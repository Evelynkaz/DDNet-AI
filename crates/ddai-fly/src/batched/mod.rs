//! The batched training backend (task 7.2b): forward-with-recording and BPTT for a whole
//! mini-batch at once, with the batch as the innermost, SIMD dimension.
//!
//! The per-sequence path of task 7.2 ([`crate::train::train_step`]) parallelises over sequences,
//! so every sequence re-reads the whole CSR (indices and weights) at every substep, forward and
//! backward. Here the state of the whole batch is `[N][NB]` cells of 8 lanes ([`lanes::L8`]; `NB =
//! ceil(B / 8)`): each edge's index and weight are loaded **once per substep for all `B` lanes**,
//! and the multiply-adds run over the lanes as vector code (auto-vectorised fixed-size array
//! loops, no `unsafe`, no intrinsics, no target features). Worker threads split the *neurons*
//! (contiguous row chunks of a [`BatchedPlan`]), never the sequences.
//!
//! The per-sequence path stays the reference: [`train_step_batched`] returns the same
//! [`BatchGradients`] as [`train_step`](crate::train::train_step) up to f32 summation order (see
//! the README for the measured bound), and the results are **bitwise independent of the thread
//! count**: chunk boundaries are a function of the graph only, partial sums are reduced per chunk
//! in a fixed order and the chunk partials are then added in chunk order.
//!
//! ## Algorithm, per substep
//! Forward, per neuron row: `V_inf = bias + sum_e w_e * r_prev[pre_e] + input` (a gather of
//! `NB` cells per in-edge), `V += decay * (V_inf - V)`, `r = f(V)`, `f'(V)`, recording `r`,
//! `X = V_inf - V_prev` and `f'(V)`. Backward, per neuron row `j` (transposed CSR): `dL/dr_j` is
//! the gather of the next substep's `delta` over `j`'s out-edges, the chain rule gives this
//! substep's `delta` and the adjoints of `b`/`theta`/inputs, and `dL/dalpha` of the *next*
//! substep is accumulated from the same `delta` rows (L1-hot from the gather) against `r_j` -- no
//! second random gather. The activation uses a vector-friendly rational `tanh` instead of libm's
//! scalar one (a few f32 ulp, see `lanes`), and neurons are internally renumbered by cell type for
//! cache locality (see `plan`).
//!
//! ## Memory
//! Per recorded substep and neuron the backward pass needs `r` (the rate that drove the substep,
//! for `dL/dalpha`), `f'(V)` (for the chain rule through the activation) and `X = V_inf - V_prev`
//! (for `dL/dtau`): three `f32` per neuron, lane and substep -- `T * S * N * B * 3 * 4` bytes (M,
//! B = 64, T = 32, S = 4: 1.25 GB). (The per-sequence recorder stores `V`, `r`, `V_inf` instead;
//! `f'(V)` is computed once, in the forward pass, from the same `tanh` that gives `r`, so the
//! membrane potential itself is a single rolling state here, not a recording.)
//! [`BatchedForwardOptions::memory_cap_bytes`] caps the engine's working set (recording,
//! adjoint state, inputs, taps, checkpoints, per-call model arrays; see
//! [`BatchedEngine::forward`] for exactly what it covers); when the full window
//! does not fit, BPTT is **chunked exactly** (gradient checkpointing): the forward pass stores `V`
//! at segment boundaries only, and the backward pass recomputes each earlier segment's recording
//! right before it is needed, carrying the adjoint state across the boundary. The gradients are
//! bit-identical to the unchunked ones (the recomputation repeats the same operations in the same
//! order); the price is one extra forward pass over all but the last segment.

pub(crate) mod kernels;
pub(crate) mod lanes;
mod plan;

use rayon::prelude::*;

use self::kernels::{BwdRead, BwdTask, Ctx, FwdRead, FwdTask, InitTask, LaneTap, bwd_chunk, fwd_chunk, init_chunk};
use self::lanes::{L8, LANES, LaneBuf, ZERO8, activation_pair_row, fix_saturated_derivative, hsum8};
use self::plan::Indices;
pub use self::plan::{BatchedPlan, DEFAULT_CHUNK_COST, DEFAULT_PAR_MIN_EDGE_CELLS};
use crate::activation::sigmoid;
use crate::backward::d_decay_d_theta_per_type;
use crate::kernel::PreIndex;
use crate::model::FlyModel;
use crate::optim::ParamGradients;
use crate::train::{BatchGradients, MemoryCapExceeded, Sequence};

/// Which backend a training loop runs its forward/backward through: the per-sequence path of task
/// 7.2 (the reference, and the default) or this module's batched one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TrainBackend {
    #[default]
    #[serde(rename = "per-seq")]
    PerSequence,
    #[serde(rename = "batched")]
    Batched,
}

/// Bytes of one 8-lane cell.
const CELL_BYTES: usize = LANES * std::mem::size_of::<f32>();

/// One sequence of a batch, as the forward pass reads it.
#[derive(Debug, Clone, Copy)]
pub struct BatchedSeqInput<'a> {
    /// `num_neurons` long.
    pub v_init: &'a [f32],
    /// One `num_inputs`-long vector per decision (`inputs.len()` is the sequence's length; a
    /// batch may mix lengths -- shorter sequences simply have no loss after their end).
    pub inputs: &'a [Vec<f32>],
}

/// One sequence's loss-side inputs of the backward pass.
#[derive(Debug, Clone, Copy)]
pub struct BatchedSeqGrad<'a> {
    /// `dL/d(dn_rates)` per decision (`num_outputs` long each); at most `inputs.len()` entries,
    /// missing trailing decisions count as zero.
    pub grad_dn: &'a [Vec<f32>],
    /// `(decision, local_substep, dL/dr)` taps, exactly as [`crate::train::Sequence::extra_taps`].
    pub extra_taps: &'a [(usize, usize, Vec<f32>)],
}

/// Options of [`BatchedEngine::forward`].
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchedForwardOptions {
    /// Cap on the engine's whole working set (see the module doc comment). `None` = unlimited.
    pub memory_cap_bytes: Option<usize>,
    /// Forces BPTT segments of this many decisions (testing / experiments); `None` = the whole
    /// window if it fits the cap, else a segment that does: the first of a shrinking sequence
    /// (`seg - 1` or `3/4 * seg`), so up to a quarter shorter than the longest that would fit.
    pub segment_decisions: Option<usize>,
    /// Also record the per-type mean rate of every decision's last substep
    /// ([`BatchedEngine::type_mean_rates`]), for an activity regulariser.
    pub type_means: bool,
}

/// What [`BatchedEngine::backward`] returns.
#[derive(Debug, Clone)]
pub struct BatchedGradients {
    /// `dL/d{a, b, theta}` summed over the whole batch.
    pub grad: ParamGradients,
    /// `grad_inputs[b][t][k]`: the input-current gradient of sequence `b`, decision `t`.
    pub grad_inputs: Vec<Vec<Vec<f32>>>,
    /// `dL/dv_init` per sequence, if requested.
    pub grad_v_init: Option<Vec<Vec<f32>>>,
}

/// Working-set estimate of a batch, see [`BatchedEngine::estimate_memory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedMemoryEstimate {
    /// Decisions per BPTT segment (`t_decisions` when the window is not chunked).
    pub segment_decisions: usize,
    pub segments: usize,
    /// The `r`/`X`/`f'(V)` recording of one segment.
    pub recording_bytes: usize,
    /// Everything else (adjoint state, inputs, taps, checkpoints, accumulators).
    pub other_bytes: usize,
}

impl BatchedMemoryEstimate {
    pub fn total_bytes(&self) -> usize {
        self.recording_bytes + self.other_bytes
    }
}

/// The state kept between [`BatchedEngine::forward`] and [`BatchedEngine::backward`].
#[derive(Debug, Clone, Copy)]
struct Run {
    batch: usize,
    nb: usize,
    t_max: usize,
    seg_decisions: usize,
    segments: usize,
    type_means: bool,
}

/// The batched forward/backward engine: a [`BatchedPlan`] plus the reusable buffers. One engine
/// serves any number of calls (buffers grow as needed, never shrink); it is `&mut`-used, so keep
/// one per training loop (the trainer wraps it in a mutex).
pub struct BatchedEngine {
    plan: BatchedPlan,
    ws: Workspace,
    arrays: ModelArrays,
    run: Option<Run>,
    /// Length (decisions) of every sequence of the batch of the last `forward`.
    lens: Vec<usize>,
}

/// Reusable buffers. Capacities only grow; every region that is read is written first.
struct Workspace {
    /// The membrane state, updated in place (one slot).
    v_state: LaneBuf,
    /// Recording: `r` (`steps + 1` slots; slot `l` is the rate that drives substep `l`), `X`
    /// (`steps` slots) and `f'(V)` (`steps + 1` slots; slot `l + 1` belongs to substep `l`'s
    /// resulting `V`, slot 0 to `V_init`).
    r_rec: LaneBuf,
    x_rec: LaneBuf,
    d_rec: LaneBuf,
    delta: [LaneBuf; 2],
    fut_dv: LaneBuf,
    grad_v_init: LaneBuf,
    inp: LaneBuf,
    gdn: LaneBuf,
    dn: LaneBuf,
    tm: LaneBuf,
    ckpt: LaneBuf,
    gin: LaneBuf,
    acc_b: Vec<f32>,
    acc_t: Vec<f32>,
    /// `dL/dalpha` lane partials, one cell per chunk-local shared-parameter slot.
    acc_a: LaneBuf,
}

fn empty() -> LaneBuf {
    LaneBuf::zeroed(0)
}

impl Workspace {
    fn new() -> Self {
        Workspace {
            v_state: empty(),
            r_rec: empty(),
            x_rec: empty(),
            d_rec: empty(),
            delta: [empty(), empty()],
            fut_dv: empty(),
            grad_v_init: empty(),
            inp: empty(),
            gdn: empty(),
            dn: empty(),
            tm: empty(),
            ckpt: empty(),
            gin: empty(),
            acc_b: Vec::new(),
            acc_t: Vec::new(),
            acc_a: empty(),
        }
    }

    /// Bytes held beyond what a call needing `needs` (cells per buffer, the same buffers
    /// [`BatchedEngine::forward`] and `backward_all` `ensure`) would use: buffers kept from an
    /// earlier, larger call.
    fn excess_bytes(&self, needs: &WorkspaceNeeds) -> usize {
        let pairs = [
            (self.v_state.len(), needs.slot),
            (self.r_rec.len(), needs.r_rec),
            (self.x_rec.len(), needs.x_rec),
            (self.d_rec.len(), needs.r_rec),
            (self.ckpt.len(), needs.ckpt),
            (self.inp.len(), needs.inp),
            (self.dn.len(), needs.dn),
            (self.tm.len(), needs.tm),
            (self.gdn.len(), needs.dn),
            (self.delta[0].len(), needs.slot),
            (self.delta[1].len(), needs.slot),
            (self.fut_dv.len(), needs.slot),
            (self.grad_v_init.len(), needs.slot),
            (self.gin.len(), needs.inp),
            (self.acc_a.len(), needs.acc_a),
        ];
        pairs
            .iter()
            .map(|&(held, need)| held.saturating_sub(need))
            .sum::<usize>()
            * CELL_BYTES
    }

    fn bytes(&self) -> usize {
        self.v_state.bytes()
            + self.r_rec.bytes()
            + self.x_rec.bytes()
            + self.d_rec.bytes()
            + self.delta[0].bytes()
            + self.delta[1].bytes()
            + self.fut_dv.bytes()
            + self.grad_v_init.bytes()
            + self.inp.bytes()
            + self.gdn.bytes()
            + self.dn.bytes()
            + self.tm.bytes()
            + self.ckpt.bytes()
            + self.gin.bytes()
            + self.acc_a.bytes()
            + (self.acc_b.len() + self.acc_t.len()) * 4
    }
}

/// Cells per workspace buffer that one batch shape needs (see [`Workspace::excess_bytes`]).
struct WorkspaceNeeds {
    slot: usize,
    r_rec: usize,
    x_rec: usize,
    ckpt: usize,
    inp: usize,
    dn: usize,
    tm: usize,
    acc_a: usize,
}

/// The model's current numbers in the engine's neuron/edge order, refreshed by every
/// [`BatchedEngine::forward`]: weights (post-major for the forward gather, pre-major for the
/// backward one), biases and decays. Kept apart from the [`Workspace`] so a kernel context can
/// borrow them while the workspace is borrowed mutably.
struct ModelArrays {
    weights: Vec<f32>,
    weights_t: Vec<f32>,
    bias: Vec<f32>,
    decay: Vec<f32>,
}

impl ModelArrays {
    fn new() -> Self {
        ModelArrays {
            weights: Vec::new(),
            weights_t: Vec::new(),
            bias: Vec::new(),
            decay: Vec::new(),
        }
    }

    fn refresh(&mut self, plan: &BatchedPlan, model: &FlyModel) {
        let w = model.weights();
        self.weights.clear();
        self.weights.extend(plan.edge_map.iter().map(|&e| w[e as usize]));
        self.weights_t.clear();
        self.weights_t
            .extend(plan.edge_of.iter().map(|&k| self.weights[k as usize]));
        self.bias.clear();
        self.bias.extend(plan.old_of.iter().map(|&o| model.bias()[o as usize]));
        self.decay.clear();
        self.decay
            .extend(plan.old_of.iter().map(|&o| model.decay()[o as usize]));
    }

    fn bytes(&self) -> usize {
        (self.weights.len() + self.weights_t.len() + self.bias.len() + self.decay.len()) * 4
    }
}

fn ensure(buf: &mut LaneBuf, cells: usize) {
    if buf.len() < cells {
        // Free the old buffer first: old and new must never be alive at once (the cap).
        *buf = empty();
        *buf = LaneBuf::zeroed(cells);
    }
}

/// Splits `s` into consecutive `&mut` pieces of the given lengths.
fn split_by_lens<T>(mut s: &mut [T], lens: impl Iterator<Item = usize>) -> Vec<&mut [T]> {
    let mut out = Vec::new();
    for len in lens {
        let (head, tail) = std::mem::take(&mut s).split_at_mut(len);
        out.push(head);
        s = tail;
    }
    out
}

fn row_lens(plan: &BatchedPlan, nb: usize) -> impl Iterator<Item = usize> + '_ {
    plan.chunks.iter().map(move |c| (c.r1 - c.r0) * nb)
}

impl BatchedEngine {
    /// An engine for `model`'s graph, with the default chunking.
    pub fn new(model: &FlyModel) -> Self {
        Self::with_plan(BatchedPlan::new(model))
    }

    pub fn with_plan(plan: BatchedPlan) -> Self {
        BatchedEngine {
            plan,
            ws: Workspace::new(),
            arrays: ModelArrays::new(),
            run: None,
            lens: Vec::new(),
        }
    }

    pub fn plan(&self) -> &BatchedPlan {
        &self.plan
    }

    /// Bytes of heap the engine currently holds (buffers + plan).
    pub fn memory_bytes(&self) -> usize {
        self.ws.bytes() + self.arrays.bytes() + self.plan.memory_bytes()
    }

    /// Bytes of workspace buffers held beyond what a batch of this shape needs.
    fn held_excess_bytes(&self, nb: usize, t_max: usize, seg_steps: usize, segments: usize, type_means: bool) -> usize {
        let slot = self.plan.n * nb;
        self.ws.excess_bytes(&WorkspaceNeeds {
            slot,
            r_rec: (seg_steps + 1) * slot,
            x_rec: seg_steps * slot,
            ckpt: segments * slot,
            inp: t_max * self.plan.num_inputs * nb,
            dn: t_max * self.plan.num_outputs * nb,
            tm: if type_means {
                t_max * self.plan.num_types * nb
            } else {
                0
            },
            acc_a: self.plan.sid_list.len(),
        })
    }

    /// The working set of a batch of `batch` sequences of up to `t_decisions` decisions, split into
    /// BPTT segments of `segment_decisions` (`None` = the first shrinking-sequence segment that fits `memory_cap_bytes`, or
    /// the whole window without a cap). Errors if not even a one-decision segment fits.
    ///
    /// This is the estimate for *this batch shape on a fresh engine*; it does not know what the
    /// engine already holds. [`BatchedEngine::forward`] accounts for that (see there). It covers
    /// the engine's buffers and the per-call copies of the model's numbers, not the plan's
    /// (graph-sized, see [`BatchedEngine::memory_bytes`]) and not anything the caller allocates.
    pub fn estimate_memory(
        &self,
        model: &FlyModel,
        batch: usize,
        t_decisions: usize,
        opts: &BatchedForwardOptions,
    ) -> Result<BatchedMemoryEstimate, MemoryCapExceeded> {
        let nb = batch.max(1).div_ceil(LANES);
        let s = model.config().substeps_per_decision as usize;
        let make = |seg: usize| -> BatchedMemoryEstimate {
            let slot = self.plan.n * nb * CELL_BYTES;
            let segments = t_decisions.div_ceil(seg.max(1)).max(1);
            let recording_bytes = (seg * s * 3 + 2) * slot;
            let other_bytes = (2 + 1 + 1 + 1) * slot // delta x2, fut_dv, grad_v_init, V state
                + segments * slot // checkpoints
                + t_decisions * self.plan.num_inputs * nb * CELL_BYTES // inputs
                + self.plan.num_inputs * t_decisions * nb * CELL_BYTES // input gradients
                + 2 * t_decisions * self.plan.num_outputs * nb * CELL_BYTES // grad_dn taps + dn_rates
                + if opts.type_means { t_decisions * self.plan.num_types * nb * CELL_BYTES } else { 0 }
                + (2 * self.plan.chunks.len() * self.plan.num_types + 2 * self.plan.nnz + 2 * self.plan.n) * 4
                + self.plan.sid_list.len() * CELL_BYTES;
            BatchedMemoryEstimate {
                segment_decisions: seg,
                segments,
                recording_bytes,
                other_bytes,
            }
        };
        if t_decisions == 0 {
            return Ok(make(1));
        }
        if let Some(seg) = opts.segment_decisions {
            let est = make(seg.clamp(1, t_decisions));
            return match opts.memory_cap_bytes {
                Some(cap) if est.total_bytes() > cap => Err(MemoryCapExceeded {
                    estimated_bytes: est.total_bytes(),
                    cap_bytes: cap,
                }),
                _ => Ok(est),
            };
        }
        let Some(cap) = opts.memory_cap_bytes else {
            return Ok(make(t_decisions));
        };
        let mut seg = t_decisions;
        loop {
            let est = make(seg);
            if est.total_bytes() <= cap {
                return Ok(est);
            }
            if seg == 1 {
                return Err(MemoryCapExceeded {
                    estimated_bytes: est.total_bytes(),
                    cap_bytes: cap,
                });
            }
            // Shrink geometrically, then refine: cheap, and the estimate is monotone in `seg`.
            seg = (seg - 1).min(seg * 3 / 4).max(1);
        }
    }

    // -----------------------------------------------------------------------------------------
    // forward
    // -----------------------------------------------------------------------------------------

    /// Runs the whole batch forward, recording what the backward pass needs, and keeps the
    /// per-decision `dn_rates` (and optionally type means) readable through
    /// [`BatchedEngine::dn_rates`]. Panics on a shape mismatch (caller-assembled input); returns
    /// [`MemoryCapExceeded`] before allocating anything if the cap cannot be met.
    ///
    /// With `memory_cap_bytes`, the cap applies to what the engine holds *after* this call and
    /// the following `backward` (buffers plus per-call model arrays, not the plan): if buffers
    /// kept from an earlier, larger call would push that over the cap, they are released first
    /// (and reallocated at their new size); otherwise they are reused. Without a cap buffers only
    /// grow. Allocations outside the engine (a caller's encoded features, the returned
    /// `grad_inputs`, logits) are not counted.
    ///
    /// `model`'s parameters must not change between this call and [`BatchedEngine::backward`].
    pub fn forward(
        &mut self,
        model: &FlyModel,
        seqs: &[BatchedSeqInput<'_>],
        opts: &BatchedForwardOptions,
    ) -> Result<(), MemoryCapExceeded> {
        self.plan.assert_matches(model);
        self.run = None;
        let n = self.plan.n;
        let k_in = self.plan.num_inputs;
        let batch = seqs.len();
        assert!(batch > 0, "BatchedEngine::forward: empty batch");
        for s in seqs {
            assert_eq!(s.v_init.len(), n, "BatchedEngine::forward: v_init length");
            for x in s.inputs {
                assert_eq!(x.len(), k_in, "BatchedEngine::forward: inputs[t] length");
            }
        }
        let t_max = seqs.iter().map(|s| s.inputs.len()).max().unwrap_or(0);
        let est = self.estimate_memory(model, batch, t_max, opts)?;
        let nb = batch.div_ceil(LANES);
        let s_sub = model.config().substeps_per_decision as usize;
        let slot = n * nb;
        let seg_steps = est.segment_decisions * s_sub;
        if let Some(cap) = opts.memory_cap_bytes {
            let excess = self.held_excess_bytes(nb, t_max, seg_steps, est.segments, opts.type_means);
            if est.total_bytes() + excess > cap {
                self.ws = Workspace::new();
            }
        }
        let ws = &mut self.ws;
        ensure(&mut ws.v_state, slot);
        ensure(&mut ws.r_rec, (seg_steps + 1) * slot);
        ensure(&mut ws.x_rec, seg_steps * slot);
        ensure(&mut ws.d_rec, (seg_steps + 1) * slot);
        ensure(&mut ws.ckpt, est.segments * slot);
        ensure(&mut ws.inp, t_max * k_in * nb);
        ensure(&mut ws.dn, t_max * self.plan.num_outputs * nb);
        if opts.type_means {
            ensure(&mut ws.tm, t_max * self.plan.num_types * nb);
        }

        // Pack `v_init` (segment 0's start) and the inputs into lane layout; padded lanes and
        // decisions past a sequence's end stay zero.
        ws.ckpt.cells_mut()[..slot].fill(ZERO8);
        {
            let c0 = &mut ws.ckpt.cells_mut()[..slot];
            for (b, s) in seqs.iter().enumerate() {
                let (cell, lane) = (b / LANES, b % LANES);
                for (i, &v) in s.v_init.iter().enumerate() {
                    c0[self.plan.new_of[i] as usize * nb + cell][lane] = v;
                }
            }
        }
        ws.inp.cells_mut()[..t_max * k_in * nb].fill(ZERO8);
        if k_in > 0 && t_max > 0 {
            let inp = &mut ws.inp.cells_mut()[..t_max * k_in * nb];
            inp.par_chunks_mut(k_in * nb).enumerate().for_each(|(t, cells)| {
                for (b, s) in seqs.iter().enumerate() {
                    if let Some(x) = s.inputs.get(t) {
                        let (cell, lane) = (b / LANES, b % LANES);
                        for (k, &val) in x.iter().enumerate() {
                            cells[k * nb + cell][lane] = val;
                        }
                    }
                }
            });
        }

        let run = Run {
            batch,
            nb,
            t_max,
            seg_decisions: est.segment_decisions,
            segments: est.segments,
            type_means: opts.type_means,
        };
        self.arrays.refresh(&self.plan, model);
        match &self.plan.indices {
            Indices::Narrow { pre, post } => {
                forward_all::<u16>(&self.plan, &self.arrays, &mut self.ws, model, (pre, post), &run);
            }
            Indices::Wide { pre, post } => {
                forward_all::<u32>(&self.plan, &self.arrays, &mut self.ws, model, (pre, post), &run);
            }
        }
        self.lens = seqs.iter().map(|s| s.inputs.len()).collect();
        self.run = Some(run);
        Ok(())
    }

    /// `dn_rates` of sequence `b` at decision `t` (as `FlyState::step_decision` would return it),
    /// after [`BatchedEngine::forward`]. `out.len() == num_outputs`.
    pub fn dn_rates(&self, b: usize, t: usize, out: &mut [f32]) {
        let run = self.run.expect("dn_rates: no forward pass has run");
        assert!(b < run.batch && t < run.t_max && out.len() == self.plan.num_outputs);
        let cells = self.ws.dn.cells();
        let (cell, lane) = (b / LANES, b % LANES);
        for (s, o) in out.iter_mut().enumerate() {
            *o = cells[(t * self.plan.num_outputs + s) * run.nb + cell][lane];
        }
    }

    /// Per-type mean `f(V)` after decision `t`'s last substep (`FlyState`'s
    /// `per_type_mean_rate`); only recorded with [`BatchedForwardOptions::type_means`].
    pub fn type_mean_rates(&self, b: usize, t: usize, out: &mut [f32]) {
        let run = self.run.expect("type_mean_rates: no forward pass has run");
        assert!(run.type_means, "type_mean_rates: forward ran without `type_means`");
        assert!(b < run.batch && t < run.t_max && out.len() == self.plan.num_types);
        let cells = self.ws.tm.cells();
        let (cell, lane) = (b / LANES, b % LANES);
        for (ty, o) in out.iter_mut().enumerate() {
            *o = cells[(t * self.plan.num_types + ty) * run.nb + cell][lane];
        }
    }

    // -----------------------------------------------------------------------------------------
    // backward
    // -----------------------------------------------------------------------------------------

    /// BPTT through the batch recorded by the last [`BatchedEngine::forward`]: the parameter
    /// gradients summed over the batch, every sequence's input-current gradients, and (if asked)
    /// every sequence's initial-state gradient. `seqs[b]` carries sequence `b`'s loss-side
    /// inputs; `model` must be the one `forward` ran with, parameters unchanged.
    pub fn backward(
        &mut self,
        model: &FlyModel,
        seqs: &[BatchedSeqGrad<'_>],
        want_v_init_grad: bool,
    ) -> BatchedGradients {
        self.plan.assert_matches(model);
        let run = self.run.expect("BatchedEngine::backward: no forward pass has run");
        assert_eq!(
            seqs.len(),
            run.batch,
            "BatchedEngine::backward: one BatchedSeqGrad per sequence"
        );
        let s_sub = model.config().substeps_per_decision as usize;
        for (b, sg) in seqs.iter().enumerate() {
            let len = self.lens[b];
            assert!(sg.grad_dn.len() <= len, "backward: grad_dn longer than the sequence");
            for g in sg.grad_dn {
                assert_eq!(g.len(), self.plan.num_outputs, "backward: grad_dn[t] length");
            }
            for (decision, local, g) in sg.extra_taps {
                assert!(*decision < len, "backward: extra tap decision out of range");
                assert!(*local < s_sub, "backward: extra tap local_substep out of range");
                assert_eq!(g.len(), self.plan.n, "backward: extra tap length");
            }
        }
        let out = match &self.plan.indices {
            Indices::Narrow { pre, post } => backward_all::<u16>(
                &self.plan,
                &self.arrays,
                &mut self.ws,
                model,
                (pre, post),
                &run,
                seqs,
                want_v_init_grad,
            ),
            Indices::Wide { pre, post } => backward_all::<u32>(
                &self.plan,
                &self.arrays,
                &mut self.ws,
                model,
                (pre, post),
                &run,
                seqs,
                want_v_init_grad,
            ),
        };
        // The recording buffers no longer hold the last segment: a new forward pass is needed.
        self.run = None;
        let mut grad_inputs = Vec::with_capacity(run.batch);
        let k_in = self.plan.num_inputs;
        let gin = self.ws.gin.cells();
        for b in 0..run.batch {
            let (cell, lane) = (b / LANES, b % LANES);
            grad_inputs.push(
                (0..self.lens[b])
                    .map(|t| {
                        (0..k_in)
                            .map(|k| gin[(self.plan.ord_of_k[k] as usize * run.t_max + t) * run.nb + cell][lane])
                            .collect::<Vec<f32>>()
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let grad_v_init = want_v_init_grad.then(|| {
            let g = self.ws.grad_v_init.cells();
            (0..run.batch)
                .map(|b| {
                    let (cell, lane) = (b / LANES, b % LANES);
                    (0..self.plan.n)
                        .map(|i| g[self.plan.new_of[i] as usize * run.nb + cell][lane])
                        .collect()
                })
                .collect()
        });
        BatchedGradients {
            grad: out,
            grad_inputs,
            grad_v_init,
        }
    }
}

/// Runs every segment forward, each from its checkpointed start state.
fn forward_all<I: PreIndex + Sync>(
    plan: &BatchedPlan,
    arrays: &ModelArrays,
    ws: &mut Workspace,
    model: &FlyModel,
    (pre, post): (&[I], &[I]),
    run: &Run,
) {
    let ctx = make_ctx(plan, arrays, model, (pre, post), run.nb);
    for seg in 0..run.segments {
        forward_segment(&ctx, ws, model, run, seg, true);
    }
}

fn make_ctx<'a, I: PreIndex + Sync>(
    plan: &'a BatchedPlan,
    arrays: &'a ModelArrays,
    model: &FlyModel,
    (pre, post): (&'a [I], &'a [I]),
    nb: usize,
) -> Ctx<'a, I> {
    let r_max = model.config().r_max;
    Ctx {
        plan,
        nb,
        row_start: &plan.row_start,
        pre_index: pre,
        post_of: post,
        weights: &arrays.weights,
        weights_t: &arrays.weights_t,
        bias: &arrays.bias,
        decay: &arrays.decay,
        r_max,
        inv_r_max: 1.0 / r_max,
        par: plan.nnz * nb >= plan.par_min_edge_cells,
    }
}

/// Runs decisions `[d0, d1)` of segment `seg` forward from its checkpointed start state,
/// recording `r`/`X`/`f'(V)` into the recording buffers. With `outputs`, also stores each
/// decision's `dn_rates` (and type means) and the next segment's checkpoint.
fn forward_segment<I: PreIndex + Sync>(
    ctx: &Ctx<'_, I>,
    ws: &mut Workspace,
    model: &FlyModel,
    run: &Run,
    seg: usize,
    outputs: bool,
) {
    let plan = ctx.plan;
    let nb = run.nb;
    let slot = plan.n * nb;
    let s_sub = model.config().substeps_per_decision as usize;
    let k_in = plan.num_inputs;
    let n_out = plan.num_outputs;
    let d0 = seg * run.seg_decisions;
    let d1 = (d0 + run.seg_decisions).min(run.t_max);
    let Workspace {
        v_state,
        r_rec,
        x_rec,
        d_rec,
        ckpt,
        inp,
        dn,
        tm,
        ..
    } = ws;
    let v = &mut v_state.cells_mut()[..slot];
    let r = r_rec.cells_mut();
    let x = x_rec.cells_mut();
    let d = d_rec.cells_mut();

    v.copy_from_slice(&ckpt.cells()[seg * slot..(seg + 1) * slot]);
    {
        let (r_max, inv) = (ctx.r_max, ctx.inv_r_max);
        let (r0, d_0) = (&mut r[..slot], &mut d[..slot]);
        r0.par_chunks_mut(nb * 64)
            .zip(d_0.par_chunks_mut(nb * 64))
            .zip(v.par_chunks(nb * 64))
            .for_each(|((rc, dc), vc)| {
                let (vf, df) = (vc.as_flattened(), dc.as_flattened_mut());
                if activation_pair_row(vf, rc.as_flattened_mut(), df, r_max, inv) {
                    fix_saturated_derivative(vf, df, inv);
                }
            });
    }

    let inp_cells = inp.cells();
    for dec in d0..d1 {
        let inp_d = &inp_cells[dec * k_in * nb..(dec + 1) * k_in * nb];
        for j in 0..s_sub {
            fwd_region(ctx, slot, v, r, x, d, inp_d, (dec - d0) * s_sub + j);
        }
        if outputs {
            let ls_last = (dec - d0) * s_sub + s_sub - 1;
            let r_before = &r[ls_last * slot..(ls_last + 1) * slot];
            let r_after = &r[(ls_last + 1) * slot..(ls_last + 2) * slot];
            let dn_cells = dn.cells_mut();
            for (s, &neuron) in plan.out_neuron.iter().enumerate() {
                let i = neuron as usize;
                for c in 0..nb {
                    let (rb, ra) = (&r_before[i * nb + c], &r_after[i * nb + c]);
                    let o = &mut dn_cells[(dec * n_out + s) * nb + c];
                    for l in 0..LANES {
                        o[l] = 0.5 * (rb[l] + ra[l]);
                    }
                }
            }
            if run.type_means {
                let types = &model.flyg().types;
                let tm_cells = &mut tm.cells_mut()[dec * plan.num_types * nb..(dec + 1) * plan.num_types * nb];
                tm_cells.fill(ZERO8);
                for i in 0..plan.n {
                    let ty = plan.type_of[i] as usize;
                    for c in 0..nb {
                        let (acc, ri) = (&mut tm_cells[ty * nb + c], &r_after[i * nb + c]);
                        for l in 0..LANES {
                            acc[l] += ri[l];
                        }
                    }
                }
                for (ty, t) in types.iter().enumerate() {
                    let cnt = t.neuron_count.max(1) as f32;
                    for cell in &mut tm_cells[ty * nb..(ty + 1) * nb] {
                        for x in cell.iter_mut() {
                            *x /= cnt;
                        }
                    }
                }
            }
        }
    }
    if outputs && seg + 1 < run.segments {
        ckpt.cells_mut()[(seg + 1) * slot..(seg + 2) * slot].copy_from_slice(v);
    }
}

/// One forward substep over all chunks in parallel.
#[allow(clippy::too_many_arguments)]
fn fwd_region<I: PreIndex + Sync>(
    ctx: &Ctx<'_, I>,
    slot: usize,
    v: &mut [L8],
    r: &mut [L8],
    x: &mut [L8],
    d: &mut [L8],
    inp: &[L8],
    ls: usize,
) {
    let plan = ctx.plan;
    let nb = ctx.nb;
    let (r_head, r_tail) = r.split_at_mut((ls + 1) * slot);
    let rd = FwdRead {
        r_prev: &r_head[ls * slot..],
        inp,
    };
    let vn = split_by_lens(v, row_lens(plan, nb));
    let rn = split_by_lens(&mut r_tail[..slot], row_lens(plan, nb));
    let xn = split_by_lens(&mut x[ls * slot..(ls + 1) * slot], row_lens(plan, nb));
    let dn = split_by_lens(&mut d[(ls + 1) * slot..(ls + 2) * slot], row_lens(plan, nb));
    let tasks: Vec<FwdTask<'_>> = vn
        .into_iter()
        .zip(rn)
        .zip(xn)
        .zip(dn)
        .enumerate()
        .map(|(chunk, (((v, r_new), x_new), d_new))| FwdTask {
            chunk,
            v,
            r_new,
            x_new,
            d_new,
        })
        .collect();
    if ctx.par {
        tasks.into_par_iter().for_each(|t| fwd_chunk(ctx, &rd, t));
    } else {
        tasks.into_iter().for_each(|t| fwd_chunk(ctx, &rd, t));
    }
}

#[allow(clippy::too_many_arguments)]
fn backward_all<I: PreIndex + Sync>(
    plan: &BatchedPlan,
    arrays: &ModelArrays,
    ws: &mut Workspace,
    model: &FlyModel,
    (pre, post): (&[I], &[I]),
    run: &Run,
    seqs: &[BatchedSeqGrad<'_>],
    want_v_init_grad: bool,
) -> ParamGradients {
    let nb = run.nb;
    let n = plan.n;
    let slot = n * nb;
    let s_sub = model.config().substeps_per_decision as usize;
    let t_max = run.t_max;
    let n_out = plan.num_outputs;
    let k_in = plan.num_inputs;
    let num_chunks = plan.chunks.len();

    // Taps and accumulators. The tap buffer is moved out of the workspace for the duration (it is
    // read-only while the regions borrow the rest of the workspace mutably).
    let mut gdn_buf = std::mem::replace(&mut ws.gdn, empty());
    ensure(&mut gdn_buf, t_max * n_out * nb);
    gdn_buf.cells_mut()[..t_max * n_out * nb].fill(ZERO8);
    {
        let gdn = gdn_buf.cells_mut();
        for (b, sg) in seqs.iter().enumerate() {
            let (cell, lane) = (b / LANES, b % LANES);
            for (t, g) in sg.grad_dn.iter().enumerate() {
                for (s, &val) in g.iter().enumerate() {
                    gdn[(t * n_out + s) * nb + cell][lane] = val;
                }
            }
        }
    }
    for buf in &mut ws.delta {
        ensure(buf, slot);
    }
    ensure(&mut ws.fut_dv, slot);
    ws.fut_dv.cells_mut()[..slot].fill(ZERO8);
    ensure(&mut ws.gin, k_in * t_max * nb);
    ws.gin.cells_mut()[..k_in * t_max * nb].fill(ZERO8);
    if want_v_init_grad {
        ensure(&mut ws.grad_v_init, slot);
    }
    let num_types = plan.num_types.max(1);
    ws.acc_b.clear();
    ws.acc_b.resize(num_chunks * num_types, 0.0);
    ws.acc_t.clear();
    ws.acc_t.resize(num_chunks * num_types, 0.0);
    ensure(&mut ws.acc_a, plan.sid_list.len());
    ws.acc_a.cells_mut()[..plan.sid_list.len()].fill(ZERO8);

    let ctx = make_ctx(plan, arrays, model, (pre, post), nb);

    // Dense per-lane taps, bucketed by flat substep.
    let lane_tap_list: Vec<(usize, usize, &[f32])> = seqs
        .iter()
        .enumerate()
        .flat_map(|(b, sg)| {
            sg.extra_taps
                .iter()
                .map(move |(d, loc, g)| (d * s_sub + loc, b, g.as_slice()))
        })
        .collect();

    let mut cur = 0usize;
    let mut have_next = false;
    for seg in (0..run.segments).rev() {
        // `forward` leaves the last segment's recording resident; earlier ones are recomputed.
        if seg + 1 != run.segments {
            forward_segment(&ctx, ws, model, run, seg, false);
        }
        let d0 = seg * run.seg_decisions;
        let d1 = (d0 + run.seg_decisions).min(t_max);
        let steps = (d1 - d0) * s_sub;
        for ls in (0..steps).rev() {
            let l = d0 * s_sub + ls;
            let gdn = gdn_buf.cells();
            let tap_dec = |t: usize| &gdn[t * n_out * nb..(t + 1) * n_out * nb];
            let tap_final = (l + 1).is_multiple_of(s_sub).then(|| tap_dec((l + 1) / s_sub - 1));
            let tap_before =
                ((l + 2).is_multiple_of(s_sub) && (l + 2) / s_sub - 1 < t_max).then(|| tap_dec((l + 2) / s_sub - 1));
            let lane_taps: Vec<LaneTap<'_>> = lane_tap_list
                .iter()
                .filter(|(step, _, _)| *step == l)
                .map(|&(_, lane, grad)| LaneTap { lane, grad })
                .collect();
            bwd_region(
                &ctx,
                ws,
                (slot, ls, cur, have_next),
                &BwdStep {
                    decision: l / s_sub,
                    t_max,
                    tap_final,
                    tap_before,
                    lane_taps: &lane_taps,
                },
            );
            cur ^= 1;
            have_next = true;
        }
    }

    // The pass before the first substep (segment 0's recording is resident now): the `dL/dalpha`
    // of substep 0 against `f(V_init)`, and, if asked, the initial-state gradient (one more
    // transposed gather, the `substeps == 1` boundary tap, the chain rule through `f(V_init)`).
    if t_max > 0 {
        let tap_before = (s_sub == 1 && n_out > 0).then(|| &gdn_buf.cells()[..n_out * nb]);
        init_region(&ctx, ws, cur ^ 1, tap_before, want_v_init_grad);
    } else if want_v_init_grad {
        ws.grad_v_init.cells_mut()[..slot].fill(ZERO8);
    }

    // Deterministic reduction: chunk partials in chunk order, the lane sum of `dL/dalpha` last.
    let mut d_decay = vec![0.0f32; plan.num_types];
    d_decay_d_theta_per_type(model, &mut d_decay);
    let mut grad = ParamGradients::zeros_like(model.params());
    let acc_a = ws.acc_a.cells();
    for c in 0..num_chunks {
        for ty in 0..plan.num_types {
            grad.b[ty] += ws.acc_b[c * num_types + ty];
            grad.theta[ty] += ws.acc_t[c * num_types + ty];
        }
        let ch = &plan.chunks[c];
        for (&sid, acc) in plan.sid_list[ch.s0..ch.s1].iter().zip(&acc_a[ch.s0..ch.s1]) {
            grad.a[sid as usize] += hsum8(acc);
        }
    }
    for (ty, g) in grad.theta.iter_mut().enumerate() {
        *g *= d_decay[ty];
    }
    for (sid, g) in grad.a.iter_mut().enumerate() {
        *g *= sigmoid(model.params().a[sid]);
    }
    ws.gdn = gdn_buf;
    grad
}

/// Per-substep inputs of [`bwd_region`] that are not buffers.
struct BwdStep<'a> {
    decision: usize,
    t_max: usize,
    tap_final: Option<&'a [L8]>,
    tap_before: Option<&'a [L8]>,
    lane_taps: &'a [LaneTap<'a>],
}

/// One backward substep (transposed gather, chain rule, `dL/dalpha` of the next substep) over all
/// chunks in parallel. `(slot, ls, cur, have_next)`: cells per slot, the substep's index in the
/// loaded segment, which delta buffer to write, and whether the other holds the next substep's
/// delta.
fn bwd_region<I: PreIndex + Sync>(
    ctx: &Ctx<'_, I>,
    ws: &mut Workspace,
    (slot, ls, cur, have_next): (usize, usize, usize, bool),
    step: &BwdStep<'_>,
) {
    let plan = ctx.plan;
    let nb = ctx.nb;
    let num_types = plan.num_types.max(1);
    let Workspace {
        r_rec,
        x_rec,
        d_rec,
        delta,
        fut_dv,
        gin,
        acc_b,
        acc_t,
        acc_a,
        ..
    } = ws;
    let (d_lo, d_hi) = delta.split_at_mut(1);
    let (cur_buf, next_buf) = if cur == 0 {
        (&mut d_lo[0], &d_hi[0])
    } else {
        (&mut d_hi[0], &d_lo[0])
    };
    let rd = BwdRead {
        d_l: &d_rec.cells()[(ls + 1) * slot..(ls + 2) * slot],
        x_l: &x_rec.cells()[ls * slot..(ls + 1) * slot],
        r_l: &r_rec.cells()[(ls + 1) * slot..(ls + 2) * slot],
        delta_next: have_next.then(|| &next_buf.cells()[..slot]),
        tap_final: step.tap_final,
        tap_before: step.tap_before,
        lane_taps: step.lane_taps,
        decision: step.decision,
        t_max: step.t_max,
    };
    let delta_cur = split_by_lens(&mut cur_buf.cells_mut()[..slot], row_lens(plan, nb));
    let fut = split_by_lens(&mut fut_dv.cells_mut()[..slot], row_lens(plan, nb));
    let k_total = plan.num_inputs * step.t_max * nb;
    let gins = split_by_lens(
        &mut gin.cells_mut()[..k_total],
        plan.chunks.iter().map(|c| (c.k1 - c.k0) * step.t_max * nb),
    );
    let ab = acc_b.chunks_mut(num_types);
    let at = acc_t.chunks_mut(num_types);
    let aa = split_by_lens(
        &mut acc_a.cells_mut()[..plan.sid_list.len()],
        plan.chunks.iter().map(|c| c.s1 - c.s0),
    );
    let tasks: Vec<BwdTask<'_>> = delta_cur
        .into_iter()
        .zip(fut)
        .zip(gins)
        .zip(ab)
        .zip(at)
        .zip(aa)
        .enumerate()
        .map(
            |(chunk, (((((delta_cur, fut_dv), gin), acc_b), acc_t), acc_a))| BwdTask {
                chunk,
                fut_dv,
                delta_cur,
                gin,
                acc_b,
                acc_t,
                acc_a,
            },
        )
        .collect();
    if ctx.par {
        tasks.into_par_iter().for_each(|t| bwd_chunk(ctx, &rd, t));
    } else {
        tasks.into_iter().for_each(|t| bwd_chunk(ctx, &rd, t));
    }
}

/// The pass before the first substep (see [`backward_all`]); `last` = the delta buffer holding the
/// first substep's `dL/dV_inf`.
fn init_region<I: PreIndex + Sync>(
    ctx: &Ctx<'_, I>,
    ws: &mut Workspace,
    last: usize,
    tap_before: Option<&[L8]>,
    want_grad_v_init: bool,
) {
    let plan = ctx.plan;
    let nb = ctx.nb;
    let slot = plan.n * nb;
    let Workspace {
        r_rec,
        x_rec,
        d_rec,
        delta,
        fut_dv,
        grad_v_init,
        acc_a,
        ..
    } = ws;
    let rd = BwdRead {
        d_l: &d_rec.cells()[..slot],
        x_l: &x_rec.cells()[..slot],
        r_l: &r_rec.cells()[..slot],
        delta_next: Some(&delta[last].cells()[..slot]),
        tap_final: None,
        tap_before,
        lane_taps: &[],
        decision: 0,
        t_max: 0,
    };
    let fut = fut_dv.cells();
    let mut gvi: Vec<Option<&mut [L8]>> = if want_grad_v_init {
        split_by_lens(&mut grad_v_init.cells_mut()[..slot], row_lens(plan, nb))
            .into_iter()
            .map(Some)
            .collect()
    } else {
        (0..plan.chunks.len()).map(|_| None).collect()
    };
    let aa = split_by_lens(
        &mut acc_a.cells_mut()[..plan.sid_list.len()],
        plan.chunks.iter().map(|c| c.s1 - c.s0),
    );
    let tasks: Vec<InitTask<'_>> = gvi
        .iter_mut()
        .zip(aa)
        .enumerate()
        .map(|(chunk, (g, acc_a))| InitTask {
            chunk,
            fut_dv: &fut[plan.chunks[chunk].r0 * nb..plan.chunks[chunk].r1 * nb],
            grad_v_init: g.take(),
            acc_a,
        })
        .collect();
    if ctx.par {
        tasks.into_par_iter().for_each(|t| init_chunk(ctx, &rd, t));
    } else {
        tasks.into_iter().for_each(|t| init_chunk(ctx, &rd, t));
    }
}

/// Drop-in batched counterpart of [`crate::train::train_step`]: the same [`Sequence`]s in, the
/// same [`BatchGradients`] out (up to f32 summation order, see the module doc comment).
pub fn train_step_batched(
    model: &FlyModel,
    engine: &mut BatchedEngine,
    sequences: &[Sequence],
    memory_cap_bytes: Option<usize>,
) -> Result<BatchGradients, MemoryCapExceeded> {
    for seq in sequences {
        seq.validate(model);
    }
    if sequences.is_empty() {
        return Ok(BatchGradients {
            grad: ParamGradients::zeros_like(model.params()),
            grad_inputs: Vec::new(),
        });
    }
    let ins: Vec<BatchedSeqInput<'_>> = sequences
        .iter()
        .map(|s| BatchedSeqInput {
            v_init: &s.v_init,
            inputs: &s.inputs,
        })
        .collect();
    engine.forward(
        model,
        &ins,
        &BatchedForwardOptions {
            memory_cap_bytes,
            ..BatchedForwardOptions::default()
        },
    )?;
    let grads: Vec<BatchedSeqGrad<'_>> = sequences
        .iter()
        .map(|s| BatchedSeqGrad {
            grad_dn: &s.grad_dn,
            extra_taps: &s.extra_taps,
        })
        .collect();
    let g = engine.backward(model, &grads, false);
    Ok(BatchGradients {
        grad: g.grad,
        grad_inputs: g.grad_inputs,
    })
}
