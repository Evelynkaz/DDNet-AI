//! [`BatchedPlan`]: everything the batched backend derives from the *topology* (and `gamma`, the
//! signs and the type table) of a graph, built once per [`FlyModel`] and reused for every training
//! step: the neuron relabeling, both CSR directions in that labeling, the neuron-row chunking the
//! worker threads split, the per-edge `sign * N_ij / Z_i` coefficient of the `a` gradient, and each
//! chunk's compact map from the shared type-pair parameters it touches to local accumulator slots.
//!
//! **Relabeling.** The engine works on its own neuron order, `(type, original index)`: neurons of
//! one cell type are wired alike (same presynaptic partners), so the rows a chunk of consecutive
//! neurons gathers from overlap far more -- on M the distinct presynaptic rows per chunk drop from
//! 64% to 38% of the gathers, ~25% less CPU time per step. Everything crossing the API (initial
//! states, inputs, outputs, taps, per-type means) is mapped at the boundary; the order of the
//! neurons *of one type* is the original one, so per-type sums run in the original order.
//!
//! **The chunking is part of the numerics, never of the schedule.** Floating-point sums are
//! reduced per chunk (in a fixed order inside the chunk) and the chunk partials are then added in
//! chunk order, so the result depends only on the chunk boundaries -- which are a pure function of
//! the graph -- and not on how many threads happen to run the chunks: bitwise identical results at
//! any thread count.

use crate::model::FlyModel;

/// Sentinel for "this neuron has no input / output slot".
pub(crate) const NO_SLOT: u32 = u32::MAX;

/// Default target "cost" of one chunk of rows: in-edges + out-edges + a per-row constant. Chosen so
/// a chunk is tens of microseconds of work at the smallest batch (enough to amortise rayon's
/// per-task cost) while M still splits into ~100 chunks for 8 threads to balance.
pub const DEFAULT_CHUNK_COST: usize = 8192;

/// Default of [`BatchedPlan::with_parallel_threshold`]: a substep's region (all chunks of one
/// forward or backward step) runs on the rayon pool only when `nnz * NB` (edges times 8-lane
/// cells) reaches this -- about 4.4 ns per edge-cell serially in the forward pass, so ~0.9 ms of
/// work. Below that the rendezvous of the pool's threads (every substep is one) can cost more than
/// the work it spreads: S with the 24-window batch of `train-demo` is ~0.7 ms per region serially,
/// and on a loaded machine each rendezvous took milliseconds (the demo ran 4x slower on 8 threads
/// than the serial regions would have). The result is bitwise the same either way: chunks, not
/// threads, define every sum.
pub const DEFAULT_PAR_MIN_EDGE_CELLS: usize = 200_000;

/// A contiguous block of neuron rows (in the engine's order) and everything that belongs to them.
#[derive(Debug, Clone)]
pub(crate) struct Chunk {
    /// Rows `[r0, r1)`.
    pub r0: usize,
    pub r1: usize,
    /// Pre-major edge positions `[t0, t1)` of those rows (= `out_row_start[r0]..out_row_start[r1]`):
    /// the out-edges of the chunk's neurons, which the backward pass walks.
    pub t0: usize,
    pub t1: usize,
    /// Input ordinals `[k0, k1)` (see [`BatchedPlan::in_ord`]) of those rows.
    pub k0: usize,
    pub k1: usize,
    /// Local accumulator slots `[s0, s1)` of the shared parameters this chunk's edges touch.
    pub s0: usize,
    pub s1: usize,
}

/// Index tables of the CSR in one width: `u16` when the graph has at most 65536 neurons (like
/// [`FlyModel`]'s own narrow copy, half the index traffic), `u32` otherwise.
#[derive(Debug, Clone)]
pub(crate) enum Indices {
    Narrow { pre: Vec<u16>, post: Vec<u16> },
    Wide { pre: Vec<u32>, post: Vec<u32> },
}

/// Topology-derived tables of the batched backend. See the module doc comment. All neuron indices
/// below are in the engine's order unless stated otherwise.
#[derive(Debug, Clone)]
pub struct BatchedPlan {
    pub(crate) n: usize,
    pub(crate) nnz: usize,
    pub(crate) num_types: usize,
    pub(crate) shared_count: usize,
    pub(crate) num_inputs: usize,
    pub(crate) num_outputs: usize,
    /// Fingerprint of the config the coefficients were derived from.
    gamma_bits: u32,

    /// `old_of[new] = original dense index` and its inverse.
    pub(crate) old_of: Vec<u32>,
    pub(crate) new_of: Vec<u32>,

    pub(crate) chunks: Vec<Chunk>,
    /// See [`DEFAULT_PAR_MIN_EDGE_CELLS`].
    pub(crate) par_min_edge_cells: usize,

    /// Post-major CSR (forward gather): `row_start[new]..row_start[new + 1]`, presynaptic neurons
    /// ascending inside a row (`pre`), plus the original edge id of every position (`edge_map`, to
    /// pull the model's weights).
    pub(crate) row_start: Vec<u32>,
    pub(crate) edge_map: Vec<u32>,
    /// Pre-major CSR (backward transposed gather): for presynaptic neuron `j`, positions
    /// `out_row_start[j]..out_row_start[j+1]` list the postsynaptic neurons (`post`, in
    /// [`Indices`]) and the post-major position of the edge (`edge_of`).
    pub(crate) out_row_start: Vec<u32>,
    pub(crate) edge_of: Vec<u32>,
    pub(crate) indices: Indices,

    /// `sign_j * N_ij / Z_i` per **pre-major** edge position (the factor of `dL/dalpha` other than
    /// `delta` and `r`).
    pub(crate) coeff_t: Vec<f32>,
    /// Per pre-major edge position: the edge's chunk-local accumulator slot.
    pub(crate) local_sid_t: Vec<u32>,
    /// Global shared-parameter id of every local slot (all chunks, concatenated).
    pub(crate) sid_list: Vec<u32>,

    pub(crate) type_of: Vec<u32>,
    /// Per neuron: its input slot `k` (index into `FlyModel::input_neuron_indices`, i.e. into the
    /// caller's `inputs[t]`), or [`NO_SLOT`].
    pub(crate) in_slot: Vec<u32>,
    /// Per neuron: its rank among the input neurons in the engine's order, or [`NO_SLOT`]; chunks
    /// own contiguous ranges of ranks.
    pub(crate) in_ord: Vec<u32>,
    /// The rank of input slot `k`.
    pub(crate) ord_of_k: Vec<u32>,
    /// Per neuron: its output slot, or [`NO_SLOT`]; and the engine-order neuron of every output
    /// slot.
    pub(crate) out_slot: Vec<u32>,
    pub(crate) out_neuron: Vec<u32>,
}

impl BatchedPlan {
    /// Builds the plan with [`DEFAULT_CHUNK_COST`].
    pub fn new(model: &FlyModel) -> Self {
        Self::with_chunk_cost(model, DEFAULT_CHUNK_COST)
    }

    /// Builds the plan with a custom chunk cost target (tests use tiny values to force many chunks
    /// on tiny graphs). `chunk_cost >= 1`.
    pub fn with_chunk_cost(model: &FlyModel, chunk_cost: usize) -> Self {
        let chunk_cost = chunk_cost.max(1);
        let flyg = model.flyg();
        let edges = &flyg.edges;
        let n = model.num_neurons();
        let nnz = edges.pre_index.len();
        let gamma = model.config().gamma;

        // Relabeling: by (type, original index).
        let mut old_of: Vec<u32> = (0..n as u32).collect();
        old_of.sort_by_key(|&i| (flyg.neurons[i as usize].type_index, i));
        let mut new_of = vec![0u32; n];
        for (new, &old) in old_of.iter().enumerate() {
            new_of[old as usize] = new as u32;
        }

        // Post-major CSR in the new labeling, presynaptic neurons ascending inside a row.
        let mut row_start = vec![0u32; n + 1];
        let mut pre32 = Vec::with_capacity(nnz);
        let mut edge_map = Vec::with_capacity(nnz);
        let mut scratch: Vec<(u32, u32)> = Vec::new();
        for (new, &old) in old_of.iter().enumerate() {
            scratch.clear();
            for e in edges.row_start[old as usize] as usize..edges.row_start[old as usize + 1] as usize {
                scratch.push((new_of[edges.pre_index[e] as usize], e as u32));
            }
            scratch.sort_unstable();
            for &(p, e) in &scratch {
                pre32.push(p);
                edge_map.push(e);
            }
            row_start[new + 1] = pre32.len() as u32;
        }

        // Transposed CSR by counting sort (pre-major, post ascending inside a row).
        let mut out_row_start = vec![0u32; n + 1];
        for &pre in &pre32 {
            out_row_start[pre as usize + 1] += 1;
        }
        for i in 0..n {
            out_row_start[i + 1] += out_row_start[i];
        }
        let mut post32 = vec![0u32; nnz];
        let mut edge_of = vec![0u32; nnz];
        let mut cursor = out_row_start.clone();
        for post in 0..n {
            let (start, end) = (row_start[post] as usize, row_start[post + 1] as usize);
            for (off, &pre) in pre32[start..end].iter().enumerate() {
                let k = start + off;
                let pre = pre as usize;
                let pos = cursor[pre] as usize;
                post32[pos] = post as u32;
                edge_of[pos] = k as u32;
                cursor[pre] += 1;
            }
        }

        // Per-edge `sign * N_ij / Z_i` (same formula and rounding as `BackwardIndex`), pre-major.
        let inv_z_old: Vec<f32> = flyg
            .neuron_input_totals
            .full_connectome
            .iter()
            .map(|&z_raw| 1.0 / (z_raw.max(1) as f32).powf(gamma))
            .collect();
        let type_pair_sign: Vec<f32> = flyg
            .type_pairs
            .iter()
            .map(|tp| f32::from(flyg.types[tp.pre_type as usize].sign.as_i8()))
            .collect();
        let mut coeff_t = vec![0.0f32; nnz];
        let mut shared_t = vec![0u32; nnz];
        for (pos, &k) in edge_of.iter().enumerate() {
            let e = edge_map[k as usize] as usize;
            let post_old = old_of[post32[pos] as usize] as usize;
            let tp = edges.type_pair_index[e] as usize;
            coeff_t[pos] = type_pair_sign[tp] * (edges.synapse_count[e] as f32) * inv_z_old[post_old];
            shared_t[pos] = flyg.type_pairs[tp].shared_param_id;
        }

        let type_of: Vec<u32> = old_of.iter().map(|&o| flyg.neurons[o as usize].type_index).collect();
        let mut in_slot = vec![NO_SLOT; n];
        for (k, &i) in model.input_neuron_indices().iter().enumerate() {
            in_slot[new_of[i as usize] as usize] = k as u32;
        }
        let mut in_ord = vec![NO_SLOT; n];
        let mut k_of_ord = Vec::new();
        let mut row_of_ord = Vec::new();
        for new in 0..n {
            if in_slot[new] != NO_SLOT {
                in_ord[new] = k_of_ord.len() as u32;
                k_of_ord.push(in_slot[new]);
                row_of_ord.push(new);
            }
        }
        let mut ord_of_k = vec![NO_SLOT; model.num_inputs()];
        for (ord, &k) in k_of_ord.iter().enumerate() {
            ord_of_k[k as usize] = ord as u32;
        }
        let mut out_slot = vec![NO_SLOT; n];
        let mut out_neuron = Vec::with_capacity(model.num_outputs());
        for (s, &i) in model.output_neuron_indices().iter().enumerate() {
            let new = new_of[i as usize];
            out_slot[new as usize] = s as u32;
            out_neuron.push(new);
        }

        // Chunk boundaries: greedy on a per-row cost of in-edges + out-edges + 8.
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut r0 = 0usize;
        let mut cost = 0usize;
        for r in 0..n {
            let in_deg = (row_start[r + 1] - row_start[r]) as usize;
            let out_deg = (out_row_start[r + 1] - out_row_start[r]) as usize;
            cost += in_deg + out_deg + 8;
            if cost >= chunk_cost || r + 1 == n {
                chunks.push(Chunk {
                    r0,
                    r1: r + 1,
                    t0: out_row_start[r0] as usize,
                    t1: out_row_start[r + 1] as usize,
                    k0: 0,
                    k1: 0,
                    s0: 0,
                    s1: 0,
                });
                r0 = r + 1;
                cost = 0;
            }
        }

        // Input-rank ranges (ranks ascend with the new index, so a chunk's are contiguous) and
        // local accumulator slots of the shared parameters.
        let shared_count = model.params().a.len();
        let mut local_sid_t = vec![0u32; nnz];
        let mut sid_list: Vec<u32> = Vec::new();
        let mut local_of_global = vec![u32::MAX; shared_count];
        let mut touched: Vec<u32> = Vec::new();
        let mut k_cursor = 0usize;
        for ch in &mut chunks {
            ch.k0 = k_cursor;
            while k_cursor < row_of_ord.len() && row_of_ord[k_cursor] < ch.r1 {
                k_cursor += 1;
            }
            ch.k1 = k_cursor;

            ch.s0 = sid_list.len();
            touched.clear();
            for pos in ch.t0..ch.t1 {
                let g = shared_t[pos] as usize;
                if local_of_global[g] == u32::MAX {
                    local_of_global[g] = (sid_list.len() - ch.s0) as u32;
                    sid_list.push(g as u32);
                    touched.push(g as u32);
                }
                local_sid_t[pos] = local_of_global[g];
            }
            ch.s1 = sid_list.len();
            for &g in &touched {
                local_of_global[g as usize] = u32::MAX;
            }
        }

        let indices = if n <= usize::from(u16::MAX) + 1 {
            Indices::Narrow {
                pre: pre32.iter().map(|&p| p as u16).collect(),
                post: post32.iter().map(|&p| p as u16).collect(),
            }
        } else {
            Indices::Wide {
                pre: pre32,
                post: post32,
            }
        };

        BatchedPlan {
            n,
            nnz,
            num_types: model.num_types(),
            shared_count,
            num_inputs: model.num_inputs(),
            num_outputs: model.num_outputs(),
            gamma_bits: gamma.to_bits(),
            old_of,
            new_of,
            chunks,
            par_min_edge_cells: DEFAULT_PAR_MIN_EDGE_CELLS,
            row_start,
            edge_map,
            out_row_start,
            edge_of,
            indices,
            coeff_t,
            local_sid_t,
            sid_list,
            type_of,
            in_slot,
            in_ord,
            ord_of_k,
            out_slot,
            out_neuron,
        }
    }

    /// Sets the work threshold (`nnz * NB`) below which a substep runs serially instead of on the
    /// rayon pool (`0` = always parallel; see [`DEFAULT_PAR_MIN_EDGE_CELLS`]). Results do not
    /// depend on it.
    pub fn with_parallel_threshold(mut self, edge_cells: usize) -> Self {
        self.par_min_edge_cells = edge_cells;
        self
    }

    /// Number of row chunks (fixed by the graph and the chunk cost, never by the thread count).
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Panics if `model` is not the graph/config this plan was built from (a caller bug: the plan
    /// is topology-only, so a model with the same graph but other parameters is fine).
    pub(crate) fn assert_matches(&self, model: &FlyModel) {
        let flyg = model.flyg();
        assert!(
            self.n == model.num_neurons()
                && self.nnz == flyg.edges.pre_index.len()
                && self.num_types == model.num_types()
                && self.shared_count == model.params().a.len()
                && self.num_inputs == model.num_inputs()
                && self.num_outputs == model.num_outputs()
                && self.gamma_bits == model.config().gamma.to_bits()
                && self.old_of.iter().zip(self.old_of.iter().skip(1)).all(|(&a, &b)| (
                    flyg.neurons[a as usize].type_index,
                    a
                ) < (
                    flyg.neurons[b as usize].type_index,
                    b
                )),
            "BatchedPlan was built for a different graph or gamma than the model it is used with"
        );
    }

    /// Bytes of heap the plan holds.
    pub fn memory_bytes(&self) -> usize {
        use std::mem::size_of;
        self.row_start.len() * 4
            + self.edge_map.len() * 4
            + self.out_row_start.len() * 4
            + self.edge_of.len() * 4
            + match &self.indices {
                Indices::Narrow { pre, post } => (pre.len() + post.len()) * size_of::<u16>(),
                Indices::Wide { pre, post } => (pre.len() + post.len()) * size_of::<u32>(),
            }
            + self.coeff_t.len() * 4
            + self.local_sid_t.len() * 4
            + self.sid_list.len() * 4
            + (self.old_of.len()
                + self.new_of.len()
                + self.type_of.len()
                + self.in_slot.len()
                + self.in_ord.len()
                + self.out_slot.len())
                * 4
    }
}
