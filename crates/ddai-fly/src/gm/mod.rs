//! `Gm`: a FlyGM-style neuron model on the fly's connectome (task 8.8, D-117; the paper is Jin et
//! al., arXiv:2602.17997, see `docs/research/fly-flygm.md` for what was verified and how it maps).
//!
//! Where the rate model ([`crate::state::FlyState`], FLY.md section 4) keeps one scalar `V` per
//! neuron with per-**type** strengths, biases and time constants, `Gm` keeps a `D`-dimensional
//! state `H[v]` per neuron and a **fixed** signed message operator `W` (the connectome: synapse
//! count over the neuron's full-connectome input count, times the presynaptic type's sign). What is
//! learned is a *shared* conditional update network `f_psi` plus a per-key descriptor `eta` (per
//! type, or per L/R-tied homolog group), the injection into the afferent (input) neurons and the
//! readout of the efferent (DN) neurons.
//!
//! One decision runs [`GmConfig::steps`] message-passing steps. Per step (all on `H`, `N x D`):
//!
//! ```text
//! afferent i:  H[i] <- softsign(bg + Wh^T H[i] + inj[type(i)] * e_i)        (the gated injection)
//! M[v]        = sum_u W[v,u] * H[u]                                           (message passing)
//! h1          = relu(W1m^T M[v] + (W1e^T eta[key(v)] + b1))                   (f_psi, layer 1)
//! out         = softsign(W2^T h1 + b2)                                        (f_psi, layer 2)
//! Plain :  H'[v] = out                                                        (the paper's equation)
//! Gated :  H'[v] = H[v] + z * (c - H[v]),  z = (1 + out_z) / 2, c = out_c     (GRU-like, per-neuron memory)
//! ```
//!
//! and the DN readout is `rate[k] = ro_b[T] + ro_w[T] . H[dn_k]` (per DN **type** `T`, tied L/R), a
//! scalar per DN slot, so the whole decoder, the frozen DN calibration and the heads of the rate
//! fly stay as they are: the readout reads **only** the efferent (DN) states.
//!
//! Hot loops are written over `[f32; D]` / `[f32; HD]` / `[f32; O]` const-generic arrays (8 lanes =
//! one AVX2 register; the workspace sets no `target-cpu`, so the compiler emits the baseline SSE2
//! form of the same code) and the layout is `H[v][d]`, so a message is one contiguous `D`-vector.
//! No `unsafe`. [`GmModel::step_once`] allocates nothing.

// The kernels index fixed-size `[f32; N]` arrays with `0..N` loops on purpose: the compiler then sees a constant trip count and bounds, and vectorises.
#![allow(clippy::needless_range_loop)]

mod backward;
mod controls;
#[cfg(test)]
mod tests;

pub use backward::{GmBackwardOut, GmScratch, gm_backward};

use ddai_flyg::{Flyg, NtClassUsed, Sign};
use serde::{Deserialize, Serialize};

use crate::config::FlyConfig;
use crate::error::FlyError;
use crate::rng::{SplitMix64, seeded_for};

/// The largest state dimension the kernels are compiled for (stack arrays in the generic code).
pub const MAX_D: usize = 16;

/// How a neuron's state is updated from its message (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GmUpdate {
    /// `H' = f_psi([M || eta])`, the equation of the paper: no direct dependence on the neuron's own
    /// previous state (the recurrence is only through the graph).
    #[default]
    Plain,
    /// `f_psi` outputs a gate and a candidate: `H' = H + z (c - H)`; the paper's appendix mentions "gated
    /// updates combining previous states and incoming messages", and this gives each neuron memory.
    Gated,
}

/// What the learnable descriptor `eta` is attached to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GmDescriptors {
    /// One descriptor per cell **type** (left and right copies share it, FLY.md section 4).
    #[default]
    PerType,
    /// One per neuron, with homologs tied: neurons that have the same `(type, group_id)` (the L/R
    /// copy of a cell) share one descriptor; a neuron without a group keeps its own. The paper's
    /// per-neuron descriptors, made mirror-friendly.
    PerNeuronTied,
}

/// Which presynaptic signs the fixed message operator uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GmSigns {
    /// Our NT rules (FLY.md section 4): ACh `+`, GABA/Glu/histamine `-`, modulators `0`.
    #[default]
    Rules,
    /// FlyGM's rule: glutamate counts as excitatory (ACh, Glu, Asp, His `+`; GABA, Gly `-`); only the
    /// glutamate types differ from `Rules` (the graph does not carry the histamine/aspartate split).
    GluExcitatory,
    /// **Control** (FLY.md section 1.3b): the type signs permuted at random (the same number of `+`, `-`, `0`).
    Shuffled { seed: u64 },
}

/// The wiring of the message operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GmWiring {
    /// The connectome.
    #[default]
    Connectome,
    /// **Control** (FLY.md section 1.3a, the paper's "DP Rewiring"): double edge swaps that keep
    /// every neuron's in-degree and out-degree and the synapse counts of its outgoing edges.
    DegreePreserving { seed: u64 },
}

/// The hyper-parameters of the `Gm` neuron model (part of the checkpoint, so a bundle replays exactly).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GmConfig {
    /// State dimension `D` per neuron: 8 or 16.
    pub d: u32,
    /// Hidden width of `f_psi`: 8, 16 or 32.
    pub hidden: u32,
    /// Message-passing steps per decision (1 is the paper's equation; more steps = more hops per decision).
    pub steps: u32,
    pub update: GmUpdate,
    pub descriptors: GmDescriptors,
    pub signs: GmSigns,
    pub wiring: GmWiring,
    /// A fixed scalar on `W` (the rate model's `alpha_init` plays this role there).
    pub msg_gain: f32,
}

impl Default for GmConfig {
    fn default() -> Self {
        GmConfig {
            d: 8,
            hidden: 16,
            steps: 2,
            update: GmUpdate::Plain,
            descriptors: GmDescriptors::PerType,
            signs: GmSigns::Rules,
            wiring: GmWiring::Connectome,
            msg_gain: 4.0,
        }
    }
}

impl GmConfig {
    pub fn validate(&self) -> Result<(), FlyError> {
        let bad = |m: String| Err(FlyError::InvalidConfig(format!("gm: {m}")));
        if !matches!(self.d, 8 | 16) {
            return bad(format!("d must be 8 or 16, got {}", self.d));
        }
        if !matches!(self.hidden, 8 | 16 | 32) {
            return bad(format!("hidden must be 8, 16 or 32, got {}", self.hidden));
        }
        if !(1..=8).contains(&self.steps) {
            return bad(format!("steps must be in 1..=8, got {}", self.steps));
        }
        if !(self.msg_gain.is_finite() && self.msg_gain > 0.0) {
            return bad(format!("msg_gain must be > 0 and finite, got {}", self.msg_gain));
        }
        Ok(())
    }

    /// Output width of `f_psi`: `D` (plain) or `2 D` (gate and candidate).
    pub fn out_dim(&self) -> usize {
        match self.update {
            GmUpdate::Plain => self.d as usize,
            GmUpdate::Gated => 2 * self.d as usize,
        }
    }

    /// A short label for logs and tables (`gm-d8-h16-k2-plain`, plus the control, if any).
    pub fn label(&self) -> String {
        let mut s = format!(
            "gm-d{}-h{}-k{}-{}",
            self.d,
            self.hidden,
            self.steps,
            match self.update {
                GmUpdate::Plain => "plain",
                GmUpdate::Gated => "gated",
            }
        );
        if matches!(self.descriptors, GmDescriptors::PerNeuronTied) {
            s.push_str("-tied");
        }
        match self.signs {
            GmSigns::Rules => {}
            GmSigns::GluExcitatory => s.push_str("-gluexc"),
            GmSigns::Shuffled { seed } => s.push_str(&format!("-shuf{seed}")),
        }
        if let GmWiring::DegreePreserving { seed } = self.wiring {
            s.push_str(&format!("-rewired{seed}"));
        }
        s
    }
}

/// How many entries of each parameter group a model has (derived from the graph and [`GmConfig`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GmShape {
    pub d: usize,
    pub hd: usize,
    pub o: usize,
    /// Descriptor keys (types, or tied homolog groups).
    pub keys: usize,
    /// Distinct afferent (input) types: each has its own injection direction.
    pub aff_types: usize,
    /// Distinct DN types: each has its own readout vector.
    pub ro_types: usize,
}

impl GmShape {
    /// Lengths of the eleven groups of [`GmParams`], in the order of [`GmParams::fields`].
    pub fn lens(&self) -> [usize; 11] {
        let (d, hd, o) = (self.d, self.hd, self.o);
        [
            self.keys * d,
            d * hd,
            d * hd,
            hd,
            hd * o,
            o,
            d * d,
            d,
            self.aff_types * d,
            self.ro_types * d,
            self.ro_types,
        ]
    }

    pub fn total(&self) -> usize {
        self.lens().iter().sum()
    }
}

/// The trainable parameters of the `Gm` model. Gradients have the same type and shape.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GmParams {
    /// Descriptors `eta`, `[key][D]`.
    pub eta: Vec<f32>,
    /// First layer of `f_psi`, message part, `[D][HD]`.
    pub w1m: Vec<f32>,
    /// First layer, descriptor part, `[D][HD]`.
    pub w1e: Vec<f32>,
    pub b1: Vec<f32>,
    /// Second layer, `[HD][O]`.
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    /// Injection gate on the afferent's previous state, `[D][D]` (row = input component).
    pub wh: Vec<f32>,
    pub bg: Vec<f32>,
    /// Injection direction of the scalar input current, per afferent type, `[aff_type][D]`.
    pub inj: Vec<f32>,
    /// DN readout vector per DN type, `[ro_type][D]`.
    pub ro_w: Vec<f32>,
    pub ro_b: Vec<f32>,
}

impl GmParams {
    pub fn zeros(shape: &GmShape) -> Self {
        let l = shape.lens();
        GmParams {
            eta: vec![0.0; l[0]],
            w1m: vec![0.0; l[1]],
            w1e: vec![0.0; l[2]],
            b1: vec![0.0; l[3]],
            w2: vec![0.0; l[4]],
            b2: vec![0.0; l[5]],
            wh: vec![0.0; l[6]],
            bg: vec![0.0; l[7]],
            inj: vec![0.0; l[8]],
            ro_w: vec![0.0; l[9]],
            ro_b: vec![0.0; l[10]],
        }
    }

    pub fn fields(&self) -> [&Vec<f32>; 11] {
        [
            &self.eta, &self.w1m, &self.w1e, &self.b1, &self.w2, &self.b2, &self.wh, &self.bg, &self.inj, &self.ro_w,
            &self.ro_b,
        ]
    }

    pub fn fields_mut(&mut self) -> [&mut Vec<f32>; 11] {
        [
            &mut self.eta,
            &mut self.w1m,
            &mut self.w1e,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
            &mut self.wh,
            &mut self.bg,
            &mut self.inj,
            &mut self.ro_w,
            &mut self.ro_b,
        ]
    }

    pub fn len(&self) -> usize {
        self.fields().iter().map(|f| f.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The parameters as one flat vector (field order), the trainer's view.
    pub fn to_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len());
        for f in self.fields() {
            out.extend_from_slice(f);
        }
        out
    }

    /// Inverse of [`GmParams::to_flat`]; `flat.len()` must equal `shape.total()`.
    pub fn from_flat(shape: &GmShape, flat: &[f32]) -> Result<Self, FlyError> {
        if flat.len() != shape.total() {
            return Err(FlyError::ParamShapeMismatch(format!(
                "gm: expected {} parameters, got {}",
                shape.total(),
                flat.len()
            )));
        }
        let mut p = GmParams::zeros(shape);
        let mut at = 0;
        for f in p.fields_mut() {
            let n = f.len();
            f.copy_from_slice(&flat[at..at + n]);
            at += n;
        }
        Ok(p)
    }

    /// `self += other` (same shape).
    pub fn add_assign(&mut self, other: &GmParams) {
        for (d, s) in self.fields_mut().into_iter().zip(other.fields()) {
            for (a, b) in d.iter_mut().zip(s) {
                *a += b;
            }
        }
    }

    pub fn validate_shape(&self, shape: &GmShape) -> Result<(), FlyError> {
        const NAMES: [&str; 11] = ["eta", "w1m", "w1e", "b1", "w2", "b2", "wh", "bg", "inj", "ro_w", "ro_b"];
        for ((f, &want), name) in self.fields().into_iter().zip(shape.lens().iter()).zip(NAMES) {
            if f.len() != want {
                return Err(FlyError::ParamShapeMismatch(format!(
                    "gm: {name}.len() == {} but the model needs {want}",
                    f.len()
                )));
            }
            if f.iter().any(|x| !x.is_finite()) {
                return Err(FlyError::ParamShapeMismatch(format!(
                    "gm: {name} contains NaN or infinity"
                )));
            }
        }
        Ok(())
    }

    /// The deterministic default initialisation (same seed, same numbers): descriptors and the
    /// first-layer descriptor weights `N(0, 1)` / `N(0, 1/D)` (so the per-key offset of the hidden units
    /// is `O(1)` and the neurons differ at the start), the rest scaled `1/sqrt(fan_in)`, biases zero.
    pub fn init(shape: &GmShape, seed: u64) -> Self {
        let mut p = GmParams::zeros(shape);
        let (d, hd) = (shape.d as f32, shape.hd as f32);
        let scales = [
            1.0,
            1.0 / d.sqrt(),
            1.0 / d.sqrt(),
            0.0,
            1.0 / hd.sqrt(),
            0.0,
            0.1 / d.sqrt(),
            0.0,
            1.0,
            1.0 / d.sqrt(),
            0.0,
        ];
        for (gi, (f, scale)) in p.fields_mut().into_iter().zip(scales).enumerate() {
            let mut rng: SplitMix64 = seeded_for(seed, 0x6D00 + gi as u64);
            for x in f.iter_mut() {
                *x = if scale == 0.0 { 0.0 } else { scale * rng.next_gaussian() };
            }
        }
        p
    }
}

/// `x / (1 + |x|)`: the bounded, smooth squashing used instead of `tanh` (a libm call per unit
/// would dominate the step; `softsign` is one division and vectorises). Derivative from the output
/// `y`: `(1 - |y|)^2`.
#[inline(always)]
pub(crate) fn softsign(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// The derivative of [`softsign`] expressed by its output.
#[inline(always)]
pub(crate) fn softsign_grad_from_out(y: f32) -> f32 {
    let t = 1.0 - y.abs();
    t * t
}

/// The fixed topology of a `Gm` model and its parameters, with everything derived precomputed.
#[derive(Debug, Clone)]
pub struct GmModel {
    config: GmConfig,
    shape: GmShape,
    params: GmParams,
    n: usize,

    /// Post-major CSR (parallel arrays): possibly rewired (control), signs and gain folded into `w`.
    row_start: Vec<u32>,
    pre: Vec<u32>,
    w: Vec<f32>,
    /// The same edges pre-major (the transpose, for the backward pass): `t_post`, `t_w` per presynaptic neuron.
    t_row_start: Vec<u32>,
    t_post: Vec<u32>,
    t_w: Vec<f32>,

    /// Descriptor key of every neuron.
    key_of: Vec<u32>,
    /// Afferent (input) neurons in input order and their injection slot.
    aff: Vec<u32>,
    aff_slot: Vec<u32>,
    /// DN (output) neurons in output order and their readout slot.
    dn: Vec<u32>,
    ro_slot: Vec<u32>,
    /// Type of every neuron and the number of neurons per type (for the per-type activity summary).
    type_of: Vec<u32>,
    type_count: Vec<f32>,

    /// `b1 + W1e^T eta[key(v)]` per neuron `[N][HD]` (derived from the parameters, recomputed by `set_params`).
    t_pre1: Vec<f32>,
}

impl GmModel {
    /// Builds the model for a graph. `inputs` / `outputs` are the dense indices of the input and DN
    /// neurons in the order the encoder and decoder use ([`crate::model::FlyModel`]).
    pub fn build(
        flyg: &Flyg,
        fly_config: &FlyConfig,
        inputs: &[u32],
        outputs: &[u32],
        config: GmConfig,
        params: Option<GmParams>,
        init_seed: u64,
    ) -> Result<Self, FlyError> {
        config.validate()?;
        let n = flyg.neurons.len();
        let edges = &flyg.edges;

        // Signs per presynaptic type.
        let mut type_sign: Vec<f32> = flyg.types.iter().map(|t| f32::from(t.sign.as_i8())).collect();
        match config.signs {
            GmSigns::Rules => {}
            GmSigns::GluExcitatory => {
                for (s, t) in type_sign.iter_mut().zip(&flyg.types) {
                    if t.nt_class_used == NtClassUsed::Glutamate && t.sign == Sign::Inhibitory {
                        *s = 1.0;
                    }
                }
            }
            GmSigns::Shuffled { seed } => controls::shuffle_in_place(&mut type_sign, seed),
        }

        // Edges, possibly rewired (the control).
        let (pre, counts) = match config.wiring {
            GmWiring::Connectome => (edges.pre_index.clone(), edges.synapse_count.clone()),
            GmWiring::DegreePreserving { seed } => {
                controls::rewire_degree_preserving(&edges.row_start, &edges.pre_index, &edges.synapse_count, seed)
            }
        };
        let row_start = edges.row_start.clone();
        let inv_z: Vec<f32> = flyg
            .neuron_input_totals
            .full_connectome
            .iter()
            .map(|&z| 1.0 / (z.max(1) as f32).powf(fly_config.gamma))
            .collect();
        let mut w = vec![0.0f32; pre.len()];
        for post in 0..n {
            for e in row_start[post] as usize..row_start[post + 1] as usize {
                let sign = type_sign[flyg.neurons[pre[e] as usize].type_index as usize];
                w[e] = config.msg_gain * sign * counts[e] as f32 * inv_z[post];
            }
        }
        // The transpose (counting sort by presynaptic neuron).
        let mut t_row_start = vec![0u32; n + 1];
        for &p in &pre {
            t_row_start[p as usize + 1] += 1;
        }
        for i in 0..n {
            t_row_start[i + 1] += t_row_start[i];
        }
        let mut cursor = t_row_start.clone();
        let mut t_post = vec![0u32; pre.len()];
        let mut t_w = vec![0.0f32; pre.len()];
        for post in 0..n {
            for e in row_start[post] as usize..row_start[post + 1] as usize {
                let p = pre[e] as usize;
                let at = cursor[p] as usize;
                t_post[at] = post as u32;
                t_w[at] = w[e];
                cursor[p] += 1;
            }
        }

        // Descriptor keys.
        let (key_of, keys) = match config.descriptors {
            GmDescriptors::PerType => (
                flyg.neurons.iter().map(|x| x.type_index).collect::<Vec<_>>(),
                flyg.types.len(),
            ),
            GmDescriptors::PerNeuronTied => controls::tied_keys(flyg),
        };

        // Afferent and DN slots.
        let slot_map = |idx: &[u32]| -> (Vec<u32>, usize) {
            let mut types: Vec<u32> = idx.iter().map(|&i| flyg.neurons[i as usize].type_index).collect();
            types.sort_unstable();
            types.dedup();
            let slots = idx
                .iter()
                .map(|&i| {
                    let t = flyg.neurons[i as usize].type_index;
                    types.binary_search(&t).expect("type is in the list") as u32
                })
                .collect();
            (slots, types.len())
        };
        let (aff_slot, aff_types) = slot_map(inputs);
        let (ro_slot, ro_types) = slot_map(outputs);

        let shape = GmShape {
            d: config.d as usize,
            hd: config.hidden as usize,
            o: config.out_dim(),
            keys,
            aff_types,
            ro_types,
        };
        let params = params.unwrap_or_else(|| GmParams::init(&shape, init_seed));
        params.validate_shape(&shape)?;

        let type_of: Vec<u32> = flyg.neurons.iter().map(|x| x.type_index).collect();
        let mut type_count = vec![0.0f32; flyg.types.len()];
        for &t in &type_of {
            type_count[t as usize] += 1.0;
        }
        let mut model = GmModel {
            config,
            shape,
            params,
            n,
            row_start,
            pre,
            w,
            t_row_start,
            t_post,
            t_w,
            key_of,
            aff: inputs.to_vec(),
            aff_slot,
            dn: outputs.to_vec(),
            ro_slot,
            type_of,
            type_count,
            t_pre1: vec![0.0; n * config.hidden as usize],
        };
        model.recompute();
        Ok(model)
    }

    fn recompute(&mut self) {
        let (d, hd) = (self.shape.d, self.shape.hd);
        let p = &self.params;
        for v in 0..self.n {
            let key = self.key_of[v] as usize;
            let eta = &p.eta[key * d..(key + 1) * d];
            let out = &mut self.t_pre1[v * hd..(v + 1) * hd];
            out.copy_from_slice(&p.b1);
            for (k, &e) in eta.iter().enumerate() {
                let row = &p.w1e[k * hd..(k + 1) * hd];
                for (o, &wv) in out.iter_mut().zip(row) {
                    *o += e * wv;
                }
            }
        }
    }

    /// Replaces the parameters (shape-checked) and recomputes what is derived from them.
    pub fn set_params(&mut self, params: GmParams) -> Result<(), FlyError> {
        params.validate_shape(&self.shape)?;
        self.params = params;
        self.recompute();
        Ok(())
    }

    pub fn config(&self) -> &GmConfig {
        &self.config
    }
    pub fn shape(&self) -> &GmShape {
        &self.shape
    }
    pub fn params(&self) -> &GmParams {
        &self.params
    }
    /// `N * D`: the length of the state vector.
    pub fn state_len(&self) -> usize {
        self.n * self.shape.d
    }
    pub fn num_neurons(&self) -> usize {
        self.n
    }
    pub fn num_afferents(&self) -> usize {
        self.aff.len()
    }
    pub fn num_outputs(&self) -> usize {
        self.dn.len()
    }
    pub fn num_edges(&self) -> usize {
        self.pre.len()
    }
    /// Multiply-adds of one message-passing step (messages plus `f_psi`), for the cost note.
    pub fn macs_per_step(&self) -> usize {
        let (d, hd, o) = (self.shape.d, self.shape.hd, self.shape.o);
        self.num_edges() * d + self.n * (d * hd + hd * o) + self.aff.len() * (d * d + d)
    }
    /// The signed, scaled edge weights (post-major, parallel to the graph's CSR), for tests and reports.
    pub fn edge_weights(&self) -> &[f32] {
        &self.w
    }
    pub fn edge_sources(&self) -> &[u32] {
        &self.pre
    }
    pub fn row_start(&self) -> &[u32] {
        &self.row_start
    }

    /// One message-passing step: `h` (the state entering, modified by the injection) -> `next`.
    /// `inputs[k]` is the current of the `k`-th afferent. Records into `rec` when given.
    pub(crate) fn step_once(&self, h: &mut [f32], next: &mut [f32], inputs: &[f32], rec: Option<&mut StepRec<'_>>) {
        let gated = self.config.update == GmUpdate::Gated;
        macro_rules! go {
            ($d:literal, $hd:literal, $o:literal) => {
                self.step_impl::<$d, $hd, $o>(h, next, inputs, rec)
            };
        }
        match (self.shape.d, self.shape.hd, gated) {
            (8, 8, false) => go!(8, 8, 8),
            (8, 8, true) => go!(8, 8, 16),
            (8, 16, false) => go!(8, 16, 8),
            (8, 16, true) => go!(8, 16, 16),
            (8, 32, false) => go!(8, 32, 8),
            (8, 32, true) => go!(8, 32, 16),
            (16, 8, false) => go!(16, 8, 16),
            (16, 8, true) => go!(16, 8, 32),
            (16, 16, false) => go!(16, 16, 16),
            (16, 16, true) => go!(16, 16, 32),
            (16, 32, false) => go!(16, 32, 16),
            (16, 32, true) => go!(16, 32, 32),
            _ => unreachable!("GmConfig::validate admits only d in {{8,16}}, hidden in {{8,16,32}}"),
        }
    }

    fn step_impl<const D: usize, const HD: usize, const O: usize>(
        &self,
        h: &mut [f32],
        next: &mut [f32],
        inputs: &[f32],
        mut rec: Option<&mut StepRec<'_>>,
    ) {
        debug_assert_eq!(h.len(), self.n * D);
        debug_assert_eq!(next.len(), self.n * D);
        debug_assert_eq!(inputs.len(), self.aff.len());
        let gated = O != D;
        let p = &self.params;
        if let Some(r) = rec.as_deref_mut() {
            r.h_pre.copy_from_slice(h);
        }

        // 1. The injection into the afferent neurons.
        for (k, &i) in self.aff.iter().enumerate() {
            let i = i as usize;
            let slot = self.aff_slot[k] as usize;
            let e = inputs[k];
            let row: &mut [f32; D] = (&mut h[i * D..(i + 1) * D]).try_into().expect("row");
            let mut pre = [0.0f32; D];
            pre.copy_from_slice(&p.bg);
            for kk in 0..D {
                let hk = row[kk];
                let w: &[f32; D] = p.wh[kk * D..(kk + 1) * D].try_into().expect("wh row");
                for d in 0..D {
                    pre[d] += hk * w[d];
                }
            }
            let dir: &[f32; D] = p.inj[slot * D..(slot + 1) * D].try_into().expect("inj row");
            for d in 0..D {
                row[d] = softsign(pre[d] + e * dir[d]);
            }
        }
        if let Some(r) = rec.as_deref_mut() {
            r.h_inj.copy_from_slice(h);
        }

        // 2. Messages and the update network, neuron by neuron.
        let h: &[f32] = h;
        for v in 0..self.n {
            let mut m = [0.0f32; D];
            for e in self.row_start[v] as usize..self.row_start[v + 1] as usize {
                let u = self.pre[e] as usize;
                let w = self.w[e];
                let hu: &[f32; D] = h[u * D..(u + 1) * D].try_into().expect("h row");
                for d in 0..D {
                    m[d] += w * hu[d];
                }
            }
            let mut a1 = [0.0f32; HD];
            a1.copy_from_slice(&self.t_pre1[v * HD..(v + 1) * HD]);
            for k in 0..D {
                let mk = m[k];
                let row: &[f32; HD] = p.w1m[k * HD..(k + 1) * HD].try_into().expect("w1m row");
                for j in 0..HD {
                    a1[j] += mk * row[j];
                }
            }
            for x in a1.iter_mut() {
                *x = x.max(0.0);
            }
            let mut out = [0.0f32; O];
            out.copy_from_slice(&p.b2);
            for j in 0..HD {
                let hj = a1[j];
                let row: &[f32; O] = p.w2[j * O..(j + 1) * O].try_into().expect("w2 row");
                for o in 0..O {
                    out[o] += hj * row[o];
                }
            }
            for x in out.iter_mut() {
                *x = softsign(*x);
            }
            if let Some(r) = rec.as_deref_mut() {
                r.m[v * D..(v + 1) * D].copy_from_slice(&m);
                r.h1[v * HD..(v + 1) * HD].copy_from_slice(&a1);
                r.a[v * O..(v + 1) * O].copy_from_slice(&out);
            }
            let nrow = &mut next[v * D..(v + 1) * D];
            if gated {
                let hv: &[f32; D] = h[v * D..(v + 1) * D].try_into().expect("h row");
                for d in 0..D {
                    let z = 0.5 + 0.5 * out[d];
                    nrow[d] = hv[d] + z * (out[D + d] - hv[d]);
                }
            } else {
                nrow.copy_from_slice(&out[..D]);
            }
        }
    }

    /// `rate[k] = ro_b[T] + ro_w[T] . H[dn_k]` for every DN slot.
    pub(crate) fn read_out(&self, h: &[f32], out: &mut [f32]) {
        let d = self.shape.d;
        let p = &self.params;
        for (k, (&i, &slot)) in self.dn.iter().zip(&self.ro_slot).enumerate() {
            let i = i as usize;
            let slot = slot as usize;
            let hv = &h[i * d..(i + 1) * d];
            let w = &p.ro_w[slot * d..(slot + 1) * d];
            let mut s = p.ro_b[slot];
            for (a, b) in hv.iter().zip(w) {
                s += a * b;
            }
            out[k] = s;
        }
    }

    /// Mean `|H|` per type (a stand-in for the rate model's per-type mean rate: the viewer's activity).
    pub(crate) fn type_activity(&self, h: &[f32], sum: &mut [f32], mean: &mut [f32]) {
        let d = self.shape.d;
        sum.fill(0.0);
        for v in 0..self.n {
            let row = &h[v * d..(v + 1) * d];
            let a: f32 = row.iter().map(|x| x.abs()).sum();
            sum[self.type_of[v] as usize] += a / d as f32;
        }
        for ((m, s), c) in mean.iter_mut().zip(sum.iter()).zip(&self.type_count) {
            *m = *s / c.max(1.0);
        }
    }
}

/// Mutable views into a [`GmRecorder`] for one step.
pub(crate) struct StepRec<'a> {
    pub h_pre: &'a mut [f32],
    pub h_inj: &'a mut [f32],
    pub m: &'a mut [f32],
    pub h1: &'a mut [f32],
    pub a: &'a mut [f32],
}

/// The trajectory a `Gm` window records for its backward pass: per step the entering state, the
/// state after the injection, the messages, the hidden activations and the squashed outputs; per
/// decision the afferent currents and the DN states the readout saw.
#[derive(Debug, Clone)]
pub struct GmRecorder {
    n: usize,
    d: usize,
    hd: usize,
    o: usize,
    n_aff: usize,
    n_out: usize,
    cap_steps: usize,
    steps: usize,
    h_pre: Vec<f32>,
    h_inj: Vec<f32>,
    m: Vec<f32>,
    h1: Vec<f32>,
    a: Vec<f32>,
    cap_decisions: usize,
    decisions: usize,
    inputs: Vec<f32>,
    dn_h: Vec<f32>,
}

impl GmRecorder {
    /// Room for `max_decisions` decisions of `model.config().steps` steps.
    pub fn new(model: &GmModel, max_decisions: usize) -> Self {
        let (n, d, hd, o) = (model.n, model.shape.d, model.shape.hd, model.shape.o);
        let cap_steps = max_decisions * model.config.steps as usize;
        GmRecorder {
            n,
            d,
            hd,
            o,
            n_aff: model.aff.len(),
            n_out: model.dn.len(),
            cap_steps,
            steps: 0,
            h_pre: vec![0.0; cap_steps * n * d],
            h_inj: vec![0.0; cap_steps * n * d],
            m: vec![0.0; cap_steps * n * d],
            h1: vec![0.0; cap_steps * n * hd],
            a: vec![0.0; cap_steps * n * o],
            cap_decisions: max_decisions,
            decisions: 0,
            inputs: vec![0.0; max_decisions * model.aff.len()],
            dn_h: vec![0.0; max_decisions * model.dn.len() * d],
        }
    }

    pub fn reset(&mut self) {
        self.steps = 0;
        self.decisions = 0;
    }

    pub fn decisions(&self) -> usize {
        self.decisions
    }

    pub(crate) fn next_step(&mut self) -> StepRec<'_> {
        assert!(self.steps < self.cap_steps, "GmRecorder is full");
        let (n, d, hd, o) = (self.n, self.d, self.hd, self.o);
        let s = self.steps;
        self.steps += 1;
        StepRec {
            h_pre: &mut self.h_pre[s * n * d..(s + 1) * n * d],
            h_inj: &mut self.h_inj[s * n * d..(s + 1) * n * d],
            m: &mut self.m[s * n * d..(s + 1) * n * d],
            h1: &mut self.h1[s * n * hd..(s + 1) * n * hd],
            a: &mut self.a[s * n * o..(s + 1) * n * o],
        }
    }

    pub(crate) fn record_decision(&mut self, model: &GmModel, inputs: &[f32], h_final: &[f32]) {
        assert!(self.decisions < self.cap_decisions, "GmRecorder is full");
        let t = self.decisions;
        self.decisions += 1;
        self.inputs[t * self.n_aff..(t + 1) * self.n_aff].copy_from_slice(inputs);
        let d = self.d;
        for (k, &i) in model.dn.iter().enumerate() {
            let i = i as usize;
            self.dn_h[(t * self.n_out + k) * d..(t * self.n_out + k + 1) * d]
                .copy_from_slice(&h_final[i * d..(i + 1) * d]);
        }
    }

    pub(crate) fn step_views(&self, s: usize) -> StepView<'_> {
        let (n, d, hd, o) = (self.n, self.d, self.hd, self.o);
        StepView {
            h_pre: &self.h_pre[s * n * d..(s + 1) * n * d],
            h_inj: &self.h_inj[s * n * d..(s + 1) * n * d],
            m: &self.m[s * n * d..(s + 1) * n * d],
            h1: &self.h1[s * n * hd..(s + 1) * n * hd],
            a: &self.a[s * n * o..(s + 1) * n * o],
        }
    }

    pub(crate) fn decision_inputs(&self, t: usize) -> &[f32] {
        &self.inputs[t * self.n_aff..(t + 1) * self.n_aff]
    }

    pub(crate) fn decision_dn_h(&self, t: usize) -> &[f32] {
        let w = self.n_out * self.d;
        &self.dn_h[t * w..(t + 1) * w]
    }
}

pub(crate) struct StepView<'a> {
    pub h_pre: &'a [f32],
    pub h_inj: &'a [f32],
    pub m: &'a [f32],
    pub h1: &'a [f32],
    pub a: &'a [f32],
}
