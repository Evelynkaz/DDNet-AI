//! The DN action decoder (task 7.3, acceptance criterion 4): frozen per-DN calibration
//! ([`DnCalibration`]) plus a handful of small heads reading a configurable DN-group subset each
//! ([`DecoderConfig`], resolved from `.flyg`'s real `output_groups` — FLY.md §6's action table,
//! task 6.3's output), turning calibrated DN z-scores into direction/jump/hook/fire/aim. Losses
//! here (cross-entropy, binary cross-entropy, von Mises NLL) all produce `dL/d(dn_rates)` in
//! [`crate::backward::backward`]'s exact input shape, so a caller trains the decoder and the
//! connectome together through one `backward` call — see `tests/gradcheck_brain.rs` for the
//! end-to-end check.
//!
//! ## Calibration protocol (acceptance criterion 4)
//! `z_i = clip((r_i - μ_i) / σ_i, ±10)`, `μ`/`σ` frozen per DN slot (never trained,
//! [`DnCalibration::fit`]'s job) — see [`crate::calibrate_from_rest`] for the one, documented
//! protocol every caller (`ddnet-ai fly brain-demo`, this module's own tests) actually uses to
//! produce the samples it's fit from (review round 1, F10, CONFIRMED: an earlier revision left
//! this as prose here and a *different*, undocumented recipe in the CLI).
//!
//! ## Mirror-symmetric heads (review round 1, F6, CONFIRMED — acceptance criterion 8)
//! The connectome's own `a`/`b`/`theta` are shared between a type's `L`/`R` copies because the
//! `.flyg` format's `shared_param_id`/`type_index` structurally force it; nothing forced this
//! module's `W_dec`/preferred angles to follow suit, so an earlier revision's dense, untied
//! weights broke criterion 8's "mirrored observation -> mirrored action distribution" the moment
//! decoder training moved a weight even slightly off symmetric. Every head here is now tied by
//! construction, built once in [`DecoderModel::new`] from the real `.flyg`'s `output_groups`
//! (task 6.3) and each member's own `side`:
//! - **`direction_left`/`direction_right`**: for every type present in *both* groups (on the real
//!   S/M graphs, every type is — `DNa01`/`DNa02`/`DNg13`/`DNp09`, one `L` + one `R` instance
//!   each), one weight `w_lr[type]` and one shared bias `b_lr` compute **both**
//!   `left_logit = b_lr + Σ_type w_lr[type] · mean(z[left_i(type)])` and the *same* formula with
//!   `right_i(type)` for `right_logit` — so `right_logit(z) == left_logit(mirror(z))` exactly
//!   (mirroring swaps a type's `L`/`R` rates, and the formula treats both sides identically).
//! - **`direction_stop`**: one weight per type, applied to the **mean over that type's members on
//!   both sides pooled together** — invariant to an `L`/`R` swap by construction (a sum/mean
//!   doesn't care which side contributed which value).
//! - **`jump`/`hook`/`fire`**: the same "one weight per type, both sides pooled" shape as `stop` —
//!   these actions have no inherent laterality (task spec's own scripted teacher never asks for
//!   "jump left"), so a mirrored observation must decode to the *same* probability, which pooling
//!   guarantees.
//! - **`aim`**: every type's `L`/`R` members are paired up (positionally, by ascending dense
//!   neuron index, when a type has more than one per side); a pair's `L` preferred angle `θ_L` is
//!   the free parameter, `θ_R` is *derived* as `π - θ_L`, never stored — see
//!   [`AimPair`]/[`population_vector`] for the exact math and why this makes the population
//!   vector's `(C, S)` transform exactly as a mirrored bearing would. Any member with no
//!   same-type opposite-side partner (side `M`/`Unknown`, or a type with unequal `L`/`R` counts —
//!   neither occurs on the real S/M graphs, but the code doesn't assume it) falls back to its own
//!   independent, untied angle (label **П**: no biological homolog to tie it to) — **not**
//!   mirror-symmetric on its own (there is no opposite-side counterpart for a mirrored observation
//!   to symmetrically swap it with), so the aim head's overall mirror guarantee below is exact
//!   only when every member is tied, which holds on both real graphs (neither has an unpaired aim
//!   member) but not in general.
//!
//! `tests::mirror` (this module) and `tests/brain_mirror.rs` (real S, action-level) check this
//! directly: on a hand-built, exactly `L`/`R`-symmetric graph, decoding a mirrored `z` gives the
//! mirrored [`DecodedAction`] to within float noise.
//!
//! ## Aim head: population vector, von Mises NLL (acceptance criterion 4's explicit choice)
//! FLY.md §6 describes the aim ensemble as a "population vector ... (обучаемые предпочтительные
//! углы, фон Мизес)" — a fan of DN neurons, each with a **learnable preferred angle**, decoded via
//! the classic population-vector estimator (Georgopoulos et al. 1986; also literally how
//! AOTU019/025 -> DNa02/03/06/13/15/16/DNg04's own "fan" structure motivates this row of the
//! table). The loss is a von Mises NLL of the *target* angle under mean direction `μ = atan2(S,
//! C)` and a **fixed** (not learnable — FLY.md §6 only calls the preferred angles learnable)
//! concentration `κ` ([`DecoderConfig::aim_kappa`]): `NLL = -κ·cos(y - μ)` (the `log(2π·I0(κ))`
//! normalizing term is dropped — it doesn't depend on any trainable quantity when `κ` is fixed).
//!
//! **A population vector, not a von Mises mixture:** the task spec allows either "choose and
//! justify" — a *mixture* would need extra machinery (per-component weights/normalization) that
//! nothing in FLY.md §6's description of a single fan of neurons motivates.
//!
//! The one numerically delicate point (documented, same "kink -> subgradient 0" convention as
//! `activation_derivative`'s relu kink elsewhere in this crate): `μ = atan2(S, C)` is undefined at
//! `(C, S) = (0, 0)`; [`aim_loss_and_grad`] returns a zero gradient into `C`/`S` there rather than
//! propagating a `NaN`.

use serde::{Deserialize, Serialize};

use crate::model::FlyModel;
use crate::rng::SplitMix64;
use crate::state::FlyState;

// --- Calibration ------------------------------------------------------------------------------

/// Frozen per-DN calibration (`μ`, `σ`), one entry per output slot (`model.num_outputs()`'s order
/// — [`FlyModel::output_neuron_indices`]). Never updated by gradient descent; see
/// [`crate::calibrate_from_rest`] for the one documented protocol every caller uses to produce the
/// samples [`DnCalibration::fit`] is fit from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DnCalibration {
    pub mu: Vec<f32>,
    pub sigma: Vec<f32>,
}

impl DnCalibration {
    /// Fits `mu`/`sigma` from `samples` (each `model.num_outputs()` long — a resting-run
    /// `DecisionOutput::dn_rates` snapshot; see [`crate::calibrate_from_rest`] for how to collect
    /// them). `sigma` is floored at `min_sigma` (never `0`, which would make `z` blow up or
    /// divide by zero for a DN slot whose resting rate happens to be perfectly constant across
    /// every sample — e.g. a dead or fully-saturated neuron).
    pub fn fit(num_outputs: usize, samples: &[Vec<f32>], min_sigma: f32) -> Result<Self, DecoderError> {
        if samples.is_empty() {
            return Err(DecoderError::InvalidConfig(
                "DnCalibration::fit needs at least one sample".to_string(),
            ));
        }
        for s in samples {
            if s.len() != num_outputs {
                return Err(DecoderError::InvalidConfig(format!(
                    "sample length {} != num_outputs {num_outputs}",
                    s.len()
                )));
            }
        }
        let n = samples.len() as f64;
        let mut mu = vec![0.0f64; num_outputs];
        for s in samples {
            for (m, &v) in mu.iter_mut().zip(s) {
                *m += f64::from(v);
            }
        }
        for m in &mut mu {
            *m /= n;
        }
        let mut var = vec![0.0f64; num_outputs];
        for s in samples {
            for (v_acc, (&v, &m)) in var.iter_mut().zip(s.iter().zip(&mu)) {
                let d = f64::from(v) - m;
                *v_acc += d * d;
            }
        }
        let sigma: Vec<f32> = var.iter().map(|&v| ((v / n).sqrt() as f32).max(min_sigma)).collect();
        Ok(DnCalibration {
            mu: mu.iter().map(|&m| m as f32).collect(),
            sigma,
        })
    }

    /// `z_i = clip((r_i - mu_i) / sigma_i, +-clip_at)`, written into `out` (allocation-free path).
    pub fn z_into(&self, dn_rates: &[f32], clip_at: f32, out: &mut [f32]) {
        for (((o, &r), &mu), &sigma) in out.iter_mut().zip(dn_rates).zip(&self.mu).zip(&self.sigma) {
            *o = ((r - mu) / sigma).clamp(-clip_at, clip_at);
        }
    }

    pub fn z(&self, dn_rates: &[f32], clip_at: f32) -> Vec<f32> {
        let mut out = vec![0.0; self.mu.len()];
        self.z_into(dn_rates, clip_at, &mut out);
        out
    }

    /// Review round 1, F11: `mu`/`sigma` must both be `num_outputs` long, finite, and `sigma`
    /// strictly positive (a `sigma <= 0` would divide by zero or flip every `z`'s sign in
    /// [`DnCalibration::z_into`]) — called from `crate::brain_checkpoint::
    /// load_brain_checkpoint_for_flyg` alongside [`DecoderModel::validate_params_shape`].
    pub fn validate_shape(&self, num_outputs: usize) -> Result<(), DecoderError> {
        if self.mu.len() != num_outputs || self.sigma.len() != num_outputs {
            return Err(DecoderError::ParamShapeMismatch(format!(
                "calibration mu.len()={}, sigma.len()={}, expected {num_outputs}",
                self.mu.len(),
                self.sigma.len()
            )));
        }
        if !self.mu.iter().all(|x| x.is_finite()) || !self.sigma.iter().all(|&s| s.is_finite() && s > 0.0) {
            return Err(DecoderError::NonFiniteParam(
                "calibration mu/sigma contain a non-finite value or a non-positive sigma".to_string(),
            ));
        }
        Ok(())
    }
}

/// The **one** documented calibration protocol (review round 1, F10, CONFIRMED: an earlier
/// revision left this as prose in this module's doc comment while `ddnet-ai fly brain-demo`'s CLI
/// implemented a *different*, undocumented recipe inline — every caller now goes through this
/// function instead): warm the connectome up from `V = 0` ([`FlyState::warm_up`]), then collect
/// [`NUM_CALIBRATION_SAMPLES`] `dn_rates` snapshots from *near*-rest decisions — a small,
/// deterministic-per-`seed` random input current (uniform over
/// `[-CALIBRATION_JITTER_AMPLITUDE/2, +CALIBRATION_JITTER_AMPLITUDE/2)`, resampled each decision)
/// rather than exactly zero input throughout.
///
/// **Mirror-symmetric jitter (review round 2, F19, CONFIRMED):** the jitter is drawn **once per
/// input *type*, per sample** — every input neuron of a given type (both `L` and `R` homologs,
/// and every member sharing a side) gets the exact same value on a given sample — not once per
/// *neuron* independently, which an earlier revision did. Independent per-neuron jitter means an
/// `L`/`R` DN homolog pair sees two different random realizations over
/// [`NUM_CALIBRATION_SAMPLES`] samples, so their measured `mu` end up subtly (but really, ~1e-3)
/// different; combined with `sigma` flooring at `min_sigma` in this near-rest regime (see below),
/// that `mu` gap alone became a `z` offset of `0.02`-`0.04` between otherwise-mirror-homologous
/// DNs — enough to break the decoder's action-level mirror guarantee through calibration alone
/// (measured: `3.7e-2` deviation on an exactly symmetric graph with independent per-neuron
/// jitter, `6.5e-7` — floating-point noise — with this fix). Per-*type* jitter makes the injected
/// current pattern itself already mirror-symmetric on every sample, so the (already
/// mirror-equivariant, by `.flyg`'s own shared `a`/`b`/`theta`) connectome's response is exactly
/// symmetric neuron-for-neuron on every sample too, not just in expectation over many.
///
/// **`sigma` is floored at `min_sigma`, not "prevented" by the jitter (review round 2, F19,
/// CONFIRMED):** an earlier revision's doc comment here claimed the jitter kept `sigma` from
/// collapsing to the floor. It doesn't: this protocol's whole point is a *resting* baseline, and
/// at `CALIBRATION_JITTER_AMPLITUDE`'s deliberately small amplitude the connectome's own damped
/// dynamics pull every sample back close enough to the same rest point that every measured
/// `sigma` sits at `min_sigma` in practice (measured on the real S graph). That is an accepted,
/// documented property of calibrating from *rest* — not a defect to paper over with a bigger,
/// otherwise-arbitrary jitter amplitude (which would just move the floor without giving `sigma`
/// any principled, non-arbitrary meaning either). A caller that wants genuinely non-floored,
/// data-driven `sigma` needs a different protocol entirely (e.g. calibrating from a distribution
/// of real, encoder-driven resting *scenes* rather than raw injected current) — out of scope for
/// this function, which stays model-and-encoder-agnostic on purpose (it only ever sees
/// `model.num_inputs()`-shaped raw current vectors, never an `Observation`).
///
/// `min_sigma` is the caller's own [`DecoderConfig::min_sigma`] (this function doesn't hardcode a
/// decoder-specific default, keeping it decoder-agnostic).
pub const NUM_CALIBRATION_SAMPLES: usize = 64;
/// See [`calibrate_from_rest`]'s doc comment.
pub const CALIBRATION_JITTER_AMPLITUDE: f32 = 0.05;

pub fn calibrate_from_rest(model: &FlyModel, seed: u64, min_sigma: f32) -> Result<DnCalibration, DecoderError> {
    let mut state = FlyState::new(model);
    let _ = state.warm_up(model); // best-effort: even a non-converged rest still gives *a* baseline to calibrate around, and the caller can check `FlyState::warm_up`'s own report separately if it needs to know.
    let mut rng = SplitMix64::new(seed ^ 0xCA11B);
    let flyg = model.flyg();
    // Review round 2, F19: one type_index per input neuron, in `model.input_neuron_indices()`'s
    // own (fixed) order -- used below to give every input neuron of the same type the exact same
    // jitter value on a given sample, regardless of side.
    let input_types: Vec<u32> = model
        .input_neuron_indices()
        .iter()
        .map(|&dense| flyg.neurons[dense as usize].type_index)
        .collect();
    let mut samples = Vec::with_capacity(NUM_CALIBRATION_SAMPLES);
    for _ in 0..NUM_CALIBRATION_SAMPLES {
        let mut jitter_by_type: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        let jittered: Vec<f32> = input_types
            .iter()
            .map(|&type_index| {
                *jitter_by_type
                    .entry(type_index)
                    .or_insert_with(|| (rng.next_f32_unit() - 0.5) * CALIBRATION_JITTER_AMPLITUDE)
            })
            .collect();
        let out = state.step_decision(model, &jittered);
        samples.push(out.dn_rates.to_vec());
    }
    DnCalibration::fit(model.num_outputs(), &samples, min_sigma)
}

// --- Config -------------------------------------------------------------------------------------

/// Which `.flyg` `output_groups` action names feed each head (FLY.md §6's table; the real S/M
/// graphs both name every action [`DecoderConfig::default`] expects — `ddnet-ai fly info` lists
/// them). The direction head's three names are given in `[left, stop, right]` order, which is
/// also [`DecodedAction::direction_probs`]'s class order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecoderConfig {
    pub direction_actions: [String; 3],
    pub jump_action: String,
    pub hook_action: String,
    pub fire_action: String,
    pub aim_action: String,
    /// Fixed von Mises concentration for the aim head's loss — see the module doc comment for why
    /// this is a hyperparameter, not a trained [`DecoderParams`] entry.
    pub aim_kappa: f32,
    /// Floor on `sigma` in [`DnCalibration::fit`].
    pub min_sigma: f32,
    /// `|z| <= this` clip (FLY.md §6: "clip |z| <= 10").
    pub z_clip: f32,
    /// L1 penalty weight on every linear head's `W` (FLY.md §6: "L1 on `W_dec`"). `0.0` disables
    /// it.
    pub l1_weight: f32,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        DecoderConfig {
            direction_actions: [
                "direction_left".to_string(),
                "direction_stop".to_string(),
                "direction_right".to_string(),
            ],
            jump_action: "jump".to_string(),
            hook_action: "hook".to_string(),
            fire_action: "fire".to_string(),
            aim_action: "aim".to_string(),
            aim_kappa: 4.0,
            min_sigma: 0.05,
            z_clip: 10.0,
            l1_weight: 1e-4,
        }
    }
}

impl DecoderConfig {
    pub fn validate(&self) -> Result<(), DecoderError> {
        if !(self.aim_kappa > 0.0 && self.aim_kappa.is_finite()) {
            return Err(DecoderError::InvalidConfig("aim_kappa must be > 0".to_string()));
        }
        if !(self.min_sigma > 0.0 && self.min_sigma.is_finite()) {
            return Err(DecoderError::InvalidConfig("min_sigma must be > 0".to_string()));
        }
        if !(self.z_clip > 0.0 && self.z_clip.is_finite()) {
            return Err(DecoderError::InvalidConfig("z_clip must be > 0".to_string()));
        }
        if !(self.l1_weight >= 0.0 && self.l1_weight.is_finite()) {
            return Err(DecoderError::InvalidConfig("l1_weight must be >= 0".to_string()));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum DecoderError {
    InvalidConfig(String),
    /// A configured action name doesn't exist in this graph's `output_groups`, or resolves to
    /// zero members — never silently treated as an empty/no-op head (acceptance criterion
    /// "validate all index inputs").
    UnknownAction(String),
    /// Review round 1, F6: a type present in one of `direction_left`/`direction_right` but not
    /// the other — the L/R tying this head is built around has no meaning for such a type.
    DirectionTypeNotOnBothSides {
        type_name: String,
    },
    /// Review round 1, F11: [`DecoderModel::validate_params_shape`]'s check — a saved
    /// [`DecoderParams`]'s field doesn't have the length this model (built from the *current*
    /// `.flyg` + [`DecoderConfig`]) expects.
    ParamShapeMismatch(String),
    /// Review round 1, F11: [`DecoderModel::validate_params_shape`] found a non-finite (`NaN`/
    /// `inf`) value in a saved [`DecoderParams`].
    NonFiniteParam(String),
}

impl std::fmt::Display for DecoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecoderError::InvalidConfig(m) => write!(f, "invalid decoder config: {m}"),
            DecoderError::UnknownAction(name) => write!(f, "unknown or empty output_groups action '{name}'"),
            DecoderError::DirectionTypeNotOnBothSides { type_name } => {
                write!(
                    f,
                    "type '{type_name}' appears in direction_left or direction_right but not both -- cannot tie"
                )
            }
            DecoderError::ParamShapeMismatch(m) => write!(f, "decoder params shape mismatch: {m}"),
            DecoderError::NonFiniteParam(m) => write!(f, "decoder params contain a non-finite value: {m}"),
        }
    }
}

impl std::error::Error for DecoderError {}

// --- Params / gradients ---------------------------------------------------------------------------

/// Every learnable decoder parameter (FLY.md §6, tied per review round 1's F6 — see the module doc
/// comment for exactly what each field ties and why). All flat `f32`/`Vec<f32>`: no per-neuron
/// weight anywhere, only per-*type* (or, for `aim`, per-pair/per-unpaired-member).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecoderParams {
    /// One weight per type shared by `direction_left`/`direction_right` (`decoder.direction_lr`'s
    /// order).
    pub direction_lr_w: Vec<f32>,
    pub direction_lr_b: f32,
    /// One weight per type in `direction_stop` (`decoder.direction_stop`'s order).
    pub direction_stop_w: Vec<f32>,
    pub direction_stop_b: f32,
    pub jump_w: Vec<f32>,
    pub jump_b: f32,
    pub hook_w: Vec<f32>,
    pub hook_b: f32,
    pub fire_w: Vec<f32>,
    pub fire_b: f32,
    /// One preferred angle per tied `L`/`R` pair (`decoder.aim_pairs`'s order) — the `L` angle;
    /// the paired `R` angle is *derived* as `π - θ`, never stored (review round 1, F6).
    pub aim_pair_theta: Vec<f32>,
    /// One independent preferred angle per member with no same-type opposite-side partner
    /// (`decoder.aim_unpaired`'s order) — label **П**.
    pub aim_unpaired_theta: Vec<f32>,
}

/// `dL/d` of every [`DecoderParams`] field, same shape.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderGradients {
    pub direction_lr_w: Vec<f32>,
    pub direction_lr_b: f32,
    pub direction_stop_w: Vec<f32>,
    pub direction_stop_b: f32,
    pub jump_w: Vec<f32>,
    pub jump_b: f32,
    pub hook_w: Vec<f32>,
    pub hook_b: f32,
    pub fire_w: Vec<f32>,
    pub fire_b: f32,
    pub aim_pair_theta: Vec<f32>,
    pub aim_unpaired_theta: Vec<f32>,
}

// --- Model ----------------------------------------------------------------------------------------

/// One type's output slots for a "one weight per type, both sides pooled" head (`direction_stop`/
/// `jump`/`hook`/`fire` — review round 1, F6).
#[derive(Debug, Clone)]
struct TypeGroup {
    slots: Vec<usize>,
}

/// One type's `L`/`R` output slots for the tied `direction_left`/`direction_right` head.
#[derive(Debug, Clone)]
struct LrTypeGroup {
    left: Vec<usize>,
    right: Vec<usize>,
}

/// One tied `aim` pair: `L`'s preferred angle is free (`DecoderParams::aim_pair_theta`'s
/// corresponding entry); `R`'s is derived as `π - θ_L`.
#[derive(Debug, Clone, Copy)]
struct AimPair {
    l_slot: usize,
    r_slot: usize,
}

/// One resolved `output_groups` member: which type, which side, and its output slot.
struct ResolvedMember {
    type_index: u32,
    side: ddai_flyg::Side,
    slot: usize,
    /// Dense neuron index — kept only to sort deterministically (ascending) when a type has more
    /// than one member on the same side.
    neuron_index: u32,
}

fn resolve_action_members(model: &FlyModel, action: &str) -> Result<Vec<ResolvedMember>, DecoderError> {
    let flyg = model.flyg();
    let group = flyg
        .output_groups
        .iter()
        .find(|g| g.action == action)
        .ok_or_else(|| DecoderError::UnknownAction(action.to_string()))?;
    if group.members.is_empty() {
        return Err(DecoderError::UnknownAction(action.to_string()));
    }
    Ok(group
        .members
        .iter()
        .map(|m| ResolvedMember {
            type_index: flyg.neurons[m.neuron_index as usize].type_index,
            side: m.side,
            slot: model
                .output_slot_for_neuron(m.neuron_index)
                .unwrap_or_else(|| panic!("output_groups member {} is not an Output-role neuron", m.neuron_index)),
            neuron_index: m.neuron_index,
        })
        .collect())
}

/// Groups `members` by type (both sides pooled), sorted by type name for determinism.
fn group_by_type_pooled(model: &FlyModel, members: &[ResolvedMember]) -> Vec<TypeGroup> {
    let flyg = model.flyg();
    let mut by_type: std::collections::BTreeMap<String, Vec<usize>> = std::collections::BTreeMap::new();
    for m in members {
        by_type
            .entry(flyg.types[m.type_index as usize].name.clone())
            .or_default()
            .push(m.slot);
    }
    by_type.into_values().map(|slots| TypeGroup { slots }).collect()
}

/// Builds the tied `direction_left`/`direction_right` structure: one [`LrTypeGroup`] per type
/// present in *both* `left`/`right`, sorted by type name. Errors if a type appears in only one
/// side (review round 1, F6: there is no meaningful tying for such a type).
fn build_lr_groups(
    model: &FlyModel,
    left: &[ResolvedMember],
    right: &[ResolvedMember],
) -> Result<Vec<LrTypeGroup>, DecoderError> {
    let flyg = model.flyg();
    let mut left_by_type: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for m in left {
        left_by_type.entry(m.type_index).or_default().push(m.slot);
    }
    let mut right_by_type: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for m in right {
        right_by_type.entry(m.type_index).or_default().push(m.slot);
    }
    let mut all_types: std::collections::BTreeSet<u32> = left_by_type.keys().copied().collect();
    all_types.extend(right_by_type.keys().copied());

    // Sort by type *name* (not raw index) for a deterministic, human-legible parameter order.
    let mut named: Vec<(String, u32)> = all_types
        .into_iter()
        .map(|ti| (flyg.types[ti as usize].name.clone(), ti))
        .collect();
    named.sort();

    named
        .into_iter()
        .map(|(name, ti)| {
            let left_slots = left_by_type.get(&ti).cloned();
            let right_slots = right_by_type.get(&ti).cloned();
            match (left_slots, right_slots) {
                (Some(left), Some(right)) => Ok(LrTypeGroup { left, right }),
                _ => Err(DecoderError::DirectionTypeNotOnBothSides { type_name: name }),
            }
        })
        .collect()
}

/// Builds the tied `aim` pairing (review round 1, F6): groups members by `(type, side)`, pairs
/// `L`/`R` members of the same type positionally (sorted by dense neuron index — deterministic,
/// not a claim of real anatomical homology, see the module doc comment's **П** label), and
/// collects anything left over (side `M`/`Unknown`, or an `L`/`R` count mismatch within a type)
/// into `unpaired`.
fn build_aim_structure(model: &FlyModel, members: &[ResolvedMember]) -> (Vec<AimPair>, Vec<usize>) {
    let mut left_by_type: std::collections::BTreeMap<u32, Vec<(u32, usize)>> = std::collections::BTreeMap::new();
    let mut right_by_type: std::collections::BTreeMap<u32, Vec<(u32, usize)>> = std::collections::BTreeMap::new();
    let mut unpaired = Vec::new();

    for m in members {
        match m.side {
            ddai_flyg::Side::L => left_by_type
                .entry(m.type_index)
                .or_default()
                .push((m.neuron_index, m.slot)),
            ddai_flyg::Side::R => right_by_type
                .entry(m.type_index)
                .or_default()
                .push((m.neuron_index, m.slot)),
            _ => unpaired.push(m.slot),
        }
    }
    let _ = model; // kept in the signature for symmetry with the other `build_*` helpers / future use.

    let mut all_types: std::collections::BTreeSet<u32> = left_by_type.keys().copied().collect();
    all_types.extend(right_by_type.keys().copied());

    let mut pairs = Vec::new();
    for ti in all_types {
        let mut l = left_by_type.remove(&ti).unwrap_or_default();
        let mut r = right_by_type.remove(&ti).unwrap_or_default();
        l.sort_by_key(|&(idx, _)| idx);
        r.sort_by_key(|&(idx, _)| idx);
        let n = l.len().min(r.len());
        for i in 0..n {
            pairs.push(AimPair {
                l_slot: l[i].1,
                r_slot: r[i].1,
            });
        }
        unpaired.extend(l[n..].iter().map(|&(_, slot)| slot));
        unpaired.extend(r[n..].iter().map(|&(_, slot)| slot));
    }
    (pairs, unpaired)
}

/// Built once from a [`FlyModel`] plus [`DecoderConfig`] (mirrors [`crate::encoder::EncoderModel`]'s
/// "build once, query many times" shape): every head's tied structure (review round 1, F6).
#[derive(Debug, Clone)]
pub struct DecoderModel {
    config: DecoderConfig,
    num_outputs: usize,
    direction_lr: Vec<LrTypeGroup>,
    direction_stop: Vec<TypeGroup>,
    jump: Vec<TypeGroup>,
    hook: Vec<TypeGroup>,
    fire: Vec<TypeGroup>,
    aim_pairs: Vec<AimPair>,
    aim_unpaired: Vec<usize>,
}

fn mean_of(z: &[f32], slots: &[usize]) -> f32 {
    if slots.is_empty() {
        return 0.0;
    }
    slots.iter().map(|&s| z[s]).sum::<f32>() / slots.len() as f32
}

impl DecoderModel {
    pub fn new(model: &FlyModel, config: DecoderConfig) -> Result<Self, DecoderError> {
        config.validate()?;
        let left_members = resolve_action_members(model, &config.direction_actions[0])?;
        let stop_members = resolve_action_members(model, &config.direction_actions[1])?;
        let right_members = resolve_action_members(model, &config.direction_actions[2])?;
        let direction_lr = build_lr_groups(model, &left_members, &right_members)?;
        let direction_stop = group_by_type_pooled(model, &stop_members);

        let jump = group_by_type_pooled(model, &resolve_action_members(model, &config.jump_action)?);
        let hook = group_by_type_pooled(model, &resolve_action_members(model, &config.hook_action)?);
        let fire = group_by_type_pooled(model, &resolve_action_members(model, &config.fire_action)?);
        let (aim_pairs, aim_unpaired) = build_aim_structure(model, &resolve_action_members(model, &config.aim_action)?);

        Ok(DecoderModel {
            config,
            num_outputs: model.num_outputs(),
            direction_lr,
            direction_stop,
            jump,
            hook,
            fire,
            aim_pairs,
            aim_unpaired,
        })
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// A default [`DecoderParams`] (all-zero `W`/`b`, evenly-spaced preferred angles) — a
    /// deterministic starting point, matching [`crate::encoder::EncoderParams::init_default`]'s
    /// own "no connectome-derived quantity to draw this from" rationale.
    pub fn init_default_params(&self) -> DecoderParams {
        let spaced = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| std::f32::consts::TAU * i as f32 / n.max(1) as f32 - std::f32::consts::PI)
                .collect()
        };
        DecoderParams {
            direction_lr_w: vec![0.0; self.direction_lr.len()],
            direction_lr_b: 0.0,
            direction_stop_w: vec![0.0; self.direction_stop.len()],
            direction_stop_b: 0.0,
            jump_w: vec![0.0; self.jump.len()],
            jump_b: 0.0,
            hook_w: vec![0.0; self.hook.len()],
            hook_b: 0.0,
            fire_w: vec![0.0; self.fire.len()],
            fire_b: 0.0,
            aim_pair_theta: spaced(self.aim_pairs.len()),
            aim_unpaired_theta: spaced(self.aim_unpaired.len()),
        }
    }

    pub fn zeros_gradients(&self) -> DecoderGradients {
        DecoderGradients {
            direction_lr_w: vec![0.0; self.direction_lr.len()],
            direction_lr_b: 0.0,
            direction_stop_w: vec![0.0; self.direction_stop.len()],
            direction_stop_b: 0.0,
            jump_w: vec![0.0; self.jump.len()],
            jump_b: 0.0,
            hook_w: vec![0.0; self.hook.len()],
            hook_b: 0.0,
            fire_w: vec![0.0; self.fire.len()],
            fire_b: 0.0,
            aim_pair_theta: vec![0.0; self.aim_pairs.len()],
            aim_unpaired_theta: vec![0.0; self.aim_unpaired.len()],
        }
    }

    /// Every field's length must match this model's own structure, and every value must be
    /// finite (review round 1, F11, CONFIRMED: an earlier revision never called any shape check
    /// on a loaded checkpoint, so editing `configs/fly/{S,M}-brain.toml` in a way that changes a
    /// head's resolved member count — e.g. reassigning which action an `output_groups` type feeds
    /// — would silently reinterpret an old checkpoint's flat `Vec<f32>` against a differently-
    /// shaped model instead of erroring). Called from `crate::brain_checkpoint::
    /// load_brain_checkpoint_for_flyg` before a caller can build a [`FlyBrain`](crate::brain::
    /// FlyBrain) from a loaded checkpoint's params.
    pub fn validate_params_shape(&self, params: &DecoderParams) -> Result<(), DecoderError> {
        let len_checks: [(&str, usize, usize); 7] = [
            ("direction_lr_w", params.direction_lr_w.len(), self.direction_lr.len()),
            (
                "direction_stop_w",
                params.direction_stop_w.len(),
                self.direction_stop.len(),
            ),
            ("jump_w", params.jump_w.len(), self.jump.len()),
            ("hook_w", params.hook_w.len(), self.hook.len()),
            ("fire_w", params.fire_w.len(), self.fire.len()),
            ("aim_pair_theta", params.aim_pair_theta.len(), self.aim_pairs.len()),
            (
                "aim_unpaired_theta",
                params.aim_unpaired_theta.len(),
                self.aim_unpaired.len(),
            ),
        ];
        for (name, actual, expected) in len_checks {
            if actual != expected {
                return Err(DecoderError::ParamShapeMismatch(format!(
                    "{name}.len() == {actual}, expected {expected}"
                )));
            }
        }
        let all_finite = params
            .direction_lr_w
            .iter()
            .chain(&params.direction_stop_w)
            .chain(&params.jump_w)
            .chain(&params.hook_w)
            .chain(&params.fire_w)
            .chain(&params.aim_pair_theta)
            .chain(&params.aim_unpaired_theta)
            .chain(std::iter::once(&params.direction_lr_b))
            .chain(std::iter::once(&params.direction_stop_b))
            .chain(std::iter::once(&params.jump_b))
            .chain(std::iter::once(&params.hook_b))
            .chain(std::iter::once(&params.fire_b))
            .all(|x| x.is_finite());
        if !all_finite {
            return Err(DecoderError::NonFiniteParam(
                "at least one field contains NaN/inf".to_string(),
            ));
        }
        Ok(())
    }
}

fn sigmoid(x: f32) -> f32 {
    crate::activation::sigmoid(x)
}

/// The decoder's output for one decision (task spec's action space): direction/jump/hook/fire as
/// probabilities (argmax/sampling is the caller's/`FlyBrain`'s policy, not this module's), aim as
/// a ring-convention angle (see `crate::encoder`'s module doc comment for the convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodedAction {
    /// `[left, stop, right]`, matching [`DecoderConfig::direction_actions`]'s order.
    pub direction_probs: [f32; 3],
    pub jump_prob: f32,
    pub hook_prob: f32,
    pub fire_prob: f32,
    pub aim_angle: f32,
}

/// `softmax` over exactly 3 logits, no allocation.
fn softmax3(logits: [f32; 3]) -> [f32; 3] {
    let max = logits[0].max(logits[1]).max(logits[2]);
    let exps = [
        (logits[0] - max).exp(),
        (logits[1] - max).exp(),
        (logits[2] - max).exp(),
    ];
    let sum = exps[0] + exps[1] + exps[2];
    [exps[0] / sum, exps[1] / sum, exps[2] / sum]
}

/// One-type-per-weight logit: `b + Σ_i w[i] * mean(z[groups[i].slots])`.
fn pooled_logit(groups: &[TypeGroup], w: &[f32], b: f32, z: &[f32]) -> f32 {
    b + groups
        .iter()
        .zip(w)
        .map(|(g, &wi)| wi * mean_of(z, &g.slots))
        .sum::<f32>()
}

/// `(C, S)` for the aim population vector — see the module doc comment for the tied-pair math.
fn population_vector(
    z: &[f32],
    pairs: &[AimPair],
    pair_theta: &[f32],
    unpaired: &[usize],
    unpaired_theta: &[f32],
) -> (f32, f32) {
    let mut c = 0.0f32;
    let mut s = 0.0f32;
    for (pair, &theta) in pairs.iter().zip(pair_theta) {
        let (zl, zr) = (z[pair.l_slot], z[pair.r_slot]);
        c += (zl - zr) * theta.cos();
        s += (zl + zr) * theta.sin();
    }
    for (&slot, &theta) in unpaired.iter().zip(unpaired_theta) {
        c += z[slot] * theta.cos();
        s += z[slot] * theta.sin();
    }
    (c, s)
}

/// Persistent scratch for [`decoder_forward_into`] — just the calibrated-`z` buffer now (review
/// round 1, F6's tying removed the old dense-concat/`max_buf` scratch entirely: every head reads
/// `z` directly through the model's own static slot lists, no per-call gather buffer needed).
#[derive(Debug, Clone)]
pub struct DecoderScratch {
    z: Vec<f32>,
}

impl DecoderScratch {
    pub fn new(decoder: &DecoderModel) -> Self {
        DecoderScratch {
            z: vec![0.0; decoder.num_outputs],
        }
    }
}

/// Runs every head forward from `dn_rates` (no gradient bookkeeping — see [`decoder_loss_and_grad`]
/// for the training path). Allocates `scratch` fresh — fine for tests/training; see
/// [`decoder_forward_into`] for the allocation-free version [`crate::brain::FlyBrain`] uses.
pub fn decoder_forward(
    decoder: &DecoderModel,
    dn_rates: &[f32],
    calib: &DnCalibration,
    params: &DecoderParams,
) -> DecodedAction {
    let mut scratch = DecoderScratch::new(decoder);
    decoder_forward_into(decoder, dn_rates, calib, params, &mut scratch)
}

/// The allocation-free path [`crate::brain::FlyBrain::decide`] actually calls.
pub fn decoder_forward_into(
    decoder: &DecoderModel,
    dn_rates: &[f32],
    calib: &DnCalibration,
    params: &DecoderParams,
    scratch: &mut DecoderScratch,
) -> DecodedAction {
    assert_eq!(dn_rates.len(), decoder.num_outputs);
    calib.z_into(dn_rates, decoder.config.z_clip, &mut scratch.z);
    let z = &scratch.z;

    let left_logit = decoder
        .direction_lr
        .iter()
        .zip(&params.direction_lr_w)
        .map(|(g, &w)| w * mean_of(z, &g.left))
        .sum::<f32>()
        + params.direction_lr_b;
    let right_logit = decoder
        .direction_lr
        .iter()
        .zip(&params.direction_lr_w)
        .map(|(g, &w)| w * mean_of(z, &g.right))
        .sum::<f32>()
        + params.direction_lr_b;
    let stop_logit = pooled_logit(
        &decoder.direction_stop,
        &params.direction_stop_w,
        params.direction_stop_b,
        z,
    );
    let dir_probs = softmax3([left_logit, stop_logit, right_logit]);

    let jump_logit = pooled_logit(&decoder.jump, &params.jump_w, params.jump_b, z);
    let hook_logit = pooled_logit(&decoder.hook, &params.hook_w, params.hook_b, z);
    let fire_logit = pooled_logit(&decoder.fire, &params.fire_w, params.fire_b, z);

    let (c, s) = population_vector(
        z,
        &decoder.aim_pairs,
        &params.aim_pair_theta,
        &decoder.aim_unpaired,
        &params.aim_unpaired_theta,
    );
    let aim_angle = s.atan2(c);

    DecodedAction {
        direction_probs: dir_probs,
        jump_prob: sigmoid(jump_logit),
        hook_prob: sigmoid(hook_logit),
        fire_prob: sigmoid(fire_logit),
        aim_angle,
    }
}

/// What to train each head towards for one decision; every field is optional so a caller (the
/// synthetic demo, a future BC loss) can supply only the heads it has a teacher signal for —
/// `None` contributes zero loss and zero gradient for that head.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecoderTargets {
    /// `0` = left, `1` = stop, `2` = right (matching [`DecodedAction::direction_probs`]).
    pub direction: Option<u8>,
    pub jump: Option<bool>,
    pub hook: Option<bool>,
    pub fire: Option<bool>,
    /// Radians, ring convention.
    pub aim: Option<f32>,
}

/// Accumulates `d_logit * mean(z[slots])` into `grad_w[i]` and scatters `d_logit * w[i] /
/// len(slots)` into `grad_z` for every type group — the shared backward step every pooled head
/// (`stop`/`jump`/`hook`/`fire`) uses.
fn pooled_backward(
    groups: &[TypeGroup],
    w: &[f32],
    d_logit: f32,
    z: &[f32],
    grad_w: &mut [f32],
    grad_b: &mut f32,
    grad_z: &mut [f32],
) {
    *grad_b += d_logit;
    for (g, (&wi, gw)) in groups.iter().zip(w.iter().zip(grad_w.iter_mut())) {
        if g.slots.is_empty() {
            continue;
        }
        *gw += d_logit * mean_of(z, &g.slots);
        let contrib = d_logit * wi / g.slots.len() as f32;
        for &slot in &g.slots {
            grad_z[slot] += contrib;
        }
    }
}

/// Cross-entropy + BCE + von Mises NLL over whichever of `targets`'s fields are `Some`, added
/// together (equal weight — a caller wanting per-head weighting scales its own targets/grad
/// upstream). Returns `(loss, grad_params, grad_dn)`; `grad_dn` (`dL/d(dn_rates)`, `num_outputs`
/// long) is exactly [`crate::backward::backward`]'s `grad_dn_rates` shape for one decision.
pub fn decoder_loss_and_grad(
    decoder: &DecoderModel,
    dn_rates: &[f32],
    calib: &DnCalibration,
    params: &DecoderParams,
    targets: &DecoderTargets,
) -> (f32, DecoderGradients, Vec<f32>) {
    assert_eq!(dn_rates.len(), decoder.num_outputs);
    let clip_at = decoder.config.z_clip;
    let z = calib.z(dn_rates, clip_at);
    let mut grad_z = vec![0.0f32; z.len()];
    let mut grads = decoder.zeros_gradients();
    let mut loss = 0.0f32;

    if let Some(target) = targets.direction {
        let left_logit = decoder
            .direction_lr
            .iter()
            .zip(&params.direction_lr_w)
            .map(|(g, &w)| w * mean_of(&z, &g.left))
            .sum::<f32>()
            + params.direction_lr_b;
        let right_logit = decoder
            .direction_lr
            .iter()
            .zip(&params.direction_lr_w)
            .map(|(g, &w)| w * mean_of(&z, &g.right))
            .sum::<f32>()
            + params.direction_lr_b;
        let stop_logit = pooled_logit(
            &decoder.direction_stop,
            &params.direction_stop_w,
            params.direction_stop_b,
            &z,
        );
        let probs = softmax3([left_logit, stop_logit, right_logit]);
        loss += -(probs[target as usize].max(1e-12)).ln();
        let [d_left, d_stop, d_right] = std::array::from_fn(|c| probs[c] - f32::from(c == target as usize));

        grads.direction_lr_b += d_left + d_right;
        for (g, (&w, gw)) in decoder
            .direction_lr
            .iter()
            .zip(params.direction_lr_w.iter().zip(grads.direction_lr_w.iter_mut()))
        {
            *gw += d_left * mean_of(&z, &g.left) + d_right * mean_of(&z, &g.right);
            if !g.left.is_empty() {
                let contrib = d_left * w / g.left.len() as f32;
                for &slot in &g.left {
                    grad_z[slot] += contrib;
                }
            }
            if !g.right.is_empty() {
                let contrib = d_right * w / g.right.len() as f32;
                for &slot in &g.right {
                    grad_z[slot] += contrib;
                }
            }
        }
        pooled_backward(
            &decoder.direction_stop,
            &params.direction_stop_w,
            d_stop,
            &z,
            &mut grads.direction_stop_w,
            &mut grads.direction_stop_b,
            &mut grad_z,
        );
    }

    macro_rules! binary_head {
        ($groups:expr, $target:expr, $w:expr, $b:expr, $grad_w:expr, $grad_b:expr) => {
            if let Some(y) = $target {
                let logit = pooled_logit($groups, $w, $b, &z);
                let p = sigmoid(logit);
                let y_f = f32::from(y);
                loss += -(y_f * p.max(1e-12).ln() + (1.0 - y_f) * (1.0 - p).max(1e-12).ln());
                let d_logit = p - y_f;
                pooled_backward($groups, $w, d_logit, &z, $grad_w, $grad_b, &mut grad_z);
            }
        };
    }
    binary_head!(
        &decoder.jump,
        targets.jump,
        &params.jump_w,
        params.jump_b,
        &mut grads.jump_w,
        &mut grads.jump_b
    );
    binary_head!(
        &decoder.hook,
        targets.hook,
        &params.hook_w,
        params.hook_b,
        &mut grads.hook_w,
        &mut grads.hook_b
    );
    binary_head!(
        &decoder.fire,
        targets.fire,
        &params.fire_w,
        params.fire_b,
        &mut grads.fire_w,
        &mut grads.fire_b
    );

    if let Some(target_angle) = targets.aim {
        let (c, s) = population_vector(
            &z,
            &decoder.aim_pairs,
            &params.aim_pair_theta,
            &decoder.aim_unpaired,
            &params.aim_unpaired_theta,
        );
        let (aim_loss, d_c, d_s) = aim_loss_and_grad(c, s, target_angle, decoder.config.aim_kappa);
        loss += aim_loss;
        for (pair, (&theta, gtheta)) in decoder
            .aim_pairs
            .iter()
            .zip(params.aim_pair_theta.iter().zip(grads.aim_pair_theta.iter_mut()))
        {
            let (zl, zr) = (z[pair.l_slot], z[pair.r_slot]);
            *gtheta += d_c * (-(zl - zr) * theta.sin()) + d_s * ((zl + zr) * theta.cos());
            grad_z[pair.l_slot] += d_c * theta.cos() + d_s * theta.sin();
            grad_z[pair.r_slot] += d_c * (-theta.cos()) + d_s * theta.sin();
        }
        for (&slot, (&theta, gtheta)) in decoder.aim_unpaired.iter().zip(
            params
                .aim_unpaired_theta
                .iter()
                .zip(grads.aim_unpaired_theta.iter_mut()),
        ) {
            let zj = z[slot];
            *gtheta += d_c * (-zj * theta.sin()) + d_s * (zj * theta.cos());
            grad_z[slot] += d_c * theta.cos() + d_s * theta.sin();
        }
    }

    // L1 on every linear head's `W` is deliberately **not** added here — see
    // [`add_l1_penalty`]'s doc comment for why it is a separate, composable function.

    // Backprop grad_z through the (frozen) calibration's clip into grad_dn: `dz/dr = 1/sigma`
    // where the clip is inactive, `0` where it's saturated (same relu-kink-style subgradient
    // convention as elsewhere in this crate).
    let mut grad_dn = vec![0.0f32; decoder.num_outputs];
    for i in 0..decoder.num_outputs {
        if grad_z[i] == 0.0 {
            continue;
        }
        let raw = (dn_rates[i] - calib.mu[i]) / calib.sigma[i];
        if raw.abs() >= clip_at {
            continue; // clipped: subgradient 0, matching this crate's kink convention.
        }
        grad_dn[i] = grad_z[i] / calib.sigma[i];
    }

    (loss, grads, grad_dn)
}

/// `NLL = -kappa * cos(target - mu)`, `mu = atan2(s, c)` — see the module doc comment. Returns
/// `(loss, dL/dc, dL/ds)`. `(c, s) approx (0, 0)` gives a zero gradient rather than propagating a
/// `NaN` through `atan2`'s undefined point.
fn aim_loss_and_grad(c: f32, s: f32, target: f32, kappa: f32) -> (f32, f32, f32) {
    let r2 = c * c + s * s;
    if r2 < 1e-12 {
        return (kappa, 0.0, 0.0);
    }
    let mu = s.atan2(c);
    let loss = -kappa * (target - mu).cos();
    let d_loss_d_mu = -kappa * (target - mu).sin();
    let d_mu_d_c = -s / r2;
    let d_mu_d_s = c / r2;
    (loss, d_loss_d_mu * d_mu_d_c, d_loss_d_mu * d_mu_d_s)
}

/// The L1 penalty's own scalar value (`weight * sum(|w|)` over every linear head's `W`) — kept
/// separate from [`add_l1_penalty`]'s gradient so a caller (`crate::brain_train::brain_train_step`,
/// review round 1 F12) can add it into a reported total loss without re-deriving the sum itself.
/// `0.0` when `weight == 0.0`, matching [`add_l1_penalty`]'s own no-op convention.
pub fn l1_penalty_value(params: &DecoderParams, weight: f32) -> f32 {
    if weight == 0.0 {
        return 0.0;
    }
    let sum_abs: f32 = params
        .direction_lr_w
        .iter()
        .chain(&params.direction_stop_w)
        .chain(&params.jump_w)
        .chain(&params.hook_w)
        .chain(&params.fire_w)
        .map(|w| w.abs())
        .sum();
    weight * sum_abs
}

/// L1 penalty on every linear head's `W` (FLY.md §6: "L1 on `W_dec`"), added directly into
/// `grads` — a separate, composable regularizer a caller applies *after* [`decoder_loss_and_grad`]
/// (mirrors `crate::optim::add_l2_pull_to_a`'s own "composed at the call site" shape). No-op when
/// `weight == 0.0`. Subgradient exactly `0` at `w == 0.0` (this crate's usual kink convention).
pub fn add_l1_penalty(grads: &mut DecoderGradients, params: &DecoderParams, weight: f32) {
    fn subgradient(w: f32) -> f32 {
        if w == 0.0 { 0.0 } else { w.signum() }
    }
    if weight == 0.0 {
        return;
    }
    for (g, &w) in grads.direction_lr_w.iter_mut().zip(&params.direction_lr_w) {
        *g += weight * subgradient(w);
    }
    for (g, &w) in grads.direction_stop_w.iter_mut().zip(&params.direction_stop_w) {
        *g += weight * subgradient(w);
    }
    for (g, &w) in grads.jump_w.iter_mut().zip(&params.jump_w) {
        *g += weight * subgradient(w);
    }
    for (g, &w) in grads.hook_w.iter_mut().zip(&params.hook_w) {
        *g += weight * subgradient(w);
    }
    for (g, &w) in grads.fire_w.iter_mut().zip(&params.fire_w) {
        *g += weight * subgradient(w);
    }
}

mod bc_head;
pub use bc_head::{decoder_bc_loss_and_grad, decoder_logits};

#[cfg(test)]
mod tests;
