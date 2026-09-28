//! The world-model head (task 7.3, acceptance criterion 5): a linear readout from `f(V)` of a
//! configurable neuron subset (default: every `Hidden`-role — "central-brain" — neuron) predicts
//! `Δpos`/`Δvel` of self and the opponent, freeze onset, ground contact, and hook hit, `k ∈ {1, 5,
//! 10}` decisions ahead. Gradients flow into 7.2's `backward` the same way the decoder's do: as an
//! `ExtraRateGrad` tap (this module's `dL/dr`) at the decision where the readout happened.
//!
//! A **separate**, non-backpropagating [`fit_ridge_probe`] implements FLY.md §7's actual "does the
//! fly understand physics" metric: a linear probe fit by closed-form ridge regression (normal
//! equations, hand-solved by Gauss-Jordan elimination — no external linear-algebra crate, matching
//! this crate's "no ML frameworks" constraint) on **frozen** network state, so the metric can't be
//! gamed by a network that has merely learned to make [`WorldModelHead`]'s own trained readout
//! look good.

use serde::{Deserialize, Serialize};

use crate::model::FlyModel;

/// One `k`-decisions-ahead regression target (task spec): `Δpos`/`Δvel` of self and the opponent,
/// 8 continuous values.
pub const NUM_REGRESSION_TARGETS: usize = 8;
/// One `k`-decisions-ahead binary target: freeze onset, ground contact, hook hit.
pub const NUM_BINARY_TARGETS: usize = 3;

/// The three horizons FLY.md §7 asks for.
pub const HORIZONS: [usize; 3] = [1, 5, 10];

#[derive(Debug)]
pub enum WorldModelError {
    InvalidConfig(String),
}

impl std::fmt::Display for WorldModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorldModelError::InvalidConfig(m) => write!(f, "invalid world-model config: {m}"),
        }
    }
}

impl std::error::Error for WorldModelError {}

/// Which neurons feed the readout (task spec: "a configurable neuron subset, default: all central-
/// brain neurons") and the loss weight it should carry relative to the main action loss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldModelConfig {
    /// Dense neuron indices the readout reads `f(V)` from, ascending, no duplicates (validated by
    /// [`WorldModelHead::new`]). `None` means "every `Hidden`-role neuron" (the task spec's
    /// default) — resolved against a specific [`FlyModel`] at construction time.
    #[serde(default)]
    pub neuron_subset: Option<Vec<u32>>,
    /// FLY.md §7: "loss with weight 0.1-0.3 of the main [loss]" — a caller combines this
    /// module's loss with the action-decoder loss using this weight; not applied inside this
    /// module itself (same "mechanism here, policy at the call site" split as `crate::optim`'s
    /// L2 pull / activity regularizer).
    pub loss_weight: f32,
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        WorldModelConfig {
            neuron_subset: None,
            loss_weight: 0.2,
        }
    }
}

impl WorldModelConfig {
    pub fn validate(&self) -> Result<(), WorldModelError> {
        if !(self.loss_weight.is_finite() && self.loss_weight >= 0.0) {
            return Err(WorldModelError::InvalidConfig("loss_weight must be >= 0".to_string()));
        }
        if let Some(subset) = &self.neuron_subset {
            if subset.is_empty() {
                return Err(WorldModelError::InvalidConfig(
                    "neuron_subset must not be empty".to_string(),
                ));
            }
            for w in subset.windows(2) {
                if w[0] >= w[1] {
                    return Err(WorldModelError::InvalidConfig(
                        "neuron_subset must be strictly ascending with no duplicates".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// One horizon `k`'s linear heads: `[NUM_REGRESSION_TARGETS x n_subset]` + bias for the
/// continuous targets, `[NUM_BINARY_TARGETS x n_subset]` + bias for the binary ones (read via
/// sigmoid, same convention as `crate::decoder`'s Bernoulli heads).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HorizonParams {
    pub w_reg: Vec<f32>,
    pub b_reg: [f32; NUM_REGRESSION_TARGETS],
    pub w_bin: Vec<f32>,
    pub b_bin: [f32; NUM_BINARY_TARGETS],
}

impl HorizonParams {
    fn zeros(n_subset: usize) -> Self {
        HorizonParams {
            w_reg: vec![0.0; NUM_REGRESSION_TARGETS * n_subset],
            b_reg: [0.0; NUM_REGRESSION_TARGETS],
            w_bin: vec![0.0; NUM_BINARY_TARGETS * n_subset],
            b_bin: [0.0; NUM_BINARY_TARGETS],
        }
    }
}

/// Every learnable world-model parameter: one [`HorizonParams`] per entry of [`HORIZONS`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldModelParams {
    pub horizons: [HorizonParams; 3],
}

/// `dL/d` of every [`WorldModelParams`] field, same shape.
#[derive(Debug, Clone, PartialEq)]
pub struct WorldModelGradients {
    pub horizons: [HorizonParams; 3],
}

/// Built once from a [`FlyModel`] plus [`WorldModelConfig`]: the resolved neuron subset (dense
/// indices into the model's full neuron array) and a default parameter set sized for it.
#[derive(Debug, Clone)]
pub struct WorldModelHead {
    subset: Vec<u32>,
    /// `config.loss_weight`, kept here (review round 1, F5, CONFIRMED) so a caller combining this
    /// head's loss/gradients with the action decoder's — `crate::brain_train::brain_train_step`
    /// — has the weight available without threading `WorldModelConfig` through separately.
    /// Applying it is the caller's job (`world_model_loss_and_grad` itself stays a plain,
    /// unweighted loss — same "mechanism here, policy at the call site" split as
    /// `crate::optim`), but forgetting to apply it is no longer possible without going out of
    /// one's way: `brain_train_step` reads it from here, not from a config the caller could pass
    /// inconsistently.
    loss_weight: f32,
}

impl WorldModelHead {
    pub fn new(model: &FlyModel, config: &WorldModelConfig) -> Result<Self, WorldModelError> {
        config.validate()?;
        let subset = match &config.neuron_subset {
            Some(explicit) => {
                let n = model.num_neurons() as u32;
                for &i in explicit {
                    if i >= n {
                        return Err(WorldModelError::InvalidConfig(format!(
                            "neuron_subset contains out-of-range dense index {i} (model has {n} neurons)"
                        )));
                    }
                }
                explicit.clone()
            }
            None => (0..model.num_neurons() as u32)
                .filter(|&i| model.flyg().neurons[i as usize].role == ddai_flyg::NeuronRole::Hidden)
                .collect(),
        };
        if subset.is_empty() {
            return Err(WorldModelError::InvalidConfig(
                "resolved neuron_subset is empty (no Hidden-role neurons on this graph?)".to_string(),
            ));
        }
        Ok(WorldModelHead {
            subset,
            loss_weight: config.loss_weight,
        })
    }

    pub fn num_subset(&self) -> usize {
        self.subset.len()
    }

    pub fn loss_weight(&self) -> f32 {
        self.loss_weight
    }

    pub fn subset(&self) -> &[u32] {
        &self.subset
    }

    pub fn init_default_params(&self) -> WorldModelParams {
        WorldModelParams {
            horizons: std::array::from_fn(|_| HorizonParams::zeros(self.subset.len())),
        }
    }

    pub fn zeros_gradients(&self) -> WorldModelGradients {
        WorldModelGradients {
            horizons: std::array::from_fn(|_| HorizonParams::zeros(self.subset.len())),
        }
    }

    /// Every horizon's `w_reg`/`w_bin` must be `NUM_REGRESSION_TARGETS`/`NUM_BINARY_TARGETS`
    /// times this head's own `num_subset()`, and every value must be finite (review round 1, F11
    /// — same rationale as `crate::decoder::DecoderModel::validate_params_shape`: an edited
    /// `neuron_subset` config changes `num_subset()`, which would otherwise silently reinterpret
    /// a loaded checkpoint's flat `Vec<f32>` at the wrong shape rather than erroring).
    pub fn validate_params_shape(&self, params: &WorldModelParams) -> Result<(), WorldModelError> {
        let expected_w_reg = NUM_REGRESSION_TARGETS * self.subset.len();
        let expected_w_bin = NUM_BINARY_TARGETS * self.subset.len();
        for (k, h) in params.horizons.iter().enumerate() {
            if h.w_reg.len() != expected_w_reg {
                return Err(WorldModelError::InvalidConfig(format!(
                    "horizons[{k}].w_reg.len() == {}, expected {expected_w_reg}",
                    h.w_reg.len()
                )));
            }
            if h.w_bin.len() != expected_w_bin {
                return Err(WorldModelError::InvalidConfig(format!(
                    "horizons[{k}].w_bin.len() == {}, expected {expected_w_bin}",
                    h.w_bin.len()
                )));
            }
            let all_finite = h
                .w_reg
                .iter()
                .chain(&h.b_reg)
                .chain(&h.w_bin)
                .chain(&h.b_bin)
                .all(|x| x.is_finite());
            if !all_finite {
                return Err(WorldModelError::InvalidConfig(format!(
                    "horizons[{k}] contains a non-finite value"
                )));
            }
        }
        Ok(())
    }

    fn gather(&self, r_full: &[f32]) -> Vec<f32> {
        self.subset.iter().map(|&i| r_full[i as usize]).collect()
    }
}

fn linear(w: &[f32], b: &[f32], x: &[f32], num_out: usize) -> Vec<f32> {
    let n = x.len();
    (0..num_out)
        .map(|o| b[o] + w[o * n..o * n + n].iter().zip(x).map(|(&wi, &xi)| wi * xi).sum::<f32>())
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    crate::activation::sigmoid(x)
}

/// One horizon's prediction: 8 continuous values, 3 probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct HorizonPrediction {
    pub regression: [f32; NUM_REGRESSION_TARGETS],
    pub binary_prob: [f32; NUM_BINARY_TARGETS],
}

pub fn world_model_forward(head: &WorldModelHead, r_full: &[f32], params: &WorldModelParams) -> [HorizonPrediction; 3] {
    let x = head.gather(r_full);
    std::array::from_fn(|k| {
        let hp = &params.horizons[k];
        let reg = linear(&hp.w_reg, &hp.b_reg, &x, NUM_REGRESSION_TARGETS);
        let bin = linear(&hp.w_bin, &hp.b_bin, &x, NUM_BINARY_TARGETS);
        HorizonPrediction {
            regression: std::array::from_fn(|i| reg[i]),
            binary_prob: std::array::from_fn(|i| sigmoid(bin[i])),
        }
    })
}

/// What to train one horizon towards; `None` (either half, or the whole horizon) contributes zero
/// loss/gradient for that part.
#[derive(Debug, Clone, Copy, Default)]
pub struct HorizonTargets {
    pub regression: Option<[f32; NUM_REGRESSION_TARGETS]>,
    pub binary: Option<[bool; NUM_BINARY_TARGETS]>,
}

/// MSE (regression) + BCE (binary), summed over every horizon that has a target. Returns `(loss,
/// grad_params, grad_r)`; `grad_r` (`dL/dr`, `num_neurons` long, dense over the **whole** model —
/// zero outside the subset) is exactly [`crate::backward::ExtraRateGrad::grad`]'s shape for the
/// decision this readout was taken at.
pub fn world_model_loss_and_grad(
    head: &WorldModelHead,
    r_full: &[f32],
    num_neurons: usize,
    params: &WorldModelParams,
    targets: &[HorizonTargets; 3],
) -> (f32, WorldModelGradients, Vec<f32>) {
    let x = head.gather(r_full);
    let n = x.len();
    let mut grads = head.zeros_gradients();
    let mut grad_x = vec![0.0f32; n];
    let mut loss = 0.0f32;

    for ((hp, g), target) in params
        .horizons
        .iter()
        .zip(grads.horizons.iter_mut())
        .zip(targets.iter())
    {
        if let Some(target) = target.regression {
            let pred = linear(&hp.w_reg, &hp.b_reg, &x, NUM_REGRESSION_TARGETS);
            for o in 0..NUM_REGRESSION_TARGETS {
                let diff = pred[o] - target[o];
                loss += diff * diff;
                let d_pred = 2.0 * diff;
                g.b_reg[o] += d_pred;
                for j in 0..n {
                    g.w_reg[o * n + j] += d_pred * x[j];
                    grad_x[j] += d_pred * hp.w_reg[o * n + j];
                }
            }
        }
        if let Some(target) = target.binary {
            let logits = linear(&hp.w_bin, &hp.b_bin, &x, NUM_BINARY_TARGETS);
            for o in 0..NUM_BINARY_TARGETS {
                let p = sigmoid(logits[o]);
                let y = f32::from(target[o]);
                loss += -(y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln());
                let d_logit = p - y;
                g.b_bin[o] += d_logit;
                for j in 0..n {
                    g.w_bin[o * n + j] += d_logit * x[j];
                    grad_x[j] += d_logit * hp.w_bin[o * n + j];
                }
            }
        }
    }

    let mut grad_r = vec![0.0f32; num_neurons];
    for (j, &dense) in head.subset.iter().enumerate() {
        grad_r[dense as usize] += grad_x[j];
    }
    (loss, grads, grad_r)
}

// --- Ridge-regression probe (FLY.md §7's "understands physics" metric; no backprop) -------------

/// A fitted linear probe: `y_hat = W . x + b`, one row of `W`/one `b` entry per target dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct RidgeProbe {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
    pub num_features: usize,
    pub num_targets: usize,
}

impl RidgeProbe {
    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        self.w
            .iter()
            .zip(&self.b)
            .map(|(row, &b)| b + row.iter().zip(x).map(|(&wi, &xi)| wi * xi).sum::<f32>())
            .collect()
    }

    /// `R²` per target dimension over `(features, targets)` pairs — the actual FLY.md §7 metric
    /// ("R²/NLL of a separate linear probe on frozen state").
    pub fn r_squared(&self, features: &[Vec<f32>], targets: &[Vec<f32>]) -> Vec<f64> {
        let n = features.len().max(1);
        let mut mean = vec![0.0f64; self.num_targets];
        for t in targets {
            for (m, &v) in mean.iter_mut().zip(t) {
                *m += f64::from(v);
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        let mut ss_res = vec![0.0f64; self.num_targets];
        let mut ss_tot = vec![0.0f64; self.num_targets];
        for (x, t) in features.iter().zip(targets) {
            let pred = self.predict(x);
            for d in 0..self.num_targets {
                let res = f64::from(t[d]) - f64::from(pred[d]);
                ss_res[d] += res * res;
                let dev = f64::from(t[d]) - mean[d];
                ss_tot[d] += dev * dev;
            }
        }
        ss_res
            .iter()
            .zip(&ss_tot)
            .map(|(&res, &tot)| if tot > 1e-12 { 1.0 - res / tot } else { 0.0 })
            .collect()
    }
}

/// Fits [`RidgeProbe`] by closed-form ridge regression (normal equations, solved by Gauss-Jordan
/// elimination — deliberately **not** gradient descent: FLY.md §7 requires this probe to *not*
/// backpropagate into the network, so its own fitting procedure must not use gradients into
/// anything at all, network or probe). `l2 > 0` keeps the system solvable even when
/// `features.len() < num_features` (a small held-out probe set is exactly this regime).
///
/// **Scaling limit (review round 1, F17, CONFIRMED):** Gauss-Jordan elimination on the `d x d`
/// normal-equations matrix (`d = num_features + 1`) is `O(d^3)` time and `O(d^2)` memory — fine
/// for a `crate::decoder`/`crate::world_model`-sized `num_features` (tens to a few hundred), but
/// not for probing the *entire* central-brain population directly on the M graph
/// (`d ~ 12000`+ dense hidden neurons would mean a `~12000 x 12000` `f64` matrix, well over 1 GiB,
/// and a fully impractical elimination cost). A caller that wants to probe a subset that large
/// should pick a smaller `features` subset (e.g. `crate::world_model::WorldModelConfig::
/// neuron_subset`'s own subsetting, or a random/PCA-reduced projection) rather than call this
/// directly on every hidden neuron — this function does not do that subsetting itself, and does
/// not implement a bigger-`d`-capable solver (conjugate gradient / Cholesky on a subset) since
/// nothing in this task's own acceptance criteria calls for probing the full M population at once.
pub fn fit_ridge_probe(features: &[Vec<f32>], targets: &[Vec<f32>], l2: f32) -> Result<RidgeProbe, WorldModelError> {
    if features.is_empty() {
        return Err(WorldModelError::InvalidConfig(
            "fit_ridge_probe needs at least one sample".to_string(),
        ));
    }
    let num_features = features[0].len();
    let num_targets = targets[0].len();
    if targets.len() != features.len() {
        return Err(WorldModelError::InvalidConfig(
            "features and targets must have the same number of samples".to_string(),
        ));
    }
    for (x, t) in features.iter().zip(targets) {
        if x.len() != num_features || t.len() != num_targets {
            return Err(WorldModelError::InvalidConfig(
                "every sample's feature/target vector must have the same length".to_string(),
            ));
        }
    }
    if !(l2 >= 0.0 && l2.is_finite()) {
        return Err(WorldModelError::InvalidConfig("l2 must be >= 0".to_string()));
    }

    // Design matrix with an appended constant column (bias) — `d = num_features + 1`.
    let d = num_features + 1;
    // Normal equations in f64 (the same "accumulate wide, store narrow" discipline
    // `crate::optim::ParamGradients::global_norm` uses): `a = X^T X + l2*I` (bias column excluded
    // from the ridge penalty, standard convention), `rhs = X^T Y`.
    let mut a = vec![0.0f64; d * d];
    let mut rhs = vec![0.0f64; d * num_targets];
    for (x, t) in features.iter().zip(targets) {
        let mut xb = vec![0.0f64; d];
        xb[..num_features].copy_from_slice(&x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
        xb[num_features] = 1.0;
        for i in 0..d {
            for j in 0..d {
                a[i * d + j] += xb[i] * xb[j];
            }
            for (o, &ty) in t.iter().enumerate() {
                rhs[i * num_targets + o] += xb[i] * f64::from(ty);
            }
        }
    }
    for i in 0..num_features {
        a[i * d + i] += f64::from(l2);
    }

    let solution = gauss_jordan_solve(&mut a, &mut rhs, d, num_targets).ok_or_else(|| {
        WorldModelError::InvalidConfig(
            "normal-equations matrix is singular even after ridge regularization".to_string(),
        )
    })?;

    let mut w = vec![vec![0.0f32; num_features]; num_targets];
    let mut b = vec![0.0f32; num_targets];
    for o in 0..num_targets {
        for f in 0..num_features {
            w[o][f] = solution[f * num_targets + o] as f32;
        }
        b[o] = solution[num_features * num_targets + o] as f32;
    }
    Ok(RidgeProbe {
        w,
        b,
        num_features,
        num_targets,
    })
}

/// Solves `a . x = rhs` for `x` (`rhs` has `num_rhs` columns, solved simultaneously) via
/// Gauss-Jordan elimination with partial pivoting, `a` (`d x d`) and `rhs` (`d x num_rhs`)
/// consumed/overwritten in place. Returns `None` if a pivot is (near-)singular even after ridge
/// regularization — never panics or divides by zero silently.
fn gauss_jordan_solve(a: &mut [f64], rhs: &mut [f64], d: usize, num_rhs: usize) -> Option<Vec<f64>> {
    for col in 0..d {
        let mut pivot_row = col;
        let mut pivot_val = a[col * d + col].abs();
        for row in (col + 1)..d {
            let v = a[row * d + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < 1e-12 {
            return None;
        }
        if pivot_row != col {
            for j in 0..d {
                a.swap(col * d + j, pivot_row * d + j);
            }
            for j in 0..num_rhs {
                rhs.swap(col * num_rhs + j, pivot_row * num_rhs + j);
            }
        }
        let pivot = a[col * d + col];
        for j in 0..d {
            a[col * d + j] /= pivot;
        }
        for j in 0..num_rhs {
            rhs[col * num_rhs + j] /= pivot;
        }
        for row in 0..d {
            if row == col {
                continue;
            }
            let factor = a[row * d + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..d {
                a[row * d + j] -= factor * a[col * d + j];
            }
            for j in 0..num_rhs {
                rhs[row * num_rhs + j] -= factor * rhs[col * num_rhs + j];
            }
        }
    }
    Some(rhs.to_vec())
}

#[cfg(test)]
mod tests;
