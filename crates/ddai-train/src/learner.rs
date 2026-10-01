//! What the trainer needs from a model, and the three implementations: the fly ([`FlyLearner`]),
//! the MLP and the GRU ([`ControlLearner`]).
//!
//! Every model exposes its parameters as **one flat vector** with a per-parameter base learning
//! rate, so the optimiser, gradient clipping, schedule, batching, evaluation and checkpointing in
//! [`crate::trainer`] are literally the same code for all of them (D-014: "trained identically").
//! The differences that remain are the model's own structure: the fly's connectome-shaped
//! parameters, its L2 anchor towards the connectome's initial strengths (FLY.md §8) and its
//! activity regulariser. Mirror augmentation is the same for all three: the fly's brain is only
//! *approximately* mirror-symmetric on the real graph (`ddai-fly/tests/brain_mirror.rs` checks
//! per-type means to a tolerance, exactness holds on a synthetic symmetric graph), so mirrored
//! windows are new data for it too, and training the controls on mirrored windows alone would give
//! them twice the data (review F4 of E-005).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ddai_controls::bundle::{ControlBundle, save_control_bundle};
use ddai_controls::features::{extract, input_dim};
use ddai_controls::net::SeqNet;
use ddai_fly::backward::BackwardIndex;
use ddai_fly::batched::{BatchedEngine, BatchedPlan, TrainBackend};
use ddai_fly::bc::{HeadLogits, HeadThresholds, LossConfig, StepLoss};
use ddai_fly::brain_bc::{BcSequence, BcStepConfig, BcStepOutput, BcWorkspace, brain_bc_forward, brain_bc_step};
use ddai_fly::brain_bc_batched::brain_bc_batched_step;
use ddai_fly::brain_config::{BrainConfig, parse_brain_config};
use ddai_fly::bundle::{BUNDLE_FORMAT_VERSION, BundleMeta, FlyBundle, save_bundle, sha256_hex_of_file};
use ddai_fly::calibration::calibrate_from_windows;
use ddai_fly::config::FlyConfig;
use ddai_fly::decoder::{DecoderGradients, DecoderModel, DecoderParams, DnCalibration, calibrate_from_rest};
use ddai_fly::encoder::{EncoderGradients, EncoderModel, EncoderParams, RayGridConfig, RayGridFeatures};
use ddai_fly::model::FlyModel;
use ddai_fly::optim::{ActivityRegularizerConfig, ParamGradients};
use ddai_fly::params::FlyParams;
use ddai_fly::state::FlyState;
use serde::{Deserialize, Serialize};

use crate::seq::Window;

/// Per-thread scratch of a learner.
pub enum Workspace {
    None,
    Fly(Box<BcWorkspace>),
}

/// What one window contributed (sums over its scored decisions).
#[derive(Debug, Clone, Copy, Default)]
pub struct WindowStats {
    pub loss: StepLoss,
    pub weight_sum: f32,
    pub activity_loss: f32,
}

pub type LearnerResult<T> = Result<T, String>;

pub trait Learner: Send + Sync {
    /// Short label for logs and tables (`fly`, `mlp-h5`, ...).
    fn label(&self) -> String;
    fn num_params(&self) -> usize;
    fn params(&self) -> Vec<f32>;
    /// Replaces every parameter (and recomputes whatever is derived from them).
    fn set_params(&mut self, flat: &[f32]) -> LearnerResult<()>;
    /// Base learning rate of every parameter (the schedule scales all of them together).
    fn base_lrs(&self) -> Vec<f32>;
    fn new_workspace(&self, max_window: usize) -> Workspace;
    /// Forward + backward over `window`; **adds** the gradient of the summed loss into `grad`.
    fn window_grad(&self, window: &Window, loss: &LossConfig, ws: &mut Workspace, grad: &mut [f32]) -> WindowStats;
    /// Whether [`Learner::batch_grad`] is available: a model with a batched backend (the fly's
    /// `backend = "batched"`) runs a whole batch of windows through one forward/backward pass, and
    /// the trainer then calls it instead of [`Learner::window_grad`] once per window.
    fn uses_batched_backend(&self) -> bool {
        false
    }
    /// Forward + backward over all of `windows` at once (see [`Learner::uses_batched_backend`]);
    /// **adds** the summed gradient into `grad` and returns every window's stats, in window order.
    fn batch_grad(
        &self,
        _windows: &[Window],
        _loss: &LossConfig,
        _grad: &mut [f32],
    ) -> LearnerResult<Vec<WindowStats>> {
        Err("this learner has no batched backend".to_string())
    }
    /// Forward only: the logits of every decision of `window`.
    fn window_logits(&self, window: &Window) -> Vec<HeadLogits>;
    /// Adds a regulariser's gradient (not normalised by the batch weight) and returns its value.
    fn regularizer_grad(&self, _grad: &mut [f32]) -> f32 {
        0.0
    }
    /// Called every `rest_refresh_every` steps (the fly recomputes its resting state).
    fn refresh(&mut self) {}
    /// Whether the trainer should randomly mirror windows for this model.
    fn mirror_augment(&self) -> bool;
    /// The decision thresholds of the jump/hook/fire heads stored in the checkpoint.
    fn thresholds(&self) -> HeadThresholds;
    fn set_thresholds(&mut self, thresholds: HeadThresholds);
    /// Writes the checkpoint the arena loads (`fly` bundle or control bundle).
    fn save(&self, path: &Path, meta: BundleMeta) -> LearnerResult<()>;
}

// --- The fly ------------------------------------------------------------------------------------

/// Learning rates and regularisers of the fly's training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlyTrainConfig {
    pub lr_a: f32,
    pub lr_b: f32,
    pub lr_theta: f32,
    pub lr_encoder: f32,
    pub lr_decoder: f32,
    /// L2 pull of `a` (the type-pair strengths) towards their initial value (FLY.md §8).
    pub l2_a: f32,
    pub activity_weight: f32,
    pub activity_low: f32,
    pub activity_high: f32,
    /// Windows used to fit the DN calibration from real scenes (`0` = the resting-state protocol).
    pub calibration_windows: usize,
    /// Initial `softplus(a)` of every type-pair strength (`0` = the crate default,
    /// `ddai_fly::params::DEFAULT_ALPHA_INIT`).
    pub alpha_init: f32,
    /// Forward/backward backend: `"per-seq"` (the default: every window through task 7.2's
    /// per-sequence BPTT, windows in parallel) or `"batched"` (task 7.2b: all windows of a batch
    /// at once, state `[neuron][window]`, threads split the neurons -- what makes the M graph
    /// trainable). Same loss and gradients up to f32 summation order.
    pub backend: TrainBackend,
    /// Cap (MiB) on the batched backend's working set (`0` = none): when a batch's `r`/`X`/`f'(V)`
    /// recording does not fit, BPTT over the windows is chunked in time with recomputation
    /// (exact gradients, one extra forward pass).
    pub batched_memory_cap_mb: usize,
    /// Work (`edges x 8-lane cells`) below which a batched substep runs on the calling thread
    /// instead of the thread pool (`None` = the engine's default, see
    /// `ddai_fly::batched::DEFAULT_PAR_MIN_EDGE_CELLS`; `0` = always on the pool). A tuning knob
    /// for the thread rendezvous cost: it never changes the results (bitwise).
    pub batched_parallel_threshold: Option<usize>,
}

impl Default for FlyTrainConfig {
    fn default() -> Self {
        FlyTrainConfig {
            lr_a: 5e-4,
            lr_b: 2e-3,
            lr_theta: 2e-3,
            lr_encoder: 2e-2,
            lr_decoder: 2e-2,
            l2_a: 1e-4,
            activity_weight: 0.0,
            activity_low: 0.02,
            activity_high: 6.0,
            calibration_windows: 300,
            alpha_init: 0.0,
            backend: TrainBackend::default(),
            batched_memory_cap_mb: 3072,
            batched_parallel_threshold: None,
        }
    }
}

/// Parameter groups of the fly's flat vector, in order.
struct Layout {
    a: usize,
    b: usize,
    theta: usize,
    g: usize,
    c: usize,
    bin: usize,
    dir_lr_w: usize,
    stop_w: usize,
    jump_w: usize,
    hook_w: usize,
    fire_w: usize,
    aim_pair: usize,
    aim_unpaired: usize,
}

impl Layout {
    fn of(fly: &FlyParams, enc: &EncoderParams, dec: &DecoderParams) -> Layout {
        Layout {
            a: fly.a.len(),
            b: fly.b.len(),
            theta: fly.theta.len(),
            g: enc.g.len(),
            c: enc.c.len(),
            bin: enc.bin_gain.len(),
            dir_lr_w: dec.direction_lr_w.len(),
            stop_w: dec.direction_stop_w.len(),
            jump_w: dec.jump_w.len(),
            hook_w: dec.hook_w.len(),
            fire_w: dec.fire_w.len(),
            aim_pair: dec.aim_pair_theta.len(),
            aim_unpaired: dec.aim_unpaired_theta.len(),
        }
    }

    fn total(&self) -> usize {
        self.a
            + self.b
            + self.theta
            + self.g
            + self.c
            + self.bin
            + self.dir_lr_w
            + 1
            + self.stop_w
            + 1
            + self.jump_w
            + 1
            + self.hook_w
            + 1
            + self.fire_w
            + 1
            + self.aim_pair
            + self.aim_unpaired
    }

    /// Start offsets of `(a, b, theta, g, c, decoder...)`.
    fn decoder_start(&self) -> usize {
        self.a + self.b + self.theta + self.g + self.c + self.bin
    }
}

fn push_decoder(out: &mut Vec<f32>, d: &DecoderParams) {
    out.extend_from_slice(&d.direction_lr_w);
    out.push(d.direction_lr_b);
    out.extend_from_slice(&d.direction_stop_w);
    out.push(d.direction_stop_b);
    out.extend_from_slice(&d.jump_w);
    out.push(d.jump_b);
    out.extend_from_slice(&d.hook_w);
    out.push(d.hook_b);
    out.extend_from_slice(&d.fire_w);
    out.push(d.fire_b);
    out.extend_from_slice(&d.aim_pair_theta);
    out.extend_from_slice(&d.aim_unpaired_theta);
}

fn take<'a>(flat: &'a [f32], at: &mut usize, n: usize) -> &'a [f32] {
    let s = &flat[*at..*at + n];
    *at += n;
    s
}

/// A decoder's parameters from a flat slice laid out by [`push_decoder`].
fn decoder_from_flat(flat: &[f32], l: &Layout) -> DecoderParams {
    let mut at = 0;
    let dir_lr_w = take(flat, &mut at, l.dir_lr_w).to_vec();
    let dir_lr_b = take(flat, &mut at, 1)[0];
    let stop_w = take(flat, &mut at, l.stop_w).to_vec();
    let stop_b = take(flat, &mut at, 1)[0];
    let jump_w = take(flat, &mut at, l.jump_w).to_vec();
    let jump_b = take(flat, &mut at, 1)[0];
    let hook_w = take(flat, &mut at, l.hook_w).to_vec();
    let hook_b = take(flat, &mut at, 1)[0];
    let fire_w = take(flat, &mut at, l.fire_w).to_vec();
    let fire_b = take(flat, &mut at, 1)[0];
    let aim_pair_theta = take(flat, &mut at, l.aim_pair).to_vec();
    let aim_unpaired_theta = take(flat, &mut at, l.aim_unpaired).to_vec();
    DecoderParams {
        direction_lr_w: dir_lr_w,
        direction_lr_b: dir_lr_b,
        direction_stop_w: stop_w,
        direction_stop_b: stop_b,
        jump_w,
        jump_b,
        hook_w,
        hook_b,
        fire_w,
        fire_b,
        aim_pair_theta,
        aim_unpaired_theta,
    }
}

/// The fly as a [`Learner`]: connectome parameters, encoder, tied decoder, frozen calibration.
pub struct FlyLearner {
    flyg_path: PathBuf,
    flyg_sha256: String,
    brain_config_toml: String,
    brain_config: BrainConfig,
    model: FlyModel,
    index: BackwardIndex,
    encoder: EncoderModel,
    decoder: DecoderModel,
    fly_params: FlyParams,
    enc_params: EncoderParams,
    dec_params: DecoderParams,
    calib: DnCalibration,
    a_init: Vec<f32>,
    v_rest: Vec<f32>,
    cfg: FlyTrainConfig,
    layout: Layout,
    thresholds: HeadThresholds,
    /// The batched engine (buffers + topology plan), present iff `cfg.backend` is `Batched`.
    batched: Option<Mutex<BatchedEngine>>,
}

impl FlyLearner {
    /// A fresh fly: default connectome parameters (`seed`), unit encoder gains, a zero decoder.
    /// `calibration_windows` (observations of real scenes) fit the DN calibration when given; an
    /// empty list falls back to [`calibrate_from_rest`].
    pub fn init(
        flyg_path: &Path,
        brain_config_path: &Path,
        seed: u64,
        cfg: FlyTrainConfig,
        calibration_windows: &[Vec<ddai_brain::Observation>],
    ) -> LearnerResult<Self> {
        let flyg = ddai_flyg::load(flyg_path).map_err(|e| format!("{}: {e}", flyg_path.display()))?;
        let flyg_sha256 = sha256_hex_of_file(flyg_path).map_err(|e| e.to_string())?;
        let toml =
            std::fs::read_to_string(brain_config_path).map_err(|e| format!("{}: {e}", brain_config_path.display()))?;
        let fly_config = FlyConfig::default();
        let fly_params = if cfg.alpha_init > 0.0 {
            FlyParams::init_default_with_alpha(&flyg, &fly_config, seed, cfg.alpha_init)
        } else {
            FlyParams::init_default(&flyg, &fly_config, seed)
        };
        let mut learner = Self::build(
            flyg,
            flyg_path.to_path_buf(),
            flyg_sha256,
            toml,
            fly_config,
            fly_params,
            None,
            None,
            None,
            cfg,
        )?;
        learner.refresh();
        let min_sigma = learner.brain_config.decoder.min_sigma;
        learner.calib = if calibration_windows.is_empty() {
            calibrate_from_rest(&learner.model, seed, min_sigma).map_err(|e| e.to_string())?
        } else {
            calibrate_from_windows(
                &learner.model,
                &learner.encoder,
                &learner.enc_params,
                &learner.v_rest,
                calibration_windows,
                calibration_windows
                    .iter()
                    .map(Vec::len)
                    .min()
                    .unwrap_or(0)
                    .saturating_sub(1)
                    .min(8),
                min_sigma,
            )
            .map_err(|e| e.to_string())?
        };
        Ok(learner)
    }

    /// Restores a fly from a bundle (and the `.flyg` it names or `flyg_path`).
    pub fn from_bundle(bundle: FlyBundle, flyg_path: &Path, cfg: FlyTrainConfig) -> LearnerResult<Self> {
        let flyg = ddai_flyg::load(flyg_path).map_err(|e| format!("{}: {e}", flyg_path.display()))?;
        let sha = sha256_hex_of_file(flyg_path).map_err(|e| e.to_string())?;
        if sha != bundle.flyg_sha256 {
            return Err(format!(
                "bundle was trained against a different .flyg ({} vs {sha})",
                bundle.flyg_sha256
            ));
        }
        let thresholds = bundle.thresholds;
        let mut l = Self::build(
            flyg,
            flyg_path.to_path_buf(),
            sha,
            bundle.brain_config_toml,
            bundle.fly_config,
            bundle.fly_params,
            Some(bundle.encoder_params),
            Some(bundle.decoder_params),
            Some(bundle.calibration),
            cfg,
        )?;
        l.thresholds = thresholds;
        l.refresh();
        Ok(l)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        flyg: ddai_flyg::Flyg,
        flyg_path: PathBuf,
        flyg_sha256: String,
        brain_config_toml: String,
        fly_config: FlyConfig,
        fly_params: FlyParams,
        enc_params: Option<EncoderParams>,
        dec_params: Option<DecoderParams>,
        calib: Option<DnCalibration>,
        cfg: FlyTrainConfig,
    ) -> LearnerResult<Self> {
        let brain_config = parse_brain_config(&brain_config_toml).map_err(|e| format!("brain config: {e}"))?;
        let model = FlyModel::new(flyg, fly_config, fly_params.clone()).map_err(|e| format!("fly model: {e}"))?;
        let index = BackwardIndex::build(&model);
        let encoder = EncoderModel::new(&model, brain_config.ray_grid, &brain_config.proprioception)
            .map_err(|e| format!("encoder: {e}"))?;
        let decoder = DecoderModel::new(&model, brain_config.decoder.clone()).map_err(|e| format!("decoder: {e}"))?;
        let enc_params = enc_params.unwrap_or_else(|| encoder.init_params());
        let dec_params = dec_params.unwrap_or_else(|| decoder.init_default_params());
        enc_params
            .validate_shape(encoder.num_params(), encoder.ray_grid_config().num_distance_bins)
            .map_err(|e| e.to_string())?;
        decoder.validate_params_shape(&dec_params).map_err(|e| e.to_string())?;
        let calib = match calib {
            Some(c) => {
                c.validate_shape(decoder.num_outputs()).map_err(|e| e.to_string())?;
                c
            }
            None => DnCalibration {
                mu: vec![0.0; decoder.num_outputs()],
                sigma: vec![1.0; decoder.num_outputs()],
            },
        };
        let layout = Layout::of(&fly_params, &enc_params, &dec_params);
        let batched = (cfg.backend == TrainBackend::Batched).then(|| {
            let mut plan = BatchedPlan::new(&model);
            if let Some(threshold) = cfg.batched_parallel_threshold {
                plan = plan.with_parallel_threshold(threshold);
            }
            Mutex::new(BatchedEngine::with_plan(plan))
        });
        Ok(FlyLearner {
            flyg_path,
            flyg_sha256,
            brain_config_toml,
            brain_config,
            a_init: fly_params.a.clone(),
            v_rest: vec![0.0; model.num_neurons()],
            model,
            index,
            encoder,
            decoder,
            fly_params,
            enc_params,
            dec_params,
            calib,
            cfg,
            layout,
            thresholds: HeadThresholds::default(),
            batched,
        })
    }

    pub fn model(&self) -> &FlyModel {
        &self.model
    }
    pub fn calibration(&self) -> &DnCalibration {
        &self.calib
    }
    pub fn brain_config(&self) -> &BrainConfig {
        &self.brain_config
    }
    pub fn train_config(&self) -> &FlyTrainConfig {
        &self.cfg
    }

    /// The bundle of the current parameters.
    pub fn to_bundle(&self, meta: BundleMeta) -> FlyBundle {
        FlyBundle {
            format_version: BUNDLE_FORMAT_VERSION,
            flyg_sha256: self.flyg_sha256.clone(),
            flyg_path_hint: self.flyg_path.to_string_lossy().into_owned(),
            brain_config_toml: self.brain_config_toml.clone(),
            fly_config: *self.model.config(),
            fly_params: self.fly_params.clone(),
            encoder_params: self.enc_params.clone(),
            decoder_params: self.dec_params.clone(),
            calibration: self.calib.clone(),
            meta,
            thresholds: self.thresholds,
        }
    }

    fn act_config(&self) -> ActivityRegularizerConfig {
        ActivityRegularizerConfig {
            weight: self.cfg.activity_weight,
            low: self.cfg.activity_low,
            high: self.cfg.activity_high,
        }
    }

    fn add_step_grads(&self, out: &BcStepOutput, grad: &mut [f32]) {
        self.add_parts_grads(&out.fly, &out.encoder, &out.decoder, grad);
    }

    /// Adds the three gradient groups of a step into the flat vector (the layout of
    /// [`Learner::params`]).
    fn add_parts_grads(
        &self,
        fly: &ParamGradients,
        encoder: &EncoderGradients,
        d: &DecoderGradients,
        grad: &mut [f32],
    ) {
        let l = &self.layout;
        let mut at = 0;
        let mut add = |src: &[f32]| {
            for (g, s) in grad[at..at + src.len()].iter_mut().zip(src) {
                *g += s;
            }
            at += src.len();
        };
        add(&fly.a);
        add(&fly.b);
        add(&fly.theta);
        add(&encoder.g);
        add(&encoder.c);
        add(&encoder.bin_gain);
        add(&d.direction_lr_w);
        add(&[d.direction_lr_b]);
        add(&d.direction_stop_w);
        add(&[d.direction_stop_b]);
        add(&d.jump_w);
        add(&[d.jump_b]);
        add(&d.hook_w);
        add(&[d.hook_b]);
        add(&d.fire_w);
        add(&[d.fire_b]);
        add(&d.aim_pair_theta);
        add(&d.aim_unpaired_theta);
        debug_assert_eq!(at, l.total());
    }
}

impl Learner for FlyLearner {
    fn label(&self) -> String {
        "fly".to_string()
    }

    fn num_params(&self) -> usize {
        self.layout.total()
    }

    fn params(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.layout.total());
        out.extend_from_slice(&self.fly_params.a);
        out.extend_from_slice(&self.fly_params.b);
        out.extend_from_slice(&self.fly_params.theta);
        out.extend_from_slice(&self.enc_params.g);
        out.extend_from_slice(&self.enc_params.c);
        out.extend_from_slice(&self.enc_params.bin_gain);
        push_decoder(&mut out, &self.dec_params);
        out
    }

    fn set_params(&mut self, flat: &[f32]) -> LearnerResult<()> {
        let l = &self.layout;
        if flat.len() != l.total() {
            return Err(format!("expected {} parameters, got {}", l.total(), flat.len()));
        }
        let mut at = 0;
        let a = take(flat, &mut at, l.a).to_vec();
        let b = take(flat, &mut at, l.b).to_vec();
        let theta = take(flat, &mut at, l.theta).to_vec();
        let g = take(flat, &mut at, l.g).to_vec();
        let c = take(flat, &mut at, l.c).to_vec();
        let bin_gain = take(flat, &mut at, l.bin).to_vec();
        let dec = decoder_from_flat(&flat[l.decoder_start()..], l);
        let fly = FlyParams { a, b, theta };
        self.model.set_params(fly.clone()).map_err(|e| e.to_string())?;
        self.fly_params = fly;
        self.enc_params = EncoderParams { g, c, bin_gain };
        self.dec_params = dec;
        Ok(())
    }

    fn base_lrs(&self) -> Vec<f32> {
        let l = &self.layout;
        let mut lrs = Vec::with_capacity(l.total());
        let cfg = &self.cfg;
        lrs.extend(std::iter::repeat_n(cfg.lr_a, l.a));
        lrs.extend(std::iter::repeat_n(cfg.lr_b, l.b));
        lrs.extend(std::iter::repeat_n(cfg.lr_theta, l.theta));
        lrs.extend(std::iter::repeat_n(cfg.lr_encoder, l.g + l.c + l.bin));
        let n_dec = l.total() - l.decoder_start();
        lrs.extend(std::iter::repeat_n(cfg.lr_decoder, n_dec));
        lrs
    }

    fn new_workspace(&self, max_window: usize) -> Workspace {
        Workspace::Fly(Box::new(BcWorkspace::new(&self.model, &self.encoder, max_window)))
    }

    fn window_grad(&self, window: &Window, loss: &LossConfig, ws: &mut Workspace, grad: &mut [f32]) -> WindowStats {
        let Workspace::Fly(ws) = ws else {
            panic!("FlyLearner needs a Fly workspace");
        };
        let seq = BcSequence {
            v_init: self.v_rest.clone(),
            observations: window.observations.clone(),
            targets: window.targets.clone(),
        };
        let cfg = BcStepConfig {
            loss: *loss,
            activity: self.act_config(),
        };
        let out = brain_bc_step(
            &self.model,
            &self.index,
            &self.encoder,
            &self.enc_params,
            &self.decoder,
            &self.dec_params,
            &self.calib,
            &seq,
            &cfg,
            ws,
        );
        self.add_step_grads(&out, grad);
        WindowStats {
            loss: out.loss,
            weight_sum: out.weight_sum,
            activity_loss: out.activity_loss,
        }
    }

    fn uses_batched_backend(&self) -> bool {
        self.batched.is_some()
    }

    fn batch_grad(&self, windows: &[Window], loss: &LossConfig, grad: &mut [f32]) -> LearnerResult<Vec<WindowStats>> {
        let engine = self
            .batched
            .as_ref()
            .ok_or("the fly learner runs the per-seq backend")?;
        let seqs: Vec<BcSequence> = windows
            .iter()
            .map(|w| BcSequence {
                v_init: self.v_rest.clone(),
                observations: w.observations.clone(),
                targets: w.targets.clone(),
            })
            .collect();
        let cfg = BcStepConfig {
            loss: *loss,
            activity: self.act_config(),
        };
        let cap = (self.cfg.batched_memory_cap_mb > 0).then(|| self.cfg.batched_memory_cap_mb.saturating_mul(1 << 20));
        let mut engine = engine.lock().map_err(|_| "batched engine lock poisoned".to_string())?;
        let out = brain_bc_batched_step(
            &self.model,
            &mut engine,
            &self.encoder,
            &self.enc_params,
            &self.decoder,
            &self.dec_params,
            &self.calib,
            &seqs,
            &cfg,
            cap,
        )
        .map_err(|e| e.to_string())?;
        // The connectome gradient is already the batch sum; the encoder/decoder parts are added
        // window by window in index order, like the per-window path's reduction.
        self.add_parts_grads(
            &out.fly,
            &self.encoder.zero_grads(),
            &self.decoder.zeros_gradients(),
            grad,
        );
        let no_fly = ParamGradients::zeros_like(self.model.params());
        let mut stats = Vec::with_capacity(out.windows.len());
        for w in &out.windows {
            self.add_parts_grads(&no_fly, &w.encoder, &w.decoder, grad);
            stats.push(WindowStats {
                loss: w.loss,
                weight_sum: w.weight_sum,
                activity_loss: w.activity_loss,
            });
        }
        Ok(stats)
    }

    fn window_logits(&self, window: &Window) -> Vec<HeadLogits> {
        brain_bc_forward(
            &self.model,
            &self.encoder,
            &self.enc_params,
            &self.decoder,
            &self.dec_params,
            &self.calib,
            &self.v_rest,
            &window.observations,
        )
    }

    fn regularizer_grad(&self, grad: &mut [f32]) -> f32 {
        if self.cfg.l2_a == 0.0 {
            return 0.0;
        }
        let mut value = 0.0f32;
        for (i, (&a, &a0)) in self.fly_params.a.iter().zip(&self.a_init).enumerate() {
            let d = a - a0;
            grad[i] += self.cfg.l2_a * d;
            value += 0.5 * self.cfg.l2_a * d * d;
        }
        value
    }

    fn refresh(&mut self) {
        let mut state = FlyState::new(&self.model);
        let _ = state.warm_up(&self.model);
        self.v_rest = state.v().to_vec();
    }

    fn mirror_augment(&self) -> bool {
        true
    }

    fn thresholds(&self) -> HeadThresholds {
        self.thresholds
    }

    fn set_thresholds(&mut self, thresholds: HeadThresholds) {
        self.thresholds = thresholds;
    }

    fn save(&self, path: &Path, meta: BundleMeta) -> LearnerResult<()> {
        save_bundle(path, &self.to_bundle(meta)).map_err(|e| e.to_string())
    }
}

// --- The controls -------------------------------------------------------------------------------

/// An MLP or GRU as a [`Learner`].
pub struct ControlLearner {
    net: Box<dyn SeqNet>,
    ray_grid: RayGridConfig,
    lr: f32,
    thresholds: HeadThresholds,
}

impl ControlLearner {
    pub fn new(net: Box<dyn SeqNet>, ray_grid: RayGridConfig, lr: f32) -> Self {
        assert_eq!(net.input_dim(), input_dim(&ray_grid));
        ControlLearner {
            net,
            ray_grid,
            lr,
            thresholds: HeadThresholds::default(),
        }
    }

    pub fn net(&self) -> &dyn SeqNet {
        self.net.as_ref()
    }

    fn features(&self, window: &Window) -> Vec<Vec<f32>> {
        let mut scratch = RayGridFeatures::new(&self.ray_grid);
        window
            .observations
            .iter()
            .map(|o| {
                let mut x = Vec::with_capacity(self.net.input_dim());
                extract(o, &self.ray_grid, &mut scratch, &mut x);
                x
            })
            .collect()
    }
}

impl Learner for ControlLearner {
    fn label(&self) -> String {
        format!("{}-h{}", self.net.kind().name(), self.net.hidden())
    }
    fn num_params(&self) -> usize {
        self.net.num_params()
    }
    fn params(&self) -> Vec<f32> {
        self.net.params().to_vec()
    }
    fn set_params(&mut self, flat: &[f32]) -> LearnerResult<()> {
        if flat.len() != self.net.num_params() {
            return Err(format!(
                "expected {} parameters, got {}",
                self.net.num_params(),
                flat.len()
            ));
        }
        self.net.params_mut().copy_from_slice(flat);
        Ok(())
    }
    fn base_lrs(&self) -> Vec<f32> {
        vec![self.lr; self.net.num_params()]
    }
    fn new_workspace(&self, _max_window: usize) -> Workspace {
        Workspace::None
    }
    fn window_grad(&self, window: &Window, loss: &LossConfig, _ws: &mut Workspace, grad: &mut [f32]) -> WindowStats {
        let xs = self.features(window);
        let out = self.net.window_grad(&xs, &window.targets, loss, grad);
        WindowStats {
            loss: out.loss,
            weight_sum: out.weight_sum,
            activity_loss: 0.0,
        }
    }
    fn window_logits(&self, window: &Window) -> Vec<HeadLogits> {
        self.net.window_logits(&self.features(window))
    }
    fn mirror_augment(&self) -> bool {
        true
    }
    fn thresholds(&self) -> HeadThresholds {
        self.thresholds
    }
    fn set_thresholds(&mut self, thresholds: HeadThresholds) {
        self.thresholds = thresholds;
    }
    fn save(&self, path: &Path, meta: BundleMeta) -> LearnerResult<()> {
        save_control_bundle(
            path,
            &ControlBundle::from_net(self.net.as_ref(), self.ray_grid, meta, self.thresholds),
        )
        .map_err(|e| e.to_string())
    }
}
