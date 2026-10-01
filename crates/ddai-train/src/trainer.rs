//! The trainer: one loop for the fly, the MLP and the GRU (task 8.2, acceptance criteria 3, 5, 7).
//!
//! * **Batches.** `batch_windows` truncated-BPTT windows of `window_len` decisions per step, a
//!   fraction of them from the human corpus; windows are built inside the workers from a
//!   per-`(seed, step, index)` RNG and the per-window gradients are summed in index order, so a
//!   step is reproducible at any thread count and a resumed run continues exactly.
//! * **Optimiser.** Adam with a per-parameter base rate ([`Learner::base_lrs`]) times a warm-up +
//!   cosine schedule, global gradient-norm clipping (FLY.md §8: 1.0), non-finite steps skipped.
//!   The loss gradient is normalised by the batch's summed step weight; the learner's regulariser
//!   (the fly's L2 anchor towards the connectome) is added un-normalised.
//! * **Run directory** (PLAN §1.3): `config.toml`, `metrics.jsonl` (append-only), `status.json`,
//!   `state.bin` (parameters + Adam moments + step, written atomically: the resume point) and
//!   `checkpoints/` (bundles the arena loads).

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ddai_fly::bc::{LossConfig, StepLoss};
use ddai_fly::bundle::{BundleMeta, read_zstd_postcard, write_zstd_postcard};
use ddai_fly::rng::SplitMix64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::learner::{Learner, WindowStats, Workspace};
use crate::metrics::{HeadReport, MetricsAccumulator};
use crate::seq::{Corpus, Window};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainConfig {
    pub seed: u64,
    pub batch_windows: usize,
    pub window_len: usize,
    pub burn_in: usize,
    /// Fraction of a batch's windows drawn from the human corpus (when it has any).
    pub human_fraction: f32,
    /// Multiplies every parameter's base learning rate.
    pub lr_scale: f32,
    pub warmup_steps: u64,
    /// The schedule ends at `lr_final_frac` of the base rate.
    pub lr_final_frac: f32,
    pub grad_clip: f32,
    pub threads: usize,
    pub loss: LossConfig,
    /// Refresh the learner's derived state (the fly's resting state) this often.
    pub refresh_every: u64,
    pub log_every: u64,
    pub eval_windows: usize,
    pub keep_checkpoints: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            seed: 1,
            batch_windows: 24,
            window_len: 32,
            burn_in: 6,
            human_fraction: 0.25,
            lr_scale: 1.0,
            warmup_steps: 100,
            lr_final_frac: 0.1,
            grad_clip: 1.0,
            threads: 6,
            loss: LossConfig::default(),
            refresh_every: 25,
            log_every: 50,
            eval_windows: 300,
            keep_checkpoints: 3,
        }
    }
}

/// Adam over one flat vector with per-element learning rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam {
    pub t: u64,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl Adam {
    pub fn new(n: usize) -> Self {
        Adam {
            t: 0,
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    /// One step: `params -= lr[i] * m_hat / (sqrt(v_hat) + eps)`. Returns `false` (touching
    /// nothing) when `grad` has a non-finite entry.
    pub fn step(&mut self, params: &mut [f32], grad: &[f32], lrs: &[f32], lr_mult: f32) -> bool {
        if !grad.iter().all(|g| g.is_finite()) {
            return false;
        }
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        self.t += 1;
        let c1 = 1.0 - B1.powi(self.t.min(1 << 30) as i32);
        let c2 = 1.0 - B2.powi(self.t.min(1 << 30) as i32);
        for i in 0..params.len() {
            self.m[i] = B1 * self.m[i] + (1.0 - B1) * grad[i];
            self.v[i] = B2 * self.v[i] + (1.0 - B2) * grad[i] * grad[i];
            let update = (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + EPS);
            params[i] -= lrs[i] * lr_mult * update;
        }
        true
    }
}

/// Warm-up then cosine decay of the learning-rate multiplier over one phase.
pub fn schedule(step_in_phase: u64, phase_steps: u64, warmup: u64, final_frac: f32) -> f32 {
    if warmup > 0 && step_in_phase < warmup {
        return (step_in_phase + 1) as f32 / warmup as f32;
    }
    let span = phase_steps.saturating_sub(warmup).max(1) as f32;
    let progress = ((step_in_phase.saturating_sub(warmup)) as f32 / span).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    final_frac + (1.0 - final_frac) * cosine
}

/// Gradient-norm clipping over the flat vector; returns the pre-clip norm.
pub fn clip_norm(grad: &mut [f32], max_norm: f32) -> f32 {
    let norm = grad.iter().map(|&g| f64::from(g) * f64::from(g)).sum::<f64>().sqrt() as f32;
    if norm.is_finite() && norm > max_norm && max_norm > 0.0 {
        let k = max_norm / norm;
        for g in grad.iter_mut() {
            *g *= k;
        }
    }
    norm
}

/// A held-out set to report per-head metrics on.
pub struct EvalSet {
    pub name: String,
    pub corpus: Corpus,
}

/// One set's metrics.
#[derive(Debug, Clone, Serialize)]
pub struct EvalRecord {
    pub set: String,
    pub report: HeadReport,
}

#[derive(Debug)]
pub struct TrainError(pub String);

impl std::fmt::Display for TrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TrainError {}

impl From<String> for TrainError {
    fn from(s: String) -> Self {
        TrainError(s)
    }
}

/// The files of a run.
#[derive(Debug, Clone)]
pub struct RunDir {
    pub root: PathBuf,
}

fn io<E: std::fmt::Display>(path: &Path) -> impl FnOnce(E) -> TrainError + '_ {
    move |e| TrainError(format!("{}: {e}", path.display()))
}

impl RunDir {
    pub fn create(root: &Path) -> Result<Self, TrainError> {
        std::fs::create_dir_all(root.join("checkpoints")).map_err(io(root))?;
        Ok(RunDir {
            root: root.to_path_buf(),
        })
    }
    pub fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
    pub fn checkpoint(&self, name: &str) -> PathBuf {
        self.root.join("checkpoints").join(name)
    }
    pub fn append_metrics(&self, v: &Value) -> Result<(), TrainError> {
        let path = self.path("metrics.jsonl");
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(io(&path))?;
        writeln!(f, "{v}").map_err(io(&path))
    }
    pub fn write_status(&self, v: &Value) -> Result<(), TrainError> {
        let path = self.path("status.json");
        let tmp = self.path("status.json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(v).unwrap_or_default()).map_err(io(&tmp))?;
        std::fs::rename(&tmp, &path).map_err(io(&path))
    }
}

#[derive(Serialize, Deserialize)]
struct SavedState {
    step: u64,
    params: Vec<f32>,
    adam: Adam,
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// SplitMix-style mixing of `(seed, step, index)` into one RNG seed.
fn mix(seed: u64, step: u64, index: u64) -> u64 {
    let mut z =
        seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ index.wrapping_add(1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// What a phase of training produced.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseSummary {
    pub phase: String,
    pub steps: u64,
    pub end_step: u64,
    pub skipped_steps: u64,
    pub mean_loss_last_log: f64,
    pub elapsed_s: f64,
    pub decisions_per_s: f64,
}

pub struct Trainer {
    pub cfg: TrainConfig,
    learner: Box<dyn Learner>,
    teacher: Corpus,
    human: Corpus,
    adam: Adam,
    base_lrs: Vec<f32>,
    pub step: u64,
    run: Option<RunDir>,
    pool: rayon::ThreadPool,
}

impl Trainer {
    pub fn new(
        learner: Box<dyn Learner>,
        cfg: TrainConfig,
        teacher: Corpus,
        human: Corpus,
        run: Option<RunDir>,
    ) -> Result<Self, TrainError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.threads.max(1))
            .build()
            .map_err(|e| TrainError(format!("thread pool: {e}")))?;
        let n = learner.num_params();
        let base_lrs = learner.base_lrs();
        Ok(Trainer {
            adam: Adam::new(n),
            base_lrs,
            cfg,
            learner,
            teacher,
            human,
            step: 0,
            run,
            pool,
        })
    }

    pub fn learner(&self) -> &dyn Learner {
        self.learner.as_ref()
    }
    pub fn learner_mut(&mut self) -> &mut dyn Learner {
        self.learner.as_mut()
    }
    pub fn run_dir(&self) -> Option<&RunDir> {
        self.run.as_ref()
    }
    pub fn set_teacher(&mut self, teacher: Corpus) {
        self.teacher = teacher;
    }
    pub fn teacher_steps(&self) -> usize {
        self.teacher.scored_steps()
    }
    pub fn human_steps(&self) -> usize {
        self.human.scored_steps()
    }

    /// Restores `state.bin` when the run directory has one. Returns whether it did.
    pub fn resume(&mut self) -> Result<bool, TrainError> {
        let Some(run) = &self.run else { return Ok(false) };
        let path = run.path("state.bin");
        if !path.exists() {
            return Ok(false);
        }
        let s: SavedState = read_zstd_postcard(&path).map_err(|e| TrainError(e.to_string()))?;
        if s.params.len() != self.learner.num_params() || s.adam.m.len() != s.params.len() {
            return Err(TrainError(format!(
                "{}: saved state has {} parameters, the model has {}",
                path.display(),
                s.params.len(),
                self.learner.num_params()
            )));
        }
        self.learner.set_params(&s.params)?;
        self.learner.refresh();
        self.adam = s.adam;
        self.step = s.step;
        Ok(true)
    }

    /// Writes `state.bin` and the named bundle (also `last.bundle`), pruning old step bundles.
    pub fn save(&self, bundle_name: Option<&str>, notes: &str) -> Result<(), TrainError> {
        let Some(run) = &self.run else { return Ok(()) };
        let state = SavedState {
            step: self.step,
            params: self.learner.params(),
            adam: self.adam.clone(),
        };
        write_zstd_postcard(&run.path("state.bin"), &state, 3).map_err(|e| TrainError(e.to_string()))?;
        let meta = BundleMeta {
            seed: self.cfg.seed,
            git_commit: None,
            steps: self.step,
            notes: notes.to_string(),
        };
        self.learner.save(&run.checkpoint("last.bundle"), meta.clone())?;
        if let Some(name) = bundle_name {
            self.learner.save(&run.checkpoint(name), meta)?;
            self.prune(run)?;
        }
        Ok(())
    }

    fn prune(&self, run: &RunDir) -> Result<(), TrainError> {
        let dir = run.root.join("checkpoints");
        let mut steps: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(io(&dir))?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("step-") && n.ends_with(".bundle"))
            })
            .collect();
        steps.sort();
        while steps.len() > self.cfg.keep_checkpoints.max(1) {
            let old = steps.remove(0);
            let _ = std::fs::remove_file(old);
        }
        Ok(())
    }

    /// One batch: summed (unnormalised) gradient and stats, deterministic in `(seed, step)`.
    fn batch(&self, step: u64) -> (Vec<f32>, WindowStats, usize) {
        let cfg = &self.cfg;
        let learner = self.learner.as_ref();
        let n = learner.num_params();
        let mirror = learner.mirror_augment();
        let results: Vec<(Vec<f32>, WindowStats, usize)> = self.pool.install(|| {
            (0..cfg.batch_windows)
                .into_par_iter()
                .map_init(
                    || learner.new_workspace(cfg.window_len),
                    |ws: &mut Workspace, i| {
                        let mut rng = SplitMix64::new(mix(cfg.seed, step, i as u64));
                        let use_human = !self.human.is_empty()
                            && (self.teacher.is_empty() || rng.next_f32_unit() < cfg.human_fraction);
                        let corpus = if use_human { &self.human } else { &self.teacher };
                        let flip = mirror && rng.next_f32_unit() < 0.5;
                        let window: Window = corpus.sample_window(&mut rng, cfg.window_len, cfg.burn_in, flip);
                        let mut grad = vec![0.0f32; n];
                        let stats = learner.window_grad(&window, &cfg.loss, ws, &mut grad);
                        (grad, stats, window.len())
                    },
                )
                .collect()
        });
        let mut grad = vec![0.0f32; n];
        let mut stats = WindowStats::default();
        let mut decisions = 0;
        for (g, s, d) in results {
            for (a, b) in grad.iter_mut().zip(&g) {
                *a += b;
            }
            stats.loss.add(&s.loss);
            stats.weight_sum += s.weight_sum;
            stats.activity_loss += s.activity_loss;
            decisions += d;
        }
        (grad, stats, decisions)
    }

    /// Trains `n_steps` steps as one schedule phase; logs to `metrics.jsonl`, checkpoints at the
    /// end, evaluates on `eval` at the end and every `eval_every` steps (`0` = end only).
    ///
    /// The phase's schedule runs over global steps `phase_first .. phase_first + phase_steps`; a
    /// resumed run passes the same `phase_first`/`phase_steps` and only the `n_steps` that remain,
    /// so the learning rate continues where it stopped.
    pub fn train_phase(
        &mut self,
        phase: &str,
        phase_first: u64,
        phase_steps: u64,
        n_steps: u64,
        eval: &[EvalSet],
        eval_every: u64,
    ) -> Result<PhaseSummary, TrainError> {
        if self.teacher.is_empty() && self.human.is_empty() {
            return Err(TrainError("no training data".to_string()));
        }
        let start_step = self.step;
        let t0 = Instant::now();
        let mut window_t = Instant::now();
        let mut acc = StepLoss::default();
        let mut acc_weight = 0.0f32;
        let mut acc_act = 0.0f32;
        let mut acc_decisions = 0usize;
        let mut acc_steps = 0u64;
        let mut acc_norm = 0.0f32;
        let mut skipped = 0u64;
        let mut last_loss = f64::NAN;
        let mut total_decisions = 0usize;
        let mut params = self.learner.params();
        for k in 0..n_steps {
            let step = start_step + k;
            let (mut grad, stats, decisions) = self.batch(step);
            total_decisions += decisions;
            if stats.weight_sum <= 0.0 {
                // A batch with nothing to score: no update, but the step still counts (the schedule,
                // the batch seeds and the resume point are all keyed by the global step).
                skipped += 1;
                self.step = step + 1;
                continue;
            }
            let inv = 1.0 / stats.weight_sum;
            for g in &mut grad {
                *g *= inv;
            }
            let reg = self.learner.regularizer_grad(&mut grad);
            let norm = clip_norm(&mut grad, self.cfg.grad_clip);
            let mult = self.cfg.lr_scale
                * schedule(
                    step - phase_first,
                    phase_steps,
                    self.cfg.warmup_steps,
                    self.cfg.lr_final_frac,
                );
            if self.adam.step(&mut params, &grad, &self.base_lrs, mult) {
                self.learner.set_params(&params)?;
            } else {
                skipped += 1;
                params = self.learner.params();
            }
            self.step = step + 1;
            if self.cfg.refresh_every > 0 && self.step.is_multiple_of(self.cfg.refresh_every) {
                self.learner.refresh();
            }
            acc.add(&stats.loss);
            acc_weight += stats.weight_sum;
            acc_act += stats.activity_loss / stats.weight_sum;
            acc_decisions += decisions;
            acc_steps += 1;
            acc_norm += norm;
            let _ = reg;
            let log_now = self.step.is_multiple_of(self.cfg.log_every.max(1)) || k + 1 == n_steps;
            if log_now && acc_weight > 0.0 {
                let mut per = acc;
                per.scale(1.0 / acc_weight);
                last_loss = f64::from(per.total);
                let dt = window_t.elapsed().as_secs_f64().max(1e-9);
                let line = json!({
                    "kind": "train", "phase": phase, "step": self.step,
                    "loss": {"total": per.total, "dir": per.dir, "jump": per.jump, "hook": per.hook, "fire": per.fire, "aim": per.aim},
                    "activity": acc_act / acc_steps as f32,
                    "grad_norm": acc_norm / acc_steps as f32,
                    "lr_mult": mult,
                    "decisions_per_s": acc_decisions as f64 / dt,
                    "skipped": skipped,
                    "unix_s": unix_seconds(),
                });
                if let Some(run) = &self.run {
                    run.append_metrics(&line)?;
                    run.write_status(&json!({
                        "phase": phase, "step": self.step, "phase_step": k + 1, "phase_steps": n_steps,
                        "elapsed_s": t0.elapsed().as_secs_f64(), "loss": per.total,
                        "unix_s": unix_seconds(),
                    }))?;
                }
                acc = StepLoss::default();
                acc_weight = 0.0;
                acc_act = 0.0;
                acc_decisions = 0;
                acc_steps = 0;
                acc_norm = 0.0;
                window_t = Instant::now();
            }
            if eval_every > 0 && !eval.is_empty() && self.step.is_multiple_of(eval_every) && k + 1 < n_steps {
                self.log_eval(phase, eval)?;
                self.save(Some(&format!("step-{:08}.bundle", self.step)), phase)?;
            }
        }
        self.learner.refresh();
        if !eval.is_empty() {
            self.calibrate_thresholds(phase, eval)?;
            self.log_eval(phase, eval)?;
        }
        self.save(Some(&format!("step-{:08}.bundle", self.step)), phase)?;
        let elapsed = t0.elapsed().as_secs_f64();
        Ok(PhaseSummary {
            phase: phase.to_string(),
            steps: n_steps,
            end_step: self.step,
            skipped_steps: skipped,
            mean_loss_last_log: last_loss,
            elapsed_s: elapsed,
            decisions_per_s: total_decisions as f64 / elapsed.max(1e-9),
        })
    }

    /// Sets the model's decision thresholds to the **rate-matched** ones on the teacher validation
    /// sets (`teacher-val` and, from round 1, `dagger-val`: unseen games of the teacher's labels on
    /// states the student visited): each of jump/hook/fire then presses as often as the teacher does (the hook head on the decisions
    /// where the own hook is not out, i.e. its start rate: [`MetricsAccumulator::rate_matched_thresholds`]).
    /// Never on holdout or human sets. Runs at the end of every phase, so a bundle always carries
    /// thresholds matching its own weights; a run without such a set keeps the current ones. The
    /// choice and its reasons are on [`crate::metrics::rate_matched_threshold`].
    pub fn calibrate_thresholds(&mut self, phase: &str, eval: &[EvalSet]) -> Result<(), TrainError> {
        let mut total = MetricsAccumulator::new();
        let mut used = Vec::new();
        for set in eval.iter().filter(|s| CALIBRATION_SETS.contains(&s.name.as_str())) {
            if let Some(acc) = accumulate_set(self.learner.as_ref(), set, &self.cfg, &self.pool) {
                total.merge(acc);
                used.push(set.name.clone());
            }
        }
        if used.is_empty() {
            return Ok(());
        }
        let th = total.rate_matched_thresholds();
        th.validate().map_err(TrainError)?;
        self.learner.set_thresholds(th);
        if let Some(run) = &self.run {
            run.append_metrics(&json!({
                "kind": "thresholds", "phase": phase, "step": self.step, "sets": used,
                "jump": th.jump, "hook": th.hook, "fire": th.fire, "unix_s": unix_seconds(),
            }))?;
        }
        Ok(())
    }

    fn log_eval(&self, phase: &str, eval: &[EvalSet]) -> Result<Vec<EvalRecord>, TrainError> {
        let recs = self.evaluate(eval);
        if let Some(run) = &self.run {
            for r in &recs {
                run.append_metrics(&json!({"kind": "eval", "phase": phase, "step": self.step, "set": r.set, "report": r.report, "unix_s": unix_seconds()}))?;
            }
        }
        Ok(recs)
    }

    /// Per-head metrics of the current model on each set, over fixed windows (the same windows at
    /// every call, whatever the model or step, so curves and tables are comparable).
    pub fn evaluate(&self, eval: &[EvalSet]) -> Vec<EvalRecord> {
        eval.iter()
            .map(|set| evaluate_set(self.learner.as_ref(), set, &self.cfg, &self.pool))
            .collect()
    }
}

/// The evaluation sets whose labels are the teacher's on unseen games of the training arenas: the
/// ones the decision thresholds are calibrated on.
pub const CALIBRATION_SETS: [&str; 2] = ["teacher-val", "dagger-val"];

/// The scores of `learner` on `set` over the fixed evaluation windows (`None` for an empty set),
/// with the learner's own decision thresholds for the point metrics.
pub fn accumulate_set(
    learner: &dyn Learner,
    set: &EvalSet,
    cfg: &TrainConfig,
    pool: &rayon::ThreadPool,
) -> Option<MetricsAccumulator> {
    if set.corpus.is_empty() {
        return None;
    }
    let name_hash = set
        .name
        .bytes()
        .fold(0u64, |h, b| h.wrapping_mul(131).wrapping_add(u64::from(b)));
    let thresholds = learner.thresholds();
    let per_window: Vec<MetricsAccumulator> = pool.install(|| {
        (0..cfg.eval_windows)
            .into_par_iter()
            .map(|i| {
                let mut rng = SplitMix64::new(mix(0xE7A1_5EED ^ name_hash, 0, i as u64));
                let w = set.corpus.sample_window(&mut rng, cfg.window_len, cfg.burn_in, false);
                let logits = learner.window_logits(&w);
                let mut acc = MetricsAccumulator::with_thresholds(thresholds);
                acc.add_window(&w, &logits);
                acc
            })
            .collect()
    });
    let mut total = MetricsAccumulator::with_thresholds(thresholds);
    for a in per_window {
        total.merge(a);
    }
    Some(total)
}

/// Metrics of `learner` on `set` (see [`Trainer::evaluate`]).
pub fn evaluate_set(learner: &dyn Learner, set: &EvalSet, cfg: &TrainConfig, pool: &rayon::ThreadPool) -> EvalRecord {
    EvalRecord {
        set: set.name.clone(),
        report: accumulate_set(learner, set, cfg, pool).map_or_else(HeadReport::default, MetricsAccumulator::finish),
    }
}
