//! The configuration of a PPO run (task 8.5b): TOML, every field with a default, unknown fields refused.
//!
//! A run is resumable and **refuses to resume under a changed configuration** ([`record_ppo_config`], the rules of `runner::record_config`
//! and `es::record_es_config`): the fields that change what the run computes are compared, the ones that only say how long or on how many
//! threads to run (`iterations`, `threads`, `max_hours`, `run_dir`) are not.

use std::path::PathBuf;

use ddai_fly::bc::LossConfig;
use serde::{Deserialize, Serialize};

use super::curriculum::CurriculumConfig;
use super::reward::PpoReward;
use crate::es::{EvalConfig, ModelStub};
use crate::teacher_data::TeacherDataConfig;
use crate::trainer::RunDir;

fn d_threads() -> usize {
    3
}
fn d_window() -> i32 {
    crate::heldblock::WINDOW_TICKS
}
fn d_burn_ticks() -> i32 {
    crate::bank::DEFAULT_BURN_IN_TICKS
}

/// Where the episodes of an iteration come from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RolloutConfig {
    /// Post-freeze episodes per iteration (starts from the bank's training part on the training halls).
    pub post_episodes: usize,
    /// Weights of the three classes of post-freeze starts (the tags of the 8.5a review): `V` the victim escapes under an idle blocker (the
    /// held-block task proper), `B` the idle blocker itself goes out (the fly's own hazard), `H` an idle blocker holds (geometry).
    pub start_mix: [f32; 3],
    /// Use only the first this many starts of every class (`0` = all): an overfitting probe, not a training setting.
    pub start_subset: usize,
    /// Full games from the spawn per iteration (the sparse first-freeze reward), on the training halls in turn.
    pub game_episodes: usize,
    /// The victim after the handover / the opponent of a game: `(spec, weight)`; a spec is `scripted`, `planner`, `fly:<bundle>` or
    /// `past` (a snapshot of this run's own earlier policies: the league; the scripted bot while there is none).
    pub opponents: Vec<(String, f32)>,
    /// Concentration of the von Mises the aim is sampled from.
    pub aim_kappa: f32,
    /// Temperature of the sampled policy (`policy_logits`): the BC fly's heads are soft, the policy it plays is sharper.
    pub temperature: f32,
    /// The hook head's own temperature (`0` = the same as `temperature`): a hook is held over many decisions, which a coin-flip head cannot do.
    pub temperature_hook: f32,
    pub window_ticks: i32,
    pub burn_in_ticks: i32,
}

impl RolloutConfig {
    pub fn temperatures(&self) -> ddai_fly::policy::Temperatures {
        let mut t = ddai_fly::policy::Temperatures::uniform(self.temperature);
        if self.temperature_hook > 0.0 {
            t.hook = self.temperature_hook;
        }
        t
    }
}

impl Default for RolloutConfig {
    fn default() -> Self {
        RolloutConfig {
            post_episodes: 48,
            start_mix: [0.5, 0.3, 0.2],
            start_subset: 0,
            game_episodes: 12,
            opponents: vec![("scripted".into(), 1.0)],
            aim_kappa: 8.0,
            temperature: 0.3,
            temperature_hook: 0.0,
            window_ticks: d_window(),
            burn_in_ticks: d_burn_ticks(),
        }
    }
}

/// The learner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PpoParams {
    pub gamma: f32,
    pub lambda: f32,
    pub clip: f32,
    /// Passes over the iteration's windows.
    pub epochs: usize,
    /// Windows per mini-batch (one Adam step each).
    pub minibatch_windows: usize,
    /// Decisions scored per BPTT window, and the burn-in before it.
    pub chunk: usize,
    pub burn_in: usize,
    pub entropy_coef: f32,
    /// Multiplies the BC learning rate of every parameter group (`FlyTrainConfig`).
    pub lr_scale: f32,
    /// Extra multiplier of the learning rate of the encoder weights of the opponent-state channels (they start at zero).
    pub lr_new_channel_mult: f32,
    pub grad_clip: f32,
    /// L2 pull of `a` towards the starting strengths (`FlyTrainConfig::l2_a`).
    pub l2_a: f32,
    /// Weight of `KL(start policy || policy)` per decision.
    pub kl_coef: f32,
    /// When positive, `kl_coef` is adapted after every iteration to keep the mean KL near this (nats per decision): up by 1.5 above
    /// `1.5 x`, down by 1.5 below `target / 1.5`, within `[kl_coef_min, kl_coef_max]`.
    pub kl_target: f32,
    pub kl_coef_min: f32,
    pub kl_coef_max: f32,
    /// Stops the epochs of an iteration once the mean `KL(old || new)` of an epoch exceeds `1.5 x` this (`0` = never).
    pub target_kl: f32,
    /// Weight of the behaviour-cloning term on teacher windows (`0` = none) and the number of teacher windows added to every mini-batch.
    pub bc_coef: f32,
    pub bc_windows: usize,
    /// Iterations at the start in which only the critic learns (the policy is not updated).
    pub critic_warmup_iters: u64,
    pub critic_hidden: usize,
    pub critic_lr: f32,
    pub critic_epochs: usize,
    pub critic_batch: usize,
    pub critic_grad_clip: f32,
    /// Windows per forward-only batch of the reference and the old policy pass.
    pub forward_chunk: usize,
    /// Cap (MiB) of the batched engine's working set (`0` = none).
    pub memory_cap_mb: usize,
}

impl Default for PpoParams {
    fn default() -> Self {
        PpoParams {
            gamma: 0.995,
            lambda: 0.95,
            clip: 0.2,
            epochs: 2,
            minibatch_windows: 48,
            chunk: 32,
            burn_in: 8,
            entropy_coef: 0.003,
            lr_scale: 0.2,
            lr_new_channel_mult: 4.0,
            grad_clip: 1.0,
            l2_a: 1e-4,
            kl_coef: 0.5,
            kl_target: 0.02,
            kl_coef_min: 0.05,
            kl_coef_max: 20.0,
            target_kl: 0.05,
            bc_coef: 0.5,
            bc_windows: 12,
            critic_warmup_iters: 3,
            critic_hidden: 128,
            critic_lr: 1e-3,
            critic_epochs: 4,
            critic_batch: 256,
            critic_grad_clip: 5.0,
            forward_chunk: 64,
            memory_cap_mb: 3072,
        }
    }
}

/// The imitation term: teacher datasets (`ddnet-ai train es collect` / the E-008 rounds) of planner labels.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BcAux {
    pub teacher_dirs: Vec<String>,
    pub teacher_data: TeacherDataConfig,
    pub loss: LossConfig,
    /// The hook head of the teacher windows is scored on the masked view (as the bundle was trained) when the bundle is two-view.
    pub seed: u64,
}

/// DAgger-style relabelling for the BC term: every `every` iterations the current fly plays `starts` post-freeze starts (deterministically, the
/// way the bot plays) and the planner labels every state it visits (`ddnet-ai train es collect --actor fly:...`, in the loop); the labelled
/// episodes join the BC corpus. The imitation of the planner's *own* trajectories (E-022's arm 3) teaches states the fly never visits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DaggerConfig {
    /// `0` = never.
    pub every: u64,
    pub starts: usize,
    /// Weights of the start classes (V, B, H) the labelled starts are drawn from.
    pub mix: [f32; 3],
    /// Sampling weight of the labelled steps in the BC corpus (round [`DAGGER_ROUND`]).
    pub round_weight: f32,
}

/// The teacher-store round number of the labelled episodes of the loop.
pub const DAGGER_ROUND: u32 = 100;

impl Default for DaggerConfig {
    fn default() -> Self {
        DaggerConfig {
            every: 0,
            starts: 48,
            mix: [0.6, 0.4, 0.0],
            round_weight: 6.0,
        }
    }
}

/// The periodic evaluation (fixed seeds), with `deny_unknown_fields` of its own.
pub type PpoEval = EvalConfig;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PpoConfig {
    pub name: String,
    #[serde(default)]
    pub model: ModelStub,
    pub flyg: String,
    /// The checkpoint to start from (upgraded with the opponent channels: `train upgrade-bundle`).
    pub init_bundle: String,
    pub arenas_dir: String,
    pub map_dir: String,
    pub run_dir: String,
    pub bank: String,
    /// Technique scenarios (`configs/scenarios`), needed when a teacher dataset of the BC term has scenario episodes.
    #[serde(default)]
    pub scenarios_dir: Option<String>,
    #[serde(default)]
    pub seed: u64,
    pub iterations: u64,
    /// The halls the episodes are played on (training-tagged only).
    pub train_arenas: Vec<String>,
    #[serde(default)]
    pub rollout: RolloutConfig,
    #[serde(default)]
    pub ppo: PpoParams,
    #[serde(default)]
    pub reward: PpoReward,
    #[serde(default)]
    pub bc: BcAux,
    #[serde(default)]
    pub curriculum: CurriculumConfig,
    #[serde(default)]
    pub dagger: DaggerConfig,
    #[serde(default)]
    pub eval: PpoEval,
    /// A snapshot of the policy is added to the league every this many iterations (`0` = never).
    #[serde(default)]
    pub snapshot_every: u64,
    #[serde(default = "d_threads")]
    pub threads: usize,
    /// Stop (cleanly, resumable) after the iteration that crosses this wall time; `0` = no limit.
    #[serde(default)]
    pub max_hours: f32,
}

impl PpoConfig {
    pub fn parse(text: &str) -> Result<PpoConfig, String> {
        let c: PpoConfig = toml::from_str(text).map_err(|e| e.to_string())?;
        c.validate()?;
        Ok(c)
    }

    pub fn validate(&self) -> Result<(), String> {
        let (r, p) = (&self.rollout, &self.ppo);
        if self.iterations == 0 {
            return Err("iterations must be positive".into());
        }
        if self.train_arenas.is_empty() {
            return Err("train_arenas is empty".into());
        }
        if self.threads == 0 || self.threads > 3 {
            return Err("threads must be 1..=3 on the shared machine".into());
        }
        if r.post_episodes + r.game_episodes == 0 {
            return Err("an iteration needs at least one episode".into());
        }
        if r.start_mix.iter().any(|w| !(*w >= 0.0 && w.is_finite())) || r.start_mix.iter().all(|w| *w == 0.0) {
            return Err("rollout.start_mix needs non-negative weights, one positive at least".into());
        }
        if !(r.temperature_hook == 0.0 || (r.temperature_hook > 0.0 && r.temperature_hook <= 1.0)) {
            return Err("rollout.temperature_hook must be 0 or in (0, 1]".into());
        }
        if !(r.temperature > 0.0 && r.temperature <= 1.0) {
            return Err("rollout.temperature must be in (0, 1]".into());
        }
        if !(r.aim_kappa >= 1.0 && r.aim_kappa.is_finite()) {
            return Err("rollout.aim_kappa must be >= 1".into());
        }
        if r.opponents.is_empty()
            || r.opponents.iter().any(|(_, w)| !(*w >= 0.0 && w.is_finite()))
            || r.opponents.iter().all(|(_, w)| *w == 0.0)
        {
            return Err("rollout.opponents needs at least one opponent with a positive weight".into());
        }
        if !(p.gamma > 0.0 && p.gamma <= 1.0 && p.lambda >= 0.0 && p.lambda <= 1.0) {
            return Err("ppo.gamma / ppo.lambda out of range".into());
        }
        if !(p.clip > 0.0 && p.clip < 1.0) {
            return Err("ppo.clip must be in (0, 1)".into());
        }
        if p.epochs == 0 || p.minibatch_windows == 0 || p.chunk == 0 || p.critic_batch == 0 || p.critic_hidden == 0 {
            return Err("ppo.epochs, minibatch_windows, chunk, critic_batch and critic_hidden must be positive".into());
        }
        if p.burn_in >= p.chunk * 4 {
            return Err("ppo.burn_in is unreasonably long".into());
        }
        if p.kl_coef < 0.0 || p.kl_coef_min < 0.0 || p.kl_coef_max < p.kl_coef_min || p.kl_target < 0.0 {
            return Err("ppo.kl_* out of range".into());
        }
        self.curriculum.validate(self.rollout.window_ticks)?;
        if self.dagger.every > 0 && !(p.bc_coef > 0.0 && p.bc_windows > 0) {
            return Err("dagger.every > 0 feeds the BC term: it needs ppo.bc_coef > 0".into());
        }
        if self.dagger.mix.iter().any(|w| !(*w >= 0.0 && w.is_finite())) || self.dagger.mix.iter().all(|w| *w == 0.0) {
            return Err("dagger.mix needs non-negative weights, one positive at least".into());
        }
        if p.bc_coef > 0.0 && p.bc_windows > 0 && self.bc.teacher_dirs.is_empty() && self.dagger.every == 0 {
            return Err("ppo.bc_coef > 0 needs bc.teacher_dirs (or set bc_coef = 0)".into());
        }
        for (spec, _) in &r.opponents {
            if !(matches!(spec.as_str(), "scripted" | "planner" | "past") || spec.starts_with("fly:")) {
                return Err(format!(
                    "unknown opponent {spec:?} (scripted, planner, past or fly:<bundle>)"
                ));
            }
        }
        Ok(())
    }

    pub fn run_path(&self) -> PathBuf {
        crate::experiment::expand_home(&self.run_dir)
    }
}

/// Fields that may change between a kill and the resume without changing what the run computes.
fn normalised_for_resume(cfg: &PpoConfig) -> PpoConfig {
    let mut c = cfg.clone();
    c.threads = 0;
    c.max_hours = 0.0;
    c.run_dir = String::new();
    c.iterations = 0;
    c
}

/// `config.toml` holds the config of the current (re)start, every earlier different one is kept as `config-before-<n>.toml`, and resuming a
/// started run (one that has a `state.bin`) under a config that differs in anything that matters is refused unless `allow_change`.
pub fn record_ppo_config(
    run: &RunDir,
    cfg: &PpoConfig,
    allow_change: bool,
    log: &mut dyn FnMut(&str),
) -> Result<(), String> {
    let path = run.path("config.toml");
    let text = toml::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    if path.exists() {
        let old_text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        if old_text == text {
            return Ok(());
        }
        let started = run.path("state.bin").exists();
        match toml::from_str::<PpoConfig>(&old_text) {
            Ok(old) if started && normalised_for_resume(&old) != normalised_for_resume(cfg) => {
                let (a, b) = (
                    toml::Value::try_from(normalised_for_resume(&old)).map_err(|e| e.to_string())?,
                    toml::Value::try_from(normalised_for_resume(cfg)).map_err(|e| e.to_string())?,
                );
                let mut sections: Vec<String> = Vec::new();
                if let (Some(a), Some(b)) = (a.as_table(), b.as_table()) {
                    for k in a.keys().chain(b.keys()) {
                        if a.get(k) != b.get(k) && !sections.contains(k) {
                            sections.push(k.clone());
                        }
                    }
                }
                if !allow_change {
                    return Err(format!(
                        "refusing to resume {}: the configuration differs from the one this run started with in {sections:?} \
                         (re-run with --allow-config-change to accept it; the old config is kept)",
                        run.path("").display()
                    ));
                }
                log(&format!(
                    "WARNING: resuming with a changed configuration ({sections:?})"
                ));
            }
            Ok(_) => {}
            Err(_) if started && !allow_change => {
                return Err(format!(
                    "refusing to resume {}: its config.toml cannot be parsed to compare (re-run with --allow-config-change)",
                    run.path("").display()
                ));
            }
            Err(_) => {}
        }
        let mut n = 1;
        while run.path(&format!("config-before-{n}.toml")).exists() {
            n += 1;
        }
        std::fs::rename(&path, run.path(&format!("config-before-{n}.toml"))).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, text).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
name = "t"
flyg = "f"
init_bundle = "b"
arenas_dir = "a"
map_dir = "m"
run_dir = "r"
bank = "bank"
iterations = 5
train_arenas = ["pit"]
[ppo]
bc_coef = 0.0
"#;

    #[test]
    fn a_minimal_config_parses_with_defaults_and_round_trips() {
        let c = PpoConfig::parse(MINIMAL).unwrap();
        assert_eq!(c.ppo.gamma, 0.995);
        assert_eq!(c.rollout.post_episodes, 48);
        let again = PpoConfig::parse(&toml::to_string_pretty(&c).unwrap()).unwrap();
        assert_eq!(c, again);
    }

    #[test]
    fn bad_configs_are_refused() {
        for (extra, what) in [
            ("[ppo]\nclip = 1.5\n", "clip"),
            ("[rollout]\nopponents = [[\"nobody\", 1.0]]\n", "opponent"),
            ("[rollout]\nstart_mix = [0.0, 0.0, 0.0]\n", "mix"),
            ("[rollout]\naim_kappa = 0.1\n", "kappa"),
            ("[ppo]\nbogus = 1\n", "unknown field"),
        ] {
            let text = MINIMAL.replace("[ppo]\nbc_coef = 0.0\n", "") + extra;
            assert!(PpoConfig::parse(&text).is_err(), "{what} should be refused");
        }
        // bc_coef > 0 without data is refused.
        assert!(PpoConfig::parse(&MINIMAL.replace("bc_coef = 0.0", "bc_coef = 0.5")).is_err());
    }

    #[test]
    fn resuming_a_started_run_under_a_changed_config_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let run = RunDir::create(dir.path()).unwrap();
        let a = PpoConfig::parse(MINIMAL).unwrap();
        record_ppo_config(&run, &a, false, &mut |_| {}).unwrap();
        // Not started yet (no state.bin): any change is fine, the old one is kept.
        let mut b = a.clone();
        b.ppo.clip = 0.1;
        record_ppo_config(&run, &b, false, &mut |_| {}).unwrap();
        assert!(run.path("config-before-1.toml").exists());
        std::fs::write(run.path("state.bin"), b"x").unwrap();
        // Started: a different clip is refused, a longer run, other threads and the wall limit are not.
        let mut c = b.clone();
        c.ppo.clip = 0.3;
        let e = record_ppo_config(&run, &c, false, &mut |_| {}).unwrap_err();
        assert!(e.contains("refusing to resume") && e.contains("ppo"), "{e}");
        let mut d = b.clone();
        (d.iterations, d.threads, d.max_hours) = (50, 1, 2.0);
        record_ppo_config(&run, &d, false, &mut |_| {}).unwrap();
        // With the flag it goes through.
        record_ppo_config(&run, &c, true, &mut |_| {}).unwrap();
    }
}
