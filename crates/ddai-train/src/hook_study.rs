//! The hook-head investigation (task 8.2, acceptance criterion 3): why was the hook head near
//! chance in 7.3, and what fixes it?
//!
//! 7.3's demo trained the fly on synthetic scenes with a scripted teacher whose hook label is
//! "the opponent is within 380 px with a clear line of sight" (`ddai_fly::demo_brain`), and the
//! hook head stayed at balanced accuracy 0.57-0.60 on the natural scene distribution. Three
//! suspects were named: the input features, class imbalance, the decoder calibration. This module
//! reruns that task through the crate's trainer with one factor changed per arm:
//!
//! * **features**: the 7.3 encoder sums a ray's four distance bins with equal weight, and the
//!   Gaussian bins partition distance (their sum is nearly constant), so "within 380 px" is
//!   invisible to it. Arm `dist-gains` learns a gain per bin ([`ddai_fly::encoder::RayGridConfig`]'s
//!   `learn_distance_gains`);
//! * **calibration**: resting-state calibration vs calibration from real scenes;
//! * **budget**: the demo's 200 tiny steps vs a longer run;
//! * **imbalance**: the teacher's hook prevalence on the evaluation scenes is reported (it is
//!   ~0.6-0.7, so the head is not starved of positives), and a positive-weighted arm is included.
//!
//! An MLP on the same features is trained alongside as the information ceiling of the features.
//! Everything is evaluated on held-out scenes of the training map and of a different map.

use std::path::Path;
use std::sync::Arc;

use ddai_controls::features::input_dim;
use ddai_controls::mlp::Mlp;
use ddai_dataset::types::ActionRec;
use ddai_fly::bc::{HeadMask, LossConfig};
use ddai_fly::brain_config::parse_brain_config;
use ddai_fly::demo_brain::{BrainDemoConfig, sample_scenario, scripted_teacher};
use ddai_fly::rng::SplitMix64;
use serde::Serialize;

use crate::learner::{ControlLearner, FlyLearner, FlyTrainConfig, Learner};
use crate::metrics::HeadReport;
use crate::seq::{Corpus, MapEntry, Seq, SeqStep, Source};
use crate::trainer::{EvalSet, TrainConfig, Trainer};
use crate::types::char_rec;

/// One scene repeated four times (the demo's `t_decisions = 4` static window); only the last
/// repetition is scored.
fn scenes(map: &Arc<MapEntry>, n: usize, seed: u64, cfg: &BrainDemoConfig) -> Vec<Seq> {
    let mut rng = SplitMix64::new(seed);
    (0..n)
        .map(|_| {
            let obs = sample_scenario(map.map.clone(), cfg, &mut rng);
            let t = scripted_teacher(&obs, cfg);
            let step = |weight: f32| SeqStep {
                tick: 0,
                me: char_rec(&obs.self_state),
                others: obs.others.iter().map(char_rec).collect(),
                target: -1,
                label: ActionRec {
                    direction: t.direction as i8 - 1,
                    jump: t.jump,
                    hook: t.hook,
                    fire: false,
                    aim: [0, -1],
                },
                soft: None,
                weight,
                mask: HeadMask {
                    fire: false,
                    aim: false,
                    ..HeadMask::ALL
                },
                latch: false,
            };
            Seq {
                map: map.clone(),
                steps: vec![step(0.0), step(0.0), step(0.0), step(1.0)],
                source: Source::Human { demo: 0 },
            }
        })
        .collect()
}

fn load_map_file(path: &Path) -> Result<Arc<MapEntry>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let loaded = ddai_map::load_map(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(MapEntry::new(Arc::new(loaded.data)))
}

#[derive(Debug, Clone, Serialize)]
pub struct ArmResult {
    pub arm: String,
    pub params: usize,
    pub steps: u64,
    /// Held-out scenes of the training map / of another map.
    pub same_map: HeadReport,
    pub other_map: HeadReport,
}

#[derive(Debug, Clone)]
pub struct HookStudyConfig {
    pub flyg: std::path::PathBuf,
    pub brain_config: std::path::PathBuf,
    pub brain_config_dist_gains: std::path::PathBuf,
    /// Distance gains with eight distance bins instead of four.
    pub brain_config_dist_gains_8: std::path::PathBuf,
    pub train_map: std::path::PathBuf,
    pub other_map: std::path::PathBuf,
    pub threads: usize,
    pub demo_steps: u64,
    pub long_steps: u64,
    pub seed: u64,
    /// Run the 2x2x2 factorial of the three changes 8.2 made together (distance-bin gains, 6x connectome learning
    /// rate, `alpha_init` 4) instead of the E-005 arm list, so each factor's effect can be separated (E-005 review
    /// F6). Every cell is one seed; the caller repeats the study over seeds.
    pub factorial: bool,
}

struct Arm {
    name: &'static str,
    dist_gains: bool,
    eight_bins: bool,
    /// The demo's constant, small connectome learning rate, or this task's schedule.
    demo_optimiser: bool,
    steps_long: bool,
    real_calibration: bool,
    hook_pos_weight: f32,
    /// Connectome learning rates (`a`, `b`, `theta`) of the schedule arms; `None` = defaults.
    fly_lr: Option<f32>,
    alpha_init: f32,
}

/// The 2x2x2 factorial: `gains` (learned distance-bin gains), `lr6` (connectome learning rate 6x), `a4`
/// (`alpha_init` 4; otherwise the default coupling). All cells use the long budget, the schedule optimiser and
/// resting-state calibration, like the E-005 accepted arm.
fn factorial_arms() -> Vec<Arm> {
    let cell = |name: &'static str, gains: bool, lr6: bool, a4: bool| Arm {
        name,
        dist_gains: gains,
        eight_bins: false,
        demo_optimiser: false,
        steps_long: true,
        real_calibration: false,
        hook_pos_weight: 1.0,
        fly_lr: lr6.then_some(3e-3),
        alpha_init: if a4 { 4.0 } else { 0.0 },
    };
    vec![
        cell("f: no gains, lr 1x, alpha default", false, false, false),
        cell("f: gains, lr 1x, alpha default", true, false, false),
        cell("f: no gains, lr 6x, alpha default", false, true, false),
        cell("f: gains, lr 6x, alpha default", true, true, false),
        cell("f: no gains, lr 1x, alpha 4", false, false, true),
        cell("f: gains, lr 1x, alpha 4", true, false, true),
        cell("f: no gains, lr 6x, alpha 4", false, true, true),
        cell("f: gains, lr 6x, alpha 4", true, true, true),
    ]
}

/// Runs every arm; `log` gets one line per finished arm.
pub fn run_hook_study(cfg: &HookStudyConfig, log: &mut dyn FnMut(&str)) -> Result<Vec<ArmResult>, String> {
    let demo_cfg = BrainDemoConfig::default();
    let train_map = load_map_file(&cfg.train_map)?;
    let other_map = load_map_file(&cfg.other_map)?;
    let train = Corpus::new(scenes(&train_map, 4000, cfg.seed, &demo_cfg));
    let sets = vec![
        EvalSet {
            name: "same-map".into(),
            corpus: Corpus::uniform(scenes(&train_map, 600, cfg.seed + 1, &demo_cfg)),
        },
        EvalSet {
            name: "other-map".into(),
            corpus: Corpus::uniform(scenes(&other_map, 600, cfg.seed + 2, &demo_cfg)),
        },
    ];
    let factorial_arms = factorial_arms();
    let legacy_arms = [
        Arm {
            name: "7.3 as built (demo budget)",
            dist_gains: false,
            eight_bins: false,
            demo_optimiser: true,
            steps_long: false,
            real_calibration: false,
            hook_pos_weight: 1.0,
            fly_lr: None,
            alpha_init: 0.0,
        },
        Arm {
            name: "+ distance gains (demo budget)",
            dist_gains: true,
            eight_bins: false,
            demo_optimiser: true,
            steps_long: false,
            real_calibration: false,
            hook_pos_weight: 1.0,
            fly_lr: None,
            alpha_init: 0.0,
        },
        Arm {
            name: "7.3 encoder, longer training",
            dist_gains: false,
            eight_bins: false,
            demo_optimiser: false,
            steps_long: true,
            real_calibration: false,
            hook_pos_weight: 1.0,
            fly_lr: None,
            alpha_init: 0.0,
        },
        Arm {
            name: "7.3 encoder, longer, real-scene calibration",
            dist_gains: false,
            eight_bins: false,
            demo_optimiser: false,
            steps_long: true,
            real_calibration: true,
            hook_pos_weight: 1.0,
            fly_lr: None,
            alpha_init: 0.0,
        },
        Arm {
            name: "7.3 encoder, longer, positive-weighted hook",
            dist_gains: false,
            eight_bins: false,
            demo_optimiser: false,
            steps_long: true,
            real_calibration: false,
            hook_pos_weight: 2.0,
            fly_lr: None,
            alpha_init: 0.0,
        },
        Arm {
            name: "+ distance gains, longer",
            dist_gains: true,
            eight_bins: false,
            demo_optimiser: false,
            steps_long: true,
            real_calibration: false,
            hook_pos_weight: 1.0,
            fly_lr: None,
            alpha_init: 0.0,
        },
        Arm {
            name: "+ distance gains, longer, real-scene calibration",
            dist_gains: true,
            eight_bins: false,
            demo_optimiser: false,
            steps_long: true,
            real_calibration: true,
            hook_pos_weight: 1.0,
            fly_lr: None,
            alpha_init: 0.0,
        },
        Arm {
            name: "+ distance gains, longer, 6x connectome lr",
            dist_gains: true,
            eight_bins: false,
            demo_optimiser: false,
            steps_long: true,
            real_calibration: false,
            hook_pos_weight: 1.0,
            fly_lr: Some(3e-3),
            alpha_init: 0.0,
        },
        Arm {
            name: "+ distance gains, longer, 6x connectome lr, alpha 6",
            dist_gains: true,
            eight_bins: false,
            demo_optimiser: false,
            steps_long: true,
            real_calibration: false,
            hook_pos_weight: 1.0,
            fly_lr: Some(3e-3),
            alpha_init: 6.0,
        },
        Arm {
            name: "+ distance gains, longer, 6x connectome lr, alpha 8",
            dist_gains: true,
            eight_bins: false,
            demo_optimiser: false,
            steps_long: true,
            real_calibration: false,
            hook_pos_weight: 1.0,
            fly_lr: Some(3e-3),
            alpha_init: 8.0,
        },
        Arm {
            name: "+ 8 distance bins with gains, 6x connectome lr, alpha 4",
            dist_gains: true,
            eight_bins: true,
            demo_optimiser: false,
            steps_long: true,
            real_calibration: false,
            hook_pos_weight: 1.0,
            fly_lr: Some(3e-3),
            alpha_init: 4.0,
        },
        Arm {
            name: "+ distance gains, longer, 6x connectome lr, alpha 4",
            dist_gains: true,
            eight_bins: false,
            demo_optimiser: false,
            steps_long: true,
            real_calibration: false,
            hook_pos_weight: 1.0,
            fly_lr: Some(3e-3),
            alpha_init: 4.0,
        },
    ];
    let arms: &[Arm] = if cfg.factorial { &factorial_arms } else { &legacy_arms };
    let mut out = Vec::new();
    for arm in arms {
        let bc_path = match (arm.dist_gains, arm.eight_bins) {
            (_, true) => &cfg.brain_config_dist_gains_8,
            (true, false) => &cfg.brain_config_dist_gains,
            (false, false) => &cfg.brain_config,
        };
        let mut fly_cfg = if arm.demo_optimiser {
            FlyTrainConfig {
                lr_a: 2e-4,
                lr_b: 2e-4,
                lr_theta: 2e-4,
                lr_encoder: 2e-2,
                lr_decoder: 2e-2,
                l2_a: 0.0,
                calibration_windows: 0,
                ..FlyTrainConfig::default()
            }
        } else {
            FlyTrainConfig::default()
        };
        if let Some(lr) = arm.fly_lr {
            fly_cfg.lr_a = lr;
            fly_cfg.lr_b = lr * 1.5;
            fly_cfg.lr_theta = lr * 1.5;
        }
        fly_cfg.alpha_init = arm.alpha_init;
        if arm.real_calibration {
            fly_cfg.calibration_windows = 300;
        } else {
            fly_cfg.calibration_windows = 0;
        }
        let windows: Vec<Vec<ddai_brain::Observation>> = if fly_cfg.calibration_windows > 0 {
            let mut rng = SplitMix64::new(cfg.seed ^ 77);
            (0..fly_cfg.calibration_windows)
                .map(|_| train.sample_window(&mut rng, 4, 0, false).observations)
                .collect()
        } else {
            Vec::new()
        };
        let learner: Box<dyn Learner> = Box::new(FlyLearner::init(&cfg.flyg, bc_path, cfg.seed, fly_cfg, &windows)?);
        let steps = if arm.steps_long { cfg.long_steps } else { cfg.demo_steps };
        out.push(run_arm(arm.name, learner, steps, arm, &train, &sets, cfg)?);
        log(&format!(
            "{}: hook auroc {:.3} / {:.3}",
            arm.name,
            out.last().map_or(0.0, |r| r.same_map.hook.auroc),
            out.last().map_or(0.0, |r| r.other_map.hook.auroc)
        ));
    }
    // The information ceiling of the features: a wide MLP.
    let text = std::fs::read_to_string(&cfg.brain_config).map_err(|e| e.to_string())?;
    let bc = parse_brain_config(&text).map_err(|e| e.to_string())?;
    let mlp: Box<dyn Learner> = Box::new(ControlLearner::new(
        Box::new(Mlp::new(input_dim(&bc.ray_grid), 32, cfg.seed)),
        bc.ray_grid,
        3e-3,
    ));
    let arm = Arm {
        name: "MLP h=32 (feature ceiling)",
        dist_gains: false,
        eight_bins: false,
        demo_optimiser: false,
        steps_long: true,
        real_calibration: false,
        hook_pos_weight: 1.0,
        fly_lr: None,
        alpha_init: 0.0,
    };
    out.push(run_arm(arm.name, mlp, cfg.long_steps, &arm, &train, &sets, cfg)?);
    log(&format!(
        "MLP ceiling: hook auroc {:.3}",
        out.last().map_or(0.0, |r| r.same_map.hook.auroc)
    ));
    Ok(out)
}

fn run_arm(
    name: &str,
    learner: Box<dyn Learner>,
    steps: u64,
    arm: &Arm,
    train: &Corpus,
    sets: &[EvalSet],
    cfg: &HookStudyConfig,
) -> Result<ArmResult, String> {
    let params = learner.num_params();
    let train_cfg = TrainConfig {
        seed: cfg.seed,
        batch_windows: 16,
        window_len: 4,
        burn_in: 0,
        human_fraction: 0.0,
        warmup_steps: if arm.demo_optimiser { 0 } else { 50 },
        lr_final_frac: if arm.demo_optimiser { 1.0 } else { 0.1 },
        threads: cfg.threads,
        eval_windows: 600,
        refresh_every: 0,
        loss: LossConfig {
            pos_weight: [1.0, arm.hook_pos_weight, 1.0],
            soft_mix: 0.0,
            ..LossConfig::default()
        },
        ..TrainConfig::default()
    };
    // The trainer takes ownership of its corpus; the scenes are cheap to share by rebuilding.
    let corpus = Corpus::new(
        train
            .seqs
            .iter()
            .map(|s| Seq {
                map: s.map.clone(),
                steps: s.steps.clone(),
                source: s.source.clone(),
            })
            .collect(),
    );
    let mut t = Trainer::new(learner, train_cfg, corpus, Corpus::new(Vec::new()), None).map_err(|e| e.to_string())?;
    t.train_phase("hook-study", 0, steps, steps, &[], 0)
        .map_err(|e| e.to_string())?;
    let recs = t.evaluate(sets);
    Ok(ArmResult {
        arm: name.to_string(),
        params,
        steps,
        same_map: recs[0].report.clone(),
        other_map: recs[1].report.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seq::testutil::room_map;

    #[test]
    fn the_factorial_has_every_combination_of_the_three_changes_exactly_once() {
        let arms = factorial_arms();
        assert_eq!(arms.len(), 8);
        let mut seen = std::collections::BTreeSet::new();
        for a in &arms {
            // Everything but the three factors is held fixed.
            assert!(!a.eight_bins && !a.demo_optimiser && a.steps_long && !a.real_calibration);
            assert_eq!(a.hook_pos_weight, 1.0);
            assert!(a.fly_lr.is_none_or(|lr| (lr - 3e-3).abs() < 1e-9));
            assert!(a.alpha_init == 0.0 || a.alpha_init == 4.0);
            seen.insert((a.dist_gains, a.fly_lr.is_some(), a.alpha_init == 4.0));
        }
        assert_eq!(seen.len(), 8, "2 x 2 x 2 distinct cells");
        let names: std::collections::BTreeSet<&str> = arms.iter().map(|a| a.name).collect();
        assert_eq!(names.len(), 8);
        // The cell E-005 never ran: the 7.3 encoder (no gains) with the other two changes.
        assert!(
            arms.iter()
                .any(|a| !a.dist_gains && a.fly_lr.is_some() && a.alpha_init == 4.0)
        );
    }

    #[test]
    fn scenes_repeat_one_observation_and_score_only_the_last_repetition() {
        let cfg = BrainDemoConfig::default();
        let a = scenes(&room_map(), 12, 5, &cfg);
        let b = scenes(&room_map(), 12, 5, &cfg);
        assert_eq!(a.len(), 12);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.steps.len(), 4);
            assert_eq!(x.steps[0].me, x.steps[3].me, "a static scene repeated");
            assert_eq!(x.steps[3].me, y.steps[3].me, "deterministic in the seed");
            let w: Vec<f32> = x.steps.iter().map(|s| s.weight).collect();
            assert_eq!(w, vec![0.0, 0.0, 0.0, 1.0]);
            assert!(!x.steps[3].mask.fire && !x.steps[3].mask.aim && x.steps[3].mask.hook);
        }
        let hooks = a.iter().filter(|s| s.steps[3].label.hook).count();
        assert!(
            hooks > 0 && hooks < a.len(),
            "the natural distribution has both hook labels ({hooks}/12)"
        );
    }
}
