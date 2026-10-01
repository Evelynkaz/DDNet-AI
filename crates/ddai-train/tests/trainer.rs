//! The trainer on a synthetic, learnable task: the loss falls and the direction head learns,
//! a step is reproducible at any thread count, and a run resumed from `state.bin` ends exactly
//! where an uninterrupted one does.

use std::sync::Arc;

use ddai_brain::CharacterObservation;
use ddai_controls::bundle::load_control_bundle;
use ddai_controls::features::input_dim;
use ddai_controls::mlp::Mlp;
use ddai_dataset::types::ActionRec;
use ddai_fly::bc::HeadMask;
use ddai_fly::bc::HeadThresholds;
use ddai_fly::encoder::RayGridConfig;
use ddai_fly::rng::SplitMix64;
use ddai_physics::map::{MapData, TILE_SOLID, Tile};
use ddai_train::learner::{ControlLearner, Learner, WindowStats, Workspace};
use ddai_train::seq::{Corpus, MapEntry, Seq, SeqStep, Source};
use ddai_train::trainer::{EvalSet, RunDir, TrainConfig, Trainer};
use ddai_train::types::char_rec;

fn map() -> Arc<MapEntry> {
    let (w, h) = (30usize, 20usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            if y >= 14 || x == 0 || x == w - 1 || y == 0 {
                game[y * w + x] = Tile {
                    index: TILE_SOLID,
                    ..Tile::default()
                };
            }
        }
    }
    MapEntry::new(Arc::new(MapData {
        width: w as u32,
        height: h as u32,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }))
}

/// The teacher's rule: run towards the opponent, hook when it is close.
fn corpus(seed: u64, n_seqs: usize) -> Corpus {
    let m = map();
    let mut rng = SplitMix64::new(seed);
    let seqs = (0..n_seqs)
        .map(|_| {
            let steps = (0..40)
                .map(|t| {
                    let x = 100.0 + rng.next_f32_unit() * 600.0;
                    let dx = (rng.next_f32_unit() - 0.5) * 500.0;
                    let mut me = CharacterObservation::at_rest(0);
                    me.pos = ddai_physics::vmath::Vec2::new(x, 400.0);
                    me.grounded = true;
                    let mut opp = CharacterObservation::at_rest(1);
                    opp.pos = ddai_physics::vmath::Vec2::new(x + dx, 400.0);
                    SeqStep {
                        tick: 2 * t,
                        me: char_rec(&me),
                        others: vec![char_rec(&opp)],
                        target: 1,
                        label: ActionRec {
                            direction: if dx > 0.0 { 1 } else { -1 },
                            jump: false,
                            hook: dx.abs() < 120.0,
                            fire: false,
                            aim: [if dx > 0.0 { 100 } else { -100 }, 0],
                        },
                        soft: None,
                        weight: 1.0,
                        mask: HeadMask::ALL,
                    }
                })
                .collect();
            Seq {
                map: m.clone(),
                steps,
                source: Source::Human { demo: 0 },
            }
        })
        .collect();
    Corpus::new(seqs)
}

fn learner() -> Box<dyn Learner> {
    let cfg = RayGridConfig::default();
    Box::new(ControlLearner::new(
        Box::new(Mlp::new(input_dim(&cfg), 8, 3)),
        cfg,
        5e-3,
    ))
}

fn cfg(threads: usize) -> TrainConfig {
    TrainConfig {
        batch_windows: 8,
        window_len: 16,
        burn_in: 2,
        warmup_steps: 5,
        threads,
        log_every: 5,
        eval_windows: 40,
        human_fraction: 0.0,
        ..TrainConfig::default()
    }
}

fn eval_set() -> EvalSet {
    EvalSet {
        name: "val".into(),
        corpus: Corpus::uniform({
            let c = corpus(99, 10);
            c.seqs
        }),
    }
}

#[test]
fn the_direction_head_learns_a_synthetic_rule_and_metrics_are_logged() {
    let dir = tempfile::tempdir().unwrap();
    let run = RunDir::create(dir.path()).unwrap();
    let mut t = Trainer::new(
        learner(),
        cfg(2),
        corpus(1, 30),
        Corpus::new(Vec::new()),
        Some(run.clone()),
    )
    .unwrap();
    let sets = [eval_set()];
    let before = t.evaluate(&sets)[0].report.dir.accuracy;
    let s = t.train_phase("bc", 0, 120, 120, &sets, 60).unwrap();
    assert_eq!((s.steps, s.end_step, s.skipped_steps), (120, 120, 0));
    let after = t.evaluate(&sets)[0].report.clone();
    assert!(
        after.dir.accuracy > 0.9 && after.dir.accuracy > before,
        "{before} -> {}",
        after.dir.accuracy
    );
    let metrics = std::fs::read_to_string(run.path("metrics.jsonl")).unwrap();
    assert!(metrics.lines().any(|l| l.contains("\"kind\":\"train\"")));
    assert!(metrics.lines().any(|l| l.contains("\"kind\":\"eval\"")));
    assert!(run.path("status.json").exists() && run.path("state.bin").exists());
    assert!(run.checkpoint("last.bundle").exists() && run.checkpoint("step-00000120.bundle").exists());
}

#[test]
fn a_step_does_not_depend_on_the_thread_count() {
    let train = |threads| {
        let mut t = Trainer::new(learner(), cfg(threads), corpus(1, 20), Corpus::new(Vec::new()), None).unwrap();
        t.train_phase("bc", 0, 12, 12, &[], 0).unwrap();
        t.learner().params()
    };
    assert_eq!(
        train(1),
        train(3),
        "summing gradients in window order makes threads irrelevant"
    );
}

#[test]
fn a_resumed_run_ends_exactly_where_an_uninterrupted_one_does() {
    let straight = {
        let mut t = Trainer::new(learner(), cfg(2), corpus(1, 20), Corpus::new(Vec::new()), None).unwrap();
        t.train_phase("bc", 0, 20, 20, &[], 0).unwrap();
        t.learner().params()
    };
    let dir = tempfile::tempdir().unwrap();
    let run = RunDir::create(dir.path()).unwrap();
    {
        let mut t = Trainer::new(
            learner(),
            cfg(2),
            corpus(1, 20),
            Corpus::new(Vec::new()),
            Some(run.clone()),
        )
        .unwrap();
        t.train_phase("bc", 0, 20, 8, &[], 0).unwrap(); // stops after 8 of 20 steps
        assert_eq!(t.step, 8);
    }
    let mut t = Trainer::new(learner(), cfg(2), corpus(1, 20), Corpus::new(Vec::new()), Some(run)).unwrap();
    assert!(t.resume().unwrap());
    assert_eq!(t.step, 8);
    t.train_phase("bc", 0, 20, 12, &[], 0).unwrap();
    assert_eq!(t.learner().params(), straight);
}

fn named_set(name: &str, seed: u64) -> EvalSet {
    EvalSet {
        name: name.into(),
        corpus: Corpus::uniform(corpus(seed, 10).seqs),
    }
}

#[test]
fn a_phase_calibrates_thresholds_on_the_teacher_validation_sets_and_the_bundle_stores_them() {
    // A hook head trained with a heavy positive-class weight presses far more often than the labels
    // at 0.5; the phase-end calibration brings its press rate back to the label rate.
    let make = || {
        let dir = tempfile::tempdir().unwrap();
        let run = RunDir::create(dir.path()).unwrap();
        let mut c = cfg(2);
        c.loss.pos_weight = [1.0, 8.0, 1.0];
        let t = Trainer::new(learner(), c, corpus(1, 30), Corpus::new(Vec::new()), Some(run.clone())).unwrap();
        (dir, run, t)
    };
    let (_d, run, mut t) = make();
    let sets = [named_set("teacher-val", 99), named_set("human-val", 7)];
    t.train_phase("bc", 0, 120, 120, &sets, 0).unwrap();
    let th = t.learner().thresholds();
    assert!(th.validate().is_ok());
    assert!(
        th.hook > 0.5,
        "an 8x-weighted head needs a higher threshold: {}",
        th.hook
    );
    let r = &t.evaluate(&sets)[0].report;
    assert!(r.hook.prevalence > 0.0 && r.hook.prevalence < 1.0);
    assert!(
        (r.hook.pred_rate - r.hook.prevalence).abs() < 0.02,
        "presses {} vs labels {}",
        r.hook.pred_rate,
        r.hook.prevalence
    );
    assert_eq!(r.hook.threshold, f64::from(th.hook));
    // The bundles written at the end of the phase carry them; the log records the choice.
    for name in ["last.bundle", "step-00000120.bundle"] {
        let b = load_control_bundle(&run.checkpoint(name)).unwrap();
        assert_eq!(b.thresholds, th, "{name}");
    }
    let metrics = std::fs::read_to_string(run.path("metrics.jsonl")).unwrap();
    assert!(metrics.lines().any(|l| l.contains("\"kind\":\"thresholds\"")));

    // Without a teacher validation set nothing is calibrated, whatever else is evaluated.
    let (_d2, run2, mut t2) = make();
    t2.train_phase("bc", 0, 20, 20, &[named_set("human-val", 7)], 0)
        .unwrap();
    assert_eq!(t2.learner().thresholds(), HeadThresholds::default());
    assert_eq!(
        load_control_bundle(&run2.checkpoint("last.bundle")).unwrap().thresholds,
        HeadThresholds::default()
    );
}

/// A learner whose every batch has nothing to score.
struct Unscored(Box<dyn Learner>);

impl Learner for Unscored {
    fn label(&self) -> String {
        self.0.label()
    }
    fn num_params(&self) -> usize {
        self.0.num_params()
    }
    fn params(&self) -> Vec<f32> {
        self.0.params()
    }
    fn set_params(&mut self, flat: &[f32]) -> Result<(), String> {
        self.0.set_params(flat)
    }
    fn base_lrs(&self) -> Vec<f32> {
        self.0.base_lrs()
    }
    fn new_workspace(&self, max_window: usize) -> Workspace {
        self.0.new_workspace(max_window)
    }
    fn window_grad(
        &self,
        _window: &ddai_train::seq::Window,
        _loss: &ddai_fly::bc::LossConfig,
        _ws: &mut Workspace,
        _grad: &mut [f32],
    ) -> WindowStats {
        WindowStats::default()
    }
    fn window_logits(&self, window: &ddai_train::seq::Window) -> Vec<ddai_fly::bc::HeadLogits> {
        self.0.window_logits(window)
    }
    fn mirror_augment(&self) -> bool {
        self.0.mirror_augment()
    }
    fn thresholds(&self) -> HeadThresholds {
        self.0.thresholds()
    }
    fn set_thresholds(&mut self, t: HeadThresholds) {
        self.0.set_thresholds(t);
    }
    fn save(&self, path: &std::path::Path, meta: ddai_fly::bundle::BundleMeta) -> Result<(), String> {
        self.0.save(path, meta)
    }
}

#[test]
fn a_batch_with_nothing_to_score_still_advances_the_global_step() {
    // The schedule, the batch seeds and the resume point are keyed by `step`; a skipped step that did
    // not count would leave a phase short of its end and make a resumed run redo it.
    let mut t = Trainer::new(
        Box::new(Unscored(learner())),
        cfg(2),
        corpus(1, 10),
        Corpus::new(Vec::new()),
        None,
    )
    .unwrap();
    let before = t.learner().params();
    let s = t.train_phase("bc", 0, 6, 6, &[], 0).unwrap();
    assert_eq!((s.skipped_steps, s.end_step, t.step), (6, 6, 6));
    assert_eq!(t.learner().params(), before, "no update without a gradient");
}
