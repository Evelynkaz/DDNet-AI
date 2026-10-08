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
use ddai_train::trainer::{EvalSet, OwnHookConfig, OwnHookMode, RunDir, TrainConfig, Trainer};
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
                        latch: false,
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

/// Task 8.8: a control that also reads the five opponent-state channels (the equal-parameter controls of the FlyGM pilot must see what the fly's
/// encoder sees) trains through the same trainer, its checkpoint stores the longer input, and the playing template accepts it.
#[test]
fn a_control_that_reads_the_opponent_state_channels_trains_and_its_checkpoint_loads() {
    use ddai_controls::ControlTemplate;
    use ddai_controls::features::{input_dim_with_opponent_state, reads_opponent_state};
    let cfg = RayGridConfig::default();
    let d = input_dim_with_opponent_state(&cfg);
    let dir = tempfile::tempdir().unwrap();
    let run = RunDir::create(dir.path()).unwrap();
    let l: Box<dyn Learner> = Box::new(ControlLearner::new(Box::new(Mlp::new(d, 3, 1)), cfg, 5e-3));
    let mut t = Trainer::new(
        l,
        self::cfg(2),
        corpus(1, 20),
        Corpus::new(Vec::new()),
        Some(run.clone()),
    )
    .unwrap();
    let before = t.learner().params();
    t.train_phase("bc", 0, 20, 20, &[], 0).unwrap();
    assert_ne!(t.learner().params(), before);
    let b = load_control_bundle(&run.checkpoint("last.bundle")).unwrap();
    assert_eq!(b.input_dim, d);
    assert_eq!(reads_opponent_state(&b.ray_grid, b.input_dim), Some(true));
    ControlTemplate::load(&run.checkpoint("last.bundle")).expect("the playing template accepts the longer input");
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
        // Reported on the windows the thresholds were *not* fitted on, so only approximately matched.
        (r.hook.pred_rate - r.hook.prevalence).abs() < 0.08,
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
    fn hook_view(&self) -> ddai_fly::bc::HookView {
        self.0.hook_view()
    }
    fn set_hook_view(&mut self, view: ddai_fly::bc::HookView) {
        self.0.set_hook_view(view);
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

#[test]
fn thresholds_are_fitted_on_one_half_of_the_validation_windows_and_reported_on_the_other() {
    use ddai_train::trainer::{WindowHalf, accumulate_set};
    let c = cfg(2);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    let net = learner();
    let set = named_set("teacher-val", 99);
    let count = |half| {
        accumulate_set(net.as_ref(), &set, &c, &pool, half)
            .unwrap()
            .finish()
            .steps
    };
    let (all, fit, report) = (
        count(WindowHalf::All),
        count(WindowHalf::Fit),
        count(WindowHalf::Report),
    );
    assert!(fit > 0 && report > 0);
    assert_eq!(fit + report, all, "the halves partition the fixed windows");

    // A trained phase: the reported rates come from the odd windows only, so the fitted rate match is no
    // longer true by construction (the thresholds were chosen on other windows).
    let mut t = Trainer::new(learner(), c.clone(), corpus(1, 30), Corpus::new(Vec::new()), None).unwrap();
    let sets = [set];
    t.train_phase("bc", 0, 60, 60, &sets, 0).unwrap();
    let th = t.learner().thresholds();
    let rep = &t.evaluate(&sets)[0].report;
    assert_eq!(
        rep.steps,
        report_steps_of(&t, &sets[0], &c, &pool),
        "evaluation uses the report half"
    );
    assert_eq!(rep.hook.threshold, f64::from(th.hook));
}

fn report_steps_of(t: &Trainer, set: &EvalSet, c: &TrainConfig, pool: &rayon::ThreadPool) -> u64 {
    use ddai_train::trainer::{WindowHalf, accumulate_set};
    accumulate_set(t.learner(), set, c, pool, WindowHalf::Report)
        .unwrap()
        .finish()
        .steps
}

/// Like [`corpus`], but the own hook state is set: out with probability `persistence` when the label presses the
/// hook and with probability `1 - persistence` when it does not (`0.5` = unrelated to the label). A high
/// persistence offers the shortcut "hook <=> my hook is already out" (E-005 review F2).
fn corpus_with_state(seed: u64, n_seqs: usize, persistence: f32) -> Corpus {
    let m = map();
    let mut rng = SplitMix64::new(seed);
    let seqs = (0..n_seqs)
        .map(|_| {
            let steps = (0..40)
                .map(|t| {
                    let x = 100.0 + rng.next_f32_unit() * 600.0;
                    let dx = (rng.next_f32_unit() - 0.5) * 500.0;
                    let hook = dx.abs() < 120.0;
                    let out = if rng.next_f32_unit() < persistence { hook } else { !hook };
                    let mut me = CharacterObservation::at_rest(0);
                    me.pos = ddai_physics::vmath::Vec2::new(x, 400.0);
                    me.grounded = true;
                    me.hook_state = if out {
                        ddai_brain::HOOK_GRABBED
                    } else {
                        ddai_brain::HOOK_IDLE
                    };
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
                            hook,
                            fire: false,
                            aim: [if dx > 0.0 { 100 } else { -100 }, 0],
                        },
                        soft: None,
                        weight: 1.0,
                        mask: HeadMask::ALL,
                        latch: false,
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

/// Trains an MLP on a corpus where the own hook state is a near-perfect shortcut, then measures how much its
/// played hook probability moves when only the own hook state of a decorrelated validation window is flipped
/// from idle to grabbed (`0` = the hook head ignores the state), and its hook AUROC among the not-out decisions.
fn hook_reliance_on_own_state(own_hook: OwnHookConfig) -> (f32, f64) {
    let mut c = cfg(2);
    c.own_hook = own_hook;
    c.window_len = 12;
    c.burn_in = 2;
    c.batch_windows = 12;
    let mut t = Trainer::new(
        learner(),
        c,
        corpus_with_state(1, 40, 0.97),
        Corpus::new(Vec::new()),
        None,
    )
    .unwrap();
    t.train_phase("bc", 0, 220, 220, &[], 0).unwrap();
    let val_corpus = corpus_with_state(99, 30, 0.5);
    let mut rng = SplitMix64::new(4);
    let (mut delta, mut n) = (0.0f32, 0usize);
    for _ in 0..30 {
        let w = val_corpus.sample_window(&mut rng, 12, 2, false);
        let mut idle = w.with_own_hook_masked(); // own hook hidden everywhere
        let mut out = w.with_own_hook_masked();
        for o in &mut out.observations {
            o.self_state.hook_state = ddai_brain::HOOK_GRABBED;
        }
        for o in &mut idle.observations {
            o.self_state.hook_state = ddai_brain::HOOK_IDLE;
        }
        let (a, b) = (
            t.learner().window_logits_played(&idle),
            t.learner().window_logits_played(&out),
        );
        for (x, y) in a.iter().zip(&b) {
            delta += (x.hook_prob() - y.hook_prob()).abs();
            n += 1;
        }
    }
    let val = EvalSet {
        name: "val".into(),
        corpus: Corpus::uniform(val_corpus.seqs),
    };
    (
        delta / n as f32,
        t.evaluate(&[val])[0].report.hook_by_state.not_out.auroc,
    )
}

#[test]
fn hiding_the_own_hook_from_the_hook_head_removes_the_copycat_shortcut() {
    let (plain, plain_auroc) = hook_reliance_on_own_state(OwnHookConfig::default());
    let (masked, masked_auroc) = hook_reliance_on_own_state(OwnHookConfig {
        mode: OwnHookMode::MaskHookHead,
        ..OwnHookConfig::default()
    });
    let (dropout, _) = hook_reliance_on_own_state(OwnHookConfig {
        mode: OwnHookMode::Dropout,
        dropout: 0.5,
        ..OwnHookConfig::default()
    });
    assert!(
        plain > 0.05,
        "trained on the shortcut, the plain hook head leans on the own hook state: {plain}"
    );
    assert_eq!(masked, 0.0, "the masked hook head cannot see it");
    assert!(
        dropout < plain,
        "dropout reduces the reliance: plain {plain} vs dropout {dropout}"
    );
    assert!(
        masked_auroc > 0.9 && plain_auroc > 0.9,
        "the rule is learnable either way: {masked_auroc}, {plain_auroc}"
    );
}

#[test]
fn the_masked_view_is_stored_with_the_model_and_the_played_logits_come_from_two_views() {
    let mut c = cfg(2);
    c.own_hook.mode = OwnHookMode::MaskHookHead;
    let dir = tempfile::tempdir().unwrap();
    let run = RunDir::create(dir.path()).unwrap();
    let mut t = Trainer::new(
        learner(),
        c,
        corpus_with_state(1, 10, 0.9),
        Corpus::new(Vec::new()),
        Some(run.clone()),
    )
    .unwrap();
    assert_eq!(t.learner().hook_view(), ddai_fly::bc::HookView::MaskedForHookHead);
    t.train_phase("bc", 0, 6, 6, &[], 0).unwrap();
    let b = load_control_bundle(&run.checkpoint("last.bundle")).unwrap();
    assert_eq!(b.hook_view, ddai_fly::bc::HookView::MaskedForHookHead);

    // The played logits: the hook head from the masked window, the rest from the full one.
    let w = corpus_with_state(5, 2, 0.9).sample_window(&mut SplitMix64::new(3), 8, 2, false);
    let net = t.learner();
    let played = net.window_logits_played(&w);
    let full = net.window_logits(&w);
    let masked = net.window_logits(&w.with_own_hook_masked());
    for k in 0..w.len() {
        assert_eq!(played[k].hook, masked[k].hook);
        assert_eq!(
            (played[k].dir, played[k].jump, played[k].fire),
            (full[k].dir, full[k].jump, full[k].fire)
        );
    }
    assert!(
        (0..w.len()).any(|k| (full[k].hook - masked[k].hook).abs() > 1e-6),
        "the own hook input does change the hook logit of an untrained-to-ignore-it model"
    );
}

#[test]
fn start_and_release_decisions_get_the_hook_loss_multiplier_and_dropout_hides_the_state() {
    use ddai_train::trainer::prepare_own_hook;
    let base = corpus_with_state(2, 6, 0.5).sample_window(&mut SplitMix64::new(1), 24, 2, false);
    let n_scored = base.targets.iter().filter(|t| t.weight > 0.0).count();
    assert!(n_scored > 5);

    // Off: untouched, no second view.
    let mut w = corpus_with_state(2, 6, 0.5).sample_window(&mut SplitMix64::new(1), 24, 2, false);
    let off = prepare_own_hook(&mut w, &OwnHookConfig::default(), &mut SplitMix64::new(9));
    assert!(off.is_none() && w.targets.iter().all(|t| t.hook_scale == 1.0));

    // A switch weight marks exactly the decisions where the label disagrees with "my hook is out".
    let mut w = corpus_with_state(2, 6, 0.5).sample_window(&mut SplitMix64::new(1), 24, 2, false);
    let cfgw = OwnHookConfig {
        switch_weight: 4.0,
        ..OwnHookConfig::default()
    };
    prepare_own_hook(&mut w, &cfgw, &mut SplitMix64::new(9));
    let mut marked = 0;
    for (t, o) in w.targets.iter().zip(&w.observations) {
        let out = matches!(
            o.self_state.hook_state,
            ddai_brain::HOOK_FLYING | ddai_brain::HOOK_GRABBED
        );
        let switch = t.weight > 0.0 && t.hook != out;
        assert_eq!(t.hook_scale, if switch { 4.0 } else { 1.0 });
        marked += usize::from(switch);
    }
    assert!(
        marked > 0 && marked < n_scored,
        "some but not all decisions are start/release: {marked}/{n_scored}"
    );

    // Dropout at p = 1 hides the state everywhere; the targets keep their switch marks from the true state.
    let mut w = corpus_with_state(2, 6, 0.5).sample_window(&mut SplitMix64::new(1), 24, 2, false);
    let all = OwnHookConfig {
        mode: OwnHookMode::Dropout,
        dropout: 1.0,
        switch_weight: 4.0,
    };
    assert!(prepare_own_hook(&mut w, &all, &mut SplitMix64::new(9)).is_none());
    assert!(
        w.observations
            .iter()
            .all(|o| o.self_state.hook_state == ddai_brain::HOOK_IDLE)
    );
    assert_eq!(w.targets.iter().filter(|t| t.hook_scale == 4.0).count(), marked);
    // p = 0 hides nothing; p = 0.5 hides about half of the grabbed ones.
    let mut w0 = corpus_with_state(2, 6, 0.5).sample_window(&mut SplitMix64::new(1), 24, 2, false);
    let none = OwnHookConfig {
        mode: OwnHookMode::Dropout,
        dropout: 0.0,
        ..OwnHookConfig::default()
    };
    prepare_own_hook(&mut w0, &none, &mut SplitMix64::new(9));
    let before = base
        .observations
        .iter()
        .filter(|o| o.self_state.hook_state != ddai_brain::HOOK_IDLE)
        .count();
    assert_eq!(
        w0.observations
            .iter()
            .filter(|o| o.self_state.hook_state != ddai_brain::HOOK_IDLE)
            .count(),
        before
    );
}
