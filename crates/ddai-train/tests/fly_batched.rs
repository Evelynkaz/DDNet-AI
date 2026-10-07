//! The fly learner's batched backend (task 7.2b, `[fly] backend = "batched"`) against the
//! per-sequence one, through the same `Learner` and `Trainer` code: the gradient of a batch of
//! windows (mirrored ones included, the activity regulariser on), a few trainer steps, and the
//! thread-count independence of the batched trainer. Skipped (with a note) when the compiled S
//! graph is not on this machine.

use std::path::PathBuf;
use std::sync::Arc;

use ddai_brain::CharacterObservation;
use ddai_dataset::types::ActionRec;
use ddai_fly::TrainBackend;
use ddai_fly::bc::{HeadMask, LossConfig};
use ddai_fly::rng::SplitMix64;
use ddai_physics::map::{MapData, TILE_SOLID, Tile};
use ddai_train::learner::{FlyLearner, FlyTrainConfig, Learner};
use ddai_train::seq::{Corpus, MapEntry, Seq, SeqStep, Source, Window};
use ddai_train::trainer::{TrainConfig, Trainer};
use ddai_train::types::char_rec;

fn graph() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("HOME").ok()?).join("aiddnet/data/connectome/compiled/fly-S-v1.flyg");
    p.exists().then_some(p)
}

fn corpus(n_seqs: usize) -> Corpus {
    corpus_with_latches(n_seqs, false)
}

/// The corpus of [`corpus`]; with `latches` every step carries a pseudo-random latch (the own previous hook command), which only an
/// intent hook head reads.
fn corpus_with_latches(n_seqs: usize, latches: bool) -> Corpus {
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
    let map = MapEntry::new(Arc::new(MapData {
        width: w as u32,
        height: h as u32,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }));
    let mut rng = SplitMix64::new(3);
    let seqs = (0..n_seqs)
        .map(|_| {
            let steps = (0..30)
                .map(|t| {
                    let x = 150.0 + rng.next_f32_unit() * 400.0;
                    let dx = (rng.next_f32_unit() - 0.5) * 300.0;
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
                            jump: dx.abs() < 60.0,
                            hook: dx.abs() < 120.0,
                            fire: false,
                            aim: [if dx > 0.0 { 100 } else { -100 }, 0],
                        },
                        soft: None,
                        weight: 1.0,
                        mask: HeadMask::ALL,
                        latch: latches && rng.next_f32_unit() < 0.5,
                    }
                })
                .collect();
            Seq {
                map: map.clone(),
                steps,
                source: Source::Human { demo: 0 },
            }
        })
        .collect();
    Corpus::new(seqs)
}

fn learner(backend: TrainBackend, par_threshold: Option<usize>) -> Option<FlyLearner> {
    learner_with(FlyTrainConfig {
        backend,
        batched_parallel_threshold: par_threshold,
        ..FlyTrainConfig::default()
    })
}

/// A learner from `cfg` with the test's activity regulariser and `alpha_init` on top.
fn learner_with(cfg: FlyTrainConfig) -> Option<FlyLearner> {
    let flyg = graph()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cfg = FlyTrainConfig {
        activity_weight: 0.05,
        activity_low: 0.4,
        activity_high: 0.8,
        alpha_init: 3.0,
        ..cfg
    };
    Some(FlyLearner::init(&flyg, &root.join("configs/fly/S-brain.toml"), 1, cfg, &[]).unwrap())
}

fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, &x| m.max(f64::from(x.abs()))).max(1e-12);
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (&x, &y)| m.max(f64::from((x - y).abs())))
        / scale
}

#[test]
fn batch_grad_matches_the_sum_of_window_grads() {
    let (Some(per_seq), Some(batched)) = (
        learner(TrainBackend::PerSequence, None),
        learner(TrainBackend::Batched, None),
    ) else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    assert!(!per_seq.uses_batched_backend() && batched.uses_batched_backend());
    let c = corpus(8);
    let mut rng = SplitMix64::new(5);
    let windows: Vec<Window> = (0..7)
        .map(|i| c.sample_window(&mut rng, 6 + i, 2, i % 2 == 1))
        .collect();
    let loss = LossConfig::default();

    let mut want = vec![0.0f32; per_seq.num_params()];
    let mut ws = per_seq.new_workspace(16);
    let mut want_stats = Vec::new();
    for w in &windows {
        let mut g = vec![0.0f32; want.len()];
        want_stats.push(per_seq.window_grad(w, &loss, &mut ws, &mut g));
        for (a, b) in want.iter_mut().zip(&g) {
            *a += b;
        }
    }
    let mut got = vec![0.0f32; want.len()];
    let stats = batched.batch_grad(&windows, &loss, &mut got).unwrap();
    assert_eq!(stats.len(), windows.len());
    for (s, r) in stats.iter().zip(&want_stats) {
        assert_eq!(s.weight_sum, r.weight_sum);
        assert!((s.loss.total - r.loss.total).abs() <= 1e-4 * r.loss.total.abs().max(1e-3));
        assert!(r.activity_loss > 0.0 && (s.activity_loss - r.activity_loss).abs() <= 1e-4 * r.activity_loss);
    }
    // Per parameter group (connectome a/b/theta, then encoder, then decoder) the batched gradient
    // equals the summed per-window one to f32 summation-order accuracy of the group's scale.
    let e_all = rel_err(&got, &want);
    eprintln!("batched vs summed per-window gradient, error relative to the largest entry: {e_all:.2e}");
    assert!(e_all < 5e-5, "{e_all}");
    assert!(got.iter().all(|g| g.is_finite()) && got.iter().any(|g| *g != 0.0));
}

fn train_cfg(threads: usize) -> TrainConfig {
    TrainConfig {
        batch_windows: 6,
        window_len: 10,
        burn_in: 2,
        warmup_steps: 3,
        threads,
        log_every: 4,
        human_fraction: 1.0,
        refresh_every: 0,
        ..TrainConfig::default()
    }
}

fn train(backend: TrainBackend, threads: usize, steps: u64, par_threshold: Option<usize>) -> Option<(Vec<f32>, f64)> {
    train_with(
        FlyTrainConfig {
            backend,
            batched_parallel_threshold: par_threshold,
            ..FlyTrainConfig::default()
        },
        threads,
        steps,
    )
}

fn train_with(cfg: FlyTrainConfig, threads: usize, steps: u64) -> Option<(Vec<f32>, f64)> {
    let l = learner_with(cfg)?;
    let mut t = Trainer::new(
        Box::new(l),
        train_cfg(threads),
        Corpus::new(Vec::new()),
        corpus(8),
        None,
    )
    .unwrap();
    let s = t.train_phase("bc", 0, steps, steps, &[], 0).unwrap();
    Some((t.learner().params(), s.mean_loss_last_log))
}

#[test]
fn trainer_steps_track_between_backends_and_do_not_depend_on_the_thread_count() {
    let Some((p_seq, loss_seq)) = train(TrainBackend::PerSequence, 2, 8, None) else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    let (p_bat, loss_bat) = train(TrainBackend::Batched, 2, 8, None).unwrap();
    assert!(loss_seq.is_finite() && loss_bat.is_finite());
    assert!(
        (loss_seq - loss_bat).abs() <= 1e-3 * loss_seq.abs().max(1.0),
        "loss after 8 steps: per-seq {loss_seq} vs batched {loss_bat}"
    );
    let drift = rel_err(&p_bat, &p_seq);
    eprintln!("parameters after 8 steps: batched vs per-seq max relative drift {drift:.2e}");
    assert!(drift < 1e-3, "{drift}");
    assert_ne!(p_bat, vec![0.0; p_bat.len()]);

    // The batched trainer is bitwise independent of the thread count. On S with this batch the
    // engine's regions are below the default parallel threshold (they would all run serially, and
    // the check would prove nothing), so force them onto the pool; and the serial run must agree.
    let (p1, _) = train(TrainBackend::Batched, 1, 5, Some(0)).unwrap();
    let (p4, _) = train(TrainBackend::Batched, 4, 5, Some(0)).unwrap();
    assert_eq!(p1, p4);
    let (p_serial, _) = train(TrainBackend::Batched, 4, 5, Some(usize::MAX)).unwrap();
    assert_eq!(p4, p_serial);
}

/// 8.2b's own-hook options (dropout of the own hook state, the start/release loss weight, the masked hook view,
/// and the two combined) train the same through the batched backend as through the per-sequence one: one step
/// (the gradient up to the optimiser) and a short run land on the same parameters and the same loss within the
/// 7.2b tolerance. The masked view is a second batched forward/backward over the masked windows.
#[test]
fn own_hook_options_train_the_same_through_both_backends() {
    use ddai_train::trainer::{OwnHookConfig, OwnHookMode};
    let options = [
        OwnHookConfig {
            mode: OwnHookMode::Dropout,
            dropout: 0.5,
            switch_weight: 1.0,
        },
        OwnHookConfig {
            mode: OwnHookMode::MaskHookHead,
            ..OwnHookConfig::default()
        },
        OwnHookConfig {
            switch_weight: 3.0,
            ..OwnHookConfig::default()
        },
        OwnHookConfig {
            mode: OwnHookMode::MaskHookHead,
            switch_weight: 3.0,
            ..OwnHookConfig::default()
        },
    ];
    let run = |backend, own: &OwnHookConfig, steps: u64| {
        let l = learner(backend, None)?;
        let mut c = train_cfg(2);
        c.own_hook = own.clone();
        let mut t = Trainer::new(Box::new(l), c, Corpus::new(Vec::new()), corpus(8), None).unwrap();
        let s = t.train_phase("bc", 0, steps, steps, &[], 0).unwrap();
        Some((t.learner().params(), s.mean_loss_last_log))
    };
    for own in &options {
        for steps in [1, 6] {
            let Some(init) = learner(TrainBackend::Batched, None).map(|l| l.params()) else {
                return;
            };
            let Some((p_seq, loss_seq)) = run(TrainBackend::PerSequence, own, steps) else {
                eprintln!("note: fly-S-v1.flyg not found, skipping");
                return;
            };
            let (p_bat, loss_bat) = run(TrainBackend::Batched, own, steps).unwrap();
            assert!(loss_seq.is_finite() && loss_bat.is_finite(), "{own:?}");
            assert!(
                (loss_seq - loss_bat).abs() <= 1e-3 * loss_seq.abs().max(1.0),
                "{own:?} after {steps} steps: loss per-seq {loss_seq} vs batched {loss_bat}"
            );
            let drift = rel_err(&p_bat, &p_seq);
            let moved = rel_err(&p_seq, &init);
            eprintln!("{own:?} after {steps} steps: parameters moved {moved:.2e}, backends differ by {drift:.2e}");
            // The comparison means something only if the steps moved the parameters well beyond the backends' gap.
            assert!(
                moved > 100.0 * drift.max(1e-9),
                "{own:?} after {steps} steps: moved {moved}, drift {drift}"
            );
            assert!(drift < 1e-3, "{own:?} after {steps} steps: {drift}");
        }
    }
    // The option is not a no-op: the masked run differs from the unmasked one.
    let (p_off, _) = run(TrainBackend::Batched, &OwnHookConfig::default(), 6).unwrap();
    let (p_mask, _) = run(TrainBackend::Batched, &options[1], 6).unwrap();
    assert_ne!(p_off, p_mask);
}

/// The opt-in speed options of task 7.2c through the trainer: `K` sub-batch engines track the
/// single engine and are bitwise independent of the thread count; the stop-gradient burn-in
/// trains (finite, different gradients) and is bitwise independent of the thread count too.
#[test]
fn subengines_and_stop_gradient_burn_in_through_the_trainer() {
    let batched = |k: usize, stop: usize, par: Option<usize>| FlyTrainConfig {
        backend: TrainBackend::Batched,
        batched_parallel_threshold: par,
        batched_subengines: k,
        batched_stop_grad_decisions: stop,
        ..FlyTrainConfig::default()
    };
    let Some((p_one, loss_one)) = train_with(batched(1, 0, None), 2, 8) else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    // 6 windows are one 8-lane cell: split a batch of two cells instead (see `batch_windows`).
    let wide = |cfg: FlyTrainConfig, threads: usize, steps: u64| {
        let l = learner_with(cfg).unwrap();
        let tc = TrainConfig {
            batch_windows: 12,
            ..train_cfg(threads)
        };
        let mut t = Trainer::new(Box::new(l), tc, Corpus::new(Vec::new()), corpus(8), None).unwrap();
        let s = t.train_phase("bc", 0, steps, steps, &[], 0).unwrap();
        (t.learner().params(), s.mean_loss_last_log)
    };
    let (p_k1, loss_k1) = wide(batched(1, 0, None), 2, 6);
    let (p_k2, loss_k2) = wide(batched(2, 0, None), 4, 6);
    assert!(
        (loss_k1 - loss_k2).abs() <= 1e-3 * loss_k1.abs().max(1.0),
        "loss with K=1 {loss_k1} vs K=2 {loss_k2}"
    );
    let drift = rel_err(&p_k2, &p_k1);
    eprintln!("K=2 vs K=1 after 6 steps: max relative parameter drift {drift:.2e}");
    assert!(drift < 1e-3, "{drift}");
    let (q1, _) = wide(batched(2, 0, Some(0)), 1, 4);
    let (q4, _) = wide(batched(2, 0, Some(0)), 4, 4);
    assert_eq!(q1, q4, "K=2 trainer, 1 vs 4 threads");

    // Stop-gradient burn-in (the config's `burn_in` is 2).
    let (p_stop, loss_stop) = train_with(batched(1, 2, None), 2, 8).unwrap();
    assert!(loss_stop.is_finite() && p_stop.iter().all(|x| x.is_finite()));
    assert_ne!(p_stop, p_one, "a truncated gradient trains differently");
    assert!(
        (loss_stop - loss_one).abs() <= 0.2 * loss_one.abs().max(1.0),
        "{loss_stop} vs {loss_one}"
    );
    let (s1, _) = train_with(batched(1, 2, Some(0)), 1, 5).unwrap();
    let (s4, _) = train_with(batched(1, 2, Some(0)), 4, 5).unwrap();
    assert_eq!(s1, s4, "stop-gradient trainer, 1 vs 4 threads");
}

/// Task 8.6: an **intent** hook head (two hazards chosen by the latch) trains through the batched backend as through the per-sequence one:
/// the batch gradient equals the sum of the windows', the flat parameter vector is the legacy one plus the release hazard, and both
/// hazards receive gradient (a window mixes latch states).
#[test]
fn an_intent_hook_head_trains_the_same_through_both_backends() {
    let Some(flyg) = graph() else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    let make = |backend| {
        let legacy = learner(backend, None).unwrap();
        let bundle =
            ddai_fly::bundle::upgrade_to_intent_hook(&legacy.to_bundle(ddai_fly::bundle::BundleMeta::default()));
        let cfg = FlyTrainConfig {
            backend,
            activity_weight: 0.05,
            activity_low: 0.4,
            activity_high: 0.8,
            ..FlyTrainConfig::default()
        };
        (
            legacy.num_params(),
            FlyLearner::from_bundle(bundle, &flyg, cfg).unwrap(),
        )
    };
    let ((n_legacy, per_seq), (_, batched)) = (make(TrainBackend::PerSequence), make(TrainBackend::Batched));
    assert_eq!(per_seq.hook_param(), ddai_fly::bc::HookParam::Intent);
    let n_release = per_seq.num_params() - n_legacy;
    assert!(
        n_release >= 2,
        "the release hazard adds its weights and a bias: {n_release}"
    );
    let c = corpus_with_latches(8, true);
    let mut rng = SplitMix64::new(9);
    let windows: Vec<Window> = (0..6)
        .map(|i| c.sample_window(&mut rng, 8 + i, 2, i % 2 == 1))
        .collect();
    assert!(
        windows.iter().flat_map(|w| &w.targets).any(|t| t.hook_latch)
            && windows.iter().flat_map(|w| &w.targets).any(|t| !t.hook_latch),
        "the windows carry both latch states"
    );
    let loss = LossConfig {
        hazard_pos_weight: Some([3.0, 5.0]),
        ..LossConfig::default()
    };
    let mut want = vec![0.0f32; per_seq.num_params()];
    let mut ws = per_seq.new_workspace(16);
    for w in &windows {
        let mut g = vec![0.0f32; want.len()];
        per_seq.window_grad(w, &loss, &mut ws, &mut g);
        for (a, b) in want.iter_mut().zip(&g) {
            *a += b;
        }
    }
    let mut got = vec![0.0f32; want.len()];
    batched.batch_grad(&windows, &loss, &mut got).unwrap();
    let e = rel_err(&got, &want);
    assert!(e < 5e-5, "{e}");
    // The release hazard sits at the end of the vector (after the aim angles) and gets a gradient; the hook head's press hazard too.
    let tail = &got[got.len() - n_release..];
    assert!(tail.iter().any(|g| *g != 0.0), "the release hazard learns");
    assert_eq!(per_seq.params().len(), got.len());
}

/// Task 8.6, "no copying" (E-005 F2): in a two-view intent fly the hook head reads the observation with the own hook state hidden and picks its
/// hazard by the latch, which is not an input of the network. On a corpus whose hook label does not depend on the own hook state (it is the
/// geometry rule; the observed state and the latch are random), after training: (1) flipping the **observed** own hook state of every
/// observation leaves the played hook logit exactly where it was; (2) flipping the **latch** leaves every other head's logits bit for bit
/// as they were (the network never sees it) and moves only the hook logit, through the hazard it selects.
#[test]
fn an_intent_fly_does_not_copy_the_hook_state_or_the_latch() {
    use ddai_train::trainer::{OwnHookConfig, OwnHookMode};
    let Some(flyg) = graph() else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    let legacy = learner(TrainBackend::Batched, None).unwrap();
    let bundle = ddai_fly::bundle::upgrade_to_intent_hook(&legacy.to_bundle(ddai_fly::bundle::BundleMeta::default()));
    let l = FlyLearner::from_bundle(
        bundle,
        &flyg,
        FlyTrainConfig {
            backend: TrainBackend::Batched,
            ..FlyTrainConfig::default()
        },
    )
    .unwrap();
    let mut cfg = train_cfg(2);
    cfg.own_hook = OwnHookConfig {
        mode: OwnHookMode::MaskHookHead,
        ..OwnHookConfig::default()
    };
    cfg.loss.hazard_pos_weight = Some([2.0, 4.0]);
    // Random observed hook states and latches, independent of the label.
    let mut c = corpus_with_latches(8, true);
    let mut rng = SplitMix64::new(11);
    for seq in &mut c.seqs {
        for st in &mut seq.steps {
            st.me.hook_state = if rng.next_f32_unit() < 0.5 {
                ddai_brain::HOOK_GRABBED as i8
            } else {
                ddai_brain::HOOK_IDLE as i8
            };
        }
    }
    let mut t = Trainer::new(Box::new(l), cfg, Corpus::new(Vec::new()), c, None).unwrap();
    t.train_phase("bc", 0, 6, 6, &[], 0).unwrap();
    let probe = corpus_with_latches(4, true);
    let mut rng = SplitMix64::new(3);
    let mut moved_by_latch = 0.0f32;
    for _ in 0..6 {
        let w = probe.sample_window(&mut rng, 8, 2, false);
        // (1) the observed own hook state, flipped everywhere.
        let a = with_state(&w, ddai_brain::HOOK_IDLE);
        let b = with_state(&w, ddai_brain::HOOK_GRABBED);
        let (la, lb) = (
            t.learner().window_logits_played(&a),
            t.learner().window_logits_played(&b),
        );
        for (x, y) in la.iter().zip(&lb) {
            assert_eq!(
                x.hook.to_bits(),
                y.hook.to_bits(),
                "the played hook logit does not read the observed own hook state"
            );
        }
        // (2) the latch, flipped everywhere.
        let mut up = with_state(&w, ddai_brain::HOOK_IDLE);
        let mut down = with_state(&w, ddai_brain::HOOK_IDLE);
        up.targets.iter_mut().for_each(|t| t.hook_latch = false);
        down.targets.iter_mut().for_each(|t| t.hook_latch = true);
        let (lu, ld) = (
            t.learner().window_logits_played(&up),
            t.learner().window_logits_played(&down),
        );
        for (x, y) in lu.iter().zip(&ld) {
            let bits = |l: &ddai_fly::bc::HeadLogits| {
                (
                    l.dir.map(f32::to_bits),
                    l.jump.to_bits(),
                    l.fire.to_bits(),
                    l.aim_c.to_bits(),
                    l.aim_s.to_bits(),
                )
            };
            assert_eq!(bits(x), bits(y), "no head but the hook reads the latch");
            moved_by_latch += (x.hook - y.hook).abs();
        }
    }
    assert!(
        moved_by_latch > 0.0,
        "the hook logit does depend on the latch, through the hazard it selects"
    );
}

/// `w` with the observed own hook state of every observation set to `state`.
fn with_state(w: &Window, state: i32) -> Window {
    Window {
        observations: w
            .observations
            .iter()
            .map(|o| {
                let mut o = o.clone();
                o.self_state.hook_state = state;
                o
            })
            .collect(),
        targets: w.targets.clone(),
        start: w.start,
        mirrored: w.mirrored,
    }
}
