//! Task 8.7: training only the hook head's readout (`[fly] readout_only`) on the real S graph. Skipped (with a note) when the compiled
//! graph is not on this machine.
//!
//! * the flat parameter vector round-trips with a wide readout, and only the hook head's parameters have a learning rate;
//! * the readout-only gradient is exactly the hook head's slice of the full BC gradient with only the hook head scored;
//! * a few trainer steps move the hook head and leave **every** other parameter bit for bit where it was, the calibration included.

use std::path::PathBuf;
use std::sync::Arc;

use ddai_brain::CharacterObservation;
use ddai_dataset::types::ActionRec;
use ddai_fly::bc::{HeadMask, HookView, LossConfig};
use ddai_fly::bundle::{BundleMeta, upgrade_hook_readout};
use ddai_fly::hook_wide::HookReadout;
use ddai_fly::rng::SplitMix64;
use ddai_physics::map::{MapData, TILE_SOLID, Tile};
use ddai_train::learner::{FlyLearner, FlyTrainConfig, Learner};
use ddai_train::seq::{Corpus, MapEntry, Seq, SeqStep, Source};
use ddai_train::trainer::{OwnHookConfig, OwnHookMode, TrainConfig, Trainer};
use ddai_train::types::char_rec;

fn graph() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("HOME").ok()?).join("aiddnet/data/connectome/compiled/fly-S-v1.flyg");
    p.exists().then_some(p)
}

fn corpus(n_seqs: usize) -> Corpus {
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
                        latch: false,
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

/// A learner whose hook head has the `kind` readout, with non-zero output weights (so the readout has a gradient into itself).
fn wide_learner(kind: HookReadout, cfg: FlyTrainConfig) -> Option<FlyLearner> {
    let flyg = graph()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let base = FlyLearner::init(
        &flyg,
        &root.join("configs/fly/S-brain.toml"),
        1,
        FlyTrainConfig {
            alpha_init: 3.0,
            ..FlyTrainConfig::default()
        },
        &[],
    )
    .unwrap();
    let mut bundle = base.to_bundle(BundleMeta::default());
    bundle.hook_view = HookView::MaskedForHookHead;
    let mut up = upgrade_hook_readout(&bundle, ddai_flyg::load(&flyg).unwrap(), kind, 4).unwrap();
    if let Some(w) = up.decoder_params.hook_wide.as_mut() {
        for (i, x) in w.w2.iter_mut().enumerate() {
            *x = 0.05 * ((i % 7) as f32 - 3.0);
        }
    }
    Some(FlyLearner::from_bundle(up, &flyg, cfg).unwrap())
}

#[test]
fn the_flat_vector_round_trips_and_only_the_hook_head_has_a_learning_rate() {
    for kind in [HookReadout::LinearDn, HookReadout::MlpDn { hidden: 8 }] {
        let cfg = FlyTrainConfig {
            readout_only: true,
            lr_hook_wide: 3e-3,
            ..FlyTrainConfig::default()
        };
        let Some(mut l) = wide_learner(kind, cfg) else {
            eprintln!("note: fly-S-v1.flyg not found, skipping");
            return;
        };
        assert_eq!(l.hook_readout(), kind);
        let p = l.params();
        l.set_params(&p).unwrap();
        assert_eq!(l.params(), p);
        let b = l.to_bundle(BundleMeta::default());
        assert_eq!(b.hook_readout, kind);
        let n_wide = {
            let w = b.decoder_params.hook_wide.as_ref().unwrap();
            w.w1.len() + w.b1.len() + w.w2.len()
        };
        let lrs = l.base_lrs();
        assert_eq!(lrs.len(), p.len());
        let moving = lrs.iter().filter(|&&x| x > 0.0).count();
        assert_eq!(moving, b.decoder_params.hook_w.len() + 1 + n_wide, "{kind:?}");
        assert!(lrs[lrs.len() - n_wide..].iter().all(|&x| x == 3e-3));
    }
}

#[test]
fn the_readout_only_gradient_is_the_hook_slice_of_the_full_gradient_with_only_the_hook_scored() {
    let kind = HookReadout::MlpDn { hidden: 8 };
    let (Some(full), Some(only)) = (
        wide_learner(kind, FlyTrainConfig::default()),
        wide_learner(
            kind,
            FlyTrainConfig {
                readout_only: true,
                ..FlyTrainConfig::default()
            },
        ),
    ) else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    let c = corpus(6);
    let mut rng = SplitMix64::new(9);
    let window = c.sample_window(&mut rng, 12, 3, false);
    // Only the hook head scored: what the trainer's second view gives.
    let loss = LossConfig {
        w_dir: 0.0,
        w_jump: 0.0,
        w_fire: 0.0,
        w_aim: 0.0,
        pos_weight: [1.0, 2.0, 1.0],
        ..LossConfig::default()
    };
    let mut g_full = vec![0.0f32; full.num_params()];
    let mut g_only = vec![0.0f32; only.num_params()];
    let s_full = full.window_grad(&window, &loss, &mut full.new_workspace(16), &mut g_full);
    let s_only = only.window_grad(&window, &loss, &mut only.new_workspace(16), &mut g_only);
    assert_eq!(s_full.weight_sum, s_only.weight_sum);
    assert!((s_full.loss.total - s_only.loss.total).abs() <= 1e-5 * s_full.loss.total.abs().max(1e-3));
    let lrs = only.base_lrs();
    let mut compared = 0;
    for (i, ((a, b), lr)) in g_full.iter().zip(&g_only).zip(&lrs).enumerate() {
        if *lr > 0.0 {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "param {i}: full {a} vs readout-only {b}"
            );
            compared += 1;
        } else {
            assert_eq!(*b, 0.0, "param {i} outside the hook head must get no gradient");
        }
    }
    assert!(compared > 0 && g_only.iter().any(|&x| x != 0.0));
}

#[test]
fn trainer_steps_move_the_hook_head_and_nothing_else() {
    let kind = HookReadout::MlpDn { hidden: 8 };
    let cfg = FlyTrainConfig {
        readout_only: true,
        ..FlyTrainConfig::default()
    };
    let Some(l) = wide_learner(kind, cfg) else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    let before = l.params();
    let lrs = l.base_lrs();
    let tcfg = TrainConfig {
        batch_windows: 6,
        window_len: 10,
        burn_in: 2,
        warmup_steps: 2,
        threads: 2,
        log_every: 2,
        human_fraction: 1.0,
        refresh_every: 0,
        own_hook: OwnHookConfig {
            mode: OwnHookMode::MaskHookHead,
            ..OwnHookConfig::default()
        },
        ..TrainConfig::default()
    };
    let mut t = Trainer::new(Box::new(l), tcfg, Corpus::new(Vec::new()), corpus(8), None).unwrap();
    let s = t.train_phase("bc", 0, 6, 6, &[], 0).unwrap();
    assert!(s.mean_loss_last_log.is_finite());
    let after = t.learner().params();
    let mut moved = 0;
    for (i, ((a, b), lr)) in before.iter().zip(&after).zip(&lrs).enumerate() {
        if *lr > 0.0 {
            moved += usize::from(a != b);
        } else {
            assert_eq!(a.to_bits(), b.to_bits(), "param {i} outside the hook head moved");
        }
    }
    assert!(moved > 0, "the hook head did not train");
}

/// Task 8.7, latency: the two-view S fly with each hook readout, decisions per second of wall clock on this machine (one core, alone in the
/// process). `#[ignore]`d: needs local bundles. `E031_BUNDLES` = comma-separated `name=path`; run with
/// `cargo test -p ddai-train --release --test hook_readout -- --ignored --nocapture latency`.
#[test]
#[ignore]
fn latency_of_the_two_view_fly_per_readout() {
    use ddai_brain::ResetContext;
    use ddai_fly::brain::{ActionSelection, FlyBrainConfig};
    use ddai_fly::bundle::FlyBrainTemplate;
    let Some(flyg) = graph() else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    let spec = std::env::var("E031_BUNDLES").expect("E031_BUNDLES=name=path,...");
    let (w, h) = (60u32, 60u32);
    let map = Arc::new(MapData {
        width: w,
        height: h,
        game: vec![Tile::default(); (w * h) as usize],
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    });
    let mut rng = SplitMix64::new(11);
    let observations: Vec<ddai_brain::Observation> = (0..4000)
        .map(|_| {
            let mut me = CharacterObservation::at_rest(0);
            me.pos = ddai_physics::vmath::Vec2::new(
                600.0 + rng.next_f32_unit() * 200.0,
                600.0 + rng.next_f32_unit() * 200.0,
            );
            me.vel =
                ddai_physics::vmath::Vec2::new(rng.next_f32_unit() * 200.0 - 100.0, rng.next_f32_unit() * 100.0 - 50.0);
            let mut opp = CharacterObservation::at_rest(1);
            opp.pos = ddai_physics::vmath::Vec2::new(
                600.0 + rng.next_f32_unit() * 400.0,
                600.0 + rng.next_f32_unit() * 400.0,
            );
            ddai_brain::Observation {
                map: map.clone(),
                tick: 0,
                self_state: me,
                others: vec![opp],
                target_id: None,
                tuning: ddai_physics::tuning::TuningParams::default(),
            }
        })
        .collect();
    for item in spec.split(',') {
        let (name, path) = item.split_once('=').expect("name=path");
        let t = FlyBrainTemplate::load(std::path::Path::new(path), Some(&flyg)).unwrap();
        let mut brain = t.instantiate_played(FlyBrainConfig {
            action_selection: ActionSelection::Argmax,
            seed: 1,
        });
        brain.reset(&ResetContext {
            map: map.clone(),
            self_id: 0,
            seed: 1,
        });
        for o in observations.iter().take(200) {
            let _ = brain.decide(o);
        }
        let mut us: Vec<f64> = observations
            .iter()
            .map(|o| {
                let t0 = std::time::Instant::now();
                std::hint::black_box(brain.decide(o));
                t0.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        us.sort_by(f64::total_cmp);
        let pct = |p: f64| us[((p / 100.0) * (us.len() - 1) as f64).round() as usize];
        eprintln!(
            "latency {name} ({}): p50 {:.0} us, p99 {:.0} us, max {:.0} us over {} decisions",
            t.hook_readout().label(),
            pct(50.0),
            pct(99.0),
            us[us.len() - 1],
            us.len()
        );
    }
}
