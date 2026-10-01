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
    let flyg = graph()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cfg = FlyTrainConfig {
        backend,
        batched_parallel_threshold: par_threshold,
        activity_weight: 0.05,
        activity_low: 0.4,
        activity_high: 0.8,
        alpha_init: 3.0,
        ..FlyTrainConfig::default()
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
    let l = learner(backend, par_threshold)?;
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
