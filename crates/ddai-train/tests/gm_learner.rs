//! The FlyGM-style neuron model (task 8.8, `[fly.gm]`) through the fly learner and the trainer on the real S graph: the flat
//! parameter layout and the gradient placed in it (checked against finite differences of the loss), a short BC run (the loss falls,
//! the result does not depend on the thread count), and the checkpoint round trip (format v6, the same decisions after a reload,
//! a `Rate` bundle still written in its old layout). Skipped (with a note) when the compiled S graph is not on this machine.

use std::path::PathBuf;
use std::sync::Arc;

use ddai_brain::CharacterObservation;
use ddai_dataset::types::ActionRec;
use ddai_fly::bc::{HeadMask, LossConfig};
use ddai_fly::bundle::{BundleMeta, FlyBrainTemplate, NeuronModel, load_bundle, peek_version, read_zstd_bytes};
use ddai_fly::gm::{GmConfig, GmUpdate};
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

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
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

fn gm_cfg(update: GmUpdate) -> FlyTrainConfig {
    FlyTrainConfig {
        gm: Some(GmConfig {
            update,
            ..GmConfig::default()
        }),
        ..FlyTrainConfig::default()
    }
}

fn learner(cfg: FlyTrainConfig) -> Option<FlyLearner> {
    let flyg = graph()?;
    Some(FlyLearner::init(&flyg, &root().join("configs/fly/S-brain.toml"), 1, cfg, &[]).unwrap())
}

#[test]
fn gm_learner_layout_has_no_connectome_parameters_and_matches_the_lr_vector() {
    let Some(l) = learner(gm_cfg(GmUpdate::Plain)) else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    let rate = learner(FlyTrainConfig::default()).unwrap();
    let flat = l.params();
    assert_eq!(flat.len(), l.num_params());
    assert_eq!(l.base_lrs().len(), l.num_params());
    // Same encoder and decoder as the rate fly; the connectome's a/b/theta (about 6.5k numbers) are replaced by the Gm network's.
    let gm_total = l.model().gm().unwrap().shape().total();
    let rate_fly = rate.model().params();
    let rate_conn = rate_fly.a.len() + rate_fly.b.len() + rate_fly.theta.len();
    assert_eq!(flat.len() + rate_conn, rate.num_params() + gm_total);
    eprintln!(
        "Gm fly: {} parameters in all, {gm_total} in the network (rate fly: {rate_conn})",
        flat.len()
    );
    // set_params(params()) is the identity; a changed parameter reaches the model.
    let mut l2 = learner(gm_cfg(GmUpdate::Plain)).unwrap();
    l2.set_params(&flat).unwrap();
    assert_eq!(l2.params(), flat);
    let mut changed = flat.clone();
    let at = l.num_params() - 3;
    changed[at] += 0.25;
    l2.set_params(&changed).unwrap();
    assert_eq!(l2.params(), changed);
    assert!(l2.set_params(&flat[1..]).is_err());
}

/// The gradient of the summed window loss lands in the right places of the flat vector: for random directions inside each region
/// (the encoder, the Gm network, the decoder) the directional derivative from `window_grad` equals the central finite difference of the
/// loss (f32 forward, so a loose but meaningful tolerance; the f64 check of the network's own backward is in `ddai-fly`'s `gm::tests`).
#[test]
fn gm_window_grad_matches_directional_finite_differences() {
    for update in [GmUpdate::Plain, GmUpdate::Gated] {
        let Some(mut l) = learner(gm_cfg(update)) else {
            eprintln!("note: fly-S-v1.flyg not found, skipping");
            return;
        };
        let c = corpus(6);
        let mut rng = SplitMix64::new(11);
        let windows: Vec<Window> = (0..4).map(|i| c.sample_window(&mut rng, 8, 2, i % 2 == 1)).collect();
        let loss_cfg = LossConfig::default();
        let theta = l.params();
        let total_loss = |l: &FlyLearner| -> f64 {
            let mut ws = l.new_workspace(16);
            let mut sum = 0.0f64;
            for w in &windows {
                let mut g = vec![0.0f32; l.num_params()];
                sum += f64::from(l.window_grad(w, &loss_cfg, &mut ws, &mut g).loss.total);
            }
            sum
        };
        let mut grad = vec![0.0f32; l.num_params()];
        {
            let mut ws = l.new_workspace(16);
            for w in &windows {
                l.window_grad(w, &loss_cfg, &mut ws, &mut grad);
            }
        }
        let layout_gm = l.model().gm().unwrap().shape().total();
        let enc_len = theta.len() - layout_gm - decoder_len(&l);
        let regions = [
            ("encoder", 0..enc_len),
            ("gm", enc_len..enc_len + layout_gm),
            ("decoder", enc_len + layout_gm..theta.len()),
        ];
        for (name, range) in regions {
            let mut u = vec![0.0f32; theta.len()];
            let mut norm = 0.0f64;
            for i in range.clone() {
                u[i] = rng.next_gaussian();
                norm += f64::from(u[i]) * f64::from(u[i]);
            }
            let norm = norm.sqrt() as f32;
            for x in &mut u {
                *x /= norm;
            }
            let analytic: f64 = grad.iter().zip(&u).map(|(&g, &d)| f64::from(g) * f64::from(d)).sum();
            // The finite difference converges to the analytic value as the step shrinks; the last (smallest) step is the one asserted.
            let mut fd = 0.0;
            for eps in [0.04f32, 0.01, 0.0025] {
                let probe = |sign: f32, l: &mut FlyLearner| {
                    let p: Vec<f32> = theta.iter().zip(&u).map(|(&t, &d)| t + sign * eps * d).collect();
                    l.set_params(&p).unwrap();
                    total_loss(l)
                };
                let lp = probe(1.0, &mut l);
                let lm = probe(-1.0, &mut l);
                l.set_params(&theta).unwrap();
                fd = (lp - lm) / (2.0 * f64::from(eps));
                eprintln!("{update:?} {name}: eps {eps}: analytic {analytic:.4} vs finite difference {fd:.4}");
            }
            assert!(
                (analytic - fd).abs() <= 0.04 * analytic.abs().max(fd.abs()) + 0.05,
                "{update:?} {name}: analytic {analytic} vs finite difference {fd}"
            );
            assert!(
                analytic.abs() > 1e-3,
                "{update:?} {name}: the gradient along the probe direction is (nearly) zero"
            );
        }
    }
}

fn decoder_len(l: &FlyLearner) -> usize {
    l.params().len() - l.decoder_param_start()
}

fn train_cfg(threads: usize) -> TrainConfig {
    TrainConfig {
        batch_windows: 6,
        window_len: 10,
        burn_in: 2,
        warmup_steps: 3,
        threads,
        log_every: 5,
        human_fraction: 1.0,
        refresh_every: 0,
        ..TrainConfig::default()
    }
}

fn train(threads: usize, steps: u64) -> Option<(Vec<f32>, f64)> {
    let l = learner(gm_cfg(GmUpdate::Plain))?;
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
fn gm_trainer_learns_a_synthetic_task_and_ignores_the_thread_count() {
    let Some((p_few, loss_few)) = train(1, 5) else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    let (p1, loss_long_1) = train(1, 60).unwrap();
    let (p3, loss_long_3) = train(3, 60).unwrap();
    eprintln!("Gm BC loss: {loss_few:.3} after 5 steps, {loss_long_1:.3} after 60");
    assert!(loss_few.is_finite() && loss_long_1.is_finite());
    assert!(
        loss_long_1 < 0.9 * loss_few,
        "the loss did not fall: {loss_few} -> {loss_long_1}"
    );
    assert_eq!(p1, p3, "the result must not depend on the thread count");
    assert_eq!(loss_long_1, loss_long_3);
    assert_ne!(p1, p_few);
}

#[test]
fn gm_bundle_round_trip_and_playing_it_gives_the_learners_decisions() {
    let Some(mut l) = learner(gm_cfg(GmUpdate::Gated)) else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    // Move the parameters away from the initialisation so a missing field cannot hide.
    let mut p = l.params();
    let mut rng = SplitMix64::new(4);
    for x in &mut p {
        *x += 0.05 * rng.next_gaussian();
    }
    l.set_params(&p).unwrap();
    l.refresh(); // the resting state of the moved network

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gm.bundle");
    l.save(&path, BundleMeta::default()).unwrap();
    let bytes = read_zstd_bytes(&path).unwrap();
    assert_eq!(
        peek_version(&path, &bytes).unwrap(),
        6,
        "a Gm bundle is format version 6"
    );
    let b = load_bundle(&path).unwrap();
    assert!(matches!(b.neuron_model, NeuronModel::Gm { .. }));
    assert_eq!(b, load_bundle(&path).unwrap());

    // Restoring the learner from the bundle gives the same parameters.
    let flyg = graph().unwrap();
    let l2 = FlyLearner::from_bundle(b.clone(), &flyg, FlyTrainConfig::default()).unwrap();
    assert_eq!(l2.params(), l.params());
    // [fly.gm] next to a bundle must name the bundle's own model (a resumed run keeps its config), nothing else.
    assert!(FlyLearner::from_bundle(b.clone(), &flyg, gm_cfg(GmUpdate::Plain)).is_err());
    assert_eq!(
        FlyLearner::from_bundle(b.clone(), &flyg, gm_cfg(GmUpdate::Gated))
            .unwrap()
            .params(),
        l.params()
    );

    // Playing the loaded bundle takes the learner's own decisions (the window logits, from the same resting state).
    let template = FlyBrainTemplate::load(&path, Some(&flyg)).unwrap();
    let c = corpus(2);
    let mut rng = SplitMix64::new(8);
    let w = c.sample_window(&mut rng, 12, 0, false);
    let want = l.window_logits(&w);
    let mut brain = template.instantiate(ddai_fly::brain::FlyBrainConfig::default());
    ddai_brain::Brain::reset(
        &mut brain,
        &ddai_brain::ResetContext {
            map: w.observations[0].map.clone(),
            self_id: 0,
            seed: 1,
        },
    );
    for (t, obs) in w.observations.iter().enumerate() {
        let got = brain.forward_logits(obs);
        for k in 0..3 {
            assert!(
                (got.dir[k] - want[t].dir[k]).abs() < 1e-4,
                "decision {t}: direction logit {k}: {got:?} vs {:?}",
                want[t]
            );
        }
        assert!((got.hook - want[t].hook).abs() < 1e-4, "decision {t}: hook logit");
    }
}

#[test]
fn a_rate_bundle_is_still_written_in_its_old_layout() {
    let Some(l) = learner(FlyTrainConfig::default()) else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rate.bundle");
    l.save(&path, BundleMeta::default()).unwrap();
    let bytes = read_zstd_bytes(&path).unwrap();
    // Legacy hook head, plain decode, pooled readout: the layout every binary since 8.2 reads.
    assert_eq!(peek_version(&path, &bytes).unwrap(), 3);
    assert!(load_bundle(&path).unwrap().neuron_model.is_rate());
}

/// Review F5: the machinery that knows only the rate model refuses a `Gm` fly with a message instead of training its inert placeholder
/// parameters (PPO, ES), and the learner refuses the batched backend and the activity regulariser for it.
#[test]
fn the_rate_only_machinery_refuses_a_gm_fly() {
    use ddai_fly::TrainBackend;
    use ddai_fly::policy::Temperatures;
    use ddai_train::es::space::{ParamSpace, SpaceConfig};
    use ddai_train::ppo::config::PpoParams;
    use ddai_train::ppo::learner::PpoLearner;
    let Some(l) = learner(gm_cfg(GmUpdate::Plain)) else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    let flyg = graph().unwrap();
    let bundle = l.to_bundle(BundleMeta::default());
    assert!(matches!(bundle.neuron_model, NeuronModel::Gm { .. }));

    let e = ParamSpace::new(&bundle, &SpaceConfig::default()).unwrap_err();
    assert!(e.contains("ES searches the rate model") && e.contains("gm-d8"), "{e}");

    let e = match PpoLearner::new(
        PpoParams::default(),
        &bundle,
        &flyg,
        4.0,
        Temperatures::uniform(1.0),
        1,
        None,
    ) {
        Err(e) => e,
        Ok(_) => panic!("PPO must refuse a Gm checkpoint"),
    };
    assert!(e.contains("rate model only") && e.contains("gm-d8"), "{e}");

    let brain = root().join("configs/fly/S-brain.toml");
    let batched = FlyTrainConfig {
        backend: TrainBackend::Batched,
        ..gm_cfg(GmUpdate::Plain)
    };
    let e = match FlyLearner::init(&flyg, &brain, 1, batched, &[]) {
        Err(e) => e,
        Ok(_) => panic!("the batched backend must refuse a Gm fly"),
    };
    assert!(e.contains("per-seq"), "{e}");
    let activity = FlyTrainConfig {
        activity_weight: 0.5,
        ..gm_cfg(GmUpdate::Plain)
    };
    let e = match FlyLearner::init(&flyg, &brain, 1, activity, &[]) {
        Err(e) => e,
        Ok(_) => panic!("the activity regulariser must refuse a Gm fly"),
    };
    assert!(e.contains("activity_weight"), "{e}");
}
