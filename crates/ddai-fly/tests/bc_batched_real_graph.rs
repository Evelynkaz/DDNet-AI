//! `brain_bc_batched_step` (task 7.2b) against the per-window `brain_bc_step` (task 8.2, the
//! reference) on the real S and M graphs: a batch of windows with different lengths, soft targets,
//! weights, head masks, a burn-in decision, mirrored-looking observations and the activity
//! regulariser on. Per window: loss, activity loss, weight sum, logits, encoder and decoder
//! gradients; per batch: the connectome gradients (summed over the windows). Skipped (with a note)
//! when a compiled graph is not on this machine.

use std::path::PathBuf;
use std::sync::Arc;

use ddai_brain::{CharacterObservation, Observation};
use ddai_fly::backward::BackwardIndex;
use ddai_fly::batched::BatchedEngine;
use ddai_fly::bc::{HeadMask, LossConfig, SoftTargets, StepTargets};
use ddai_fly::brain_bc::{BcSequence, BcStepConfig, BcWorkspace, brain_bc_step};
use ddai_fly::brain_bc_batched::brain_bc_batched_step;
use ddai_fly::config::FlyConfig;
use ddai_fly::decoder::{DecoderModel, DnCalibration};
use ddai_fly::encoder::{EncoderModel, EncoderParams};
use ddai_fly::model::FlyModel;
use ddai_fly::optim::ActivityRegularizerConfig;
use ddai_fly::params::FlyParams;
use ddai_fly::rng::SplitMix64;

fn graph(name: &str) -> Option<ddai_flyg::Flyg> {
    let home = std::env::var("HOME").ok()?;
    let path = PathBuf::from(home).join(format!("aiddnet/data/connectome/compiled/fly-{name}-v1.flyg"));
    path.exists().then(|| ddai_flyg::load(&path).expect("load graph"))
}

fn open_map() -> ddai_physics::map::MapData {
    ddai_physics::map::MapData {
        width: 40,
        height: 40,
        game: vec![Default::default(); 1600],
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

fn obs(map: &Arc<ddai_physics::map::MapData>, ox: f32, oy: f32, vx: f32) -> Observation {
    let mut me = CharacterObservation::at_rest(0);
    me.pos = ddai_physics::vmath::Vec2::new(640.0, 640.0);
    me.vel = ddai_physics::vmath::Vec2::new(vx, -20.0);
    let mut opp = CharacterObservation::at_rest(1);
    opp.pos = ddai_physics::vmath::Vec2::new(ox, oy);
    opp.vel = ddai_physics::vmath::Vec2::new(-20.0, 15.0);
    Observation {
        map: map.clone(),
        tick: 0,
        self_state: me,
        others: vec![opp],
        target_id: None,
        tuning: ddai_physics::tuning::TuningParams::default(),
    }
}

fn targets_for(rng: &mut SplitMix64, len: usize) -> Vec<StepTargets> {
    (0..len)
        .map(|t| {
            if t == 0 {
                return StepTargets::burn_in();
            }
            let soft = (rng.next_f32_unit() < 0.5).then(|| {
                let a = rng.next_f32_unit();
                let b = (1.0 - a) * rng.next_f32_unit();
                SoftTargets {
                    dir: [a, b, 1.0 - a - b],
                    jump: rng.next_f32_unit(),
                    hook: rng.next_f32_unit(),
                    fire: rng.next_f32_unit(),
                }
            });
            StepTargets {
                dir: (rng.next_u64() % 3) as u8,
                jump: rng.next_f32_unit() < 0.4,
                hook: rng.next_f32_unit() < 0.5,
                fire: rng.next_f32_unit() < 0.3,
                aim: rng.next_f32_unit() * 6.0 - 3.0,
                soft,
                mask: HeadMask {
                    fire: rng.next_f32_unit() < 0.7,
                    ..HeadMask::ALL
                },
                weight: 0.5 + rng.next_f32_unit(),
                hook_scale: 1.0,
            }
        })
        .collect()
}

/// `max|a - b| / max(max|b|, tiny)`: the error relative to the group's own scale.
fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f64, |m, &x| m.max(f64::from(x.abs()))).max(1e-12);
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (&x, &y)| m.max(f64::from((x - y).abs())))
        / scale
}

/// The model, networks and batch of windows both tests below run.
struct Fixture {
    model: FlyModel,
    index: BackwardIndex,
    encoder: EncoderModel,
    encoder_params: EncoderParams,
    decoder: DecoderModel,
    decoder_params: ddai_fly::decoder::DecoderParams,
    calib: DnCalibration,
    cfg: BcStepConfig,
    seqs: Vec<BcSequence>,
}

fn fixture(name: &str, brain_config: &str, v_scale: f32) -> Option<Fixture> {
    let Some(flyg) = graph(name) else {
        eprintln!("skipped: fly-{name}-v1.flyg not found");
        return None;
    };
    let brain_cfg = ddai_fly::brain_config::load_brain_config(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../configs/fly/{brain_config}")),
    )
    .unwrap();
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 11);
    let model = FlyModel::new(flyg, config, params).unwrap();
    let index = BackwardIndex::build(&model);
    let encoder = EncoderModel::new(&model, brain_cfg.ray_grid, &brain_cfg.proprioception).unwrap();
    let mut encoder_params = EncoderParams::init_default(encoder.num_params());
    for (i, g) in encoder_params.g.iter_mut().enumerate() {
        *g = 0.9 + 0.01 * (i % 23) as f32;
    }
    let decoder = DecoderModel::new(&model, brain_cfg.decoder.clone()).unwrap();
    let mut decoder_params = decoder.init_default_params();
    for (i, w) in decoder_params.hook_w.iter_mut().enumerate() {
        *w = 0.05 * (i as f32 - 2.0);
    }
    for (i, w) in decoder_params.jump_w.iter_mut().enumerate() {
        *w = 0.04 * (i as f32);
    }
    for (i, w) in decoder_params.direction_lr_w.iter_mut().enumerate() {
        *w = 0.03 * (i as f32 - 1.5);
    }
    for (i, t) in decoder_params.aim_pair_theta.iter_mut().enumerate() {
        *t = 0.3 + 0.2 * i as f32;
    }
    let calib = DnCalibration {
        mu: vec![0.2; model.num_outputs()],
        sigma: vec![1.0; model.num_outputs()],
    };
    let cfg = BcStepConfig {
        loss: LossConfig {
            soft_mix: 0.5,
            pos_weight: [2.0, 1.5, 1.0],
            ..LossConfig::default()
        },
        activity: ActivityRegularizerConfig {
            weight: 0.05,
            low: 0.4,
            high: 0.8,
        },
    };
    let map = Arc::new(open_map());
    let mut rng = SplitMix64::new(77);
    let lens = [6usize, 9, 3, 6, 8, 5, 6, 7, 4, 9];
    let seqs: Vec<BcSequence> = lens
        .iter()
        .enumerate()
        .map(|(w, &len)| BcSequence {
            v_init: (0..model.num_neurons())
                .map(|_| rng.next_f32_unit() * v_scale)
                .collect(),
            observations: (0..len)
                .map(|t| {
                    obs(
                        &map,
                        700.0 + 30.0 * w as f32 + 10.0 * t as f32,
                        560.0 + 20.0 * ((w + t) % 5) as f32,
                        -40.0 + 15.0 * ((w * 3 + t) % 7) as f32,
                    )
                })
                .collect(),
            targets: targets_for(&mut rng, len),
        })
        .collect();
    Some(Fixture {
        model,
        index,
        encoder,
        encoder_params,
        decoder,
        decoder_params,
        calib,
        cfg,
        seqs,
    })
}

fn check_graph(name: &str, brain_config: &str, v_scale: f32) {
    let Some(Fixture {
        model,
        index,
        encoder,
        encoder_params,
        decoder,
        decoder_params,
        calib,
        cfg,
        seqs,
    }) = fixture(name, brain_config, v_scale)
    else {
        return;
    };

    // Reference: window by window.
    let mut ws = BcWorkspace::new(&model, &encoder, 16);
    let refs: Vec<_> = seqs
        .iter()
        .map(|s| {
            brain_bc_step(
                &model,
                &index,
                &encoder,
                &encoder_params,
                &decoder,
                &decoder_params,
                &calib,
                s,
                &cfg,
                &mut ws,
            )
        })
        .collect();
    let mut engine = BatchedEngine::new(&model);
    let got = brain_bc_batched_step(
        &model,
        &mut engine,
        &encoder,
        &encoder_params,
        &decoder,
        &decoder_params,
        &calib,
        &seqs,
        &cfg,
        None,
    )
    .unwrap();
    assert_eq!(got.windows.len(), refs.len());

    let mut worst = [0.0f64; 5];
    for (w, (g, r)) in got.windows.iter().zip(&refs).enumerate() {
        assert_eq!(g.weight_sum, r.weight_sum, "window {w}: weight_sum");
        assert!(r.activity_loss > 0.0, "the test band must penalise something");
        let loss_rel = |a: f32, b: f32| f64::from((a - b).abs()) / f64::from(b.abs()).max(1e-6);
        assert!(
            loss_rel(g.loss.total, r.loss.total) < 1e-4,
            "window {w}: loss {} vs {}",
            g.loss.total,
            r.loss.total
        );
        assert!(
            loss_rel(g.activity_loss, r.activity_loss) < 1e-4,
            "window {w}: activity"
        );
        assert_eq!(g.logits.len(), r.logits.len());
        for (a, b) in g.logits.iter().zip(&r.logits) {
            assert!(
                (a.hook - b.hook).abs() < 2e-4 && (a.dir[0] - b.dir[0]).abs() < 2e-4 && (a.jump - b.jump).abs() < 2e-4,
                "window {w}: {a:?} vs {b:?}"
            );
        }
        let e_g = rel_err(&g.encoder.g, &r.encoder.g);
        let e_c = rel_err(&g.encoder.c, &r.encoder.c);
        let e_h = rel_err(&g.decoder.hook_w, &r.decoder.hook_w);
        let e_d = rel_err(&g.decoder.direction_lr_w, &r.decoder.direction_lr_w);
        let e_a = rel_err(&g.decoder.aim_pair_theta, &r.decoder.aim_pair_theta);
        for (slot, e) in [e_g, e_c, e_h, e_d, e_a].into_iter().enumerate() {
            worst[slot.min(4)] = worst[slot.min(4)].max(e);
            assert!(e < 2e-3, "window {w}: gradient group {slot} off by {e}");
        }
    }
    // Connectome gradients, summed over the windows.
    let mut sum = ddai_fly::optim::ParamGradients::zeros_like(model.params());
    for r in &refs {
        sum.add_assign(&r.fly);
    }
    let (ea, eb, et) = (
        rel_err(&got.fly.a, &sum.a),
        rel_err(&got.fly.b, &sum.b),
        rel_err(&got.fly.theta, &sum.theta),
    );
    eprintln!(
        "{name}: connectome gradient error relative to group scale: a {ea:.2e}, b {eb:.2e}, theta {et:.2e}; \
         worst per-window encoder g/c {:.2e}/{:.2e}, decoder hook/dir/aim {:.2e}/{:.2e}/{:.2e}",
        worst[0], worst[1], worst[2], worst[3], worst[4]
    );
    assert!(
        ea < 2e-3 && eb < 2e-3 && et < 2e-3,
        "connectome gradients: a {ea}, b {eb}, theta {et}"
    );
}

#[test]
fn batched_bc_step_matches_per_window_bc_step_on_the_real_s_graph() {
    check_graph("S", "S-brain.toml", 0.5);
}

#[test]
fn batched_bc_step_matches_per_window_bc_step_on_the_real_m_graph() {
    check_graph("M", "M-brain.toml", 0.5);
}

/// The opt-in stop-gradient burn-in (`BatchedEngine::with_stop_grad_decisions`, task 7.2c) on the
/// real S graph. The forward side of a step does not change at all -- loss, logits, decoder
/// gradients are bitwise those of the full BPTT when the prefix decisions are the unscored
/// burn-in anyway -- while the connectome gradient is a (finite, different) truncated one, and a
/// prefix as long as the window leaves nothing to train.
#[test]
fn stop_gradient_burn_in_leaves_the_forward_side_of_a_bc_step_untouched() {
    let Some(mut fx) = fixture("S", "S-brain.toml", 0.2) else {
        return;
    };
    let prefix = 3usize;
    fx.cfg.activity.weight = 0.0; // the regulariser's taps in the prefix are dropped by design
    for seq in &mut fx.seqs {
        for t in seq.targets.iter_mut().take(prefix) {
            *t = StepTargets::burn_in();
        }
    }
    let run = |engine: &mut BatchedEngine| {
        brain_bc_batched_step(
            &fx.model,
            engine,
            &fx.encoder,
            &fx.encoder_params,
            &fx.decoder,
            &fx.decoder_params,
            &fx.calib,
            &fx.seqs,
            &fx.cfg,
            None,
        )
        .unwrap()
    };
    let full = run(&mut BatchedEngine::new(&fx.model));
    let stopped = run(&mut BatchedEngine::new(&fx.model).with_stop_grad_decisions(prefix));
    assert_eq!(full.windows.len(), stopped.windows.len());
    for (w, (a, b)) in full.windows.iter().zip(&stopped.windows).enumerate() {
        assert_eq!(a.loss.total, b.loss.total, "window {w}: loss bits");
        assert_eq!(a.weight_sum, b.weight_sum, "window {w}: weight_sum");
        assert_eq!(a.logits.len(), b.logits.len());
        for (x, y) in a.logits.iter().zip(&b.logits) {
            assert_eq!(
                (x.hook, x.dir, x.jump),
                (y.hook, y.dir, y.jump),
                "window {w}: logits bits"
            );
        }
        assert_eq!(a.decoder.hook_w, b.decoder.hook_w, "window {w}: decoder gradient bits");
        assert_eq!(a.decoder.direction_lr_w, b.decoder.direction_lr_w);
    }
    assert!(
        stopped
            .fly
            .a
            .iter()
            .chain(&stopped.fly.b)
            .chain(&stopped.fly.theta)
            .all(|x| x.is_finite())
    );
    assert_ne!(stopped.fly.b, full.fly.b, "a truncated gradient");
    assert!(stopped.fly.b.iter().any(|&x| x != 0.0));
    // Nothing is scored, nothing flows: a prefix covering every window.
    let longest = fx.seqs.iter().map(|s| s.observations.len()).max().unwrap();
    let none = run(&mut BatchedEngine::new(&fx.model).with_stop_grad_decisions(longest));
    assert!(
        none.fly
            .a
            .iter()
            .chain(&none.fly.b)
            .chain(&none.fly.theta)
            .all(|&x| x == 0.0)
    );
    assert!(none.windows.iter().all(|w| w.weight_sum == 0.0));
}
