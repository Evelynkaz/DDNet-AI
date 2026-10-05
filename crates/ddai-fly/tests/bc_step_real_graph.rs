//! `brain_bc_step` on the real S graph (task 8.2): finite-difference check of the gradients of
//! every trainable group (encoder `g`, fly `a`/`b`/`theta`, decoder hook weights and aim angles)
//! for a soft-target, weighted, masked window with the activity regulariser on, and equality of
//! the training pass's logits with the forward-only evaluation pass. Skipped (with a note) when
//! the compiled graph is not on this machine.

use std::sync::Arc;

use ddai_brain::{CharacterObservation, Observation};
use ddai_fly::backward::BackwardIndex;
use ddai_fly::bc::{HeadMask, LossConfig, SoftTargets, StepTargets};
use ddai_fly::brain_bc::{BcSequence, BcStepConfig, BcWorkspace, brain_bc_forward, brain_bc_step};
use ddai_fly::config::FlyConfig;
use ddai_fly::decoder::{DecoderModel, DnCalibration};
use ddai_fly::encoder::{EncoderModel, EncoderParams};
use ddai_fly::model::FlyModel;
use ddai_fly::optim::ActivityRegularizerConfig;
use ddai_fly::params::FlyParams;

fn real_s() -> Option<ddai_flyg::Flyg> {
    let home = std::env::var("HOME").ok()?;
    let path = std::path::PathBuf::from(home).join("aiddnet/data/connectome/compiled/fly-S-v1.flyg");
    path.exists().then(|| ddai_flyg::load(&path).expect("load S"))
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

fn largest(values: &[f32], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[b].abs().partial_cmp(&values[a].abs()).unwrap());
    idx.truncate(n);
    idx
}

#[test]
fn bc_step_gradients_match_finite_differences_on_the_real_s_graph() {
    let Some(flyg) = real_s() else {
        eprintln!("skipped: fly-S-v1.flyg not found");
        return;
    };
    let brain_cfg = ddai_fly::brain_config::load_brain_config(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/fly/S-brain.toml"),
    )
    .unwrap();
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 11);
    let mut model = FlyModel::new(flyg, config, params).unwrap();
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
    let map = Arc::new(open_map());
    let observations = vec![
        obs(&map, 760.0, 600.0, 60.0),
        obs(&map, 820.0, 660.0, 30.0),
        obs(&map, 700.0, 500.0, -10.0),
    ];
    let soft = SoftTargets {
        dir: [0.2, 0.3, 0.5],
        jump: 0.3,
        hook: 0.7,
        fire: 0.1,
    };
    let targets = vec![
        StepTargets::burn_in(),
        StepTargets {
            dir: 2,
            jump: false,
            hook: true,
            fire: false,
            aim: 0.4,
            soft: Some(soft),
            mask: HeadMask::ALL,
            weight: 1.5,
            hook_scale: 1.0,
        },
        StepTargets {
            dir: 0,
            jump: true,
            hook: false,
            fire: true,
            aim: -1.1,
            soft: None,
            mask: HeadMask {
                fire: false,
                ..HeadMask::ALL
            },
            weight: 1.0,
            hook_scale: 1.0,
        },
    ];
    let cfg = BcStepConfig {
        loss: LossConfig {
            soft_mix: 0.5,
            pos_weight: [2.0, 1.5, 1.0],
            ..LossConfig::default()
        },
        // A band the resting activity is partly outside, so the regulariser has gradient.
        activity: ActivityRegularizerConfig {
            weight: 0.05,
            low: 0.4,
            high: 0.8,
        },
    };
    let v_init = vec![0.0f32; model.num_neurons()];
    let seq = BcSequence {
        v_init: v_init.clone(),
        observations: observations.clone(),
        targets,
    };
    let mut ws = BcWorkspace::new(&model, &encoder, 8);
    let out = brain_bc_step(
        &model,
        &index,
        &encoder,
        &encoder_params,
        &decoder,
        &decoder_params,
        &calib,
        &seq,
        &cfg,
        &mut ws,
    );
    assert!((out.weight_sum - 2.5).abs() < 1e-6);
    assert!(
        out.activity_loss > 0.0,
        "the test band must actually penalise something"
    );

    // Forward-only evaluation must give the same logits as the training pass.
    let fwd = brain_bc_forward(
        &model,
        &encoder,
        &encoder_params,
        &decoder,
        &decoder_params,
        &calib,
        &v_init,
        &observations,
    );
    assert_eq!(fwd.len(), out.logits.len());
    for (a, b) in fwd.iter().zip(&out.logits) {
        assert!(
            (a.hook - b.hook).abs() < 1e-5 && (a.dir[0] - b.dir[0]).abs() < 1e-5,
            "{a:?} vs {b:?}"
        );
    }

    let total = |model: &FlyModel, ep: &EncoderParams, dp: &ddai_fly::decoder::DecoderParams| -> f64 {
        let mut ws = BcWorkspace::new(model, &encoder, 8);
        let o = brain_bc_step(model, &index, &encoder, ep, &decoder, dp, &calib, &seq, &cfg, &mut ws);
        f64::from(o.loss.total) + f64::from(o.activity_loss)
    };
    let check = |name: &str, analytic: &[f32], numeric: &dyn Fn(usize, f32) -> f64, eps: f32, n: usize| {
        let max_g = analytic.iter().fold(0.0f32, |m, &g| m.max(g.abs()));
        assert!(max_g > 0.0, "{name}: the gradient group is identically zero");
        for i in largest(analytic, n) {
            let num = ((numeric(i, eps) - numeric(i, -eps)) / (2.0 * f64::from(eps))) as f32;
            assert!(
                (analytic[i] - num).abs() < 0.05 * max_g + 1e-4,
                "{name}[{i}]: analytic {} vs numeric {num} (max |g| {max_g})",
                analytic[i]
            );
        }
    };
    check(
        "encoder.g",
        &out.encoder.g,
        &|i, e| {
            let mut p = encoder_params.clone();
            p.g[i] += e;
            total(&model, &p, &decoder_params)
        },
        2e-3,
        4,
    );
    check(
        "decoder.hook_w",
        &out.decoder.hook_w,
        &|i, e| {
            let mut p = decoder_params.clone();
            p.hook_w[i] += e;
            total(&model, &encoder_params, &p)
        },
        2e-3,
        3,
    );
    check(
        "decoder.aim_pair_theta",
        &out.decoder.aim_pair_theta,
        &|i, e| {
            let mut p = decoder_params.clone();
            p.aim_pair_theta[i] += e;
            total(&model, &encoder_params, &p)
        },
        2e-3,
        2,
    );
    let base = model.params().clone();
    let mut fly_check = |name: &str, analytic: &[f32], which: u8, n: usize| {
        let max_g = analytic.iter().fold(0.0f32, |m, &g| m.max(g.abs()));
        assert!(max_g > 0.0, "{name}: zero gradient");
        for i in largest(analytic, n) {
            let mut eval = |e: f32| {
                let mut p = base.clone();
                match which {
                    0 => p.a[i] += e,
                    1 => p.b[i] += e,
                    _ => p.theta[i] += e,
                }
                model.set_params(p).unwrap();
                total(&model, &encoder_params, &decoder_params)
            };
            let eps = 2e-3f32;
            let num = ((eval(eps) - eval(-eps)) / (2.0 * f64::from(eps))) as f32;
            assert!(
                (analytic[i] - num).abs() < 0.08 * max_g + 1e-4,
                "{name}[{i}]: analytic {} vs numeric {num} (max |g| {max_g})",
                analytic[i]
            );
        }
        model.set_params(base.clone()).unwrap();
    };
    fly_check("fly.a", &out.fly.a, 0, 4);
    fly_check("fly.b", &out.fly.b, 1, 3);
    fly_check("fly.theta", &out.fly.theta, 2, 3);
}
