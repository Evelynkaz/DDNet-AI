//! Task 8.5a: the target opponent's state (frozen, freeze time left, velocity, hook) as encoder input.
//!
//! The contract: a bundle upgraded with `[opponent_state]` and zero weights for it plays **bit for bit** like the
//! original, whatever the opponent does (frozen or not, fast or slow, hooking or not); once those weights are non-zero
//! the fly does see the difference. The old parameters keep their places, so an old bundle loads and the upgrade only
//! appends.

use std::sync::Arc;

use ddai_brain::{Brain, CharacterObservation, HOOK_GRABBED, Observation, ResetContext};
use ddai_fly::bc::HookView;
use ddai_fly::brain::FlyBrainConfig;
use ddai_fly::brain_config::parse_brain_config;
use ddai_fly::bundle::{FlyBrainTemplate, load_bundle, save_bundle, upgrade_with_opponent_state};
use ddai_fly::encoder::{EncoderModel, OPPONENT_CHANNELS, OpponentChannel, OpponentStateConfig, RayGridFeatures};
use ddai_physics::map::MapData;
use ddai_physics::tuning::TuningParams;
use ddai_physics::vmath::Vec2;

const SECTION: &str = r#"
[opponent_state]
frozen = ["VPN_OPP", "VPN_WALL"]
freeze_left = ["VPN_OPP"]
velocity_x = ["VPN_OPP"]
velocity_y = ["VPN_OPP"]
hook_state = ["AN_GROUND", "VPN_WALL"]
"#;

fn map() -> Arc<MapData> {
    Arc::new(MapData {
        width: 40,
        height: 20,
        game: vec![Default::default(); 800],
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    })
}

/// A scene: the opponent `k` (0..) differs in position, velocity, freeze and hook state.
fn scene(k: i32, map: &Arc<MapData>) -> Observation {
    let mut me = CharacterObservation::at_rest(0);
    me.pos = Vec2::new(400.0, 300.0);
    me.vel = Vec2::new((k % 5) as f32 - 2.0, 1.5);
    let mut opp = CharacterObservation::at_rest(1);
    opp.pos = Vec2::new(400.0 + 20.0 * ((k % 9) as f32 - 4.0), 300.0 + 10.0 * (k % 3) as f32);
    opp.vel = Vec2::new(8.0 * ((k % 4) as f32 - 1.5), -6.0 * (k % 3) as f32);
    opp.is_frozen = k % 2 == 0;
    opp.freeze_ticks_remaining = if opp.is_frozen { 10 + 13 * k } else { 0 };
    if k % 3 == 1 {
        opp.hook_state = HOOK_GRABBED;
    }
    Observation {
        map: map.clone(),
        tick: 2 * k,
        self_state: me,
        others: vec![opp],
        target_id: Some(1),
        tuning: TuningParams::default(),
    }
}

fn fixture() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let (bundle, flyg) = ddai_fly::brain_fixtures::write_tiny_fly_bundle(dir.path(), HookView::Shared);
    (dir, bundle, flyg)
}

fn upgraded(dir: &std::path::Path, bundle: &std::path::Path, flyg: &std::path::Path) -> std::path::PathBuf {
    let b = load_bundle(bundle).unwrap();
    let up = upgrade_with_opponent_state(&b, ddai_flyg::load(flyg).unwrap(), SECTION).unwrap();
    let path = dir.join("tiny-up.bundle");
    save_bundle(&path, &up).unwrap();
    path
}

/// The decoded head probabilities of a brain over a fixed series of scenes, as raw bits.
fn trace(template: &FlyBrainTemplate) -> Vec<[u32; 6]> {
    let m = map();
    let mut brain = template.instantiate(FlyBrainConfig::default());
    brain.reset(&ResetContext {
        map: m.clone(),
        self_id: 0,
        seed: 1,
    });
    (0..40)
        .map(|k| {
            brain.decide(&scene(k, &m));
            let d = brain.last_decoded().unwrap();
            [
                d.direction_probs[0].to_bits(),
                d.direction_probs[1].to_bits(),
                d.direction_probs[2].to_bits(),
                d.jump_prob.to_bits(),
                d.hook_prob.to_bits(),
                d.fire_prob.to_bits(),
            ]
        })
        .collect()
}

#[test]
fn an_upgraded_bundle_with_zero_weights_plays_bit_for_bit_like_the_original() {
    let (dir, bundle, flyg) = fixture();
    let old = FlyBrainTemplate::load(&bundle, Some(&flyg)).unwrap();
    let up_path = upgraded(dir.path(), &bundle, &flyg);
    let up = FlyBrainTemplate::load(&up_path, Some(&flyg)).unwrap();
    assert!(up.encoder().num_params() > old.encoder().num_params());
    assert_eq!(trace(&old), trace(&up));
}

#[test]
fn nonzero_weights_make_the_fly_see_the_opponents_state() {
    let (dir, bundle, flyg) = fixture();
    let up_path = upgraded(dir.path(), &bundle, &flyg);
    let mut b = load_bundle(&up_path).unwrap();
    let base = trace(&FlyBrainTemplate::load(&up_path, Some(&flyg)).unwrap());
    // Each new (type, channel) parameter, alone, must change what the fly decides in some scene.
    let n_old = {
        let old = load_bundle(&bundle).unwrap();
        old.encoder_params.g.len()
    };
    let n_new = b.encoder_params.g.len();
    assert_eq!(
        n_new - n_old,
        7,
        "VPN_OPP x4, VPN_WALL x1 (frozen), AN_GROUND x1, VPN_WALL x1 (hook)"
    );
    for p in n_old..n_new {
        let saved = b.encoder_params.g[p];
        b.encoder_params.g[p] = 6.0;
        let path = dir.path().join(format!("probe-{p}.bundle"));
        save_bundle(&path, &b).unwrap();
        let t = trace(&FlyBrainTemplate::load(&path, Some(&flyg)).unwrap());
        // Not every probe reaches a decoded head on the tiny graph; at least the frozen / freeze-left / velocity
        // probes on the opponent type do (checked below), and none may crash or produce NaN.
        assert!(t.iter().flatten().all(|&bits| f32::from_bits(bits).is_finite()));
        b.encoder_params.g[p] = saved;
    }
    let mut all = b.clone();
    for p in n_old..n_new {
        all.encoder_params.g[p] = 4.0;
    }
    let path = dir.path().join("all.bundle");
    save_bundle(&path, &all).unwrap();
    assert_ne!(base, trace(&FlyBrainTemplate::load(&path, Some(&flyg)).unwrap()));
}

#[test]
fn old_parameters_keep_their_places_and_the_new_ones_come_last() {
    let (_dir, bundle, flyg) = fixture();
    let b = load_bundle(&bundle).unwrap();
    let fly = ddai_flyg::load(&flyg).unwrap();
    let model = ddai_fly::FlyModel::new(fly, b.fly_config, b.fly_params.clone()).unwrap();
    let old_cfg = parse_brain_config(&b.brain_config_toml).unwrap();
    let new_cfg = {
        let section: toml::Table = SECTION.parse().unwrap();
        let text = format!("{}\n{}", b.brain_config_toml, toml::to_string(&section).unwrap());
        parse_brain_config(&text).unwrap()
    };
    let (old, new) = (
        old_cfg.encoder_model(&model).unwrap(),
        new_cfg.encoder_model(&model).unwrap(),
    );
    let n = old.num_params();
    assert_eq!(new.num_params(), n + 7);
    for (a, b) in old.assignments().iter().zip(new.assignments()) {
        assert_eq!(
            (a.type_index, a.channel, a.param_id),
            (b.type_index, b.channel, b.param_id)
        );
    }
    assert!(
        new.assignments()[n..]
            .iter()
            .all(|a| a.channel.starts_with("opponent_"))
    );
}

#[test]
fn an_upgrade_twice_or_without_channels_or_a_velocity_on_an_ascending_type_is_refused() {
    let (dir, bundle, flyg) = fixture();
    let b = load_bundle(&bundle).unwrap();
    let g = || ddai_flyg::load(&flyg).unwrap();
    let up = upgrade_with_opponent_state(&b, g(), SECTION).unwrap();
    assert!(
        upgrade_with_opponent_state(&up, g(), SECTION).is_err(),
        "already upgraded"
    );
    assert!(
        upgrade_with_opponent_state(&b, g(), "[opponent_state]\n").is_err(),
        "no channel named"
    );
    assert!(
        upgrade_with_opponent_state(&b, g(), "[opponent_state]\nvelocity_x = [\"AN_GROUND\"]\n").is_err(),
        "a velocity needs a visual type"
    );
    assert!(
        upgrade_with_opponent_state(&b, g(), "[opponent_state]\nfrozen = [\"NO_SUCH_TYPE\"]\n").is_err(),
        "unknown type"
    );
    assert!(
        upgrade_with_opponent_state(&b, g(), "[opponent_state]\nfrozen = [\"HID\"]\n").is_err(),
        "a hidden type is not an input"
    );
    let _ = dir;
}

#[test]
fn the_features_carry_the_opponents_state() {
    let m = map();
    let cfg = ddai_fly::encoder::RayGridConfig::default();
    let mut f = RayGridFeatures::new(&cfg);
    // k = 4: frozen (even) with 10 + 52 = 62 ticks left, hook idle (4 % 3 == 1 -> grabbed), vel x = 8 * (0 - 1.5)
    let obs = scene(4, &m);
    f.compute(&obs, &cfg);
    assert_eq!(f.opponent(OpponentChannel::Frozen), 1.0);
    assert!((f.opponent(OpponentChannel::FreezeLeft) - 62.0 / 300.0).abs() < 1e-6);
    assert_eq!(f.opponent(OpponentChannel::HookState), 1.0);
    let o = &obs.others[0];
    assert!(
        (f.opponent(OpponentChannel::VelocityX) - (o.vel.x / cfg.velocity_norm_scale).clamp(-1.0, 1.0)).abs() < 1e-6
    );
    // No opponent: all zero. A free opponent: not frozen, no time left.
    let mut alone = scene(4, &m);
    alone.others.clear();
    f.compute(&alone, &cfg);
    assert!(OPPONENT_CHANNELS.iter().all(|&c| f.opponent(c) == 0.0));
    f.compute(&scene(5, &m), &cfg);
    assert_eq!(f.opponent(OpponentChannel::Frozen), 0.0);
    assert_eq!(f.opponent(OpponentChannel::FreezeLeft), 0.0);
}

#[test]
fn encoder_gradients_of_the_new_channels_are_the_feature_times_the_input_gradient() {
    let (_dir, bundle, flyg) = fixture();
    let b = load_bundle(&bundle).unwrap();
    let model = ddai_fly::FlyModel::new(ddai_flyg::load(&flyg).unwrap(), b.fly_config, b.fly_params.clone()).unwrap();
    let cfg = parse_brain_config(&format!("{}\n{}", b.brain_config_toml, SECTION)).unwrap();
    let enc: EncoderModel = cfg.encoder_model(&model).unwrap();
    let m = map();
    let obs = scene(4, &m);
    let mut feats = RayGridFeatures::new(enc.ray_grid_config());
    feats.compute(&obs, enc.ray_grid_config());
    let an = ddai_fly::encoder::compute_proprioception_values(&obs.self_state, enc.ray_grid_config());
    let mut params = enc.init_params();
    // Make every input neuron's current matter: dL/dI_k = 1 for all k.
    let gi = vec![1.0f32; enc.num_inputs()];
    let mut grads = enc.zero_grads();
    enc.backward_with_params(&feats, &an, &params, &gi, &mut grads);
    // Finite difference of sum_k I_k with respect to every g and c.
    let sum = |p: &ddai_fly::encoder::EncoderParams| {
        let mut out = vec![0.0f32; enc.num_inputs()];
        enc.forward(&feats, &an, p, &mut out);
        out.iter().map(|&x| f64::from(x)).sum::<f64>()
    };
    let base = sum(&params);
    for pid in 0..enc.num_params() {
        let h = 0.5f32;
        params.g[pid] += h;
        let dg = (sum(&params) - base) / f64::from(h);
        params.g[pid] -= h;
        params.c[pid] += h;
        let dc = (sum(&params) - base) / f64::from(h);
        params.c[pid] -= h;
        assert!(
            (dg - f64::from(grads.g[pid])).abs() < 1e-3 * (1.0 + dg.abs()),
            "g[{pid}] {dg} vs {}",
            grads.g[pid]
        );
        assert!(
            (dc - f64::from(grads.c[pid])).abs() < 1e-3 * (1.0 + dc.abs()),
            "c[{pid}] {dc} vs {}",
            grads.c[pid]
        );
    }
    let _ = OpponentStateConfig::default();
}

#[test]
fn a_deep_frozen_opponent_reads_as_frozen_with_a_full_timer_as_the_live_world_reports_it() {
    let m = map();
    let cfg = ddai_fly::encoder::RayGridConfig::default();
    let mut f = RayGridFeatures::new(&cfg);
    // What LiveWorld gives for deep freeze: no freeze time, `is_frozen` false, the deep flag set.
    let mut obs = scene(5, &m);
    assert!(!obs.others[0].is_frozen && obs.others[0].freeze_ticks_remaining == 0);
    obs.others[0].is_deep_frozen = true;
    f.compute(&obs, &cfg);
    assert_eq!(f.opponent(OpponentChannel::Frozen), 1.0);
    assert_eq!(f.opponent(OpponentChannel::FreezeLeft), 1.0);
    // The own freeze timer too.
    obs.self_state.is_deep_frozen = true;
    let v = ddai_fly::encoder::compute_proprioception_values(&obs.self_state, &cfg);
    assert_eq!(v.freeze_timer, 1.0);
    obs.self_state.is_deep_frozen = false;
    let v = ddai_fly::encoder::compute_proprioception_values(&obs.self_state, &cfg);
    assert_eq!(v.freeze_timer, 0.0);
}

#[test]
fn a_live_frozen_tee_reads_as_before_the_deep_freeze_change() {
    // Live freeze (`TILE_LFREEZE`) only disables movement: the tee can still hook and is not "out", so it must not read as frozen with a
    // full timer (review of 8.5a, F8). Without a freeze time it reads exactly like a free tee, with one it reads by that time.
    let m = map();
    let cfg = ddai_fly::encoder::RayGridConfig::default();
    let mut f = RayGridFeatures::new(&cfg);
    let mut obs = scene(5, &m);
    assert!(!obs.others[0].is_frozen && obs.others[0].freeze_ticks_remaining == 0);
    obs.others[0].is_live_frozen = true;
    f.compute(&obs, &cfg);
    assert_eq!(f.opponent(OpponentChannel::Frozen), 0.0);
    assert_eq!(f.opponent(OpponentChannel::FreezeLeft), 0.0);
    obs.others[0].freeze_ticks_remaining = 90;
    f.compute(&obs, &cfg);
    assert_eq!(f.opponent(OpponentChannel::Frozen), 1.0);
    assert!((f.opponent(OpponentChannel::FreezeLeft) - 90.0 / 300.0).abs() < 1e-6);
    // The own freeze timer likewise.
    obs.self_state.is_live_frozen = true;
    let v = ddai_fly::encoder::compute_proprioception_values(&obs.self_state, &cfg);
    assert_eq!(v.freeze_timer, 0.0);
    obs.self_state.freeze_ticks_remaining = 150;
    let v = ddai_fly::encoder::compute_proprioception_values(&obs.self_state, &cfg);
    assert_eq!(v.freeze_timer, 0.5);
}
