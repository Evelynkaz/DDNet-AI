//! Task 8.6: the hysteresis decode, the intent hook head and bundle format v4.
//!
//! What these tests pin down:
//! * a v3 bundle loads as `legacy` / `plain` and plays **bit for bit** like the same weights saved as v4, and like their intent upgrade
//!   (the release hazard is the mirror image of the hook head, so the two hazards together are the legacy head);
//! * the hysteresis decode is latched on the fly's **own previous hook command**, not on the observed hook state;
//! * the latch selects a hazard and never reaches the network: with the latch flipped the DN rates and every other head are
//!   bit-identical ("no copying", the trap of E-005 F2), and only the selected hazard gets a gradient.

use std::sync::Arc;

use ddai_brain::{Brain, CharacterObservation, Observation, ResetContext};
use serde::Serialize;

use crate::bc::{HookDecode, HookParam, HookView};
use crate::brain::{ActionSelection, FlyBrainConfig};
use crate::brain_fixtures::write_tiny_fly_bundle;
use crate::bundle::{
    FlyBrainTemplate, FlyBundle, load_bundle, save_bundle, upgrade_to_intent_hook, write_zstd_postcard,
};
use crate::decoder::HookRelease;

fn tiny_map() -> Arc<ddai_physics::map::MapData> {
    Arc::new(ddai_physics::map::MapData {
        width: 20,
        height: 20,
        game: vec![Default::default(); 400],
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    })
}

fn obs(map: &Arc<ddai_physics::map::MapData>, tick: i32, opp_x: f32, hook_state: i32) -> Observation {
    let mut me = CharacterObservation::at_rest(0);
    me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
    me.hook_state = hook_state;
    let mut opp = CharacterObservation::at_rest(1);
    opp.pos = ddai_physics::vmath::Vec2::new(opp_x, 300.0);
    Observation {
        map: map.clone(),
        tick,
        self_state: me,
        others: vec![opp],
        target_id: None,
        tuning: ddai_physics::tuning::TuningParams::default(),
    }
}

fn reset_ctx(map: &Arc<ddai_physics::map::MapData>) -> ResetContext {
    ResetContext {
        map: map.clone(),
        self_id: 0,
        seed: 3,
    }
}

fn brain_config() -> FlyBrainConfig {
    FlyBrainConfig {
        action_selection: ActionSelection::Argmax,
        seed: 1,
    }
}

/// The bundle layout of format version 3, as it was written before task 8.6 (for the loader test).
#[derive(Serialize)]
struct V3Out<'a> {
    format_version: u32,
    flyg_sha256: &'a str,
    flyg_path_hint: &'a str,
    brain_config_toml: &'a str,
    fly_config: &'a crate::config::FlyConfig,
    fly_params: &'a crate::params::FlyParams,
    encoder_params: &'a crate::encoder::EncoderParams,
    decoder_params: crate::decoder::DecoderParamsV3,
    calibration: &'a crate::decoder::DnCalibration,
    meta: &'a crate::bundle::BundleMeta,
    thresholds: &'a crate::bc::HeadThresholds,
    hook_view: HookView,
}

fn write_v3(path: &std::path::Path, b: &FlyBundle) {
    let d = &b.decoder_params;
    assert!(d.hook_release.is_none(), "a v3 file has no release hazard");
    let v3 = V3Out {
        format_version: 3,
        flyg_sha256: &b.flyg_sha256,
        flyg_path_hint: &b.flyg_path_hint,
        brain_config_toml: &b.brain_config_toml,
        fly_config: &b.fly_config,
        fly_params: &b.fly_params,
        encoder_params: &b.encoder_params,
        decoder_params: crate::decoder::DecoderParamsV3 {
            direction_lr_w: d.direction_lr_w.clone(),
            direction_lr_b: d.direction_lr_b,
            direction_stop_w: d.direction_stop_w.clone(),
            direction_stop_b: d.direction_stop_b,
            jump_w: d.jump_w.clone(),
            jump_b: d.jump_b,
            hook_w: d.hook_w.clone(),
            hook_b: d.hook_b,
            fire_w: d.fire_w.clone(),
            fire_b: d.fire_b,
            aim_pair_theta: d.aim_pair_theta.clone(),
            aim_unpaired_theta: d.aim_unpaired_theta.clone(),
        },
        calibration: &b.calibration,
        meta: &b.meta,
        thresholds: &b.thresholds,
        hook_view: b.hook_view,
    };
    write_zstd_postcard(path, &v3, 3).unwrap();
}

/// What a played brain does over a fixed script of observations (the opponent moves, the own hook state cycles through every state
/// whatever the brain did): the action and the bits of every head probability of the full view.
fn scripted_state(i: i32) -> i32 {
    let states = [
        ddai_brain::HOOK_IDLE,
        ddai_brain::HOOK_FLYING,
        ddai_brain::HOOK_GRABBED,
        ddai_brain::HOOK_RETRACT_START,
        ddai_brain::HOOK_IDLE,
        ddai_brain::HOOK_IDLE,
    ];
    states[i as usize % states.len()]
}

fn play(template: &FlyBrainTemplate, map: &Arc<ddai_physics::map::MapData>) -> Vec<(ddai_brain::Action, [u32; 2])> {
    let mut brain = template.instantiate_played(brain_config());
    brain.reset(&reset_ctx(map));
    (0..60)
        .map(|i| {
            let o = obs(map, 2 * i, 250.0 + 7.0 * (i % 23) as f32, scripted_state(i));
            let a = brain.decide(&o);
            // The telemetry JSON prints every head probability; its last field is the wall-clock latency, which is not a decision.
            let t = brain.telemetry().expect("telemetry");
            let t = t.rsplit_once(",\"latency_us\"").expect("a latency field").0.to_string();
            let h = |s: &str| {
                let mut x = 0xcbf2_9ce4_8422_2325u64;
                for b in s.bytes() {
                    x = (x ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
                }
                x
            };
            let hh = h(&t);
            (a, [hh as u32, (hh >> 32) as u32])
        })
        .collect()
}

fn template_of(path: &std::path::Path, flyg: &std::path::Path) -> FlyBrainTemplate {
    FlyBrainTemplate::load(path, Some(flyg)).unwrap()
}

#[test]
fn a_v3_bundle_loads_as_legacy_plain_and_plays_bit_for_bit_like_v4_legacy_and_the_intent_upgrade() {
    let map = tiny_map();
    for view in [HookView::Shared, HookView::MaskedForHookHead] {
        let dir = tempfile::tempdir().unwrap();
        let (v4_path, flyg) = write_tiny_fly_bundle(dir.path(), view);
        let v4 = load_bundle(&v4_path).unwrap();
        assert_eq!((v4.hook_param, v4.hook_decode), (HookParam::Legacy, HookDecode::Plain));

        // The same weights as a version 3 file: it loads as legacy / plain, equal to the v4 bundle in every field.
        let v3_path = dir.path().join("v3.bundle");
        write_v3(&v3_path, &v4);
        let v3 = load_bundle(&v3_path).unwrap();
        assert_eq!(
            v3, v4,
            "a v3 file loads as the same bundle (legacy hook head, plain decode)"
        );

        // And plays the same: v3 file, v4 file, and the intent upgrade (press = the hook head, release = its mirror image).
        let intent = upgrade_to_intent_hook(&v4);
        assert_eq!(intent.hook_param, HookParam::Intent);
        assert!(intent.decoder_params.hook_release.is_some());
        let intent_path = dir.path().join("intent.bundle");
        save_bundle(&intent_path, &intent).unwrap();
        // A threshold at the middle of the probabilities this script gives (the tiny fly's hook probability is ~0.3 with the hook hidden,
        // so the default 0.5 would never press in the two-view play), so that both keys occur.
        let probs = |t: &FlyBrainTemplate| -> Vec<f32> {
            let mut brain = t.instantiate_played(brain_config());
            brain.reset(&reset_ctx(&map));
            (0..60)
                .map(|i| {
                    let _ = brain.decide(&obs(&map, 2 * i, 250.0 + 7.0 * (i % 23) as f32, scripted_state(i)));
                    let j = brain.telemetry().unwrap();
                    let at = j.find("\"hook_prob\":").unwrap() + "\"hook_prob\":".len();
                    j[at..].split([',', '}']).next().unwrap().parse().unwrap()
                })
                .collect()
        };
        let mut sorted = probs(&template_of(&v4_path, &flyg));
        sorted.sort_by(f32::total_cmp);
        let mid = 0.5 * (sorted[29] + sorted[30]);
        assert!(
            sorted[0] < mid && mid < sorted[59],
            "the probabilities of the script vary: {sorted:?}"
        );
        let mut v4 = v4;
        v4.thresholds.hook = mid;
        save_bundle(&v4_path, &v4).unwrap();
        write_v3(&v3_path, &v4);
        let intent = upgrade_to_intent_hook(&v4);
        save_bundle(&intent_path, &intent).unwrap();
        let reference = play(&template_of(&v4_path, &flyg), &map);
        assert!(
            reference.iter().any(|(a, _)| a.hook) && reference.iter().any(|(a, _)| !a.hook),
            "the script must exercise both hook keys ({view:?})"
        );
        assert_eq!(
            play(&template_of(&v3_path, &flyg), &map),
            reference,
            "v3 vs v4, {view:?}"
        );
        assert_eq!(
            play(&template_of(&intent_path, &flyg), &map),
            reference,
            "intent vs legacy, {view:?}"
        );
    }
}

#[test]
fn a_bundle_whose_hook_kind_disagrees_with_its_parameters_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (path, flyg) = write_tiny_fly_bundle(dir.path(), HookView::Shared);
    let mut b = load_bundle(&path).unwrap();
    b.hook_param = HookParam::Intent; // but no release hazard
    let bad = dir.path().join("bad.bundle");
    save_bundle(&bad, &b).unwrap();
    assert!(load_bundle(&bad).unwrap_err().0.contains("no release hazard"));
    let mut b = upgrade_to_intent_hook(&load_bundle(&path).unwrap());
    b.hook_param = HookParam::Legacy; // but a release hazard
    save_bundle(&bad, &b).unwrap();
    assert!(load_bundle(&bad).unwrap_err().0.contains("has a release hazard"));
    let mut b = load_bundle(&path).unwrap();
    b.hook_decode = HookDecode::Latched { hi: 0.4, lo: 1.0 }; // not a probability
    save_bundle(&bad, &b).unwrap();
    assert!(load_bundle(&bad).unwrap_err().0.contains("not inside"));
    let _ = flyg;
}

/// A script of observations whose own hook state swings between idle and grabbed, so that the tiny fly's hook probability (it reads
/// DN_HOOK, which the own-hook input drives: ~0.3 idle, ~0.8 grabbed in the fixture) swings through any band.
fn swing_script(map: &Arc<ddai_physics::map::MapData>) -> Vec<Observation> {
    (0..48)
        .map(|i| {
            let state = if (i / 6) % 2 == 0 {
                ddai_brain::HOOK_GRABBED
            } else {
                ddai_brain::HOOK_IDLE
            };
            obs(map, 2 * i, 400.0, state)
        })
        .collect()
}

/// `(hook probability, hook key)` of a template's single-view brain over the script.
fn run_script(
    template: &FlyBrainTemplate,
    script: &[Observation],
    map: &Arc<ddai_physics::map::MapData>,
) -> Vec<(f32, bool)> {
    let mut brain = template.instantiate(brain_config());
    brain.reset(&reset_ctx(map));
    script
        .iter()
        .map(|o| {
            let a = brain.decide(o);
            (brain.last_decoded().unwrap().hook_prob, a.hook)
        })
        .collect()
}

#[test]
fn the_hysteresis_decode_is_latched_on_the_own_previous_command() {
    let map = tiny_map();
    let dir = tempfile::tempdir().unwrap();
    let (path, flyg) = write_tiny_fly_bundle(dir.path(), HookView::Shared);
    let with = |decode: HookDecode, hook_threshold: f32| {
        let mut b = load_bundle(&path).unwrap();
        b.hook_decode = decode;
        b.thresholds.hook = hook_threshold;
        let p = dir.path().join("x.bundle");
        save_bundle(&p, &b).unwrap();
        template_of(&p, &flyg)
    };
    let script = swing_script(&map);
    let plain = run_script(&with(HookDecode::Plain, 0.6), &script, &map);
    let probs: Vec<f32> = plain.iter().map(|x| x.0).collect();
    let (lo_p, hi_p) = probs
        .iter()
        .skip(4)
        .fold((1.0f32, 0.0f32), |(a, b), &p| (a.min(p), b.max(p)));
    assert!(
        lo_p < 0.45 && hi_p > 0.65,
        "the script must swing the probability across the band: {lo_p}..{hi_p}"
    );
    let (hi, lo) = (0.65f32, 0.4f32);

    // The reference: the rule written out on the probabilities (they do not depend on the decode: the key never reaches the network).
    let mut latch = false;
    let reference: Vec<bool> = probs
        .iter()
        .map(|&p| {
            latch = p >= if latch { lo } else { hi };
            latch
        })
        .collect();
    let latched = run_script(&with(HookDecode::Latched { hi, lo }, 0.6), &script, &map);
    assert_eq!(
        latched.iter().map(|x| x.0.to_bits()).collect::<Vec<_>>(),
        probs.iter().map(|p| p.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(latched.iter().map(|x| x.1).collect::<Vec<_>>(), reference);
    // It is a genuine hysteresis: neither plain threshold gives the same keys (it holds through the dips between lo and hi, and does not press in the band).
    for t in [lo, hi] {
        let keys: Vec<bool> = probs.iter().map(|&p| p >= t).collect();
        assert_ne!(keys, reference, "plain {t} must differ from the hysteresis");
    }
    assert!(reference.iter().any(|&k| k) && reference.iter().any(|&k| !k));
    // lo = hi is the plain rule.
    let same = run_script(&with(HookDecode::Latched { hi: 0.6, lo: 0.6 }, 0.6), &script, &map);
    assert_eq!(
        same.iter().map(|x| x.1).collect::<Vec<_>>(),
        probs.iter().map(|&p| p >= 0.6).collect::<Vec<_>>()
    );
}

#[test]
fn the_latch_is_the_own_command_not_the_observed_hook_state() {
    let map = tiny_map();
    let dir = tempfile::tempdir().unwrap();
    let (path, flyg) = write_tiny_fly_bundle(dir.path(), HookView::Shared);
    let mut b = load_bundle(&path).unwrap();
    // The fly always sees an idle hook (probability ~0.3); the band [0.2, 0.9] holds a pressed key but never starts one.
    b.hook_decode = HookDecode::Latched { hi: 0.9, lo: 0.2 };
    let p = dir.path().join("x.bundle");
    save_bundle(&p, &b).unwrap();
    let t = template_of(&p, &flyg);
    let idle = |i: i32| obs(&map, 2 * i, 400.0, ddai_brain::HOOK_IDLE);
    let mut released = t.instantiate(brain_config());
    released.reset(&reset_ctx(&map));
    assert!((0..10).all(|i| !released.decide(&idle(i)).hook), "never starts: p < hi");
    let mut held = t.instantiate(brain_config());
    held.reset(&reset_ctx(&map));
    held.set_hook_latch(true); // as if its last command had been a press
    assert!(
        (0..10).all(|i| held.decide(&idle(i)).hook),
        "keeps holding: p >= lo, though the observed state says idle"
    );
    assert!(held.hook_latch());
    held.reset(&reset_ctx(&map));
    assert!(!held.hook_latch(), "a reset clears the latch");
}

#[test]
fn the_latch_selects_the_hazard_and_never_reaches_the_network() {
    let map = tiny_map();
    let dir = tempfile::tempdir().unwrap();
    let (path, flyg) = write_tiny_fly_bundle(dir.path(), HookView::Shared);
    let mut b = upgrade_to_intent_hook(&load_bundle(&path).unwrap());
    // Make the release hazard its own head, no longer the mirror image of the press hazard.
    b.decoder_params.hook_release = Some(HookRelease { w: vec![-1.5], b: 0.7 });
    let p = dir.path().join("intent.bundle");
    save_bundle(&p, &b).unwrap();
    let t = template_of(&p, &flyg);
    let o = obs(&map, 0, 400.0, ddai_brain::HOOK_GRABBED);
    let run = |latch: bool| {
        let mut brain = t.instantiate(brain_config());
        brain.reset(&reset_ctx(&map));
        brain.set_hook_latch(latch);
        let l = brain.forward_logits(&o);
        (l, brain.state_v().to_vec())
    };
    let (released, v_released) = run(false);
    let (held, v_held) = run(true);
    // The network state (and so the DN rates every head reads) is bit-identical: the latch is not an input of the network.
    assert_eq!(v_released, v_held);
    let bits = |l: &crate::bc::HeadLogits| {
        (
            l.dir.map(f32::to_bits),
            l.jump.to_bits(),
            l.fire.to_bits(),
            l.aim_c.to_bits(),
            l.aim_s.to_bits(),
        )
    };
    assert_eq!(
        bits(&released),
        bits(&held),
        "every head but the hook is the same bit for bit"
    );
    assert_ne!(released.hook, held.hook, "the hook head picks its hazard by the latch");
    // The press hazard is the hook head, the release hazard its own: hold logit = minus the release logit.
    let z_hook = {
        // DN z of the hook group = the hazards' common input; recover it from the press hazard (weight 3.0, bias -2.0 in the fixture).
        (released.hook - (-2.0)) / 3.0
    };
    let want_hold = -(0.7 + (-1.5) * z_hook);
    assert!((held.hook - want_hold).abs() < 1e-4, "{} vs {want_hold}", held.hook);
}

/// A `Legacy` + `Plain` bundle is written as version 3 (older binaries keep reading it); an intent head or a latched decode as version 4. A
/// version 3 file that is loaded and saved again comes back **byte for byte** the same.
#[test]
fn legacy_plain_bundles_are_written_as_v3_and_a_loaded_v3_file_is_saved_back_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _flyg) = write_tiny_fly_bundle(dir.path(), HookView::MaskedForHookHead);
    let version_of = |p: &std::path::Path| {
        let bytes = crate::bundle::read_zstd_bytes(p).unwrap();
        crate::bundle::peek_version(p, &bytes).unwrap()
    };
    assert_eq!(version_of(&path), 3, "a legacy / plain bundle is a v3 file");
    // A v3 file written by the old layout (this test's mirror of it) loads and saves back to the same bytes.
    let original = load_bundle(&path).unwrap();
    let old = dir.path().join("old-v3.bundle");
    write_v3(&old, &original);
    let again = dir.path().join("again.bundle");
    save_bundle(&again, &load_bundle(&old).unwrap()).unwrap();
    assert_eq!(
        std::fs::read(&old).unwrap(),
        std::fs::read(&again).unwrap(),
        "byte-identical"
    );
    assert_eq!(std::fs::read(&old).unwrap(), std::fs::read(&path).unwrap());
    // The new kinds are v4 and round-trip.
    let intent = upgrade_to_intent_hook(&original);
    let (p_intent, p_latched) = (dir.path().join("i.bundle"), dir.path().join("l.bundle"));
    save_bundle(&p_intent, &intent).unwrap();
    let mut latched = original.clone();
    latched.hook_decode = HookDecode::Latched { hi: 0.7, lo: 0.4 };
    save_bundle(&p_latched, &latched).unwrap();
    assert_eq!((version_of(&p_intent), version_of(&p_latched)), (4, 4));
    assert_eq!(load_bundle(&p_intent).unwrap(), intent);
    assert_eq!(load_bundle(&p_latched).unwrap(), latched);
}

/// The hybrid's proposer, the live bot's loader and `es critical` refuse a fly whose hook depends on the latch (review F3): the latch would
/// follow the fly's own command, not the action really played.
#[test]
fn a_latch_dependent_fly_is_refused_where_the_played_hook_is_not_its_own() {
    let dir = tempfile::tempdir().unwrap();
    let (path, flyg) = write_tiny_fly_bundle(dir.path(), HookView::Shared);
    let plain = template_of(&path, &flyg);
    assert!(plain.require_unlatched("test").is_ok());
    assert!(crate::proposer::FlyProposer::from_template(&plain, brain_config(), 1).is_ok());
    let mut latched = load_bundle(&path).unwrap();
    latched.hook_decode = HookDecode::Latched { hi: 0.7, lo: 0.4 };
    let intent = upgrade_to_intent_hook(&load_bundle(&path).unwrap());
    for (name, b) in [("latched", latched), ("intent", intent)] {
        let p = dir.path().join(format!("{name}.bundle"));
        save_bundle(&p, &b).unwrap();
        let t = template_of(&p, &flyg);
        let e = t.require_unlatched("test").unwrap_err().0;
        assert!(e.contains("can diverge"), "{name}: {e}");
        assert!(
            crate::proposer::FlyProposer::from_template(&t, brain_config(), 1).is_err(),
            "{name}"
        );
    }
}
