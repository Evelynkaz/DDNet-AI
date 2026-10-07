//! Task 8.7: the wide hook readout and bundle format v5.
//!
//! * a v4 file loads as the `pooled` readout and plays **bit for bit** like the same weights saved as v5;
//! * a wide readout with zero output weights (what `upgrade_hook_readout` makes) plays bit for bit like the bundle it came from, in
//!   both views; with non-zero weights it moves the hook probability and nothing else;
//! * a v5 bundle with a wide readout round-trips through the file and plays the same;
//! * a readout that disagrees with its parameters, or is combined with an intent hook head, is refused.

use std::sync::Arc;

use serde::Serialize;

use crate::bc::{HookDecode, HookParam, HookView};
use crate::brain_fixtures::write_tiny_fly_bundle;
use crate::bundle::{
    FlyBrainTemplate, FlyBundle, load_bundle, save_bundle, upgrade_hook_readout, upgrade_to_intent_hook,
    write_zstd_postcard,
};
use crate::hook_intent_tests::{brain_config, obs, play, reset_ctx, scripted_state, template_of, tiny_map};
use crate::hook_wide::HookReadout;

/// The bundle layout of format version 4, as it was written before task 8.7.
#[derive(Serialize)]
struct V4Out<'a> {
    format_version: u32,
    flyg_sha256: &'a str,
    flyg_path_hint: &'a str,
    brain_config_toml: &'a str,
    fly_config: &'a crate::config::FlyConfig,
    fly_params: &'a crate::params::FlyParams,
    encoder_params: &'a crate::encoder::EncoderParams,
    decoder_params: crate::decoder::DecoderParamsV4,
    calibration: &'a crate::decoder::DnCalibration,
    meta: &'a crate::bundle::BundleMeta,
    thresholds: &'a crate::bc::HeadThresholds,
    hook_view: HookView,
    hook_param: HookParam,
    hook_decode: HookDecode,
}

fn write_v4(path: &std::path::Path, b: &FlyBundle) {
    let d = &b.decoder_params;
    assert!(d.hook_wide.is_none());
    let v4 = V4Out {
        format_version: 4,
        flyg_sha256: &b.flyg_sha256,
        flyg_path_hint: &b.flyg_path_hint,
        brain_config_toml: &b.brain_config_toml,
        fly_config: &b.fly_config,
        fly_params: &b.fly_params,
        encoder_params: &b.encoder_params,
        decoder_params: crate::decoder::DecoderParamsV4 {
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
            hook_release: d.hook_release.clone(),
        },
        calibration: &b.calibration,
        meta: &b.meta,
        thresholds: &b.thresholds,
        hook_view: b.hook_view,
        hook_param: b.hook_param,
        hook_decode: b.hook_decode,
    };
    write_zstd_postcard(path, &v4, 3).unwrap();
}

/// Sets the hook threshold in the middle of the probabilities the script gives, so both keys occur (as in the intent tests).
fn with_mid_threshold(
    path: &std::path::Path,
    flyg: &std::path::Path,
    map: &Arc<ddai_physics::map::MapData>,
) -> FlyBundle {
    let mut b = load_bundle(path).unwrap();
    let t = template_of(path, flyg);
    let mut brain = t.instantiate_played(brain_config());
    brain.reset(&reset_ctx(map));
    let mut probs: Vec<f32> = (0..60)
        .map(|i| {
            let _ = brain.decide(&obs(map, 2 * i, 250.0 + 7.0 * (i % 23) as f32, scripted_state(i)));
            let j = brain.telemetry().unwrap();
            let at = j.find("\"hook_prob\":").unwrap() + "\"hook_prob\":".len();
            j[at..].split([',', '}']).next().unwrap().parse().unwrap()
        })
        .collect();
    probs.sort_by(f32::total_cmp);
    b.thresholds.hook = 0.5 * (probs[29] + probs[30]);
    save_bundle(path, &b).unwrap();
    b
}

fn g(path: &std::path::Path) -> ddai_flyg::Flyg {
    ddai_flyg::load(path).unwrap()
}

#[test]
fn a_v4_bundle_loads_as_pooled_and_plays_bit_for_bit_like_v5() {
    let map = tiny_map();
    for view in [HookView::Shared, HookView::MaskedForHookHead] {
        let dir = tempfile::tempdir().unwrap();
        let (v5_path, flyg) = write_tiny_fly_bundle(dir.path(), view);
        let b = with_mid_threshold(&v5_path, &flyg, &map);
        assert_eq!(b.hook_readout, HookReadout::Pooled);
        assert_eq!(b.format_version, crate::bundle::BUNDLE_FORMAT_VERSION);
        let v4_path = dir.path().join("v4.bundle");
        write_v4(&v4_path, &b);
        assert_eq!(
            load_bundle(&v4_path).unwrap(),
            b,
            "a v4 file loads as the same bundle, {view:?}"
        );
        let reference = play(&template_of(&v5_path, &flyg), &map);
        assert!(reference.iter().any(|(a, _)| a.hook) && reference.iter().any(|(a, _)| !a.hook));
        assert_eq!(play(&template_of(&v4_path, &flyg), &map), reference, "{view:?}");
    }
}

#[test]
fn a_zero_weight_wide_readout_plays_bit_for_bit_like_the_original_and_a_trained_one_moves_only_the_hook() {
    let map = tiny_map();
    for view in [HookView::Shared, HookView::MaskedForHookHead] {
        for kind in [HookReadout::LinearDn, HookReadout::MlpDn { hidden: 6 }] {
            let dir = tempfile::tempdir().unwrap();
            let (path, flyg) = write_tiny_fly_bundle(dir.path(), view);
            let base = with_mid_threshold(&path, &flyg, &map);
            let up = upgrade_hook_readout(&base, g(&flyg), kind, 5).unwrap();
            assert_eq!(up.hook_readout, kind);
            let w = up.decoder_params.hook_wide.as_ref().unwrap();
            assert!(w.w2.iter().all(|&x| x == 0.0));
            if matches!(kind, HookReadout::MlpDn { .. }) {
                assert!(
                    w.w1.iter().any(|&x| x != 0.0),
                    "the first layer is drawn, the output weights are zero"
                );
            }
            let up_path = dir.path().join("up.bundle");
            save_bundle(&up_path, &up).unwrap();
            let reference = play(&template_of(&path, &flyg), &map);
            assert_eq!(
                play(&template_of(&up_path, &flyg), &map),
                reference,
                "{kind:?} {view:?}"
            );

            // Trained (non-zero output weights): the file round-trips, the hook probability moves, every other head is bit-identical.
            let mut trained = up.clone();
            let wide = trained.decoder_params.hook_wide.as_mut().unwrap();
            for (i, x) in wide.w2.iter_mut().enumerate() {
                *x = 0.9 - 0.4 * i as f32;
            }
            let tr_path = dir.path().join("tr.bundle");
            save_bundle(&tr_path, &trained).unwrap();
            assert_eq!(load_bundle(&tr_path).unwrap(), trained);
            let moved = play(&template_of(&tr_path, &flyg), &map);
            assert_ne!(
                moved, reference,
                "{kind:?} {view:?}: the readout must change the hook probability"
            );
            // The telemetry prints every head probability; compare it with the hook probability removed.
            let strip = |t: &str| -> String {
                let at = t.find("\"hook_prob\":").unwrap();
                let end = at + t[at..].find(',').unwrap();
                format!("{}{}", &t[..at], &t[end..])
            };
            let tele = |p: &std::path::Path| -> Vec<String> {
                let mut brain = template_of(p, &flyg).instantiate_played(brain_config());
                brain.reset(&reset_ctx(&map));
                (0..60)
                    .map(|i| {
                        let _ = brain.decide(&obs(&map, 2 * i, 250.0 + 7.0 * (i % 23) as f32, scripted_state(i)));
                        let t = brain.telemetry().unwrap();
                        strip(t.rsplit_once(",\"latency_us\"").unwrap().0)
                    })
                    .collect()
            };
            // (In two-view play the viewer's hook values are the masked view's; the other heads are the full view's.)
            assert_eq!(
                tele(&tr_path),
                tele(&path),
                "only the hook head may move ({kind:?} {view:?})"
            );
        }
    }
}

#[test]
fn a_readout_that_disagrees_with_its_parameters_or_meets_an_intent_head_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (path, flyg) = write_tiny_fly_bundle(dir.path(), HookView::Shared);
    let base = load_bundle(&path).unwrap();
    let up = upgrade_hook_readout(&base, g(&flyg), HookReadout::MlpDn { hidden: 4 }, 1).unwrap();
    let bad = dir.path().join("bad.bundle");

    let mut b = up.clone();
    b.hook_readout = HookReadout::Pooled; // but wide parameters
    write_zstd_postcard(&bad, &b, 3).unwrap(); // deliberately invalid: bypasses save_bundle's own check
    assert!(load_bundle(&bad).unwrap_err().0.contains("pooled but"));
    let mut b = up.clone();
    b.decoder_params.hook_wide = None; // but a wide readout
    write_zstd_postcard(&bad, &b, 3).unwrap(); // deliberately invalid: bypasses save_bundle's own check
    assert!(load_bundle(&bad).unwrap_err().0.contains("no wide readout"));
    let mut b = up.clone();
    b.decoder_params.hook_wide.as_mut().unwrap().w1.pop(); // wrong shape: the template refuses it
    write_zstd_postcard(&bad, &b, 3).unwrap(); // deliberately invalid: bypasses save_bundle's own check
    assert!(FlyBrainTemplate::load(&bad, Some(&flyg)).is_err());
    let mut b = up.clone();
    b.decoder_params.hook_wide.as_mut().unwrap().w2[0] = f32::NAN;
    write_zstd_postcard(&bad, &b, 3).unwrap(); // deliberately invalid: bypasses save_bundle's own check
    assert!(FlyBrainTemplate::load(&bad, Some(&flyg)).is_err());

    // Intent + wide, either way round.
    assert!(upgrade_hook_readout(&upgrade_to_intent_hook(&base), g(&flyg), HookReadout::LinearDn, 1).is_err());
    let mut b = up.clone();
    b.hook_param = HookParam::Intent;
    b.decoder_params = b.decoder_params.with_intent_hook();
    write_zstd_postcard(&bad, &b, 3).unwrap(); // deliberately invalid: bypasses save_bundle's own check
    assert!(load_bundle(&bad).unwrap_err().0.contains("intent"));
    // Twice, and the pooled kind is not an upgrade.
    assert!(upgrade_hook_readout(&up, g(&flyg), HookReadout::LinearDn, 1).is_err());
    assert!(HookReadout::parse("pooled").is_ok());
}

/// Review F5: `save_bundle` refuses a bundle whose hook kind disagrees with its parameters instead of writing a file the loader would refuse.
#[test]
fn save_bundle_refuses_an_inconsistent_hook_kind() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = write_tiny_fly_bundle(dir.path(), HookView::Shared);
    let mut b = load_bundle(&path).unwrap();
    b.hook_readout = HookReadout::LinearDn; // but no wide parameters
    let out = dir.path().join("x.bundle");
    assert!(save_bundle(&out, &b).unwrap_err().0.contains("not written"));
    assert!(!out.exists());
}

/// Review F1: the encoder-input control readout is refused where a fly plays (the proposer; the live bot's loader calls the same check) and
/// stays loadable for the arena and the BC tools; the DN readouts are not refused.
#[test]
fn the_control_readout_is_refused_where_a_fly_plays_but_not_in_the_arena_tools() {
    let dir = tempfile::tempdir().unwrap();
    let (path, flyg) = write_tiny_fly_bundle(dir.path(), HookView::Shared);
    let base = load_bundle(&path).unwrap();
    for (kind, ok) in [
        (HookReadout::EncoderMlp { hidden: 3 }, false),
        (HookReadout::LinearDn, true),
        (HookReadout::MlpDn { hidden: 3 }, true),
    ] {
        let b = upgrade_hook_readout(&base, g(&flyg), kind, 1).unwrap();
        let p = dir.path().join("w.bundle");
        save_bundle(&p, &b).unwrap();
        let t = template_of(&p, &flyg); // the template still loads (es eval / hook-eval use it)
        assert_eq!(t.require_fly_readout("test").is_ok(), ok, "{kind:?}");
        assert_eq!(
            crate::proposer::FlyProposer::from_template(&t, brain_config(), 1).is_ok(),
            ok,
            "{kind:?}"
        );
        if !ok {
            assert!(
                t.require_fly_readout("test")
                    .unwrap_err()
                    .0
                    .contains("encoder-input control")
            );
        }
    }
    assert!(template_of(&path, &flyg).require_fly_readout("test").is_ok());
}

/// The oldest layout that plays the bundle correctly is the one written (task 8.6 F2, extended to v5): a pooled legacy bundle is a v3
/// file, a pooled intent head a v4 file, a wide readout a v5 file, and each loads back as the same bundle.
#[test]
fn a_bundle_is_written_in_the_oldest_layout_that_plays_it_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let (path, flyg) = write_tiny_fly_bundle(dir.path(), HookView::MaskedForHookHead);
    let base = load_bundle(&path).unwrap();
    let version = |b: &FlyBundle| -> u32 {
        let p = dir.path().join("probe.bundle");
        save_bundle(&p, b).unwrap();
        let bytes = crate::bundle::read_zstd_bytes(&p).unwrap();
        let v = crate::bundle::peek_version(&p, &bytes).unwrap();
        assert_eq!(
            &load_bundle(&p).unwrap(),
            b,
            "version {v} reads back as the same bundle"
        );
        v
    };
    assert_eq!(version(&base), 3);
    assert_eq!(version(&upgrade_to_intent_hook(&base)), 4);
    let mut latched = base.clone();
    latched.hook_decode = HookDecode::Latched { hi: 0.7, lo: 0.4 };
    assert_eq!(version(&latched), 4);
    for kind in [
        HookReadout::LinearDn,
        HookReadout::MlpDn { hidden: 3 },
        HookReadout::EncoderMlp { hidden: 3 },
    ] {
        let wide = upgrade_hook_readout(&base, g(&flyg), kind, 1).unwrap();
        assert_eq!(version(&wide), 5, "{kind:?}");
    }
}
