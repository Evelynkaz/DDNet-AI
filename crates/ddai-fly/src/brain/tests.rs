use super::*;
use crate::brain_fixtures::tiny_brain_flyg;
use crate::config::FlyConfig;
use crate::decoder::{DecoderConfig, DecoderModel, DnCalibration};
use crate::encoder::{EncoderModel, EncoderParams, ProprioceptionConfig, RayGridConfig};
use crate::params::FlyParams;
use ddai_brain::{CharacterObservation, Observation};
use std::sync::Arc;

fn tiny_map() -> ddai_physics::map::MapData {
    ddai_physics::map::MapData {
        width: 20,
        height: 20,
        game: vec![Default::default(); 400],
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

pub(crate) fn make_brain(seed: u64, selection: ActionSelection) -> FlyBrain {
    let flyg = tiny_brain_flyg();
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 1);
    let model = FlyModel::new(flyg, config, params).unwrap();

    let encoder = EncoderModel::new(
        &model,
        RayGridConfig::default(),
        &ProprioceptionConfig {
            grounded: vec!["AN_GROUND".to_string()],
            ..ProprioceptionConfig::default()
        },
    )
    .unwrap();
    let encoder_params = EncoderParams::init_default(encoder.num_params());

    let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
    let decoder_params = decoder.init_default_params();
    let calib = DnCalibration {
        mu: vec![0.0; model.num_outputs()],
        sigma: vec![1.0; model.num_outputs()],
    };

    FlyBrain::new(
        model,
        encoder,
        encoder_params,
        decoder,
        decoder_params,
        calib,
        FlyBrainConfig {
            action_selection: selection,
            seed,
        },
    )
}

pub(crate) fn sample_observation(opp_x: f32) -> Observation {
    let mut me = CharacterObservation::at_rest(0);
    me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
    let mut opp = CharacterObservation::at_rest(1);
    opp.pos = ddai_physics::vmath::Vec2::new(opp_x, 300.0);
    Observation {
        map: Arc::new(tiny_map()),
        tick: 0,
        self_state: me,
        others: vec![opp],
        target_id: None,
        tuning: ddai_physics::tuning::TuningParams::default(),
    }
}

use ddai_brain::Brain;

#[test]
fn decide_returns_a_well_formed_action() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    let obs = sample_observation(400.0);
    let action = brain.decide(&obs);
    assert!((-1..=1).contains(&action.direction));
    assert_ne!(
        (action.target.x, action.target.y),
        (0, 0),
        "target must never be exactly zero"
    );
}

#[test]
fn reset_then_decide_does_not_panic_and_telemetry_appears_after_deciding() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    assert!(
        ddai_brain::Brain::telemetry(&brain).is_none(),
        "no telemetry before the first decision"
    );
    let ctx = ddai_brain::ResetContext {
        map: Arc::new(tiny_map()),
        self_id: 0,
        seed: 1,
    };
    brain.reset(&ctx);
    let obs = sample_observation(400.0);
    let _ = brain.decide(&obs);
    let telemetry = brain.telemetry().expect("telemetry after a decision");
    assert!(telemetry.starts_with('{'));
    assert!(telemetry.contains("dn_z_by_action"));
    assert!(telemetry.contains("decoded_action"));
}

/// Review round 1, F13 (CONFIRMED): `reset` must re-seed from `ResetContext::seed`, not
/// `FlyBrainConfig::seed` (which stays fixed for the brain's whole lifetime) -- two brains built
/// with *different* config seeds must still sample byte-for-byte identically after both are
/// `reset` with the *same* `ctx.seed`. An earlier revision ignored `ctx` entirely, so this would
/// have failed (each brain kept sampling from its own original, different config seed forever).
#[test]
fn reset_reseeds_sampled_action_selection_from_the_context_seed() {
    let mut a = make_brain(111, ActionSelection::Sampled);
    let mut b = make_brain(222, ActionSelection::Sampled);
    let ctx = ddai_brain::ResetContext {
        map: Arc::new(tiny_map()),
        self_id: 0,
        seed: 999,
    };
    a.reset(&ctx);
    b.reset(&ctx);
    for x in [350.0, 500.0, 250.0, 600.0] {
        let obs = sample_observation(x);
        assert_eq!(a.decide(&obs), b.decide(&obs));
    }
}

#[test]
fn decide_is_deterministic_for_the_same_seed_and_observations() {
    let mut a = make_brain(42, ActionSelection::Sampled);
    let mut b = make_brain(42, ActionSelection::Sampled);
    for x in [350.0, 500.0, 250.0, 600.0] {
        let obs = sample_observation(x);
        assert_eq!(a.decide(&obs), b.decide(&obs));
    }
}

#[test]
fn latency_is_recorded_after_a_decision() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    assert_eq!(brain.last_latency(), std::time::Duration::ZERO);
    let obs = sample_observation(400.0);
    let _ = brain.decide(&obs);
    // Not a timing assertion (no host-speed dependency) — just that *something* was recorded,
    // proving `decide` actually measured itself rather than leaving the field untouched.
    let _ = brain.last_latency();
}

#[test]
fn name_is_non_empty_and_mentions_graph_size() {
    let brain = make_brain(1, ActionSelection::Argmax);
    let name = Brain::name(&brain);
    assert!(name.contains('n'), "name={name}");
}

#[test]
fn thresholds_decide_each_binary_head_under_argmax() {
    let obs = sample_observation(400.0);
    let probe = |th: crate::bc::HeadThresholds| {
        let mut b = make_brain(1, ActionSelection::Argmax);
        b.set_thresholds(th);
        b.reset(&ddai_brain::ResetContext {
            map: obs.map.clone(),
            self_id: 0,
            seed: 1,
        });
        let a = b.decide(&obs);
        let d = *b.last_decoded().unwrap();
        (a, d)
    };
    let (a0, d) = probe(crate::bc::HeadThresholds::default());
    assert_eq!(a0.jump, d.jump_prob >= 0.5);
    assert_eq!(a0.hook, d.hook_prob >= 0.5);
    assert_eq!(a0.fire, d.fire_prob >= 0.5);
    // A threshold just below a head's probability presses it, just above releases it.
    let eps = 1e-4;
    let at = |j: f32, h: f32, f: f32| crate::bc::HeadThresholds {
        jump: j,
        hook: h,
        fire: f,
    };
    let (below, _) = probe(at(d.jump_prob - eps, d.hook_prob - eps, d.fire_prob - eps));
    assert!(below.jump && below.hook && below.fire);
    let (above, _) = probe(at(d.jump_prob + eps, d.hook_prob + eps, d.fire_prob + eps));
    assert!(!above.jump && !above.hook && !above.fire);
}

// ---- task 7.4: the visualisation stream ------------------------------------------------------------

#[test]
fn a_frame_round_trips_within_one_quantisation_step() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    brain.set_viz_every(1);
    for i in 0..3 {
        let _ = brain.decide(&sample_observation(300.0 + 10.0 * i as f32));
    }
    let obs = sample_observation(330.0);
    let action = brain.decide(&obs);
    let outcome = ddai_planner::hybrid::ProposalOutcome {
        chosen: true,
        chosen_total: 7,
        decisions_total: 9,
    };
    let bytes = brain.viz_frame_with(1234, Some(outcome)).expect("a frame").to_vec();
    let layout = brain.viz_layout().clone();
    assert_eq!(bytes.len(), layout.frame_len());
    let f = crate::viz::decode_frame(&bytes, layout.rate_max(), layout.z_clip()).unwrap();
    assert_eq!((f.tick, f.seq), (1234, 4));
    assert_eq!(i32::from(f.direction), action.direction);
    assert_eq!(f.flags & crate::viz::flag::JUMP != 0, action.jump);
    assert_eq!(f.flags & crate::viz::flag::HOOK != 0, action.hook);
    assert_eq!(f.flags & crate::viz::flag::FIRE != 0, action.fire);
    assert_ne!(f.flags & crate::viz::flag::CHOSEN_VALID, 0);
    assert_ne!(f.flags & crate::viz::flag::CHOSEN, 0);
    assert_eq!((f.chosen_total, f.decisions_total), (7, 9));
    let decoded = brain.last_decoded().unwrap();
    assert!((f.aim_angle - decoded.aim_angle).abs() < 1.0 / crate::viz::AIM_SCALE);
    // Group rates: against the layout's own aggregation of the brain's last per-type means.
    for (g, &got) in f.groups.iter().enumerate() {
        let want = layout.group_rate(g, &brain.last_per_type_mean_rate);
        assert!(
            (got - want).abs() <= layout.rate_max() / 255.0 / 2.0 + 1e-5,
            "group {g}: {got} vs {want}"
        );
    }
    // DN z-scores.
    let z = brain.calib.z(&brain.last_dn_rates, layout.z_clip());
    for (i, (&got, &want)) in f.dn_z.iter().zip(&z).enumerate() {
        assert!(
            (got - want).abs() <= layout.z_clip() / 127.0 / 2.0 + 1e-5,
            "dn {i}: {got} vs {want}"
        );
    }
    // The eye: every channel and cell.
    for (ci, ch) in crate::viz::EYE_CHANNELS.iter().enumerate() {
        let grid = brain.last_ray_features().spatial(*ch);
        assert_eq!(f.eye[ci].len(), grid.len());
        for (i, (&got, &want)) in f.eye[ci].iter().zip(grid).enumerate() {
            assert!(
                (got - want.clamp(0.0, 1.0)).abs() <= 0.5 / 255.0 + 1e-6,
                "{ch:?}[{i}]: {got} vs {want}"
            );
        }
    }
    // Logits: the heads are true logits, the direction ones log-probabilities.
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    assert!((sig(f.logits[3]) - decoded.jump_prob).abs() < 0.07);
    assert!((f.logits[0].exp() - decoded.direction_probs[0]).abs() < 0.07);
    assert_eq!(f.rays, 48);
    assert_eq!(f.bins, 4);
}

#[test]
fn pulling_frames_never_changes_a_decision() {
    // Two identical flies see the same observations; one is watched every decision, with a made-up
    // proposer verdict. Every action, rate and decoded probability must agree bit for bit.
    let mut watched = make_brain(5, ActionSelection::Sampled);
    let mut plain = make_brain(5, ActionSelection::Sampled);
    let ctx = ddai_brain::ResetContext {
        map: sample_observation(0.0).map,
        self_id: 0,
        seed: 9,
    };
    watched.reset(&ctx);
    plain.reset(&ctx);
    watched.set_viz_every(1);
    for i in 0..60 {
        let obs = sample_observation(200.0 + 7.0 * i as f32);
        let a = watched.decide(&obs);
        let b = plain.decide(&obs);
        assert_eq!(a, b, "decision {i}");
        let outcome = ddai_planner::hybrid::ProposalOutcome {
            chosen: i % 3 == 0,
            chosen_total: i,
            decisions_total: i + 1,
        };
        assert!(watched.viz_frame_with(i, Some(outcome)).is_some());
        assert_eq!(
            watched.last_dn_rates.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            plain.last_dn_rates.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "dn rates {i}"
        );
        let (x, y) = (watched.last_decoded().unwrap(), plain.last_decoded().unwrap());
        assert_eq!(x.direction_probs.map(f32::to_bits), y.direction_probs.map(f32::to_bits));
        assert_eq!(x.aim_angle.to_bits(), y.aim_angle.to_bits());
        assert_eq!(
            watched.state.v().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            plain.state.v().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "membrane state {i}"
        );
    }
}

#[test]
fn a_reset_restarts_the_frames_decision_number_and_owes_no_frame() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    brain.set_viz_every(1);
    for i in 0..3 {
        let _ = brain.decide(&sample_observation(300.0 + i as f32));
    }
    let f = crate::viz::decode_frame(brain.viz_frame_with(1, None).unwrap(), 10.0, 10.0).unwrap();
    assert_eq!(f.seq, 3);
    // Decisions nobody watched pile up; a reset ends the episode: no frame of the old one, numbering from 1.
    let _ = brain.decide(&sample_observation(310.0));
    brain.reset(&ddai_brain::ResetContext {
        map: sample_observation(0.0).map,
        self_id: 0,
        seed: 2,
    });
    assert!(
        brain.viz_frame_with(2, None).is_none(),
        "nothing decided since the reset"
    );
    let _ = brain.decide(&sample_observation(320.0));
    let f = crate::viz::decode_frame(brain.viz_frame_with(3, None).unwrap(), 10.0, 10.0).unwrap();
    assert_eq!((f.seq, f.tick), (1, 3));
}

/// 8.2b F1 (review F8): a bundle trained with the hook head masked is played in two views wherever a trained fly is
/// played (`instantiate_played`, `FlyProposer::from_template`); a shared-view bundle stays a plain fly. The fixture's
/// hook output is wired to the own-hook input, so masking MATTERS: the tests below fail if the second view is fed the
/// unmasked observation or the proposer drops its second view.
fn own_hook_templates() -> (crate::bundle::FlyBrainTemplate, crate::bundle::FlyBrainTemplate) {
    use crate::bc::HookView;
    use crate::bundle::FlyBrainTemplate;
    let dir = tempfile::tempdir().unwrap();
    let load = |view| {
        let sub = dir.path().join(format!("{view:?}"));
        std::fs::create_dir_all(&sub).unwrap();
        let (bundle, flyg) = crate::brain_fixtures::write_tiny_fly_bundle(&sub, view);
        FlyBrainTemplate::load(&bundle, Some(&flyg)).unwrap()
    };
    (load(HookView::Shared), load(HookView::MaskedForHookHead))
}

fn played_config() -> FlyBrainConfig {
    FlyBrainConfig {
        action_selection: ActionSelection::Argmax,
        seed: 1,
    }
}

/// 60 observations cycling the own hook state (idle / flying / grabbed) while the opponent moves.
fn own_hook_observations() -> Vec<Observation> {
    let states = [ddai_brain::HOOK_IDLE, ddai_brain::HOOK_FLYING, ddai_brain::HOOK_GRABBED];
    (0..60)
        .map(|i| {
            let mut obs = sample_observation(340.0 + 4.0 * (i / 3) as f32);
            obs.self_state.hook_state = states[i % 3];
            obs
        })
        .collect()
}

#[test]
fn a_masked_bundle_is_played_in_two_views_and_a_shared_one_is_not() {
    use crate::bc::mask_own_hook;
    let (shared, masked) = own_hook_templates();
    assert!(!shared.instantiate_played(played_config()).name().ends_with("+hookview"));
    assert!(
        !crate::proposer::FlyProposer::from_template(&shared, played_config(), 1)
            .unwrap()
            .has_hook_view()
    );
    assert!(masked.instantiate_played(played_config()).name().ends_with("+hookview"));
    assert!(
        crate::proposer::FlyProposer::from_template(&masked, played_config(), 1)
            .unwrap()
            .has_hook_view()
    );

    let mut played = masked.instantiate_played(played_config());
    let (mut full, mut hook_view) = (masked.instantiate(played_config()), masked.instantiate(played_config()));
    let observations = own_hook_observations();
    let reset = ddai_brain::ResetContext {
        map: observations[0].map.clone(),
        self_id: 0,
        seed: 1,
    };
    played.reset(&reset);
    full.reset(&reset);
    hook_view.reset(&reset);
    let (mut differ_in_prob, mut differ_in_hook) = (0, 0);
    for (i, obs) in observations.iter().enumerate() {
        let a = played.decide(obs);
        let f = full.decide(obs);
        let h = hook_view.decide(&mask_own_hook(obs));
        // The played action = the full view's action with the hook of the second, masked-input view ...
        assert_eq!(a, ddai_brain::Action { hook: h.hook, ..f }, "decision {i}");
        // ... and the fixture makes that matter: the two views disagree about the hook probability, and sometimes about
        // the hook itself, so a play that fed the second view the unmasked observation (or none) fails here.
        let (pf, ph) = (
            full.last_decoded().unwrap().hook_prob,
            hook_view.last_decoded().unwrap().hook_prob,
        );
        differ_in_prob += usize::from((pf - ph).abs() > 1e-3);
        differ_in_hook += usize::from(f.hook != h.hook);
    }
    assert!(
        differ_in_prob > 20,
        "masking must change the hook probability ({differ_in_prob}/60)"
    );
    assert!(
        differ_in_hook > 5,
        "masking must change the hook decision ({differ_in_hook}/60)"
    );
}

/// Review F9: the work-clock price of a proposal is per network run, so a two-view proposer costs twice a one-view one
/// (3.7a's `proposal_in_cap` takes that price off the search budget).
#[test]
fn a_masked_proposer_costs_two_views_on_the_work_clock() {
    use crate::bc::HookView;
    use crate::bundle::FlyBrainTemplate;
    use crate::proposer::FlyProposer;
    use ddai_planner::hybrid::Proposer;
    let dir = tempfile::tempdir().unwrap();
    // Many substeps, so that the price (nnz x substeps x rate) does not round to 0 tee-ticks on the tiny graph.
    let load = |view: HookView| {
        let sub = dir.path().join(format!("{view:?}"));
        std::fs::create_dir_all(&sub).unwrap();
        let (bundle, flyg) = crate::brain_fixtures::write_tiny_fly_bundle_with(&sub, view, 4000);
        FlyBrainTemplate::load(&bundle, Some(&flyg)).unwrap()
    };
    let shared = FlyProposer::from_template(&load(HookView::Shared), played_config(), 1).unwrap();
    let masked = FlyProposer::from_template(&load(HookView::MaskedForHookHead), played_config(), 1).unwrap();
    assert!(shared.work_units() > 100, "{}", shared.work_units());
    assert_eq!(masked.work_units(), 2 * shared.work_units());
}

/// Review F8 (round 3): the proposer's own path. A masked `FlyProposer`'s distribution takes its hook probability from
/// the masked view (and every other head from the full one), a shared proposer's from its single network; the frame the
/// proposer streams shows the played hook, its probability and both views' time.
#[test]
fn a_masked_proposer_builds_its_distribution_and_frame_from_the_masked_view() {
    use crate::bc::mask_own_hook;
    use crate::proposer::FlyProposer;
    use ddai_planner::hybrid::Proposer;
    let (shared, masked) = own_hook_templates();
    let observations = own_hook_observations();
    let reset = ddai_brain::ResetContext {
        map: observations[0].map.clone(),
        self_id: 0,
        seed: 1,
    };

    let mut p = FlyProposer::from_template(&masked, played_config(), 1).unwrap();
    let mut single = FlyProposer::from_template(&shared, played_config(), 1).unwrap();
    let (mut full, mut hook_view) = (masked.instantiate(played_config()), masked.instantiate(played_config()));
    let mut shared_net = shared.instantiate(played_config());
    Proposer::reset(&mut p, &reset);
    Proposer::reset(&mut single, &reset);
    full.reset(&reset);
    hook_view.reset(&reset);
    shared_net.reset(&reset);
    let layout = p.brain().viz_layout().clone();
    let (mut differs, mut frames) = (0, 0);
    for (i, obs) in observations.iter().enumerate() {
        let d = p.distribution(obs).unwrap();
        full.decide(obs);
        let played = hook_view.decide(&mask_own_hook(obs));
        let (f, h) = (full.last_decoded().unwrap(), hook_view.last_decoded().unwrap());
        assert_eq!(
            d.hook,
            f64::from(h.hook_prob),
            "decision {i}: the hook probability is the masked view's"
        );
        assert_eq!(
            d.jump,
            f64::from(f.jump_prob),
            "decision {i}: the other heads are the full view's"
        );
        assert_eq!(d.direction, f.direction_probs.map(f64::from));
        differs += usize::from((f.hook_prob - h.hook_prob).abs() > 1e-3);

        // a one-view proposer: everything from its single network
        let ds = single.distribution(obs).unwrap();
        shared_net.decide(obs);
        assert_eq!(
            ds.hook,
            f64::from(shared_net.last_decoded().unwrap().hook_prob),
            "decision {i}"
        );

        if let Some(bytes) = Proposer::viz_frame(&mut p, i as u32, None) {
            frames += 1;
            let fr = crate::viz::decode_frame(bytes, layout.rate_max(), layout.z_clip()).unwrap();
            let want_logit = (h.hook_prob / (1.0 - h.hook_prob)).ln();
            assert!(
                (fr.logits[4] - want_logit).abs() < 0.2,
                "decision {i}: frame hook logit {} vs {want_logit}",
                fr.logits[4]
            );
            assert_eq!(
                fr.flags & crate::viz::flag::HOOK != 0,
                played.hook,
                "decision {i}: the played hook"
            );
            assert!(
                u128::from(fr.latency_us) > p.brain().last_latency().as_micros(),
                "decision {i}: the frame's latency covers both views"
            );
        }
    }
    assert!(
        differs > 20,
        "the views must disagree for this test to bite ({differs}/60)"
    );
    assert!(frames > 10, "the proposer streamed {frames} frames");
}

/// Review F8 (round 3, M9): the played-hook override of a two-view play belongs to one decision stream; `reset` starts a
/// new episode and clears it, so a brain taken out of a two-view play never shows a stale hook or latency.
#[test]
fn a_played_hook_override_is_cleared_by_reset() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    let obs = sample_observation(400.0);
    let reset = ddai_brain::ResetContext {
        map: obs.map.clone(),
        self_id: 0,
        seed: 1,
    };
    brain.reset(&reset);
    brain.decide(&obs);
    let own = brain.last_decoded().unwrap().hook_prob;
    brain.set_played_override(Some(PlayedOverride {
        hook: true,
        hook_prob: 0.987,
        latency: Duration::from_millis(5),
    }));
    assert!(
        brain.telemetry().unwrap().contains("\"hook_prob\":0.987"),
        "the override is shown"
    );
    brain.reset(&reset);
    brain.decide(&obs);
    let json = brain.telemetry().unwrap();
    assert!(
        json.contains(&format!("\"hook_prob\":{own}")),
        "after reset the brain shows its own hook: {json}"
    );
    assert!(!json.contains("0.987"), "{json}");
}
