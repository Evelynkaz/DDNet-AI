//! Playing a model trained with `HookView::MaskedForHookHead` (task 8.2b): [`TwoViewFly`] for a fly and
//! [`TwoViewBrain`] for any other brain (the MLP/GRU controls).
//!
//! In training the hook head of such a model never saw the own hook state: it was trained on the observation with that
//! state hidden ([`crate::bc::mask_own_hook`]) and the full view carries **no** hook loss. So the model must be played
//! in two views: the full network decides everything but the hook, a second instance of the same network sees the same
//! observations with the own hook state hidden and decides the hook. Two recurrent states, one action.
//!
//! Every place that plays a trained fly (the arena's `ModelBrains`, the live bot's `--fly-bundle`) must go through
//! [`crate::bundle::FlyBrainTemplate::instantiate_played`] / [`crate::proposer::FlyProposer::from_template`], which
//! pick the right form from the bundle's `hook_view`; a single-view play of a masked bundle is an unevaluated policy.

use ddai_brain::{Action, Brain, Observation, ResetContext, WorldView};

use crate::brain::{FlyBrain, PlayedOverride};

/// `obs` with the own hook state hidden, written into `scratch` without allocating once its `others` has the capacity
/// (`mask_own_hook` clones the whole observation, a heap allocation per decision).
fn masked_into<'a>(scratch: &'a mut Option<Observation>, obs: &Observation) -> &'a Observation {
    // Destructured so that a field added to `Observation` fails to compile here instead of being silently dropped.
    let Observation {
        map,
        tick,
        self_state,
        others,
        target_id,
        tuning,
    } = obs;
    let s = scratch.get_or_insert_with(|| obs.clone());
    s.map.clone_from(map);
    s.tick = *tick;
    s.self_state = *self_state;
    s.others.clone_from(others);
    s.target_id = *target_id;
    s.tuning = *tuning;
    s.self_state.hook_state = ddai_brain::HOOK_IDLE;
    s
}

/// A fly played in two views: `full` decides everything but the hook, `masked` (the same weights, fed the observation
/// without the own hook state) decides the hook. The viewer's frame and telemetry show the **played** hook and its
/// probability (the masked view's) and the time of both views.
pub struct TwoViewFly {
    full: FlyBrain,
    masked: FlyBrain,
    scratch: Option<Observation>,
    name: String,
}

impl TwoViewFly {
    pub fn new(full: FlyBrain, masked: FlyBrain) -> Self {
        let name = format!("{}+hookview", full.name());
        TwoViewFly {
            full,
            masked,
            scratch: None,
            name,
        }
    }

    fn combine(&mut self, a: Action, b: Action) -> Action {
        if let Some(d) = self.masked.last_decoded() {
            self.full.set_played_override(Some(PlayedOverride {
                hook: b.hook,
                hook_prob: d.hook_prob,
                latency: self.full.last_latency() + self.masked.last_latency(),
            }));
        }
        Action { hook: b.hook, ..a }
    }
}

impl Brain for TwoViewFly {
    fn reset(&mut self, ctx: &ResetContext) {
        self.full.reset(ctx);
        self.masked.reset(ctx);
    }

    // One code path: the arena and the live bot call `decide_in`, so `decide` is `decide_in` without a world view.
    fn decide(&mut self, obs: &Observation) -> Action {
        self.decide_in(obs, None)
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let a = self.full.decide_in(obs, view);
        let b = self.masked.decide_in(masked_into(&mut self.scratch, obs), view);
        self.combine(a, b)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn telemetry(&self) -> Option<String> {
        self.full.telemetry()
    }

    fn viz_meta(&self) -> Option<String> {
        self.full.viz_meta()
    }

    fn viz_frame(&mut self, tick: u32) -> Option<&[u8]> {
        self.full.viz_frame(tick)
    }
}

/// Any brain played in two views (the MLP/GRU controls): `full` decides everything but the hook, `masked` decides it.
pub struct TwoViewBrain {
    full: Box<dyn Brain>,
    masked: Box<dyn Brain>,
    scratch: Option<Observation>,
    name: String,
}

impl TwoViewBrain {
    pub fn new(full: Box<dyn Brain>, masked: Box<dyn Brain>) -> Self {
        let name = format!("{}+hookview", full.name());
        TwoViewBrain {
            full,
            masked,
            scratch: None,
            name,
        }
    }
}

impl Brain for TwoViewBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.full.reset(ctx);
        self.masked.reset(ctx);
    }

    // One code path (see `TwoViewFly`).
    fn decide(&mut self, obs: &Observation) -> Action {
        self.decide_in(obs, None)
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let a = self.full.decide_in(obs, view);
        let b = self.masked.decide_in(masked_into(&mut self.scratch, obs), view);
        Action { hook: b.hook, ..a }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bc::{HookView, mask_own_hook};
    use crate::brain::{ActionSelection, FlyBrainConfig};
    use crate::bundle::FlyBrainTemplate;

    fn template(view: HookView) -> (tempfile::TempDir, FlyBrainTemplate) {
        let dir = tempfile::tempdir().unwrap();
        let (bundle, flyg) = crate::brain_fixtures::write_tiny_fly_bundle(dir.path(), view);
        let t = FlyBrainTemplate::load(&bundle, Some(&flyg)).unwrap();
        (dir, t)
    }

    fn config() -> FlyBrainConfig {
        FlyBrainConfig {
            action_selection: ActionSelection::Argmax,
            seed: 1,
        }
    }

    fn observation(opp_x: f32, hook_state: i32) -> Observation {
        let mut me = ddai_brain::CharacterObservation::at_rest(0);
        me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
        me.hook_state = hook_state;
        let mut opp = ddai_brain::CharacterObservation::at_rest(1);
        opp.pos = ddai_physics::vmath::Vec2::new(opp_x, 300.0);
        let map = ddai_physics::map::MapData {
            width: 20,
            height: 20,
            game: vec![Default::default(); 400],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        Observation {
            map: std::sync::Arc::new(map),
            tick: 0,
            self_state: me,
            others: vec![opp],
            target_id: None,
            tuning: ddai_physics::tuning::TuningParams::default(),
        }
    }

    /// Review F11: the viewer's frame and telemetry of a two-view play show the played hook (the masked view's), its
    /// probability and the time of both views, not the full view's own (whose hook head was never trained).
    #[test]
    fn the_frame_and_telemetry_show_the_played_hook_and_both_views_latency() {
        let (_dir, t) = template(HookView::MaskedForHookHead);
        let mut played = TwoViewFly::new(t.instantiate(config()), t.instantiate(config()));
        let mut mirror = t.instantiate(config());
        let obs = observation(400.0, ddai_brain::HOOK_GRABBED);
        let reset = ResetContext {
            map: obs.map.clone(),
            self_id: 0,
            seed: 1,
        };
        played.reset(&reset);
        mirror.reset(&reset);
        played.full.set_viz_every(1);
        for _ in 0..3 {
            let a = played.decide(&obs);
            mirror.decide(&mask_own_hook(&obs));
            let want = mirror.last_decoded().unwrap().hook_prob;
            let full_own = played.full.last_decoded().unwrap().hook_prob;
            assert!(
                (want - full_own).abs() > 0.1,
                "the two views disagree: {want} vs {full_own}"
            );
            assert!(!a.hook, "masked view: not hooking");

            let layout = played.full.viz_layout().clone();
            let frame = played.viz_frame(7).expect("a frame").to_vec();
            let f = crate::viz::decode_frame(&frame, layout.rate_max(), layout.z_clip()).unwrap();
            let want_logit = (want / (1.0 - want)).ln();
            assert!(
                (f.logits[4] - want_logit).abs() < 0.2,
                "frame hook logit {} vs played {want_logit}",
                f.logits[4]
            );
            assert_eq!(f.flags & crate::viz::flag::HOOK != 0, a.hook);
            let both = (played.full.last_latency() + played.masked.last_latency()).as_micros();
            assert_eq!(u128::from(f.latency_us), both.min(65535), "both views' time");
            assert!(u128::from(f.latency_us) > played.full.last_latency().as_micros());

            let json = played.telemetry().unwrap();
            assert!(json.contains(&format!("\"hook_prob\":{want}")), "{json}");
        }
    }
}
