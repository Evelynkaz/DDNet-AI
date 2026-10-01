//! [`ControlBrain`]: a control network playing through `ddai_brain::Brain`, and
//! [`ControlTemplate`], the loaded-once, instantiate-per-game handle the arena factory uses.
//!
//! The action rule is the fly's ([`ddai_fly::brain::FlyBrain`]): argmax over the direction logits,
//! `p >= 0.5` for jump/hook/fire, the aim population-vector angle turned into a target vector by
//! [`ddai_fly::brain::aim_angle_to_target`].

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ddai_brain::{Action, Brain, Observation, ResetContext};
use ddai_fly::bc::HeadThresholds;
use ddai_fly::brain::ActionSelection;
use ddai_fly::brain::aim_angle_to_target;
use ddai_fly::bundle::{BundleError, BundleMeta};
use ddai_fly::encoder::{RayGridConfig, RayGridFeatures};
use ddai_fly::rng::SplitMix64;

use crate::bundle::load_control_bundle;
use crate::features::{extract, input_dim};
use crate::net::SeqNet;

pub struct ControlBrain {
    net: Arc<dyn SeqNet>,
    ray_grid: RayGridConfig,
    state: Vec<f32>,
    scratch: RayGridFeatures,
    x: Vec<f32>,
    name: String,
    last_latency: Duration,
    selection: ActionSelection,
    thresholds: HeadThresholds,
    rng: SplitMix64,
}

impl ControlBrain {
    pub fn new(net: Arc<dyn SeqNet>, ray_grid: RayGridConfig) -> Self {
        Self::with_selection(net, ray_grid, ActionSelection::Argmax)
    }

    /// `Sampled` draws every head from its probabilities (seeded from the reset context, so a game
    /// stays reproducible); `Argmax` takes the most likely value.
    pub fn with_selection(net: Arc<dyn SeqNet>, ray_grid: RayGridConfig, selection: ActionSelection) -> Self {
        assert_eq!(
            net.input_dim(),
            input_dim(&ray_grid),
            "control net / ray grid input size mismatch"
        );
        let name = format!("{}-h{}", net.kind().name(), net.hidden());
        ControlBrain {
            state: vec![0.0; net.state_size()],
            scratch: RayGridFeatures::new(&ray_grid),
            x: Vec::with_capacity(net.input_dim()),
            net,
            ray_grid,
            name,
            last_latency: Duration::ZERO,
            selection,
            thresholds: HeadThresholds::default(),
            rng: SplitMix64::new(0),
        }
    }

    /// Decision thresholds of the jump/hook/fire heads under argmax selection (default `0.5`).
    pub fn with_thresholds(mut self, thresholds: HeadThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    pub fn last_latency(&self) -> Duration {
        self.last_latency
    }
}

impl Brain for ControlBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.rng = SplitMix64::new(ctx.seed);
        self.state.fill(0.0);
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        let t0 = Instant::now();
        extract(obs, &self.ray_grid, &mut self.scratch, &mut self.x);
        let l = self.net.step(&mut self.state, &self.x);
        let p = l.dir_probs();
        let (best, jump, hook, fire) = match self.selection {
            ActionSelection::Argmax => (
                (0..3).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap_or(1),
                l.jump_on(&self.thresholds),
                l.hook_on(&self.thresholds),
                l.fire_on(&self.thresholds),
            ),
            ActionSelection::Sampled => {
                let u = self.rng.next_f32_unit();
                let dir = if u < p[0] {
                    0
                } else if u < p[0] + p[1] {
                    1
                } else {
                    2
                };
                (
                    dir,
                    self.rng.next_f32_unit() < l.jump_prob(),
                    self.rng.next_f32_unit() < l.hook_prob(),
                    self.rng.next_f32_unit() < l.fire_prob(),
                )
            }
        };
        let action = Action {
            direction: [-1, 0, 1][best],
            jump,
            hook,
            fire,
            target: aim_angle_to_target(l.aim_angle()),
            wanted_weapon: None,
        };
        self.last_latency = t0.elapsed();
        action
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A loaded control network: build any number of independent brains from it.
#[derive(Clone)]
pub struct ControlTemplate {
    net: Arc<dyn SeqNet>,
    ray_grid: RayGridConfig,
    thresholds: HeadThresholds,
    pub meta: BundleMeta,
}

impl ControlTemplate {
    pub fn new(net: Arc<dyn SeqNet>, ray_grid: RayGridConfig, meta: BundleMeta) -> Self {
        ControlTemplate {
            net,
            ray_grid,
            thresholds: HeadThresholds::default(),
            meta,
        }
    }

    /// The same template with calibrated decision thresholds.
    pub fn with_thresholds(mut self, thresholds: HeadThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    pub fn thresholds(&self) -> HeadThresholds {
        self.thresholds
    }

    pub fn load(path: &Path) -> Result<Self, BundleError> {
        let b = load_control_bundle(path)?;
        let net: Arc<dyn SeqNet> = Arc::from(b.build()?);
        if net.input_dim() != input_dim(&b.ray_grid) {
            return Err(BundleError(format!(
                "{}: stored input size {} does not match its ray grid ({})",
                path.display(),
                net.input_dim(),
                input_dim(&b.ray_grid)
            )));
        }
        Ok(ControlTemplate {
            net,
            ray_grid: b.ray_grid,
            thresholds: b.thresholds,
            meta: b.meta,
        })
    }

    pub fn net(&self) -> &dyn SeqNet {
        self.net.as_ref()
    }

    pub fn ray_grid(&self) -> &RayGridConfig {
        &self.ray_grid
    }

    pub fn instantiate(&self) -> ControlBrain {
        ControlBrain::new(self.net.clone(), self.ray_grid).with_thresholds(self.thresholds)
    }

    pub fn instantiate_with(&self, selection: ActionSelection) -> ControlBrain {
        ControlBrain::with_selection(self.net.clone(), self.ray_grid, selection).with_thresholds(self.thresholds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gru::Gru;
    use crate::mlp::Mlp;
    use ddai_brain::CharacterObservation;
    use ddai_physics::map::{MapData, TILE_SOLID, Tile};
    use ddai_physics::tuning::TuningParams;

    fn room_obs() -> Observation {
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
        let map = Arc::new(MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        });
        let mut me = CharacterObservation::at_rest(0);
        me.pos = ddai_physics::vmath::Vec2::new(300.0, 400.0);
        let mut opp = CharacterObservation::at_rest(1);
        opp.pos = ddai_physics::vmath::Vec2::new(500.0, 400.0);
        Observation {
            map,
            tick: 0,
            self_state: me,
            others: vec![opp],
            target_id: Some(1),
            tuning: TuningParams::default(),
        }
    }

    fn reset(b: &mut dyn Brain, obs: &Observation) {
        b.reset(&ResetContext {
            map: obs.map.clone(),
            self_id: 0,
            seed: 1,
        });
    }

    #[test]
    fn a_brain_is_deterministic_and_reset_forgets_the_recurrent_state() {
        let cfg = RayGridConfig::default();
        let obs = room_obs();
        for net in [
            Arc::new(Mlp::new(input_dim(&cfg), 4, 3)) as Arc<dyn SeqNet>,
            Arc::new(Gru::new(input_dim(&cfg), 3, 3)),
        ] {
            let tpl = ControlTemplate::new(net, cfg, BundleMeta::default());
            let run = || {
                let mut b = tpl.instantiate();
                reset(&mut b, &obs);
                (0..5).map(|_| b.decide(&obs)).collect::<Vec<_>>()
            };
            assert_eq!(run(), run());
            let mut b = tpl.instantiate();
            reset(&mut b, &obs);
            let first: Vec<Action> = (0..3).map(|_| b.decide(&obs)).collect();
            reset(&mut b, &obs);
            let again: Vec<Action> = (0..3).map(|_| b.decide(&obs)).collect();
            assert_eq!(first, again, "reset must restore the initial state");
            assert!(b.last_latency() > Duration::ZERO);
        }
    }

    #[test]
    fn the_brain_decodes_its_logits_like_the_fly() {
        // A net whose heads are forced: bias-only outputs (zero weights) pick the action.
        let cfg = RayGridConfig::default();
        let mut m = Mlp::new(input_dim(&cfg), 2, 1);
        let head = 2 * input_dim(&cfg) + 2;
        let p = m.params_mut();
        for v in &mut p[head..head + 16] {
            *v = 0.0;
        }
        let b0 = head + 16;
        // dir logits (left, stop, right) = (0, 0, 5); jump -5; hook +5; fire -5; aim (c, s) = (0, 3): straight up.
        p[b0..b0 + 8].copy_from_slice(&[0.0, 0.0, 5.0, -5.0, 5.0, -5.0, 0.0, 3.0]);
        let tpl = ControlTemplate::new(Arc::new(m), cfg, BundleMeta::default());
        let obs = room_obs();
        let mut b = tpl.instantiate();
        reset(&mut b, &obs);
        let a = b.decide(&obs);
        assert_eq!(a.direction, 1);
        assert!(!a.jump && a.hook && !a.fire);
        assert_eq!((a.target.x, a.target.y), (0, -1000), "ring angle pi/2 is straight up");
    }

    #[test]
    fn sampled_selection_is_seeded_and_draws_from_the_probabilities() {
        // Zero weights, biases: dir (0, 0, 0) -> uniform, jump logit 0 -> p = 0.5.
        let cfg = RayGridConfig::default();
        let mut m = Mlp::new(input_dim(&cfg), 2, 1);
        let n = m.params().len();
        for v in &mut m.params_mut()[..n] {
            *v = 0.0;
        }
        let tpl = ControlTemplate::new(Arc::new(m), cfg, BundleMeta::default());
        let obs = room_obs();
        let run = |seed: u64| {
            let mut b = tpl.instantiate_with(ActionSelection::Sampled);
            b.reset(&ResetContext {
                map: obs.map.clone(),
                self_id: 0,
                seed,
            });
            (0..200).map(|_| b.decide(&obs)).collect::<Vec<_>>()
        };
        let (a, b) = (run(7), run(7));
        assert_eq!(a, b, "same seed, same draws");
        assert_ne!(a, run(8), "another seed draws differently");
        let jumps = a.iter().filter(|x| x.jump).count();
        assert!((60..=140).contains(&jumps), "p = 0.5 over 200 draws gave {jumps} jumps");
        let dirs: std::collections::BTreeSet<i32> = a.iter().map(|x| x.direction).collect();
        assert_eq!(dirs.len(), 3, "a uniform direction head visits every direction");
        // Argmax on the same net is constant.
        let mut g = tpl.instantiate();
        g.reset(&ResetContext {
            map: obs.map.clone(),
            self_id: 0,
            seed: 7,
        });
        let first = g.decide(&obs);
        assert!((0..20).all(|_| g.decide(&obs) == first));
    }

    #[test]
    fn thresholds_move_the_argmax_decision_of_each_binary_head() {
        // All-zero weights: every binary head sits at p = 0.5.
        let cfg = RayGridConfig::default();
        let obs = room_obs();
        let decide = |th: HeadThresholds| {
            let mut m = Mlp::new(input_dim(&cfg), 2, 1);
            let n = m.params().len();
            for v in &mut m.params_mut()[..n] {
                *v = 0.0;
            }
            let tpl = ControlTemplate::new(Arc::new(m), cfg, BundleMeta::default()).with_thresholds(th);
            let mut b = tpl.instantiate();
            reset(&mut b, &obs);
            b.decide(&obs)
        };
        let a = decide(HeadThresholds::default());
        assert!(a.jump && a.hook && a.fire, "p = 0.5 >= 0.5 presses");
        let a = decide(HeadThresholds {
            jump: 0.6,
            hook: 0.4,
            fire: 0.51,
        });
        assert!(!a.jump && a.hook && !a.fire, "only the lowered threshold still presses");
    }
}
