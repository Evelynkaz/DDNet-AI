//! The PPO **actor**: the fly playing through its policy ([`ddai_fly::policy`]) and recording what the learner needs (task 8.5b).
//!
//! It is a [`Brain`] (the arena plays it like any other), built from a [`FlyBrainTemplate`], so the network is exactly the one a
//! bundle plays with: in two views when the bundle was trained with the hook head masked (a second [`FlyBrain`] on the observation
//! with the own hook state hidden decides the hook, as [`ddai_fly::two_view::TwoViewFly`] does). In `Sample` mode every head is drawn
//! from the policy (the calibrated thresholds are part of it, see [`ddai_fly::policy`]); in `Argmax` mode it plays the deterministic
//! decoding and the identity test shows it equals `ActionSelection::Argmax`.
//!
//! Every decision is pushed to a shared sink as a [`Decision`]: the observation, the sampled action and, at the start of the
//! BPTT windows of the learner, the membrane state of both views *before* the decision (R2D2's stored state). The decisions before
//! `acting_from_tick` (the burn-in of a bank start: the fly decides but the logged blocker acts) are recorded unacted: they only roll
//! the recurrent state.

use std::sync::{Arc, Mutex};

use ddai_brain::{Action, Brain, HOOK_IDLE, Observation, ResetContext, WorldView};
use ddai_fly::bc::{HeadLogits, HeadThresholds, HookView, combine_hook_view, mask_own_hook};
use ddai_fly::brain::{FlyBrain, FlyBrainConfig, aim_angle_to_target};
use ddai_fly::bundle::FlyBrainTemplate;
use ddai_fly::policy::{self, PolicyAction};
use ddai_fly::rng::SplitMix64;
use serde::{Deserialize, Serialize};

/// One recorded decision.
#[derive(Debug, Clone)]
pub struct Decision {
    pub obs: Observation,
    /// `false` for a burn-in decision (the logged action was played, this action is a placeholder).
    pub acted: bool,
    pub action: PolicyAction,
    /// The membrane state `(full view, masked view)` before this decision, at the decisions where a learner window's burn-in starts.
    pub snap: Option<Box<(Vec<f32>, Vec<f32>)>>,
}

pub type Sink = Arc<Mutex<Vec<Decision>>>;

/// How the actor picks its action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActMode {
    Sample,
    Argmax,
}

/// The learner's window grid, which tells the actor where to store the recurrent state: a window of `T = chunk` decisions with `K =
/// burn_in` decisions of burn-in before it starts at the decisions `a` with `(a + K) % T == 0`, `a = tick / decide_every`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowGrid {
    pub chunk: usize,
    pub burn_in: usize,
    pub decide_every: i32,
}

impl WindowGrid {
    /// Whether the decision at `tick` (the `index`-th recorded one) starts a burn-in segment.
    pub fn snapshot_here(&self, index: usize, tick: i32) -> bool {
        let a = (tick / self.decide_every.max(1)) as usize;
        index == 0 || (a + self.burn_in).is_multiple_of(self.chunk)
    }
}

pub struct PpoActor {
    full: FlyBrain,
    masked: Option<FlyBrain>,
    thresholds: HeadThresholds,
    aim_kappa: f32,
    temps: policy::Temperatures,
    mode: ActMode,
    rng: SplitMix64,
    acting_from_tick: i32,
    grid: WindowGrid,
    sink: Sink,
    scratch: Option<Observation>,
    recorded: usize,
    name: String,
}

impl PpoActor {
    pub fn new(
        template: &FlyBrainTemplate,
        mode: ActMode,
        aim_kappa: f32,
        temps: policy::Temperatures,
        acting_from_tick: i32,
        grid: WindowGrid,
        sink: Sink,
    ) -> PpoActor {
        let full = template.instantiate(FlyBrainConfig::default());
        let masked = (template.hook_view() == HookView::MaskedForHookHead)
            .then(|| template.instantiate(FlyBrainConfig::default()));
        PpoActor {
            thresholds: template.thresholds(),
            full,
            masked,
            aim_kappa,
            temps,
            mode,
            rng: SplitMix64::new(0),
            acting_from_tick,
            grid,
            sink,
            scratch: None,
            recorded: 0,
            name: "ppo-fly".into(),
        }
    }

    /// The played head logits of this observation (both views, the hook head from the masked one) and the states before it.
    fn logits(&mut self, obs: &Observation) -> HeadLogits {
        let lf = self.full.forward_logits(obs);
        match &mut self.masked {
            Some(m) => {
                let masked_obs = self.scratch.insert(mask_own_hook(obs));
                let lm = m.forward_logits(masked_obs);
                combine_hook_view(&lf, &lm)
            }
            None => lf,
        }
    }
}

fn to_action(a: &PolicyAction) -> Action {
    Action {
        direction: [-1, 0, 1][usize::from(a.dir.min(2))],
        jump: a.jump,
        hook: a.hook,
        fire: a.fire,
        target: aim_angle_to_target(a.aim),
        wanted_weapon: None,
    }
}

impl Brain for PpoActor {
    fn reset(&mut self, ctx: &ResetContext) {
        self.full.reset(ctx);
        if let Some(m) = &mut self.masked {
            m.reset(ctx);
        }
        self.rng = SplitMix64::new(ctx.seed ^ 0x5A17_9011_C100);
        self.recorded = 0;
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        self.decide_in(obs, None)
    }

    fn decide_in(&mut self, obs: &Observation, _view: Option<&WorldView<'_>>) -> Action {
        let snap = self.grid.snapshot_here(self.recorded, obs.tick).then(|| {
            Box::new((
                self.full.state_v().to_vec(),
                self.masked
                    .as_ref()
                    .map_or_else(|| self.full.state_v().to_vec(), |m| m.state_v().to_vec()),
            ))
        });
        let raw = self.logits(obs);
        let acted = obs.tick >= self.acting_from_tick;
        let mut action = if !acted {
            PolicyAction::default()
        } else {
            match self.mode {
                ActMode::Sample => policy::sample(
                    &policy::policy_logits(&raw, &self.thresholds, &self.temps),
                    self.aim_kappa,
                    &mut self.rng,
                ),
                ActMode::Argmax => policy::argmax(&raw, &self.thresholds),
            }
        };
        // The aim is part of the decision's probability only when it reaches the game: a throw (the observed hook state is idle) or a shot.
        action.aim_counts =
            acted && policy::counts_aim(action.hook, action.fire, obs.self_state.hook_state == HOOK_IDLE);
        self.sink.lock().expect("sink lock").push(Decision {
            obs: obs.clone(),
            acted,
            action,
            snap,
        });
        self.recorded += 1;
        if acted { to_action(&action) } else { Action::neutral() }
    }

    fn name(&self) -> &str {
        &self.name
    }
}
