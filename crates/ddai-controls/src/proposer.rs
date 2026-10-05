//! [`ControlProposer`] (8.2b): an MLP or GRU as the *proposing* half of the hybrid brain, the control twin of
//! `ddai_fly::proposer::FlyProposer`. One `decide()` of the wrapped [`ControlBrain`] gives the head
//! probabilities of the decision; [`ddai_planner::hybrid::plans_from_distribution`] turns them into `K` plans (the
//! argmax plan plus sampled ones) that the exact search scores next to its own candidates. A proposal is only a
//! candidate: the network never decides.

use ddai_brain::{Brain, ResetContext};
use ddai_fly::rng::SplitMix64;
use ddai_planner::hybrid::{ActionDistribution, ProposeCtx, Proposer, plans_from_distribution};
use ddai_planner::planner::PlanStep;

use crate::brain::ControlBrain;

/// A control network as a [`Proposer`].
pub struct ControlProposer {
    brain: ControlBrain,
    /// A second network instance for the hook head of a model played in two views (see `HookView`).
    hook_brain: Option<ControlBrain>,
    rng: SplitMix64,
    name: String,
    /// Decisions made (cost reporting).
    pub decisions: u64,
}

impl ControlProposer {
    pub fn new(brain: ControlBrain, seed: u64) -> ControlProposer {
        let name = format!("control:{}", brain.name());
        ControlProposer {
            brain,
            hook_brain: None,
            rng: SplitMix64::new(seed),
            name,
            decisions: 0,
        }
    }

    pub fn brain(&self) -> &ControlBrain {
        &self.brain
    }

    /// A proposer for a model played in two views: `hook_brain` (over the observation with the own hook state
    /// hidden) supplies the hook probability.
    pub fn with_hook_brain(mut self, hook_brain: ControlBrain) -> ControlProposer {
        self.hook_brain = Some(hook_brain);
        self
    }
}

impl Proposer for ControlProposer {
    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self, ctx: &ResetContext) {
        self.brain.reset(ctx);
        if let Some(h) = &mut self.hook_brain {
            h.reset(ctx);
        }
        self.rng = SplitMix64::new(ctx.seed ^ 0xC0C0_C0C0);
    }

    fn propose(&mut self, ctx: &ProposeCtx<'_>, out: &mut Vec<Vec<PlanStep>>) {
        let _ = self.brain.decide(ctx.obs);
        let hook_prob = self.hook_brain.as_mut().and_then(|h| {
            let _ = h.decide(&ddai_fly::bc::mask_own_hook(ctx.obs));
            h.last_logits().map(|l| l.hook_prob())
        });
        self.decisions += 1;
        let Some(l) = self.brain.last_logits() else {
            return;
        };
        let p = l.dir_probs();
        // The ring angle of the aim head `(cos a, -sin a)` becomes the planner's `atan2(dy, dx)` (y down).
        let dist = ActionDistribution {
            direction: p.map(f64::from),
            jump: f64::from(l.jump_prob()),
            hook: f64::from(hook_prob.unwrap_or_else(|| l.hook_prob())),
            fire: f64::from(l.fire_prob()),
            aim_angle: -f64::from(l.aim_angle()),
        };
        let rng = &mut self.rng;
        plans_from_distribution(&dist, ctx.steps, ctx.k, &mut || f64::from(rng.next_f32_unit()), out);
    }
}
