//! [`IdleBrain`]: the do-nothing controller (task 8.1) — the control in arena baselines and the
//! "doing nothing must not win" check of the technique scenarios.

use crate::action::Action;
use crate::brain::{Brain, ResetContext};
use crate::observation::Observation;

/// Always returns [`Action::neutral`]. Stateless, so `reset` is a no-op.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdleBrain;

impl Brain for IdleBrain {
    fn reset(&mut self, _ctx: &ResetContext) {}

    fn decide(&mut self, _obs: &Observation) -> Action {
        Action::neutral()
    }

    fn name(&self) -> &str {
        "idle"
    }
}
