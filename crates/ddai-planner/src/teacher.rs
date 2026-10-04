//! The fixed-iteration planner as a *teacher* (task 8.2, D-017): labels a state with its chosen
//! action and, where the CEM ran, the elite set's first-step statistics
//! ([`crate::elite::EliteFirstStep`]) as a soft target.
//!
//! Same search as [`crate::brains::PlannerBrain`] in [`crate::brains::PlannerMode::Fixed`] (the
//! deterministic, clock-free path: the same seed and state give the same label), with one
//! difference that matters for DAgger: the planner's "previous input" (`prev`, which its
//! flip-hysteresis and hook logic read) can be set to what was *actually executed* by whoever
//! played the last decision ([`TeacherPlanner::note_executed`]). When the student acts, the
//! teacher then answers "what would you do from here, having just done what the student did",
//! not "having just done what I decided".
//!
//! The labelling itself *is* `PlannerBrain::decide_in` (a wrapped fixed-mode [`PlannerBrain`]:
//! its world handling, roll-forward through the in-flight inputs, bystanders, boxed planner), so
//! the label equals the planner's own decision on a state by construction; this file adds the
//! `prev` override and the elite statistics, and is off the TS-parity path.

use ddai_brain::{Action, Brain, Observation, ResetContext, WorldView};

use crate::brains::{ClockKind, PlannerBrain, PlannerBrainConfig, PlannerMode, PlannerPreset, input_from_action};
use crate::elite::EliteFirstStep;
use crate::planner::DecisionInfo;
use crate::types::PlayerInput;

/// What the teacher says about one state.
#[derive(Debug, Clone, Copy)]
pub struct TeacherLabel {
    /// The planner's decision (after shield and post-processing).
    pub action: Action,
    /// The elite set's first-step statistics; `None` when the planner did not run a CEM search
    /// (no target, target or self dead, a committed decision).
    pub elite: Option<EliteFirstStep>,
    pub info: DecisionInfo,
}

/// A fixed-iteration planner that labels states. Build one per thread that uses it (it owns a
/// planner of hundreds of kB, boxed inside the wrapped [`PlannerBrain`] so rayon's 2 MB worker
/// stacks do not carry it).
pub struct TeacherPlanner {
    brain: PlannerBrain,
    prev_before_label: PlayerInput,
}

impl TeacherPlanner {
    pub fn new(preset: PlannerPreset) -> Self {
        TeacherPlanner {
            brain: PlannerBrain::new(PlannerBrainConfig {
                preset,
                mode: PlannerMode::Fixed,
                clock: ClockKind::Wall,
            }),
            prev_before_label: crate::types::empty_input(),
        }
    }

    /// Start of an episode: reseeds the search from `ctx.seed` and forgets the previous input.
    pub fn reset(&mut self, ctx: &ResetContext) {
        self.brain.reset(ctx);
        self.prev_before_label = crate::types::empty_input();
    }

    /// Tells the teacher which action was really played for the decision it just labelled
    /// (`label` assumes its own answer was played). No-op before the first `label`.
    pub fn note_executed(&mut self, played: &Action) {
        let prev = input_from_action(played, &self.prev_before_label);
        self.brain.set_prev_input(prev);
    }

    /// The plan behind the last label (task 3.7b, the loss diagnosis): the planner's best plan, `None` when the last
    /// label did not search (no target, a dead tee).
    pub fn last_plan(&mut self) -> Option<Vec<crate::planner::PlanStep>> {
        let planner = self.brain.planner_mut();
        planner.last_info.searched.then(|| planner.warm.clone()).flatten()
    }

    /// Labels the state of `view` (the exact arena world). `obs` supplies the target choice
    /// (`Observation::target_or_nearest`, like `PlannerBrain`). Without a target the label is the
    /// neutral action.
    pub fn label(&mut self, obs: &Observation, view: &WorldView<'_>) -> TeacherLabel {
        self.prev_before_label = self.brain.prev_input();
        {
            let planner = self.brain.planner_mut();
            planner.last_elite = None;
            planner.last_info = DecisionInfo::default();
        }
        let action = self.brain.decide_in(obs, Some(view));
        let planner = self.brain.planner_mut();
        TeacherLabel {
            action,
            elite: planner.last_elite,
            info: planner.last_info,
        }
    }
}
