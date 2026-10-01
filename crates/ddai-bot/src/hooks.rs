//! Trait hooks for task 4.2 (navigation, wayblock, trek) — no-ops for now, with the exact place in
//! the pipeline where each is called, so 4.2 fills in bodies instead of reshaping the loop.
//!
//! Pipeline (`docs/research/orig-bot.md` §4.2, steps 19-29), as [`crate::bot::Bot::on_snapshot`]
//! runs it:
//!
//! 1. [`WayBlock`] — `holding()` decides whether the wayblock mode is on (Copy Love Box only); its
//!    [`WayBlock::filter`] is consulted for every candidate inside the target selection (step 26's
//!    `wb` block), and [`WayBlock::wants_kill`] after the unstick rules (the "lying in freeze on the
//!    wayblock" rule, `WB_LYING_TICKS`/`WB_KILL_COOLDOWN_TICKS`).
//! 2. [`Navigator::drive`] — step 25: a goto/seek/home walk in progress takes over the snapshot and
//!    returns the input (or a kill) itself; `None` means "not navigating, carry on".
//! 3. target selection (ported here), then [`Trek::goal`] — step 27/30: an intermediate point the
//!    planner should head for when the target is far or behind a wall (`pathGoal`/`trekGoal`).
//! 4. the brain.
//!
//! [`Navigator::in_dead_zone`] is the TS `deadZone` grid (`route.ts`): cells from which no route leads
//! back to the game. The unstick's "trapped" rule reads it.
//!
//! [`RouteFinder`](crate::reach::RouteFinder) (reachability) is the fourth hook; its default is the
//! flood fill in [`crate::reach`].

use ddai_brain::Action;
use ddai_physics::map::MapData;
use ddai_physics::vmath::Vec2;

use crate::mapgrid::MapGrid;
use crate::players::PlayerTable;
use crate::reach::{FloodRoute, RouteFinder};
use crate::tees::{Tee, TeeSet};

/// Read-only view of the situation for a hook.
pub struct HookContext<'a> {
    pub tick: i32,
    pub own: &'a Tee,
    pub tees: &'a TeeSet,
    pub players: &'a PlayerTable,
    pub grid: &'a MapGrid,
}

/// What a navigator asks of the bot this snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NavStep {
    /// Send this input instead of deciding (the post-filters still run on it, as for a brain).
    Input(Action),
    /// `nav.takeKill()` (`bot.ts:1990`): give up and kill (subject to the kill cooldown).
    Kill,
}

/// Goto / seek / home navigation (task 4.2).
pub trait Navigator {
    /// A new map was loaded.
    fn on_map(&mut self, _map: &MapData) {}
    /// We (re)spawned (`nav.respawned()`).
    fn respawned(&mut self) {}
    /// Step 25: take over this snapshot, or `None`.
    fn drive(&mut self, _ctx: &HookContext<'_>) -> Option<NavStep> {
        None
    }
    /// The dead-zone grid query used by the unstick rules.
    fn in_dead_zone(&self, _pos: Vec2<f32>) -> bool {
        false
    }
}

/// The wayblock mode (Copy Love Box, task 4.2).
pub trait WayBlock {
    /// Wayblock mode is active (`wbHolding()`).
    fn holding(&self) -> bool {
        false
    }
    /// For each candidate of the target selection: `skip` drops it (the wayblock's own admission
    /// rules), `in_zone` adds the `+300` zone bonus.
    fn filter(&mut self, _ctx: &HookContext<'_>, _candidate: &Tee) -> WbFilter {
        WbFilter::default()
    }
    /// The "lying frozen on the wayblock" kill rule.
    fn wants_kill(&mut self, _ctx: &HookContext<'_>, _frozen_for: i32) -> bool {
        false
    }
}

/// [`WayBlock::filter`]'s verdict.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WbFilter {
    pub skip: bool,
    pub in_zone: bool,
}

/// Trek / path goals (task 4.2).
pub trait Trek {
    /// The intermediate point the brain should head for, if any (`planner.setTravelGoal`).
    fn goal(&mut self, _ctx: &HookContext<'_>, _target: &Tee) -> Option<Vec2<f32>> {
        None
    }
}

/// The no-op navigator.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoNavigator;
impl Navigator for NoNavigator {}

/// The no-op wayblock.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoWayBlock;
impl WayBlock for NoWayBlock {}

/// The no-op trek.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoTrek;
impl Trek for NoTrek {}

/// All hooks of one bot.
pub struct Hooks {
    pub navigator: Box<dyn Navigator>,
    pub wayblock: Box<dyn WayBlock>,
    pub trek: Box<dyn Trek>,
    pub route: Box<dyn RouteFinder>,
}

impl Default for Hooks {
    fn default() -> Self {
        Hooks {
            navigator: Box::new(NoNavigator),
            wayblock: Box::new(NoWayBlock),
            trek: Box::new(NoTrek),
            route: Box::<FloodRoute>::default(),
        }
    }
}
