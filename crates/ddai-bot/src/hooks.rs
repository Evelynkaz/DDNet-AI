//! Trait hooks of navigation, wayblock and trek (task 4.2; `crate::nav_hooks` has the real bodies, the
//! `No*` types here are the no-ops the bot's own tests run with), with the exact place in the pipeline
//! where each is called (`docs/research/orig-bot.md` §4.2, steps 19-29), as
//! [`crate::bot::Bot::on_snapshot`] runs it:
//!
//! 1. [`Navigator::poll`] — pending commands, the end of a walk, the freeze memory; may change the
//!    mode (a goto begins: `goto`; it ends: back to the mode it started from) and hands the brain
//!    fresh [`MapKnowledge`].
//! 2. [`WayBlock`] — `holding()` decides whether the wayblock mode is on (Copy Love Box only); its
//!    [`WayBlock::filter`] is consulted for every candidate inside the target selection (step 26's
//!    `wb` block), and [`WayBlock::wants_kill`] after the unstick rules (the "lying in freeze on the
//!    wayblock" rule, `WB_LYING_TICKS`/`WB_KILL_COOLDOWN_TICKS`).
//! 3. [`Navigator::drive`] — step 25 (`driveNav`): a goto/follow walk in progress takes over the
//!    snapshot and returns the input (or a kill) itself; `None` means "not navigating, carry on".
//!    The bot guards the input unless the hook says not to, and tells it about vetoes and kills.
//! 4. target selection (ported here), then [`Trek::steer`] (the walk-to-the-game / seek / home rules)
//!    and, for the brain, [`Trek::goal`] — step 27/30: an intermediate point the planner should head
//!    for when the target is far or behind a wall (`pathGoal`/`trekGoal`), plus
//!    [`WayBlock::brain_hints`].
//! 5. the brain.
//!
//! [`Navigator::in_dead_zone`] is the TS `deadZone` grid (`route.ts`): cells from which no route leads
//! back to the game. The unstick's "trapped" rule reads it.
//!
//! [`RouteFinder`](crate::reach::RouteFinder) (reachability) is the fourth hook; its default is the
//! flood fill in [`crate::reach`].

use std::sync::Arc;

use ddai_brain::{Action, MapKnowledge, WbHints};
use ddai_physics::map::MapData;
use ddai_physics::vmath::Vec2;
use ddai_physics::world::World;

use crate::activity::ActivityClock;
use crate::bot::Mode;
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
    /// The activity clock (AFK, frozen-since, "at us" memory).
    pub clock: &'a ActivityClock,
    /// The reconstructed world at the snapshot tick.
    pub world: &'a World<f32>,
    /// Ticks between this snapshot and the earliest tick a new input can take effect (the input lag
    /// the TS navigator was told as `lagTicks()`).
    pub lag_ticks: i32,
    /// The bot's current mode.
    pub mode: Mode,
}

/// Which map is loaded: the name decides which wayblock applies, the hash keys the freeze memory.
#[derive(Debug, Clone, Default)]
pub struct MapIdent {
    pub name: String,
    pub sha256: [u8; 32],
}

/// What a navigator asks of the bot this snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NavStep {
    /// Send this input instead of deciding. `guard`: run the shield on it first (`driveNav` guards the
    /// input unless the navigator is crossing a tube or walking a planned freeze); a changed input is
    /// reported back through [`Navigator::vetoed`].
    Input { action: Action, guard: bool },
    /// `nav.takeKill()` (`bot.ts:1990`): give up and kill (subject to the kill cooldown); the action is
    /// sent meanwhile. The bot reports a kill that went out through [`Navigator::kill_sent`].
    Kill { action: Action },
}

/// What [`Navigator::poll`] hands back.
#[derive(Debug, Default)]
pub struct Poll {
    /// Change the bot's mode (a goto began or ended).
    pub mode: Option<Mode>,
    /// New map knowledge (dead zone, freeze memory) for the brain.
    pub knowledge: Option<MapKnowledge>,
}

/// Goto / seek / home navigation (task 4.2).
pub trait Navigator {
    /// A new map was loaded.
    fn on_map(&mut self, _map: &Arc<MapData>, _ident: &MapIdent) {}
    /// The map is being replaced: forget what only made sense on it (a goto, a trek, the home), save.
    fn on_map_changing(&mut self) {}
    /// We (re)spawned (`nav.respawned()`).
    fn respawned(&mut self) {}
    /// First thing each snapshot while our tee lives: pending commands, the end of a walk, the freeze
    /// memory. Returns a mode change and fresh map knowledge for the brain, if any.
    fn poll(&mut self, _ctx: &HookContext<'_>) -> Poll {
        Poll::default()
    }
    /// Step 25: take over this snapshot, or `None`.
    fn drive(&mut self, _ctx: &HookContext<'_>) -> Option<NavStep> {
        None
    }
    /// The shield changed the navigator's input (`nav.vetoed()`).
    fn vetoed(&mut self) {}
    /// A `Cl_Kill` of ours went out at `tick` (TS `lastKillTick`: the unstick's and the trek's too, so a
    /// followed walk does not count them as deaths on the way). `by_route` is true when the navigator
    /// itself asked for it (a respawn step of its route: TS `routeKillTick`).
    fn kill_sent(&mut self, _tick: i32, _by_route: bool) {}
    /// `SV_KILLMSG`: `victim` died (ours and a followed player's deaths end walks).
    fn on_kill(&mut self, _victim: i32, _own_id: i32, _tick: i32) {}
    /// The dead-zone grid query used by the unstick rules.
    fn in_dead_zone(&self, _pos: Vec2<f32>) -> bool {
        false
    }
    /// The run ends: save what is kept on disk.
    fn stop(&mut self) {}
}

/// The wayblock mode (Copy Love Box, task 4.2).
pub trait WayBlock {
    /// Wayblock mode is active (`wbHolding()`).
    fn holding(&self) -> bool {
        false
    }
    /// For each candidate of the target selection: `skip` drops it (the wayblock's own admission
    /// rules), `in_zone` adds the `+300` zone bonus, `finish_zone` marks a frozen tee in the zone that
    /// we hold from inside the hall (`wbFinish`: kept as a finishing target unless sealed).
    fn filter(&mut self, _ctx: &HookContext<'_>, _candidate: &Tee) -> WbFilter {
        WbFilter::default()
    }
    /// The "lying frozen on the wayblock" kill rule.
    fn wants_kill(&mut self, _ctx: &HookContext<'_>, _frozen_for: i32) -> bool {
        false
    }
    /// What the brain is told about the wayblock this decision (`wbBand`, `wbPlanOverrides`).
    fn brain_hints(&mut self, _ctx: &HookContext<'_>) -> WbHints {
        WbHints::default()
    }
    /// With nobody to fight: where to stand and what to watch (`wander(…, anchorX, lookAt)`).
    fn wander_hint(&mut self, _ctx: &HookContext<'_>) -> Option<WanderHint> {
        None
    }
}

/// [`WayBlock::wander_hint`]'s answer, pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WanderHint {
    pub anchor_x: f32,
    pub look_at: Option<(f32, f32)>,
}

/// [`WayBlock::filter`]'s verdict.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WbFilter {
    pub skip: bool,
    pub in_zone: bool,
    pub finish_zone: bool,
}

/// Trek / path goals (task 4.2).
pub trait Trek {
    /// The intermediate point the brain should head for, if any (`planner.setTravelGoal`).
    fn goal(&mut self, _ctx: &HookContext<'_>, _target: &Tee) -> Option<Vec2<f32>> {
        None
    }
    /// After the target pick, when nothing navigates: the walk-to-the-game, seek and home rules
    /// (`bot.ts:2519-2576`). `target` is the picked id or -1. May start a navigation (the next
    /// snapshot's [`Navigator::drive`] takes over).
    fn steer(&mut self, _ctx: &HookContext<'_>, _target: i32) {}
    /// A trek passed a respawn step: send `Cl_Kill` (subject to the kill cooldown).
    fn take_kill(&mut self) -> bool {
        false
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
