//! The smart self-kill policy (task 4.12, D-108): kill only when a kill is truly needed.
//!
//! **Why.** The legacy rules of [`crate::unstick`] kill on fixed timers (frozen 200/400 ticks, wedged 200/450, wayblock lying 25). A
//! timer cannot tell a tee that thaws in a moment from one that never will, so it kills tees that were about to thaw, tees a friend is
//! pulling out and tees that were holding a block. [`SelfKillPolicy::Smart`] asks the physics and the room instead.
//!
//! **What it decides** ([`judge_frozen`], for a frozen tee; [`judge_wedged`], for a free one that has not moved). A kill costs the
//! respawn plus the walk back from the nearest spawn ([`kill_cost_ticks`]); it pays only when waiting costs more:
//!
//! * **no kill** while a friend (a *helper* tee: friend, ignored or clan-friend list, unfrozen, not AFK) hooks us (up to
//!   [`HELPED_LIMIT_TICKS`], the legacy bound), while such a friend is within hook reach ([`RESCUE_REACH_PX`]) for a short grace
//!   ([`FRIEND_GRACE_TICKS`], the legacy in-tile timer: time to start hooking), while somebody else hooks us (up to
//!   [`FROZEN_HARD_LIMIT_TICKS`]), while the exact
//!   passive forecast ([`ddai_planner::forecast::passive_forecast`]: our tee alone on the real physics, no input) says we thaw within the
//!   cost of a kill, or that we die anyway;
//! * **kill** when the forecast says we never thaw (deep freeze, resting on a freeze tile: no exit), when we sit in the dead zone (a respawn
//!   is needed after the thaw too), or thaw later than the cost of a
//!   kill, and we have been frozen [`SMART_MIN_FROZEN_TICKS`] (the wayblock's own request: [`WB_LYING_TICKS`]);
//! * the legacy timers stay as **upper bounds**: frozen [`FROZEN_HARD_LIMIT_TICKS`] kills whatever the forecast says (the forecast
//!   ignores the other tees; nothing can make the bot wait longer than before except a friend, bounded by [`HELPED_LIMIT_TICKS`]).
//!
//! A free tee that has not moved for the legacy wedge window is killed unless a block of ours is being held ([`judge_wedged`]).
//!
//! The cooldown ([`crate::consts::KILL_COOLDOWN_TICKS`]), the kill-protection learning and the `/kill` fallback of D-078 are not touched:
//! they sit behind this decision.

use ddai_planner::forecast::Forecast;

use crate::consts::*;

/// `--selfkill-policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelfKillPolicy {
    /// The timers of D-058/D-078 as they always were (the default until the owner switches it, D-108).
    #[default]
    Legacy,
    /// Cost/benefit: kill only when waiting costs more than a kill.
    Smart,
}

impl SelfKillPolicy {
    pub fn name(self) -> &'static str {
        match self {
            SelfKillPolicy::Legacy => "legacy",
            SelfKillPolicy::Smart => "smart",
        }
    }

    pub fn is_smart(self) -> bool {
        self == SelfKillPolicy::Smart
    }
}

impl std::str::FromStr for SelfKillPolicy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "legacy" => Ok(SelfKillPolicy::Legacy),
            "smart" => Ok(SelfKillPolicy::Smart),
            other => Err(format!("unknown self-kill policy {other:?} (legacy|smart)")),
        }
    }
}

/// Ticks from `Cl_Kill` to a new life (`sv_kill_delay` and the spawn; the server respawns at once, a little slack for the snapshot).
pub const RESPAWN_TICKS: i32 = 25;
/// How fast the walk back from the spawn is taken to be (px per tick; the run speed is ~10, a route is ~1.5 times the straight line).
pub const WALK_PX_PER_TICK: f32 = 6.0;
/// A frozen tee is never killed by the smart policy before it has been frozen this long (1 s): a freeze is 3 s at most unless renewed.
pub const SMART_MIN_FROZEN_TICKS: i32 = 50;
/// A freeze that ends within this many ticks (2 s) is never worth a death, whatever the walk back costs: a kill also loses the tee's
/// place, its target and its block.
pub const SLOW_THAW_FLOOR_TICKS: i32 = 100;
/// How far a friend can still help: the hook reach.
pub const RESCUE_REACH_PX: f32 = HOOK_LENGTH_PX;
/// A friend merely within reach (not hooking) holds a kill back only this long (the legacy in-tile timer, 4 s): time to start hooking.
/// Only an actual hook of his extends the wait to [`HELPED_LIMIT_TICKS`] (review 4.12, F3).
pub const FRIEND_GRACE_TICKS: i32 = FROZEN_IN_TILE_TICKS;
const _: () = assert!(
    FRIEND_GRACE_TICKS <= HELPED_LIMIT_TICKS,
    "a friend in reach never holds longer than the legacy bound"
);
/// The horizon of the forecast (the held-block window, 5 s: more than the 3 s a freeze lasts).
pub const FORECAST_HORIZON_TICKS: i32 = ddai_planner::forecast::HELD_HORIZON_TICKS;

/// What a kill costs: the respawn and the walk back from the nearest spawn to where we are (straight line; no spawn known: a flat 100).
pub fn kill_cost_ticks(spawns: &[(f64, f64)], pos: (f32, f32)) -> i32 {
    let walk = spawns
        .iter()
        .map(|&(x, y)| ddai_libm::hypot(f64::from(pos.0) - x, f64::from(pos.1) - y) as f32)
        .fold(f32::INFINITY, f32::min);
    if walk.is_finite() {
        RESPAWN_TICKS + (walk / WALK_PX_PER_TICK).ceil() as i32
    } else {
        100
    }
}

/// Why the smart policy did not kill or did (log line and tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmartWhy {
    // -- kill
    /// Deep frozen: no thaw, ever, without an undeep tile.
    DeepFreeze,
    /// Resting on a freeze tile (or sealed in): the freeze is renewed every second, no exit.
    NoExit,
    /// Thaws later than a kill costs.
    SlowThaw,
    /// In the dead zone (no way back to the game): a respawn is needed after the thaw anyway.
    DeadZone,
    /// The legacy timer (an upper bound) ran out.
    UpperBound,
    /// Free and wedged for the legacy window, nothing to hold.
    Wedged,
    // -- wait
    /// Frozen for less than [`SMART_MIN_FROZEN_TICKS`].
    TooEarly,
    /// A friend hooks us.
    FriendHooking,
    /// A friend is within hook reach (a short grace, [`FRIEND_GRACE_TICKS`]).
    FriendNear,
    /// Someone else hooks us.
    Hooked,
    /// The forecast says we thaw within the cost of a kill.
    ThawSoon,
    /// The forecast says we die anyway.
    DiesAnyway,
    /// A block of ours is being held.
    HoldingBlock,
    /// The forecast was not made (not frozen, or no physics yet): the legacy timers decide.
    NoForecast,
}

impl SmartWhy {
    pub fn name(self) -> &'static str {
        match self {
            SmartWhy::DeepFreeze => "deep-freeze",
            SmartWhy::NoExit => "no-exit",
            SmartWhy::SlowThaw => "slow-thaw",
            SmartWhy::DeadZone => "dead-zone",
            SmartWhy::UpperBound => "upper-bound",
            SmartWhy::Wedged => "wedged",
            SmartWhy::TooEarly => "too-early",
            SmartWhy::FriendHooking => "friend-hooking",
            SmartWhy::FriendNear => "friend-near",
            SmartWhy::Hooked => "hooked",
            SmartWhy::ThawSoon => "thaw-soon",
            SmartWhy::DiesAnyway => "dies-anyway",
            SmartWhy::HoldingBlock => "holding-block",
            SmartWhy::NoForecast => "no-forecast",
        }
    }
}

/// The verdict of one judgement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Call {
    Kill(SmartWhy),
    Wait(SmartWhy),
}

impl Call {
    pub fn is_kill(self) -> bool {
        matches!(self, Call::Kill(_))
    }
    pub fn why(self) -> SmartWhy {
        match self {
            Call::Kill(w) | Call::Wait(w) => w,
        }
    }
}

/// What [`judge_frozen`] needs.
#[derive(Debug, Clone, Copy)]
pub struct FrozenFacts {
    /// Ticks frozen so far.
    pub frozen_for: i32,
    /// The wayblock's own request (lying frozen on the wayblock) is what asks now: no minimum frozen time beyond its own.
    pub wb_request: bool,
    /// A legacy timer is due (overdue, or the stuck window): the upper bound.
    pub legacy_due: bool,
    /// A friend (helper) tee hooks us.
    pub hooked_by_helper: bool,
    /// Somebody hooks us.
    pub hooked_by_any: bool,
    /// A friend (helper tee: friend, ignored, clan friend; unfrozen, not AFK) within [`RESCUE_REACH_PX`].
    pub helper_in_reach: bool,
    pub deep_frozen: bool,
    /// In the navigator's dead zone and not hooked: no route leads back to the game from here, so a respawn is needed after the thaw too.
    pub trapped: bool,
    /// The passive forecast of our tee; `None`: not made.
    pub forecast: Option<Forecast>,
    /// What a kill costs ([`kill_cost_ticks`]).
    pub cost_ticks: i32,
}

/// The smart verdict for a **frozen** tee (see the module docs).
pub fn judge_frozen(f: &FrozenFacts) -> Call {
    let floor = if f.wb_request { 0 } else { SMART_MIN_FROZEN_TICKS };
    if f.frozen_for < floor && !f.legacy_due {
        return Call::Wait(SmartWhy::TooEarly);
    }
    if f.hooked_by_helper && f.frozen_for < HELPED_LIMIT_TICKS {
        return Call::Wait(SmartWhy::FriendHooking);
    }
    if f.helper_in_reach && f.frozen_for < FRIEND_GRACE_TICKS {
        return Call::Wait(SmartWhy::FriendNear);
    }
    if f.hooked_by_any && f.frozen_for < FROZEN_HARD_LIMIT_TICKS {
        return Call::Wait(SmartWhy::Hooked);
    }
    if f.frozen_for >= FROZEN_HARD_LIMIT_TICKS {
        return Call::Kill(SmartWhy::UpperBound);
    }
    let Some(fc) = f.forecast else {
        return if f.legacy_due {
            Call::Kill(SmartWhy::UpperBound)
        } else {
            Call::Wait(SmartWhy::NoForecast)
        };
    };
    if f.deep_frozen {
        return Call::Kill(SmartWhy::DeepFreeze);
    }
    if f.trapped {
        return Call::Kill(SmartWhy::DeadZone);
    }
    if fc.died {
        return Call::Wait(SmartWhy::DiesAnyway);
    }
    match fc.free_in {
        Some(t) if t <= f.cost_ticks.max(SLOW_THAW_FLOOR_TICKS) => Call::Wait(SmartWhy::ThawSoon),
        Some(_) => Call::Kill(SmartWhy::SlowThaw),
        None => Call::Kill(SmartWhy::NoExit),
    }
}

/// What [`judge_wedged`] needs.
#[derive(Debug, Clone, Copy)]
pub struct WedgedFacts {
    /// A block of ours is being held (a victim we froze within the held-block window and has not escaped).
    pub holding_block: bool,
}

/// The smart verdict for a **free** tee that has not left a 48 px circle for the legacy window with a target in sight.
pub fn judge_wedged(f: &WedgedFacts) -> Call {
    if f.holding_block {
        Call::Wait(SmartWhy::HoldingBlock)
    } else {
        Call::Kill(SmartWhy::Wedged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> FrozenFacts {
        FrozenFacts {
            frozen_for: 100,
            wb_request: false,
            legacy_due: false,
            hooked_by_helper: false,
            hooked_by_any: false,
            helper_in_reach: false,
            deep_frozen: false,
            trapped: false,
            forecast: Some(Forecast {
                free_in: None,
                died: false,
                steps: 1,
            }),
            cost_ticks: 80,
        }
    }

    fn free_in(t: Option<i32>) -> Option<Forecast> {
        Some(Forecast {
            free_in: t,
            died: false,
            steps: 1,
        })
    }

    #[test]
    fn a_tee_that_never_thaws_is_killed_after_the_floor_and_not_before() {
        let mut f = facts();
        f.frozen_for = SMART_MIN_FROZEN_TICKS - 1;
        assert_eq!(judge_frozen(&f), Call::Wait(SmartWhy::TooEarly));
        f.frozen_for = SMART_MIN_FROZEN_TICKS;
        assert_eq!(judge_frozen(&f), Call::Kill(SmartWhy::NoExit));
        f.deep_frozen = true;
        assert_eq!(judge_frozen(&f), Call::Kill(SmartWhy::DeepFreeze));
    }

    #[test]
    fn the_wayblock_request_has_no_floor_of_its_own() {
        let mut f = facts();
        f.frozen_for = 25;
        f.wb_request = true;
        assert_eq!(judge_frozen(&f), Call::Kill(SmartWhy::NoExit));
        f.forecast = free_in(Some(30));
        assert_eq!(judge_frozen(&f), Call::Wait(SmartWhy::ThawSoon));
    }

    #[test]
    fn the_dead_zone_kills_even_when_the_tee_would_thaw() {
        let mut f = facts();
        f.trapped = true;
        f.forecast = free_in(Some(20));
        assert_eq!(judge_frozen(&f), Call::Kill(SmartWhy::DeadZone));
        f.helper_in_reach = true;
        assert_eq!(
            judge_frozen(&f),
            Call::Wait(SmartWhy::FriendNear),
            "a friend can still pull us out of it"
        );
    }

    #[test]
    fn a_thaw_within_the_cost_of_a_kill_waits_and_a_later_one_kills() {
        let mut f = facts();
        f.cost_ticks = 180;
        f.forecast = free_in(Some(180));
        assert_eq!(
            judge_frozen(&f),
            Call::Wait(SmartWhy::ThawSoon),
            "exactly the cost: waiting is not dearer"
        );
        f.forecast = free_in(Some(181));
        assert_eq!(judge_frozen(&f), Call::Kill(SmartWhy::SlowThaw));
    }

    #[test]
    fn a_thaw_within_two_seconds_waits_however_cheap_the_kill_is() {
        let mut f = facts();
        f.cost_ticks = RESPAWN_TICKS; // standing on the spawn
        f.forecast = free_in(Some(SLOW_THAW_FLOOR_TICKS));
        assert_eq!(judge_frozen(&f), Call::Wait(SmartWhy::ThawSoon));
        f.forecast = free_in(Some(SLOW_THAW_FLOOR_TICKS + 1));
        assert_eq!(judge_frozen(&f), Call::Kill(SmartWhy::SlowThaw));
    }

    #[test]
    fn a_friend_who_hooks_us_is_waited_for_up_to_the_legacy_bound() {
        let mut f = facts();
        f.hooked_by_helper = true;
        f.hooked_by_any = true;
        assert_eq!(judge_frozen(&f), Call::Wait(SmartWhy::FriendHooking));
        f.frozen_for = 399;
        assert_eq!(
            judge_frozen(&f),
            Call::Wait(SmartWhy::FriendHooking),
            "even past the legacy 400"
        );
        f.frozen_for = HELPED_LIMIT_TICKS - 1;
        assert_eq!(judge_frozen(&f), Call::Wait(SmartWhy::FriendHooking));
        f.frozen_for = HELPED_LIMIT_TICKS;
        assert_eq!(judge_frozen(&f), Call::Kill(SmartWhy::UpperBound));
    }

    /// Review 4.12 F3: a friend merely within reach is a short grace (time to start hooking), not 1500 ticks.
    #[test]
    fn a_friend_merely_in_reach_gets_only_a_short_grace() {
        let mut f = facts();
        f.helper_in_reach = true;
        f.frozen_for = SMART_MIN_FROZEN_TICKS;
        assert_eq!(judge_frozen(&f), Call::Wait(SmartWhy::FriendNear));
        f.frozen_for = FRIEND_GRACE_TICKS - 1;
        assert_eq!(judge_frozen(&f), Call::Wait(SmartWhy::FriendNear));
        f.frozen_for = FRIEND_GRACE_TICKS;
        assert_eq!(
            judge_frozen(&f),
            Call::Kill(SmartWhy::NoExit),
            "the grace is over: a hopeless freeze is killed"
        );
        // Once he hooks, the long wait applies.
        f.hooked_by_helper = true;
        f.hooked_by_any = true;
        assert_eq!(judge_frozen(&f), Call::Wait(SmartWhy::FriendHooking));
    }

    #[test]
    fn somebody_elses_hook_waits_until_the_hard_limit() {
        let mut f = facts();
        f.hooked_by_any = true;
        assert_eq!(judge_frozen(&f), Call::Wait(SmartWhy::Hooked));
        f.frozen_for = FROZEN_HARD_LIMIT_TICKS;
        assert_eq!(judge_frozen(&f), Call::Kill(SmartWhy::UpperBound));
    }

    #[test]
    fn the_hard_limit_kills_whatever_the_forecast_says() {
        let mut f = facts();
        f.frozen_for = FROZEN_HARD_LIMIT_TICKS;
        f.forecast = free_in(Some(5));
        assert_eq!(judge_frozen(&f), Call::Kill(SmartWhy::UpperBound));
    }

    #[test]
    fn without_a_forecast_only_a_due_legacy_timer_kills() {
        let mut f = facts();
        f.forecast = None;
        assert_eq!(judge_frozen(&f), Call::Wait(SmartWhy::NoForecast));
        f.legacy_due = true;
        assert_eq!(judge_frozen(&f), Call::Kill(SmartWhy::UpperBound));
    }

    #[test]
    fn a_tee_that_dies_anyway_is_not_killed() {
        let mut f = facts();
        f.forecast = Some(Forecast {
            free_in: None,
            died: true,
            steps: 3,
        });
        assert_eq!(judge_frozen(&f), Call::Wait(SmartWhy::DiesAnyway));
    }

    #[test]
    fn a_held_block_keeps_a_wedged_tee_alive() {
        assert_eq!(
            judge_wedged(&WedgedFacts { holding_block: true }),
            Call::Wait(SmartWhy::HoldingBlock)
        );
        assert_eq!(
            judge_wedged(&WedgedFacts { holding_block: false }),
            Call::Kill(SmartWhy::Wedged)
        );
    }

    #[test]
    fn the_cost_of_a_kill_grows_with_the_walk_back() {
        let spawns = [(100.0, 100.0), (4000.0, 100.0)];
        assert_eq!(kill_cost_ticks(&spawns, (100.0, 100.0)), RESPAWN_TICKS);
        assert_eq!(kill_cost_ticks(&spawns, (700.0, 100.0)), RESPAWN_TICKS + 100);
        assert_eq!(
            kill_cost_ticks(&spawns, (3700.0, 100.0)),
            RESPAWN_TICKS + 50,
            "the nearer spawn counts"
        );
        assert_eq!(kill_cost_ticks(&[], (0.0, 0.0)), 100, "no spawn known");
    }

    #[test]
    fn the_policy_name_parses_both_ways() {
        for p in [SelfKillPolicy::Legacy, SelfKillPolicy::Smart] {
            assert_eq!(p.name().parse::<SelfKillPolicy>(), Ok(p));
        }
        assert!("Smart".parse::<SelfKillPolicy>().is_err());
        assert_eq!(SelfKillPolicy::default(), SelfKillPolicy::Legacy);
    }
}
