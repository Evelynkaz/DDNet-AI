//! Getting unstuck — `maybeUnstick` (`bot.ts:4574-4693`, `docs/research/orig-bot.md` §8.6) with the
//! exact constants ([`crate::consts`]), but the kill is the protocol message **`Cl_Kill`**
//! (`NETMSGTYPE_CL_KILL`, sent through [`ddai_client::Client::kill`]) and **never** `/kill` in chat
//! (D-007; the client API has no chat path at all).
//!
//! A kill is requested when, with the bot acting and (free or frozen):
//!
//! 1. **overdue**: frozen `>= 400` ticks, or frozen `>= 200` while standing with a body corner in a
//!    freeze tile and not hooked, or frozen `>= 75` in the navigator's dead zone while not hooked —
//!    unless a helper (a friend/ignored/clan-friend tee, unfrozen, within 140 px) is near, in which
//!    case only after `HELPED_LIMIT_TICKS` (1500); or
//! 2. the wayblock hook says so (the "lying frozen on the wayblock" rule, own cooldown 100), or
//! 3. **stuck**: the tee stayed within 48 px of an anchor for 450 ticks while frozen (200 while free
//!    *and* with a target). Exceptions: someone hooks us; a helper is near (for < 1500 ticks); the
//!    frozen tee is not actually on a freeze tile; free and the held target is frozen and within
//!    hook reach or roped to us.
//!
//! Every kill honours `KILL_COOLDOWN_TICKS` (500) since the previous one.

use ddai_physics::vmath::Vec2;

use crate::consts::*;
use crate::mapgrid::MapGrid;
use crate::players::PlayerTable;
use crate::smartkill::{self, Call, FrozenFacts, SelfKillPolicy, SmartWhy, WedgedFacts};
use crate::tees::{Tee, TeeSet, dist};
use ddai_planner::forecast::Forecast;

const NEVER: i32 = i32::MIN / 2;

/// Why a kill was requested (logged; tests assert on it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillReason {
    /// Frozen too long / standing in freeze / trapped in the dead zone.
    Overdue,
    /// The wayblock hook's own rule.
    WayBlockLying,
    /// No progress for the stuck window.
    Stuck,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    None,
    Kill(KillReason),
}

/// Everything one step reads.
pub struct UnstickCtx<'a> {
    pub tick: i32,
    pub own: &'a Tee,
    pub tees: &'a TeeSet,
    pub players: &'a PlayerTable,
    pub grid: &'a MapGrid,
    /// The current target (-1 none) — a free tee with no target is never "wedged".
    pub target: i32,
    /// `acting`: false in hold mode (the anchor is dropped and nothing is checked).
    pub acting: bool,
    /// `Navigator::in_dead_zone(own.pos)`.
    pub in_dead_zone: bool,
    /// `WayBlock::wants_kill`.
    pub wayblock_wants_kill: bool,
    /// Task 4.12 (the smart policy only): the passive forecast of our tee ([`Unstick::wants_forecast`] says when it is worth making).
    pub forecast: Option<Forecast>,
    /// Task 4.12: what a kill costs in ticks ([`smartkill::kill_cost_ticks`]).
    pub cost_ticks: i32,
    /// Task 4.12: a block of ours is being held ([`crate::activity::ActivityClock::holding_block`]).
    pub holding_block: bool,
}

#[derive(Debug, Clone, Copy)]
struct Anchor {
    pos: Vec2<f32>,
    tick: i32,
    frozen: bool,
}

pub struct Unstick {
    frozen_since: i32,
    anchor: Option<Anchor>,
    last_kill_tick: i32,
    kills: u32,
    /// `--no-selfkill` (D-102): no kill verdict is ever returned and nothing is recorded as fired (the cooldown is not started).
    no_kill: bool,
    /// Task 4.12 (`--selfkill-policy`).
    policy: SelfKillPolicy,
    /// The smart policy held back a kill the legacy timers would have sent: `(tick, why)`, taken by [`Unstick::take_skip`].
    skip: Option<(i32, SmartWhy)>,
    last_skip_tick: i32,
    /// The why of the last kill verdict of the smart policy.
    last_why: Option<SmartWhy>,
    /// The smart policy's call at the last [`Unstick::step`] that judged a frozen tee (for the replay tool's trace).
    last_call: Option<Call>,
}

impl Default for Unstick {
    fn default() -> Self {
        Self::new()
    }
}

impl Unstick {
    pub fn new() -> Self {
        Unstick {
            frozen_since: -1,
            anchor: None,
            last_kill_tick: NEVER,
            kills: 0,
            no_kill: false,
            policy: SelfKillPolicy::Legacy,
            skip: None,
            last_skip_tick: NEVER,
            last_why: None,
            last_call: None,
        }
    }

    /// `--selfkill-policy` (task 4.12, D-108).
    pub fn set_policy(&mut self, policy: SelfKillPolicy) {
        self.policy = policy;
    }

    pub fn policy(&self) -> SelfKillPolicy {
        self.policy
    }

    /// Whether the smart policy needs the passive forecast of our tee at `tick` (it is frozen long enough for a kill to be asked for).
    /// Not asked for when no kill could be sent anyway (the duel switch is on, or the kill cooldown still runs: 500 ticks, the wayblock
    /// request's own 100 when `wb_request`), so a frozen bot in a duel pays nothing for the policy (review 4.12, F4).
    pub fn wants_forecast(&self, tick: i32, own: &Tee, wb_request: bool) -> bool {
        self.policy.is_smart()
            && !self.no_kill
            && (if wb_request {
                tick - self.last_kill_tick >= WB_KILL_COOLDOWN_TICKS
            } else {
                self.cooldown_ready(tick)
            })
            && own.frozen
            && self.frozen_for(tick, own) >= WB_LYING_TICKS
    }

    /// A kill the smart policy held back since the last call (rate limited: one per 100 ticks), for the log.
    pub fn take_skip(&mut self) -> Option<(i32, SmartWhy)> {
        self.skip.take()
    }

    /// The smart policy's call at the last step (`None`: not judged: not frozen, cooldown, the duel switch).
    pub fn last_call(&self) -> Option<Call> {
        self.last_call
    }

    /// Why the smart policy sent the last kill verdict (`None` for the legacy policy).
    pub fn last_why(&self) -> Option<SmartWhy> {
        self.last_why
    }

    fn note_skip(&mut self, tick: i32, why: SmartWhy) {
        if tick - self.last_skip_tick >= 100 {
            self.last_skip_tick = tick;
            self.skip = Some((tick, why));
        }
    }

    /// `--no-selfkill` (D-102): never return [`Verdict::Kill`]. Only the three `fire()` / `Verdict::Kill` returns are gated: the frozen
    /// clock and the anchor keep running, so lifting the switch finds them current (no kill on a stale anchor), and the kill cooldown
    /// is not started, so the owner's `!kill` is not held up by a kill that was never sent.
    pub fn set_no_kill(&mut self, off: bool) {
        self.no_kill = off;
    }

    /// Kills requested so far (`stats.selfKills`).
    pub fn kills(&self) -> u32 {
        self.kills
    }

    pub fn last_kill_tick(&self) -> i32 {
        self.last_kill_tick
    }

    /// Forget tick-dependent state: the frozen clock, the anchor **and the kill cooldown** (a game
    /// tick that went backwards makes the old tick numbers meaningless). Safe: any kill needs at
    /// least 200 ticks of its conditions after a reset, so a reset cannot cause kill spam.
    pub fn reset_ticks(&mut self) {
        self.frozen_since = -1;
        self.anchor = None;
        self.last_kill_tick = NEVER;
    }

    /// Records that a kill was sent for a reason outside this state machine (a navigator's), so the
    /// cooldown covers it too.
    pub fn note_external_kill(&mut self, tick: i32) {
        self.last_kill_tick = tick;
        self.anchor = None;
        self.frozen_since = -1;
        self.kills += 1;
    }

    /// Whether a kill is allowed now (`KILL_COOLDOWN_TICKS` since the last).
    pub fn cooldown_ready(&self, tick: i32) -> bool {
        tick - self.last_kill_tick >= KILL_COOLDOWN_TICKS
    }

    /// Ticks the bot has been frozen (as of the previous [`Unstick::step`], which is what updates the
    /// clock): the duration the wayblock kill hook gets, evaluated before this snapshot's step.
    pub fn frozen_for(&self, tick: i32, own: &Tee) -> i32 {
        if own.frozen && self.frozen_since >= 0 {
            tick - self.frozen_since
        } else {
            0
        }
    }

    pub fn step(&mut self, c: &UnstickCtx<'_>) -> Verdict {
        let (tick, me) = (c.tick, c.own);
        self.last_why = None;
        self.last_call = None;
        if !me.frozen {
            self.frozen_since = -1;
        } else if self.frozen_since < 0 {
            self.frozen_since = tick;
        }
        let frozen_for = if me.frozen && self.frozen_since >= 0 {
            tick - self.frozen_since
        } else {
            0
        };
        if !c.acting {
            self.anchor = None;
            return Verdict::None;
        }
        if !me.frozen && c.target == -1 {
            self.anchor = None;
            return Verdict::None;
        }

        let in_tiles = me.frozen
            && [-HALF_TEE_PX, HALF_TEE_PX].iter().any(|&dx| {
                [-HALF_TEE_PX, HALF_TEE_PX]
                    .iter()
                    .any(|&dy| c.grid.is_freeze(me.pos.x + dx, me.pos.y + dy))
            });
        let hooked = c.tees.iter().any(|o| o.id != me.id && o.hooked_player == me.id);
        let trapped = !hooked && c.in_dead_zone;
        let helped = helper_near(me, c.tees, c.players);
        let overdue_raw = frozen_for >= FROZEN_HARD_LIMIT_TICKS
            || (in_tiles && !hooked && frozen_for >= FROZEN_IN_TILE_TICKS)
            || (trapped && frozen_for >= TRAPPED_TICKS);
        let overdue = overdue_raw && (!helped || frozen_for >= HELPED_LIMIT_TICKS);
        let smart = self.policy.is_smart();

        if !smart || !me.frozen {
            if c.wayblock_wants_kill && !self.no_kill && tick - self.last_kill_tick >= WB_KILL_COOLDOWN_TICKS {
                self.fire(tick, true);
                return Verdict::Kill(KillReason::WayBlockLying);
            }
            if overdue && !self.no_kill && self.cooldown_ready(tick) {
                self.fire(tick, true);
                return Verdict::Kill(KillReason::Overdue);
            }
        }

        let anchor_moved = match self.anchor {
            None => true,
            Some(a) => dist(me.pos, a.pos) > STUCK_RADIUS_PX || me.frozen != a.frozen,
        };
        if anchor_moved {
            self.anchor = Some(Anchor {
                pos: me.pos,
                tick,
                frozen: me.frozen,
            });
            if !(smart && me.frozen) {
                return Verdict::None;
            }
        }
        let anchor = self.anchor.expect("anchor is set: anchor_moved was false");
        let stuck_ticks = tick - anchor.tick;

        if smart && me.frozen {
            return self.smart_frozen(c, frozen_for, overdue_raw, stuck_ticks, hooked);
        }

        if stuck_ticks
            < if me.frozen {
                STUCK_FROZEN_TICKS
            } else {
                STUCK_WEDGED_TICKS
            }
        {
            return Verdict::None;
        }
        if me.frozen {
            if hooked {
                return Verdict::None;
            }
            if helped && stuck_ticks < HELPED_LIMIT_TICKS {
                return Verdict::None;
            }
            if !c.grid.is_freeze(me.pos.x, me.pos.y) {
                return Verdict::None;
            }
        } else if let Some(held) = c.tees.get(c.target)
            && held.frozen
            && (me.hooked_player == held.id || dist(me.pos, held.pos) < HOOK_LENGTH_PX)
        {
            return Verdict::None;
        }
        if self.no_kill || !self.cooldown_ready(tick) {
            return Verdict::None;
        }
        if smart {
            // A free tee wedged for the legacy window: killed unless a block of ours is being held.
            let call = smartkill::judge_wedged(&WedgedFacts {
                holding_block: c.holding_block,
            });
            if !call.is_kill() {
                self.note_skip(tick, call.why());
                return Verdict::None;
            }
            self.last_why = Some(call.why());
        }
        self.fire(tick, false);
        Verdict::Kill(KillReason::Stuck)
    }

    /// The smart policy for a frozen tee (task 4.12): the legacy timers as upper bounds, [`smartkill::judge_frozen`] in between.
    fn smart_frozen(
        &mut self,
        c: &UnstickCtx<'_>,
        frozen_for: i32,
        overdue_raw: bool,
        stuck_ticks: i32,
        hooked: bool,
    ) -> Verdict {
        let (tick, me) = (c.tick, c.own);
        let wb_request = c.wayblock_wants_kill;
        let stuck_due = stuck_ticks >= STUCK_FROZEN_TICKS && !hooked && c.grid.is_freeze(me.pos.x, me.pos.y);
        let legacy_due = wb_request || overdue_raw || stuck_due;
        let cooldown_ok = if wb_request {
            tick - self.last_kill_tick >= WB_KILL_COOLDOWN_TICKS
        } else {
            self.cooldown_ready(tick)
        };
        if self.no_kill || !cooldown_ok {
            return Verdict::None;
        }
        let (hooked_by_helper, hooked_by_any, helper_in_reach) = rescuers(me, c.tees, c.players);
        let call = smartkill::judge_frozen(&FrozenFacts {
            frozen_for,
            wb_request,
            legacy_due,
            hooked_by_helper,
            hooked_by_any,
            helper_in_reach,
            deep_frozen: me.deep_frozen,
            trapped: !hooked && c.in_dead_zone,
            forecast: c.forecast,
            cost_ticks: c.cost_ticks,
        });
        self.last_call = Some(call);
        match call {
            Call::Kill(why) => {
                self.last_why = Some(why);
                self.fire(tick, true);
                Verdict::Kill(if wb_request {
                    KillReason::WayBlockLying
                } else {
                    KillReason::Overdue
                })
            }
            Call::Wait(why) => {
                if legacy_due && why != SmartWhy::TooEarly {
                    self.note_skip(tick, why);
                }
                Verdict::None
            }
        }
    }

    fn fire(&mut self, tick: i32, reset_frozen: bool) {
        self.last_kill_tick = tick;
        self.anchor = None;
        if reset_frozen {
            self.frozen_since = -1;
        }
        self.kills += 1;
    }
}

/// Task 4.12: who could pull a frozen tee out: `(a friend hooks us, somebody hooks us, a friend is within hook reach)`. A friend is a
/// helper tee ([`crate::players::PlayerSlot::flags`]: friend, ignored, clan friend), alive, unfrozen and not AFK or paused.
pub fn rescuers(me: &Tee, tees: &TeeSet, players: &PlayerTable) -> (bool, bool, bool) {
    let (mut by_helper, mut by_any, mut near) = (false, false, false);
    for o in tees.iter().filter(|o| o.id != me.id) {
        let slot = players.get(o.id);
        let helper = slot.is_some_and(|s| s.flags.helper());
        if o.hooked_player == me.id {
            by_any = true;
            by_helper |= helper;
        }
        // A friend who is AFK or paused is no help.
        let active = slot.is_some_and(|s| !s.server_afk() && !s.not_playing());
        if helper && active && !o.frozen && dist(o.pos, me.pos) <= smartkill::RESCUE_REACH_PX {
            near = true;
        }
    }
    (by_helper, by_any, near)
}

/// `helperNear` (`bot.ts:4418-4432`): a friend/ignored/clan-friend tee, alive, unfrozen, within
/// 140 px.
pub fn helper_near(me: &Tee, tees: &TeeSet, players: &PlayerTable) -> bool {
    tees.iter().any(|o| {
        o.id != me.id
            && !o.frozen
            && dist(o.pos, me.pos) <= HELPER_RANGE_PX
            && players.get(o.id).is_some_and(|s| s.flags.helper())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapgrid::test_maps::*;
    use crate::players::test_support::player;
    use crate::relations::{ListKind, Relations};

    struct F {
        u: Unstick,
        tees: TeeSet,
        players: PlayerTable,
        grid: MapGrid,
    }

    const FREEZE_TX: u32 = 5;

    impl F {
        fn new() -> Self {
            // 12x12 room with one freeze tile at (5,5).
            let grid = MapGrid::new(&room(12, 12, &[(FREEZE_TX, 5, FREEZE)]));
            let mut players = PlayerTable::new([0; 16]);
            players.update(
                &[
                    player(0, "me", "", true, 0, Some(0)),
                    player(1, "other", "", false, 0, Some(0)),
                    player(2, "pal", "", false, 0, Some(0)),
                ],
                &Relations::new(),
            );
            let mut f = F {
                u: Unstick::new(),
                tees: TeeSet::new(),
                players,
                grid,
            };
            f.put(0, 5.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0, false); // on the freeze tile
            f.put(1, 300.0, 300.0, false);
            f
        }
        fn put(&mut self, id: i32, x: f32, y: f32, frozen: bool) {
            self.tees.set_for_test(Tee {
                id,
                alive: true,
                pos: Vec2::new(x, y),
                frozen,
                freeze_ticks_left: if frozen { 100 } else { 0 },
                ..Tee::DEAD
            });
        }
        fn set_frozen(&mut self, frozen: bool) {
            let mut t = *self.tees.get(0).unwrap();
            t.frozen = frozen;
            self.tees.set_for_test(t);
        }
        fn step_with(&mut self, tick: i32, target: i32, acting: bool, dead: bool, wb: bool) -> Verdict {
            let own = *self.tees.get(0).unwrap();
            self.u.step(&UnstickCtx {
                tick,
                own: &own,
                tees: &self.tees,
                players: &self.players,
                grid: &self.grid,
                target,
                acting,
                in_dead_zone: dead,
                wayblock_wants_kill: wb,
                forecast: None,
                cost_ticks: 100,
                holding_block: false,
            })
        }
        fn step(&mut self, tick: i32, target: i32) -> Verdict {
            self.step_with(tick, target, true, false, false)
        }
        /// A smart-policy step: `forecast` is what the physics says, `dead` the dead zone, `holding` a block being held.
        fn step_smart(
            &mut self,
            tick: i32,
            target: i32,
            forecast: Option<Forecast>,
            dead: bool,
            holding: bool,
        ) -> Verdict {
            let own = *self.tees.get(0).unwrap();
            self.u.step(&UnstickCtx {
                tick,
                own: &own,
                tees: &self.tees,
                players: &self.players,
                grid: &self.grid,
                target,
                acting: true,
                in_dead_zone: dead,
                wayblock_wants_kill: false,
                forecast,
                cost_ticks: 100,
                holding_block: holding,
            })
        }
    }

    /// Frozen for `n` ticks, sampled every 2 ticks, never on a freeze tile unless the fixture says so.
    fn run_frozen(f: &mut F, from: i32, to: i32, target: i32) -> Option<i32> {
        for tick in (from..=to).step_by(2) {
            if let Verdict::Kill(_) = f.step(tick, target) {
                return Some(tick);
            }
        }
        None
    }

    #[test]
    fn frozen_in_the_tiles_kills_after_200_ticks_not_before() {
        let mut f = F::new();
        f.set_frozen(true);
        // A body corner (+-14 px) is in the freeze tile: `in_tiles`; overdue at 200 ticks frozen.
        let at = run_frozen(&mut f, 1000, 1400, 1).expect("a kill within the window");
        assert_eq!(at, 1200, "frozen since 1000, 200 ticks standing in freeze");
        assert_eq!(f.u.kills(), 1);
    }

    #[test]
    fn frozen_outside_any_tile_waits_for_the_hard_limit_of_400() {
        let mut f = F::new();
        f.put(0, 2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0, true); // no freeze tile near
        assert_eq!(
            run_frozen(&mut f, 0, 398, 1),
            None,
            "anchor rule needs 450 + a freeze tile"
        );
        assert_eq!(run_frozen(&mut f, 400, 402, 1), Some(400), "FROZEN_HARD_LIMIT_TICKS");
    }

    #[test]
    fn being_hooked_by_someone_prevents_the_in_tile_kill_and_the_stuck_kill() {
        let mut f = F::new();
        f.set_frozen(true);
        let mut o = *f.tees.get(1).unwrap();
        o.hooked_player = 0;
        f.tees.set_for_test(o);
        // in_tiles && !hooked is false; hard limit at 400 still applies (overdue).
        assert_eq!(run_frozen(&mut f, 0, 398, 1), None);
        assert_eq!(
            run_frozen(&mut f, 400, 402, 1),
            Some(400),
            "the 400-tick limit ignores a hook"
        );
    }

    #[test]
    fn a_friend_nearby_delays_the_kill_to_1500_ticks() {
        let mut f = F::new();
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "pal");
        f.players.update(
            &[
                player(0, "me", "", true, 0, Some(0)),
                player(1, "other", "", false, 0, Some(0)),
                player(2, "pal", "", false, 0, Some(0)),
            ],
            &rel,
        );
        f.set_frozen(true);
        f.put(2, 5.0 * 32.0 + 16.0 + 60.0, 5.0 * 32.0 + 16.0, false);
        assert_eq!(run_frozen(&mut f, 0, 1498, 1), None, "a helper is near: patience");
        assert_eq!(run_frozen(&mut f, 1500, 1502, 1), Some(1500), "HELPED_LIMIT_TICKS");
    }

    #[test]
    fn a_frozen_friend_is_no_helper_and_a_far_friend_neither() {
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "pal");
        let mut f = F::new();
        f.players.update(
            &[
                player(0, "me", "", true, 0, Some(0)),
                player(2, "pal", "", false, 0, Some(0)),
            ],
            &rel,
        );
        f.put(2, 5.0 * 32.0 + 16.0 + 60.0, 5.0 * 32.0 + 16.0, true);
        let me = *f.tees.get(0).unwrap();
        assert!(!helper_near(&me, &f.tees, &f.players), "frozen");
        f.put(2, 5.0 * 32.0 + 16.0 + 200.0, 5.0 * 32.0 + 16.0, false);
        let me = *f.tees.get(0).unwrap();
        assert!(!helper_near(&me, &f.tees, &f.players), "200 px > 140 px");
        f.put(2, 5.0 * 32.0 + 16.0 + 140.0, 5.0 * 32.0 + 16.0, false);
        assert!(helper_near(&me, &f.tees, &f.players), "exactly 140 px counts");
    }

    #[test]
    fn the_dead_zone_kills_after_75_ticks_frozen() {
        let mut f = F::new();
        f.put(0, 2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0, true);
        let mut killed = None;
        for tick in (0..200).step_by(2) {
            if let Verdict::Kill(r) = f.step_with(tick, 1, true, true, false) {
                killed = Some((tick, r));
                break;
            }
        }
        assert_eq!(killed, Some((76, KillReason::Overdue)), "first step >= 75 frozen ticks");
    }

    #[test]
    fn the_kill_cooldown_is_500_ticks_for_every_path() {
        let mut f = F::new();
        f.set_frozen(true);
        assert_eq!(run_frozen(&mut f, 0, 400, 1), Some(200));
        // Still frozen on the tile: frozen_since was reset at the kill, so it re-arms from 202.
        // The cooldown (500) blocks the next kill until tick 700 even though overdue at 402.
        let mut kills = Vec::new();
        for tick in (202..=1300).step_by(2) {
            if let Verdict::Kill(_) = f.step(tick, 1) {
                kills.push(tick);
            }
        }
        assert_eq!(kills.first().copied(), Some(700), "200 + KILL_COOLDOWN_TICKS");
        for w in kills.windows(2) {
            assert!(w[1] - w[0] >= KILL_COOLDOWN_TICKS, "{kills:?}");
        }
    }

    #[test]
    fn wedged_free_with_a_target_kills_after_200_ticks_but_not_without_one() {
        let mut f = F::new();
        f.put(0, 2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0, false);
        assert_eq!(run_frozen(&mut f, 0, 400, -1), None, "no target: never wedged");
        let mut f = F::new();
        f.put(0, 2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0, false);
        assert_eq!(
            run_frozen(&mut f, 0, 400, 1),
            Some(200),
            "anchor at tick 0 -> 200 ticks later"
        );
    }

    #[test]
    fn moving_more_than_48_px_resets_the_anchor() {
        let mut f = F::new();
        f.put(0, 2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0, false);
        for tick in (0..190).step_by(2) {
            assert_eq!(f.step(tick, 1), Verdict::None);
        }
        // A 60 px step at tick 190 moves the anchor; 190+200 is the new deadline.
        f.put(0, 2.0 * 32.0 + 16.0 + 60.0, 2.0 * 32.0 + 16.0, false);
        for tick in (190..388).step_by(2) {
            assert_eq!(f.step(tick, 1), Verdict::None, "tick {tick}");
        }
        assert_eq!(run_frozen(&mut f, 388, 392, 1), Some(390));
    }

    #[test]
    fn a_frozen_target_within_hook_reach_keeps_a_free_bot_from_being_killed() {
        let mut f = F::new();
        f.put(0, 2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0, false);
        f.put(1, 2.0 * 32.0 + 16.0 + 200.0, 2.0 * 32.0 + 16.0, true);
        assert_eq!(
            run_frozen(&mut f, 0, 600, 1),
            None,
            "holding a frozen target it could hook"
        );
        let mut f = F::new();
        f.put(0, 2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0, false);
        f.put(1, 2.0 * 32.0 + 16.0 + 500.0, 2.0 * 32.0 + 16.0, true);
        assert_eq!(
            run_frozen(&mut f, 0, 600, 1),
            Some(200),
            "frozen but out of reach: wedged"
        );
    }

    #[test]
    fn hold_mode_never_kills_and_drops_the_anchor() {
        let mut f = F::new();
        f.set_frozen(true);
        for tick in (0..1000).step_by(2) {
            assert_eq!(f.step_with(tick, 1, false, false, false), Verdict::None);
        }
        assert_eq!(f.u.kills(), 0);
    }

    #[test]
    fn the_wayblock_hook_can_request_a_kill_with_its_own_short_cooldown() {
        let mut f = F::new();
        assert_eq!(
            f.step_with(10, 1, true, false, true),
            Verdict::Kill(KillReason::WayBlockLying)
        );
        assert_eq!(f.step_with(50, 1, true, false, true), Verdict::None, "within 100 ticks");
        assert_eq!(
            f.step_with(110, 1, true, false, true),
            Verdict::Kill(KillReason::WayBlockLying)
        );
    }

    #[test]
    fn an_external_kill_starts_the_cooldown() {
        let mut f = F::new();
        f.u.note_external_kill(100);
        assert!(!f.u.cooldown_ready(599));
        assert!(f.u.cooldown_ready(600));
        assert_eq!(f.u.kills(), 1);
    }

    /// D-102: with the duel switch on, no kill verdict (wayblock lying, overdue frozen, stuck) and no cooldown is started.
    #[test]
    fn the_duel_switch_returns_no_kill_and_starts_no_cooldown() {
        let mut f = F::new();
        f.u.set_no_kill(true);
        assert_eq!(f.step_with(10, 1, true, false, true), Verdict::None, "wayblock lying");
        f.set_frozen(true);
        for tick in (0..2000).step_by(2) {
            assert_eq!(f.step(tick, 1), Verdict::None, "frozen at {tick}");
        }
        assert_eq!(f.u.kills(), 0);
        assert!(f.u.cooldown_ready(2000), "nothing was fired, so no cooldown runs");
    }

    /// D-102 review F1: the anchor keeps running while the switch is on, so lifting it does not fire a `Stuck` kill on an anchor
    /// from before (the bot was back at the old spot only 10 ticks earlier). It kills exactly when it would have without the switch.
    #[test]
    fn lifting_the_duel_switch_does_not_fire_on_a_stale_anchor() {
        let run = |toggle: bool| -> Option<i32> {
            let mut f = F::new();
            let a = (100.0, 100.0);
            f.put(0, a.0, a.1, false);
            for t in (0..=10).step_by(2) {
                assert!(matches!(f.step(t, 1), Verdict::None));
            }
            if toggle {
                f.u.set_no_kill(true);
            }
            for (i, t) in (12..=1000).step_by(2).enumerate() {
                f.put(0, if i % 2 == 0 { 250.0 } else { 320.0 }, 100.0, false);
                assert!(matches!(f.step(t, 1), Verdict::None), "moving: no kill at {t}");
            }
            f.put(0, a.0, a.1, false);
            for t in (1002..=1010).step_by(2) {
                let _ = f.step(t, 1);
            }
            if toggle {
                f.u.set_no_kill(false);
            }
            for t in (1012..=1300).step_by(2) {
                if let Verdict::Kill(_) = f.step(t, 1) {
                    return Some(t);
                }
            }
            None
        };
        let without = run(false);
        assert!(without.is_some_and(|t| t >= 1202), "control: {without:?}");
        assert_eq!(run(true), without);
    }

    // ---- task 4.12 (D-108): the smart policy -------------------------------------------------------------------------

    fn held() -> Option<Forecast> {
        Some(Forecast {
            free_in: None,
            died: false,
            steps: 1,
        })
    }

    fn thaws_in(t: i32) -> Option<Forecast> {
        Some(Forecast {
            free_in: Some(t),
            died: false,
            steps: 1,
        })
    }

    /// First kill tick of a frozen tee sampled every 2 ticks from 0 under the smart policy.
    fn smart_first_kill(f: &mut F, to: i32, forecast: Option<Forecast>, dead: bool) -> Option<(i32, Option<SmartWhy>)> {
        for tick in (0..=to).step_by(2) {
            if let Verdict::Kill(_) = f.step_smart(tick, 1, forecast, dead, false) {
                return Some((tick, f.u.last_why()));
            }
        }
        None
    }

    #[test]
    fn smart_kills_a_tee_resting_in_the_freeze_at_50_ticks_where_legacy_waits_for_200() {
        let mut f = F::new();
        f.u.set_policy(SelfKillPolicy::Smart);
        f.set_frozen(true);
        assert_eq!(
            smart_first_kill(&mut f, 400, held(), false),
            Some((50, Some(SmartWhy::NoExit)))
        );
        // Legacy, same tee: 200 (frozen_in_the_tiles_kills_after_200_ticks_not_before).
    }

    #[test]
    fn smart_does_not_kill_a_tee_whose_freeze_runs_out_soon_even_past_the_legacy_timer() {
        let mut f = F::new();
        f.u.set_policy(SelfKillPolicy::Smart);
        f.set_frozen(true);
        // On the freeze tile the legacy timer is due at 200; the forecast says it thaws in 60 ticks (forever, in this synthetic run):
        // the smart policy holds back until the 400-tick upper bound and says why, once per 100 ticks.
        let mut skips = Vec::new();
        let mut killed = None;
        for tick in (0..=500).step_by(2) {
            if let Verdict::Kill(_) = f.step_smart(tick, 1, thaws_in(60), false, false) {
                killed = Some(tick);
                break;
            }
            if let Some(s) = f.u.take_skip() {
                skips.push(s);
            }
        }
        assert_eq!(killed, Some(400), "FROZEN_HARD_LIMIT_TICKS is the upper bound");
        assert_eq!(
            skips,
            vec![(200, SmartWhy::ThawSoon), (300, SmartWhy::ThawSoon)],
            "one line per 100 ticks, from the legacy 200"
        );
        assert_eq!(f.u.last_why(), Some(SmartWhy::UpperBound));
    }

    #[test]
    fn smart_without_a_forecast_falls_back_to_the_legacy_timers() {
        let mut f = F::new();
        f.u.set_policy(SelfKillPolicy::Smart);
        f.set_frozen(true);
        assert_eq!(
            smart_first_kill(&mut f, 400, None, false),
            Some((200, Some(SmartWhy::UpperBound)))
        );
    }

    #[test]
    fn smart_kills_in_the_dead_zone_at_the_floor_even_if_the_tee_would_thaw() {
        let mut f = F::new();
        f.u.set_policy(SelfKillPolicy::Smart);
        f.put(0, 2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0, true);
        assert_eq!(
            smart_first_kill(&mut f, 400, thaws_in(20), true),
            Some((50, Some(SmartWhy::DeadZone)))
        );
    }

    #[test]
    fn smart_waits_for_a_friend_within_hook_reach_and_for_a_hooking_one() {
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "pal");
        for hook in [false, true] {
            let mut f = F::new();
            f.u.set_policy(SelfKillPolicy::Smart);
            f.players.update(
                &[
                    player(0, "me", "", true, 0, Some(0)),
                    player(1, "other", "", false, 0, Some(0)),
                    player(2, "pal", "", false, 0, Some(0)),
                ],
                &rel,
            );
            f.set_frozen(true);
            f.put(2, 5.0 * 32.0 + 16.0 + 300.0, 5.0 * 32.0 + 16.0, false); // beyond the legacy 140 px, within the hook's reach
            if hook {
                let mut p = *f.tees.get(2).unwrap();
                p.hooked_player = 0;
                f.tees.set_for_test(p);
            }
            if hook {
                assert_eq!(smart_first_kill(&mut f, 1498, held(), false), None, "a hooking friend");
                assert!(
                    matches!(
                        smart_first_kill(&mut f, 1502, held(), false),
                        Some((1500, Some(SmartWhy::UpperBound)))
                    ),
                    "HELPED_LIMIT_TICKS"
                );
            } else {
                // Review 4.12 F3: a friend merely in reach is a grace of FRIEND_GRACE_TICKS, then the hopeless freeze is killed.
                assert_eq!(
                    smart_first_kill(&mut f, 1502, held(), false),
                    Some((200, Some(SmartWhy::NoExit))),
                    "a friend who does not hook"
                );
            }
        }
    }

    /// Review 4.12 F4: the forecast is asked for only when a kill could go out (round 2: exactly, the wayblock request has its own 100).
    #[test]
    fn the_forecast_is_asked_for_only_when_a_kill_could_be_sent() {
        let mut f = F::new();
        f.set_frozen(true);
        f.u.set_policy(SelfKillPolicy::Smart);
        let own = *f.tees.get(0).unwrap();
        // Frozen 40 ticks, no forecast given: nothing fires (the floor is 50).
        for t in (0..=40).step_by(2) {
            assert_eq!(f.step_smart(t, 1, None, false, false), Verdict::None);
        }
        assert!(
            f.u.wants_forecast(40, &own, false),
            "frozen 40 ticks, no switch, no cooldown"
        );
        f.u.set_no_kill(true);
        assert!(!f.u.wants_forecast(40, &own, false), "the duel switch is on");
        assert!(
            !f.u.wants_forecast(40, &own, true),
            "the duel switch is on, wayblock request too"
        );
        f.u.set_no_kill(false);
        // A kill at 50 starts the cooldown; the clock runs again from the next step.
        assert!(matches!(f.step_smart(50, 1, held(), false, false), Verdict::Kill(_)));
        for t in (52..=148).step_by(2) {
            f.step_smart(t, 1, None, false, false);
            assert!(
                !f.u.wants_forecast(t, &own, false),
                "tick {t}: the cooldown (500) still runs"
            );
            assert!(
                !f.u.wants_forecast(t, &own, true),
                "tick {t}: the wayblock's 100 still runs"
            );
        }
        f.step_smart(150, 1, None, false, false);
        assert!(!f.u.wants_forecast(150, &own, false), "500 not over");
        assert!(
            f.u.wants_forecast(150, &own, true),
            "the wayblock request's 100 is over"
        );
        assert!(!f.u.wants_forecast(549, &own, false), "499 ticks since the kill");
        assert!(f.u.wants_forecast(550, &own, false), "500 over");
        // The legacy policy never asks.
        let mut g = F::new();
        g.set_frozen(true);
        for t in (0..=60).step_by(2) {
            g.step(t, 1);
        }
        let own = *g.tees.get(0).unwrap();
        assert!(!g.u.wants_forecast(60, &own, false), "legacy policy");
    }

    #[test]
    fn an_afk_friend_is_no_rescuer() {
        use ddai_net::generated::enums::explayerflagflag::AFK;
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "pal");
        let mut f = F::new();
        f.u.set_policy(SelfKillPolicy::Smart);
        f.players.update(
            &[
                player(0, "me", "", true, 0, Some(0)),
                player(1, "other", "", false, 0, Some(0)),
                player(2, "pal", "", false, 0, Some(AFK)),
            ],
            &rel,
        );
        f.set_frozen(true);
        f.put(2, 5.0 * 32.0 + 16.0 + 300.0, 5.0 * 32.0 + 16.0, false);
        assert_eq!(smart_first_kill(&mut f, 400, held(), false).map(|k| k.0), Some(50));
    }

    #[test]
    fn smart_lets_a_wedged_free_tee_hold_a_block_but_kills_it_otherwise() {
        for holding in [true, false] {
            let mut f = F::new();
            f.u.set_policy(SelfKillPolicy::Smart);
            f.put(0, 2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0, false);
            let mut kill = None;
            for tick in (0..=600).step_by(2) {
                if let Verdict::Kill(KillReason::Stuck) = f.step_smart(tick, 1, None, false, holding) {
                    kill = Some(tick);
                    break;
                }
            }
            assert_eq!(kill, if holding { None } else { Some(200) }, "holding {holding}");
            if holding {
                assert_eq!(f.u.take_skip().map(|s| s.1), Some(SmartWhy::HoldingBlock));
            }
        }
    }

    #[test]
    fn the_duel_switch_beats_the_smart_policy_too() {
        let mut f = F::new();
        f.u.set_policy(SelfKillPolicy::Smart);
        f.u.set_no_kill(true);
        f.set_frozen(true);
        for tick in (0..2000).step_by(2) {
            assert_eq!(f.step_smart(tick, 1, held(), true, false), Verdict::None, "tick {tick}");
        }
        assert_eq!(f.u.kills(), 0);
        assert!(f.u.cooldown_ready(2000));
    }

    #[test]
    fn the_smart_policy_keeps_the_500_tick_cooldown() {
        let mut f = F::new();
        f.u.set_policy(SelfKillPolicy::Smart);
        f.set_frozen(true);
        let mut kills = Vec::new();
        for tick in (0..=1600).step_by(2) {
            if let Verdict::Kill(_) = f.step_smart(tick, 1, held(), false, false) {
                kills.push(tick);
            }
        }
        assert!(kills.len() >= 2 && kills[0] == 50, "{kills:?}");
        for w in kills.windows(2) {
            assert!(w[1] - w[0] >= KILL_COOLDOWN_TICKS, "{kills:?}");
        }
    }

    #[test]
    fn the_legacy_policy_ignores_the_smart_inputs() {
        let mut f = F::new();
        f.set_frozen(true);
        // Same tee as the smart test above, legacy: a held forecast does not make it earlier, a thaw soon does not make it later.
        for fc in [held(), thaws_in(5), None] {
            let mut g = F::new();
            g.set_frozen(true);
            let first = (0..=400)
                .step_by(2)
                .find(|&t| matches!(g.step_smart(t, 1, fc, false, true), Verdict::Kill(_)));
            assert_eq!(first, Some(200));
        }
        assert_eq!(f.u.policy(), SelfKillPolicy::Legacy);
    }
}
