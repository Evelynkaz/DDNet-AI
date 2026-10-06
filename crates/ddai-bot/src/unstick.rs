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
use crate::tees::{Tee, TeeSet, dist};

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
        let overdue = (frozen_for >= FROZEN_HARD_LIMIT_TICKS
            || (in_tiles && !hooked && frozen_for >= FROZEN_IN_TILE_TICKS)
            || (trapped && frozen_for >= TRAPPED_TICKS))
            && (!helped || frozen_for >= HELPED_LIMIT_TICKS);

        if c.wayblock_wants_kill && !self.no_kill && tick - self.last_kill_tick >= WB_KILL_COOLDOWN_TICKS {
            self.fire(tick, true);
            return Verdict::Kill(KillReason::WayBlockLying);
        }
        if overdue && !self.no_kill && self.cooldown_ready(tick) {
            self.fire(tick, true);
            return Verdict::Kill(KillReason::Overdue);
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
            return Verdict::None;
        }
        let anchor = self.anchor.expect("anchor is set: anchor_moved was false");
        let stuck_ticks = tick - anchor.tick;
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
        self.fire(tick, false);
        Verdict::Kill(KillReason::Stuck)
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
            })
        }
        fn step(&mut self, tick: i32, target: i32) -> Verdict {
            self.step_with(tick, target, true, false, false)
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
}
