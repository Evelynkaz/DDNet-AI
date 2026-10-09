//! The activity clock and block attribution — `refreshInputClock` (`bot.ts:2916-2991`),
//! `inputIdle`/`afk` (`2993-3018`), `onFreezeOnset` (`3773-3788`), `docs/research/orig-bot.md` §7.4.
//!
//! **Activity / AFK.** A player is "active" when its aim angle, attack tick or keys
//! ([`input_keys_of`]) change (a frozen tee's keys do not count). `changed` is the last tick such a
//! change was seen, except inside the `INPUT_SETTLE_TICKS` window after a death (a respawn is not a
//! player waking up). `input_idle`: no record -> `strict`; never changed -> idle once seen for more
//! than `AFK_TICKS` (or at once when `strict`); else idle when `tick - changed > AFK_TICKS`.
//!
//! **Attribution** (stats and telemetry only — the score does not use it). `last_touch[victim] =
//! {by, tick}` is set when someone hooks the victim (keeping the previous toucher while they still
//! hook it, otherwise the lowest id) and when a hammer swing's point `pos + dir(angle) * 21` lands
//! within 56 px of the victim; it is forgotten on a teleport (> 200 px within 4 ticks) and a death.
//! A *freeze onset* (not a re-freeze within 6 ticks of thawing) within `BLOCK_CREDIT_TICKS` (50) of a
//! touch is a block (ours) or a "blocked by" (theirs), unless the toucher is ourselves/a friend.
//!
//! Deterministic order: tees are visited in ascending client id (the TS used map-insertion order).
//! All storage is fixed-size: no allocation per snapshot.

use ddai_physics::vmath::Vec2;

use crate::consts::*;
use crate::players::{MAX_CLIENTS, PlayerTable};
use crate::tees::{HELD_MOVE_MIN, Tee, TeeSet, dist, input_keys_of, keys_neutral};

const NEVER: i32 = i32::MIN / 2;

#[derive(Debug, Clone, Copy)]
struct Seen {
    valid: bool,
    generation: u32,
    angle: i32,
    attack: i32,
    keys: i32,
    /// Where the tee was at the last snapshot (held keys while it moves are activity too).
    pos: Vec2<f32>,
    at: i32,
    first_seen: i32,
    changed: i32,
    settle_until: i32,
}

impl Seen {
    const NONE: Seen = Seen {
        valid: false,
        generation: 0,
        angle: 0,
        attack: 0,
        keys: 0,
        pos: Vec2 { x: 0.0, y: 0.0 },
        at: 0,
        first_seen: 0,
        changed: -1,
        settle_until: 0,
    };
}

#[derive(Debug, Clone, Copy)]
struct Touch {
    by: i32,
    tick: i32,
}

#[derive(Debug, Clone, Copy)]
struct LastPos {
    pos: Vec2<f32>,
    tick: i32,
    valid: bool,
}

/// What the attribution noticed this snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockEvent {
    /// We froze `victim` within the credit window of our touch.
    Block { victim: i32 },
    /// `by` froze us.
    BlockedBy { by: i32 },
    /// Task 3.10: a block of ours was still on `HELD_BLOCK_TICKS` later (`died`: the victim died inside the window, which is out of
    /// the fight as well).
    Held { victim: i32, died: bool },
    /// Task 3.10: the victim of a block of ours was free again `after` ticks after the block.
    Escaped { victim: i32, after: i32 },
}

/// The window a block must stay on to count as held (the arena's held-block metric: 250 ticks, 5 s).
pub const HELD_BLOCK_TICKS: i32 = 250;
/// Blocks watched at once (more are not watched; the oldest watch is dropped first).
const HELD_WATCH: usize = 8;

/// Running block counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockStats {
    pub blocks: u32,
    pub blocked_by: u32,
    /// Task 3.10: of our blocks, those that were still on 5 s later (or whose victim died), and those whose victim got free before.
    pub held: u32,
    pub escaped: u32,
    /// Of `held`: the victim died inside the window (a kill tile or its own `/kill`), not frozen the whole time.
    pub died: u32,
}

pub struct ActivityClock {
    seen: Box<[Seen; MAX_CLIENTS]>,
    at_us: Box<[i32; MAX_CLIENTS]>,
    at_friend: Box<[i32; MAX_CLIENTS]>,
    frozen_since: Box<[i32; MAX_CLIENTS]>,
    thaw_tick: Box<[i32; MAX_CLIENTS]>,
    last_touch: Box<[Option<Touch>; MAX_CLIENTS]>,
    last_pos: Box<[LastPos; MAX_CLIENTS]>,
    was_alive: Box<[bool; MAX_CLIENTS]>,
    /// Ids that were out of the game (spectating, paused, team -1) at the last snapshot.
    out_of_game: Box<[bool; MAX_CLIENTS]>,
    stats: BlockStats,
    events: Vec<BlockEvent>,
    /// `(victim, block tick)` of the blocks being followed (`victim < 0`: free slot).
    held_watch: [(i32, i32); HELD_WATCH],
}

impl Default for ActivityClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Room for the events of one snapshot; more are counted but not queued.
const EVENT_CAP: usize = 32;

impl ActivityClock {
    pub fn new() -> Self {
        ActivityClock {
            seen: Box::new([Seen::NONE; MAX_CLIENTS]),
            at_us: Box::new([NEVER; MAX_CLIENTS]),
            at_friend: Box::new([NEVER; MAX_CLIENTS]),
            frozen_since: Box::new([-1; MAX_CLIENTS]),
            thaw_tick: Box::new([NEVER; MAX_CLIENTS]),
            last_touch: Box::new([None; MAX_CLIENTS]),
            last_pos: Box::new(
                [LastPos {
                    pos: Vec2::new(0.0, 0.0),
                    tick: 0,
                    valid: false,
                }; MAX_CLIENTS],
            ),
            was_alive: Box::new([false; MAX_CLIENTS]),
            out_of_game: Box::new([false; MAX_CLIENTS]),
            stats: BlockStats::default(),
            events: Vec::with_capacity(EVENT_CAP),
            held_watch: [(-1, 0); HELD_WATCH],
        }
    }

    /// Forgets everything tick-dependent (`onTickReset`, `bot.ts:1258-1293`: the game tick went
    /// backwards — a map restart). Counters stay.
    pub fn reset(&mut self) {
        let stats = self.stats;
        *self = ActivityClock::new();
        self.stats = stats;
    }

    pub fn stats(&self) -> BlockStats {
        self.stats
    }

    /// Task 4.12: a block of ours is being followed (made within [`HELD_BLOCK_TICKS`], not yet held for good, escaped or died): the smart
    /// self-kill policy does not kill a tee that stands over a block it is holding.
    pub fn holding_block(&self) -> bool {
        self.held_watch.iter().any(|w| w.0 >= 0)
    }

    /// The block events since the last call (oldest first).
    pub fn drain_events(&mut self) -> std::vec::Drain<'_, BlockEvent> {
        self.events.drain(..)
    }

    /// `onKill` (`bot.ts:2120-2135`): a kill message for `victim` forgets its touch/thaw and starts
    /// its input-settle window.
    pub fn on_kill(&mut self, victim: i32, tick: i32) {
        let Some(i) = Self::idx(victim) else { return };
        // Task 3.10: a victim that dies inside the window of a block of ours is out of the fight (the block is held, `died`), even when it is back
        // on its feet by the next snapshot (a kill tile or its own `/kill` respawns it at once).
        for w in self.held_watch.iter_mut().filter(|w| w.0 == victim) {
            *w = (-1, 0);
            self.stats.held += 1;
            self.stats.died += 1;
            if self.events.len() < EVENT_CAP {
                self.events.push(BlockEvent::Held { victim, died: true });
            }
        }
        self.last_touch[i] = None;
        self.thaw_tick[i] = NEVER;
        if self.seen[i].valid {
            self.seen[i].settle_until = tick + INPUT_SETTLE_TICKS;
        }
    }

    fn idx(id: i32) -> Option<usize> {
        usize::try_from(id).ok().filter(|&i| i < MAX_CLIENTS)
    }

    /// Last tick `id` swung at us or hooked us (`atUsById`), for the target score's memory window.
    pub fn at_us_within(&self, id: i32, tick: i32, window: i32) -> bool {
        Self::idx(id).is_some_and(|i| tick - self.at_us[i] < window)
    }

    /// Same for friends (`atFriendById`).
    pub fn at_friend_within(&self, id: i32, tick: i32, window: i32) -> bool {
        Self::idx(id).is_some_and(|i| tick - self.at_friend[i] < window)
    }

    /// Ticks `id` has been frozen (0 when free) — `frozenFor` (`bot.ts:3073`).
    pub fn frozen_for(&self, tee: &Tee, tick: i32) -> i32 {
        if !tee.frozen {
            return 0;
        }
        Self::idx(tee.id).map_or(0, |i| {
            let since = self.frozen_since[i];
            if since < 0 { 0 } else { tick - since }
        })
    }

    /// `inputIdle` (`bot.ts:2993-2998`).
    pub fn input_idle(&self, id: i32, tick: i32, strict: bool) -> bool {
        let Some(i) = Self::idx(id) else { return strict };
        let s = &self.seen[i];
        if !s.valid {
            return strict;
        }
        if s.changed < 0 {
            return strict || tick - s.first_seen > AFK_TICKS;
        }
        tick - s.changed > AFK_TICKS
    }

    /// `awayInGame(t)`: AFK while in the game (a spectator or a paused player whose tee is still on the map
    /// is not "away": it is fought, outside the AFK room; `afk` still counts it).
    pub fn away_in_game(&self, id: i32, tick: i32, players: &PlayerTable) -> bool {
        let slot = players.get(id);
        !slot.is_some_and(|s| s.not_playing())
            && (slot.is_some_and(|s| s.server_afk()) || self.input_idle(id, tick, false))
    }

    /// `afk` (`bot.ts:3000-3003`): server AFK flag, not playing, or no input for 10 s.
    pub fn afk(&self, id: i32, tick: i32, players: &PlayerTable, strict: bool) -> bool {
        let slot = players.get(id);
        slot.is_some_and(|s| s.server_afk() || s.not_playing()) || self.input_idle(id, tick, strict)
    }

    /// One snapshot's worth of bookkeeping. `own_id` is our client id.
    pub fn update(&mut self, tick: i32, tees: &TeeSet, players: &PlayerTable, own_id: i32) {
        let roster_known = players.present().next().is_some();
        let me = tees.get(own_id).copied();
        if roster_known {
            for i in 0..MAX_CLIENTS {
                match players.get(i as i32).filter(|s| s.present) {
                    None => {
                        self.seen[i].valid = false;
                        self.at_us[i] = NEVER;
                        self.at_friend[i] = NEVER;
                        self.out_of_game[i] = false;
                    }
                    Some(slot) => {
                        // A player back from the spectators (or a pause) starts over: its old inputs say
                        // nothing about whether it is away now (`notPlayingIds`).
                        let out = slot.not_playing() || slot.team == -1;
                        if !out && self.out_of_game[i] {
                            self.seen[i].valid = false;
                        }
                        self.out_of_game[i] = out;
                    }
                }
            }
        }

        // Deaths: a tee alive last snapshot and gone now (`!tee.alive` branch, `bot.ts:2961-2964`).
        for i in 0..MAX_CLIENTS {
            if self.was_alive[i] && tees.get(i as i32).is_none() {
                self.thaw_tick[i] = NEVER;
                self.last_touch[i] = None;
                self.frozen_since[i] = -1;
                if self.seen[i].valid {
                    self.seen[i].settle_until = tick + INPUT_SETTLE_TICKS;
                }
            }
        }

        // Teleports forget the touch (`bot.ts:2934-2937`).
        for a in tees.iter() {
            let i = a.id as usize;
            let lp = self.last_pos[i];
            if lp.valid && tick - lp.tick <= 4 && dist(a.pos, lp.pos) > TELEPORT_JUMP_PX {
                self.last_touch[i] = None;
            }
            self.last_pos[i] = LastPos {
                pos: a.pos,
                tick,
                valid: true,
            };
        }

        // Hooks: every hooker of a victim touches it (`bot.ts:2938-2948`).
        let mut first_hooker = [-1i32; MAX_CLIENTS];
        for a in tees.iter() {
            if a.hooked_player >= 0 && (a.hooked_player as usize) < MAX_CLIENTS {
                let v = a.hooked_player as usize;
                if first_hooker[v] < 0 {
                    first_hooker[v] = a.id; // ascending ids: the first is the minimum
                }
            }
        }
        for (v, &min_id) in first_hooker.iter().enumerate() {
            if min_id < 0 {
                continue;
            }
            let keeps_previous = self.last_touch[v]
                .and_then(|t| tees.get(t.by))
                .is_some_and(|prev| prev.hooked_player == v as i32);
            let by = if keeps_previous {
                self.last_touch[v].map_or(min_id, |t| t.by)
            } else {
                min_id
            };
            self.last_touch[v] = Some(Touch { by, tick });
        }

        // Hammer swings (`bot.ts:2949-2955`).
        for a in tees.iter() {
            let i = a.id as usize;
            let s = self.seen[i];
            if !s.valid || s.at == tick || a.attack_tick == s.attack || !a.holding_hammer() {
                continue;
            }
            let r = a.aim_rad();
            let hp = Vec2::new(
                a.pos.x + ddai_libm::cosf(r) * HAMMER_REACH_AHEAD_PX,
                a.pos.y + ddai_libm::sinf(r) * HAMMER_REACH_AHEAD_PX,
            );
            for b in tees.iter() {
                if b.id != a.id && dist(hp, b.pos) < HAMMER_REACH_PX {
                    self.last_touch[b.id as usize] = Some(Touch { by: a.id, tick });
                }
            }
        }

        // Per tee: freeze tracking, then the input clock.
        for tee in tees.iter() {
            let i = tee.id as usize;
            if !tee.frozen && self.frozen_since[i] >= 0 {
                self.thaw_tick[i] = tick;
            }
            if !tee.frozen {
                self.frozen_since[i] = -1;
            } else if self.frozen_since[i] < 0 {
                self.frozen_since[i] = tick;
                if tick - self.thaw_tick[i] > REFREEZE_TICKS {
                    self.on_freeze_onset(tee, me.as_ref(), tick, players);
                }
            }

            let generation = players.get(tee.id).map_or(0, |s| s.generation);
            let keys = input_keys_of(tee);
            let seen = &mut self.seen[i];
            if !seen.valid || seen.generation != generation {
                *seen = Seen {
                    valid: true,
                    generation,
                    angle: tee.angle,
                    attack: tee.attack_tick,
                    keys,
                    pos: tee.pos,
                    at: tick,
                    first_seen: tick,
                    changed: -1,
                    settle_until: 0,
                };
                continue;
            }
            if seen.at == tick {
                continue;
            }
            // Held keys count as activity while the tee moves (running, swinging on the hook), even when
            // the keys themselves did not change.
            let changed = tee.angle != seen.angle
                || tee.attack_tick != seen.attack
                || (keys >= 0 && seen.keys >= 0 && keys != seen.keys)
                || (keys >= 0 && !keys_neutral(keys) && dist(tee.pos, seen.pos) >= HELD_MOVE_MIN);
            if changed && tick >= seen.settle_until {
                seen.changed = tick;
            }
            let swung = tee.attack_tick != seen.attack;
            if let Some(me) = me.as_ref()
                && tee.id != me.id
                && ((swung && dist(tee.pos, me.pos) < SWING_AT_US_PX) || tee.hooked_player == me.id)
            {
                self.at_us[i] = tick;
            }
            if tee.id != own_id {
                let at_friend = tees.iter().any(|f| {
                    f.id != tee.id
                        && players.get(f.id).is_some_and(|s| s.flags.friendly())
                        && f.id != own_id
                        && ((swung && !f.frozen && dist(tee.pos, f.pos) < SWING_AT_US_PX) || tee.hooked_player == f.id)
                });
                if at_friend {
                    self.at_friend[i] = tick;
                }
            }
            let seen = &mut self.seen[i];
            seen.angle = tee.angle;
            seen.attack = tee.attack_tick;
            if keys >= 0 {
                seen.keys = keys;
            }
            seen.pos = tee.pos;
            seen.at = tick;
        }

        self.follow_blocks(tick, tees, players);

        for i in 0..MAX_CLIENTS {
            self.was_alive[i] = tees.get(i as i32).is_some();
        }
    }

    /// `onFreezeOnset` (`bot.ts:3773-3788`).
    fn on_freeze_onset(&mut self, tee: &Tee, me: Option<&Tee>, tick: i32, players: &PlayerTable) {
        let Some(me) = me else { return };
        let Some(last) = Self::idx(tee.id).and_then(|i| self.last_touch[i]) else {
            return;
        };
        if tick - last.tick > BLOCK_CREDIT_TICKS {
            return;
        }
        let friendly = |id: i32| players.get(id).is_some_and(|s| s.flags.friendly());
        if tee.id == me.id {
            if last.by == me.id || friendly(last.by) {
                return;
            }
            self.stats.blocked_by += 1;
            if self.events.len() < EVENT_CAP {
                self.events.push(BlockEvent::BlockedBy { by: last.by });
            }
            return;
        }
        if last.by != me.id || friendly(tee.id) {
            return;
        }
        self.stats.blocks += 1;
        if self.events.len() < EVENT_CAP {
            self.events.push(BlockEvent::Block { victim: tee.id });
        }
        // Follow it: the slot of this victim's older watch, else a free one, else the oldest.
        let slot = self
            .held_watch
            .iter()
            .position(|w| w.0 == tee.id)
            .or_else(|| self.held_watch.iter().position(|w| w.0 < 0))
            .unwrap_or_else(|| {
                (0..HELD_WATCH)
                    .min_by_key(|&k| self.held_watch[k].1)
                    .expect("HELD_WATCH > 0")
            });
        self.held_watch[slot] = (tee.id, tick);
    }

    /// Task 3.10: the verdicts of the blocks being followed. Free (not frozen) before the window ends: escaped; still frozen when it ends: held; died: only
    /// by a kill message (`on_kill`); gone from the snapshot without one: no verdict.
    fn follow_blocks(&mut self, tick: i32, tees: &TeeSet, players: &PlayerTable) {
        for k in 0..HELD_WATCH {
            let (victim, since) = self.held_watch[k];
            if victim < 0 {
                continue;
            }
            let verdict = match tees.get(victim) {
                // Gone from the snapshot: dead, out of view or in the spectators -- the snapshot cannot tell which. A death is announced by its kill message
                // (`on_kill`, which ends the watch as held); without one there is no verdict, and the watch just runs out (the clip tool's `OutOfView`).
                None if players.get(victim).is_some_and(|s| s.present) && tick - since < HELD_BLOCK_TICKS => continue,
                None => {
                    self.held_watch[k] = (-1, 0);
                    continue;
                }
                Some(t) if !t.frozen && tick > since => Some(BlockEvent::Escaped {
                    victim,
                    after: tick - since,
                }),
                Some(_) if tick - since >= HELD_BLOCK_TICKS => Some(BlockEvent::Held { victim, died: false }),
                Some(_) => None,
            };
            let Some(v) = verdict else { continue };
            self.held_watch[k] = (-1, 0);
            match v {
                BlockEvent::Held { died, .. } => {
                    self.stats.held += 1;
                    self.stats.died += u32::from(died);
                }
                _ => self.stats.escaped += 1,
            }
            if self.events.len() < EVENT_CAP {
                self.events.push(v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::players::test_support::player;
    use crate::relations::{ListKind, Relations};
    use crate::tees::{HOOK_GRABBED, HOOK_IDLE};

    struct Fixture {
        clock: ActivityClock,
        tees: TeeSet,
        players: PlayerTable,
        rel: Relations,
    }

    impl Fixture {
        fn new(n: i32) -> Self {
            let rel = Relations::new();
            let mut players = PlayerTable::new([0; 16]);
            let list: Vec<_> = (0..n)
                .map(|i| player(i, &format!("p{i}"), "", i == 0, 0, Some(0)))
                .collect();
            players.update(&list, &rel);
            let mut f = Fixture {
                clock: ActivityClock::new(),
                tees: TeeSet::new(),
                players,
                rel,
            };
            for i in 0..n {
                f.set(Tee {
                    id: i,
                    alive: true,
                    pos: Vec2::new(i as f32 * 1000.0, 0.0),
                    ..Tee::DEAD
                });
            }
            f
        }
        fn set(&mut self, t: Tee) {
            self.tees.set_for_test(t);
        }
        fn tee(&self, id: i32) -> Tee {
            *self.tees.get(id).unwrap()
        }
        fn step(&mut self, tick: i32) {
            self.clock.update(tick, &self.tees, &self.players, 0);
        }
        fn kill(&mut self, id: i32) {
            let mut t = self.tee(id);
            t.alive = false;
            self.set(t);
        }
    }

    #[test]
    fn a_tee_that_never_changes_becomes_afk_after_500_ticks_and_one_that_moves_does_not() {
        let mut f = Fixture::new(3);
        for tick in (0..=700).step_by(2) {
            // tee 1 sits still; tee 2 wiggles its aim every 100 ticks.
            let mut t2 = f.tee(2);
            t2.angle = (tick / 100) * 50;
            f.set(t2);
            f.step(tick);
        }
        assert!(f.clock.input_idle(1, 700, false), "no change for 700 ticks");
        assert!(!f.clock.input_idle(2, 700, false), "aim keeps changing");
        assert!(!f.clock.input_idle(1, 400, false), "not yet at 400 ticks");
        assert!(f.clock.afk(1, 700, &f.players, false));
        assert!(!f.clock.afk(2, 700, &f.players, false));
    }

    #[test]
    fn idle_boundaries_are_exactly_500_ticks_and_strict_means_unseen_is_idle() {
        let mut f = Fixture::new(2);
        f.step(0);
        assert!(
            !f.clock.input_idle(1, 500, false),
            "seen 500 ticks ago is not yet > AFK_TICKS"
        );
        assert!(f.clock.input_idle(1, 501, false));
        assert!(f.clock.input_idle(1, 0, true), "strict: never changed -> idle at once");
        assert!(!f.clock.input_idle(99, 0, false), "unknown id: not strict -> not idle");
        assert!(f.clock.input_idle(99, 0, true), "unknown id, strict -> idle");
    }

    #[test]
    fn changes_right_after_a_death_do_not_count_as_activity() {
        let mut f = Fixture::new(2);
        f.step(0);
        f.clock.on_kill(1, 10);
        for (tick, angle) in [(12, 60), (14, 120), (16, 180)] {
            let mut t = f.tee(1);
            t.angle = angle;
            f.set(t);
            f.step(tick);
        }
        assert!(
            f.clock.input_idle(1, 600, false),
            "angle changes within the 50-tick window ignored"
        );
        let mut t = f.tee(1);
        t.angle = 300;
        f.set(t);
        f.step(70);
        assert!(
            !f.clock.input_idle(1, 100, false),
            "after the window a change counts (changed at 70)"
        );
    }

    #[test]
    fn frozen_keys_are_ignored_but_aim_changes_count() {
        let mut f = Fixture::new(2);
        let mut t = f.tee(1);
        t.frozen = true;
        f.set(t);
        f.step(0);
        t.direction = 1; // keys are -1 while frozen: no change
        f.set(t);
        f.step(2);
        assert!(f.clock.input_idle(1, 600, false));
        t.angle = 77;
        f.set(t);
        f.step(4);
        assert!(!f.clock.input_idle(1, 100, false), "the aim changed at tick 4");
    }

    #[test]
    fn swinging_near_us_or_hooking_us_marks_at_us_and_far_swings_do_not() {
        let mut f = Fixture::new(3);
        let mut me = f.tee(0);
        me.pos = Vec2::new(0.0, 0.0);
        f.set(me);
        let mut near = f.tee(1);
        near.pos = Vec2::new(100.0, 0.0);
        f.set(near);
        let mut far = f.tee(2);
        far.pos = Vec2::new(300.0, 0.0);
        f.set(far);
        f.step(0);
        let (mut n, mut fr) = (f.tee(1), f.tee(2));
        n.attack_tick = 5;
        fr.attack_tick = 5;
        f.set(n);
        f.set(fr);
        f.step(2);
        assert!(
            f.clock.at_us_within(1, 2, AGGRESSOR_MEMORY_TICKS),
            "a swing within 128 px"
        );
        assert!(
            !f.clock.at_us_within(2, 2, AGGRESSOR_MEMORY_TICKS),
            "a swing at 300 px is not at us"
        );
        assert!(
            !f.clock.at_us_within(1, 2 + 150, AGGRESSOR_MEMORY_TICKS),
            "forgotten after 150 ticks"
        );
        let mut far2 = f.tee(2);
        far2.hooked_player = 0;
        f.set(far2);
        f.step(4);
        assert!(
            f.clock.at_us_within(2, 4, AGGRESSOR_MEMORY_TICKS),
            "hooking us counts at any range"
        );
    }

    /// Task 3.10: a block is followed for 250 ticks: free again before that is "escaped", still frozen at the end is "held", gone from the
    /// tees without a kill message gets no verdict (a kill message ends the watch as held at once); a victim that leaves the server is dropped.
    #[test]
    fn a_block_is_followed_until_the_victim_is_free_or_the_window_ends() {
        // Escaped: thaws 120 ticks after the block.
        let mut f = Fixture::new(2);
        f.step(0);
        hook_then_freeze(&mut f, 0, 1, 10, 30);
        f.clock.drain_events().for_each(drop);
        let mut v = f.tee(1);
        v.frozen = true;
        f.set(v);
        f.step(100);
        assert!(
            f.clock.drain_events().next().is_none(),
            "still frozen at 70 ticks: no verdict yet"
        );
        v.frozen = false;
        f.set(v);
        f.step(150);
        assert_eq!(
            f.clock.drain_events().collect::<Vec<_>>(),
            vec![BlockEvent::Escaped { victim: 1, after: 120 }]
        );
        assert_eq!((f.clock.stats().held, f.clock.stats().escaped), (0, 1));
        f.step(400);
        assert!(f.clock.drain_events().next().is_none(), "one verdict per block");

        // Held: frozen on the 250th tick.
        let mut g = Fixture::new(2);
        g.step(0);
        hook_then_freeze(&mut g, 0, 1, 10, 30);
        g.clock.drain_events().for_each(drop);
        let mut v = g.tee(1);
        v.frozen = true;
        g.set(v);
        g.step(200);
        assert!(g.clock.drain_events().next().is_none());
        g.step(280);
        assert_eq!(
            g.clock.drain_events().collect::<Vec<_>>(),
            vec![BlockEvent::Held { victim: 1, died: false }]
        );
        assert_eq!((g.clock.stats().held, g.clock.stats().escaped), (1, 0));

        // A kill message (a kill tile, its own /kill) ends the watch at once as held, even if the victim is back on its feet next snapshot.
        let mut k = Fixture::new(2);
        k.step(0);
        hook_then_freeze(&mut k, 0, 1, 10, 30);
        k.clock.drain_events().for_each(drop);
        k.clock.on_kill(1, 60);
        let mut v = k.tee(1);
        v.frozen = false;
        k.set(v);
        k.step(70);
        assert_eq!(
            k.clock.drain_events().collect::<Vec<_>>(),
            vec![BlockEvent::Held { victim: 1, died: true }]
        );
        assert_eq!(
            (k.clock.stats().held, k.clock.stats().died, k.clock.stats().escaped),
            (1, 1, 0)
        );

        // Gone from the snapshot with no kill message (out of view, in the spectators): no verdict, not even at the end of the window.
        let mut d = Fixture::new(2);
        d.step(0);
        hook_then_freeze(&mut d, 0, 1, 10, 30);
        d.clock.drain_events().for_each(drop);
        d.kill(1);
        d.step(60);
        d.step(200);
        d.step(400);
        assert!(
            d.clock.drain_events().next().is_none(),
            "no verdict without a kill message"
        );
        assert_eq!((d.clock.stats().held, d.clock.stats().escaped), (0, 0));
    }

    #[test]
    fn swinging_at_a_friend_marks_at_friend() {
        let mut f = Fixture::new(3);
        f.rel.add(ListKind::Friend, "p2");
        let ps: Vec<_> = (0..3)
            .map(|i| player(i, &format!("p{i}"), "", i == 0, 0, Some(0)))
            .collect();
        f.players.update(&ps, &f.rel);
        let mut attacker = f.tee(1);
        attacker.pos = Vec2::new(500.0, 0.0);
        let mut friend = f.tee(2);
        friend.pos = Vec2::new(550.0, 0.0);
        f.set(attacker);
        f.set(friend);
        f.step(0);
        attacker.attack_tick = 9;
        f.set(attacker);
        f.step(2);
        assert!(f.clock.at_friend_within(1, 2, AGGRESSOR_MEMORY_TICKS));
    }

    /// The hooker hooks at `hook_tick`, lets go, and the victim freezes at `freeze_tick`.
    fn hook_then_freeze(f: &mut Fixture, hooker: i32, victim: i32, hook_tick: i32, freeze_tick: i32) {
        let mut h = f.tee(hooker);
        h.hooked_player = victim;
        h.hook_state = HOOK_GRABBED;
        f.set(h);
        f.step(hook_tick);
        h.hooked_player = -1;
        h.hook_state = HOOK_IDLE;
        f.set(h);
        let mut v = f.tee(victim);
        v.frozen = true;
        f.set(v);
        f.step(freeze_tick);
    }

    #[test]
    fn a_freeze_soon_after_our_hook_is_a_block() {
        let mut f = Fixture::new(2);
        f.step(0);
        hook_then_freeze(&mut f, 0, 1, 10, 30);
        assert_eq!(
            f.clock.stats(),
            BlockStats {
                blocks: 1,
                blocked_by: 0,
                held: 0,
                escaped: 0,
                died: 0
            }
        );
        assert_eq!(
            f.clock.drain_events().collect::<Vec<_>>(),
            vec![BlockEvent::Block { victim: 1 }]
        );
    }

    #[test]
    fn credit_is_at_most_50_ticks_old_and_a_refreeze_within_6_ticks_is_not_new() {
        let mut f = Fixture::new(2);
        f.step(0);
        // Hook at 10, freeze at 61: 51 ticks later -> no credit.
        hook_then_freeze(&mut f, 0, 1, 10, 61);
        assert_eq!(f.clock.stats().blocks, 0, "51 ticks is outside the window");
        // Exactly 50 ticks is inside.
        let mut f = Fixture::new(2);
        f.step(0);
        hook_then_freeze(&mut f, 0, 1, 10, 60);
        assert_eq!(f.clock.stats().blocks, 1);
        // Thaw at 62, refreeze at 66 (< 6 ticks after the thaw): not a new block.
        let mut v = f.tee(1);
        v.frozen = false;
        f.set(v);
        f.step(62);
        v.frozen = true;
        f.set(v);
        f.step(66);
        assert_eq!(f.clock.stats().blocks, 1, "a refreeze within REFREEZE_TICKS");
    }

    #[test]
    fn our_own_freeze_after_their_hook_is_blocked_by_unless_a_friend_did_it() {
        let mut f = Fixture::new(3);
        f.step(0);
        hook_then_freeze(&mut f, 1, 0, 10, 20);
        assert_eq!(
            f.clock.stats(),
            BlockStats {
                blocks: 0,
                blocked_by: 1,
                held: 0,
                escaped: 0,
                died: 0
            }
        );
        assert_eq!(
            f.clock.drain_events().collect::<Vec<_>>(),
            vec![BlockEvent::BlockedBy { by: 1 }]
        );

        let mut g = Fixture::new(3);
        g.rel.add(ListKind::Friend, "p2");
        let ps: Vec<_> = (0..3)
            .map(|i| player(i, &format!("p{i}"), "", i == 0, 0, Some(0)))
            .collect();
        g.players.update(&ps, &g.rel);
        g.step(0);
        hook_then_freeze(&mut g, 2, 0, 10, 20);
        assert_eq!(
            g.clock.stats(),
            BlockStats::default(),
            "a friend hooked us: no attribution"
        );
    }

    #[test]
    fn a_friend_we_hooked_is_not_a_block_and_someone_elses_victim_is_not_ours() {
        let mut g = Fixture::new(3);
        g.rel.add(ListKind::Friend, "p1");
        let ps: Vec<_> = (0..3)
            .map(|i| player(i, &format!("p{i}"), "", i == 0, 0, Some(0)))
            .collect();
        g.players.update(&ps, &g.rel);
        g.step(0);
        hook_then_freeze(&mut g, 0, 1, 10, 20);
        assert_eq!(g.clock.stats().blocks, 0, "friend");
        let mut h = Fixture::new(3);
        h.step(0);
        hook_then_freeze(&mut h, 2, 1, 10, 20);
        assert_eq!(h.clock.stats(), BlockStats::default(), "tee 2 hooked tee 1, not us");
    }

    #[test]
    fn the_previous_hooker_keeps_the_credit_while_still_hooking() {
        let mut f = Fixture::new(3);
        f.step(0);
        // 2 hooks tee 1 first; then 0 joins. Touch stays with 2 (still hooking), so no block for us.
        let mut h2 = f.tee(2);
        h2.hooked_player = 1;
        f.set(h2);
        f.step(2);
        let mut h0 = f.tee(0);
        h0.hooked_player = 1;
        f.set(h0);
        f.step(4);
        let mut v = f.tee(1);
        v.frozen = true;
        f.set(v);
        f.step(6);
        assert_eq!(
            f.clock.stats().blocks,
            0,
            "credit stays with the earlier, still-hooking tee"
        );
        // Once 2 lets go the credit moves to the lowest id hooking: us.
        let mut f = Fixture::new(3);
        f.step(0);
        let mut h2 = f.tee(2);
        h2.hooked_player = 1;
        f.set(h2);
        f.step(2);
        h2.hooked_player = -1;
        f.set(h2);
        let mut h0 = f.tee(0);
        h0.hooked_player = 1;
        f.set(h0);
        f.step(4);
        let mut v = f.tee(1);
        v.frozen = true;
        f.set(v);
        f.step(6);
        assert_eq!(f.clock.stats().blocks, 1);
    }

    #[test]
    fn a_hammer_swing_that_lands_on_the_victim_is_a_touch() {
        let mut f = Fixture::new(2);
        let mut me = f.tee(0);
        me.pos = Vec2::new(0.0, 0.0);
        me.weapon = 0; // hammer
        me.angle = 0; // aiming right
        let mut v = f.tee(1);
        v.pos = Vec2::new(40.0, 0.0);
        f.set(me);
        f.set(v);
        f.step(0);
        me.attack_tick = 4;
        f.set(me);
        f.step(2);
        let mut v = f.tee(1);
        v.frozen = true;
        f.set(v);
        f.step(6);
        assert_eq!(
            f.clock.stats().blocks,
            1,
            "swing point 21 px ahead is within 56 px of the victim"
        );
    }

    #[test]
    fn a_swing_that_misses_or_a_different_weapon_is_no_touch() {
        let mut f = Fixture::new(2);
        let mut me = f.tee(0);
        me.pos = Vec2::new(0.0, 0.0);
        me.angle = 0;
        let mut v = f.tee(1);
        v.pos = Vec2::new(200.0, 0.0);
        f.set(me);
        f.set(v);
        f.step(0);
        me.attack_tick = 4;
        f.set(me);
        f.step(2);
        v.frozen = true;
        f.set(v);
        f.step(4);
        assert_eq!(f.clock.stats().blocks, 0, "200 px away");
        let mut g = Fixture::new(2);
        let mut me = g.tee(0);
        me.weapon = 1; // gun
        let mut v = g.tee(1);
        v.pos = Vec2::new(40.0, 0.0);
        g.set(me);
        g.set(v);
        g.step(0);
        me.attack_tick = 4;
        g.set(me);
        g.step(2);
        v.frozen = true;
        g.set(v);
        g.step(4);
        assert_eq!(g.clock.stats().blocks, 0, "only the hammer touches");
    }

    #[test]
    fn a_teleport_and_a_death_forget_the_touch() {
        let mut f = Fixture::new(2);
        f.step(0);
        let mut h = f.tee(0);
        h.hooked_player = 1;
        f.set(h);
        f.step(2);
        h.hooked_player = -1;
        f.set(h);
        let mut v = f.tee(1);
        v.pos = Vec2::new(1000.0 + 500.0, 0.0); // jumped 500 px in 2 ticks
        f.set(v);
        f.step(4);
        v.frozen = true;
        f.set(v);
        f.step(6);
        assert_eq!(f.clock.stats().blocks, 0, "a teleport between the touch and the freeze");

        let mut g = Fixture::new(2);
        g.step(0);
        let mut h = g.tee(0);
        h.hooked_player = 1;
        g.set(h);
        g.step(2);
        h.hooked_player = -1;
        g.set(h);
        let mut v = g.tee(1);
        g.kill(1);
        g.step(4);
        v.alive = true;
        v.frozen = true;
        g.set(v);
        g.step(6);
        assert_eq!(
            g.clock.stats().blocks,
            0,
            "it died and respawned frozen: the old touch is gone"
        );
    }

    #[test]
    fn a_tee_rejoining_under_a_new_name_restarts_its_record() {
        let mut f = Fixture::new(2);
        f.step(0);
        assert!(f.clock.input_idle(1, 600, false));
        let ps = [
            player(0, "p0", "", true, 0, Some(0)),
            player(1, "someone else", "", false, 0, Some(0)),
        ];
        f.players.update(&ps, &f.rel);
        f.step(600);
        assert!(!f.clock.input_idle(1, 600, false), "a fresh record: just seen");
    }

    #[test]
    fn frozen_for_counts_from_the_onset() {
        let mut f = Fixture::new(2);
        f.step(0);
        let mut v = f.tee(1);
        v.frozen = true;
        f.set(v);
        f.step(10);
        assert_eq!(f.clock.frozen_for(&f.tee(1), 110), 100);
        let free = Tee {
            frozen: false,
            ..f.tee(1)
        };
        assert_eq!(f.clock.frozen_for(&free, 110), 0);
        let _ = HOOK_IDLE;
    }

    #[test]
    fn steady_state_update_allocates_nothing() {
        let mut f = Fixture::new(6);
        for tick in (0..40).step_by(2) {
            f.step(tick);
        }
        let info = allocation_counter::measure(|| {
            for tick in (40..400).step_by(2) {
                f.clock.update(tick, &f.tees, &f.players, 0);
            }
        });
        assert_eq!(info.count_total, 0, "{info:?}");
    }
}
