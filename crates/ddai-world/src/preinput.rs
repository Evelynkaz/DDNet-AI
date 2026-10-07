//! Server pre-inputs (`Sv_PreInput`, task 3.20, D-112): the other tees' REAL inputs, sent by a DDNet >= 19.4 server before the tick they are used on.
//!
//! **What the server does** (`engine/server/server.cpp:1934-1980`, `game/server/gamecontext.cpp:1530`): when a client's input arrives for
//! `IntendedTick <= Tick() + 4 * TickSpeed + 1` and the input differs from the client's previous pre-input in anything but the aim, the server sends
//! `Sv_PreInput { direction, target, jump, fire, hook, weapons, owner, intended_tick }` to every client of the same DDRace team that is not a
//! spectator or AFK, has version >= 19040, and for whom the owner's character is snapped and not network-clipped. **Only changes are sent**
//! (the aim is attached to a change but does not make one), so between two messages the owner's input is the first message's input.
//!
//! **What the DDNet client does** (`gameclient.cpp:1269`, `ApplyPreInputs` at 2572): keeps the message at `[intended_tick % 200]` per owner and feeds it
//! to the prediction world on exactly that tick; the character keeps its input after that.
//!
//! **What this module does.** [`PreInputStore`] is that ring (allocation-free), plus the newest tick heard per owner and counters. The prediction of
//! [`crate::LiveWorld`] reads it through [`Roll`]: a step on tick `t` for owner `id` plays the owner's input as of `t` -- the newest message with
//! `intended_tick <= t` -- **only while `t <= newest(id)`** (up to there the absence of a message really means "unchanged"; beyond it nothing is
//! known and the assumed input -- the window model's or "hold" -- plays on). The state at the snapshot tick is trusted only if its direction equals
//! the snapshot's (a lost message, a spent `sv_max_preinputs_per_tick` budget).
//!
//! Two details of the mapping to the physics input: the **aim** is the message's only on its own tick (it is stale on the ticks in between: the
//! assumed input's aim plays then); the **fire** is not mapped at all: the server fires when the input arrives, before the intended tick, so a swing may already be in the snapshot
//! (review round 2, F9); the assumed fire stays. Weapon fields are not mapped either.

use ddai_physics::core::{MAX_CLIENTS, PlayerInput};

/// Ring size per owner (DDNet: `m_aPreInputs[200]`).
pub const RING: usize = 200;
/// Histogram of the lead of a message over the latest snapshot tick: `lead = intended_tick - snapshot_tick`, clamped to `LEAD_MIN..=LEAD_MAX`.
pub const LEAD_MIN: i32 = -4;
pub const LEAD_MAX: i32 = 11;
pub const LEAD_BINS: usize = (LEAD_MAX - LEAD_MIN + 1) as usize;

#[derive(Clone, Copy)]
struct Slot {
    tick: i32,
    input: PlayerInput,
}

const EMPTY: Slot = Slot {
    tick: -1,
    input: PlayerInput {
        direction: 0,
        target_x: 0,
        target_y: 0,
        jump: 0,
        fire: 0,
        hook: 0,
        player_flags: 0,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    },
};

/// What [`PreInputStore::insert`] did with a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inserted {
    /// Stored in its slot.
    Stored,
    /// The same owner and tick again (a resend): replaced.
    Duplicate,
    /// A message older than what its slot holds (a newer tick of the same residue): ignored.
    Stale,
    /// An owner outside `0..128` or a negative tick: ignored.
    Invalid,
}

/// Counters since the start (STATUS and the periodic log line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreInputCounts {
    pub received: u64,
    pub stored: u64,
    pub stale: u64,
    pub invalid: u64,
    pub duplicate: u64,
    /// Of the stored: those with `intended_tick` ahead of the latest snapshot tick when they arrived (the useful ones) and those at or behind it.
    pub ahead: u64,
    pub behind: u64,
    /// Steps of predictions in which a pre-input drove a character (one per character per step).
    pub used: u64,
    /// Characters whose state at the snapshot disagreed with the pre-input (direction): not trusted.
    pub distrusted: u64,
    /// `lead` of the stored messages, bins `LEAD_MIN..=LEAD_MAX` (the ends hold everything beyond).
    pub lead: [u64; LEAD_BINS],
    /// At each snapshot, for each other tee we have messages of: `newest message tick - snapshot tick`, same bins. This is what a decision
    /// made from that snapshot can use: the ticks past the snapshot the real input is known for (bins above 0).
    pub known_ahead: [u64; LEAD_BINS],
}

impl Default for PreInputCounts {
    fn default() -> Self {
        PreInputCounts {
            received: 0,
            stored: 0,
            stale: 0,
            invalid: 0,
            duplicate: 0,
            ahead: 0,
            behind: 0,
            used: 0,
            distrusted: 0,
            lead: [0; LEAD_BINS],
            known_ahead: [0; LEAD_BINS],
        }
    }
}

pub struct PreInputStore {
    slots: Box<[Slot]>,
    newest: [i32; MAX_CLIENTS],
    counts: PreInputCounts,
    /// Per owner: messages stored (STATUS shows the top few, never names).
    per_owner: [u32; MAX_CLIENTS],
}

impl Default for PreInputStore {
    fn default() -> Self {
        Self::new()
    }
}

impl PreInputStore {
    pub fn new() -> PreInputStore {
        PreInputStore {
            slots: vec![EMPTY; MAX_CLIENTS * RING].into_boxed_slice(),
            newest: [-1; MAX_CLIENTS],
            counts: PreInputCounts::default(),
            per_owner: [0; MAX_CLIENTS],
        }
    }

    fn slot(&self, owner: usize, tick: i32) -> &Slot {
        &self.slots[owner * RING + tick.rem_euclid(RING as i32) as usize]
    }

    /// Stores one message. `snapshot_tick` is the latest snapshot's tick (for the lead histogram; `None` before the first).
    pub fn insert(
        &mut self,
        owner: i32,
        intended_tick: i32,
        input: PlayerInput,
        snapshot_tick: Option<i32>,
    ) -> Inserted {
        self.counts.received += 1;
        let Some(o) = usize::try_from(owner).ok().filter(|&o| o < MAX_CLIENTS) else {
            self.counts.invalid += 1;
            return Inserted::Invalid;
        };
        if intended_tick < 0 {
            self.counts.invalid += 1;
            return Inserted::Invalid;
        }
        let idx = o * RING + intended_tick.rem_euclid(RING as i32) as usize;
        let held = self.slots[idx].tick;
        if held > intended_tick {
            self.counts.stale += 1;
            return Inserted::Stale;
        }
        let dup = held == intended_tick;
        self.slots[idx] = Slot {
            tick: intended_tick,
            input,
        };
        self.newest[o] = self.newest[o].max(intended_tick);
        self.per_owner[o] = self.per_owner[o].saturating_add(1);
        if dup {
            self.counts.duplicate += 1;
            return Inserted::Duplicate;
        }
        self.counts.stored += 1;
        if let Some(s) = snapshot_tick {
            let lead = intended_tick - s;
            if lead > 0 {
                self.counts.ahead += 1;
            } else {
                self.counts.behind += 1;
            }
            self.counts.lead[(lead.clamp(LEAD_MIN, LEAD_MAX) - LEAD_MIN) as usize] += 1;
        }
        Inserted::Stored
    }

    /// Forgets everything of `owner` (it left, died, changed team: what it did before is no longer what it does).
    pub fn forget(&mut self, owner: i32) {
        if let Some(o) = usize::try_from(owner).ok().filter(|&o| o < MAX_CLIENTS) {
            for s in &mut self.slots[o * RING..(o + 1) * RING] {
                *s = EMPTY;
            }
            self.newest[o] = -1;
        }
    }

    /// Forgets everyone (a new map, a reconnect).
    pub fn clear(&mut self) {
        for o in 0..MAX_CLIENTS as i32 {
            self.forget(o);
        }
    }

    /// The newest `intended_tick` heard of `owner` (`-1`: none).
    pub fn newest(&self, owner: i32) -> i32 {
        usize::try_from(owner)
            .ok()
            .and_then(|o| self.newest.get(o).copied())
            .unwrap_or(-1)
    }

    /// The message `owner` has for exactly `tick`, if any.
    pub fn exact(&self, owner: i32, tick: i32) -> Option<&PlayerInput> {
        let o = usize::try_from(owner).ok().filter(|&o| o < MAX_CLIENTS)?;
        let s = self.slot(o, tick);
        (s.tick == tick).then_some(&s.input)
    }

    /// The newest message of `owner` with `intended_tick <= tick` within one ring (the input in force at `tick` if nothing was missed), with its tick.
    pub fn as_of(&self, owner: i32, tick: i32) -> Option<(i32, &PlayerInput)> {
        let o = usize::try_from(owner).ok().filter(|&o| o < MAX_CLIENTS)?;
        (0..RING as i32).find_map(|k| {
            let t = tick - k;
            let s = self.slot(o, t);
            (s.tick == t && t >= 0).then_some((t, &s.input))
        })
    }

    pub fn counts(&self) -> PreInputCounts {
        self.counts
    }

    /// The owners with the most messages: `(id, count)`, at most `n`, most first. Ids only.
    pub fn busiest(&self, n: usize) -> Vec<(i32, u32)> {
        let mut v: Vec<(i32, u32)> = self
            .per_owner
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(i, &c)| (i as i32, c))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(n);
        v
    }

    /// A snapshot at `snapshot_tick` was taken in: records how far past it the messages of each of the `ids` reach.
    pub(crate) fn note_snapshot(&mut self, snapshot_tick: i32, ids: impl Iterator<Item = i32>) {
        for id in ids {
            let n = self.newest(id);
            if n >= 0 {
                self.counts.known_ahead[((n - snapshot_tick).clamp(LEAD_MIN, LEAD_MAX) - LEAD_MIN) as usize] += 1;
            }
        }
    }

    pub(crate) fn note_used(&mut self, n: u64) {
        self.counts.used += n;
    }

    pub(crate) fn note_distrusted(&mut self) {
        self.counts.distrusted += 1;
    }
}

/// Which step of which owner a [`Roll::input`] call is for.
#[derive(Debug, Clone, Copy)]
pub struct Step {
    pub owner: i32,
    /// The tick the step goes into.
    pub tick: i32,
    /// The snapshot's tick, and the direction it shows for the owner (the trust check).
    pub base_tick: i32,
    pub snapshot_dir: i32,
    /// What the snapshot shows of the hook and the jump key (`Some(held)`), or `None` when it cannot tell (frozen, or the jump flags are ambiguous).
    pub snapshot_hook: Option<bool>,
    pub snapshot_jump: Option<bool>,
    /// The assumed input is the window model's prediction (it plays past the newest message), not hold (the last real input does).
    pub assumed_is_model: bool,
}

/// The state of one prediction roll over the store: per owner whether the state at the snapshot was trusted. Stack-only.
pub struct Roll {
    /// The owner's state at the snapshot was checked: trusted or not.
    trusted: [Option<bool>; MAX_CLIENTS],
}

impl Roll {
    pub fn new() -> Roll {
        Roll {
            trusted: [None; MAX_CLIENTS],
        }
    }

    /// The input of `owner` for the step into `tick`, or `None` when the store knows nothing for it (then the assumed input plays).
    ///
    /// `assumed` is the input that would play without pre-inputs (its aim and flags are kept where the message is stale); `snapshot_dir` is the
    /// direction the snapshot at `base_tick` shows for the owner. Counts a use in the store.
    pub fn input(&mut self, store: &mut PreInputStore, step: &Step, assumed: &PlayerInput) -> Option<PlayerInput> {
        let Step {
            owner,
            tick,
            base_tick,
            snapshot_dir,
            snapshot_hook,
            snapshot_jump,
            assumed_is_model,
        } = *step;
        let o = usize::try_from(owner).ok().filter(|&o| o < MAX_CLIENTS)?;
        let newest = store.newest(owner);
        // Past the newest message nothing is known of a change. The window model's prediction plays then; without one, the last real input
        // persists (a better "hold" than the snapshot-derived one: it has the jump and hook keys).
        if tick > newest && assumed_is_model {
            return None;
        }
        let (at, msg) = store.as_of(owner, tick.min(newest))?;
        let msg = *msg;
        // The state at the snapshot must agree with what the snapshot shows, once per roll and owner.
        if self.trusted[o].is_none() {
            let ok = match store.as_of(owner, base_tick) {
                // `Sv_PreInput` is neither vital nor repeated (the server sends changes only), so a lost message leaves a stale state behind: the
                // state at the snapshot must agree on the direction and, when the snapshot can tell, on the hook and the jump key too.
                Some((_, m)) => {
                    m.direction.clamp(-1, 1) == snapshot_dir
                        && snapshot_hook.is_none_or(|h| h == (m.hook != 0))
                        && snapshot_jump.is_none_or(|j| j == (m.jump != 0))
                }
                // Nothing at or before the snapshot: the first message is a change from the unknown; trust what it says from its tick on.
                None => true,
            };
            if !ok {
                store.note_distrusted();
            }
            self.trusted[o] = Some(ok);
        }
        if self.trusted[o] == Some(false) {
            return None;
        }
        let mut input = *assumed;
        input.direction = msg.direction.clamp(-1, 1);
        input.jump = msg.jump;
        input.hook = msg.hook;
        // `fire` is deliberately NOT taken from the message: the server fires when the input ARRIVES (`OnClientDirectInput` -> `FireWeapon`,
        // `server.cpp:1989`), earlier than the intended tick, so a swing may already be in the snapshot and replaying it would add a phantom one.
        // The assumed fire stays (D-112).
        if at == tick {
            // The aim is the message's only on its own tick.
            input.target_x = msg.target_x;
            input.target_y = msg.target_y;
        }
        store.note_used(1);
        Some(input)
    }
}

impl Default for Roll {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(owner: i32, tick: i32, base_tick: i32, snapshot_dir: i32, assumed_is_model: bool) -> Step {
        Step {
            owner,
            tick,
            base_tick,
            snapshot_dir,
            snapshot_hook: None,
            snapshot_jump: None,
            assumed_is_model,
        }
    }

    fn msg(direction: i32, hook: i32, fire: i32) -> PlayerInput {
        PlayerInput {
            direction,
            hook,
            fire,
            target_x: 10,
            target_y: -20,
            ..PlayerInput::default()
        }
    }

    #[test]
    fn a_message_is_found_by_its_tick_and_persists_until_the_next_one() {
        let mut s = PreInputStore::new();
        assert_eq!(s.insert(3, 100, msg(1, 0, 0), Some(98)), Inserted::Stored);
        assert_eq!(s.insert(3, 104, msg(-1, 1, 0), Some(98)), Inserted::Stored);
        assert_eq!(s.exact(3, 100).map(|m| m.direction), Some(1));
        assert!(s.exact(3, 101).is_none());
        assert_eq!(s.as_of(3, 103).map(|(t, m)| (t, m.direction)), Some((100, 1)));
        assert_eq!(s.as_of(3, 104).map(|(t, m)| (t, m.direction)), Some((104, -1)));
        assert_eq!(s.as_of(3, 99), None, "nothing before the first message");
        assert_eq!(s.newest(3), 104);
        assert_eq!(s.newest(4), -1);
        let c = s.counts();
        assert_eq!((c.received, c.stored, c.ahead, c.behind), (2, 2, 2, 0));
        assert_eq!(c.lead[(2 - LEAD_MIN) as usize], 1);
        assert_eq!(c.lead[(6 - LEAD_MIN) as usize], 1);
    }

    #[test]
    fn stale_duplicate_and_invalid_messages_are_told_apart() {
        let mut s = PreInputStore::new();
        s.insert(1, 300, msg(1, 0, 0), None);
        // The same residue (300 % 200 == 100), an older tick: the slot keeps the newer message.
        assert_eq!(s.insert(1, 100, msg(-1, 0, 0), None), Inserted::Stale);
        assert_eq!(s.exact(1, 300).map(|m| m.direction), Some(1));
        assert_eq!(s.insert(1, 300, msg(1, 1, 0), None), Inserted::Duplicate);
        assert_eq!(s.exact(1, 300).map(|m| m.hook), Some(1), "a resend replaces");
        assert_eq!(s.insert(-1, 5, msg(0, 0, 0), None), Inserted::Invalid);
        assert_eq!(s.insert(128, 5, msg(0, 0, 0), None), Inserted::Invalid);
        assert_eq!(s.insert(1, -5, msg(0, 0, 0), None), Inserted::Invalid);
        let c = s.counts();
        assert_eq!((c.received, c.stored, c.stale, c.duplicate, c.invalid), (6, 1, 1, 1, 3));
    }

    #[test]
    fn an_out_of_order_message_does_not_hide_a_newer_one() {
        let mut s = PreInputStore::new();
        s.insert(2, 110, msg(1, 0, 0), None);
        s.insert(2, 105, msg(-1, 0, 0), None); // arrives late
        assert_eq!(s.as_of(2, 112).map(|(t, _)| t), Some(110));
        assert_eq!(s.as_of(2, 107).map(|(t, _)| t), Some(105));
        assert_eq!(s.newest(2), 110);
    }

    #[test]
    fn forgetting_an_owner_clears_only_that_owner() {
        let mut s = PreInputStore::new();
        s.insert(2, 110, msg(1, 0, 0), None);
        s.insert(5, 110, msg(1, 0, 0), None);
        s.forget(2);
        assert!(s.as_of(2, 120).is_none() && s.newest(2) == -1);
        assert!(s.as_of(5, 120).is_some());
        s.clear();
        assert!(s.as_of(5, 120).is_none());
    }

    #[test]
    fn the_roll_plays_messages_up_to_the_newest_and_the_assumed_input_beyond() {
        let mut s = PreInputStore::new();
        s.insert(1, 101, msg(1, 0, 0), Some(100));
        s.insert(1, 103, msg(-1, 1, 0), Some(100));
        let assumed = PlayerInput {
            direction: 0,
            target_x: 7,
            target_y: 8,
            player_flags: 1,
            ..PlayerInput::default()
        };
        let mut r = Roll::new();
        // The snapshot at 100 shows direction 0 and there is no message at or before it: trusted.
        let at = |r: &mut Roll, s: &mut PreInputStore, t| r.input(s, &st(1, t, 100, 0, false), &assumed);
        let a = at(&mut r, &mut s, 101).unwrap();
        assert_eq!(
            (a.direction, a.target_x, a.player_flags),
            (1, 10, 1),
            "exact tick: the message's aim"
        );
        let b = at(&mut r, &mut s, 102).unwrap();
        assert_eq!((b.direction, b.target_x), (1, 7), "persisted: the assumed aim");
        let c = at(&mut r, &mut s, 103).unwrap();
        assert_eq!((c.direction, c.hook, c.target_x), (-1, 1, 10));
        // Beyond the newest message: with a window model the model plays; without, the last real input persists (its keys, the assumed aim).
        assert!(r.input(&mut s, &st(1, 104, 100, 0, true), &assumed).is_none());
        let d = at(&mut r, &mut s, 104).unwrap();
        assert_eq!((d.direction, d.hook, d.target_x), (-1, 1, 7));
        assert_eq!(s.counts().used, 4);
        // Another owner is untouched.
        assert!(r.input(&mut s, &st(2, 101, 100, 0, false), &assumed).is_none());
    }

    #[test]
    fn a_state_that_contradicts_the_snapshot_is_not_trusted() {
        let mut s = PreInputStore::new();
        s.insert(1, 95, msg(1, 0, 0), Some(94));
        s.insert(1, 102, msg(1, 0, 0), Some(100));
        let assumed = PlayerInput::default();
        let mut r = Roll::new();
        // The message in force at the snapshot says "right", the snapshot shows "left": a message was lost. Nothing is played.
        assert!(r.input(&mut s, &st(1, 102, 100, -1, false), &assumed).is_none());
        assert!(r.input(&mut s, &st(1, 102, 100, -1, false), &assumed).is_none());
        assert_eq!(s.counts().distrusted, 1, "once per roll and owner");
        assert_eq!(s.counts().used, 0);
    }

    #[test]
    fn fire_is_never_taken_from_a_message_so_no_phantom_swing_is_replayed() {
        let mut s = PreInputStore::new();
        s.insert(1, 100, msg(0, 0, 41), Some(98));
        s.insert(1, 101, msg(0, 0, 42), Some(98)); // a press at the first rolled tick
        s.insert(1, 103, msg(0, 0, 43), Some(98));
        let assumed = PlayerInput {
            fire: 7,
            ..PlayerInput::default()
        };
        let mut r = Roll::new();
        for t in 100..=103 {
            let got = r.input(&mut s, &st(1, t, 99, 0, false), &assumed).unwrap();
            assert_eq!(got.fire, 7, "tick {t}: the assumed fire plays");
        }
    }

    #[test]
    fn a_lost_release_of_the_hook_or_the_jump_key_is_noticed_from_the_snapshot() {
        let mut s = PreInputStore::new();
        // "hook down" and "jump down" at 95; the release messages were lost. The snapshot at 100 shows neither held.
        s.insert(1, 95, msg(0, 1, 0), Some(94));
        let mut j = msg(0, 0, 0);
        j.jump = 1;
        s.insert(2, 95, j, Some(94));
        s.insert(1, 102, msg(0, 1, 0), Some(100));
        s.insert(2, 102, j, Some(100));
        let assumed = PlayerInput::default();
        let step = |owner, hook, jump| Step {
            snapshot_hook: hook,
            snapshot_jump: jump,
            ..st(owner, 102, 100, 0, false)
        };
        let mut r = Roll::new();
        assert!(
            r.input(&mut s, &step(1, Some(false), None), &assumed).is_none(),
            "stale hook"
        );
        assert!(
            r.input(&mut s, &step(2, None, Some(false)), &assumed).is_none(),
            "stale jump"
        );
        assert_eq!(s.counts().distrusted, 2);
        // Agreeing snapshots, and a snapshot that cannot tell (frozen), are trusted.
        let mut r = Roll::new();
        assert!(r.input(&mut s, &step(1, Some(true), None), &assumed).is_some());
        assert!(r.input(&mut s, &step(2, None, Some(true)), &assumed).is_some());
        let mut r = Roll::new();
        assert!(r.input(&mut s, &step(1, None, None), &assumed).is_some());
    }
}
