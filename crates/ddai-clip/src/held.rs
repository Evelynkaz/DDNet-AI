//! Task 3.10: **what became of every block we made** (the live diagnosis of `docs/research/held-block.md`).
//!
//! A clip is a 30 s window of snapshot frames with the bot's own state (`BotRec`: the picked target and the D-059
//! attribution counters) and the real events. A *block* is a frame where `BotRec::blocks` goes up: we froze an
//! opponent. [`block_fates`] follows that victim for [`HELD_TICKS`] (5 s, longer than `sv_freeze_delay` = 3 s) and says whether the
//! block was held (the victim stayed frozen or died), escaped (it thawed and was free again), or could not be told (the victim left the
//! recorded tees, or the clip ended first). For an escape it names the likely cause from what the bot did meanwhile.
//!
//! The causes are read off the recording, not guessed from the plans (a clip holds no candidate pool), so each has a stated rule:
//!
//! * [`Why::Switched`]: the picked target was not the victim at some frame between the block and the escape (the bot let go);
//! * [`Why::NoReach`]: it kept the victim as its target but never touched it again (no hook attach, no hammer hit) before the escape: no plan got there;
//! * [`Why::LateInput`]: it touched the victim (the finishing contact happened) but at least one input of that stretch was applied
//!   later than aimed (`SentRec::tick` past `BotRec::aimed_tick`);
//! * [`Why::Slipped`]: it touched the victim in time and nothing was late, yet the victim got out: the push or the prediction was wrong.
//!
//! Nothing here reads a nickname: players are ids and 4.1 tags.

use crate::format::{ClipEvent, Frame};

/// The window a block must stay on to count as held: 250 ticks (5 s).
pub const HELD_TICKS: i32 = 250;

/// What became of the victim of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fate {
    /// Frozen for the whole window.
    Held,
    /// Died (a kill message): `weapon` is `-1` the world (a kill tile), `-2` its own `/kill`, `-3` the game's end. A death is out of the fight (the
    /// block counts as held), but a `/kill` is also how a frozen player escapes a block, so it is told apart.
    Killed { weapon: i32 },
    /// Free again `after` ticks after the block.
    Escaped { after: i32, why: Why },
    /// Not decidable from the clip (see [`Unknown`]).
    Unknown(Unknown),
}

/// Why an escaped block escaped (see the module docs for the rules).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Why {
    Switched,
    NoReach,
    LateInput,
    Slipped,
}

/// Why a block's fate is not known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unknown {
    /// The clip ends before the window is over, and the victim was still out at its end.
    ClipEnds,
    /// The victim left the recorded tees (far away or gone) while still out, with no kill seen.
    OutOfView,
}

/// One block and what became of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockFate {
    /// The block's frame index and tick.
    pub frame: usize,
    pub tick: i32,
    pub victim: i32,
    pub fate: Fate,
    /// Ticks the victim stayed out in view (frozen or dead) until the fate was known.
    pub out_ticks: i32,
    /// The victim was still our target on the frame the block was counted.
    pub target_at_block: bool,
    /// Frames (of those up to the escape or the window's end) where the target was not the victim.
    pub frames_off_target: u32,
    /// Hook attaches and hammer hits of ours on the victim after the block, up to the fate.
    pub touches: u32,
}

/// Whether we touched `id` (hook attach/release on it, a hammer hit on it, or our rope on it in a frame) in the 40 frames before frame `i`.
fn touched_by_us(frames: &[Frame], i: usize, own_id: i32, id: i32) -> bool {
    frames[i.saturating_sub(40)..=i].iter().any(|f| {
        f.tee(own_id).is_some_and(|t| t.ch.hooked_player == id)
            || f.events.iter().any(|e| match *e {
                ClipEvent::HookAttach { id: by, target } | ClipEvent::HookRelease { id: by, target, .. } => {
                    by == own_id && target == id
                }
                ClipEvent::HammerHit { from, to } => from == own_id && to == id,
                _ => false,
            })
    })
}

/// The victim of the block counted on frame `i`: of the opponents that went frozen on the frame or the two before, the one we touched
/// (a crowd freezes several at once and only ours is the block); else the first; else the target.
fn victim_of(frames: &[Frame], i: usize, own_id: i32) -> Option<i32> {
    let mut first = None;
    for k in [i, i.saturating_sub(1), i.saturating_sub(2)] {
        for e in &frames[k].events {
            if let ClipEvent::FreezeOnset { id } = e
                && *id != own_id
            {
                if touched_by_us(frames, i, own_id, *id) {
                    return Some(*id);
                }
                first.get_or_insert(*id);
            }
        }
    }
    first.or_else(|| {
        let t = frames[i].bot.target;
        (t >= 0 && t != own_id).then_some(t)
    })
}

/// The fate of every block in `frames` (`own_id`: our client id). Blocks whose victim cannot be named are skipped.
pub fn block_fates(frames: &[Frame], own_id: i32) -> Vec<BlockFate> {
    let mut out = Vec::new();
    for i in 1..frames.len() {
        if frames[i].bot.blocks <= frames[i - 1].bot.blocks {
            continue;
        }
        let Some(victim) = victim_of(frames, i, own_id) else {
            continue;
        };
        out.push(follow(frames, i, own_id, victim));
    }
    out
}

fn follow(frames: &[Frame], i: usize, own_id: i32, victim: i32) -> BlockFate {
    let t0 = frames[i].tick;
    let target_at_block = frames[i].bot.target == victim;
    let mut last_out = t0;
    let mut touches = 0u32;
    let mut off_target = 0u32;
    let mut late = false;
    let mut left_target_at = None::<i32>;
    let mut fate = None;
    let mut dead = None::<i32>;
    for f in &frames[i + 1..] {
        let dt = f.tick - t0;
        if dt > HELD_TICKS {
            break;
        }
        for e in &f.events {
            match *e {
                ClipEvent::HookAttach { id, target } if id == own_id && target == victim => touches += 1,
                ClipEvent::HammerHit { from, to } if from == own_id && to == victim => touches += 1,
                ClipEvent::Kill { victim: v, weapon, .. } if v == victim => dead = Some(weapon),
                _ => {}
            }
        }
        if f.bot.target != victim {
            off_target += 1;
            left_target_at.get_or_insert(f.tick);
        }
        if f.bot.aimed_tick > 0 && f.sent.iter().any(|s| s.tick > f.bot.aimed_tick) {
            late = true;
        }
        if let Some(weapon) = dead {
            // A kill is a block that holds (the victim is out of the game until its respawn).
            last_out = f.tick;
            fate = Some(Fate::Killed { weapon });
            break;
        }
        match f.tee(victim) {
            Some(t) if t.frozen => last_out = f.tick,
            Some(_) => {
                let why = if left_target_at.is_some() {
                    Why::Switched
                } else if touches == 0 {
                    Why::NoReach
                } else if late {
                    Why::LateInput
                } else {
                    Why::Slipped
                };
                fate = Some(Fate::Escaped { after: dt, why });
                break;
            }
            None => {}
        }
    }
    let fate = fate.unwrap_or_else(|| {
        let end = frames.last().map_or(t0, |f| f.tick);
        let seen_to = last_out - t0;
        if end - t0 >= HELD_TICKS {
            // The window is covered by frames; the victim either stayed out of view or kept frozen.
            if seen_to >= HELD_TICKS - 8 {
                Fate::Held
            } else {
                Fate::Unknown(Unknown::OutOfView)
            }
        } else {
            Fate::Unknown(Unknown::ClipEnds)
        }
    });
    BlockFate {
        frame: i,
        tick: t0,
        victim,
        fate,
        out_ticks: last_out - t0,
        target_at_block,
        frames_off_target: off_target,
        touches,
    }
}

/// Counts for a set of fates: held, escaped by cause, unknown.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FateCounts {
    pub blocks: u32,
    pub held: u32,
    pub killed: u32,
    pub killed_self: u32,
    pub switched: u32,
    pub no_reach: u32,
    pub late_input: u32,
    pub slipped: u32,
    pub clip_ends: u32,
    pub out_of_view: u32,
}

impl FateCounts {
    pub fn add(&mut self, f: &BlockFate) {
        self.blocks += 1;
        match f.fate {
            Fate::Held => self.held += 1,
            Fate::Killed { weapon } => {
                self.killed += 1;
                self.killed_self += u32::from(weapon == -2);
            }
            Fate::Escaped { why, .. } => match why {
                Why::Switched => self.switched += 1,
                Why::NoReach => self.no_reach += 1,
                Why::LateInput => self.late_input += 1,
                Why::Slipped => self.slipped += 1,
            },
            Fate::Unknown(Unknown::ClipEnds) => self.clip_ends += 1,
            Fate::Unknown(Unknown::OutOfView) => self.out_of_view += 1,
        }
    }

    pub fn escaped(&self) -> u32 {
        self.switched + self.no_reach + self.late_input + self.slipped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BotRec, CharRec, TeeRec};

    fn tee(id: i32, frozen: bool) -> TeeRec {
        TeeRec {
            id,
            ch: CharRec::default(),
            frozen,
            ..TeeRec::default()
        }
    }

    fn frame(tick: i32, blocks: u16, target: i32, victim_frozen: Option<bool>, events: Vec<ClipEvent>) -> Frame {
        let mut tees = vec![tee(0, false)];
        if let Some(fz) = victim_frozen {
            tees.push(tee(1, fz));
        }
        Frame {
            tick,
            own_alive: true,
            tees,
            events,
            bot: BotRec {
                target,
                blocks,
                ..BotRec::default()
            },
            ..Frame::default()
        }
    }

    /// Frames every 2 ticks from tick 100; the block is counted on the second frame. `victim_at(dt)` is the victim's state.
    fn run(
        n: usize,
        target_at: impl Fn(i32) -> i32,
        victim_at: impl Fn(i32) -> Option<bool>,
        events_at: impl Fn(i32) -> Vec<ClipEvent>,
    ) -> Vec<BlockFate> {
        let mut frames = vec![frame(100, 0, 1, Some(false), vec![])];
        for k in 1..n {
            let tick = 100 + 2 * k as i32;
            let dt = tick - 102;
            let mut ev = events_at(dt);
            if k == 1 {
                ev.push(ClipEvent::FreezeOnset { id: 1 });
            }
            frames.push(frame(tick, 1, target_at(dt), victim_at(dt), ev));
        }
        block_fates(&frames, 0)
    }

    #[test]
    fn a_victim_frozen_for_the_whole_window_is_held() {
        let f = run(200, |_| 1, |_| Some(true), |_| vec![]);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].victim, 1);
        assert_eq!(f[0].fate, Fate::Held);
        assert!(f[0].target_at_block);
    }

    #[test]
    fn a_thawed_victim_escaped_and_the_cause_follows_the_rules() {
        // Still the target, never touched again: no plan reached it.
        let f = run(200, |_| 1, |dt| Some(dt < 100), |_| vec![]);
        assert_eq!(
            f[0].fate,
            Fate::Escaped {
                after: 100,
                why: Why::NoReach
            }
        );
        // The bot let go of the target before the escape.
        let f = run(200, |dt| if dt > 40 { -1 } else { 1 }, |dt| Some(dt < 100), |_| vec![]);
        assert!(matches!(f[0].fate, Fate::Escaped { why: Why::Switched, .. }));
        assert!(f[0].frames_off_target > 0);
        // A hook attach on the victim after the block: it did reach it, and nothing was late.
        let touch = |dt| {
            if dt == 20 {
                vec![ClipEvent::HookAttach { id: 0, target: 1 }]
            } else {
                vec![]
            }
        };
        let f = run(200, |_| 1, |dt| Some(dt < 100), touch);
        assert!(
            matches!(f[0].fate, Fate::Escaped { why: Why::Slipped, .. }),
            "{:?}",
            f[0]
        );
        assert_eq!(f[0].touches, 1);
    }

    #[test]
    fn a_kill_holds_and_a_short_clip_or_a_lost_victim_is_unknown() {
        let kill = |dt| {
            if dt == 30 {
                vec![ClipEvent::Kill {
                    killer: -1,
                    victim: 1,
                    weapon: -1,
                }]
            } else {
                vec![]
            }
        };
        assert_eq!(
            run(200, |_| 1, |dt| (dt < 30).then_some(true), kill)[0].fate,
            Fate::Killed { weapon: -1 }
        );
        assert_eq!(
            run(40, |_| 1, |_| Some(true), |_| vec![])[0].fate,
            Fate::Unknown(Unknown::ClipEnds)
        );
        assert_eq!(
            run(200, |_| 1, |dt| (dt < 30).then_some(true), |_| vec![])[0].fate,
            Fate::Unknown(Unknown::OutOfView)
        );
    }

    #[test]
    fn counts_add_up() {
        let mut c = FateCounts::default();
        for f in run(200, |_| 1, |_| Some(true), |_| vec![]) {
            c.add(&f);
        }
        assert_eq!((c.blocks, c.held, c.escaped()), (1, 1, 0));
    }
}
