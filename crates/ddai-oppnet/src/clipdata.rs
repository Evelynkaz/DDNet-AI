//! Task 3.21 (E-036): live clips as training and evaluation data for the opponent-input predictor.
//!
//! A clip is the live bot's ring of snapshot frames, 2 ticks apart. The opponent's raw inputs are not in it, so the labels are **what the later
//! snapshots show**: the snapshot at tick `T + 2` shows the direction, the aim and the hook state of the step `T + 1` (so window tick `k = 1`), and the tick
//! of the opponent's last weapon use (`attack_tick`), which says exactly *when* in `T .. T + 2` a swing happened (`k = 0` or `k = 1`). The steps
//! `k = 0, 2, 4, ..` are invisible, except for the swings. [`ClipLabels`] keeps these facts per head and per tick as masks.
//!
//! The data types are plain `serde` records, the conversion from `ddai_clip::format::Clip` (through `LiveWorld`, exactly the path the live bot takes) is
//! in `ddai-env`'s `live_data` example, so this crate does not depend on the clip crate. **No nicknames**: a game carries a source tag and a session number.

use serde::{Deserialize, Serialize};

use crate::frame::{InputRec, N_RAYS, TeeFrame};

/// One snapshot frame of a clip, as the live bot's history sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipTick {
    /// The snapshot tick.
    pub tick: i32,
    /// `[us, the opponent]`, from the world `LiveWorld` rebuilds out of the snapshot.
    pub frames: [TeeFrame; 2],
    /// The geometry rays (`frame::rays`) around us and around the opponent, in the order of `frames`.
    pub rays: [[f32; N_RAYS]; 2],
    /// Our inputs in force at the ticks `tick - 1` and `tick` (the two ticks since the previous frame), as sent.
    pub sent: [Option<InputRec>; 2],
    /// The opponent's raw `attack_tick` and weapon (the tick of its last weapon use).
    pub opp_attack_tick: i32,
    pub opp_weapon: i8,
    /// The frame is part of a duel the model is asked about: exactly the two tees of the pair (none dropped), both alive, close enough (the regime gate).
    pub duel: bool,
}

/// A run of consecutive frames (2 ticks apart) of one clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipGame {
    /// A source tag without any player name (`s<session>-<clip number>`).
    pub source: String,
    /// The session the clip comes from (the unit of the train / held-out split).
    pub session: u8,
    pub ticks: Vec<ClipTick>,
}

impl ClipGame {
    /// The input of ours in force at world tick `tag` (the step into tick `tag`), searched from frame `i` on.
    pub fn sent_at(&self, i: usize, tag: i32) -> Option<InputRec> {
        self.ticks.get(i..)?.iter().take(6).find_map(|t| {
            if tag == t.tick - 1 {
                t.sent[0]
            } else if tag == t.tick {
                t.sent[1]
            } else {
                None
            }
        })
    }

    /// Whether the frames `i - back ..= i + fwd` exist and are consecutive (2 ticks apart).
    pub fn consecutive(&self, i: usize, back: usize, fwd: usize) -> bool {
        let (Some(lo), hi) = (i.checked_sub(back), i + fwd) else {
            return false;
        };
        if hi >= self.ticks.len() {
            return false;
        }
        (lo..hi).all(|j| self.ticks[j + 1].tick == self.ticks[j].tick + 2)
    }
}

/// The most window ticks a [`ClipLabels`] holds.
pub const MAX_H: usize = 8;

/// What the later snapshots show of the opponent's inputs for window ticks `0 .. horizon`, with masks for the ticks they show.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ClipLabels {
    /// Bit `k`: the direction / hook / aim of window tick `k` is shown by a snapshot (`k` odd, the next frame exists and the opponent is free).
    pub v_obs: u8,
    /// Bit `k`: a swing is decided for window tick `k` (`k = 0, 1` need the frame at `T + 2`, `k = 2, 3` the frame at `T + 4`, ...).
    pub v_press: u8,
    pub dir: [i8; MAX_H],
    /// The hook state of the step is above idle (the reading of the guard and of hold).
    pub hook: u8,
    /// Bit `k`: the weapon was used in step `k`.
    pub press: u8,
    /// Applied aim angle minus the angle of the sample's frame (radians, wrapped) -- from the angle the snapshot shows.
    pub aim_delta: [f32; MAX_H],
}

/// Labels of the sample at frame `i`: needs the frames up to `i + (horizon + 1) / 2` consecutive (a shorter run labels fewer ticks). The opponent must be alive and
/// free (not frozen) in the frame that shows a tick, and so in the frame of the sample.
pub fn labels_at(g: &ClipGame, i: usize, horizon: usize) -> ClipLabels {
    let h = horizon.min(MAX_H);
    let mut l = ClipLabels::default();
    let t0 = &g.ticks[i];
    let base = f64::from(t0.frames[1].angle);
    for m in 1..=h.div_ceil(2) {
        let Some(next) = g.ticks.get(i + m) else { break };
        if next.tick != t0.tick + 2 * m as i32 || !g.consecutive(i, 0, m) {
            break;
        }
        let f = &next.frames[1];
        if !f.alive || f.freeze_left > 0 || !next.duel {
            break;
        }
        let k = 2 * m - 1;
        if k < h {
            l.v_obs |= 1 << k;
            l.dir[k] = f.direction;
            if f.hook_state > 0 {
                l.hook |= 1 << k;
            }
            l.aim_delta[k] = crate::feature::wrap_angle(f64::from(f.angle) - base) as f32;
        }
        // Swings in the two steps `k = 2m - 2, 2m - 1`: the tick of the weapon use is the step's own tick.
        let prev_attack = g.ticks[i + m - 1].opp_attack_tick;
        for kk in [2 * m - 2, 2 * m - 1] {
            if kk < h {
                l.v_press |= 1 << kk;
            }
        }
        if next.opp_attack_tick != prev_attack {
            let k_press = next.opp_attack_tick - t0.tick;
            if (2 * m as i32 - 2..=2 * m as i32 - 1).contains(&k_press) && (k_press as usize) < h {
                l.press |= 1 << k_press;
            }
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tick(t: i32, dir: i8, attack: i32, hook: i8) -> ClipTick {
        let f = TeeFrame {
            alive: true,
            direction: dir,
            hook_state: hook,
            ..TeeFrame::default()
        };
        ClipTick {
            tick: t,
            frames: [f, f],
            rays: [[1.0; N_RAYS]; 2],
            sent: [None, None],
            opp_attack_tick: attack,
            opp_weapon: 0,
            duel: true,
        }
    }

    fn game(ticks: Vec<ClipTick>) -> ClipGame {
        ClipGame {
            source: "t".into(),
            session: 0,
            ticks,
        }
    }

    #[test]
    fn a_later_frame_labels_the_odd_tick_and_the_swing_tick() {
        // Frames at 100, 102, 104; a swing at tick 101 (k = 1 of the sample at 100), none else.
        let g = game(vec![tick(100, 0, 90, 0), tick(102, 1, 101, 5), tick(104, -1, 101, 0)]);
        let l = labels_at(&g, 0, 4);
        assert_eq!(l.v_obs, 0b1010, "k = 1 and k = 3");
        assert_eq!((l.dir[1], l.dir[3]), (1, -1));
        assert_eq!(l.hook, 0b0010);
        assert_eq!(l.v_press, 0b1111);
        assert_eq!(l.press, 0b0010, "the swing of tick 101 is window tick 1");
        // A swing at tick 100 is window tick 0.
        let g = game(vec![tick(100, 0, 90, 0), tick(102, 0, 100, 0), tick(104, 0, 100, 0)]);
        assert_eq!(labels_at(&g, 0, 4).press, 0b0001);
        // A swing at 103 belongs to the second frame pair: k = 3.
        let g = game(vec![tick(100, 0, 90, 0), tick(102, 0, 90, 0), tick(104, 0, 103, 0)]);
        assert_eq!(labels_at(&g, 0, 4).press, 0b1000);
    }

    #[test]
    fn a_gap_or_a_frozen_opponent_stops_the_labels() {
        let g = game(vec![tick(100, 0, 90, 0), tick(102, 1, 90, 0), tick(106, 1, 90, 0)]);
        let l = labels_at(&g, 0, 4);
        assert_eq!(l.v_obs, 0b0010, "the frame at 106 is not the one after 104");
        let mut frozen = tick(102, 1, 90, 0);
        frozen.frames[1].freeze_left = 5;
        let g = game(vec![tick(100, 0, 90, 0), frozen]);
        assert_eq!(labels_at(&g, 0, 4), ClipLabels::default());
        // The last frame has no future.
        let g = game(vec![tick(100, 0, 90, 0)]);
        assert_eq!(labels_at(&g, 0, 4).v_obs, 0);
    }

    #[test]
    fn sent_inputs_are_found_by_their_tag_and_runs_are_checked() {
        let mut a = tick(100, 0, 0, 0);
        let mut b = tick(102, 0, 0, 0);
        let i = |d| InputRec {
            direction: d,
            ..InputRec::default()
        };
        a.sent = [Some(i(1)), Some(i(2))];
        b.sent = [Some(i(3)), Some(i(4))];
        let g = game(vec![a, b, tick(104, 0, 0, 0)]);
        assert_eq!(g.sent_at(0, 100).map(|x| x.direction), Some(2));
        assert_eq!(g.sent_at(0, 101).map(|x| x.direction), Some(3));
        assert_eq!(g.sent_at(0, 102).map(|x| x.direction), Some(4));
        assert_eq!(g.sent_at(0, 105), None);
        assert!(g.consecutive(1, 1, 1));
        assert!(!g.consecutive(1, 2, 0));
        assert!(!g.consecutive(2, 0, 1));
    }
}
