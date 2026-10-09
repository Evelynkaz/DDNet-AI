//! Task 3.24 (E-039): games of **real human players** from DDNet demos as training and evaluation data for the opponent-input
//! predictor.
//!
//! A demo records the other players' `Sv_PreInput` messages (their real input changes, forwarded by the server before their tick), so
//! unlike a live clip (see [`crate::clipdata`]) the opponent's input is known at **every** tick and for **every** head, including the
//! jump key and the aim: the labels are the true inputs, not what the next snapshots show. A [`HumanGame`] is a run of consecutive
//! snapshot frames (2 ticks apart, the pair `[us, opponent]`: either human can play either role) and the real inputs of both players
//! tick by tick -- where the demo has them: the server forwards pre-inputs to the other clients, never to their owner, so the player who
//! recorded the demo has none (`None`) and only its opponents can be modelled.
//!
//! **No nicknames**: a game carries a source tag (`h<demo number>-<label>-<label>p<part>`) built from anonymous per-demo labels.

use serde::{Deserialize, Serialize};

use crate::clipdata::ClipTick;
use crate::frame::InputRec;

/// A run of consecutive frames of one ordered pair of humans, with their real inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanGame {
    /// `h<demo number>-<us label>-<opponent label>p<part>`: anonymous.
    pub source: String,
    /// The demo number: the unit of the train / held-out split (a demo is never divided).
    pub session: u8,
    /// The frames, 2 ticks apart: `frames[0]` is us (the player whose in-flight inputs the model sees), `frames[1]` the opponent being
    /// predicted. `sent` and `opp_attack_tick` are filled as in a clip.
    pub ticks: Vec<ClipTick>,
    /// The world tick of `inputs[0]`.
    pub first: i32,
    /// `inputs[j]` = the real inputs `[us, opponent]` applied in the step into tick `first + j` (`None` = not known: no real input was in
    /// force for that player).
    pub inputs: Vec<[Option<InputRec>; 2]>,
}

impl HumanGame {
    /// The real input of `who` (0 = us, 1 = the opponent) applied in the step into world tick `tick`.
    pub fn input_at(&self, who: usize, tick: i32) -> Option<InputRec> {
        let j = usize::try_from(tick.checked_sub(self.first)?).ok()?;
        self.inputs.get(j)?.get(who).copied().flatten()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{N_RAYS, TeeFrame};

    fn game() -> HumanGame {
        let tick = |t: i32| ClipTick {
            tick: t,
            frames: [TeeFrame::default(); 2],
            rays: [[1.0; N_RAYS]; 2],
            sent: [None, None],
            opp_attack_tick: 0,
            opp_weapon: 0,
            duel: true,
        };
        HumanGame {
            source: "h0-1-2p1".into(),
            session: 0,
            ticks: vec![tick(100), tick(102), tick(104), tick(108)],
            first: 99,
            inputs: vec![
                [Some(InputRec::default()), None],
                [
                    Some(InputRec {
                        direction: 1,
                        ..InputRec::default()
                    }),
                    Some(InputRec::default()),
                ],
            ],
        }
    }

    #[test]
    fn inputs_are_found_by_world_tick_and_unknown_is_none() {
        let g = game();
        assert_eq!(g.input_at(0, 99), Some(InputRec::default()));
        assert_eq!(g.input_at(1, 99), None, "no real input in force");
        assert_eq!(g.input_at(0, 100).map(|i| i.direction), Some(1));
        assert_eq!(g.input_at(0, 98), None, "before the first tick");
        assert_eq!(g.input_at(0, 101), None, "past the last");
    }

    #[test]
    fn consecutive_frames_are_two_ticks_apart() {
        let g = game();
        assert!(g.consecutive(0, 0, 2));
        assert!(!g.consecutive(0, 0, 3), "104 -> 108 is a gap");
        assert!(!g.consecutive(2, 0, 1));
        assert!(!g.consecutive(0, 1, 0), "no frame before the first");
    }
}
