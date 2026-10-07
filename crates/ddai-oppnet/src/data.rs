//! The dataset: one [`GameRec`] per arena game, the per-tick states of both tees and the inputs applied.
//!
//! Slot 0 is **us** (the hybrid), slot 1 the **opponent** being modelled (`live-v2`). `ticks[j]` holds the state of the world at tick
//! `tick0 + j` and the inputs applied in the step that led to it (the step of tick `tick0 + j - 1`). The rays are those around slot 1.

use serde::{Deserialize, Serialize};

use crate::frame::{InputRec, N_RAYS, TeeFrame};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickRec {
    pub frames: [TeeFrame; 2],
    pub applied: [InputRec; 2],
    pub rays: [f32; N_RAYS],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameRec {
    pub arena: String,
    pub seed: u64,
    /// Input lag in ticks: `[us, opponent]`.
    pub lag: [u8; 2],
    pub swap: bool,
    /// World ticks between decisions (the arena's `decide_every`).
    pub decide_every: u8,
    /// World tick of `ticks[0]`.
    pub tick0: i32,
    pub ticks: Vec<TickRec>,
}

impl GameRec {
    /// Index into `ticks` of world tick `t`.
    pub fn index_of(&self, t: i32) -> Option<usize> {
        usize::try_from(t - self.tick0).ok().filter(|&i| i < self.ticks.len())
    }
}

/// A dataset file's content.
pub type Chunk = Vec<GameRec>;
