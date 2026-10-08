//! The v2 arena dataset: like [`crate::data`], with the geometry rays around **both** tees.

use serde::{Deserialize, Serialize};

use crate::frame::{InputRec, N_RAYS, TeeFrame};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickRec {
    pub frames: [TeeFrame; 2],
    pub applied: [InputRec; 2],
    /// Around slot 0 (us) and slot 1 (the opponent).
    pub rays: [[f32; N_RAYS]; 2],
}

/// One arena game. Slot 0 is **us**, slot 1 the opponent being modelled; `ticks[j]` holds the world at tick `tick0 + j` and the inputs applied in the step that led to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameRec {
    pub arena: String,
    pub seed: u64,
    /// Input lag in ticks: `[us, opponent]`.
    pub lag: [u8; 2],
    pub swap: bool,
    pub decide_every: u8,
    pub tick0: i32,
    pub ticks: Vec<TickRec>,
}
