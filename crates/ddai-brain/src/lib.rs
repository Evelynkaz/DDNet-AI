//! `ddai-brain`: the shared decision-maker interface (task 7.3) every controller implements —
//! the planner (phase 8's teacher/eval baseline), a scripted bot, the fly (`ddai-fly`'s
//! `FlyBrain`), and later human-replay. Plain data in, plain data out ([`Observation`] /
//! [`Action`]), so none of them, nor the live bot/arena code that drives whichever one is
//! currently deciding, needs to know or care which implementation is behind the trait.
//!
//! No heavy dependencies: only [`ddai_physics`] (itself dependency-free), for the map tile
//! format and the wire-format `PlayerInput`/tuning types this crate's own types are built from or
//! convert to.

mod action;
mod brain;
mod idle;
mod mirror;
mod observation;

pub use action::{Action, IVec2};
pub use brain::{Brain, ResetContext, WorldView};
pub use idle::IdleBrain;
pub use mirror::mirror_map_data;
pub use observation::{
    CharacterObservation, HOOK_FLYING, HOOK_GRABBED, HOOK_IDLE, HOOK_RETRACT_END, HOOK_RETRACT_START, HOOK_RETRACTED,
    Observation, jumps_left,
};
