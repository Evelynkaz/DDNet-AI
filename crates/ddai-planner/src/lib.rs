//! Task 3.2: Rust port of the legacy TS block planner (`src/plan/*.ts`, `src/env/{scripted,action,
//! obs}.ts`) — decision-for-decision identical to TS on the [`ts_adapter`] backend
//! (`ddai-tsworld::SimWorld`, feature `ts-parity`), and the precise-search half of the live brain
//! (D-041) on the [`physics_adapter`] backend (`ddai_physics::World<f32>`).
//!
//! See `docs/research/orig-plan.md` (the phase-0 research this crate is ported from) and
//! `docs/DECISIONS.md` D-017/D-018/D-021/D-035/D-041 for the design rationale.

pub mod action;
pub mod brains;
pub mod clock;
pub mod config;
pub mod elite;
pub mod fields;
pub mod hybrid;
pub mod memory;
pub mod opponent_profile;
pub mod physics_adapter;
pub mod plan_world;
pub mod planner;
pub mod prof;
pub mod scripted;
pub mod seal;
pub mod shield;
pub mod teacher;
pub mod throw_lines;
#[cfg(feature = "ts-parity")]
pub mod ts_adapter;
pub mod tuning;
pub mod types;
pub mod vmath;

pub use plan_world::{PlanCollision, PlanWorld};
pub use planner::{Decision, DecisionInfo, PlanStep, Planner};
pub use types::PlayerInput;
