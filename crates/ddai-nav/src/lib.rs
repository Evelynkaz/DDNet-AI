//! `ddai-nav` (task 4.2): navigation for the live bot — see `docs/research/nav.md` for the
//! TS-function-to-Rust mapping and `crates/ddai-nav/README.md` for the overview.
//!
//! Everything here is generic over [`ddai_planner::plan_world::PlanCollision`] /
//! [`ddai_planner::plan_world::PlanWorld`], like the planner (3.2): on `ddai-tsworld` (f64, feature
//! `ts-parity`) it is proven decision-for-decision against the real TS sources; on
//! `ddai_physics::World<f32>` it plays live.

pub mod crossbench;
pub mod crossing;
pub mod follow;
pub mod grid;
pub mod harness;
pub mod home;
pub mod memory;
pub mod navigator;
pub mod route;
pub mod runner;
pub mod trek;
pub mod wayblock;

pub use grid::NavGrid;
pub use route::{MoveKind, RouteOpts, RouteResult, RouteStep, Router, dead_zone, route_move_key, spawn_tiles};
pub use runner::{RouteRunner, RunnerState};

/// A tile is 32 px.
pub const TILE_PX: i32 = 32;
