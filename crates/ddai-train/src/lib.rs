//! `ddai-train` (task 8.2): behaviour cloning and DAgger for the fly and its controls.
//!
//! * [`types`], [`store`]: the teacher dataset (chunked postcard + zstd + manifest);
//! * (more modules are added as the crate grows; see `README.md`).

// The per-iteration metrics line of `ppo` is one big `json!`.
#![recursion_limit = "256"]

pub mod bank;
pub mod bank_collect;
pub mod collect;
pub mod es;
pub mod experiment;
pub mod heldblock;
pub mod hook_study;
pub mod human;
pub mod learner;
pub mod metrics;
pub mod play_stats;
pub mod ppo;
pub mod runner;
pub mod seq;
pub mod store;
pub mod teacher_data;
pub mod trainer;
pub mod types;

pub use ddai_env::EnvError;
