//! `ddai-fly`: the fly's inference engine (phase 7.1). Loads a compiled `.flyg` connectome
//! subgraph (`ddai-flyg`) plus trainable parameters and runs the continuous-time rate model
//! (`docs/FLY.md` §4, in the main repo) forward in real time — one game decision (4 exponential-
//! Euler substeps by default) in well under the 25Hz budget on one core. See `README.md` for the
//! model, the API's shape, and the performance/init-tuning numbers this crate's own tests were
//! checked against.
//!
//! Task 7.3 adds the ray-grid input encoder ([`encoder`]), the DN action decoder ([`decoder`]),
//! the world-model head ([`world_model`]), and [`brain::FlyBrain`] (the `ddai_brain::Brain`
//! implementation that glues all three to 7.1/7.2's forward/backward). Training (phase 8) is
//! still not here.

pub mod activation;
pub mod backward;
mod bench;
pub mod brain_checkpoint;
pub mod brain_config;
pub mod brain_train;
pub mod checkpoint;
pub mod config;
pub mod decoder;
#[doc(hidden)] // demo/reporting glue shared by tests and the `ddnet-ai fly train-demo` CLI, not covered by semver.
pub mod demo;
pub mod demo_brain;
pub mod encoder;
mod error;
pub mod flat_adam;
mod kernel;
pub mod model;
pub mod optim;
pub mod params;
pub mod proposer;
pub mod recorder;
#[doc(hidden)] // test/bench support only (deterministic PRNG for reproducible synthetic input traffic).
pub mod rng;
pub mod state;
pub mod train;
pub mod world_model;

pub mod brain;

#[doc(hidden)] // test/bench support only, not covered by semver — see its own doc comment.
pub mod brain_fixtures;
#[doc(hidden)] // test/bench support only, not covered by semver — see its own doc comment.
pub mod test_fixtures;

pub use backward::{BackwardIndex, BpttGradients, BpttScratch, ExtraRateGrad, backward};
pub use bench::{DutyCycleReport, refresh_inputs_partial, run_duty_cycle};
pub use checkpoint::{
    Checkpoint, CheckpointMeta, load_checkpoint, load_checkpoint_for_flyg, save_checkpoint, sha256_hex_of_file,
};
pub use config::FlyConfig;
pub use decoder::calibrate_from_rest;
pub use error::FlyError;
pub use model::FlyModel;
pub use optim::{
    ActivityRegularizerConfig, AdamConfig, AdamState, GuardedAdamConfig, GuardedAdamState, GuardedStepOutcome,
    ParamGradients, activity_regularizer_rate_grad, adam_step, add_l2_pull_to_a, clip_grad_norm, guarded_adam_step,
};
pub use params::FlyParams;
pub use recorder::TrajectoryRecorder;
pub use state::{DecisionOutput, FlyState, WarmUpReport};
pub use train::{
    BatchGradients, MemoryCapExceeded, Sequence, batch_memory_bytes, concurrent_batch_memory_bytes,
    sequence_memory_bytes, train_step,
};
