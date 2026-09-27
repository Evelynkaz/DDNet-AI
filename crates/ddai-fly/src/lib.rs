//! `ddai-fly`: the fly's inference engine (phase 7.1). Loads a compiled `.flyg` connectome
//! subgraph (`ddai-flyg`) plus trainable parameters and runs the continuous-time rate model
//! (`docs/FLY.md` §4, in the main repo) forward in real time — one game decision (4 exponential-
//! Euler substeps by default) in well under the 25Hz budget on one core. See `README.md` for the
//! model, the API's shape, and the performance/init-tuning numbers this crate's own tests were
//! checked against.
//!
//! Explicitly **not** here (see `README.md`'s "Scope"): the backward pass (7.2), the ray-grid
//! input encoder / DN action decoder (7.3), and training (phase 8). What *is* here for those:
//! [`TrajectoryRecorder`] (a hook for 7.2) and [`FlyModel::output_slot_for_neuron`] /
//! `flyg().output_groups` (hooks for 7.3).

pub mod activation;
pub mod backward;
mod bench;
pub mod checkpoint;
pub mod config;
#[doc(hidden)] // demo/reporting glue shared by tests and the `ddnet-ai fly train-demo` CLI, not covered by semver.
pub mod demo;
mod error;
mod kernel;
pub mod model;
pub mod optim;
pub mod params;
pub mod recorder;
#[doc(hidden)] // test/bench support only (deterministic PRNG for reproducible synthetic input traffic).
pub mod rng;
pub mod state;
pub mod train;

#[doc(hidden)] // test/bench support only, not covered by semver — see its own doc comment.
pub mod test_fixtures;

pub use backward::{BackwardIndex, BpttGradients, BpttScratch, ExtraRateGrad, backward};
pub use bench::{DutyCycleReport, refresh_inputs_partial, run_duty_cycle};
pub use checkpoint::{
    Checkpoint, CheckpointMeta, load_checkpoint, load_checkpoint_for_flyg, save_checkpoint, sha256_hex_of_file,
};
pub use config::FlyConfig;
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
