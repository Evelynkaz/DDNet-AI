//! `ddai-dataset` (task 8.4c): human block play from DDNet client demos as training data for the
//! fly - `(Observation, Action, meta)` samples with reconstructed inputs whose quality is
//! *measured* (physics replay), technique tags (T1-T18, D-048), skill signals (D-030), a
//! chunked postcard + zstd dataset format and a filtering reader. See `README.md` and
//! `docs/formats.md` section 20.
//!
//! Data flow: [`ingest`] (demo -> anonymised recorder frames) -> [`pipeline`] (world state, input
//! reconstruction, replay check) -> [`analysis`] / [`skill`] / [`technique`] -> [`demo`] (one demo)
//! -> [`run`] (all demos, deterministic) -> [`dataset`] (write / read) and [`report`].
//!
//! Nicknames never leave [`ingest`]: everything downstream sees only anonymous per-demo labels.

pub mod analysis;
pub mod config;
pub mod dataset;
pub mod demo;
pub mod ingest;
pub mod pipeline;
pub mod privacy;
pub mod replay;
pub mod report;
pub mod run;
pub mod skill;
pub mod store;
#[cfg(feature = "test-util")]
pub mod synth;
pub mod tags;
pub mod technique;
pub mod testutil;
pub mod types;
