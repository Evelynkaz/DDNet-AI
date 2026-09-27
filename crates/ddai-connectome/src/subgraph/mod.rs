//! `build-subgraph`/`flyg-info`: selects the fly's connectome subgraph (task 6.3) and compiles it
//! into a `.flyg` file. See this crate's README for the selection algorithm in FLY.md-style prose
//! and `docs/formats.md` for the `.flyg` on-disk format.

pub mod an_pick;
pub mod build;
pub mod common;
pub mod config;
pub mod report;
pub mod rf;
pub mod select;
pub mod signs;

pub use build::{BuildSubgraphSummary, run_build_subgraph};
