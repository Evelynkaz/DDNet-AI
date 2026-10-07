//! `ddai-env`: the offline evaluation arena (task 8.1) -- the measuring stick for the D-041
//! search, fly training and regressions.
//!
//! * [`arena`]: maps + spawn rules as TOML data, train/holdout tags;
//! * [`game`]: one N-player game with the phase-0 harness's rules on the bit-exact
//!   `ddai_physics::World<f32>`, every player any `ddai_brain::Brain`;
//! * [`run`]: rayon batches, reproducible at any thread count;
//! * [`report`]: W:L:D:T with Wilson 95% intervals, the run record, the Russian markdown table;
//! * [`scenario`]: technique scenarios (T1-T18) with fixed start states and success predicates;
//! * [`output`]: JSONL / summary files for a whole run config;
//! * [`oppdata`]: a game recorded as a dataset record of the opponent-input predictor (task 3.15).

pub mod arena;
pub mod brains;
pub mod config;
pub mod game;
pub mod models;
pub mod observe;
pub mod oppdata;
pub mod output;
pub mod report;
pub mod run;
pub mod scenario;
pub mod sim;
pub mod stats;

use std::fmt;

/// Every fallible operation in this crate reports a human-readable message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvError(String);

impl EnvError {
    pub fn new(msg: impl Into<String>) -> Self {
        EnvError(msg.into())
    }
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EnvError {}
