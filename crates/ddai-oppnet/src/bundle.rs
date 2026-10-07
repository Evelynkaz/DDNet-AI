//! The model file: the network, what it was trained on, and a feature-layout version so a model never meets features it was not trained on.
//!
//! Model files are never committed (`.gitignore` and `tools/ci/no-weights.sh` refuse `*.oppnet` and `*.opp`); they live in `~/aiddnet/data/runs/E-028/`.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::blob::{read_blob, write_blob};
use crate::feature::{FD, HORIZON, IF_DIM, IF_SLOTS, INPUT_DIM, K_HIST, OUT_DIM, STRIDE};
use crate::frame::N_RAYS;
use crate::net::Mlp;

/// Bumped whenever [`crate::feature`] changes what an input or output number means.
pub const FEATURE_VERSION: u32 = 1;
pub const BUNDLE_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OppBundle {
    pub format_version: u32,
    pub feature_version: u32,
    /// `[K_HIST, STRIDE, FD, N_RAYS, IF_SLOTS, IF_DIM, HORIZON]` of the features the network was trained with.
    pub layout: [u32; 7],
    pub net: Mlp,
    pub seed: u64,
    pub epochs: u32,
    pub val_loss: f64,
    pub notes: String,
}

pub fn current_layout() -> [u32; 7] {
    [K_HIST, STRIDE, FD, N_RAYS, IF_SLOTS, IF_DIM, HORIZON].map(|v| v as u32)
}

impl OppBundle {
    pub fn new(net: Mlp, seed: u64, epochs: u32, val_loss: f64, notes: String) -> OppBundle {
        OppBundle {
            format_version: BUNDLE_VERSION,
            feature_version: FEATURE_VERSION,
            layout: current_layout(),
            net,
            seed,
            epochs,
            val_loss,
            notes,
        }
    }

    /// Checks the bundle against this build's feature layout.
    pub fn validate(&self) -> Result<(), String> {
        if self.format_version != BUNDLE_VERSION {
            return Err(format!(
                "opponent model: format version {} (this build reads {BUNDLE_VERSION})",
                self.format_version
            ));
        }
        if self.feature_version != FEATURE_VERSION || self.layout != current_layout() {
            return Err("opponent model: trained with another feature layout".into());
        }
        if self.net.n_in != INPUT_DIM || self.net.n_out != OUT_DIM || !self.net.is_consistent() {
            return Err(
                "opponent model: the network does not fit the feature layout (or holds a non-finite weight)".into(),
            );
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        write_blob(path, self, 3)
    }

    pub fn load(path: &Path) -> Result<OppBundle, String> {
        let b: OppBundle = read_blob(path)?;
        b.validate().map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bundle_round_trips_and_a_foreign_layout_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.oppnet");
        let b = OppBundle::new(Mlp::new(INPUT_DIM, 8, 4, OUT_DIM, 1), 1, 2, 0.5, "t".into());
        b.save(&p).unwrap();
        assert_eq!(OppBundle::load(&p).unwrap(), b);
        let mut bad = b.clone();
        bad.layout[0] += 1;
        assert!(bad.validate().is_err());
        let mut bad = b.clone();
        bad.net.n_in += 1;
        assert!(bad.validate().is_err());
        let mut bad = b;
        bad.net.params[3] = f32::NAN;
        assert!(bad.validate().is_err());
    }
}
