//! Loads task 7.3's `configs/fly/{S,M}-brain.toml` — the ray-grid/decoder/world-model/
//! proprioception configuration layered on top of a compiled `.flyg`'s own `input_channels`
//! (task 6.3). See `configs/fly/S-brain.toml`'s header comment for why this file exists
//! separately from `configs/fly/{S,M}.toml`.

use std::path::Path;

use serde::Deserialize;

use crate::decoder::DecoderConfig;
use crate::encoder::{OpponentStateConfig, ProprioceptionConfig, RayGridConfig};
use crate::world_model::WorldModelConfig;

#[derive(Debug, Deserialize)]
struct BrainConfigFile {
    ray_grid: RayGridConfig,
    decoder: DecoderConfig,
    world_model: WorldModelConfig,
    proprioception: ProprioceptionConfig,
    /// Task 8.5a; absent in every config written before it.
    #[serde(default)]
    opponent_state: OpponentStateConfig,
}

/// Everything a `configs/fly/{S,M}-brain.toml` file carries, already parsed.
#[derive(Debug)]
pub struct BrainConfig {
    pub ray_grid: RayGridConfig,
    pub decoder: DecoderConfig,
    pub world_model: WorldModelConfig,
    pub proprioception: ProprioceptionConfig,
    pub opponent_state: OpponentStateConfig,
}

impl BrainConfig {
    /// The encoder this config describes over `model` (the ray grid, the proprioception and the opponent-state channels).
    pub fn encoder_model(
        &self,
        model: &crate::model::FlyModel,
    ) -> Result<crate::encoder::EncoderModel, crate::encoder::EncoderError> {
        crate::encoder::EncoderModel::with_opponent_state(
            model,
            self.ray_grid,
            &self.proprioception,
            &self.opponent_state,
        )
    }
}

#[derive(Debug)]
pub enum BrainConfigError {
    Io(std::io::Error),
    Toml(toml::de::Error),
}

impl std::fmt::Display for BrainConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrainConfigError::Io(e) => write!(f, "I/O error reading brain config: {e}"),
            BrainConfigError::Toml(e) => write!(f, "failed to parse brain config TOML: {e}"),
        }
    }
}

impl std::error::Error for BrainConfigError {}

pub fn load_brain_config(path: &Path) -> Result<BrainConfig, BrainConfigError> {
    let text = std::fs::read_to_string(path).map_err(BrainConfigError::Io)?;
    parse_brain_config(&text)
}

/// [`load_brain_config`] on already-read text (a training bundle embeds the config it was trained
/// with, so a checkpoint stays loadable when `configs/fly/*.toml` later changes).
pub fn parse_brain_config(text: &str) -> Result<BrainConfig, BrainConfigError> {
    let file: BrainConfigFile = toml::from_str(text).map_err(BrainConfigError::Toml)?;
    Ok(BrainConfig {
        ray_grid: file.ray_grid,
        decoder: file.decoder,
        world_model: file.world_model,
        proprioception: file.proprioception,
        opponent_state: file.opponent_state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_the_real_s_brain_config() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/fly/S-brain.toml");
        let cfg = load_brain_config(&path).expect("configs/fly/S-brain.toml should parse");
        assert_eq!(cfg.ray_grid.num_directions, 48);
        assert_eq!(cfg.decoder.direction_actions[0], "direction_left");
        assert_eq!(cfg.proprioception.grounded.len(), 3);
    }

    #[test]
    fn loads_the_real_m_brain_config() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/fly/M-brain.toml");
        let cfg = load_brain_config(&path).expect("configs/fly/M-brain.toml should parse");
        assert_eq!(cfg.proprioception.speed.len(), 7);
    }

    #[test]
    fn missing_file_is_a_clean_error_not_a_panic() {
        let result = load_brain_config(std::path::Path::new("/nonexistent/brain.toml"));
        assert!(matches!(result, Err(BrainConfigError::Io(_))));
    }

    #[test]
    fn malformed_toml_is_a_clean_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.toml");
        std::fs::write(&path, "this is not valid toml {{{").unwrap();
        let result = load_brain_config(&path);
        assert!(matches!(result, Err(BrainConfigError::Toml(_))));
    }
}
