//! On-disk checkpoint for task 7.3's own learnable state: [`crate::encoder::EncoderParams`],
//! [`crate::decoder::DecoderParams`] plus its frozen [`crate::decoder::DnCalibration`], and
//! [`crate::world_model::WorldModelParams`]. A **separate** file/format from
//! [`crate::checkpoint::Checkpoint`] (task 7.2's `FlyParams`-only checkpoint, format version
//! [`crate::checkpoint::FLY_CHECKPOINT_FORMAT_VERSION`]) — deliberately not folded into that
//! struct, per the task constraint "must not change 7.1/7.2 behaviour": every existing
//! `Checkpoint` reader/writer, and its own format-version/bit-flip tests, stay untouched.
//! Otherwise this mirrors that module's on-disk discipline exactly: postcard + zstd (content
//! checksum enabled) + an explicit sha256 of the postcard bytes, atomic write (temp file +
//! rename), and a checkpoint-format-version header checked before a full decode is attempted.
//!
//! FLY.md §6's "frozen per-DN calibration ... stored in the checkpoint" (acceptance criterion 4)
//! is what this module is for: [`BrainCheckpoint::calibration`] travels with the decoder
//! parameters it was fit alongside, so loading one checkpoint restores a fully consistent,
//! ready-to-decide `FlyBrain`.

use std::io::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::decoder::{DecoderModel, DecoderParams, DnCalibration};
use crate::encoder::{EncoderModel, EncoderParams};
use crate::world_model::{WorldModelHead, WorldModelParams};

pub const BRAIN_CHECKPOINT_FORMAT_VERSION: u32 = 1;

const PAYLOAD_HASH_LEN: usize = 32;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BrainCheckpointMeta {
    pub seed: u64,
    pub git_commit: Option<String>,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainCheckpoint {
    pub format_version: u32,
    /// sha256 (hex) of the `.flyg` file this checkpoint's params were built for — same convention
    /// as `crate::checkpoint::Checkpoint::flyg_sha256`.
    pub flyg_sha256: String,
    /// sha256 (hex) of the `configs/fly/{S,M}-brain.toml` file this checkpoint's params were
    /// built against (review round 1, F11, CONFIRMED): `flyg_sha256` alone pins the connectome
    /// and its `.flyg`-carried `input_channels`/`output_groups` tables, but every trainable
    /// param's own *shape* also depends on this crate's brain config (`RayGridConfig`'s
    /// `num_directions`/`num_distance_bins` sizing the encoder's spatial channels;
    /// `DecoderConfig`'s action names resolving to a specific set of `output_groups` members;
    /// `WorldModelConfig`'s `neuron_subset`) — editing that file (even without touching the
    /// `.flyg`) can silently change which index means what in an old checkpoint's flat
    /// `Vec<f32>`s. `load_brain_checkpoint_for_flyg` checks this the same way it checks
    /// `flyg_sha256`; `validate_brain_checkpoint_shapes` is the *shape*-level backstop for a
    /// change this hash's exact-file-match wouldn't catch on its own (e.g. two config files that
    /// happen to differ only in a comment).
    pub brain_config_sha256: String,
    pub encoder_params: EncoderParams,
    pub decoder_params: DecoderParams,
    pub calibration: DnCalibration,
    pub world_model_params: WorldModelParams,
    pub meta: BrainCheckpointMeta,
}

#[derive(Debug)]
pub enum BrainCheckpointError {
    Io(std::io::Error),
    Zstd(std::io::Error),
    Decode(postcard::Error),
    Encode(postcard::Error),
    FormatVersionMismatch {
        found: u32,
        expected: u32,
    },
    FlygMismatch {
        expected: String,
        found: String,
    },
    /// Review round 1, F11: same check as `FlygMismatch`, for `brain_config_sha256`.
    BrainConfigMismatch {
        expected: String,
        found: String,
    },
    Truncated {
        len: usize,
        min_len: usize,
    },
    ChecksumMismatch,
    /// Review round 1, F11: `validate_brain_checkpoint_shapes` found a shape mismatch or
    /// non-finite value in one of the checkpoint's own param groups.
    InvalidEncoderParams(String),
    InvalidDecoderParams(String),
    InvalidCalibration(String),
    InvalidWorldModelParams(String),
}

impl std::fmt::Display for BrainCheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrainCheckpointError::Io(e) => write!(f, "I/O error: {e}"),
            BrainCheckpointError::Zstd(e) => write!(f, "zstd (de)compression failed: {e}"),
            BrainCheckpointError::Decode(e) => write!(f, "failed to decode brain checkpoint (postcard): {e}"),
            BrainCheckpointError::Encode(e) => write!(f, "failed to encode brain checkpoint (postcard): {e}"),
            BrainCheckpointError::FormatVersionMismatch { found, expected } => {
                write!(f, "brain checkpoint format version {found} != expected {expected}")
            }
            BrainCheckpointError::FlygMismatch { expected, found } => write!(
                f,
                "brain checkpoint was trained against a different .flyg: expected sha256 {expected}, this graph hashes to {found}"
            ),
            BrainCheckpointError::BrainConfigMismatch { expected, found } => write!(
                f,
                "brain checkpoint was trained against a different brain config (configs/fly/{{S,M}}-brain.toml): expected sha256 {expected}, current file hashes to {found}"
            ),
            BrainCheckpointError::Truncated { len, min_len } => {
                write!(
                    f,
                    "brain checkpoint file is truncated: {len} bytes, need at least {min_len}"
                )
            }
            BrainCheckpointError::ChecksumMismatch => {
                write!(f, "brain checkpoint payload hash does not match — file is corrupted")
            }
            BrainCheckpointError::InvalidEncoderParams(m) => write!(f, "invalid encoder_params: {m}"),
            BrainCheckpointError::InvalidDecoderParams(m) => write!(f, "invalid decoder_params: {m}"),
            BrainCheckpointError::InvalidCalibration(m) => write!(f, "invalid calibration: {m}"),
            BrainCheckpointError::InvalidWorldModelParams(m) => write!(f, "invalid world_model_params: {m}"),
        }
    }
}

impl std::error::Error for BrainCheckpointError {}

#[derive(Debug, Deserialize)]
struct CheckpointHeader {
    format_version: u32,
}

fn sha256_raw(bytes: &[u8]) -> [u8; PAYLOAD_HASH_LEN] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = [0u8; PAYLOAD_HASH_LEN];
    out.copy_from_slice(&digest);
    out
}

fn sha256_hex_of_bytes(bytes: &[u8]) -> String {
    sha256_raw(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_hex_of_file(path: &Path) -> Result<String, BrainCheckpointError> {
    let bytes = std::fs::read(path).map_err(BrainCheckpointError::Io)?;
    Ok(sha256_hex_of_bytes(&bytes))
}

fn unique_sibling_tmp_path(path: &Path) -> std::path::PathBuf {
    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
    path.with_file_name(format!("{file_name}.{}.tmp", std::process::id()))
}

fn write_checkpoint(path: &Path, checkpoint: &BrainCheckpoint) -> Result<(), BrainCheckpointError> {
    let bytes = postcard::to_allocvec(checkpoint).map_err(BrainCheckpointError::Encode)?;
    let payload_hash = sha256_raw(&bytes);

    let mut compressed = Vec::new();
    {
        let mut encoder = zstd::stream::Encoder::new(&mut compressed, 3).map_err(BrainCheckpointError::Zstd)?;
        encoder.include_checksum(true).map_err(BrainCheckpointError::Zstd)?;
        encoder.write_all(&bytes).map_err(BrainCheckpointError::Zstd)?;
        encoder.finish().map_err(BrainCheckpointError::Zstd)?;
    }

    let mut out = Vec::with_capacity(PAYLOAD_HASH_LEN + compressed.len());
    out.extend_from_slice(&payload_hash);
    out.extend_from_slice(&compressed);

    let tmp_path = unique_sibling_tmp_path(path);
    std::fs::write(&tmp_path, &out).map_err(BrainCheckpointError::Io)?;
    std::fs::rename(&tmp_path, path).map_err(BrainCheckpointError::Io)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn save_brain_checkpoint(
    path: &Path,
    flyg_path: &Path,
    brain_config_path: &Path,
    encoder_params: &EncoderParams,
    decoder_params: &DecoderParams,
    calibration: &DnCalibration,
    world_model_params: &WorldModelParams,
    meta: BrainCheckpointMeta,
) -> Result<(), BrainCheckpointError> {
    let flyg_sha256 = sha256_hex_of_file(flyg_path)?;
    let brain_config_sha256 = sha256_hex_of_file(brain_config_path)?;
    let checkpoint = BrainCheckpoint {
        format_version: BRAIN_CHECKPOINT_FORMAT_VERSION,
        flyg_sha256,
        brain_config_sha256,
        encoder_params: encoder_params.clone(),
        decoder_params: decoder_params.clone(),
        calibration: calibration.clone(),
        world_model_params: world_model_params.clone(),
        meta,
    };
    write_checkpoint(path, &checkpoint)
}

pub fn load_brain_checkpoint(path: &Path) -> Result<BrainCheckpoint, BrainCheckpointError> {
    let all_bytes = std::fs::read(path).map_err(BrainCheckpointError::Io)?;
    if all_bytes.len() < PAYLOAD_HASH_LEN {
        return Err(BrainCheckpointError::Truncated {
            len: all_bytes.len(),
            min_len: PAYLOAD_HASH_LEN,
        });
    }
    let (stored_hash, compressed) = all_bytes.split_at(PAYLOAD_HASH_LEN);
    let bytes = zstd::stream::decode_all(compressed).map_err(BrainCheckpointError::Zstd)?;

    let actual_hash = sha256_raw(&bytes);
    if actual_hash.as_slice() != stored_hash {
        return Err(BrainCheckpointError::ChecksumMismatch);
    }

    let (header, _): (CheckpointHeader, _) = postcard::take_from_bytes(&bytes).map_err(BrainCheckpointError::Decode)?;
    if header.format_version != BRAIN_CHECKPOINT_FORMAT_VERSION {
        return Err(BrainCheckpointError::FormatVersionMismatch {
            found: header.format_version,
            expected: BRAIN_CHECKPOINT_FORMAT_VERSION,
        });
    }

    let checkpoint: BrainCheckpoint = postcard::from_bytes(&bytes).map_err(BrainCheckpointError::Decode)?;
    Ok(checkpoint)
}

/// Loads a checkpoint and checks it was trained against exactly this `.flyg` *and* exactly this
/// brain config file (review round 1, F11) — but does **not** validate param shapes/finiteness;
/// see [`validate_brain_checkpoint_shapes`] for that (it needs the already-built
/// `EncoderModel`/`DecoderModel`/`WorldModelHead` this function doesn't have).
pub fn load_brain_checkpoint_for_flyg(
    path: &Path,
    flyg_path: &Path,
    brain_config_path: &Path,
) -> Result<BrainCheckpoint, BrainCheckpointError> {
    let checkpoint = load_brain_checkpoint(path)?;
    let actual_flyg = sha256_hex_of_file(flyg_path)?;
    if actual_flyg != checkpoint.flyg_sha256 {
        return Err(BrainCheckpointError::FlygMismatch {
            expected: checkpoint.flyg_sha256.clone(),
            found: actual_flyg,
        });
    }
    let actual_config = sha256_hex_of_file(brain_config_path)?;
    if actual_config != checkpoint.brain_config_sha256 {
        return Err(BrainCheckpointError::BrainConfigMismatch {
            expected: checkpoint.brain_config_sha256.clone(),
            found: actual_config,
        });
    }
    Ok(checkpoint)
}

/// Shape/finiteness validation (review round 1, F11, CONFIRMED — "`validate_shape` never
/// called"): every param group must match the shape `encoder`/`decoder`/`world_model` (already
/// built from the *current* `.flyg` + brain config) expect, and every value must be finite. Call
/// this after [`load_brain_checkpoint_for_flyg`]'s hash checks, before trusting a loaded
/// checkpoint's params enough to build a [`crate::brain::FlyBrain`] from them — the hash checks
/// alone only catch a *different* file; a bit flip that happens to preserve length, or two config
/// files that hash differently but happen to resolve to the same shapes, would pass them but not
/// this.
pub fn validate_brain_checkpoint_shapes(
    checkpoint: &BrainCheckpoint,
    encoder: &EncoderModel,
    decoder: &DecoderModel,
    world_model: &WorldModelHead,
) -> Result<(), BrainCheckpointError> {
    checkpoint
        .encoder_params
        .validate_shape(encoder.num_params(), encoder.ray_grid_config().num_distance_bins)
        .map_err(|e| BrainCheckpointError::InvalidEncoderParams(e.to_string()))?;
    decoder
        .validate_params_shape(&checkpoint.decoder_params)
        .map_err(|e| BrainCheckpointError::InvalidDecoderParams(e.to_string()))?;
    checkpoint
        .calibration
        .validate_shape(decoder.num_outputs())
        .map_err(|e| BrainCheckpointError::InvalidCalibration(e.to_string()))?;
    world_model
        .validate_params_shape(&checkpoint.world_model_params)
        .map_err(|e| BrainCheckpointError::InvalidWorldModelParams(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{DecoderConfig, DecoderModel};
    use crate::encoder::{EncoderModel, ProprioceptionConfig, RayGridConfig};
    use crate::world_model::{WorldModelConfig, WorldModelHead};

    fn tiny_flyg_with_everything() -> ddai_flyg::Flyg {
        use crate::brain_fixtures::{FxInputChannel, FxNeuron, FxOutputGroup, FxType, build_brain_flyg};
        use ddai_flyg::{NeuronRole, Side, Sign};

        let types = vec![
            FxType {
                name: "VPN_T",
                sign: Sign::Excitatory,
            },
            FxType {
                name: "HID_T",
                sign: Sign::Excitatory,
            },
            FxType {
                name: "DIR_LR",
                sign: Sign::Excitatory,
            },
            FxType {
                name: "DIR_S",
                sign: Sign::Excitatory,
            },
            FxType {
                name: "JMP",
                sign: Sign::Excitatory,
            },
            FxType {
                name: "HK",
                sign: Sign::Excitatory,
            },
            FxType {
                name: "FR",
                sign: Sign::Excitatory,
            },
            FxType {
                name: "AIM",
                sign: Sign::Excitatory,
            },
        ];
        let neurons = vec![
            FxNeuron {
                type_index: 0,
                role: NeuronRole::InputVisual,
                side: Side::M,
                full_connectome_in: 1000,
                rf: (0.0, 0.0),
            },
            FxNeuron {
                type_index: 1,
                role: NeuronRole::Hidden,
                side: Side::M,
                full_connectome_in: 1000,
                rf: (0.0, 0.0),
            },
            // DIR_LR: one tied L/R pair on the same type (review round 1, F6 -- matches how the
            // real .flyg's own output_groups assign each side's DN to its own action name).
            FxNeuron {
                type_index: 2,
                role: NeuronRole::Output,
                side: Side::L,
                full_connectome_in: 1000,
                rf: (0.0, 0.0),
            },
            FxNeuron {
                type_index: 2,
                role: NeuronRole::Output,
                side: Side::R,
                full_connectome_in: 1000,
                rf: (0.0, 0.0),
            },
            FxNeuron {
                type_index: 3,
                role: NeuronRole::Output,
                side: Side::M,
                full_connectome_in: 1000,
                rf: (0.0, 0.0),
            },
            FxNeuron {
                type_index: 4,
                role: NeuronRole::Output,
                side: Side::M,
                full_connectome_in: 1000,
                rf: (0.0, 0.0),
            },
            FxNeuron {
                type_index: 5,
                role: NeuronRole::Output,
                side: Side::M,
                full_connectome_in: 1000,
                rf: (0.0, 0.0),
            },
            FxNeuron {
                type_index: 6,
                role: NeuronRole::Output,
                side: Side::M,
                full_connectome_in: 1000,
                rf: (0.0, 0.0),
            },
            FxNeuron {
                type_index: 7,
                role: NeuronRole::Output,
                side: Side::M,
                full_connectome_in: 1000,
                rf: (0.0, 0.0),
            },
        ];
        build_brain_flyg(
            &types,
            &neurons,
            &[],
            &[FxInputChannel {
                type_name: "VPN_T",
                channels: vec!["walls"],
            }],
            &[
                FxOutputGroup {
                    action: "direction_left",
                    member_type_names: vec!["DIR_LR"],
                    side_filter: Some(Side::L),
                },
                FxOutputGroup {
                    action: "direction_right",
                    member_type_names: vec!["DIR_LR"],
                    side_filter: Some(Side::R),
                },
                FxOutputGroup {
                    action: "direction_stop",
                    member_type_names: vec!["DIR_S"],
                    side_filter: None,
                },
                FxOutputGroup {
                    action: "jump",
                    member_type_names: vec!["JMP"],
                    side_filter: None,
                },
                FxOutputGroup {
                    action: "hook",
                    member_type_names: vec!["HK"],
                    side_filter: None,
                },
                FxOutputGroup {
                    action: "fire",
                    member_type_names: vec!["FR"],
                    side_filter: None,
                },
                FxOutputGroup {
                    action: "aim",
                    member_type_names: vec!["AIM"],
                    side_filter: None,
                },
            ],
        )
    }

    #[test]
    fn round_trip_preserves_everything() {
        let flyg = tiny_flyg_with_everything();
        let config = crate::config::FlyConfig::default();
        let params = crate::params::FlyParams::init_default(&flyg, &config, 1);
        let model = crate::model::FlyModel::new(flyg, config, params).unwrap();

        let encoder = EncoderModel::new(&model, RayGridConfig::default(), &ProprioceptionConfig::default()).unwrap();
        let encoder_params = EncoderParams::init_default(encoder.num_params());
        let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
        let decoder_params = decoder.init_default_params();
        let calib = DnCalibration {
            mu: vec![0.1; model.num_outputs()],
            sigma: vec![0.9; model.num_outputs()],
        };
        let wm_head = WorldModelHead::new(&model, &WorldModelConfig::default()).unwrap();
        let wm_params = wm_head.init_default_params();

        let dir = tempfile::tempdir().unwrap();
        let flyg_path = dir.path().join("s.flyg");
        ddai_flyg::save(model.flyg(), &flyg_path).unwrap();
        let brain_config_path = dir.path().join("s-brain.toml");
        std::fs::write(&brain_config_path, b"# a brain config, for its sha256 only").unwrap();
        let ckpt_path = dir.path().join("brain.ckpt");
        let meta = BrainCheckpointMeta {
            seed: 7,
            git_commit: Some("deadbeef".to_string()),
            notes: "round trip".to_string(),
        };
        save_brain_checkpoint(
            &ckpt_path,
            &flyg_path,
            &brain_config_path,
            &encoder_params,
            &decoder_params,
            &calib,
            &wm_params,
            meta.clone(),
        )
        .unwrap();

        let loaded = load_brain_checkpoint_for_flyg(&ckpt_path, &flyg_path, &brain_config_path).unwrap();
        assert_eq!(loaded.encoder_params, encoder_params);
        assert_eq!(loaded.decoder_params, decoder_params);
        assert_eq!(loaded.calibration, calib);
        assert_eq!(loaded.world_model_params, wm_params);
        assert_eq!(loaded.meta, meta);
        assert_eq!(
            loaded.brain_config_sha256,
            sha256_hex_of_file(&brain_config_path).unwrap()
        );

        validate_brain_checkpoint_shapes(&loaded, &encoder, &decoder, &wm_head)
            .expect("a freshly round-tripped checkpoint must validate against the model it came from");
    }

    #[test]
    fn loading_against_a_different_flyg_fails_clearly() {
        let flyg = tiny_flyg_with_everything();
        let dir = tempfile::tempdir().unwrap();
        let flyg_path = dir.path().join("a.flyg");
        ddai_flyg::save(&flyg, &flyg_path).unwrap();
        let brain_config_path = dir.path().join("s-brain.toml");
        std::fs::write(&brain_config_path, b"# a brain config, for its sha256 only").unwrap();

        let config = crate::config::FlyConfig::default();
        let params = crate::params::FlyParams::init_default(&flyg, &config, 1);
        let model = crate::model::FlyModel::new(flyg, config, params).unwrap();
        let encoder = EncoderModel::new(&model, RayGridConfig::default(), &ProprioceptionConfig::default()).unwrap();
        let encoder_params = EncoderParams::init_default(encoder.num_params());
        let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
        let decoder_params = decoder.init_default_params();
        let calib = DnCalibration {
            mu: vec![0.0; model.num_outputs()],
            sigma: vec![1.0; model.num_outputs()],
        };
        let wm_head = WorldModelHead::new(&model, &WorldModelConfig::default()).unwrap();
        let wm_params = wm_head.init_default_params();

        let ckpt_path = dir.path().join("brain.ckpt");
        save_brain_checkpoint(
            &ckpt_path,
            &flyg_path,
            &brain_config_path,
            &encoder_params,
            &decoder_params,
            &calib,
            &wm_params,
            BrainCheckpointMeta::default(),
        )
        .unwrap();

        let other_path = dir.path().join("b.flyg");
        std::fs::write(&other_path, b"not the same file").unwrap();
        let err = load_brain_checkpoint_for_flyg(&ckpt_path, &other_path, &brain_config_path).unwrap_err();
        assert!(matches!(err, BrainCheckpointError::FlygMismatch { .. }));
    }

    #[test]
    fn loading_against_a_different_brain_config_fails_clearly() {
        let flyg = tiny_flyg_with_everything();
        let dir = tempfile::tempdir().unwrap();
        let flyg_path = dir.path().join("a.flyg");
        ddai_flyg::save(&flyg, &flyg_path).unwrap();
        let brain_config_path = dir.path().join("s-brain.toml");
        std::fs::write(&brain_config_path, b"# original brain config").unwrap();

        let config = crate::config::FlyConfig::default();
        let params = crate::params::FlyParams::init_default(&flyg, &config, 1);
        let model = crate::model::FlyModel::new(flyg, config, params).unwrap();
        let encoder = EncoderModel::new(&model, RayGridConfig::default(), &ProprioceptionConfig::default()).unwrap();
        let encoder_params = EncoderParams::init_default(encoder.num_params());
        let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
        let decoder_params = decoder.init_default_params();
        let calib = DnCalibration {
            mu: vec![0.0; model.num_outputs()],
            sigma: vec![1.0; model.num_outputs()],
        };
        let wm_head = WorldModelHead::new(&model, &WorldModelConfig::default()).unwrap();
        let wm_params = wm_head.init_default_params();

        let ckpt_path = dir.path().join("brain.ckpt");
        save_brain_checkpoint(
            &ckpt_path,
            &flyg_path,
            &brain_config_path,
            &encoder_params,
            &decoder_params,
            &calib,
            &wm_params,
            BrainCheckpointMeta::default(),
        )
        .unwrap();

        let other_config_path = dir.path().join("edited-brain.toml");
        std::fs::write(&other_config_path, b"# a config edited after training").unwrap();
        let err = load_brain_checkpoint_for_flyg(&ckpt_path, &flyg_path, &other_config_path).unwrap_err();
        assert!(matches!(err, BrainCheckpointError::BrainConfigMismatch { .. }));
    }

    #[test]
    fn validate_brain_checkpoint_shapes_catches_a_shape_mismatch() {
        let flyg = tiny_flyg_with_everything();
        let config = crate::config::FlyConfig::default();
        let params = crate::params::FlyParams::init_default(&flyg, &config, 1);
        let model = crate::model::FlyModel::new(flyg, config, params).unwrap();
        let encoder = EncoderModel::new(&model, RayGridConfig::default(), &ProprioceptionConfig::default()).unwrap();
        let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
        let wm_head = WorldModelHead::new(&model, &WorldModelConfig::default()).unwrap();

        let mut checkpoint = BrainCheckpoint {
            format_version: BRAIN_CHECKPOINT_FORMAT_VERSION,
            flyg_sha256: String::new(),
            brain_config_sha256: String::new(),
            encoder_params: EncoderParams::init_default(encoder.num_params()),
            decoder_params: decoder.init_default_params(),
            calibration: DnCalibration {
                mu: vec![0.0; model.num_outputs()],
                sigma: vec![1.0; model.num_outputs()],
            },
            world_model_params: wm_head.init_default_params(),
            meta: BrainCheckpointMeta::default(),
        };
        // A shorter `jump_w` than this decoder expects -- exactly what an edited `.flyg`/config
        // that changed the `jump` action's member count would produce (review round 1, F11).
        checkpoint.decoder_params.jump_w.pop();
        let err = validate_brain_checkpoint_shapes(&checkpoint, &encoder, &decoder, &wm_head).unwrap_err();
        assert!(matches!(err, BrainCheckpointError::InvalidDecoderParams(_)));
    }

    #[test]
    fn validate_brain_checkpoint_shapes_catches_a_non_finite_value() {
        let flyg = tiny_flyg_with_everything();
        let config = crate::config::FlyConfig::default();
        let params = crate::params::FlyParams::init_default(&flyg, &config, 1);
        let model = crate::model::FlyModel::new(flyg, config, params).unwrap();
        let encoder = EncoderModel::new(&model, RayGridConfig::default(), &ProprioceptionConfig::default()).unwrap();
        let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
        let wm_head = WorldModelHead::new(&model, &WorldModelConfig::default()).unwrap();

        let mut checkpoint = BrainCheckpoint {
            format_version: BRAIN_CHECKPOINT_FORMAT_VERSION,
            flyg_sha256: String::new(),
            brain_config_sha256: String::new(),
            encoder_params: EncoderParams::init_default(encoder.num_params()),
            decoder_params: decoder.init_default_params(),
            calibration: DnCalibration {
                mu: vec![0.0; model.num_outputs()],
                sigma: vec![1.0; model.num_outputs()],
            },
            world_model_params: wm_head.init_default_params(),
            meta: BrainCheckpointMeta::default(),
        };
        checkpoint.calibration.sigma[0] = f32::NAN;
        let err = validate_brain_checkpoint_shapes(&checkpoint, &encoder, &decoder, &wm_head).unwrap_err();
        assert!(matches!(err, BrainCheckpointError::InvalidCalibration(_)));
    }

    #[test]
    fn truncated_file_is_rejected_without_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.ckpt");
        std::fs::write(&path, [1, 2, 3]).unwrap();
        let result = std::panic::catch_unwind(|| load_brain_checkpoint(&path));
        assert!(matches!(result, Ok(Err(BrainCheckpointError::Truncated { .. }))));
    }

    #[test]
    fn corrupted_bytes_are_rejected_without_panic() {
        let flyg = tiny_flyg_with_everything();
        let config = crate::config::FlyConfig::default();
        let params = crate::params::FlyParams::init_default(&flyg, &config, 1);
        let model = crate::model::FlyModel::new(flyg, config, params).unwrap();
        let encoder = EncoderModel::new(&model, RayGridConfig::default(), &ProprioceptionConfig::default()).unwrap();
        let encoder_params = EncoderParams::init_default(encoder.num_params());
        let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
        let decoder_params = decoder.init_default_params();
        let calib = DnCalibration {
            mu: vec![0.0; model.num_outputs()],
            sigma: vec![1.0; model.num_outputs()],
        };
        let wm_head = WorldModelHead::new(&model, &WorldModelConfig::default()).unwrap();
        let wm_params = wm_head.init_default_params();

        let dir = tempfile::tempdir().unwrap();
        let flyg_path = dir.path().join("s.flyg");
        ddai_flyg::save(model.flyg(), &flyg_path).unwrap();
        let brain_config_path = dir.path().join("s-brain.toml");
        std::fs::write(&brain_config_path, b"# a brain config, for its sha256 only").unwrap();
        let ckpt_path = dir.path().join("brain.ckpt");
        save_brain_checkpoint(
            &ckpt_path,
            &flyg_path,
            &brain_config_path,
            &encoder_params,
            &decoder_params,
            &calib,
            &wm_params,
            BrainCheckpointMeta::default(),
        )
        .unwrap();

        let mut bytes = std::fs::read(&ckpt_path).unwrap();
        let mid = bytes.len() / 2;
        for b in bytes.iter_mut().skip(mid).take(8) {
            *b ^= 0xFF;
        }
        std::fs::write(&ckpt_path, &bytes).unwrap();
        let result = std::panic::catch_unwind(|| load_brain_checkpoint(&ckpt_path));
        assert!(matches!(result, Ok(Err(_))));
    }
}
