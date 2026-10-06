//! The self-contained fly checkpoint used by training and the arena (task 8.2): everything a
//! [`crate::brain::FlyBrain`] needs except the `.flyg` graph itself, which is pinned by sha256.
//!
//! [`crate::brain_checkpoint::BrainCheckpoint`] (7.3) holds the encoder/decoder/world-model side
//! only and leaves the connectome's own `a`/`b`/`theta` to [`crate::checkpoint::Checkpoint`]
//! (7.2), so restoring a brain took two files plus a config file that could have changed since.
//! The bundle is one file: fly parameters and hyper-parameters, encoder and decoder parameters,
//! the frozen DN calibration and the *text* of the brain config it was trained with. The on-disk
//! discipline is the same as the other two (`[sha256 of the postcard bytes] ++ zstd`, atomic
//! rename), and [`FlyBrainTemplate`] builds any number of independent brains from one loaded
//! bundle, which is what an arena batch needs (one load, a brain per game).

use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::bc::{HeadThresholds, HookView};
use crate::brain::{FlyBrain, FlyBrainConfig};
use crate::brain_config::{BrainConfig, parse_brain_config};
use crate::config::FlyConfig;
use crate::decoder::{DecoderModel, DecoderParams, DnCalibration};
use crate::encoder::{EncoderModel, EncoderParams};
use crate::model::FlyModel;
use crate::params::FlyParams;

/// Version 2 added [`FlyBundle::thresholds`], version 3 [`FlyBundle::hook_view`]; version 1 and 2 files
/// (everything trained before 8.2b) still load, with the default thresholds / the shared hook view.
pub const BUNDLE_FORMAT_VERSION: u32 = 3;

/// Anything that can go wrong saving or loading a bundle; the message says what.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleError(pub String);

impl std::fmt::Display for BundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BundleError {}

fn err<E: std::fmt::Display>(what: &str) -> impl FnOnce(E) -> BundleError + '_ {
    move |e| BundleError(format!("{what}: {e}"))
}

/// Provenance carried with the weights.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BundleMeta {
    pub seed: u64,
    pub git_commit: Option<String>,
    /// Optimiser steps the weights have been trained for.
    pub steps: u64,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlyBundle {
    pub format_version: u32,
    /// sha256 (hex) of the `.flyg` the parameters were built for.
    pub flyg_sha256: String,
    /// Where that `.flyg` was when the bundle was written; only a hint (the hash decides).
    pub flyg_path_hint: String,
    /// The `configs/fly/*-brain.toml` text the encoder/decoder were built from.
    pub brain_config_toml: String,
    pub fly_config: FlyConfig,
    pub fly_params: FlyParams,
    pub encoder_params: EncoderParams,
    pub decoder_params: DecoderParams,
    pub calibration: DnCalibration,
    pub meta: BundleMeta,
    /// Decision thresholds of the jump/hook/fire heads (format v2).
    pub thresholds: HeadThresholds,
    /// How the hook head sees the own hook state (format v3); a masked model is played in two views.
    pub hook_view: HookView,
}

/// The format-version-2 layout (no hook view), kept to read old files.
#[derive(Deserialize)]
struct FlyBundleV2 {
    #[allow(dead_code)] // decoded to keep the layout; the version was peeked already
    format_version: u32,
    flyg_sha256: String,
    flyg_path_hint: String,
    brain_config_toml: String,
    fly_config: FlyConfig,
    fly_params: FlyParams,
    encoder_params: EncoderParams,
    decoder_params: DecoderParams,
    calibration: DnCalibration,
    meta: BundleMeta,
    thresholds: HeadThresholds,
}

/// The format-version-1 layout (no thresholds), kept to read old files.
#[derive(Deserialize)]
struct FlyBundleV1 {
    #[allow(dead_code)] // decoded to keep the layout; the version was peeked already
    format_version: u32,
    flyg_sha256: String,
    flyg_path_hint: String,
    brain_config_toml: String,
    fly_config: FlyConfig,
    fly_params: FlyParams,
    encoder_params: EncoderParams,
    decoder_params: DecoderParams,
    calibration: DnCalibration,
    meta: BundleMeta,
}

/// Only the leading version field of any bundle, to pick the layout before decoding.
#[derive(Deserialize)]
pub struct VersionHead {
    pub format_version: u32,
}

const HASH_LEN: usize = 32;

fn sha256_raw(bytes: &[u8]) -> [u8; HASH_LEN] {
    let digest = Sha256::digest(bytes);
    let mut out = [0u8; HASH_LEN];
    out.copy_from_slice(&digest);
    out
}

/// sha256 (hex) of a file's bytes.
pub fn sha256_hex_of_file(path: &Path) -> Result<String, BundleError> {
    let bytes = std::fs::read(path).map_err(err(&format!("reading {}", path.display())))?;
    Ok(sha256_raw(&bytes).iter().map(|b| format!("{b:02x}")).collect())
}

/// Writes `value` as `[sha256 of the postcard bytes] ++ zstd(postcard bytes)` (level `level`,
/// content checksum on), atomically (sibling temp file + rename).
pub fn write_zstd_postcard<T: Serialize>(path: &Path, value: &T, level: i32) -> Result<(), BundleError> {
    let bytes = postcard::to_allocvec(value).map_err(err("encoding"))?;
    let hash = sha256_raw(&bytes);
    let mut compressed = Vec::new();
    {
        let mut enc = zstd::stream::Encoder::new(&mut compressed, level).map_err(err("zstd"))?;
        enc.include_checksum(true).map_err(err("zstd"))?;
        enc.write_all(&bytes).map_err(err("zstd"))?;
        enc.finish().map_err(err("zstd"))?;
    }
    let mut out = Vec::with_capacity(HASH_LEN + compressed.len());
    out.extend_from_slice(&hash);
    out.extend_from_slice(&compressed);
    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
    let tmp = path.with_file_name(format!("{file_name}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, &out).map_err(err(&format!("writing {}", tmp.display())))?;
    std::fs::rename(&tmp, path).map_err(err(&format!("renaming to {}", path.display())))?;
    Ok(())
}

/// Reads what [`write_zstd_postcard`] wrote and returns the verified postcard bytes.
pub fn read_zstd_bytes(path: &Path) -> Result<Vec<u8>, BundleError> {
    let all = std::fs::read(path).map_err(err(&format!("reading {}", path.display())))?;
    if all.len() < HASH_LEN {
        return Err(BundleError(format!(
            "{}: truncated ({} bytes)",
            path.display(),
            all.len()
        )));
    }
    let (stored, compressed) = all.split_at(HASH_LEN);
    let bytes = zstd::stream::decode_all(compressed).map_err(err(&format!("{}: zstd", path.display())))?;
    if sha256_raw(&bytes).as_slice() != stored {
        return Err(BundleError(format!(
            "{}: payload hash mismatch (corrupt file)",
            path.display()
        )));
    }
    Ok(bytes)
}

/// Reads what [`write_zstd_postcard`] wrote, verifying the hash before decoding.
pub fn read_zstd_postcard<T: DeserializeOwned>(path: &Path) -> Result<T, BundleError> {
    let bytes = read_zstd_bytes(path)?;
    postcard::from_bytes(&bytes).map_err(err(&format!("{}: decoding", path.display())))
}

/// Decodes a verified payload as `T`, naming `what` in the error.
pub fn decode_payload<T: DeserializeOwned>(path: &Path, bytes: &[u8], what: &str) -> Result<T, BundleError> {
    postcard::from_bytes(bytes).map_err(|e| BundleError(format!("{}: decoding {what}: {e}", path.display())))
}

/// The format version a verified payload starts with.
pub fn peek_version(path: &Path, bytes: &[u8]) -> Result<u32, BundleError> {
    postcard::from_bytes::<VersionHead>(bytes)
        .map(|h| h.format_version)
        .map_err(err(&format!("{}: decoding the version", path.display())))
}

/// Writes a bundle.
pub fn save_bundle(path: &Path, bundle: &FlyBundle) -> Result<(), BundleError> {
    write_zstd_postcard(path, bundle, 3)
}

/// Reads a bundle, checking its format version (version 1 files load with default thresholds).
pub fn load_bundle(path: &Path) -> Result<FlyBundle, BundleError> {
    let bytes = read_zstd_bytes(path)?;
    match peek_version(path, &bytes)? {
        BUNDLE_FORMAT_VERSION => {
            let b: FlyBundle = decode_payload(path, &bytes, "a v3 bundle")?;
            b.thresholds
                .validate()
                .map_err(|e| BundleError(format!("{}: {e}", path.display())))?;
            Ok(b)
        }
        1 => {
            let b: FlyBundleV1 = decode_payload(path, &bytes, "a v1 bundle")?;
            Ok(FlyBundle {
                format_version: BUNDLE_FORMAT_VERSION,
                flyg_sha256: b.flyg_sha256,
                flyg_path_hint: b.flyg_path_hint,
                brain_config_toml: b.brain_config_toml,
                fly_config: b.fly_config,
                fly_params: b.fly_params,
                encoder_params: b.encoder_params,
                decoder_params: b.decoder_params,
                calibration: b.calibration,
                meta: b.meta,
                thresholds: HeadThresholds::default(),
                hook_view: HookView::Shared,
            })
        }
        2 => {
            let b: FlyBundleV2 = decode_payload(path, &bytes, "a v2 bundle")?;
            b.thresholds
                .validate()
                .map_err(|e| BundleError(format!("{}: {e}", path.display())))?;
            Ok(FlyBundle {
                format_version: BUNDLE_FORMAT_VERSION,
                flyg_sha256: b.flyg_sha256,
                flyg_path_hint: b.flyg_path_hint,
                brain_config_toml: b.brain_config_toml,
                fly_config: b.fly_config,
                fly_params: b.fly_params,
                encoder_params: b.encoder_params,
                decoder_params: b.decoder_params,
                calibration: b.calibration,
                meta: b.meta,
                thresholds: b.thresholds,
                hook_view: HookView::Shared,
            })
        }
        v => Err(BundleError(format!(
            "{}: bundle format version {v} is not supported (expected 1 to {BUNDLE_FORMAT_VERSION})",
            path.display()
        ))),
    }
}

/// A copy of `bundle` whose encoder also reads the target opponent's state (task 8.5a): the `[opponent_state]` section
/// given as TOML text (only that section is read) is appended to the embedded brain config, and the encoder
/// parameters of the new `(type, channel)` pairs are **zero** (`g = 0`, `c = 0`), so the upgraded fly plays bit for bit
/// like `bundle` until training moves them. Needs the graph the bundle was built for. The format version is unchanged:
/// the encoder's shape follows from the embedded config, which an old bundle simply does not have a section in.
pub fn upgrade_with_opponent_state(
    bundle: &FlyBundle,
    flyg: ddai_flyg::Flyg,
    opponent_state_toml: &str,
) -> Result<FlyBundle, BundleError> {
    #[derive(Deserialize)]
    struct Section {
        opponent_state: crate::encoder::OpponentStateConfig,
    }
    #[derive(Serialize)]
    struct SectionOut<'a> {
        opponent_state: &'a crate::encoder::OpponentStateConfig,
    }
    let section: Section = toml::from_str(opponent_state_toml).map_err(err("the [opponent_state] section"))?;
    if section.opponent_state.is_empty() {
        return Err(BundleError("the [opponent_state] section names no input type".into()));
    }
    let old_cfg = parse_brain_config(&bundle.brain_config_toml).map_err(err("embedded brain config"))?;
    if !old_cfg.opponent_state.is_empty() {
        return Err(BundleError("the bundle already has opponent-state channels".into()));
    }
    let new_toml = format!(
        "{}\n\n# Task 8.5a: the target opponent's state (frozen, freeze time left, velocity, hook), zero-initialised.\n{}",
        bundle.brain_config_toml.trim_end(),
        toml::to_string(&SectionOut {
            opponent_state: &section.opponent_state
        })
        .map_err(err("encoding the section"))?
    );
    let new_cfg = parse_brain_config(&new_toml).map_err(err("the upgraded brain config"))?;
    let model = FlyModel::new(flyg, bundle.fly_config, bundle.fly_params.clone()).map_err(err("fly model"))?;
    let old_enc = old_cfg.encoder_model(&model).map_err(err("encoder"))?;
    let new_enc = new_cfg.encoder_model(&model).map_err(err("upgraded encoder"))?;
    let n_old = old_enc.num_params();
    if new_enc.assignments().len() < n_old
        || new_enc.assignments()[..n_old]
            .iter()
            .zip(old_enc.assignments())
            .any(|(a, b)| a.type_index != b.type_index || a.channel != b.channel)
    {
        return Err(BundleError(
            "the upgraded encoder does not keep the old parameters in place".into(),
        ));
    }
    bundle
        .encoder_params
        .validate_shape(n_old, old_enc.ray_grid_config().num_distance_bins)
        .map_err(err("the bundle's encoder params"))?;
    let extra = new_enc.num_params() - n_old;
    let nb = new_enc.ray_grid_config().num_distance_bins;
    let mut out = bundle.clone();
    out.brain_config_toml = new_toml;
    out.encoder_params.g.extend(std::iter::repeat_n(0.0, extra));
    out.encoder_params.c.extend(std::iter::repeat_n(0.0, extra));
    if !out.encoder_params.bin_gain.is_empty() {
        out.encoder_params.bin_gain.extend(std::iter::repeat_n(1.0, extra * nb));
    }
    out.encoder_params
        .validate_shape(new_enc.num_params(), nb)
        .map_err(err("the upgraded encoder params"))?;
    Ok(out)
}

/// Everything shared by the brains built from one bundle: the models and parameters, immutable.
/// `Sync`, so an arena batch shares one template across its worker threads.
pub struct FlyBrainTemplate {
    model: FlyModel,
    encoder: EncoderModel,
    encoder_params: EncoderParams,
    decoder: DecoderModel,
    decoder_params: DecoderParams,
    calib: DnCalibration,
    brain_config: BrainConfig,
    thresholds: HeadThresholds,
    hook_view: HookView,
    /// The resting state, warmed up once (each brain restores it on `reset`).
    rest: crate::state::FlyState,
    rest_converged: bool,
    pub meta: BundleMeta,
    /// `(name, sha256)` of the bundle file this template was loaded from (`None` when built from
    /// parts): what the web panel shows. Never a path.
    identity: Option<(String, String)>,
}

/// A short display name for a bundle file: `<run>/<stem>` for a file under `<run>/checkpoints/` or
/// `<run>/rounds/` (`e008-p1-fly-base-s1/final`), else the file stem. Never the full path.
pub fn bundle_display_name(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map_or_else(|| "bundle".to_string(), |s| s.to_string_lossy().into_owned());
    let dir = path.parent();
    let in_run_dir = dir
        .and_then(Path::file_name)
        .is_some_and(|d| d == "checkpoints" || d == "rounds");
    match dir.and_then(Path::parent).and_then(Path::file_name) {
        Some(run) if in_run_dir => format!("{}/{stem}", run.to_string_lossy()),
        _ => stem,
    }
}

impl FlyBrainTemplate {
    /// Builds the models from `bundle` and an already loaded graph; verifies the graph hash,
    /// every parameter shape and finiteness.
    pub fn from_parts(bundle: FlyBundle, flyg: ddai_flyg::Flyg, flyg_sha256: &str) -> Result<Self, BundleError> {
        if flyg_sha256 != bundle.flyg_sha256 {
            return Err(BundleError(format!(
                "bundle was trained against a different .flyg: expected sha256 {}, this graph is {flyg_sha256}",
                bundle.flyg_sha256
            )));
        }
        let brain_config = parse_brain_config(&bundle.brain_config_toml).map_err(err("embedded brain config"))?;
        let model = FlyModel::new(flyg, bundle.fly_config, bundle.fly_params).map_err(err("fly model"))?;
        let encoder = brain_config.encoder_model(&model).map_err(err("encoder"))?;
        let decoder = DecoderModel::new(&model, brain_config.decoder.clone()).map_err(err("decoder"))?;
        bundle
            .encoder_params
            .validate_shape(encoder.num_params(), encoder.ray_grid_config().num_distance_bins)
            .map_err(err("encoder params"))?;
        decoder
            .validate_params_shape(&bundle.decoder_params)
            .map_err(err("decoder params"))?;
        bundle
            .calibration
            .validate_shape(decoder.num_outputs())
            .map_err(err("calibration"))?;
        bundle.thresholds.validate().map_err(err("thresholds"))?;
        let mut rest = crate::state::FlyState::new(&model);
        let rest_converged = rest.warm_up(&model).converged;
        Ok(FlyBrainTemplate {
            model,
            encoder,
            encoder_params: bundle.encoder_params,
            decoder,
            decoder_params: bundle.decoder_params,
            calib: bundle.calibration,
            brain_config,
            thresholds: bundle.thresholds,
            hook_view: bundle.hook_view,
            rest,
            rest_converged,
            meta: bundle.meta,
            identity: None,
        })
    }

    /// Loads a bundle file and the `.flyg` it names: `flyg_override` if given, else the path the
    /// bundle was written with.
    pub fn load(bundle_path: &Path, flyg_override: Option<&Path>) -> Result<Self, BundleError> {
        let bundle = load_bundle(bundle_path)?;
        let identity = (bundle_display_name(bundle_path), sha256_hex_of_file(bundle_path)?);
        let flyg_path: PathBuf = flyg_override
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(&bundle.flyg_path_hint));
        let sha = sha256_hex_of_file(&flyg_path)?;
        let flyg = ddai_flyg::load(&flyg_path).map_err(err(&format!("loading {}", flyg_path.display())))?;
        let mut template = Self::from_parts(bundle, flyg, &sha)?;
        template.identity = Some(identity);
        Ok(template)
    }

    pub fn model(&self) -> &FlyModel {
        &self.model
    }
    pub fn encoder(&self) -> &EncoderModel {
        &self.encoder
    }
    pub fn decoder(&self) -> &DecoderModel {
        &self.decoder
    }
    pub fn brain_config(&self) -> &BrainConfig {
        &self.brain_config
    }
    pub fn thresholds(&self) -> HeadThresholds {
        self.thresholds
    }
    /// How the hook head sees the own hook state; a masked model must be played in two views
    /// ([`crate::two_view::TwoViewFly`], via [`FlyBrainTemplate::instantiate_played`]).
    pub fn hook_view(&self) -> HookView {
        self.hook_view
    }

    /// A fresh brain (its own state and scratch buffers) over copies of the shared models.
    pub fn instantiate(&self, config: FlyBrainConfig) -> FlyBrain {
        let mut brain = FlyBrain::new(
            self.model.clone(),
            self.encoder.clone(),
            self.encoder_params.clone(),
            self.decoder.clone(),
            self.decoder_params.clone(),
            self.calib.clone(),
            config,
        );
        brain.adopt_rest(&self.rest, self.rest_converged);
        brain.set_thresholds(self.thresholds);
        if let Some((name, sha256)) = &self.identity {
            brain.set_identity(name.clone(), sha256.clone());
        }
        brain
    }

    /// The brain to **play** with this bundle: a plain fly, or, for a model trained with the hook head masked
    /// (`hook_view() == MaskedForHookHead`), the two-view brain ([`crate::two_view::TwoViewFly`]). The arena and the
    /// live bot both come through here so that a masked bundle is never played single-view.
    pub fn instantiate_played(&self, config: FlyBrainConfig) -> Box<dyn ddai_brain::Brain> {
        if self.hook_view == HookView::MaskedForHookHead {
            Box::new(crate::two_view::TwoViewFly::new(
                self.instantiate(config.clone()),
                self.instantiate(config),
            ))
        } else {
            Box::new(self.instantiate(config))
        }
    }

    /// Whether the template's one warm-up converged (a non-converged rest is still a usable
    /// starting state, but worth reporting).
    pub fn rest_converged(&self) -> bool {
        self.rest_converged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zstd_postcard_round_trips_and_rejects_corruption_and_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.bin");
        let value: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
        write_zstd_postcard(&path, &value, 3).unwrap();
        assert_eq!(read_zstd_postcard::<Vec<f32>>(&path).unwrap(), value);

        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 5;
        bytes[last] ^= 0x40;
        let bad = dir.path().join("bad.bin");
        std::fs::write(&bad, &bytes).unwrap();
        assert!(read_zstd_postcard::<Vec<f32>>(&bad).is_err());

        std::fs::write(&bad, &bytes[..10]).unwrap();
        let e = read_zstd_postcard::<Vec<f32>>(&bad).unwrap_err();
        assert!(e.0.contains("truncated"), "{e}");
    }

    #[test]
    fn peek_version_reads_the_leading_field_and_v1_layout_decodes_without_thresholds() {
        #[derive(Serialize)]
        struct Old {
            format_version: u32,
            name: String,
        }
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct New {
            format_version: u32,
            name: String,
            thresholds: HeadThresholds,
        }
        let dir = tempfile::tempdir().unwrap();
        let old = dir.path().join("old.bin");
        write_zstd_postcard(
            &old,
            &Old {
                format_version: 1,
                name: "x".into(),
            },
            3,
        )
        .unwrap();
        let bytes = read_zstd_bytes(&old).unwrap();
        assert_eq!(peek_version(&old, &bytes).unwrap(), 1);
        // Decoding the old bytes as the new layout fails (too short), so the loader must branch on the version.
        assert!(decode_payload::<New>(&old, &bytes, "new").is_err());
        let new = dir.path().join("new.bin");
        let v = New {
            format_version: 2,
            name: "y".into(),
            thresholds: HeadThresholds {
                jump: 0.3,
                hook: 0.6,
                fire: 0.8,
            },
        };
        write_zstd_postcard(&new, &v, 3).unwrap();
        let bytes = read_zstd_bytes(&new).unwrap();
        assert_eq!(peek_version(&new, &bytes).unwrap(), 2);
        assert_eq!(decode_payload::<New>(&new, &bytes, "new").unwrap(), v);
    }

    #[test]
    fn thresholds_must_be_probabilities() {
        assert!(HeadThresholds::default().validate().is_ok());
        for bad in [0.0f32, 1.0, -0.1, 1.5, f32::NAN] {
            let t = HeadThresholds {
                jump: 0.5,
                hook: bad,
                fire: 0.5,
            };
            assert!(t.validate().is_err(), "{bad}");
        }
    }
}
