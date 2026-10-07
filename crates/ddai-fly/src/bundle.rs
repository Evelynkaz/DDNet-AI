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

use crate::bc::{HeadThresholds, HookDecode, HookParam, HookView};
use crate::brain::{FlyBrain, FlyBrainConfig};
use crate::brain_config::{BrainConfig, parse_brain_config};
use crate::config::FlyConfig;
use crate::decoder::{DecoderModel, DecoderParams, DecoderParamsV3, DecoderParamsV4, DnCalibration};
use crate::encoder::{EncoderModel, EncoderParams};
use crate::hook_wide::HookReadout;
use crate::model::FlyModel;
use crate::params::FlyParams;

/// Version 2 added [`FlyBundle::thresholds`], version 3 [`FlyBundle::hook_view`], version 4 (task 8.6) [`FlyBundle::hook_param`] and
/// [`FlyBundle::hook_decode`] (and the release hazard in `decoder_params`), version 5 (task 8.7) [`FlyBundle::hook_readout`] (and the wide
/// readout's parameters in `decoder_params`); version 1 to 4 files (everything trained before 8.7) still load, with the default thresholds /
/// the shared hook view / the `legacy` hook head and the plain decode / the `pooled` readout, and **play bit for bit** as before.
pub const BUNDLE_FORMAT_VERSION: u32 = 5;
/// The version a `Legacy` + `Plain` + `Pooled` bundle is still written as (readable by every binary since 8.2).
pub const LEGACY_BUNDLE_FORMAT_VERSION: u32 = 3;
/// The version an intent head or a latched decode with the `Pooled` readout is still written as (readable by the 8.6 binaries).
pub const INTENT_BUNDLE_FORMAT_VERSION: u32 = 4;

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
    /// How the hook head is parameterised (format v4): `Intent` iff `decoder_params.hook_release` is set.
    pub hook_param: HookParam,
    /// How the hook probability becomes the hook key (format v4): the plain threshold or the hysteresis decode.
    pub hook_decode: HookDecode,
    /// How the hook head reads the network (format v5): `Pooled` (the twelve-parameter head of every earlier bundle) or a wide readout whose
    /// parameters are `decoder_params.hook_wide`.
    pub hook_readout: HookReadout,
}

impl FlyBundle {
    /// The kind of the hook head must agree with the parameters (an `Intent` bundle carries the release hazard, a `Legacy` one does not;
    /// a wide readout carries its parameters and is not combined with an intent head) and the decode must be valid.
    pub fn validate_hook(&self) -> Result<(), String> {
        match (self.hook_readout, self.decoder_params.hook_wide.is_some()) {
            (HookReadout::Pooled, true) => {
                return Err("hook_readout is pooled but decoder_params has a wide readout".into());
            }
            (r, false) if r != HookReadout::Pooled => {
                return Err(format!(
                    "hook_readout is {} but decoder_params has no wide readout",
                    r.label()
                ));
            }
            _ => {}
        }
        if self.hook_readout != HookReadout::Pooled && self.hook_param == HookParam::Intent {
            return Err("a wide hook readout is not combined with an intent hook head".into());
        }
        match (self.hook_param, self.decoder_params.hook_release.is_some()) {
            (HookParam::Intent, false) => {
                return Err("hook_param is intent but decoder_params has no release hazard".into());
            }
            (HookParam::Legacy, true) => {
                return Err("hook_param is legacy but decoder_params has a release hazard".into());
            }
            _ => {}
        }
        self.hook_decode.validate()
    }
}

/// The format-version-4 layout (no hook readout), kept to read old files.
#[derive(Deserialize)]
struct FlyBundleV4 {
    #[allow(dead_code)] // decoded to keep the layout; the version was peeked already
    format_version: u32,
    flyg_sha256: String,
    flyg_path_hint: String,
    brain_config_toml: String,
    fly_config: FlyConfig,
    fly_params: FlyParams,
    encoder_params: EncoderParams,
    decoder_params: DecoderParamsV4,
    calibration: DnCalibration,
    meta: BundleMeta,
    thresholds: HeadThresholds,
    hook_view: HookView,
    hook_param: HookParam,
    hook_decode: HookDecode,
}

/// The format-version-3 layout (no hook parameterisation, no hook decode), kept to read old files.
#[derive(Deserialize)]
struct FlyBundleV3 {
    #[allow(dead_code)] // decoded to keep the layout; the version was peeked already
    format_version: u32,
    flyg_sha256: String,
    flyg_path_hint: String,
    brain_config_toml: String,
    fly_config: FlyConfig,
    fly_params: FlyParams,
    encoder_params: EncoderParams,
    decoder_params: DecoderParamsV3,
    calibration: DnCalibration,
    meta: BundleMeta,
    thresholds: HeadThresholds,
    hook_view: HookView,
}

/// The version-3 layout to **write**: a `Legacy` + `Plain` bundle is saved as version 3 so that every binary built before task 8.6 (the
/// rollback copies) still reads it; only an intent head or a latched decode, which such a binary would play wrong, needs version 4.
#[derive(Serialize)]
struct FlyBundleV3Out<'a> {
    format_version: u32,
    flyg_sha256: &'a str,
    flyg_path_hint: &'a str,
    brain_config_toml: &'a str,
    fly_config: &'a FlyConfig,
    fly_params: &'a FlyParams,
    encoder_params: &'a EncoderParams,
    decoder_params: DecoderParamsV3,
    calibration: &'a DnCalibration,
    meta: &'a BundleMeta,
    thresholds: &'a HeadThresholds,
    hook_view: HookView,
}

/// The version-4 layout to **write**: a `Pooled` bundle with an intent head or a latched decode is saved as version 4, so the 8.6 binaries still
/// read it; only a wide hook readout, which they could not play, needs version 5.
#[derive(Serialize)]
struct FlyBundleV4Out<'a> {
    format_version: u32,
    flyg_sha256: &'a str,
    flyg_path_hint: &'a str,
    brain_config_toml: &'a str,
    fly_config: &'a FlyConfig,
    fly_params: &'a FlyParams,
    encoder_params: &'a EncoderParams,
    decoder_params: DecoderParamsV4,
    calibration: &'a DnCalibration,
    meta: &'a BundleMeta,
    thresholds: &'a HeadThresholds,
    hook_view: HookView,
    hook_param: HookParam,
    hook_decode: HookDecode,
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
    decoder_params: DecoderParamsV3,
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
    decoder_params: DecoderParamsV3,
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

/// Writes a bundle in the **oldest layout that plays it correctly**: version 3 for a `Legacy` + `Plain` + `Pooled` one (readable by every older
/// binary), version 4 for a `Pooled` one with an intent head or a latched decode (readable by the 8.6 binaries), version 5 for a wide hook
/// readout. An older binary then never reads a file it would play wrong, and keeps reading every file it can.
pub fn save_bundle(path: &Path, bundle: &FlyBundle) -> Result<(), BundleError> {
    // Never write what the loader would refuse (task 8.7, review F5).
    bundle
        .validate_hook()
        .map_err(|e| BundleError(format!("{}: not written: {e}", path.display())))?;
    if bundle.hook_readout == HookReadout::Pooled && bundle.decoder_params.hook_wide.is_none() {
        if bundle.hook_param == HookParam::Legacy
            && bundle.hook_decode == HookDecode::Plain
            && let Some(decoder_params) = bundle.decoder_params.to_v3()
        {
            let v3 = FlyBundleV3Out {
                format_version: LEGACY_BUNDLE_FORMAT_VERSION,
                flyg_sha256: &bundle.flyg_sha256,
                flyg_path_hint: &bundle.flyg_path_hint,
                brain_config_toml: &bundle.brain_config_toml,
                fly_config: &bundle.fly_config,
                fly_params: &bundle.fly_params,
                encoder_params: &bundle.encoder_params,
                decoder_params,
                calibration: &bundle.calibration,
                meta: &bundle.meta,
                thresholds: &bundle.thresholds,
                hook_view: bundle.hook_view,
            };
            return write_zstd_postcard(path, &v3, 3);
        }
        let v4 = FlyBundleV4Out {
            format_version: INTENT_BUNDLE_FORMAT_VERSION,
            flyg_sha256: &bundle.flyg_sha256,
            flyg_path_hint: &bundle.flyg_path_hint,
            brain_config_toml: &bundle.brain_config_toml,
            fly_config: &bundle.fly_config,
            fly_params: &bundle.fly_params,
            encoder_params: &bundle.encoder_params,
            decoder_params: bundle.decoder_params.to_v4(),
            calibration: &bundle.calibration,
            meta: &bundle.meta,
            thresholds: &bundle.thresholds,
            hook_view: bundle.hook_view,
            hook_param: bundle.hook_param,
            hook_decode: bundle.hook_decode,
        };
        return write_zstd_postcard(path, &v4, 3);
    }
    let mut v5 = bundle.clone();
    v5.format_version = BUNDLE_FORMAT_VERSION;
    write_zstd_postcard(path, &v5, 3)
}

/// Reads a bundle, checking its format version (version 1 files load with default thresholds; versions 1 to 3 with the `legacy` hook head
/// and the plain hook decode; versions 1 to 4 with the `pooled` hook readout).
pub fn load_bundle(path: &Path) -> Result<FlyBundle, BundleError> {
    let bytes = read_zstd_bytes(path)?;
    let bad = |e: String| BundleError(format!("{}: {e}", path.display()));
    match peek_version(path, &bytes)? {
        BUNDLE_FORMAT_VERSION => {
            let b: FlyBundle = decode_payload(path, &bytes, "a v5 bundle")?;
            b.thresholds.validate().map_err(bad)?;
            b.validate_hook().map_err(bad)?;
            Ok(b)
        }
        4 => {
            let b: FlyBundleV4 = decode_payload(path, &bytes, "a v4 bundle")?;
            b.thresholds.validate().map_err(bad)?;
            let out = FlyBundle {
                format_version: BUNDLE_FORMAT_VERSION,
                flyg_sha256: b.flyg_sha256,
                flyg_path_hint: b.flyg_path_hint,
                brain_config_toml: b.brain_config_toml,
                fly_config: b.fly_config,
                fly_params: b.fly_params,
                encoder_params: b.encoder_params,
                decoder_params: b.decoder_params.into(),
                calibration: b.calibration,
                meta: b.meta,
                thresholds: b.thresholds,
                hook_view: b.hook_view,
                hook_param: b.hook_param,
                hook_decode: b.hook_decode,
                hook_readout: HookReadout::Pooled,
            };
            out.validate_hook().map_err(bad)?;
            Ok(out)
        }
        3 => {
            let b: FlyBundleV3 = decode_payload(path, &bytes, "a v3 bundle")?;
            b.thresholds.validate().map_err(bad)?;
            Ok(FlyBundle {
                format_version: BUNDLE_FORMAT_VERSION,
                flyg_sha256: b.flyg_sha256,
                flyg_path_hint: b.flyg_path_hint,
                brain_config_toml: b.brain_config_toml,
                fly_config: b.fly_config,
                fly_params: b.fly_params,
                encoder_params: b.encoder_params,
                decoder_params: b.decoder_params.into(),
                calibration: b.calibration,
                meta: b.meta,
                thresholds: b.thresholds,
                hook_view: b.hook_view,
                hook_param: HookParam::Legacy,
                hook_decode: HookDecode::Plain,
                hook_readout: HookReadout::Pooled,
            })
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
                decoder_params: b.decoder_params.into(),
                calibration: b.calibration,
                meta: b.meta,
                thresholds: HeadThresholds::default(),
                hook_view: HookView::Shared,
                hook_param: HookParam::Legacy,
                hook_decode: HookDecode::Plain,
                hook_readout: HookReadout::Pooled,
            })
        }
        2 => {
            let b: FlyBundleV2 = decode_payload(path, &bytes, "a v2 bundle")?;
            b.thresholds.validate().map_err(bad)?;
            Ok(FlyBundle {
                format_version: BUNDLE_FORMAT_VERSION,
                flyg_sha256: b.flyg_sha256,
                flyg_path_hint: b.flyg_path_hint,
                brain_config_toml: b.brain_config_toml,
                fly_config: b.fly_config,
                fly_params: b.fly_params,
                encoder_params: b.encoder_params,
                decoder_params: b.decoder_params.into(),
                calibration: b.calibration,
                meta: b.meta,
                thresholds: b.thresholds,
                hook_view: HookView::Shared,
                hook_param: HookParam::Legacy,
                hook_decode: HookDecode::Plain,
                hook_readout: HookReadout::Pooled,
            })
        }
        v => Err(BundleError(format!(
            "{}: bundle format version {v} is not supported (expected 1 to {BUNDLE_FORMAT_VERSION})",
            path.display()
        ))),
    }
}

/// A copy of `bundle` with the hook head turned into an intent head (task 8.6): the press hazard is the hook head, the release hazard
/// its mirror image, so the intent fly plays **exactly** like the legacy one (same hook probability whatever the latch is) until training
/// moves the two apart. The decode becomes the hysteresis decode at the legacy threshold on both sides (`hi = lo = thresholds.hook`),
/// which is the plain decode: the same decisions. Already intent: unchanged.
pub fn upgrade_to_intent_hook(bundle: &FlyBundle) -> FlyBundle {
    let mut out = bundle.clone();
    out.decoder_params = bundle.decoder_params.with_intent_hook();
    out.hook_param = HookParam::Intent;
    if out.hook_decode == HookDecode::Plain {
        let t = out.thresholds.hook;
        out.hook_decode = HookDecode::Latched { hi: t, lo: t };
    }
    out
}

/// A copy of `bundle` whose hook head also reads the network through the wide readout `readout` (task 8.7). The wide part is a residual on the
/// pooled head with **zero output weights**, so the upgraded fly plays bit for bit like `bundle` until it is trained; the first layer is drawn
/// deterministically from `seed`. Needs the graph the bundle was built for. Refused for an intent hook head and for a bundle that
/// already has a wide readout.
pub fn upgrade_hook_readout(
    bundle: &FlyBundle,
    flyg: ddai_flyg::Flyg,
    readout: HookReadout,
    seed: u64,
) -> Result<FlyBundle, BundleError> {
    if bundle.hook_param == HookParam::Intent {
        return Err(BundleError(
            "a wide hook readout is not combined with an intent hook head".into(),
        ));
    }
    if bundle.hook_readout != HookReadout::Pooled {
        return Err(BundleError(format!(
            "the bundle already has the {} hook readout",
            bundle.hook_readout.label()
        )));
    }
    let cfg = parse_brain_config(&bundle.brain_config_toml).map_err(err("embedded brain config"))?;
    let model = FlyModel::new(flyg, bundle.fly_config, bundle.fly_params.clone()).map_err(err("fly model"))?;
    let mut decoder = DecoderModel::new(&model, cfg.decoder.clone()).map_err(err("decoder"))?;
    decoder.set_hook_readout(&model, readout).map_err(err("hook readout"))?;
    let mut out = bundle.clone();
    out.hook_readout = readout;
    out.decoder_params.hook_wide = decoder.init_hook_wide(seed);
    out.validate_hook()
        .map_err(|e| BundleError(format!("hook readout: {e}")))?;
    decoder
        .validate_params_shape(&out.decoder_params)
        .map_err(err("decoder params"))?;
    Ok(out)
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
    hook_param: HookParam,
    hook_decode: HookDecode,
    hook_readout: HookReadout,
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
        bundle.validate_hook().map_err(err("hook head"))?;
        let brain_config = parse_brain_config(&bundle.brain_config_toml).map_err(err("embedded brain config"))?;
        let model = FlyModel::new(flyg, bundle.fly_config, bundle.fly_params).map_err(err("fly model"))?;
        let encoder = brain_config.encoder_model(&model).map_err(err("encoder"))?;
        let mut decoder = DecoderModel::new(&model, brain_config.decoder.clone()).map_err(err("decoder"))?;
        decoder
            .set_hook_readout(&model, bundle.hook_readout)
            .map_err(err("hook readout"))?;
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
            hook_param: bundle.hook_param,
            hook_decode: bundle.hook_decode,
            hook_readout: bundle.hook_readout,
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
    /// Refuses a fly whose hook depends on the latch (an intent head or a latched decode) where the latch can diverge from the hook that is
    /// actually played: the live bot (a server pause and its resume do not call the brain, the guard and the hook veto change the hook after
    /// `decide`), the hybrid's [`crate::proposer::FlyProposer`] (the latch follows the fly's own argmax, not the hybrid's played action) and
    /// the critical-decision swaps (`ddai_train::critical`). A plain legacy fly has no latch to diverge. Lift this only with a `Brain` hook
    /// that tells the brain the action really sent.
    pub fn require_unlatched(&self, what: &str) -> Result<(), BundleError> {
        if self.hook_param == HookParam::Intent || self.hook_decode != HookDecode::Plain {
            return Err(BundleError(format!(
                "{what}: this checkpoint has an intent hook head or a latched hook decode; its latch (the fly's own previous hook command) \
                 can diverge from the hook actually played here, so it is refused (task 8.6, review F3)"
            )));
        }
        Ok(())
    }

    /// Refuses the **encoder-input control** readout (`mlp-enc-<H>`, task 8.7, review F1) where a fly really plays: the live bot and the hybrid's
    /// proposer. That readout is an MLP on the encoder's input with no connectome (FLY.md section 1 point 3); it exists to be measured in the
    /// arena (`es eval`, `hook-eval`, the BC tools), never to play as "the fly". `linear-dn` and `mlp-dn-<H>` read DN slots only and are allowed.
    pub fn require_fly_readout(&self, what: &str) -> Result<(), BundleError> {
        if self.hook_readout.reads_encoder() {
            return Err(BundleError(format!(
                "{what}: this checkpoint's hook head is the encoder-input control ({}), an MLP with no connectome; it is a measurement control, \
                 not a fly, and is refused where a fly plays (task 8.7, review F1)",
                self.hook_readout.label()
            )));
        }
        Ok(())
    }

    /// How the hook head is parameterised (`Legacy` for every bundle written before format v4).
    pub fn hook_param(&self) -> HookParam {
        self.hook_param
    }
    /// How the hook probability becomes the hook key (`Plain` for every bundle written before format v4).
    pub fn hook_decode(&self) -> HookDecode {
        self.hook_decode
    }
    /// How the hook head reads the network (`Pooled` for every bundle written before format v5).
    pub fn hook_readout(&self) -> HookReadout {
        self.hook_readout
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
        brain.set_hook_decode(self.hook_decode);
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
