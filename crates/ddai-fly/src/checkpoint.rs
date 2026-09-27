//! Checkpoint format (acceptance criterion 1e): `FlyParams` + `FlyConfig` + the sha256 of the
//! `.flyg` file the params were trained against + free-form metadata, postcard + zstd (same
//! on-disk convention as `ddai_flyg::io` / `ddai_connectome::tables` — atomic write via a sibling
//! temp file + rename).
//!
//! Deliberately does *not* store a copy of the `.flyg` itself (that would duplicate a
//! multi-megabyte file per checkpoint and belongs in `~/aiddnet/data/connectome/compiled/`, not in
//! a training run's checkpoint directory) — only its hash, so [`load_checkpoint_for_flyg`] can
//! refuse to apply params trained against a different graph instead of silently misinterpreting
//! `a`/`b`/`theta` against the wrong `shared_param_id`/`type_index` numbering.
//!
//! **On-disk layout** (review round 1, F2 — the original version had no integrity check at all
//! beyond postcard/zstd happening to fail to *decode*; a fuzz test flipping single bits in a real
//! checkpoint found 389 of 400 flips loaded "successfully" with silently different params):
//! `[32 raw bytes: sha256 of the postcard-encoded `Checkpoint`] ++ [zstd stream, with its content
//! checksum enabled, of those same postcard bytes]`. Two independent checks, deliberately not
//! just one:
//! - zstd's own content checksum (`Encoder::include_checksum(true)`) is verified *by zstd itself*
//!   during decompression (`ZSTD_decompressStream`), and catches most corruption of the
//!   compressed bytes before this module ever sees decompressed data at all.
//! - The sha256 over the *decompressed* postcard bytes, checked explicitly in [`load_checkpoint`]
//!   before postcard even attempts to decode them, is what actually gets exercised by this
//!   crate's own bit-flip test (`tests::every_bit_flip_on_an_incompressible_payload_is_rejected`)
//!   — belt and suspenders against relying on zstd's checksum alone.

use std::io::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::FlyConfig;
use crate::error::FlyError;
use crate::optim::AdamState;
use crate::params::FlyParams;

/// Length of the raw (not hex) sha256 digest prepended to every checkpoint file — see the module
/// doc comment's "on-disk layout".
const PAYLOAD_HASH_LEN: usize = 32;

/// Bumped whenever the checkpoint format itself changes in a way an old reader would
/// misinterpret. Independent of `ddai_flyg::FLYG_FORMAT_VERSION` and of `.flyg` content — a
/// checkpoint's `flyg_sha256` is what ties it to one specific compiled graph, not this version.
///
/// **v2** (task 7.2, acceptance criterion 6): added [`Checkpoint::optimizer`] (Adam's moments +
/// step counter). A v1 file has no such field at all, so it cannot be decoded by a v2 reader —
/// [`load_checkpoint`] checks this explicitly (from just the header, before attempting to decode
/// the rest — review round 1, F6's convention) and returns a clear
/// [`FlyError::CheckpointFormatVersionMismatch`] rather than a confusing postcard decode error.
pub const FLY_CHECKPOINT_FORMAT_VERSION: u32 = 2;

/// Free-form provenance, per acceptance criterion 1e ("seed, git commit, notes").
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub seed: u64,
    pub git_commit: Option<String>,
    pub notes: String,
}

/// The complete on-disk checkpoint contents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub format_version: u32,
    /// sha256 (hex, lowercase) of the raw bytes of the `.flyg` file these params were built for.
    pub flyg_sha256: String,
    pub config: FlyConfig,
    pub params: FlyParams,
    pub meta: CheckpointMeta,
    /// Adam's per-parameter moments and step counter (task 7.2, acceptance criterion 6):
    /// `Some` for a checkpoint saved mid-training (so resuming reproduces the exact same next
    /// `adam_step`, not one that restarts the moment estimates from zero); `None` for one saved
    /// before training ever ran an optimiser step (e.g. straight after `FlyParams::init_default`).
    pub optimizer: Option<AdamState>,
}

/// Just [`Checkpoint`]'s first field, decoded with `postcard::take_from_bytes` (which — unlike
/// `from_bytes` — doesn't require the rest of the buffer to be a valid `CheckpointHeader` too) so
/// [`load_checkpoint`] can check `format_version` clearly before attempting to decode a
/// `Checkpoint` whose shape might have changed (review round 1, F6). Field name and type must
/// keep matching `Checkpoint`'s first field exactly.
#[derive(Debug, Deserialize)]
struct CheckpointHeader {
    format_version: u32,
}

fn sha256_hex_of_bytes(bytes: &[u8]) -> String {
    sha256_raw(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_raw(bytes: &[u8]) -> [u8; PAYLOAD_HASH_LEN] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = [0u8; PAYLOAD_HASH_LEN];
    out.copy_from_slice(&digest);
    out
}

/// sha256 (hex) of a file's raw bytes — used both to fill in [`Checkpoint::flyg_sha256`] at save
/// time and to check it at load time. Public so a caller building a `Checkpoint` by hand (rather
/// than through [`save_checkpoint`]) can compute the same hash.
pub fn sha256_hex_of_file(path: &Path) -> Result<String, FlyError> {
    let bytes = std::fs::read(path).map_err(FlyError::Io)?;
    Ok(sha256_hex_of_bytes(&bytes))
}

/// Encodes `checkpoint` (postcard, then zstd level 3 with its content checksum enabled) and
/// writes `[sha256 of the postcard bytes] ++ [zstd stream]` atomically to `path` (temp file +
/// rename, so a reader never observes a half-written file) — see the module doc comment's
/// "on-disk layout" for why both a checksum and an explicit hash.
fn write_checkpoint(path: &Path, checkpoint: &Checkpoint) -> Result<(), FlyError> {
    let bytes = postcard::to_allocvec(checkpoint).map_err(FlyError::Encode)?;
    let payload_hash = sha256_raw(&bytes);

    let mut compressed = Vec::new();
    {
        let mut encoder = zstd::stream::Encoder::new(&mut compressed, 3).map_err(FlyError::Zstd)?;
        encoder.include_checksum(true).map_err(FlyError::Zstd)?;
        encoder.write_all(&bytes).map_err(FlyError::Zstd)?;
        encoder.finish().map_err(FlyError::Zstd)?;
    }

    let mut out = Vec::with_capacity(PAYLOAD_HASH_LEN + compressed.len());
    out.extend_from_slice(&payload_hash);
    out.extend_from_slice(&compressed);

    let tmp_path = unique_sibling_tmp_path(path);
    std::fs::write(&tmp_path, &out).map_err(FlyError::Io)?;
    std::fs::rename(&tmp_path, path).map_err(FlyError::Io)?;
    Ok(())
}

/// A sibling temp path for an atomic write (review round 1, F7: the original
/// `path.with_extension("flyckpt.tmp")` collides for `a.x`/`a.y` — both `with_extension` to the
/// exact same `a.flyckpt.tmp`, so two concurrent saves to differently-named checkpoints sharing a
/// stem could clobber each other's temp file). This appends `.tmp` to the *full* file name
/// instead of replacing the extension, so `a.x.tmp` and `a.y.tmp` never collide, plus the
/// process id for good measure (two processes racing to save the same checkpoint path is still
/// naturally sequenced by the final `rename`, which is atomic).
fn unique_sibling_tmp_path(path: &Path) -> std::path::PathBuf {
    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
    path.with_file_name(format!("{file_name}.{}.tmp", std::process::id()))
}

/// Hashes `flyg_path`, assembles a [`Checkpoint`] from `config`/`params`/`meta`/`optimizer`, and
/// writes it to `path`. `optimizer` is `None` for a checkpoint that hasn't trained yet (see
/// [`Checkpoint::optimizer`]'s doc comment).
pub fn save_checkpoint(
    path: &Path,
    flyg_path: &Path,
    config: &FlyConfig,
    params: &FlyParams,
    meta: CheckpointMeta,
    optimizer: Option<&AdamState>,
) -> Result<(), FlyError> {
    let flyg_sha256 = sha256_hex_of_file(flyg_path)?;
    let checkpoint = Checkpoint {
        format_version: FLY_CHECKPOINT_FORMAT_VERSION,
        flyg_sha256,
        config: *config,
        params: params.clone(),
        meta,
        optimizer: optimizer.cloned(),
    };
    write_checkpoint(path, &checkpoint)
}

/// Reads, splits off and checks the payload hash, zstd-decompresses (which independently checks
/// its own content checksum — see the module doc comment), postcard-decodes, and checks
/// `format_version` — a truncated or corrupted file is rejected here with a [`FlyError`], never a
/// panic, at whichever of those checks first notices (checked in exactly that order, cheapest and
/// most-likely-to-catch-corruption first). Does **not** check the `.flyg` hash (there is no graph
/// to check it against without one being named); see [`load_checkpoint_for_flyg`] for the version
/// that does.
pub fn load_checkpoint(path: &Path) -> Result<Checkpoint, FlyError> {
    let all_bytes = std::fs::read(path).map_err(FlyError::Io)?;
    if all_bytes.len() < PAYLOAD_HASH_LEN {
        return Err(FlyError::Truncated {
            len: all_bytes.len(),
            min_len: PAYLOAD_HASH_LEN,
        });
    }
    let (stored_hash, compressed) = all_bytes.split_at(PAYLOAD_HASH_LEN);

    let bytes = zstd::stream::decode_all(compressed).map_err(FlyError::Zstd)?;

    let actual_hash = sha256_raw(&bytes);
    if actual_hash.as_slice() != stored_hash {
        return Err(FlyError::ChecksumMismatch);
    }

    // Review round 1, F6: check `format_version` from just the header before attempting to decode
    // the full `Checkpoint` — a future format version that changes the struct's shape should fail
    // with a clear `CheckpointFormatVersionMismatch`, not whatever confusing `Decode` error
    // postcard happens to produce trying to fit new bytes into the old struct definition.
    // `format_version` is `Checkpoint`'s first field, and postcard's `take_from_bytes` (unlike
    // `from_bytes`) doesn't require consuming the rest of the buffer, so this only ever looks at
    // the leading few bytes regardless of what the rest of the format looks like.
    let (header, _): (CheckpointHeader, _) = postcard::take_from_bytes(&bytes).map_err(FlyError::Decode)?;
    if header.format_version != FLY_CHECKPOINT_FORMAT_VERSION {
        return Err(FlyError::CheckpointFormatVersionMismatch {
            found: header.format_version,
            expected: FLY_CHECKPOINT_FORMAT_VERSION,
        });
    }

    let checkpoint: Checkpoint = postcard::from_bytes(&bytes).map_err(FlyError::Decode)?;

    // Review round 1, F7d: a checkpoint whose `optimizer` doesn't shape-match its own `params`
    // (both individually well-formed postcard-wise, but disagreeing with each other — a
    // hand-edited or corrupted file could do this even though the top-level checksum passed)
    // used to load "successfully" and only panic later, inside `adam_step`, whenever a caller
    // finally tried to resume training with it.
    if let Some(optimizer) = &checkpoint.optimizer
        && !optimizer.matches_shape(&checkpoint.params)
    {
        return Err(FlyError::OptimizerShapeMismatch(format!(
            "params: a.len()={}, b.len()={}, theta.len()={}; optimizer: m_a.len()={}, m_b.len()={}, m_theta.len()={}",
            checkpoint.params.a.len(),
            checkpoint.params.b.len(),
            checkpoint.params.theta.len(),
            optimizer.m_a.len(),
            optimizer.m_b.len(),
            optimizer.m_theta.len(),
        )));
    }

    Ok(checkpoint)
}

/// [`load_checkpoint`], then checks `flyg_path`'s hash against `checkpoint.flyg_sha256` —
/// "loading against a different graph fails clearly" (acceptance criterion 1e).
pub fn load_checkpoint_for_flyg(path: &Path, flyg_path: &Path) -> Result<Checkpoint, FlyError> {
    let checkpoint = load_checkpoint(path)?;
    let actual = sha256_hex_of_file(flyg_path)?;
    if actual != checkpoint.flyg_sha256 {
        return Err(FlyError::FlygMismatch {
            expected: checkpoint.flyg_sha256.clone(),
            found: actual,
        });
    }
    Ok(checkpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::tiny_chain_flyg;
    use std::io::Write;

    fn write_flyg(dir: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
        let flyg = tiny_chain_flyg();
        let path = dir.path().join(name);
        ddai_flyg::save(&flyg, &path).expect("save fixture .flyg");
        path
    }

    #[test]
    fn round_trip_preserves_everything() {
        let dir = tempfile::tempdir().unwrap();
        let flyg_path = write_flyg(&dir, "s.flyg");
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 42);
        let meta = CheckpointMeta {
            seed: 42,
            git_commit: Some("deadbeef".to_string()),
            notes: "round-trip test".to_string(),
        };
        let ckpt_path = dir.path().join("ckpt.flyckpt");
        save_checkpoint(&ckpt_path, &flyg_path, &config, &params, meta.clone(), None).unwrap();

        let loaded = load_checkpoint_for_flyg(&ckpt_path, &flyg_path).unwrap();
        assert_eq!(loaded.config, config);
        assert_eq!(loaded.params, params);
        assert_eq!(loaded.meta, meta);
        assert_eq!(loaded.format_version, FLY_CHECKPOINT_FORMAT_VERSION);
        assert_eq!(loaded.flyg_sha256, sha256_hex_of_file(&flyg_path).unwrap());
    }

    #[test]
    fn loading_against_a_different_flyg_fails_clearly() {
        let dir = tempfile::tempdir().unwrap();
        let flyg_path = write_flyg(&dir, "a.flyg");
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let ckpt_path = dir.path().join("ckpt.flyckpt");
        save_checkpoint(
            &ckpt_path,
            &flyg_path,
            &config,
            &params,
            CheckpointMeta::default(),
            None,
        )
        .unwrap();

        // A byte-different .flyg file (still valid, just not the same one the checkpoint names).
        let other_path = dir.path().join("b.flyg");
        std::fs::write(&other_path, b"not actually the same file").unwrap();

        let err = load_checkpoint_for_flyg(&ckpt_path, &other_path).unwrap_err();
        assert!(
            matches!(err, FlyError::FlygMismatch { .. }),
            "expected FlygMismatch, got {err:?}"
        );
    }

    #[test]
    fn truncated_file_is_rejected_without_panic() {
        let dir = tempfile::tempdir().unwrap();
        let flyg_path = write_flyg(&dir, "s.flyg");
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let ckpt_path = dir.path().join("ckpt.flyckpt");
        save_checkpoint(
            &ckpt_path,
            &flyg_path,
            &config,
            &params,
            CheckpointMeta::default(),
            None,
        )
        .unwrap();

        let mut bytes = std::fs::read(&ckpt_path).unwrap();
        bytes.truncate(bytes.len() / 2);
        std::fs::write(&ckpt_path, &bytes).unwrap();

        let result = std::panic::catch_unwind(|| load_checkpoint(&ckpt_path));
        match result {
            Ok(Err(_)) => {} // rejected cleanly, as expected
            Ok(Ok(_)) => panic!("truncated checkpoint should not load successfully"),
            Err(_) => panic!("truncated checkpoint must be rejected with an Err, not a panic"),
        }
    }

    #[test]
    fn garbage_file_is_rejected_without_panic() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt_path = dir.path().join("garbage.flyckpt");
        let mut f = std::fs::File::create(&ckpt_path).unwrap();
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02]).unwrap();
        drop(f);

        let result = std::panic::catch_unwind(|| load_checkpoint(&ckpt_path));
        assert!(
            matches!(result, Ok(Err(_))),
            "garbage checkpoint must be Err, not Ok or a panic"
        );
    }

    #[test]
    fn corrupted_middle_bytes_are_rejected_without_panic() {
        let dir = tempfile::tempdir().unwrap();
        let flyg_path = write_flyg(&dir, "s.flyg");
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let ckpt_path = dir.path().join("ckpt.flyckpt");
        save_checkpoint(
            &ckpt_path,
            &flyg_path,
            &config,
            &params,
            CheckpointMeta::default(),
            None,
        )
        .unwrap();

        let mut bytes = std::fs::read(&ckpt_path).unwrap();
        let mid = bytes.len() / 2;
        for b in bytes.iter_mut().skip(mid).take(8) {
            *b ^= 0xFF;
        }
        std::fs::write(&ckpt_path, &bytes).unwrap();

        let result = std::panic::catch_unwind(|| load_checkpoint(&ckpt_path));
        assert!(
            matches!(result, Ok(Err(_))),
            "corrupted checkpoint must be Err, not Ok or a panic"
        );
    }

    /// Review round 1 (F2): the reviewer's own fuzzer flipped single bits in a real (S-scale)
    /// checkpoint with high-entropy (effectively random, so barely compressible — the worst case
    /// for relying on compression-induced structure to "accidentally" catch corruption) `a`
    /// values, and found 389 of 400 flips loaded "successfully" with silently different params.
    /// This samples 2000 distinct bit positions (deterministic seed, so this test is itself
    /// reproducible) across a similarly-sized, similarly high-entropy checkpoint and requires
    /// *every one* to be rejected.
    #[test]
    fn every_sampled_bit_flip_on_a_high_entropy_payload_is_rejected() {
        let mut rng = crate::rng::SplitMix64::new(20260927);
        // S-scale: shared_param_count=5368, types=432 (see docs/FLY.md §3.1) — comparable size
        // and entropy to the reviewer's own probe.
        let a: Vec<f32> = (0..5368).map(|_| rng.next_f32_unit() * 10.0 - 5.0).collect();
        let b: Vec<f32> = (0..432).map(|_| rng.next_f32_unit() * 2.0 - 1.0).collect();
        let theta: Vec<f32> = (0..432).map(|_| rng.next_f32_unit() * 2.0 - 1.0).collect();
        let params = FlyParams { a, b, theta };
        // A real mid-training checkpoint also carries Adam's moments (task 7.2) — included here
        // so this fuzz target's entropy/size stays representative of what's actually on disk now.
        let mut optimizer = AdamState::new(&params);
        for x in optimizer
            .m_a
            .iter_mut()
            .chain(&mut optimizer.v_a)
            .chain(&mut optimizer.m_b)
            .chain(&mut optimizer.v_b)
            .chain(&mut optimizer.m_theta)
            .chain(&mut optimizer.v_theta)
        {
            *x = rng.next_f32_unit() * 4.0 - 2.0;
        }
        optimizer.step = 12345;
        let checkpoint = Checkpoint {
            format_version: FLY_CHECKPOINT_FORMAT_VERSION,
            flyg_sha256: "0".repeat(64),
            config: FlyConfig::default(),
            params,
            meta: CheckpointMeta {
                seed: 1,
                git_commit: None,
                notes: "fuzz target".to_string(),
            },
            optimizer: Some(optimizer),
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fuzz.flyckpt");
        write_checkpoint(&path, &checkpoint).unwrap();
        let original = std::fs::read(&path).unwrap();
        let total_bits = original.len() * 8;
        eprintln!("fuzz target: {} bytes ({total_bits} bits)", original.len());

        let num_samples = 2000usize.min(total_bits);
        let mut tested = std::collections::HashSet::new();
        let mut accepted: Vec<usize> = Vec::new();
        while tested.len() < num_samples {
            let bit = (rng.next_u64() % total_bits as u64) as usize;
            if !tested.insert(bit) {
                continue;
            }
            let mut corrupted = original.clone();
            corrupted[bit / 8] ^= 1 << (bit % 8);
            std::fs::write(&path, &corrupted).unwrap();
            if load_checkpoint(&path).is_ok() {
                accepted.push(bit);
            }
        }

        assert!(
            accepted.is_empty(),
            "{}/{} sampled single-bit flips were NOT rejected (bit positions: {:?})",
            accepted.len(),
            num_samples,
            &accepted[..accepted.len().min(20)]
        );
    }

    #[test]
    fn wrong_format_version_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let flyg_path = write_flyg(&dir, "s.flyg");
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let mut checkpoint = Checkpoint {
            format_version: FLY_CHECKPOINT_FORMAT_VERSION + 1,
            flyg_sha256: sha256_hex_of_file(&flyg_path).unwrap(),
            config,
            params,
            meta: CheckpointMeta::default(),
            optimizer: None,
        };
        checkpoint.format_version = FLY_CHECKPOINT_FORMAT_VERSION + 1;
        let ckpt_path = dir.path().join("ckpt.flyckpt");
        write_checkpoint(&ckpt_path, &checkpoint).unwrap();

        let err = load_checkpoint(&ckpt_path).unwrap_err();
        assert!(matches!(err, FlyError::CheckpointFormatVersionMismatch { .. }));
    }

    #[test]
    fn round_trip_preserves_optimizer_state() {
        let dir = tempfile::tempdir().unwrap();
        let flyg_path = write_flyg(&dir, "s.flyg");
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 42);
        let mut optimizer = AdamState::new(&params);
        optimizer.step = 7;
        optimizer.m_a[0] = 0.5;
        optimizer.v_a[0] = 0.02;

        let ckpt_path = dir.path().join("ckpt.flyckpt");
        save_checkpoint(
            &ckpt_path,
            &flyg_path,
            &config,
            &params,
            CheckpointMeta::default(),
            Some(&optimizer),
        )
        .unwrap();

        let loaded = load_checkpoint_for_flyg(&ckpt_path, &flyg_path).unwrap();
        assert_eq!(loaded.optimizer, Some(optimizer));
    }

    #[test]
    fn checkpoint_saved_before_any_training_step_has_no_optimizer_state() {
        let dir = tempfile::tempdir().unwrap();
        let flyg_path = write_flyg(&dir, "s.flyg");
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);

        let ckpt_path = dir.path().join("ckpt.flyckpt");
        save_checkpoint(
            &ckpt_path,
            &flyg_path,
            &config,
            &params,
            CheckpointMeta::default(),
            None,
        )
        .unwrap();

        let loaded = load_checkpoint_for_flyg(&ckpt_path, &flyg_path).unwrap();
        assert_eq!(loaded.optimizer, None);
    }

    /// Review round 1, F7d: a checkpoint whose `optimizer` doesn't shape-match its own `params`
    /// (both individually well-formed, but disagreeing with each other — built directly via
    /// `write_checkpoint`, bypassing `save_checkpoint`, precisely because `save_checkpoint`'s own
    /// normal usage always builds a matching pair; this simulates a hand-edited/corrupted file
    /// that still decodes and passes the sha256/format-version checks) must be rejected by
    /// `load_checkpoint` itself, not left for `adam_step` to panic on later.
    #[test]
    fn load_checkpoint_rejects_an_optimizer_that_does_not_shape_match_its_own_params() {
        let dir = tempfile::tempdir().unwrap();
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let mut mismatched_optimizer = AdamState::new(&params);
        mismatched_optimizer.m_a.push(0.0); // now one longer than params.a

        let checkpoint = Checkpoint {
            format_version: FLY_CHECKPOINT_FORMAT_VERSION,
            flyg_sha256: "0".repeat(64),
            config,
            params,
            meta: CheckpointMeta::default(),
            optimizer: Some(mismatched_optimizer),
        };
        let ckpt_path = dir.path().join("mismatched.flyckpt");
        write_checkpoint(&ckpt_path, &checkpoint).unwrap();

        let err = load_checkpoint(&ckpt_path).unwrap_err();
        assert!(
            matches!(err, FlyError::OptimizerShapeMismatch(_)),
            "expected OptimizerShapeMismatch, got {err:?}"
        );
    }

    /// Acceptance criterion 6: "resuming training reproduces the same next step" — save mid-run,
    /// load into a fresh `FlyParams`/`AdamState` pair, and check that continuing from the loaded
    /// state produces **bit-identical** params to continuing without ever saving/loading at all.
    #[test]
    fn resuming_training_from_a_checkpoint_reproduces_the_same_next_step() {
        use crate::optim::{AdamConfig, ParamGradients, adam_step};

        let dir = tempfile::tempdir().unwrap();
        let flyg_path = write_flyg(&dir, "s.flyg");
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let adam_config = AdamConfig::default();

        let mut params_continuous = FlyParams::init_default(&flyg, &config, 1);
        let mut state_continuous = AdamState::new(&params_continuous);

        // Some fixed sequence of "gradients" (any deterministic values will do — this test is
        // about optimiser-state continuity, not gradient correctness, which `tests/
        // backward_correctness.rs` already covers).
        let grads: Vec<ParamGradients> = (0..5)
            .map(|i| ParamGradients {
                a: params_continuous
                    .a
                    .iter()
                    .enumerate()
                    .map(|(j, _)| 0.01 * (i + j) as f32)
                    .collect(),
                b: params_continuous
                    .b
                    .iter()
                    .enumerate()
                    .map(|(j, _)| -0.02 * (i + j) as f32)
                    .collect(),
                theta: params_continuous
                    .theta
                    .iter()
                    .enumerate()
                    .map(|(j, _)| 0.03 * (i + j) as f32)
                    .collect(),
            })
            .collect();

        // Run 3 steps continuously, save a checkpoint, then run 2 more steps.
        for g in &grads[0..3] {
            adam_step(&mut params_continuous, g, &mut state_continuous, &adam_config);
        }
        let ckpt_path = dir.path().join("mid_training.flyckpt");
        save_checkpoint(
            &ckpt_path,
            &flyg_path,
            &config,
            &params_continuous,
            CheckpointMeta::default(),
            Some(&state_continuous),
        )
        .unwrap();
        for g in &grads[3..5] {
            adam_step(&mut params_continuous, g, &mut state_continuous, &adam_config);
        }

        // Now: load the checkpoint fresh and run the same 2 remaining steps.
        let loaded = load_checkpoint_for_flyg(&ckpt_path, &flyg_path).unwrap();
        let mut params_resumed = loaded.params;
        let mut state_resumed = loaded.optimizer.expect("checkpoint was saved with Some(&state)");
        for g in &grads[3..5] {
            adam_step(&mut params_resumed, g, &mut state_resumed, &adam_config);
        }

        assert_eq!(
            params_resumed, params_continuous,
            "resumed params must exactly match the never-interrupted run"
        );
        assert_eq!(
            state_resumed, state_continuous,
            "resumed optimizer state must exactly match the never-interrupted run"
        );
    }
}
