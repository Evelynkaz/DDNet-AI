//! Hand-rolled error types, no `anyhow`/`thiserror` — same rationale as `ddai-flyg`: this crate is
//! meant to be linked by the live bot, so it stays light. See `ddai_flyg::error` for the sibling
//! convention this mirrors.

use std::fmt;

/// Everything that can go wrong constructing a [`crate::model::FlyModel`], applying a
/// [`crate::params::FlyParams`] to one, or saving/loading a [`crate::checkpoint::Checkpoint`].
#[derive(Debug)]
pub enum FlyError {
    /// A [`crate::params::FlyParams`] array's length doesn't match the graph it's being applied
    /// to (e.g. `a.len() != flyg.summary.shared_param_count`). Carries a human-readable
    /// explanation rather than a structured variant per mismatch — there are several independent
    /// length checks and none of them need to be matched on separately by callers.
    ParamShapeMismatch(String),
    /// A [`crate::config::FlyConfig`] field is out of its valid range (e.g. `substeps_per_decision
    /// == 0`).
    InvalidConfig(String),
    Io(std::io::Error),
    /// zstd (de)compression failed. zstd's own API returns plain `io::Error` for this, so this
    /// variant exists only to attach a clearer message at the call site (mirrors
    /// `ddai_flyg::FlygError::Zstd`).
    Zstd(std::io::Error),
    Decode(postcard::Error),
    Encode(postcard::Error),
    CheckpointFormatVersionMismatch {
        found: u32,
        expected: u32,
    },
    /// The checkpoint's `flyg_sha256` doesn't match the `.flyg` file it's being loaded against —
    /// the whole point of storing that hash (acceptance criterion 1e): loading a checkpoint
    /// trained against a different graph fails clearly instead of silently indexing params that
    /// mean something else.
    FlygMismatch {
        expected: String,
        found: String,
    },
    /// The checkpoint file is shorter than the fixed-size payload-hash prefix (review round 1,
    /// F2/F7's "on-disk layout") — always corrupt/truncated, never a valid checkpoint.
    Truncated {
        len: usize,
        min_len: usize,
    },
    /// The sha256 stored in the checkpoint file doesn't match the sha256 of its (successfully
    /// zstd-decompressed) postcard payload — the file is corrupt (review round 1, F2: a bit-flip
    /// fuzz test on a real checkpoint found this catches essentially every single-bit flip that
    /// zstd's own content checksum, checked earlier during decompression, didn't already reject).
    ChecksumMismatch,
    /// The checkpoint's `optimizer` (`Some`) doesn't shape-match its own `params` (review round 1,
    /// F7d): [`crate::optim::AdamState::matches_shape`] said no. Checked once, right in
    /// [`crate::checkpoint::load_checkpoint`], rather than leaving every caller to remember to
    /// call `matches_shape` itself before trusting a loaded checkpoint's optimiser state (the
    /// `flyg_sha256` check already rules out a mismatched *graph*, but a hand-edited or corrupted
    /// checkpoint could still have `params` and `optimizer` individually well-formed yet
    /// disagree with each other in length).
    OptimizerShapeMismatch(String),
}

impl fmt::Display for FlyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlyError::ParamShapeMismatch(msg) => write!(f, "params do not match graph shape: {msg}"),
            FlyError::InvalidConfig(msg) => write!(f, "invalid FlyConfig: {msg}"),
            FlyError::Io(e) => write!(f, "I/O error: {e}"),
            FlyError::Zstd(e) => write!(f, "zstd (de)compression failed: {e}"),
            FlyError::Decode(e) => write!(f, "failed to decode checkpoint contents (postcard): {e}"),
            FlyError::Encode(e) => write!(f, "failed to encode checkpoint contents (postcard): {e}"),
            FlyError::CheckpointFormatVersionMismatch { found, expected } => {
                write!(f, "checkpoint format version {found} != expected {expected}")
            }
            FlyError::FlygMismatch { expected, found } => write!(
                f,
                "checkpoint was trained against a different .flyg: expected sha256 {expected}, this graph hashes to {found}"
            ),
            FlyError::Truncated { len, min_len } => {
                write!(f, "checkpoint file is truncated: {len} bytes, need at least {min_len}")
            }
            FlyError::ChecksumMismatch => write!(f, "checkpoint payload hash does not match — file is corrupted"),
            FlyError::OptimizerShapeMismatch(msg) => {
                write!(f, "checkpoint's optimizer state does not match its own params: {msg}")
            }
        }
    }
}

impl std::error::Error for FlyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FlyError::Io(e) => Some(e),
            FlyError::Zstd(e) => Some(e),
            FlyError::Decode(e) | FlyError::Encode(e) => Some(e),
            _ => None,
        }
    }
}
