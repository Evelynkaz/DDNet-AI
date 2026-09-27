//! Hand-rolled error types (this crate deliberately has no `anyhow`/`thiserror` dependency — see
//! the crate-level docs and `Cargo.toml`: the format crate the fly model links against should
//! stay minimal).

use std::fmt;

/// A single validation failure, with enough context to find the offending row without re-running
/// validation under a debugger. [`crate::validate::validate`] returns the *first* one it finds
/// (validation stops there — see that function's docs for why continuing would not be
/// meaningfully more helpful).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError(pub String);

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid .flyg data: {}", self.0)
    }
}

impl std::error::Error for ValidationError {}

/// Everything that can go wrong loading or saving a `.flyg` file.
#[derive(Debug)]
pub enum FlygError {
    Io(std::io::Error),
    /// zstd (de)compression failed. zstd's own API returns plain `io::Error` for this, so this
    /// variant exists only to attach a clearer message at the call site (see `io.rs`).
    Zstd(std::io::Error),
    Decode(postcard::Error),
    Encode(postcard::Error),
    FormatVersionMismatch {
        found: u32,
        expected: u32,
    },
    Validation(ValidationError),
}

impl fmt::Display for FlygError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlygError::Io(e) => write!(f, "I/O error: {e}"),
            FlygError::Zstd(e) => write!(f, "zstd (de)compression failed: {e}"),
            FlygError::Decode(e) => write!(f, "failed to decode .flyg contents (postcard): {e}"),
            FlygError::Encode(e) => write!(f, "failed to encode .flyg contents (postcard): {e}"),
            FlygError::FormatVersionMismatch { found, expected } => write!(
                f,
                ".flyg format version {found} != expected {expected} — rebuild with `build-subgraph`"
            ),
            FlygError::Validation(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for FlygError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FlygError::Io(e) => Some(e),
            FlygError::Zstd(e) => Some(e),
            FlygError::Decode(e) | FlygError::Encode(e) => Some(e),
            FlygError::FormatVersionMismatch { .. } => None,
            FlygError::Validation(e) => Some(e),
        }
    }
}

impl From<ValidationError> for FlygError {
    fn from(e: ValidationError) -> Self {
        FlygError::Validation(e)
    }
}
