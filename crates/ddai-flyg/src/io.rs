//! Save/load a [`Flyg`] as postcard + zstd, the same on-disk convention as
//! `ddai_connectome::tables` (see that crate's `save_tables`/`load_tables`): a single file,
//! written to a sibling temp file and renamed into place so a reader never sees a half-written
//! file and a failed write never clobbers a good one.

use std::path::Path;

use crate::error::FlygError;
use crate::format::{FLYG_FORMAT_VERSION, Flyg};
use crate::validate::validate;

/// Serializes `flyg` (postcard, then zstd level 3 — datasets, not archives) and writes it
/// atomically to `path`. Does **not** validate `flyg` first — callers that build a `Flyg` from
/// scratch should call [`validate`] themselves before saving (the builder in `ddai-connectome`
/// does); this function's job is only the encode/write, so [`load`] round-trips whatever bytes
/// were actually asked for, corrupted input included, for the corruption-rejection tests.
pub fn save(flyg: &Flyg, path: &Path) -> Result<(), FlygError> {
    let bytes = postcard::to_allocvec(flyg).map_err(FlygError::Encode)?;
    let compressed = zstd::stream::encode_all(bytes.as_slice(), 3).map_err(FlygError::Zstd)?;
    let tmp_path = path.with_extension("flyg.tmp");
    std::fs::write(&tmp_path, compressed).map_err(FlygError::Io)?;
    std::fs::rename(&tmp_path, path).map_err(FlygError::Io)?;
    Ok(())
}

/// Inverse of [`save`]: reads, zstd-decompresses, postcard-decodes, checks
/// `header.format_version`, then runs [`validate`] — a `.flyg` file that decodes fine but fails
/// structural validation (out-of-range index, malformed CSR, NaN, …) is rejected here too, not
/// just when a caller happens to call `validate` separately.
pub fn load(path: &Path) -> Result<Flyg, FlygError> {
    let compressed = std::fs::read(path).map_err(FlygError::Io)?;
    let bytes = zstd::stream::decode_all(compressed.as_slice()).map_err(FlygError::Zstd)?;
    let flyg: Flyg = postcard::from_bytes(&bytes).map_err(FlygError::Decode)?;
    if flyg.header.format_version != FLYG_FORMAT_VERSION {
        return Err(FlygError::FormatVersionMismatch {
            found: flyg.header.format_version,
            expected: FLYG_FORMAT_VERSION,
        });
    }
    validate(&flyg)?;
    Ok(flyg)
}
