//! One file format for datasets and model bundles: `sha256(postcard bytes) ++ zstd(postcard bytes)`, written atomically
//! (the same discipline as `ddai_fly::bundle`, without its dependencies).

use std::io::Write;
use std::path::Path;

use serde::Serialize;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

const HASH_LEN: usize = 32;

/// Lower-case hex of the sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Writes `value` to `path` (sibling temp file + rename).
pub fn write_blob<T: Serialize>(path: &Path, value: &T, level: i32) -> Result<(), String> {
    let bytes = postcard::to_allocvec(value).map_err(|e| format!("encoding: {e}"))?;
    let hash = Sha256::digest(&bytes);
    let mut compressed = Vec::new();
    {
        let mut enc = zstd::stream::Encoder::new(&mut compressed, level).map_err(|e| format!("zstd: {e}"))?;
        enc.include_checksum(true).map_err(|e| format!("zstd: {e}"))?;
        enc.write_all(&bytes).map_err(|e| format!("zstd: {e}"))?;
        enc.finish().map_err(|e| format!("zstd: {e}"))?;
    }
    let mut out = Vec::with_capacity(HASH_LEN + compressed.len());
    out.extend_from_slice(&hash);
    out.extend_from_slice(&compressed);
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let tmp = path.with_file_name(format!("{name}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, &out).map_err(|e| format!("writing {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("renaming to {}: {e}", path.display()))
}

/// Reads what [`write_blob`] wrote, verifying the hash.
pub fn read_blob<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let all = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if all.len() < HASH_LEN {
        return Err(format!("{}: truncated ({} bytes)", path.display(), all.len()));
    }
    let (stored, compressed) = all.split_at(HASH_LEN);
    let bytes = zstd::stream::decode_all(compressed).map_err(|e| format!("{}: zstd: {e}", path.display()))?;
    if Sha256::digest(&bytes).as_slice() != stored {
        return Err(format!("{}: payload hash mismatch (corrupt file)", path.display()));
    }
    postcard::from_bytes(&bytes).map_err(|e| format!("{}: decoding: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blob_round_trips_and_a_flipped_byte_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.blob");
        let v: Vec<(u32, String)> = vec![(1, "a".into()), (2, "b".into())];
        write_blob(&p, &v, 3).unwrap();
        assert_eq!(read_blob::<Vec<(u32, String)>>(&p).unwrap(), v);
        let mut raw = std::fs::read(&p).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xff;
        std::fs::write(&p, raw).unwrap();
        assert!(read_blob::<Vec<(u32, String)>>(&p).is_err());
    }
}
