//! The fly's visualisation stream as the web unit sees it (task 7.4, `docs/formats.md` §27): what a
//! `FLYMETA` or `FLY` message of the bot bridge must look like before it is let into the hub and on to
//! browsers. The bytes come from a `0600` socket of the same user, but they are still checked: the
//! web never forwards a frame it has not measured.
//!
//! The frame layout is the bot side's (`ddai_fly::viz`, 44-byte header and a body whose size the header's
//! counts give); this crate does not depend on the fly and re-implements only the length check, with its own
//! golden test of the documented layout.

/// `DFLY` v1.
pub const MAGIC: &[u8; 4] = b"DFLY";
pub const VERSION: u8 = 1;
/// Bytes before the body.
pub const HEADER_LEN: usize = 44;
/// Largest frame let through (a real one is ~1.7 KB).
pub const MAX_FRAME: usize = 64 * 1024;
/// Largest layout description let through (a real one is ~5 KB).
pub const MAX_META: usize = 64 * 1024;

/// Checks magic, version and that the length is what the counts of the header say:
/// `44 + groups + dn + channels * rays * bins + scalars`.
pub fn validate_frame(b: &[u8]) -> Result<(), String> {
    if b.len() < HEADER_LEN {
        return Err(format!("a fly frame of {} bytes is shorter than its header", b.len()));
    }
    if b.len() > MAX_FRAME {
        return Err(format!("a fly frame of {} bytes is too large", b.len()));
    }
    if &b[0..4] != MAGIC {
        return Err("a fly frame with a bad magic".to_string());
    }
    if b[4] != VERSION {
        return Err(format!("a fly frame of unsupported version {}", b[4]));
    }
    let u16_at = |i: usize| usize::from(u16::from_le_bytes([b[i], b[i + 1]]));
    let want = HEADER_LEN
        + u16_at(34)
        + u16_at(36)
        + usize::from(b[41]) * u16_at(38) * usize::from(b[40])
        + usize::from(b[42]);
    if b.len() != want {
        return Err(format!("a fly frame of {} bytes where its counts say {want}", b.len()));
    }
    Ok(())
}

/// A `FLYMETA` payload: empty means "this brain has no stream" (`None`); otherwise one JSON object of
/// at most [`MAX_META`] bytes with `"v": 1`.
pub fn validate_meta(payload: &[u8]) -> Result<Option<String>, String> {
    if payload.is_empty() {
        return Ok(None);
    }
    if payload.len() > MAX_META {
        return Err(format!("a fly layout of {} bytes is too large", payload.len()));
    }
    let value: serde_json::Value =
        serde_json::from_slice(payload).map_err(|e| format!("the fly layout is not JSON: {e}"))?;
    if !value.is_object() || value["v"] != 1 {
        return Err("the fly layout is not a version 1 object".to_string());
    }
    String::from_utf8(payload.to_vec()).map(Some).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame with 2 groups, 3 DN, 4 rays x 2 bins x 7 channels and 3 scalars: the documented layout.
    pub(crate) fn sample_frame() -> Vec<u8> {
        let (groups, dn, rays, bins, channels, scalars) = (2usize, 3usize, 4usize, 2usize, 7usize, 3usize);
        let mut f = vec![0u8; HEADER_LEN + groups + dn + channels * rays * bins + scalars];
        f[0..4].copy_from_slice(b"DFLY");
        f[4] = 1;
        f[34..36].copy_from_slice(&(groups as u16).to_le_bytes());
        f[36..38].copy_from_slice(&(dn as u16).to_le_bytes());
        f[38..40].copy_from_slice(&(rays as u16).to_le_bytes());
        f[40] = bins as u8;
        f[41] = channels as u8;
        f[42] = scalars as u8;
        f
    }

    #[test]
    fn a_frame_of_the_documented_layout_passes_and_every_deviation_does_not() {
        let f = sample_frame();
        assert_eq!(f.len(), 44 + 2 + 3 + 56 + 3);
        assert_eq!(validate_frame(&f), Ok(()));
        assert!(validate_frame(&f[..43]).is_err(), "shorter than the header");
        assert!(validate_frame(&f[..f.len() - 1]).is_err(), "truncated body");
        let mut long = f.clone();
        long.push(0);
        assert!(validate_frame(&long).is_err(), "extra byte");
        let mut bad = f.clone();
        bad[0] = b'W';
        assert!(validate_frame(&bad).unwrap_err().contains("magic"));
        let mut bad = f.clone();
        bad[4] = 2;
        assert!(validate_frame(&bad).unwrap_err().contains("version"));
        // Counts that claim far more than the bytes carry cannot make a huge allocation or a panic.
        let mut bad = f;
        bad[34..36].copy_from_slice(&u16::MAX.to_le_bytes());
        bad[38..40].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(validate_frame(&bad).is_err());
        assert!(validate_frame(&vec![b'D'; MAX_FRAME + 1]).is_err());
    }

    #[test]
    fn the_layout_must_be_a_small_version_1_json_object_or_empty() {
        assert_eq!(validate_meta(b""), Ok(None));
        assert!(validate_meta(br#"{"v":1,"rays":48}"#).unwrap().is_some());
        assert!(validate_meta(br#"{"v":2}"#).is_err());
        assert!(validate_meta(br#"[1]"#).is_err());
        assert!(validate_meta(b"not json").is_err());
        assert!(validate_meta(&vec![b' '; MAX_META + 1]).is_err());
    }
}
