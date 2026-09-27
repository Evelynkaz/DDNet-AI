//! Parsing GCS's `x-goog-hash` response header, e.g. `crc32c=vjz9cg==, md5=UKdxh3DFciDxYLpPQxq4ng==`.
//!
//! GCS may send this as one header with comma-separated `key=base64` pairs, or as several
//! repeated headers each holding one pair (both forms are handled here; see
//! `~/aiddnet/data/connectome/samples/*.headers` for real examples of the first form).

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;

use crate::hashing::to_hex;

/// Extracts the MD5 checksum from one or more `x-goog-hash` header values, returned as
/// lowercase hex. Returns `None` if no `md5=...` pair is present or it fails to base64-decode.
pub fn parse_goog_md5_hex<'a>(header_values: impl IntoIterator<Item = &'a str>) -> Option<String> {
    for value in header_values {
        for part in value.split(',') {
            let part = part.trim();
            if let Some(b64) = part.strip_prefix("md5=")
                && let Ok(bytes) = BASE64.decode(b64)
            {
                return Some(to_hex(&bytes));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real header line captured for `body-annotations-male-cns-v1.0-minconf-0.5.feather` in
    /// `~/aiddnet/data/connectome/samples/body-annotations-male-cns-v1.0-minconf-0.5.feather.headers`
    /// (single header, both hashes comma-separated). Expected hex cross-checked against the GCS
    /// JSON API's `md5Hash` for the same object.
    const ANNOTATIONS_HEADER: &str = "crc32c=vjz9cg==, md5=UKdxh3DFciDxYLpPQxq4ng==";
    const ANNOTATIONS_MD5_HEX: &str = "50a7718770c57220f160ba4f431ab89e";

    const NEUROTRANSMITTERS_HEADER: &str = "crc32c=jcpNFg==, md5=PYQrEv5cSe763lKNfdJKHw==";
    const NEUROTRANSMITTERS_MD5_HEX: &str = "3d842b12fe5c49eefade528d7dd24a1f";

    #[test]
    fn parses_single_header_with_both_hashes() {
        assert_eq!(
            parse_goog_md5_hex([ANNOTATIONS_HEADER]),
            Some(ANNOTATIONS_MD5_HEX.to_string())
        );
        assert_eq!(
            parse_goog_md5_hex([NEUROTRANSMITTERS_HEADER]),
            Some(NEUROTRANSMITTERS_MD5_HEX.to_string())
        );
    }

    #[test]
    fn parses_repeated_single_pair_headers() {
        // The same information, but as two separate `x-goog-hash` header lines (as real HTTP/2
        // responses from GCS send them — see the `.headers` fixture files), each with one pair.
        let values = ["crc32c=vjz9cg==", "md5=UKdxh3DFciDxYLpPQxq4ng=="];
        assert_eq!(parse_goog_md5_hex(values), Some(ANNOTATIONS_MD5_HEX.to_string()));
    }

    #[test]
    fn returns_none_without_md5_pair() {
        assert_eq!(parse_goog_md5_hex(["crc32c=vjz9cg=="]), None);
        assert_eq!(parse_goog_md5_hex(Vec::<&str>::new()), None);
    }

    #[test]
    fn returns_none_for_invalid_base64() {
        assert_eq!(parse_goog_md5_hex(["md5=not-valid-base64!!"]), None);
    }
}
