//! Hashing used by the trace format and by fixture/scenario integrity checks.
//!
//! Two unrelated hashes, for two unrelated jobs: **FNV-1a 64** is the *canonical per-tick state
//! hash* (see `docs/formats.md`) — small, fast, and specified precisely enough that the C++
//! oracle computes the exact same value byte-for-byte, so it is safe to use as the thing Rust
//! physics (task 1.3) is checked against; it is hand-written here (below) since it needs to be
//! pinned exactly, not "whatever some crate's version happens to do". **SHA-256** is only used
//! as a content-integrity check (map/scenario file identity, golden fixture pinning) — it is
//! never compared against a value computed by different code, so this side deliberately uses
//! the well-audited `sha2` crate (RustCrypto) instead of a from-scratch implementation (review
//! round 1, finding F7); the C++ oracle keeps its own small hand-written SHA-256
//! (`tools/ddnet-oracle/sha256.h`, cross-checked against the same FIPS 180-4 vectors this
//! module's tests use) since pulling in an external crypto library for a single build script is
//! not worth it there.

/// FNV-1a 64 hash, run over an arbitrary byte stream.
///
/// This is the FNV-1a variant (XOR-then-multiply, not multiply-then-XOR) with the standard
/// 64-bit offset basis and prime. See <http://www.isthe.com/chongo/tech/comp/fnv/> for the
/// reference definition; `docs/formats.md` pins the exact byte layout callers must feed it for
/// the canonical per-tick state hash.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET_BASIS;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// A streaming FNV-1a 64 accumulator, for hashing a value field-by-field without building an
/// intermediate `Vec<u8>` (used to compute the canonical per-tick state hash across many
/// characters' worth of fields).
#[derive(Debug, Clone)]
pub struct Fnv1a64 {
    hash: u64,
}

impl Default for Fnv1a64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv1a64 {
    pub fn new() -> Self {
        Fnv1a64 {
            hash: 0xcbf29ce484222325,
        }
    }

    pub fn update(&mut self, bytes: &[u8]) -> &mut Self {
        const PRIME: u64 = 0x100000001b3;
        for &b in bytes {
            self.hash ^= b as u64;
            self.hash = self.hash.wrapping_mul(PRIME);
        }
        self
    }

    pub fn finish(&self) -> u64 {
        self.hash
    }
}

/// SHA-256 of `bytes`, as a raw 32-byte digest. Delegates to the `sha2` crate (RustCrypto) —
/// see the module doc comment for why this side uses a crate while the C++ oracle doesn't.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Renders a digest as lowercase hex, e.g. for embedding in JSON metadata or file names.
pub fn to_hex(digest: &[u8]) -> String {
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_matches_known_test_vector() {
        // Reference vectors from http://www.isthe.com/chongo/src/fnv/test_fnv.c ("" and "a").
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
    }

    #[test]
    fn streaming_matches_one_shot() {
        let data = b"hello, ddnet";
        let one_shot = fnv1a64(data);
        let mut streamed = Fnv1a64::new();
        streamed.update(&data[..5]).update(&data[5..]);
        assert_eq!(streamed.finish(), one_shot);
    }

    #[test]
    fn sha256_matches_known_test_vector() {
        // SHA-256("abc") from FIPS 180-4.
        let digest = sha256(b"abc");
        assert_eq!(
            to_hex(&digest),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn empty_input_matches_known_digest() {
        let digest = sha256(b"");
        assert_eq!(
            to_hex(&digest),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn multi_block_input_matches_known_digest() {
        // FIPS 180-4 test vector: one million repetitions of 'a'. 1_000_000 / 64 = 15_625 exactly
        // (no remainder), so this exercises the full-block loop at scale (15,625 iterations) plus
        // a padding block that is entirely new content (0x80 then zeros then the length) rather
        // than a mix of real trailing bytes and padding, unlike every other test here.
        let data = vec![b'a'; 1_000_000];
        let digest = sha256(&data);
        assert_eq!(
            to_hex(&digest),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn one_byte_short_of_a_block_boundary_pads_into_a_second_block() {
        // 63 bytes + the mandatory 0x80 byte = 64, leaving no room for the 8-byte length in the
        // same block as the padding start — must spill into a second block. Digest computed
        // independently with Python's hashlib for this exact input.
        let data = vec![0u8; 63];
        let digest = sha256(&data);
        assert_eq!(
            to_hex(&digest),
            "c7723fa1e0127975e49e62e753db53924c1bd84b8ac1ac08df78d09270f3d971"
        );
    }
}
