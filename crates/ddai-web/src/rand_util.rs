//! Small helpers around [`getrandom`] and base64url encoding used throughout this crate for
//! session ids, CSRF tokens, and key material. All randomness in this crate goes through
//! [`random_bytes`], which reads from the OS CSPRNG (`getrandom`) — never a seeded/deterministic
//! RNG, unlike the rest of the workspace's ML code which needs reproducible seeds.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;

/// Fills and returns an `N`-byte array of OS-random bytes.
///
/// # Panics
/// Panics if the OS RNG is unavailable. `getrandom` failing means the process cannot safely
/// generate session ids, passwords, or key material at all; there is no sane fallback, so we
/// fail loudly and immediately rather than risk predictable "random" values.
pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut buf = [0u8; N];
    getrandom::fill(&mut buf).expect("OS RNG (getrandom) must be available");
    buf
}

pub fn encode_b64(bytes: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

pub fn decode_b64(s: &str) -> Result<Vec<u8>, base64::DecodeError> {
    URL_SAFE_NO_PAD.decode(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_bytes_are_not_all_zero_and_vary() {
        let a: [u8; 32] = random_bytes();
        let b: [u8; 32] = random_bytes();
        assert_ne!(a, [0u8; 32], "getrandom returned an all-zero buffer");
        assert_ne!(a, b, "two consecutive calls returned identical bytes");
    }

    #[test]
    fn b64_roundtrips() {
        let bytes = random_bytes::<32>();
        let encoded = encode_b64(&bytes);
        let decoded = decode_b64(&encoded).expect("decode");
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn b64_rejects_garbage() {
        assert!(decode_b64("not valid base64!!").is_err());
    }
}
