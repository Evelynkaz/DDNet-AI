//! CSRF token comparison for state-changing routes (acceptance criterion 4): the client echoes
//! the token it received at login back in the `X-CSRF-Token` header, and we compare it to the
//! one stored in that session's server-side record.

use subtle::ConstantTimeEq;

use crate::auth::session::CsrfToken;
use crate::rand_util::decode_b64;

pub const CSRF_HEADER_NAME: &str = "x-csrf-token";

/// Decodes a base64url CSRF token header value and compares it in constant time against the
/// session's real token. Any decoding failure (wrong length, invalid base64) is treated as a
/// mismatch, not an error — from the caller's point of view a malformed token is just as
/// unauthorized as a wrong one.
pub fn matches(expected: &CsrfToken, header_value: &str) -> bool {
    let Ok(provided) = decode_b64(header_value) else {
        return false;
    };
    if provided.len() != expected.len() {
        return false;
    }
    bool::from(expected.ct_eq(&provided))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rand_util::{encode_b64, random_bytes};

    #[test]
    fn matching_token_is_accepted() {
        let token: CsrfToken = random_bytes();
        assert!(matches(&token, &encode_b64(&token)));
    }

    #[test]
    fn wrong_token_is_rejected() {
        let token: CsrfToken = random_bytes();
        let other: CsrfToken = random_bytes();
        assert!(!matches(&token, &encode_b64(&other)));
    }

    #[test]
    fn wrong_length_is_rejected_not_panicking() {
        let token: CsrfToken = random_bytes();
        assert!(!matches(&token, &encode_b64(b"short")));
        assert!(!matches(&token, ""));
    }

    #[test]
    fn invalid_base64_is_rejected() {
        let token: CsrfToken = random_bytes();
        assert!(!matches(&token, "not valid base64!!"));
    }
}
