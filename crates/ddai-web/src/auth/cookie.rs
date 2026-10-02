//! The session and trusted-device cookies: encoding/verifying their signed values, and building
//! the `Set-Cookie` a browser actually stores.
//!
//! Acceptance criterion 2 asks for "axum-extra (cookie-private) OR a hand-rolled
//! signed/encrypted cookie" carrying "a random 256-bit id". We take the hand-rolled route: a
//! cookie value is `base64url(id) "." base64url(HMAC-SHA256(key, context || id))`. The id's own
//! 256 bits of entropy already make it unguessable, so the HMAC does not add confidentiality — it
//! adds a *fast, cheap* rejection of a tampered/foreign cookie value before we ever touch the
//! session/device table (a lookup miss on a tampered id would reject just as surely, but
//! verifying the signature first means a probing client can't use timing or table-population side
//! effects to distinguish "malformed" from "well-formed but unknown"). This is what acceptance
//! criterion 2's "session key material (cookie encryption/signing key) generated on first run" is
//! for — see [`crate::secrets::load_or_create_session_key`].
//!
//! The session cookie and the trusted-device cookie (review finding F7) share that one key but
//! are signed with distinct `context` prefixes (`b"session"` / `b"device"`), so a valid value for
//! one can never be replayed as the other even though both are opaque 32-byte ids under the same
//! key — cheap domain separation.
//!
//! Cookie names follow the `__Host-` prefix rules (RFC 6265bis): that prefix requires `Secure`,
//! `Path=/`, and no `Domain` attribute, so we only use it when `secure` is true. Local, plain-HTTP
//! testing runs with `secure = false` and plain cookie names instead, since a real browser
//! silently refuses to store a `Secure` cookie (let alone a `__Host-` one) sent over plain HTTP.

use axum_extra::extract::cookie::{Cookie, SameSite};
use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

use crate::auth::device::DeviceId;
use crate::auth::session::SessionId;
use crate::rand_util::{decode_b64, encode_b64};

type HmacSha256 = Hmac<Sha256>;

const SESSION_CONTEXT: &[u8] = b"session";
const DEVICE_CONTEXT: &[u8] = b"device";
const AUDIT_CONTEXT: &[u8] = b"audit";

pub const SESSION_COOKIE_NAME_SECURE: &str = "__Host-session";
pub const SESSION_COOKIE_NAME_INSECURE: &str = "ddai_session";
pub const DEVICE_COOKIE_NAME_SECURE: &str = "__Host-device";
pub const DEVICE_COOKIE_NAME_INSECURE: &str = "ddai_device";

pub fn cookie_name(secure: bool) -> &'static str {
    if secure {
        SESSION_COOKIE_NAME_SECURE
    } else {
        SESSION_COOKIE_NAME_INSECURE
    }
}

pub fn device_cookie_name(secure: bool) -> &'static str {
    if secure {
        DEVICE_COOKIE_NAME_SECURE
    } else {
        DEVICE_COOKIE_NAME_INSECURE
    }
}

fn hmac_for(key: &[u8], context: &[u8], id: &[u8; 32]) -> HmacSha256 {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC-SHA256 accepts a key of any length");
    mac.update(context);
    mac.update(id);
    mac
}

fn encode_signed(key: &[u8], context: &[u8], id: &[u8; 32]) -> String {
    let sig = hmac_for(key, context, id).finalize().into_bytes();
    format!("{}.{}", encode_b64(id), encode_b64(&sig))
}

fn decode_signed(key: &[u8], context: &[u8], value: &str) -> Option<[u8; 32]> {
    let (id_part, sig_part) = value.split_once('.')?;
    let id_bytes = decode_b64(id_part).ok()?;
    let sig_bytes = decode_b64(sig_part).ok()?;
    let id: [u8; 32] = id_bytes.as_slice().try_into().ok()?;
    hmac_for(key, context, &id).verify_slice(&sig_bytes).ok()?;
    Some(id)
}

/// An opaque, non-reversible tag of a session for the bot's audit log (task 5.6): 8 bytes of
/// `HMAC-SHA256(key, "audit" || id)` as lower-case hex. It is **not** the session id nor the cookie (the bot's log must
/// not be able to impersonate a session), it is stable for a session's life, and different sessions get different
/// tags. Domain-separated from the cookie signatures by its own context.
pub fn audit_tag(key: &[u8], id: &SessionId) -> String {
    let full = hmac_for(key, AUDIT_CONTEXT, id).finalize().into_bytes();
    full[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// Encodes a session id into its signed cookie value.
pub fn encode_session_cookie_value(key: &[u8], id: &SessionId) -> String {
    encode_signed(key, SESSION_CONTEXT, id)
}

/// Verifies and decodes a session cookie value. Returns `None` for anything malformed or whose
/// signature doesn't match — including a value signed under a *different* context (e.g. a device
/// cookie value replayed here) or a different key (e.g. after a key rotation), both intentionally
/// indistinguishable from "not our cookie".
pub fn decode_session_cookie_value(key: &[u8], value: &str) -> Option<SessionId> {
    decode_signed(key, SESSION_CONTEXT, value)
}

/// Encodes a trusted-device id into its signed cookie value.
pub fn encode_device_cookie_value(key: &[u8], id: &DeviceId) -> String {
    encode_signed(key, DEVICE_CONTEXT, id)
}

/// Verifies and decodes a trusted-device cookie value. See
/// [`decode_session_cookie_value`] for what `None` covers.
pub fn decode_device_cookie_value(key: &[u8], value: &str) -> Option<DeviceId> {
    decode_signed(key, DEVICE_CONTEXT, value)
}

fn build_cookie(name: &'static str, secure: bool, value: String, max_age: cookie::time::Duration) -> Cookie<'static> {
    Cookie::build((name, value))
        .path("/")
        .http_only(true)
        .secure(secure)
        .same_site(SameSite::Strict)
        .max_age(max_age)
        .build()
}

/// Builds the `Set-Cookie` header value for a freshly created session. `max_age` should track the
/// session's absolute timeout, so the browser stops sending a cookie whose session can no longer
/// possibly be valid — the server enforces the real (idle + absolute) expiry independently.
pub fn build_session_cookie(secure: bool, value: String, max_age: cookie::time::Duration) -> Cookie<'static> {
    build_cookie(cookie_name(secure), secure, value, max_age)
}

/// Builds the `Set-Cookie` header value for a (re)confirmed trusted device (review finding F7).
pub fn build_device_cookie(secure: bool, value: String, max_age: cookie::time::Duration) -> Cookie<'static> {
    build_cookie(device_cookie_name(secure), secure, value, max_age)
}

/// Builds a `Set-Cookie` that instructs the browser to delete the session cookie (logout). Must
/// match the original cookie's `Path`/`Secure`/`SameSite` for the browser to actually overwrite
/// (rather than add a second, distinct) it.
pub fn build_removal_cookie(secure: bool) -> Cookie<'static> {
    build_cookie(cookie_name(secure), secure, String::new(), cookie::time::Duration::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rand_util::random_bytes;

    #[test]
    fn encode_decode_roundtrip() {
        let key = random_bytes::<32>();
        let id: SessionId = random_bytes::<32>();
        let value = encode_session_cookie_value(&key, &id);
        let decoded = decode_session_cookie_value(&key, &value).expect("should decode");
        assert_eq!(decoded, id);
    }

    #[test]
    fn the_audit_tag_is_stable_per_session_valid_for_the_bot_and_not_the_cookie() {
        let key = random_bytes::<32>();
        let a: SessionId = random_bytes::<32>();
        let b: SessionId = random_bytes::<32>();
        let tag = audit_tag(&key, &a);
        assert_eq!(tag, audit_tag(&key, &a), "stable");
        assert_ne!(tag, audit_tag(&key, &b), "one per session");
        assert_ne!(tag, audit_tag(&random_bytes::<32>(), &a), "keyed");
        assert!(ddai_botctl::proto::valid_session_tag(&tag), "{tag}");
        assert_eq!(tag.len(), 16);
        // It reveals nothing of the id or the signed cookie value.
        let cookie = encode_session_cookie_value(&key, &a);
        assert!(!cookie.contains(&tag) && !encode_b64(&a).contains(&tag));
    }

    #[test]
    fn device_encode_decode_roundtrip() {
        let key = random_bytes::<32>();
        let id: DeviceId = random_bytes::<32>();
        let value = encode_device_cookie_value(&key, &id);
        let decoded = decode_device_cookie_value(&key, &value).expect("should decode");
        assert_eq!(decoded, id);
    }

    #[test]
    fn session_value_does_not_decode_as_a_device_value_and_vice_versa() {
        let key = random_bytes::<32>();
        let id: [u8; 32] = random_bytes::<32>();
        let session_value = encode_session_cookie_value(&key, &id);
        let device_value = encode_device_cookie_value(&key, &id);
        assert_ne!(
            session_value, device_value,
            "context separation should change the signature"
        );
        assert!(decode_device_cookie_value(&key, &session_value).is_none());
        assert!(decode_session_cookie_value(&key, &device_value).is_none());
    }

    #[test]
    fn tampered_id_is_rejected() {
        let key = random_bytes::<32>();
        let id: SessionId = random_bytes::<32>();
        let value = encode_session_cookie_value(&key, &id);
        let (_, sig_part) = value.split_once('.').unwrap();
        let other_id: SessionId = random_bytes::<32>();
        let tampered = format!("{}.{sig_part}", encode_b64(&other_id));
        assert!(decode_session_cookie_value(&key, &tampered).is_none());
    }

    #[test]
    fn tampered_signature_is_rejected() {
        let key = random_bytes::<32>();
        let id: SessionId = random_bytes::<32>();
        let value = encode_session_cookie_value(&key, &id);
        let (id_part, _) = value.split_once('.').unwrap();
        let bogus_sig = encode_b64(&random_bytes::<32>());
        let tampered = format!("{id_part}.{bogus_sig}");
        assert!(decode_session_cookie_value(&key, &tampered).is_none());
    }

    #[test]
    fn wrong_key_is_rejected() {
        let key_a = random_bytes::<32>();
        let key_b = random_bytes::<32>();
        let id: SessionId = random_bytes::<32>();
        let value = encode_session_cookie_value(&key_a, &id);
        assert!(decode_session_cookie_value(&key_b, &value).is_none());
    }

    #[test]
    fn malformed_values_are_rejected() {
        let key = random_bytes::<32>();
        assert!(decode_session_cookie_value(&key, "").is_none());
        assert!(decode_session_cookie_value(&key, "no-dot-here").is_none());
        assert!(decode_session_cookie_value(&key, "not base64!.also not base64!").is_none());
        assert!(decode_session_cookie_value(&key, &format!("{}.", encode_b64(&[1, 2, 3]))).is_none());
    }

    #[test]
    fn cookie_name_follows_host_prefix_rules() {
        assert_eq!(cookie_name(true), "__Host-session");
        assert!(!cookie_name(false).starts_with("__Host-"));
        assert_eq!(device_cookie_name(true), "__Host-device");
        assert!(!device_cookie_name(false).starts_with("__Host-"));
    }

    #[test]
    fn built_cookie_has_expected_attributes() {
        let cookie = build_session_cookie(true, "abc".to_string(), cookie::time::Duration::days(7));
        assert_eq!(cookie.name(), "__Host-session");
        assert_eq!(cookie.path(), Some("/"));
        assert_eq!(cookie.http_only(), Some(true));
        assert_eq!(cookie.secure(), Some(true));
        assert_eq!(cookie.same_site(), Some(SameSite::Strict));
    }

    #[test]
    fn built_device_cookie_has_its_own_name() {
        let cookie = build_device_cookie(false, "abc".to_string(), cookie::time::Duration::days(90));
        assert_eq!(cookie.name(), "ddai_device");
        assert_eq!(cookie.http_only(), Some(true));
        assert_eq!(cookie.same_site(), Some(SameSite::Strict));
    }

    #[test]
    fn removal_cookie_has_zero_max_age() {
        let cookie = build_removal_cookie(false);
        assert_eq!(cookie.max_age(), Some(cookie::time::Duration::ZERO));
        assert_eq!(cookie.value(), "");
    }
}
