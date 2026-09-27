//! `POST /api/login`, `POST /api/logout`, `GET /api/me` (acceptance criterion 3, 4).

use axum::Json;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum_extra::extract::cookie::CookieJar;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

use crate::auth::cookie::{
    build_device_cookie, build_removal_cookie, build_session_cookie, decode_device_cookie_value, device_cookie_name,
    encode_device_cookie_value, encode_session_cookie_value,
};
use crate::auth::csrf;
use crate::auth::device::password_fingerprint;
use crate::origin::client_ip;
use crate::rand_util::{encode_b64, random_bytes};
use crate::secrets;
use crate::session_guard::{current_session, request_is_same_origin};
use crate::state::SharedState;

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: &'static str,
}

fn json_error(status: StatusCode, error: &'static str) -> Response {
    (status, Json(ErrorBody { error })).into_response()
}

fn forwarded_for(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::HeaderName::from_static("x-forwarded-for"))?
        .to_str()
        .ok()
}

fn peer_ip(state: &SharedState, connect_info: SocketAddr, headers: &HeaderMap) -> std::net::IpAddr {
    client_ip(connect_info.ip(), state.config.trust_proxy, forwarded_for(headers))
}

fn duration_to_cookie_max_age(d: std::time::Duration) -> cookie::time::Duration {
    cookie::time::Duration::seconds(d.as_secs().min(i64::MAX as u64) as i64)
}

// ---------------------------------------------------------------------------------------------
// POST /api/login
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct LoginJsonBody {
    password: String,
}

#[derive(Debug, Serialize)]
struct LoginOkBody {
    ok: bool,
    csrf_token: String,
}

/// Parses the password out of a JSON or form-encoded body (acceptance criterion 3: "JSON or
/// form"). Anything else (wrong content type, malformed body) is `None`.
fn extract_password(headers: &HeaderMap, body: &[u8]) -> Option<String> {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if content_type.starts_with("application/json") {
        serde_json::from_slice::<LoginJsonBody>(body).ok().map(|b| b.password)
    } else if content_type.starts_with("application/x-www-form-urlencoded") {
        let text = std::str::from_utf8(body).ok()?;
        parse_form_field(text, "password")
    } else {
        None
    }
}

fn parse_form_field(body: &str, field: &str) -> Option<String> {
    body.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (percent_decode(k) == field).then(|| percent_decode(v))
    })
}

/// Minimal `application/x-www-form-urlencoded` value decoder: `+` -> space, `%XX` -> byte. We
/// only ever need this for our own login form's single `password` field, so a full
/// `form_urlencoded`-crate dependency would be disproportionate.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
                match hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    Some(byte) => {
                        out.push(byte);
                        i += 3;
                    }
                    None => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub async fn login(
    State(state): State<SharedState>,
    ConnectInfo(connect_info): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    request_jar: CookieJar,
    body: axum::body::Bytes,
) -> Response {
    if !request_is_same_origin(&headers) {
        return json_error(StatusCode::FORBIDDEN, "cross_origin");
    }

    let ip = peer_ip(&state, connect_info, &headers);

    // Loaded up front (a cheap file read) so a trusted-device cookie can be checked against the
    // *current* password hash before we even touch the rate limiter (review finding F7), and
    // reused below for the actual password verification — one read, two uses.
    let auth = match secrets::load_password_auth(&state.secrets_paths) {
        Ok(auth) => auth,
        Err(error) => {
            tracing::error!(%ip, %error, "failed to load password auth");
            return json_error(StatusCode::INTERNAL_SERVER_ERROR, "internal_error");
        }
    };

    let presented_device_id = request_jar
        .get(device_cookie_name(state.config.cookie_secure))
        .and_then(|c| decode_device_cookie_value(&state.session_key, c.value()));
    // Review round 2, finding F9c: store/compare a keyed fingerprint of the hash, never the raw
    // PHC string — computed once here and reused below (the underlying `auth.hash_phc` doesn't
    // change between now and the `confirm` call further down).
    let current_fingerprint = auth
        .as_ref()
        .map(|auth| password_fingerprint(&state.session_key, &auth.hash_phc));
    let bypass_global = match (&current_fingerprint, presented_device_id) {
        (Some(fingerprint), Some(id)) => state.devices.is_trusted(&id, fingerprint),
        _ => false,
    };

    if let Err(limited) = state.login_rate_limiter.check(ip, bypass_global) {
        let mut response = json_error(StatusCode::TOO_MANY_REQUESTS, "rate_limited");
        if let Ok(value) = header::HeaderValue::from_str(&limited.retry_after.as_secs().max(1).to_string()) {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
        return response;
    }

    let Some(password) = extract_password(&headers, &body) else {
        return json_error(StatusCode::BAD_REQUEST, "bad_request");
    };

    let Some(auth) = auth else {
        tracing::warn!(%ip, "login attempt but no password has been set up yet (run `ddnet-ai web-passwd`)");
        return json_error(StatusCode::UNAUTHORIZED, "invalid_credentials");
    };

    // Review finding F4: argon2id verification is deliberately synchronous, CPU-bound work
    // (~0.2s at production params, see `secrets::Argon2Params::default`). Running it inline on
    // an async worker thread blocked every other request scheduled on that same thread — a
    // handful of concurrent login attempts was enough to make an unrelated `GET /api/me` take
    // ~0.8s. `spawn_blocking` moves the CPU-bound work to tokio's dedicated blocking pool, and
    // the semaphore bounds concurrent memory use (each verification allocates the full `m_cost`,
    // ~64 MiB at production params, so `ARGON2_MAX_CONCURRENT` permits caps that at ~128 MiB).
    let verified = {
        let _permit = state
            .argon2_semaphore
            .acquire()
            .await
            .expect("argon2 semaphore is never closed");
        let hash_phc = auth.hash_phc.clone();
        let password_for_blocking = password.clone();
        tokio::task::spawn_blocking(move || secrets::verify_password(&password_for_blocking, &hash_phc))
            .await
            .expect("argon2 verification task panicked")
            .unwrap_or_else(|error| {
                tracing::error!(%ip, %error, "failed to verify password hash");
                false
            })
    };

    // Audit log line: IP and outcome only, never the password (acceptance criterion 3). Actually
    // reaching a log sink requires a tracing subscriber to be installed — see
    // `ddnet-ai`'s `web_cmd::init_tracing` (review finding F1: this call was always correct, but
    // was a silent no-op until something installed a subscriber at all).
    tracing::info!(%ip, success = verified, bypassed_global_rate_limit = bypass_global, "login attempt");

    if !verified {
        return json_error(StatusCode::UNAUTHORIZED, "invalid_credentials");
    }

    state.login_rate_limiter.record_success(ip);

    let new_session = state.sessions.create(ip);
    let session_cookie_value = encode_session_cookie_value(&state.session_key, &new_session.id);
    let session_cookie = build_session_cookie(
        state.config.cookie_secure,
        session_cookie_value,
        duration_to_cookie_max_age(state.config.absolute_timeout),
    );

    // Review round 2, finding F9b: only ever REFRESH an id that was ALREADY trusted (this
    // request's own `bypass_global`, computed above from the state *before* this login) — never
    // mint trust for a presented id that `is_trusted` had just said "no" to. The previous
    // behavior (reuse whatever id was presented, trusted or not, "so a returning browser doesn't
    // accumulate a fresh device record on every login") had a sharp edge: an old device cookie's
    // id and HMAC signature don't change across a password rotation, only its stored fingerprint
    // does — so re-confirming a presented-but-untrusted id under the NEW fingerprint (purely
    // because the legitimate browser happened to log back in with it) also silently re-authorized
    // ANY OTHER copy of that exact cookie value an attacker might have captured earlier. A
    // presented-but-untrusted id is now simply left alone: it stays exactly as untrusted as it
    // was (bounded by its own TTL, or by `MAX_TRACKED_DEVICES`' eviction — finding F10), and this
    // login instead earns a brand-new id that no earlier copy could have known in advance.
    let device_id = if bypass_global {
        presented_device_id.expect("bypass_global is only ever true when a device id was both presented and trusted")
    } else {
        random_bytes()
    };
    let fingerprint = current_fingerprint.expect("auth is Some at this point, so current_fingerprint was too");
    state.devices.confirm(device_id, &fingerprint);
    let device_cookie_value = encode_device_cookie_value(&state.session_key, &device_id);
    let device_cookie = build_device_cookie(
        state.config.cookie_secure,
        device_cookie_value,
        duration_to_cookie_max_age(state.config.trusted_device_ttl),
    );

    let response_jar = CookieJar::new().add(session_cookie).add(device_cookie);

    (
        response_jar,
        Json(LoginOkBody {
            ok: true,
            csrf_token: encode_b64(&new_session.csrf_token),
        }),
    )
        .into_response()
}

// ---------------------------------------------------------------------------------------------
// POST /api/logout
// ---------------------------------------------------------------------------------------------

pub async fn logout(State(state): State<SharedState>, headers: HeaderMap, jar: CookieJar) -> Response {
    if !request_is_same_origin(&headers) {
        return json_error(StatusCode::FORBIDDEN, "cross_origin");
    }

    let Some((id, info)) = current_session(&state, &jar) else {
        return json_error(StatusCode::UNAUTHORIZED, "unauthenticated");
    };

    let Some(csrf_header) = headers.get(csrf::CSRF_HEADER_NAME).and_then(|v| v.to_str().ok()) else {
        return json_error(StatusCode::FORBIDDEN, "missing_csrf");
    };
    if !csrf::matches(&info.csrf_token, csrf_header) {
        return json_error(StatusCode::FORBIDDEN, "bad_csrf");
    }

    // Also notifies any open WebSocket for this session to close itself immediately instead of
    // running until its next periodic validity check (review finding F2). Deliberately does not
    // touch the trusted-device cookie: device trust is meant to survive an ordinary logout.
    state.invalidate_session(id);

    let removal = build_removal_cookie(state.config.cookie_secure);
    let response_jar = CookieJar::new().add(removal);
    (response_jar, Json(serde_json::json!({ "ok": true }))).into_response()
}

// ---------------------------------------------------------------------------------------------
// GET /api/me
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct MeBody {
    authenticated: bool,
    csrf_token: Option<String>,
}

pub async fn me(State(state): State<SharedState>, jar: CookieJar) -> Response {
    match current_session(&state, &jar) {
        Some((_, info)) => Json(MeBody {
            authenticated: true,
            csrf_token: Some(encode_b64(&info.csrf_token)),
        })
        .into_response(),
        None => Json(MeBody {
            authenticated: false,
            csrf_token: None,
        })
        .into_response(),
    }
}
