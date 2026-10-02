//! Small helpers shared by every handler that needs "is this request authenticated?" or "is this
//! request same-origin?" — the login/logout/me HTTP handlers and the WebSocket upgrade all go
//! through the same two checks (acceptance criteria 4).

use axum::http::HeaderMap;
use axum_extra::extract::cookie::CookieJar;

use crate::auth::cookie::{cookie_name, decode_session_cookie_value};
use crate::auth::session::{SessionId, SessionInfo};
use crate::origin::{is_same_origin, is_strict_same_origin};
use crate::state::AppState;

/// Extracts and validates the session cookie, refreshing its idle timeout if it's still valid
/// (see [`crate::auth::session::SessionStore::touch`]).
pub fn current_session(state: &AppState, jar: &CookieJar) -> Option<(SessionId, SessionInfo)> {
    let cookie = jar.get(cookie_name(state.config.cookie_secure))?;
    let id = decode_session_cookie_value(&state.session_key, cookie.value())?;
    let info = state.sessions.touch(&id)?;
    Some((id, info))
}

/// Like [`current_session`], but **does not refresh** the idle timeout ([`crate::auth::session::SessionStore::is_valid`]):
/// for polls that no person caused (the bot status panel refreshes itself every couple of seconds). A page left open
/// must not keep the session alive forever, exactly as the open WebSocket does not (5.3, review F2). Returns the session
/// id when it is still valid.
pub fn peek_session(state: &AppState, jar: &CookieJar) -> Option<SessionId> {
    let cookie = jar.get(cookie_name(state.config.cookie_secure))?;
    let id = decode_session_cookie_value(&state.session_key, cookie.value())?;
    state.sessions.is_valid(&id).then_some(id)
}

/// Same-origin check for a request's headers (acceptance criterion 4: state-changing HTTP routes
/// and the WebSocket upgrade all require this).
pub fn request_is_same_origin(headers: &HeaderMap) -> bool {
    let host = headers.get(axum::http::header::HOST).and_then(|v| v.to_str().ok());
    let origin = headers.get(axum::http::header::ORIGIN).and_then(|v| v.to_str().ok());
    let sec_fetch_site = headers.get("sec-fetch-site").and_then(|v| v.to_str().ok());
    is_same_origin(host, origin, sec_fetch_site)
}

/// The strict same-origin check of the bot-control routes ([`is_strict_same_origin`]); `require_https` follows the
/// deployment (`WebConfig::cookie_secure`).
pub fn request_is_strict_same_origin(headers: &HeaderMap, require_https: bool) -> bool {
    let host = headers.get(axum::http::header::HOST).and_then(|v| v.to_str().ok());
    let origin = headers.get(axum::http::header::ORIGIN).and_then(|v| v.to_str().ok());
    let sec_fetch_site = headers.get("sec-fetch-site").and_then(|v| v.to_str().ok());
    is_strict_same_origin(host, origin, sec_fetch_site, require_https)
}
