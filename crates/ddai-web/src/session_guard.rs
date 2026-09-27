//! Small helpers shared by every handler that needs "is this request authenticated?" or "is this
//! request same-origin?" — the login/logout/me HTTP handlers and the WebSocket upgrade all go
//! through the same two checks (acceptance criteria 4).

use axum::http::HeaderMap;
use axum_extra::extract::cookie::CookieJar;

use crate::auth::cookie::{cookie_name, decode_session_cookie_value};
use crate::auth::session::{SessionId, SessionInfo};
use crate::origin::is_same_origin;
use crate::state::AppState;

/// Extracts and validates the session cookie, refreshing its idle timeout if it's still valid
/// (see [`crate::auth::session::SessionStore::touch`]).
pub fn current_session(state: &AppState, jar: &CookieJar) -> Option<(SessionId, SessionInfo)> {
    let cookie = jar.get(cookie_name(state.config.cookie_secure))?;
    let id = decode_session_cookie_value(&state.session_key, cookie.value())?;
    let info = state.sessions.touch(&id)?;
    Some((id, info))
}

/// Same-origin check for a request's headers (acceptance criterion 4: state-changing HTTP routes
/// and the WebSocket upgrade all require this).
pub fn request_is_same_origin(headers: &HeaderMap) -> bool {
    let host = headers.get(axum::http::header::HOST).and_then(|v| v.to_str().ok());
    let origin = headers.get(axum::http::header::ORIGIN).and_then(|v| v.to_str().ok());
    let sec_fetch_site = headers.get("sec-fetch-site").and_then(|v| v.to_str().ok());
    is_same_origin(host, origin, sec_fetch_site)
}
