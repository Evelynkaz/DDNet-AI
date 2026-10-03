//! The owner-only bot control routes (task 5.6, D-070):
//!
//! | route | |
//! |---|---|
//! | `GET /api/bot/status` | the live bot's status (from the read-only bridge) for the status panel |
//! | `POST /api/bot/command` | one typed command ([`ddai_botctl::proto::ControlCommand`]) to the bot, its reply back |
//! | `GET /api/bot/relations` | the friend / war / ignore / clan lists, as normalised names |
//! | `POST /api/bot/relations` | add / remove one name, persist, and ask the running bot to reload |
//!
//! **Every route is behind the session** (`401` without one). **Every POST** also needs the **strict** same-origin check
//! (`Origin` present and equal to `Host`, `Sec-Fetch-Site: same-origin` when sent, `https://` behind HTTPS —
//! [`crate::origin::is_strict_same_origin`]), the CSRF token of the session in `X-CSRF-Token`, and a JSON body. They are
//! under `no-store` like every `/api/*` route, and the security headers are the unchanged global ones.
//!
//! **What the web cannot do through here.** The only things it can send the bot are the variants of `ControlCommand`:
//! no chat, no quit, no connect, no allow-list and no `ready` flag exist in that vocabulary (and the web process never
//! writes `live-servers.toml`). The relations editor writes only the lists file.
//!
//! **Names.** The lists are returned to the logged-in owner only. Nothing here logs a name: log lines carry the audit
//! tag of the session, the command's tag and counts.

use std::time::Duration;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum_extra::extract::cookie::CookieJar;
use ddai_botctl::proto::{ControlCommand, ControlReply, ReplyCode};
use ddai_botctl::relations::ListKind;
use serde::{Deserialize, Serialize};

use crate::auth::cookie::audit_tag;
use crate::auth::csrf;
use crate::control::client::SendError;
use crate::control::relations::{MAX_ENTRIES_PER_LIST, MAX_NAME_BYTES, StoreError};
use crate::live::source::SourceKind;
use crate::session_guard::{current_session, peek_session, request_is_strict_same_origin};
use crate::state::SharedState;

/// A bridge status older than this means the bot is not (or no longer) there.
pub const STATUS_STALE: Duration = Duration::from_secs(3);

fn json_error(status: StatusCode, error: &'static str) -> Response {
    (status, Json(serde_json::json!({ "error": error }))).into_response()
}

fn json_error_detail(status: StatusCode, error: &'static str, detail: &str) -> Response {
    (status, Json(serde_json::json!({ "error": error, "detail": detail }))).into_response()
}

/// Who asked: the opaque audit tag of the session (never the cookie).
struct Owner {
    tag: String,
}

/// Task 5.7: the offline demo is what the site shows **and** there is no live bot to command: its control socket is not
/// there. The demo takes no commands and has no control socket; `control.sock` is the live bot's alone. A bot that is running
/// but whose bridge is away (started with `--no-bridge`, or the site's bridge connection is reconnecting) still has its control
/// socket, so it stays commandable while the demo is on show. A stale socket file with nobody behind it is refused later, by
/// the connection (`bot_unavailable`).
async fn demo_only(state: &SharedState) -> bool {
    let demo_on_show = state
        .live_hub
        .as_ref()
        .and_then(|hub| hub.latest_source())
        .is_some_and(|(kind, _)| kind == SourceKind::Demo);
    demo_on_show && !state.control.socket_present().await
}

/// A refusal: the status and the `error` code of the JSON body (small, so `Result` stays small).
type Refusal = (StatusCode, &'static str);

/// `refresh`: a request a person made refreshes the idle timeout; the status poll (nobody's action) does not.
fn authorize_get(state: &SharedState, jar: &CookieJar, refresh: bool) -> Result<Owner, Refusal> {
    let id = if refresh {
        current_session(state, jar).map(|(id, _info)| id)
    } else {
        peek_session(state, jar)
    };
    let Some(id) = id else {
        return Err((StatusCode::UNAUTHORIZED, "unauthenticated"));
    };
    Ok(Owner {
        tag: audit_tag(&state.session_key, &id),
    })
}

/// Origin (strict) -> session -> CSRF -> JSON content type, in that order; the first failure answers.
fn authorize_post(state: &SharedState, headers: &HeaderMap, jar: &CookieJar) -> Result<Owner, Refusal> {
    if !request_is_strict_same_origin(headers, state.config.cookie_secure) {
        return Err((StatusCode::FORBIDDEN, "cross_origin"));
    }
    let Some((id, info)) = current_session(state, jar) else {
        return Err((StatusCode::UNAUTHORIZED, "unauthenticated"));
    };
    let Some(token) = headers.get(csrf::CSRF_HEADER_NAME).and_then(|v| v.to_str().ok()) else {
        return Err((StatusCode::FORBIDDEN, "missing_csrf"));
    };
    if !csrf::matches(&info.csrf_token, token) {
        return Err((StatusCode::FORBIDDEN, "bad_csrf"));
    }
    // The media type itself, exactly (parameters such as `; charset=utf-8` are fine, `application/jsonfoo` is not).
    let is_json = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|ct| ct.split(';').next())
        .is_some_and(|media| media.trim().eq_ignore_ascii_case("application/json"));
    if !is_json {
        return Err((StatusCode::UNSUPPORTED_MEDIA_TYPE, "json_required"));
    }
    Ok(Owner {
        tag: audit_tag(&state.session_key, &id),
    })
}

// ---------------------------------------------------------------------------------------------
// GET /api/bot/status
// ---------------------------------------------------------------------------------------------

pub async fn status(State(state): State<SharedState>, jar: CookieJar) -> Response {
    // Polled by the page on a timer: must not keep the session alive (see `peek_session`).
    if let Err((status, error)) = authorize_get(&state, &jar, false) {
        return json_error(status, error);
    }
    let latest = state.live_hub.as_ref().and_then(|hub| hub.latest_bot_status());
    let (live, age_ms, status) = match latest {
        Some((age, json)) => {
            let live = age <= STATUS_STALE;
            // A stale status is not shown as if it were current.
            let status = if live {
                serde_json::from_str::<serde_json::Value>(&json).ok()
            } else {
                None
            };
            (live, Some(u64::try_from(age.as_millis()).unwrap_or(u64::MAX)), status)
        }
        None => (false, None, None),
    };
    let source = state
        .live_hub
        .as_ref()
        .and_then(|hub| hub.latest_source())
        .map(|(kind, _)| kind.as_str());
    Json(serde_json::json!({
        // Whether this web unit was started with a bot bridge at all (`--bot-socket`).
        "bridge": state.config.bot_socket.is_some(),
        // Task 5.7: what the site shows (`live`, `demo`, `none`; null without a multiplexer). The status below is only ever
        // the live bot's: with the demo on show `live` is false and `status` null.
        "source": source,
        "demo_configured": state.config.demo_socket.is_some(),
        "live": live,
        "age_ms": age_ms,
        "status": status,
        "control_socket": state.control.socket_present().await,
    }))
    .into_response()
}

// ---------------------------------------------------------------------------------------------
// POST /api/bot/command
// ---------------------------------------------------------------------------------------------

#[derive(Serialize)]
struct ReplyBody<'a> {
    ok: bool,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<ReplyCode>,
}

fn reply_response(reply: &ControlReply) -> Response {
    let status = match reply.code {
        None => StatusCode::OK,
        Some(ReplyCode::BadRequest) => StatusCode::BAD_REQUEST,
        Some(ReplyCode::RateLimited) => StatusCode::TOO_MANY_REQUESTS,
        Some(ReplyCode::Busy | ReplyCode::Gone) => StatusCode::SERVICE_UNAVAILABLE,
        Some(ReplyCode::Timeout) => StatusCode::GATEWAY_TIMEOUT,
    };
    (
        status,
        Json(ReplyBody {
            ok: reply.ok,
            text: &reply.text,
            code: reply.code,
        }),
    )
        .into_response()
}

fn send_error_response(e: SendError) -> Response {
    match e {
        SendError::Unavailable => json_error(StatusCode::SERVICE_UNAVAILABLE, "bot_unavailable"),
        SendError::Timeout => json_error(StatusCode::GATEWAY_TIMEOUT, "bot_timeout"),
        SendError::Protocol => json_error(StatusCode::BAD_GATEWAY, "bot_protocol"),
    }
}

pub async fn command(State(state): State<SharedState>, headers: HeaderMap, jar: CookieJar, body: Bytes) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    // Task 5.7: commands are the live bot's. With the demo on show and no live bot to take them, nothing is sent anywhere.
    if demo_only(&state).await {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "demo_only");
    }
    // A closed vocabulary: anything that is not exactly a `ControlCommand` (a `say`, a `quit`, an extra field) is a
    // parse error here, before it could reach the socket.
    let cmd: ControlCommand = match serde_json::from_slice(&body) {
        Ok(c) => c,
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "bad_request"),
    };
    if let Err(e) = cmd.validate() {
        return json_error_detail(StatusCode::BAD_REQUEST, "invalid", &e.to_string());
    }
    let tag = cmd.tag();
    match state.control.send(&owner.tag, cmd).await {
        Ok(reply) => {
            tracing::info!(session = %owner.tag, cmd = %tag, ok = reply.ok, code = ?reply.code, "bot command from the web");
            reply_response(&reply)
        }
        Err(e) => {
            tracing::info!(session = %owner.tag, cmd = %tag, error = %e, "bot command from the web failed");
            send_error_response(e)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GET / POST /api/bot/relations
// ---------------------------------------------------------------------------------------------

fn store_error_response(e: StoreError) -> Response {
    match e {
        StoreError::Empty => json_error_detail(StatusCode::BAD_REQUEST, "invalid_name", "empty"),
        StoreError::TooLong => json_error_detail(StatusCode::BAD_REQUEST, "invalid_name", "too_long"),
        StoreError::Control => json_error_detail(StatusCode::BAD_REQUEST, "invalid_name", "control"),
        StoreError::Full => json_error(StatusCode::CONFLICT, "list_full"),
        StoreError::Corrupt => json_error(StatusCode::INTERNAL_SERVER_ERROR, "relations_unreadable"),
        StoreError::Write => json_error(StatusCode::INTERNAL_SERVER_ERROR, "relations_write_failed"),
    }
}

pub async fn relations_get(State(state): State<SharedState>, jar: CookieJar) -> Response {
    if let Err((status, error)) = authorize_get(&state, &jar, true) {
        return json_error(status, error);
    }
    match state.relations.view().await {
        Ok(view) => Json(serde_json::json!({
            "lists": view,
            "limits": { "max_entries": MAX_ENTRIES_PER_LIST, "max_name_bytes": MAX_NAME_BYTES },
        }))
        .into_response(),
        Err(e) => store_error_response(e),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Op {
    Add,
    Remove,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RelationsEdit {
    op: Op,
    kind: String,
    name: String,
}

/// How the running bot took the reload that follows a saved edit. `digest` is the fingerprint of the lists the web
/// just wrote; the bot's reply carries the fingerprint of the lists it now holds, and they must be the same, or the bot
/// is reading another file (its `--relations` differs from the site's) and the owner must be told so.
fn applied_state(result: &Result<ControlReply, SendError>, digest: &str) -> &'static str {
    match result {
        Ok(r) if r.ok => match r.data.as_ref().and_then(|d| d.get("digest")).and_then(|d| d.as_str()) {
            Some(bot) if bot == digest => "applied",
            Some(_) => "mismatch",
            None => "unverified",
        },
        Ok(r) => match r.code {
            Some(ReplyCode::RateLimited) => "rate_limited",
            Some(ReplyCode::Busy) => "busy",
            Some(ReplyCode::Timeout) => "timeout",
            _ => "bot_refused",
        },
        Err(SendError::Unavailable) => "unavailable",
        Err(SendError::Timeout) => "timeout",
        Err(SendError::Protocol) => "error",
    }
}

pub async fn relations_post(
    State(state): State<SharedState>,
    headers: HeaderMap,
    jar: CookieJar,
    body: Bytes,
) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    let edit: RelationsEdit = match serde_json::from_slice(&body) {
        Ok(e) => e,
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "bad_request"),
    };
    let Some(kind) = ListKind::parse(&edit.kind) else {
        return json_error(StatusCode::BAD_REQUEST, "bad_request");
    };
    let (op_name, folded, changed, moved_from, view, digest) = match edit.op {
        Op::Add => match state.relations.add(kind, &edit.name).await {
            Ok(o) => ("add", o.folded, o.changed, o.moved_from, o.view, o.digest),
            Err(e) => return store_error_response(e),
        },
        Op::Remove => match state.relations.remove(kind, &edit.name).await {
            Ok(o) => ("remove", o.folded, o.changed, Vec::new(), o.view, o.digest),
            Err(e) => return store_error_response(e),
        },
    };
    // Applies to the running bot through the command channel. An edit that changed nothing needs no reload.
    let (applied, applied_text) = if changed && demo_only(&state).await {
        // The demo is on show and the bot has no control socket: there is no bot to tell (the lists wait for its next start).
        ("unavailable", String::new())
    } else if changed {
        let result = state.control.send(&owner.tag, ControlCommand::ReloadRelations {}).await;
        let text = match &result {
            Ok(r) => r.text.clone(),
            Err(e) => e.to_string(),
        };
        (applied_state(&result, &digest), text)
    } else {
        ("unchanged", String::new())
    };
    // Counts and tags only: the name is not logged.
    tracing::info!(
        session = %owner.tag,
        op = op_name,
        kind = kind.name(),
        changed,
        applied,
        entries = view.list(kind).len(),
        "relations edit from the web"
    );
    Json(serde_json::json!({
        "ok": true,
        "op": op_name,
        "kind": kind.name(),
        "normalised": folded,
        "changed": changed,
        "moved_from": moved_from.iter().map(|k| k.name()).collect::<Vec<_>>(),
        "lists": view,
        "applied": applied,
        "applied_text": applied_text,
    }))
    .into_response()
}
