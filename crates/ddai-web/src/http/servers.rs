//! The «Серверы» tab's routes (task 5.12, D-099): the server list, the owner's favourites and the proxy profiles.
//!
//! | route | |
//! |---|---|
//! | `GET /api/servers` | the cached master list (rows, when it was fetched, what the last fetch did) |
//! | `POST /api/servers/refresh` | asks for a fresh list: rewrites `launch/servers-refresh`; a path unit runs the fetch in a unit that has network |
//! | `GET /api/favourites` | the favourites, each with its proxy and whether the bot's ban memory has it closed |
//! | `POST /api/favourites/{add,update,remove,reopen}` | change the favourites file (strictly validated; the owner's consent is required to add) |
//! | `GET /api/proxies` | the proxy profiles (never a password or a user name) and the last «Проверить» result |
//! | `POST /api/proxies/{save,remove,check}` | write or remove a profile (0600), or ask a helper unit to check one |
//!
//! **No privilege and no network.** Every mutating route needs the session, the strict same-origin check, the CSRF token and a JSON
//! body ([`authorize_post`]) plus a rate limit of its own ([`crate::serverbrowser::RateGate`]; every attempt counts, valid or not).
//! The web only writes files in directories it already owned (`launch/`, `secrets/`); the root helper (`ddnet-ai launch apply`) judges
//! a favourite again before the bot is ever pointed at it, and **a ban or kick closes the favourite until the owner presses «Открыть
//! снова»** (`POST /api/favourites/reopen`, which only writes `reopened_at`; the helper compares it with its own memory). Nothing here
//! ever picks, switches or falls back to another proxy: a favourite's connection is what the owner wrote.
//!
//! Logs carry the audit tag of the session and the kind of change, never an address, a name or any proxy value.

use std::net::SocketAddr;
use std::path::Path;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum_extra::extract::cookie::CookieJar;
use ddai_client::favourites::{self, DEFAULT_NICK, Favourite, FavouriteError};
use ddai_client::live_servers::LiveServers;
use serde::{Deserialize, Serialize};

use crate::http::bot::{Refusal, authorize_get, authorize_post, json_error};
use crate::launch::{REQUEST_STALE_SECS, read_regular_nofollow, unix_now, write_atomic};
use crate::serverbrowser::cache::{CacheProblem, read_master, read_refresh};
use crate::serverbrowser::favourites::StoreError;
use crate::serverbrowser::proxies::{ProxyError, ProxyForm};
use crate::serverbrowser::{
    BLOCKED_FILE, BlockedEntry, BlockedFile, MAX_BLOCKED_BYTES, MAX_PROXY_CHECK_BYTES, PROXY_CHECK_REQUEST_FILE,
    PROXY_CHECK_RESULT_FILE, ProxyCheckRequest, ProxyCheckResult, REFRESH_TRIGGER_FILE,
};
use crate::state::SharedState;

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

fn parse_body<T: serde::de::DeserializeOwned>(body: &Bytes) -> Result<T, Refusal> {
    serde_json::from_slice(body).map_err(|_| (StatusCode::BAD_REQUEST, "bad_request"))
}

fn refuse((status, error): Refusal) -> Response {
    json_error(status, error)
}

fn too_many_edits() -> Response {
    json_error(StatusCode::TOO_MANY_REQUESTS, "rate_limited")
}

/// The helper's list of closed servers, from the root-owned status directory (empty when there is none or it cannot be read).
fn read_blocked(status_dir: &Path) -> Vec<BlockedEntry> {
    let Ok(bytes) = read_regular_nofollow(&status_dir.join(BLOCKED_FILE), MAX_BLOCKED_BYTES) else {
        return Vec::new();
    };
    match serde_json::from_slice::<BlockedFile>(&bytes) {
        Ok(f) if f.v == 1 => f.blocked,
        _ => Vec::new(),
    }
}

/// The newest kick or ban that still closes `fav`: one recorded for its address or for any address on the same IP (a ban is the
/// machine's), not yet lifted by the favourite's own `reopened_at`.
pub(crate) fn closing_block(fav: &Favourite, blocks: &[BlockedEntry]) -> Option<BlockedEntry> {
    let addr = fav.socket_addr()?;
    blocks
        .iter()
        .filter(|b| {
            b.address == fav.address
                || b.address
                    .parse::<SocketAddr>()
                    .is_ok_and(|a| a.ip().to_canonical() == addr.ip().to_canonical())
        })
        .filter(|b| fav.reopened_at <= b.at)
        .max_by_key(|b| b.at)
        .cloned()
}

#[derive(Serialize)]
struct FavouriteView<'a> {
    address: &'a str,
    name: &'a str,
    nick: &'a str,
    connection: &'a str,
    proxy: Option<&'a str>,
    consent_at: u64,
    notes: &'a str,
    added_at: u64,
    reopened_at: u64,
    blocked: Option<BlockedView>,
}

#[derive(Serialize)]
struct BlockedView {
    at: u64,
    code: i32,
}

fn favourite_view<'a>(f: &'a Favourite, blocks: &[BlockedEntry]) -> FavouriteView<'a> {
    FavouriteView {
        address: &f.address,
        name: &f.name,
        nick: &f.nick,
        connection: &f.connection,
        proxy: f.proxy_name(),
        consent_at: f.consent_at,
        notes: &f.notes,
        added_at: f.added_at,
        reopened_at: f.reopened_at,
        blocked: closing_block(f, blocks).map(|b| BlockedView { at: b.at, code: b.code }),
    }
}

fn store_refusal(e: StoreError) -> Refusal {
    let status = match e {
        StoreError::Refused(FavouriteError::Duplicate | FavouriteError::TooMany) => StatusCode::CONFLICT,
        StoreError::Refused(_) => StatusCode::BAD_REQUEST,
        StoreError::NotFound => StatusCode::NOT_FOUND,
        StoreError::FileUnusable(_) | StoreError::AllowListNotLiteral => StatusCode::CONFLICT,
        StoreError::WriteFailed => StatusCode::SERVICE_UNAVAILABLE,
    };
    (status, e.code())
}

fn store_error(e: StoreError) -> Response {
    refuse(store_refusal(e))
}

fn proxy_refusal(e: ProxyError) -> Refusal {
    let status = match e {
        ProxyError::Exists | ProxyError::TooMany | ProxyError::NotManaged => StatusCode::CONFLICT,
        ProxyError::NotFound => StatusCode::NOT_FOUND,
        ProxyError::WriteFailed => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::BAD_REQUEST,
    };
    (status, e.code())
}

fn proxy_error(e: ProxyError) -> Response {
    refuse(proxy_refusal(e))
}

// ---------------------------------------------------------------------------------------------
// the server list
// ---------------------------------------------------------------------------------------------

pub async fn servers_get(State(state): State<SharedState>, jar: CookieJar) -> Response {
    // Polled by the page: must not keep the session alive.
    if let Err((status, error)) = authorize_get(&state, &jar, false) {
        return json_error(status, error);
    }
    let dir = state.config.servers_dir.clone();
    let body = tokio::task::spawn_blocking(move || {
        let refresh = read_refresh(&dir);
        match read_master(&dir) {
            Ok(cache) => {
                let age = unix_now().saturating_sub(cache.fetched_at);
                serde_json::json!({
                    "enabled": true,
                    "problem": serde_json::Value::Null,
                    "fetched_at": cache.fetched_at,
                    "age_s": age,
                    "master": cache.master,
                    "min_refresh_s": ddai_client::server_list::MIN_REFRESH_SECS,
                    "refresh": refresh,
                    "servers": cache.servers,
                })
            }
            Err(problem) => serde_json::json!({
                "enabled": dir.is_dir(),
                "problem": match problem { CacheProblem::Missing => "missing", CacheProblem::Invalid => "invalid" },
                "refresh": refresh,
                "servers": [],
            }),
        }
    })
    .await;
    match body {
        Ok(b) => Json(b).into_response(),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "servers_unavailable"),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}

pub async fn refresh_post(
    State(state): State<SharedState>,
    headers: HeaderMap,
    jar: CookieJar,
    body: Bytes,
) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    if let Err(r) = parse_body::<Empty>(&body) {
        return refuse(r);
    }
    if !state.refresh_gate.try_take() {
        return json_error(StatusCode::TOO_MANY_REQUESTS, "rate_limited");
    }
    let launch_dir = state.config.launch_dir.clone();
    let servers_dir = state.config.servers_dir.clone();
    let result = tokio::task::spawn_blocking(move || -> Result<bool, (StatusCode, &'static str)> {
        if !launch_dir.is_dir() {
            return Err((StatusCode::SERVICE_UNAVAILABLE, "launcher_unavailable"));
        }
        // A cache younger than the fetch's own minimum is not asked for again.
        if let Ok(cache) = read_master(&servers_dir)
            && unix_now().saturating_sub(cache.fetched_at) < ddai_client::server_list::MIN_REFRESH_SECS
        {
            return Ok(false);
        }
        // Rewritten in place (truncate, write, close): the path unit's `PathChanged=` fires on the close after a write. The content
        // means nothing. Never through a symlink.
        use std::io::Write;
        let mut file = ddai_os::nofollow::create_truncate_nofollow(&launch_dir.join(REFRESH_TRIGGER_FILE), 0o644)
            .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "refresh_write_failed"))?;
        file.write_all(unix_now().to_string().as_bytes())
            .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "refresh_write_failed"))?;
        Ok(true)
    })
    .await;
    match result {
        Ok(Ok(asked)) => {
            tracing::info!(session = %owner.tag, asked, "server list refresh requested from the web");
            (
                StatusCode::ACCEPTED,
                Json(serde_json::json!({"ok": true, "asked": asked})),
            )
                .into_response()
        }
        Ok(Err((status, error))) => json_error(status, error),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "refresh_write_failed"),
    }
}

// ---------------------------------------------------------------------------------------------
// favourites
// ---------------------------------------------------------------------------------------------

pub async fn favourites_get(State(state): State<SharedState>, jar: CookieJar) -> Response {
    if let Err((status, error)) = authorize_get(&state, &jar, false) {
        return json_error(status, error);
    }
    let body = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            let blocks = read_blocked(&state.config.status_dir);
            match state.favourites.load() {
                Ok(list) => serde_json::json!({
                    "enabled": state.config.launch_dir.is_dir(),
                    "error": serde_json::Value::Null,
                    "favourites": list.favourites.iter().map(|f| favourite_view(f, &blocks)).collect::<Vec<_>>(),
                    "max": favourites::MAX_FAVOURITES,
                }),
                Err(code) => serde_json::json!({
                    "enabled": state.config.launch_dir.is_dir(),
                    "error": code,
                    "favourites": [],
                    "max": favourites::MAX_FAVOURITES,
                }),
            }
        }
    })
    .await;
    match body {
        Ok(b) => Json(b).into_response(),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "favourites_unavailable"),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AddForm {
    address: String,
    name: String,
    nick: Option<String>,
    /// `direct` (default) or `proxy:<name>`.
    connection: Option<String>,
    /// The owner confirms that the server's admin allows the bot. Must be `true`.
    consent: bool,
    notes: Option<String>,
}

/// The proxy a connection names must be a profile that exists (the helper checks it loads, again, at launch).
fn check_connection(state: &SharedState, connection: &str) -> Result<(), Refusal> {
    favourites::valid_connection(connection).map_err(|e| (StatusCode::BAD_REQUEST, e.code()))?;
    if let Some(name) = connection.strip_prefix("proxy:")
        && !state.proxies.exists(name)
    {
        return Err((StatusCode::BAD_REQUEST, "proxy_unknown"));
    }
    Ok(())
}

pub async fn favourite_add(
    State(state): State<SharedState>,
    headers: HeaderMap,
    jar: CookieJar,
    body: Bytes,
) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    if !state.servers_edit_gate.try_take() {
        return too_many_edits();
    }
    let form: AddForm = match parse_body(&body) {
        Ok(f) => f,
        Err(r) => return refuse(r),
    };
    if !form.consent {
        return json_error(StatusCode::BAD_REQUEST, "consent_required");
    }
    let connection = form.connection.unwrap_or_else(|| "direct".to_string());
    if let Err(r) = check_connection(&state, &connection) {
        return refuse(r);
    }
    let now = unix_now();
    let fav = Favourite {
        address: form.address,
        name: form.name,
        nick: form.nick.unwrap_or_else(|| DEFAULT_NICK.to_string()),
        connection,
        consent_at: now,
        notes: form.notes.unwrap_or_default(),
        added_at: now,
        reopened_at: 0,
    };
    let rules = state.favourites.rules();
    if let Err(e) = fav.validate(rules) {
        return store_error(StoreError::Refused(e));
    }
    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            // One statement per server: an address the owner's allow-list also names is refused (the helper refuses it too).
            let live = LiveServers::load_or_empty(&state.config.live_servers).unwrap_or_default();
            if fav.socket_addr().is_some_and(|a| live.listed(a)) {
                return Err(StoreError::Refused(FavouriteError::Duplicate));
            }
            // A host name in the allow-list can hide the very address a favourite names (neither the web nor the root helper resolves
            // names): no favourites while one exists (review 5.12 F2).
            if live.has_non_literal_entry() {
                return Err(StoreError::AllowListNotLiteral);
            }
            state.favourites.change(|list| {
                list.favourites.push(fav.clone());
                Ok(())
            })
        }
    })
    .await;
    match result {
        Ok(Ok(())) => {
            tracing::info!(session = %owner.tag, "favourite added from the web");
            (StatusCode::CREATED, Json(serde_json::json!({"ok": true}))).into_response()
        }
        Ok(Err(e)) => store_error(e),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "favourites_write_failed"),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateForm {
    address: String,
    name: Option<String>,
    nick: Option<String>,
    connection: Option<String>,
    notes: Option<String>,
}

pub async fn favourite_update(
    State(state): State<SharedState>,
    headers: HeaderMap,
    jar: CookieJar,
    body: Bytes,
) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    if !state.servers_edit_gate.try_take() {
        return too_many_edits();
    }
    let form: UpdateForm = match parse_body(&body) {
        Ok(f) => f,
        Err(r) => return refuse(r),
    };
    if let Some(c) = &form.connection
        && let Err(r) = check_connection(&state, c)
    {
        return refuse(r);
    }
    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            state.favourites.change(|list| {
                let fav = list
                    .favourites
                    .iter_mut()
                    .find(|f| f.address == form.address)
                    .ok_or(StoreError::NotFound)?;
                if let Some(v) = form.name {
                    fav.name = v;
                }
                if let Some(v) = form.nick {
                    fav.nick = v;
                }
                if let Some(v) = form.connection {
                    fav.connection = v;
                }
                if let Some(v) = form.notes {
                    fav.notes = v;
                }
                // `consent_at`, `added_at` and `reopened_at` are never touched here: a change of nick or proxy is not a re-opening.
                Ok(())
            })
        }
    })
    .await;
    match result {
        Ok(Ok(())) => {
            tracing::info!(session = %owner.tag, "favourite changed from the web");
            Json(serde_json::json!({"ok": true})).into_response()
        }
        Ok(Err(e)) => store_error(e),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "favourites_write_failed"),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AddressForm {
    address: String,
}

pub async fn favourite_remove(
    State(state): State<SharedState>,
    headers: HeaderMap,
    jar: CookieJar,
    body: Bytes,
) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    if !state.servers_edit_gate.try_take() {
        return too_many_edits();
    }
    let form: AddressForm = match parse_body(&body) {
        Ok(f) => f,
        Err(r) => return refuse(r),
    };
    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            state.favourites.change(|list| {
                let before = list.favourites.len();
                list.favourites.retain(|f| f.address != form.address);
                if list.favourites.len() == before {
                    Err(StoreError::NotFound)
                } else {
                    Ok(())
                }
            })
        }
    })
    .await;
    match result {
        Ok(Ok(())) => {
            tracing::info!(session = %owner.tag, "favourite removed from the web");
            Json(serde_json::json!({"ok": true})).into_response()
        }
        Ok(Err(e)) => store_error(e),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "favourites_write_failed"),
    }
}

enum Reopened {
    Done,
    NotBlocked,
    ClockSkew,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReopenForm {
    address: String,
    /// Must be `true`: re-opening is a conscious act.
    confirm: bool,
}

/// «Открыть снова»: the owner's explicit decision to let the bot try a server again after a kick or a ban. It writes `reopened_at`
/// (now, or just after the ban if the clocks disagree) and nothing else: the helper's own memory decides whether that lifts the block.
/// It works only on a favourite that is **closed** right now, so it can never be used to pre-empt a ban.
pub async fn favourite_reopen(
    State(state): State<SharedState>,
    headers: HeaderMap,
    jar: CookieJar,
    body: Bytes,
) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    if !state.servers_edit_gate.try_take() {
        return too_many_edits();
    }
    let form: ReopenForm = match parse_body(&body) {
        Ok(f) => f,
        Err(r) => return refuse(r),
    };
    if !form.confirm {
        return json_error(StatusCode::BAD_REQUEST, "confirm_required");
    }
    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            let blocks = read_blocked(&state.config.status_dir);
            state.favourites.change(|list| {
                let fav = list
                    .favourites
                    .iter_mut()
                    .find(|f| f.address == form.address)
                    .ok_or(StoreError::NotFound)?;
                let Some(block) = closing_block(fav, &blocks) else {
                    return Ok(Reopened::NotBlocked);
                };
                let at = unix_now().max(block.at.saturating_add(1));
                // A ban dated in the future would need a future re-opening, which the helper refuses (review 5.12 F1).
                if at > unix_now().saturating_add(favourites::MAX_FUTURE_SKEW_SECS) {
                    return Ok(Reopened::ClockSkew);
                }
                fav.reopened_at = at;
                Ok(Reopened::Done)
            })
        }
    })
    .await;
    match result {
        Ok(Ok(Reopened::Done)) => {
            tracing::info!(session = %owner.tag, "favourite re-opened from the web");
            Json(serde_json::json!({"ok": true})).into_response()
        }
        Ok(Ok(Reopened::NotBlocked)) => json_error(StatusCode::CONFLICT, "not_blocked"),
        Ok(Ok(Reopened::ClockSkew)) => json_error(StatusCode::CONFLICT, "clock_skew"),
        Ok(Err(e)) => store_error(e),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "favourites_write_failed"),
    }
}

// ---------------------------------------------------------------------------------------------
// proxies
// ---------------------------------------------------------------------------------------------

fn read_check_result(launch_dir: &Path) -> Option<ProxyCheckResult> {
    let bytes = read_regular_nofollow(&launch_dir.join(PROXY_CHECK_RESULT_FILE), MAX_PROXY_CHECK_BYTES).ok()?;
    let r: ProxyCheckResult = serde_json::from_slice(&bytes).ok()?;
    (r.v == 1).then_some(r)
}

/// The age in seconds of a request file somebody left in the launch directory (a regular file only), if there is one.
fn request_age(path: &Path) -> Option<u64> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    Some(
        meta.modified()
            .ok()
            .and_then(|m| std::time::SystemTime::now().duration_since(m).ok())
            .map_or(0, |d| d.as_secs()),
    )
}

pub async fn proxies_get(State(state): State<SharedState>, jar: CookieJar) -> Response {
    if let Err((status, error)) = authorize_get(&state, &jar, false) {
        return json_error(status, error);
    }
    let body = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            let last = state
                .proxycheck_last
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            // A result is shown only for the request the page asked for last.
            let check = read_check_result(&state.config.launch_dir)
                .filter(|r| last.as_ref().is_some_and(|(id, name)| &r.id == id && &r.proxy == name));
            let pending = request_age(&state.config.launch_dir.join(PROXY_CHECK_REQUEST_FILE)).is_some();
            serde_json::json!({
                "enabled": state.config.launch_dir.is_dir(),
                "proxies": state.proxies.list(),
                "check": check,
                "last_check_id": last.map(|(id, _)| id),
                "pending": pending,
            })
        }
    })
    .await;
    match body {
        Ok(b) => Json(b).into_response(),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "proxies_unavailable"),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SaveForm {
    /// `true` makes a new profile (an existing name is refused); `false` edits one the site made.
    create: bool,
    name: String,
    host: String,
    port: u16,
    user: Option<String>,
    pass: Option<String>,
    relay: Option<String>,
    session_pick: Option<u8>,
}

pub async fn proxy_save(State(state): State<SharedState>, headers: HeaderMap, jar: CookieJar, body: Bytes) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    if !state.servers_edit_gate.try_take() {
        return too_many_edits();
    }
    let form: SaveForm = match parse_body(&body) {
        Ok(f) => f,
        Err(r) => return refuse(r),
    };
    let create = form.create;
    let proxy_form = ProxyForm {
        name: form.name,
        host: form.host,
        port: form.port,
        user: form.user,
        pass: form.pass,
        relay: form.relay,
        session_pick: form.session_pick,
    };
    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            if create {
                state.proxies.create(&proxy_form)
            } else {
                state.proxies.update(&proxy_form)
            }
        }
    })
    .await;
    match result {
        Ok(Ok(())) => {
            // Never the profile's values: the kind of change only.
            tracing::info!(session = %owner.tag, create, "proxy profile saved from the web");
            (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response()
        }
        Ok(Err(e)) => proxy_error(e),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "proxy_write_failed"),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NameForm {
    name: String,
}

pub async fn proxy_remove(
    State(state): State<SharedState>,
    headers: HeaderMap,
    jar: CookieJar,
    body: Bytes,
) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    if !state.servers_edit_gate.try_take() {
        return too_many_edits();
    }
    let form: NameForm = match parse_body(&body) {
        Ok(f) => f,
        Err(r) => return refuse(r),
    };
    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || -> Result<(), Refusal> {
            // A proxy a favourite uses stays; and with a favourites file that cannot be read nothing can be proved unused.
            let list = state.favourites.load().map_err(|code| (StatusCode::CONFLICT, code))?;
            if list
                .favourites
                .iter()
                .any(|f| f.proxy_name() == Some(form.name.as_str()))
            {
                return Err((StatusCode::CONFLICT, "proxy_in_use"));
            }
            state.proxies.remove(&form.name).map_err(proxy_refusal)
        }
    })
    .await;
    match result {
        Ok(Ok(())) => {
            tracing::info!(session = %owner.tag, "proxy profile removed from the web");
            Json(serde_json::json!({"ok": true})).into_response()
        }
        Ok(Err(r)) => refuse(r),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "proxy_write_failed"),
    }
}

/// «Проверить»: drops a small request for `ddnet-ai launch check-proxy` (its own unit; the web has no network) and remembers its id.
pub async fn proxy_check(
    State(state): State<SharedState>,
    headers: HeaderMap,
    jar: CookieJar,
    body: Bytes,
) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    if !state.proxycheck_gate.try_take() {
        return json_error(StatusCode::TOO_MANY_REQUESTS, "rate_limited");
    }
    let form: NameForm = match parse_body(&body) {
        Ok(f) => f,
        Err(r) => return refuse(r),
    };
    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || -> Result<String, (StatusCode, &'static str)> {
            let dir = state.config.launch_dir.clone();
            if !dir.is_dir() {
                return Err((StatusCode::SERVICE_UNAVAILABLE, "launcher_unavailable"));
            }
            if !ddai_client::proxy::valid_proxy_name(&form.name) {
                return Err((StatusCode::BAD_REQUEST, "bad_name"));
            }
            if !state.proxies.exists(&form.name) {
                return Err((StatusCode::NOT_FOUND, "proxy_not_found"));
            }
            let path = dir.join(PROXY_CHECK_REQUEST_FILE);
            match request_age(&path) {
                Some(age) if age <= REQUEST_STALE_SECS => return Err((StatusCode::CONFLICT, "pending")),
                // Nobody consumed it for a minute: the check unit is down. Remove what we left, and ask again below.
                Some(_) => {
                    let _ = std::fs::remove_file(&path);
                }
                None => {}
            }
            let id: String = crate::rand_util::random_bytes::<8>()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            let req = ProxyCheckRequest {
                v: crate::launch::PROTOCOL_VERSION,
                id: id.clone(),
                ts: unix_now(),
                proxy: form.name.clone(),
            };
            let bytes =
                serde_json::to_vec(&req).map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "check_write_failed"))?;
            write_atomic(&dir, PROXY_CHECK_REQUEST_FILE, &bytes, 0o644)
                .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "check_write_failed"))?;
            *state
                .proxycheck_last
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((id.clone(), form.name));
            Ok(id)
        }
    })
    .await;
    match result {
        Ok(Ok(id)) => {
            tracing::info!(session = %owner.tag, id = %id, "proxy check requested from the web");
            (StatusCode::ACCEPTED, Json(serde_json::json!({"ok": true, "id": id}))).into_response()
        }
        Ok(Err((status, error))) => json_error(status, error),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "check_write_failed"),
    }
}

// ---------------------------------------------------------------------------------------------
// shared with the launcher routes
// ---------------------------------------------------------------------------------------------

/// What the launcher card may offer next to «Локальный сервер» and the allow-list: the favourites that are valid, with their state.
/// A favourites file that cannot be trusted offers none (and says so).
pub(crate) struct FavouriteChoices {
    pub favourites: Vec<Favourite>,
    pub blocks: Vec<BlockedEntry>,
    pub error: Option<&'static str>,
}

pub(crate) fn favourite_choices(state: &SharedState) -> FavouriteChoices {
    let blocks = read_blocked(&state.config.status_dir);
    match state.favourites.load() {
        Ok(list) => FavouriteChoices {
            favourites: list.favourites,
            blocks,
            error: None,
        },
        Err(code) => FavouriteChoices {
            favourites: Vec::new(),
            blocks,
            error: Some(code),
        },
    }
}
