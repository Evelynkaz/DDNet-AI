//! The launcher routes (task 5.9, D-089), for the «Бот» tab's «Запуск» card:
//!
//! | route | |
//! |---|---|
//! | `GET /api/bot/launch` | what the owner may choose (servers: local, the allow-list and the favourites; brains, durations), the active fly bundle's run name, the helper's last status, whether a request is waiting |
//! | `POST /api/bot/launch` | one start or stop request: validated here against the same choices, then written **atomically** to `<launch-dir>/request.json` |
//!
//! **The web gains no privilege here.** It writes one small file into its own directory and reads another; a root systemd path
//! unit runs `ddnet-ai launch apply`, which re-validates everything against fixed allow-lists and does the `systemctl`. Nothing
//! the browser sends is a path, an address or a command line: the server is `"local"`, the exact `address` of a `ready = true`
//! entry of the owner's `live-servers.toml` (which the web only reads), or (task 5.12, D-099) the exact address of one of the owner's
//! favourites; the helper resolves it again by those files, and a favourite the bot was kicked or banned from stays closed
//! (`409 blocked_after_ban`) until the owner re-opens it on the «Серверы» tab.
//!
//! Protections are those of every mutating route (D-070): the session, the strict same-origin check, the CSRF token and a JSON
//! body ([`crate::http::bot::authorize_post`]); plus a rate limit of its own (one request per `launch_min_gap`) and a refusal while
//! the previous request still waits to be consumed.
//!
//! The status is read from the root-owned `status_dir` (`/run/ddnet-ai`), never from the web's own directory. A request nobody
//! consumed for [`REQUEST_STALE_SECS`] is removed by the web, and the page is told the launcher is down (`launcher_down`).

use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum_extra::extract::cookie::CookieJar;
use serde::Deserialize;

use crate::http::bot::{authorize_get, authorize_post, json_error};
use crate::http::servers::{closing_block, favourite_choices};
use crate::launch::{
    Action, Brain, DurationChoice, Finish, LOCAL_SERVER, LaunchConfig, LaunchRequest, LaunchStatus, MAX_SPARRING,
    MAX_STATUS_BYTES, Mirror, PROTOCOL_VERSION, REQUEST_FILE, REQUEST_STALE_SECS, STATUS_FILE, WbSmart,
    bundle_run_name, read_regular_nofollow, unix_now, write_atomic,
};
use crate::state::SharedState;

/// The allow-list file is a few hundred bytes.
const MAX_LIVE_SERVERS_BYTES: usize = 64 * 1024;

/// The `address` of every `ready = true` entry of the allow-list, in file order. A missing or unreadable file means none.
pub fn ready_servers(path: &Path) -> Vec<String> {
    let Ok(bytes) = read_regular_nofollow(path, MAX_LIVE_SERVERS_BYTES) else {
        return Vec::new();
    };
    let Ok(text) = String::from_utf8(bytes) else {
        return Vec::new();
    };
    let Ok(table) = text.parse::<toml::Table>() else {
        return Vec::new();
    };
    let Some(toml::Value::Array(entries)) = table.get("server") else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|e| e.as_table())
        .filter(|e| e.get("ready").and_then(toml::Value::as_bool) == Some(true))
        .filter_map(|e| e.get("address").and_then(toml::Value::as_str))
        .filter(|a| !a.is_empty() && a.len() <= 64)
        .map(str::to_string)
        .collect()
}

/// The fly bundle the helper will use: the root-owned config's `fly_bundle`, else the default under `<data-dir>`.
fn bundle_path(state: &SharedState) -> std::path::PathBuf {
    read_regular_nofollow(&state.config.launch_config, 4096)
        .ok()
        .and_then(|b| String::from_utf8(b).ok())
        .and_then(|t| toml::from_str::<LaunchConfig>(&t).ok())
        .and_then(|c| c.fly_bundle)
        .unwrap_or_else(|| state.config.data_dir.join(crate::launch::DEFAULT_BUNDLE_REL))
}

/// The opponent-input model the helper will use (task 3.17): `<data-dir>/bot/models/opp-m1.oppnet`.
fn window_model_path(state: &SharedState) -> std::path::PathBuf {
    state.config.data_dir.join(crate::launch::DEFAULT_WINDOW_MODEL_REL)
}

/// The age of the request file, if one is waiting.
fn pending_age(launch_dir: &Path) -> Option<Duration> {
    let meta = std::fs::symlink_metadata(launch_dir.join(REQUEST_FILE)).ok()?;
    Some(
        meta.modified()
            .ok()
            .and_then(|m| SystemTime::now().duration_since(m).ok())
            .unwrap_or_default(),
    )
}

fn read_status(status_dir: &Path) -> Option<LaunchStatus> {
    let bytes = read_regular_nofollow(&status_dir.join(STATUS_FILE), MAX_STATUS_BYTES).ok()?;
    serde_json::from_slice(&bytes).ok()
}

pub async fn launch_get(State(state): State<SharedState>, jar: CookieJar) -> Response {
    // Polled by the page: must not keep the session alive.
    if let Err((status, error)) = authorize_get(&state, &jar, false) {
        return json_error(status, error);
    }
    let dir = state.config.launch_dir.clone();
    let enabled = dir.is_dir();
    let body = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            let bundle = bundle_path(&state);
            let mut servers = vec![serde_json::json!({"id": LOCAL_SERVER, "label": LOCAL_SERVER, "kind": "local"})];
            servers.extend(
                ready_servers(&state.config.live_servers)
                    .into_iter()
                    .map(|a| serde_json::json!({"id": a, "label": a, "kind": "allowlist"})),
            );
            // Task 5.12: the owner's favourites, with whether the bot's ban memory has them closed (the helper decides again).
            let fav = favourite_choices(&state);
            servers.extend(fav.favourites.iter().map(|f| {
                serde_json::json!({
                    "id": f.address,
                    "label": f.address,
                    "kind": "favourite",
                    "name": f.name,
                    "nick": f.nick,
                    "proxy": f.proxy_name(),
                    "blocked": closing_block(f, &fav.blocks).is_some(),
                })
            }));
            let mut pending = pending_age(&dir);
            // A request nobody consumed for too long: the launcher is down. Remove what we left behind, and remember when.
            if pending.is_some_and(|a| a.as_secs() > REQUEST_STALE_SECS) {
                let _ = std::fs::remove_file(dir.join(REQUEST_FILE));
                *state
                    .launch_stalled_at
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(unix_now());
                pending = None;
            }
            let status = if enabled {
                read_status(&state.config.status_dir)
            } else {
                None
            };
            let stalled_at = *state
                .launch_stalled_at
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Down until the helper has written anything newer than the stall.
            let launcher_down = stalled_at.is_some_and(|t| status.as_ref().is_none_or(|s| s.at < t));
            serde_json::json!({
                "enabled": enabled,
                "servers": servers,
                "favourites_error": fav.error,
                "brains": ["hybrid", "hybrid-fly", "fly"],
                "durations": ["15m", "60m", "unlimited"],
                "max_sparring": MAX_SPARRING,
                "bundle": bundle_run_name(&bundle),
                "bundle_present": bundle.is_file(),
                "window_model_present": window_model_path(&state).is_file(),
                "status": status,
                "launcher_down": launcher_down,
                "pending": pending.is_some(),
                "pending_age_s": pending.map(|a| a.as_secs()),
            })
        }
    })
    .await;
    match body {
        Ok(b) => Json(b).into_response(),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "launch_unavailable"),
    }
}

/// What the browser sends. No `v`, no `id`: the web makes those.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchForm {
    action: Action,
    brain: Option<Brain>,
    server: Option<String>,
    duration: Option<DurationChoice>,
    sparring: Option<u8>,
    mirror: Option<Mirror>,
    finish: Option<Finish>,
    wb_smart: Option<WbSmart>,
    no_selfkill: Option<bool>,
    window_model: Option<bool>,
    preinput: Option<bool>,
}

fn new_id() -> String {
    crate::rand_util::random_bytes::<8>()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Checks the form against the same choices the helper will check, and builds the request.
fn build_request(
    form: LaunchForm,
    ready: &[String],
    bundle_ok: bool,
    model_ok: bool,
) -> Result<LaunchRequest, &'static str> {
    let id = new_id();
    match form.action {
        Action::Stop => {
            if form.brain.is_some()
                || form.server.is_some()
                || form.duration.is_some()
                || form.sparring.is_some()
                || form.mirror.is_some()
                || form.finish.is_some()
                || form.wb_smart.is_some()
                || form.no_selfkill.is_some()
                || form.window_model.is_some()
                || form.preinput.is_some()
            {
                return Err("bad_request");
            }
            Ok(LaunchRequest {
                v: PROTOCOL_VERSION,
                id,
                ts: unix_now(),
                action: Action::Stop,
                brain: None,
                server: None,
                duration: None,
                sparring: None,
                mirror: None,
                finish: None,
                wb_smart: None,
                no_selfkill: None,
                window_model: None,
                preinput: None,
            })
        }
        Action::Start => {
            let (Some(brain), Some(server), Some(duration)) = (form.brain, form.server, form.duration) else {
                return Err("bad_request");
            };
            if server != LOCAL_SERVER && !ready.contains(&server) {
                return Err("server_not_allowed");
            }
            let sparring = form.sparring.unwrap_or(0);
            if sparring > MAX_SPARRING {
                return Err("bad_request");
            }
            if sparring > 0 && server != LOCAL_SERVER {
                return Err("sparring_local_only");
            }
            if brain.needs_bundle() && !bundle_ok {
                return Err("bundle_missing");
            }
            // Task 5.13: finishing was measured with the hybrid (and the planner) only; the pure fly gets the target picker's tee as its
            // opponent but was never trained or measured on a held victim, so «off» is the only finishing it takes (same code as the helper).
            if brain == Brain::Fly && form.finish.is_some_and(Finish::is_on) {
                return Err("finish_hybrid_only");
            }
            // Task 3.17 (D-111): the opponent-input model belongs to the hybrid (its lag window); the pure fly has no such window. A model file that
            // is not there is refused here, so the owner sees it at once instead of a bot that does not start.
            if form.window_model == Some(true) {
                if brain == Brain::Fly {
                    return Err("window_model_hybrid_only");
                }
                if !model_ok {
                    return Err("window_model_missing");
                }
            }
            // Task 3.20b (D-112): the server's pre-inputs are played in the hybrid's prediction; the pure fly is not offered them (same code as the helper).
            if form.preinput == Some(true) && brain == Brain::Fly {
                return Err("preinput_hybrid_only");
            }
            Ok(LaunchRequest {
                v: PROTOCOL_VERSION,
                id,
                ts: unix_now(),
                action: Action::Start,
                brain: Some(brain),
                server: Some(server),
                duration: Some(duration),
                sparring: Some(sparring),
                mirror: form.mirror,
                finish: form.finish,
                // Task 5.15: both are bot-level switches (navigation, target choice, self-kills), so every brain takes them.
                wb_smart: form.wb_smart,
                no_selfkill: form.no_selfkill,
                window_model: form.window_model,
                preinput: form.preinput,
            })
        }
    }
}

pub async fn launch_post(
    State(state): State<SharedState>,
    headers: HeaderMap,
    jar: CookieJar,
    body: Bytes,
) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    let form: LaunchForm = match serde_json::from_slice(&body) {
        Ok(f) => f,
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "bad_request"),
    };
    let dir = state.config.launch_dir.clone();
    if !dir.is_dir() {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "launcher_unavailable");
    }
    // Our own rate limit, taken only by a request that is going to be written.
    {
        let mut gate = state
            .launch_gate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while gate.front().is_some_and(|t| t.elapsed() >= Duration::from_secs(60)) {
            gate.pop_front();
        }
        if gate.back().is_some_and(|t| t.elapsed() < state.config.launch_min_gap)
            || gate.len() >= state.config.launch_max_per_minute
        {
            return json_error(StatusCode::TOO_MANY_REQUESTS, "rate_limited");
        }
    }
    let action = form.action;
    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || -> Result<LaunchRequest, (StatusCode, &'static str)> {
            let mut ready = ready_servers(&state.config.live_servers);
            // Task 5.12: a favourite is a server the owner may choose like a ready allow-list entry; one the bot was kicked or banned
            // from stays closed until «Открыть снова» (the helper holds the memory and judges again).
            let fav = favourite_choices(&state);
            ready.extend(fav.favourites.iter().map(|f| f.address.clone()));
            let bundle_ok = bundle_path(&state).is_file();
            let model_ok = window_model_path(&state).is_file();
            let req = build_request(form, &ready, bundle_ok, model_ok).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
            if let Some(server) = req.server.as_deref()
                && fav
                    .favourites
                    .iter()
                    .any(|f| f.address == server && closing_block(f, &fav.blocks).is_some())
            {
                return Err((StatusCode::CONFLICT, "blocked_after_ban"));
            }
            if pending_age(&dir).is_some_and(|a| a.as_secs() <= REQUEST_STALE_SECS) {
                return Err((StatusCode::CONFLICT, "pending"));
            }
            let bytes =
                serde_json::to_vec(&req).map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "launch_write_failed"))?;
            write_atomic(&dir, REQUEST_FILE, &bytes, 0o644)
                .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "launch_write_failed"))?;
            Ok(req)
        }
    })
    .await;
    match result {
        Ok(Ok(req)) => {
            state
                .launch_gate
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push_back(Instant::now());
            // Tags and choices only (the server is logged as local / listed: an address is the owner's, but not needed here).
            tracing::info!(
                session = %owner.tag,
                action = ?action,
                brain = ?req.brain,
                local = req.server.as_deref() == Some(LOCAL_SERVER),
                sparring = ?req.sparring,
                id = %req.id,
                "launch request from the web"
            );
            (
                StatusCode::ACCEPTED,
                Json(serde_json::json!({"ok": true, "id": req.id, "action": action})),
            )
                .into_response()
        }
        Ok(Err((status, error))) => json_error(status, error),
        Err(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "launch_write_failed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(v: serde_json::Value) -> LaunchForm {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn only_ready_entries_are_offered_and_junk_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live-servers.toml");
        std::fs::write(
            &path,
            "[[server]]\naddress=\"1.2.3.4:8303\"\nnick=\"a\"\nready=false\n[[server]]\naddress=\"5.6.7.8:8308\"\nnick=\"b\"\nready=true\n[[server]]\nnick=\"c\"\nready=true\n[[server]]\naddress=\"9.9.9.9:1\"\nready=\"yes\"\n",
        )
        .unwrap();
        assert_eq!(ready_servers(&path), vec!["5.6.7.8:8308".to_string()]);
        assert!(ready_servers(&dir.path().join("nope.toml")).is_empty());
        std::fs::write(&path, "not toml [[[").unwrap();
        assert!(ready_servers(&path).is_empty());
    }

    #[test]
    fn the_form_is_checked_against_the_choices() {
        let ready = vec!["5.6.7.8:8308".to_string()];
        let ok = |v| build_request(form(v), &ready, true, true);
        let r = ok(
            serde_json::json!({"action":"start","brain":"hybrid-fly","server":"local","duration":"15m","sparring":2}),
        )
        .unwrap();
        assert_eq!((r.sparring, r.server.as_deref()), (Some(2), Some("local")));
        assert!(crate::launch::valid_id(&r.id));
        // The request the web builds is one the strict parser accepts.
        assert!(crate::launch::parse_request(&serde_json::to_vec(&r).unwrap()).is_ok());
        assert!(
            ok(serde_json::json!({"action":"start","brain":"hybrid","server":"5.6.7.8:8308","duration":"unlimited"}))
                .is_ok()
        );
        let refused = |v, ready: &[String], bundle| build_request(form(v), ready, bundle, true).unwrap_err();
        assert_eq!(
            refused(
                serde_json::json!({"action":"start","brain":"hybrid","server":"1.2.3.4:8303","duration":"15m"}),
                &ready,
                true
            ),
            "server_not_allowed"
        );
        assert_eq!(
            refused(
                serde_json::json!({"action":"start","brain":"hybrid","server":"5.6.7.8:8308","duration":"15m","sparring":1}),
                &ready,
                true
            ),
            "sparring_local_only"
        );
        assert_eq!(
            refused(
                serde_json::json!({"action":"start","brain":"hybrid","server":"local","duration":"15m","sparring":4}),
                &ready,
                true
            ),
            "bad_request"
        );
        assert_eq!(
            refused(
                serde_json::json!({"action":"start","brain":"fly","server":"local","duration":"15m"}),
                &ready,
                false
            ),
            "bundle_missing"
        );
        assert_eq!(
            refused(serde_json::json!({"action":"start","brain":"hybrid"}), &ready, true),
            "bad_request"
        );
        // The opponent-model switch (D-090): `on`, `off` or absent; anything else, or on a stop, is refused.
        let m = |v: &str| {
            ok(serde_json::json!({"action":"start","brain":"hybrid","server":"local","duration":"15m","mirror":v}))
        };
        assert_eq!(m("off").unwrap().mirror, Some(Mirror::Off));
        assert_eq!(m("on").unwrap().mirror, Some(Mirror::On));
        assert_eq!(
            ok(serde_json::json!({"action":"start","brain":"hybrid","server":"local","duration":"15m"}))
                .unwrap()
                .mirror,
            None
        );
        assert!(serde_json::from_value::<LaunchForm>(serde_json::json!({"action":"start","mirror":"maybe"})).is_err());
        assert_eq!(
            refused(serde_json::json!({"action":"stop","mirror":"off"}), &ready, true),
            "bad_request"
        );
        assert_eq!(
            refused(serde_json::json!({"action":"stop","server":"local"}), &ready, true),
            "bad_request"
        );
        assert!(ok(serde_json::json!({"action":"stop"})).is_ok());
    }

    #[test]
    fn the_finishing_switch_is_a_closed_list_and_the_pure_fly_takes_only_off() {
        let ready: Vec<String> = Vec::new();
        let start = |brain: &str, finish: serde_json::Value| {
            let mut v = serde_json::json!({"action":"start","brain":brain,"server":"local","duration":"15m"});
            if !finish.is_null() {
                v["finish"] = finish;
            }
            serde_json::from_value::<LaunchForm>(v).map(|f| build_request(f, &ready, true, true))
        };
        for brain in ["hybrid", "hybrid-fly"] {
            for (word, want) in [
                ("off", Finish::Off),
                ("target", Finish::Target),
                ("wb", Finish::Wb),
                ("full", Finish::Full),
            ] {
                let req = start(brain, serde_json::json!(word)).unwrap().unwrap();
                assert_eq!(req.finish, Some(want), "{brain} {word}");
            }
            assert_eq!(
                start(brain, serde_json::Value::Null).unwrap().unwrap().finish,
                None,
                "absent stays absent"
            );
        }
        // The pure fly: off (or nothing) only.
        assert_eq!(
            start("fly", serde_json::json!("off")).unwrap().unwrap().finish,
            Some(Finish::Off)
        );
        assert_eq!(start("fly", serde_json::Value::Null).unwrap().unwrap().finish, None);
        for word in ["target", "wb", "full"] {
            assert_eq!(
                start("fly", serde_json::json!(word)).unwrap().unwrap_err(),
                "finish_hybrid_only",
                "{word}"
            );
        }
        // Anything outside the list never reaches `build_request`.
        for bad in [
            "on",
            "Target",
            "TARGET",
            "target ",
            "WB",
            "wb ",
            "maybe",
            "",
            "target --x",
            "wb --wb left",
        ] {
            assert!(start("hybrid", serde_json::json!(bad)).is_err(), "{bad:?}");
        }
        // A stop carries no finishing.
        let stop = serde_json::from_value::<LaunchForm>(serde_json::json!({"action":"stop","finish":"off"})).unwrap();
        assert_eq!(build_request(stop, &ready, true, true).unwrap_err(), "bad_request");
    }

    #[test]
    fn the_smart_wayblock_and_the_duel_switch_are_closed_values_and_every_brain_takes_them() {
        let ready: Vec<String> = Vec::new();
        let start = |brain: &str, key: &str, val: serde_json::Value| {
            let mut v = serde_json::json!({"action":"start","brain":brain,"server":"local","duration":"15m"});
            if !val.is_null() {
                v[key] = val;
            }
            serde_json::from_value::<LaunchForm>(v).map(|f| build_request(f, &ready, true, true))
        };
        // The pure fly takes both too: they are navigation, target choice and self-kill rules the bot runs under every brain.
        for brain in ["hybrid", "hybrid-fly", "fly"] {
            for (word, want) in [("off", WbSmart::Off), ("on", WbSmart::On)] {
                let req = start(brain, "wb_smart", serde_json::json!(word)).unwrap().unwrap();
                assert_eq!(req.wb_smart, Some(want), "{brain} {word}");
            }
            for want in [false, true] {
                let req = start(brain, "no_selfkill", serde_json::json!(want)).unwrap().unwrap();
                assert_eq!(req.no_selfkill, Some(want), "{brain} {want}");
            }
            let bare = start(brain, "wb_smart", serde_json::Value::Null).unwrap().unwrap();
            assert_eq!((bare.wb_smart, bare.no_selfkill), (None, None), "absent stays absent");
        }
        // Closed values: never reach `build_request`.
        for bad in ["true", "On", "ON", "on ", "maybe", "", "on --x", "$(id)"] {
            assert!(start("hybrid", "wb_smart", serde_json::json!(bad)).is_err(), "{bad:?}");
        }
        for bad in [
            serde_json::json!("true"),
            serde_json::json!("false"),
            serde_json::json!(1),
            serde_json::json!(""),
        ] {
            assert!(start("hybrid", "no_selfkill", bad.clone()).is_err(), "{bad}");
        }
        // A stop carries neither.
        for (key, val) in [
            ("wb_smart", serde_json::json!("off")),
            ("no_selfkill", serde_json::json!(false)),
        ] {
            let stop = serde_json::from_value::<LaunchForm>(serde_json::json!({"action":"stop", key: val})).unwrap();
            assert_eq!(
                build_request(stop, &ready, true, true).unwrap_err(),
                "bad_request",
                "{key}"
            );
        }
    }

    #[test]
    fn the_opponent_predictor_is_a_boolean_for_the_hybrids_only_and_needs_its_file() {
        // Task 3.17 (D-111).
        let ready: Vec<String> = Vec::new();
        let start = |brain: &str, val: serde_json::Value, model_ok: bool| {
            let mut v = serde_json::json!({"action":"start","brain":brain,"server":"local","duration":"15m"});
            if !val.is_null() {
                v["window_model"] = val;
            }
            serde_json::from_value::<LaunchForm>(v).map(|f| build_request(f, &ready, true, model_ok))
        };
        for brain in ["hybrid", "hybrid-fly"] {
            for want in [false, true] {
                let req = start(brain, serde_json::json!(want), true).unwrap().unwrap();
                assert_eq!(req.window_model, Some(want), "{brain} {want}");
            }
            let bare = start(brain, serde_json::Value::Null, true).unwrap().unwrap();
            assert_eq!(bare.window_model, None, "absent stays absent");
        }
        // The pure fly has no lag window to put the model in; «off» is harmless, «on» is refused.
        assert_eq!(
            start("fly", serde_json::json!(true), true).unwrap().unwrap_err(),
            "window_model_hybrid_only"
        );
        assert_eq!(
            start("fly", serde_json::json!(false), true)
                .unwrap()
                .unwrap()
                .window_model,
            Some(false)
        );
        // A model that is not on the disk is refused when asked for, never when not.
        assert_eq!(
            start("hybrid", serde_json::json!(true), false).unwrap().unwrap_err(),
            "window_model_missing"
        );
        assert!(start("hybrid", serde_json::json!(false), false).unwrap().is_ok());
        assert!(start("hybrid", serde_json::Value::Null, false).unwrap().is_ok());
        // Nothing but a JSON boolean ever gets as far as `build_request`: no path, no word.
        for bad in [
            serde_json::json!("true"),
            serde_json::json!("on"),
            serde_json::json!("/etc/passwd"),
            serde_json::json!(1),
            serde_json::json!(""),
        ] {
            assert!(start("hybrid", bad.clone(), true).is_err(), "{bad}");
        }
        // A stop carries nothing.
        let stop =
            serde_json::from_value::<LaunchForm>(serde_json::json!({"action":"stop","window_model":false})).unwrap();
        assert_eq!(build_request(stop, &ready, true, true).unwrap_err(), "bad_request");
    }

    #[test]
    fn the_preinput_toggle_is_a_boolean_for_the_hybrids_only() {
        // Task 3.20b (D-112). No file to check: the switch names nothing but itself.
        let ready: Vec<String> = Vec::new();
        let start = |brain: &str, val: serde_json::Value| {
            let mut v = serde_json::json!({"action":"start","brain":brain,"server":"local","duration":"15m"});
            if !val.is_null() {
                v["preinput"] = val;
            }
            serde_json::from_value::<LaunchForm>(v).map(|f| build_request(f, &ready, true, false))
        };
        for brain in ["hybrid", "hybrid-fly"] {
            for want in [false, true] {
                let req = start(brain, serde_json::json!(want)).unwrap().unwrap();
                assert_eq!(req.preinput, Some(want), "{brain} {want}");
            }
            let bare = start(brain, serde_json::Value::Null).unwrap().unwrap();
            assert_eq!(bare.preinput, None, "absent stays absent");
        }
        // The pure fly: `false` is harmless, `true` is refused with its own code.
        assert_eq!(
            start("fly", serde_json::json!(true)).unwrap().unwrap_err(),
            "preinput_hybrid_only"
        );
        assert_eq!(
            start("fly", serde_json::json!(false)).unwrap().unwrap().preinput,
            Some(false)
        );
        // Nothing but a JSON boolean ever gets as far as `build_request`: no word, no path.
        for bad in [
            serde_json::json!("true"),
            serde_json::json!("on"),
            serde_json::json!("off"),
            serde_json::json!("/etc/passwd"),
            serde_json::json!(1),
            serde_json::json!(""),
        ] {
            assert!(start("hybrid", bad.clone()).is_err(), "{bad}");
        }
        // A stop carries nothing.
        let stop = serde_json::from_value::<LaunchForm>(serde_json::json!({"action":"stop","preinput":false})).unwrap();
        assert_eq!(build_request(stop, &ready, true, true).unwrap_err(), "bad_request");
    }
}
