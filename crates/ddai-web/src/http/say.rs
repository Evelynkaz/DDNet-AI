//! `POST /api/bot/say` (task 4.9, D-094): the owner types a line on the authenticated website and the bot says it in the game chat.
//!
//! The only chat the bot may send is the typed `/kill` fallback (D-078) and this. Nothing is automatic. Since task 4.9b (the owner's
//! decision of 2026-10-05, amending D-094) a line may start with `/`: it is then a server command (`/spec`, `/emote`, `/w`, ...) typed by
//! the owner, and goes through exactly the same checks and pacing as any other line.
//!
//! **Behind everything the other bot POSTs are behind** (D-070): the session (`401`), the strict same-origin check (`403`), the CSRF
//! token (`403`) and a JSON body (`415`), in that order ([`super::bot::authorize_post`]).
//!
//! **Validated twice.** Here, with `OwnerText::normalise` (the very rules the bot applies, and no sendable value is made here: only the
//! bot's dispatcher holds the `OwnerChannel` that makes an `OwnerText`; not empty, at most 255 bytes, no control or invisible
//! character or line break, not the server's bot-trap line, trimmed), before anything is sent; and again by the bot, which also paces the lines (3 s
//! apart, 10 a minute, a queue of 3) and says nothing while it is not in the game. What goes to the control socket is the trimmed text.
//!
//! **Its own rate limit**, kept here ([`crate::config::WebConfig::say_burst`] in [`say_burst_window`], and
//! [`say_max_per_minute`] a minute): a request that passed validation takes a slot before it is sent. A line the validation refuses
//! costs nothing (a typo is not a flood).
//!
//! **Privacy.** The text is not logged, anywhere: the log line carries the session's audit tag, the team flag, the length and the
//! outcome. The bot's answers never contain it either.
//!
//! [`say_burst_window`]: crate::config::WebConfig::say_burst_window
//! [`say_max_per_minute`]: crate::config::WebConfig::say_max_per_minute

use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum_extra::extract::cookie::CookieJar;
use ddai_botctl::proto::{ControlCommand, OwnerText, OwnerTextError, ReplyCode, SayText};
use serde::Deserialize;

use super::bot::{authorize_post, demo_only, json_error, json_error_detail};
use crate::control::client::SendError;
use crate::state::SharedState;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SayForm {
    /// Team chat instead of all chat.
    #[serde(default)]
    team: bool,
    text: String,
}

/// The `detail` of a refused text: a fixed word per reason, never the text.
fn detail_of(e: OwnerTextError) -> &'static str {
    match e {
        OwnerTextError::Empty => "empty",
        OwnerTextError::TooLong => "too_long",
        OwnerTextError::Control => "control",
        OwnerTextError::Reserved => "reserved",
    }
}

/// Takes a slot of the web's own rate limit, or says no.
fn take_slot(state: &SharedState, now: Instant) -> bool {
    let mut gate = state.say_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    while gate
        .front()
        .is_some_and(|t| now.saturating_duration_since(*t) >= Duration::from_secs(60))
    {
        gate.pop_front();
    }
    let in_burst = gate
        .iter()
        .filter(|t| now.saturating_duration_since(**t) < state.config.say_burst_window)
        .count();
    if in_burst >= state.config.say_burst || gate.len() >= state.config.say_max_per_minute {
        return false;
    }
    gate.push_back(now);
    true
}

pub async fn say(State(state): State<SharedState>, headers: HeaderMap, jar: CookieJar, body: Bytes) -> Response {
    let owner = match authorize_post(&state, &headers, &jar) {
        Ok(o) => o,
        Err((status, error)) => return json_error(status, error),
    };
    let form: SayForm = match serde_json::from_slice(&body) {
        Ok(f) => f,
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "bad_request"),
    };
    // The web's check, with the constructor the bot uses too.
    let text = match OwnerText::normalise(&form.text) {
        Ok(t) => t,
        Err(e) => return json_error_detail(StatusCode::BAD_REQUEST, "invalid_text", detail_of(e)),
    };
    // The live bot's command: with the demo on show and no live bot there is nobody to say it.
    if demo_only(&state).await {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "demo_only");
    }
    if !take_slot(&state, Instant::now()) {
        tracing::info!(session = %owner.tag, team = form.team, len = text.len(), "owner chat refused by the web's rate limit");
        return json_error(StatusCode::TOO_MANY_REQUESTS, "rate_limited");
    }
    let cmd = ControlCommand::Say {
        team: form.team,
        text: SayText::new(text),
    };
    match state.control.send(&owner.tag, cmd).await {
        Ok(reply) => {
            let reason = reply
                .data
                .as_ref()
                .and_then(|d| d.get("reason"))
                .and_then(|r| r.as_str())
                .map(str::to_owned);
            let status = if reply.ok {
                StatusCode::OK
            } else {
                match (reply.code, reason.as_deref()) {
                    (Some(ReplyCode::BadRequest), _) => StatusCode::BAD_REQUEST,
                    (Some(ReplyCode::RateLimited), _) | (_, Some("queue_full" | "rate_limited")) => {
                        StatusCode::TOO_MANY_REQUESTS
                    }
                    (_, Some("not_in_game" | "chat_disabled")) => StatusCode::CONFLICT,
                    (Some(ReplyCode::Busy | ReplyCode::Gone), _) => StatusCode::SERVICE_UNAVAILABLE,
                    (Some(ReplyCode::Timeout), _) => StatusCode::GATEWAY_TIMEOUT,
                    _ => StatusCode::CONFLICT,
                }
            };
            tracing::info!(
                session = %owner.tag,
                team = form.team,
                len = text.len(),
                ok = reply.ok,
                reason = reason.as_deref().unwrap_or("-"),
                "owner chat from the web"
            );
            (
                status,
                Json(serde_json::json!({
                    "ok": reply.ok,
                    "text": reply.text,
                    "reason": reason,
                })),
            )
                .into_response()
        }
        Err(e) => {
            tracing::info!(session = %owner.tag, team = form.team, len = text.len(), error = %e, "owner chat from the web failed");
            match e {
                SendError::Unavailable => json_error(StatusCode::SERVICE_UNAVAILABLE, "bot_unavailable"),
                SendError::Timeout => json_error(StatusCode::GATEWAY_TIMEOUT, "bot_timeout"),
                SendError::Protocol => json_error(StatusCode::BAD_GATEWAY, "bot_protocol"),
            }
        }
    }
}
