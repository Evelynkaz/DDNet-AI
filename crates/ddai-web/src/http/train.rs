//! The owner-only, **read-only** training routes (task 5.8, D-077):
//!
//! | route | |
//! |---|---|
//! | `GET /api/train/runs` | experiments and runs with their state (running / done / not updating), phase and step |
//! | `GET /api/train/run?exp=…&run=…` | one run: curves, DAgger rounds, arena results with Wilson intervals, checkpoints, summaries |
//!
//! Both need the session (`401` without one) and sit under `no-store` like every `/api/*` route. They are `GET`s that change
//! nothing, so there is no CSRF token or Origin check to add; a request that carries `poll=1` is the page's timer (it does
//! not refresh the session's idle timeout, exactly like the bot status poll). Names in the query are identifiers only
//! ([`crate::training::paths`]); what is read, how much and where from is the business of [`crate::training`]. Nothing here
//! logs a name from the request.

use axum::Json;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum_extra::extract::cookie::CookieJar;
use serde::{Deserialize, Serialize};

use crate::session_guard::{current_session, peek_session};
use crate::state::SharedState;
use crate::training::{DetailError, Listing, RunDetail, ScanError, unix_now};

/// How often the page should refresh a running job (the page's own constant; sent so the two cannot drift apart).
pub const POLL_SECS: u64 = 10;

fn json_error(status: StatusCode, error: &'static str) -> Response {
    (status, Json(serde_json::json!({ "error": error }))).into_response()
}

/// Session check; a person's request refreshes the idle timeout, the page's timer (`poll=1`) does not.
fn authorized(state: &SharedState, jar: &CookieJar, poll: bool) -> bool {
    if poll {
        peek_session(state, jar).is_some()
    } else {
        current_session(state, jar).is_some()
    }
}

fn is_poll(flag: Option<&str>) -> bool {
    flag == Some("1")
}

#[derive(Deserialize)]
pub struct ListQuery {
    poll: Option<String>,
}

#[derive(Serialize)]
struct ListOut {
    generated_unix: u64,
    poll_secs: u64,
    #[serde(flatten)]
    listing: Listing,
}

fn scan_error(e: ScanError) -> Response {
    match e {
        ScanError::Busy => json_error(StatusCode::SERVICE_UNAVAILABLE, "busy"),
        ScanError::Failed => json_error(StatusCode::INTERNAL_SERVER_ERROR, "scan_failed"),
    }
}

/// The queries are `Result<Query<…>, QueryRejection>` so that the session is checked first: a malformed query string (a repeated key, say)
/// is answered `401` JSON without a session and `400 bad_name` JSON with one, never with the extractor's plain-text 400.
pub async fn runs(
    State(state): State<SharedState>,
    jar: CookieJar,
    q: Result<Query<ListQuery>, QueryRejection>,
) -> Response {
    let poll = q.as_ref().is_ok_and(|Query(q)| is_poll(q.poll.as_deref()));
    if !authorized(&state, &jar, poll) {
        return json_error(StatusCode::UNAUTHORIZED, "unauthenticated");
    }
    let now = unix_now();
    match state.training.scan(move |store| store.list(now)).await {
        Ok(listing) => Json(ListOut {
            generated_unix: now,
            poll_secs: POLL_SECS,
            listing,
        })
        .into_response(),
        Err(e) => scan_error(e),
    }
}

#[derive(Deserialize)]
pub struct RunQuery {
    exp: Option<String>,
    run: Option<String>,
    poll: Option<String>,
}

#[derive(Serialize)]
struct RunOut {
    generated_unix: u64,
    poll_secs: u64,
    #[serde(flatten)]
    detail: RunDetail,
}

pub async fn run(
    State(state): State<SharedState>,
    jar: CookieJar,
    q: Result<Query<RunQuery>, QueryRejection>,
) -> Response {
    let poll = q.as_ref().is_ok_and(|Query(q)| is_poll(q.poll.as_deref()));
    if !authorized(&state, &jar, poll) {
        return json_error(StatusCode::UNAUTHORIZED, "unauthenticated");
    }
    let Ok(Query(RunQuery {
        exp: Some(exp),
        run: Some(run),
        ..
    })) = q
    else {
        return json_error(StatusCode::BAD_REQUEST, "bad_name");
    };
    let now = unix_now();
    match state
        .training
        .scan(move |store| store.run_detail(&exp, &run, now))
        .await
    {
        Ok(Ok(detail)) => Json(RunOut {
            generated_unix: now,
            poll_secs: POLL_SECS,
            detail,
        })
        .into_response(),
        Ok(Err(DetailError::BadName)) => json_error(StatusCode::BAD_REQUEST, "bad_name"),
        Ok(Err(DetailError::NotFound)) => json_error(StatusCode::NOT_FOUND, "not_found"),
        Err(e) => scan_error(e),
    }
}
