//! `GET /api/map/<sha256>` (acceptance criterion 1): the classified, compressed scene for a map
//! the live source has already reported (via a `map` WS message) — authenticated, like every
//! other route.
//!
//! **Cache policy.** A map's classified [`crate::live::scene::MapScene`] is immutable for a given
//! sha256 (the same input bytes always classify to the same output, `MapScene::build` is a pure
//! function) — unlike every other `/api/*` route, this one is deliberately *not* under
//! `headers::no_store` (see `server::build_router`'s doc comment at the merge site). Instead it
//! sends `ETag: "<sha256-hex>"` and `Cache-Control: private, max-age=604800, immutable`:
//! `private` because this is still behind login, not a public CDN-cacheable resource; `immutable`
//! and a week-long `max-age` because the content genuinely never changes for a given URL, so a
//! browser (or this same client reconnecting later) never needs to re-fetch it at all, and a
//! conditional `If-None-Match` revalidation (if the browser ever asks anyway) always resolves to
//! `304` for free.

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum_extra::extract::cookie::CookieJar;

use crate::live::scene::encode_compressed;
use crate::session_guard::current_session;
use crate::state::SharedState;

fn decode_hex_sha256(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(hex.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

pub async fn get_map(
    State(state): State<SharedState>,
    Path(sha256_hex): Path<String>,
    headers: HeaderMap,
    jar: CookieJar,
) -> Response {
    if current_session(&state, &jar).is_none() {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let Some(sha256) = decode_hex_sha256(&sha256_hex) else {
        return StatusCode::BAD_REQUEST.into_response();
    };

    let Some(hub) = &state.live_hub else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Some(scene) = hub.map_cache.get(&sha256) else {
        return StatusCode::NOT_FOUND.into_response();
    };

    let etag = format!("\"{sha256_hex}\"");
    if headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == etag)
    {
        return StatusCode::NOT_MODIFIED.into_response();
    }

    // Review round 1, finding F11: `encode_compressed` runs `Compression::best` over the whole
    // `kinds` grid — measured at ~30ms for `BlmapChill` — which used to run synchronously on this
    // request's own async runtime worker thread on *every* request, blocking whatever else that
    // thread could otherwise be doing (e.g. another connection's WS message loop) even though the
    // result is 100% deterministic in `sha256` and therefore identical on every call. Cache the
    // compressed bytes (`MapCache::get_compressed`/`insert_compressed`, keyed by the same sha256
    // as the scene itself) so only the very first request for a given map pays the cost at all,
    // and run even that first compression on the blocking-task pool (`spawn_blocking`) rather
    // than the async worker thread, so it never competes with other connections' scheduling.
    let body = match hub.map_cache.get_compressed(&sha256) {
        Some(cached) => cached,
        None => {
            let scene_for_blocking = scene.clone();
            // `encode_compressed` only ever fails via an `expect()` on in-memory `Vec` I/O, which
            // never actually fails (see its own doc comment) — a `JoinError` here would mean that
            // invariant broke (a genuine panic), and there is no safe compressed body to fall
            // back to, so this reports it as a server error rather than unwrapping past it.
            let Ok(compressed) = tokio::task::spawn_blocking(move || encode_compressed(&scene_for_blocking)).await
            else {
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            };
            hub.map_cache.insert_compressed(sha256, compressed)
        }
    };
    (
        [
            (header::CONTENT_TYPE, "application/octet-stream".to_string()),
            (header::ETAG, etag),
            (header::CACHE_CONTROL, "private, max-age=604800, immutable".to_string()),
        ],
        // `body` is the `Arc<Vec<u8>>` cached above — axum's `IntoResponse` has no impl for that
        // type directly, only for `Vec<u8>`, so this clones the (already-compressed, so small:
        // low tens of KB at most for this corpus's maps) final bytes out of the `Arc`. That clone
        // is a plain `memcpy`, nothing like the `Compression::best` pass the cache above exists
        // to avoid — copying a few KB costs microseconds, not the ~30ms this finding is about.
        (*body).clone(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_hex_sha256_round_trips() {
        let hex = "ab".repeat(32);
        let decoded = decode_hex_sha256(&hex).expect("should decode");
        assert_eq!(decoded, [0xab; 32]);
    }

    #[test]
    fn decode_hex_sha256_rejects_wrong_length() {
        assert_eq!(decode_hex_sha256("abcd"), None);
    }

    #[test]
    fn decode_hex_sha256_rejects_non_hex_characters() {
        let bad = "zz".repeat(32);
        assert_eq!(decode_hex_sha256(&bad), None);
    }
}
