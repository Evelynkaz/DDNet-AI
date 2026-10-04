//! Shared server state, handed to every handler through axum's `State` extractor.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::{Semaphore, broadcast};

use crate::auth::device::DeviceStore;
use crate::auth::rate_limit::LoginRateLimiter;
use crate::auth::session::{SessionId, SessionStore};
use crate::config::WebConfig;
use crate::control::client::ControlClient;
use crate::control::relations::RelationsStore;
use crate::live::hub::LiveHub;
use crate::secrets::SecretsPaths;
use crate::training::{Limits, TrainingStore};

/// Everything handlers need, wrapped once in an `Arc` and cloned cheaply per request.
pub type SharedState = Arc<AppState>;

/// Argon2id verification (review finding F4) runs on tokio's blocking thread pool, bounded by
/// this many concurrent hashes — at the production params (64 MiB) that's ~128 MiB of memory
/// even under a login flood, and it stops a burst of concurrent login attempts from starving the
/// async worker threads (and therefore every other request, including unrelated `GET /api/me`
/// calls) the way running argon2 inline on a worker thread did.
const ARGON2_MAX_CONCURRENT: usize = 2;

/// Capacity of the session-invalidation broadcast channel (review finding F2): how many logouts
/// can be pending delivery to WebSocket tasks before a slow subscriber starts missing individual
/// notices (it still has the per-tick fallback check, see `ws.rs`). Comfortably larger than any
/// realistic burst for a single-owner admin panel.
const SESSION_INVALIDATED_CHANNEL_CAPACITY: usize = 64;

pub struct AppState {
    pub config: WebConfig,
    pub secrets_paths: SecretsPaths,
    /// HMAC key for the session cookie (acceptance criterion 2). Loaded once at startup; a
    /// server restart is required to pick up a rotated key (which invalidates all sessions
    /// anyway, so there is nothing to "hot reload" here).
    pub session_key: [u8; 32],
    pub sessions: SessionStore,
    /// Trusted devices that may skip the global login rate limit (review finding F7).
    pub devices: DeviceStore,
    pub login_rate_limiter: LoginRateLimiter,
    pub started_at: Instant,
    /// Count of currently open WebSocket connections per session id, for the "max concurrent WS
    /// per session" limit (acceptance criterion 5).
    pub ws_conns_per_session: Mutex<HashMap<SessionId, u32>>,
    /// Bounds concurrent argon2id verification (review finding F4).
    pub argon2_semaphore: Semaphore,
    /// Published to whenever a session is explicitly invalidated (logout), so any open WebSocket
    /// for that session can close itself immediately instead of waiting for its next periodic
    /// validity check (review finding F2). Session *expiry* (idle/absolute timeout elapsing) has
    /// no corresponding event — those are caught by the WS task's own periodic check.
    pub session_invalidated: broadcast::Sender<SessionId>,
    /// Task 5.2a: the live map view's `FrameSource` hub, when `config.replay_source` was
    /// configured — `None` otherwise (the WS still works; it just never gets `map`/`live`
    /// messages). See `crate::live::hub`.
    pub live_hub: Option<Arc<LiveHub>>,
    /// Task 5.6: the client of the bot's control socket (connect only).
    pub control: ControlClient,
    /// Task 5.6: the friend / war / ignore lists file.
    pub relations: RelationsStore,
    /// Task 5.8: the read-only view of the training runs directory.
    pub training: Arc<TrainingStore>,
    /// Task 5.9: when the launcher requests of the last minute were accepted (the web's own rate limit, on top of the helper's).
    pub launch_gate: Mutex<std::collections::VecDeque<Instant>>,
    /// Task 5.9: unix time when a request nobody consumed was found and removed (the launcher is down until a newer status).
    pub launch_stalled_at: Mutex<Option<u64>>,
}

impl AppState {
    /// Constructs state with no live-view source attached (`live_hub: None`) — used by every
    /// existing test that doesn't care about the live map view, and equivalent to
    /// [`AppState::new_with_live_hub`] with `live_hub: None`.
    pub fn new(config: WebConfig, session_key: [u8; 32]) -> Self {
        Self::new_with_live_hub(config, session_key, None)
    }

    pub fn new_with_live_hub(config: WebConfig, session_key: [u8; 32], live_hub: Option<Arc<LiveHub>>) -> Self {
        let secrets_paths = SecretsPaths::new(&config.data_dir);
        let sessions = SessionStore::new(config.idle_timeout, config.absolute_timeout);
        // Review round 2, finding F8a: loads any previously-trusted devices from disk so a
        // restart/redeploy doesn't strand the owner without their one way past the global login
        // rate limit — see `auth::device`'s module doc comment.
        let devices = DeviceStore::load_or_empty(&secrets_paths, config.trusted_device_ttl);
        let login_rate_limiter = LoginRateLimiter::new(config.login_rate_limit);
        let (session_invalidated, _rx) = broadcast::channel(SESSION_INVALIDATED_CHANNEL_CAPACITY);
        let control = ControlClient::new(config.control_socket.clone());
        let relations = RelationsStore::new(config.relations_path.clone());
        let training = Arc::new(TrainingStore::new(config.runs_dir.clone(), Limits::default()));
        Self {
            config,
            secrets_paths,
            session_key,
            sessions,
            devices,
            login_rate_limiter,
            started_at: Instant::now(),
            ws_conns_per_session: Mutex::new(HashMap::new()),
            argon2_semaphore: Semaphore::new(ARGON2_MAX_CONCURRENT),
            session_invalidated,
            live_hub,
            control,
            relations,
            training,
            launch_gate: Mutex::new(std::collections::VecDeque::new()),
            launch_stalled_at: Mutex::new(None),
        }
    }

    pub fn uptime_secs(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    /// Invalidates a session and notifies any open WebSocket for it to close (review finding
    /// F2). Returns `true` if the session existed. The single call site every handler should use
    /// instead of `self.sessions.invalidate` directly, so the broadcast is never forgotten.
    pub fn invalidate_session(&self, id: SessionId) -> bool {
        let existed = self.sessions.invalidate(&id);
        // No receivers (e.g. no open WS for this session) is not an error — `send` returning
        // `Err` just means nobody was listening.
        let _ = self.session_invalidated.send(id);
        existed
    }

    /// Tries to reserve one WebSocket connection slot for `session_id`. Returns `true` if the
    /// caller may proceed (and must eventually call [`AppState::release_ws_slot`]); `false` if
    /// the session is already at [`crate::config::WebConfig::max_ws_per_session`].
    pub fn try_reserve_ws_slot(&self, session_id: SessionId) -> bool {
        let mut counts = self
            .ws_conns_per_session
            .lock()
            .expect("ws conn counter mutex poisoned");
        let count = counts.entry(session_id).or_insert(0);
        if *count >= self.config.max_ws_per_session {
            false
        } else {
            *count += 1;
            true
        }
    }

    pub fn release_ws_slot(&self, session_id: SessionId) {
        let mut counts = self
            .ws_conns_per_session
            .lock()
            .expect("ws conn counter mutex poisoned");
        if let Some(count) = counts.get_mut(&session_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                counts.remove(&session_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rand_util::random_bytes;
    use std::net::{Ipv4Addr, SocketAddr};
    use std::path::PathBuf;

    fn state() -> AppState {
        let mut config = WebConfig::new(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            PathBuf::from("/tmp/ddai-web-state-test"),
        );
        config.max_ws_per_session = 2;
        AppState::new(config, random_bytes())
    }

    #[test]
    fn ws_slot_reservation_respects_limit() {
        let state = state();
        let session: SessionId = random_bytes();
        assert!(state.try_reserve_ws_slot(session));
        assert!(state.try_reserve_ws_slot(session));
        assert!(!state.try_reserve_ws_slot(session), "third slot should be refused");
        state.release_ws_slot(session);
        assert!(
            state.try_reserve_ws_slot(session),
            "releasing a slot should free capacity"
        );
    }

    #[test]
    fn ws_slots_are_independent_per_session() {
        let state = state();
        let a: SessionId = random_bytes();
        let b: SessionId = random_bytes();
        assert!(state.try_reserve_ws_slot(a));
        assert!(state.try_reserve_ws_slot(a));
        assert!(
            state.try_reserve_ws_slot(b),
            "a different session must have its own budget"
        );
    }

    #[test]
    fn releasing_below_zero_does_not_underflow() {
        let state = state();
        let session: SessionId = random_bytes();
        state.release_ws_slot(session); // no reservation was ever made
        assert!(state.try_reserve_ws_slot(session));
    }
}
