//! The server-side session table: `id -> {expires, csrf token, created_ip}` (acceptance
//! criterion 2). Deliberately not `tower-sessions`' `MemoryStore`, which never evicts expired
//! entries (see `docs/research/rust-stack.md` §1) — this store enforces both an idle timeout and
//! an absolute timeout on every lookup, plus an explicit [`SessionStore::purge_expired`] for the
//! periodic background sweep.
//!
//! Sessions live only in memory: restarting the server invalidates every session, which is fine
//! for a single-owner admin panel and simpler than persisting session state across restarts.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::rand_util::random_bytes;

pub type SessionId = [u8; 32];
pub type CsrfToken = [u8; 32];

struct SessionRecord {
    csrf_token: CsrfToken,
    created_ip: IpAddr,
    absolute_deadline: Instant,
    idle_deadline: Instant,
}

/// Info returned about a still-valid session.
#[derive(Debug, Clone, Copy)]
pub struct SessionInfo {
    pub csrf_token: CsrfToken,
    pub created_ip: IpAddr,
}

/// A freshly created session.
#[derive(Debug, Clone, Copy)]
pub struct NewSession {
    pub id: SessionId,
    pub csrf_token: CsrfToken,
}

pub struct SessionStore {
    sessions: Mutex<HashMap<SessionId, SessionRecord>>,
    idle_timeout: Duration,
    absolute_timeout: Duration,
}

impl SessionStore {
    pub fn new(idle_timeout: Duration, absolute_timeout: Duration) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            idle_timeout,
            absolute_timeout,
        }
    }

    /// Creates a new session for a login from `created_ip` and inserts it into the table.
    pub fn create(&self, created_ip: IpAddr) -> NewSession {
        let now = Instant::now();
        let mut sessions = self.sessions.lock().expect("session store mutex poisoned");
        loop {
            let id: SessionId = random_bytes();
            if sessions.contains_key(&id) {
                // 256 bits of randomness: this branch is not expected to ever be taken in
                // practice, but looping is cheap insurance against a collision.
                continue;
            }
            let csrf_token: CsrfToken = random_bytes();
            sessions.insert(
                id,
                SessionRecord {
                    csrf_token,
                    created_ip,
                    absolute_deadline: now + self.absolute_timeout,
                    idle_deadline: now + self.idle_timeout,
                },
            );
            return NewSession { id, csrf_token };
        }
    }

    /// Validates `id`. If the session exists and neither timeout has elapsed, refreshes its idle
    /// deadline (this request counts as activity) and returns its info; otherwise removes it (if
    /// present) and returns `None`. Every authenticated request should call this exactly once.
    pub fn touch(&self, id: &SessionId) -> Option<SessionInfo> {
        let now = Instant::now();
        let mut sessions = self.sessions.lock().expect("session store mutex poisoned");
        let record = sessions.get_mut(id)?;
        if now >= record.absolute_deadline || now >= record.idle_deadline {
            sessions.remove(id);
            return None;
        }
        record.idle_deadline = now + self.idle_timeout;
        Some(SessionInfo {
            csrf_token: record.csrf_token,
            created_ip: record.created_ip,
        })
    }

    /// Checks whether `id` is currently valid *without* touching it — no idle-deadline refresh,
    /// no removal on expiry (review finding F2: used by an open WebSocket to periodically notice
    /// its session went away, where merely having the socket open shouldn't itself count as
    /// activity that keeps extending the idle timeout).
    pub fn is_valid(&self, id: &SessionId) -> bool {
        let now = Instant::now();
        let sessions = self.sessions.lock().expect("session store mutex poisoned");
        sessions
            .get(id)
            .is_some_and(|record| now < record.absolute_deadline && now < record.idle_deadline)
    }

    /// Removes a session (logout). Returns `true` if it existed.
    pub fn invalidate(&self, id: &SessionId) -> bool {
        self.sessions
            .lock()
            .expect("session store mutex poisoned")
            .remove(id)
            .is_some()
    }

    /// Removes every expired session. Returns how many were removed. Called periodically by the
    /// background purge task, and directly by tests.
    pub fn purge_expired(&self) -> usize {
        let now = Instant::now();
        let mut sessions = self.sessions.lock().expect("session store mutex poisoned");
        let before = sessions.len();
        sessions.retain(|_, record| now < record.absolute_deadline && now < record.idle_deadline);
        before - sessions.len()
    }

    /// Number of sessions currently in the table, expired or not. Test/observability only.
    pub fn len(&self) -> usize {
        self.sessions.lock().expect("session store mutex poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;
    use std::thread::sleep;

    fn ip() -> IpAddr {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    }

    #[test]
    fn is_valid_does_not_refresh_idle_deadline() {
        let store = SessionStore::new(Duration::from_millis(150), Duration::from_secs(3600));
        let session = store.create(ip());
        // Repeatedly calling is_valid should NOT keep the session alive past its idle timeout,
        // unlike touch() — stay comfortably under the timeout while doing so.
        for _ in 0..3 {
            assert!(store.is_valid(&session.id));
            sleep(Duration::from_millis(30));
        }
        // Now comfortably past the 150ms idle timeout (only ~90ms elapsed above).
        sleep(Duration::from_millis(200));
        assert!(
            !store.is_valid(&session.id),
            "is_valid must not have refreshed the idle deadline"
        );
    }

    #[test]
    fn is_valid_returns_false_for_unknown_session() {
        let store = SessionStore::new(Duration::from_secs(3600), Duration::from_secs(3600));
        let bogus: SessionId = [0u8; 32];
        assert!(!store.is_valid(&bogus));
    }

    #[test]
    fn is_valid_returns_false_after_invalidate() {
        let store = SessionStore::new(Duration::from_secs(3600), Duration::from_secs(3600));
        let session = store.create(ip());
        assert!(store.is_valid(&session.id));
        store.invalidate(&session.id);
        assert!(!store.is_valid(&session.id));
    }

    #[test]
    fn create_then_touch_succeeds() {
        let store = SessionStore::new(Duration::from_secs(3600), Duration::from_secs(7 * 24 * 3600));
        let session = store.create(ip());
        let info = store.touch(&session.id).expect("session should be valid");
        assert_eq!(info.csrf_token, session.csrf_token);
        assert_eq!(info.created_ip, ip());
    }

    #[test]
    fn touch_unknown_id_returns_none() {
        let store = SessionStore::new(Duration::from_secs(3600), Duration::from_secs(3600));
        let bogus: SessionId = [0u8; 32];
        assert!(store.touch(&bogus).is_none());
    }

    #[test]
    fn idle_timeout_expires_session() {
        let store = SessionStore::new(Duration::from_millis(30), Duration::from_secs(3600));
        let session = store.create(ip());
        assert!(store.touch(&session.id).is_some());
        sleep(Duration::from_millis(60));
        assert!(store.touch(&session.id).is_none(), "session should be idle-expired");
        assert_eq!(store.len(), 0, "expired session must be removed on touch");
    }

    #[test]
    fn activity_refreshes_idle_timeout() {
        let store = SessionStore::new(Duration::from_millis(80), Duration::from_secs(3600));
        let session = store.create(ip());
        // Touch repeatedly, staying under the idle window each time; the session should survive
        // longer than a single idle window because each touch refreshes it.
        for _ in 0..4 {
            sleep(Duration::from_millis(40));
            assert!(
                store.touch(&session.id).is_some(),
                "activity should keep the session alive"
            );
        }
    }

    #[test]
    fn absolute_timeout_expires_session_even_with_activity() {
        let store = SessionStore::new(Duration::from_secs(3600), Duration::from_millis(50));
        let session = store.create(ip());
        assert!(store.touch(&session.id).is_some());
        sleep(Duration::from_millis(80));
        assert!(
            store.touch(&session.id).is_none(),
            "absolute timeout must expire the session regardless of activity"
        );
    }

    #[test]
    fn invalidate_removes_session() {
        let store = SessionStore::new(Duration::from_secs(3600), Duration::from_secs(3600));
        let session = store.create(ip());
        assert!(store.invalidate(&session.id));
        assert!(store.touch(&session.id).is_none());
        assert!(!store.invalidate(&session.id), "second invalidate should find nothing");
    }

    #[test]
    fn purge_expired_removes_only_expired_sessions() {
        let store = SessionStore::new(Duration::from_millis(30), Duration::from_secs(3600));
        let fresh_ttl_store = &store;
        let expiring = fresh_ttl_store.create(ip());
        sleep(Duration::from_millis(60));
        let fresh = store.create(ip());
        let removed = store.purge_expired();
        assert_eq!(removed, 1);
        assert!(store.touch(&expiring.id).is_none());
        assert!(store.touch(&fresh.id).is_some());
    }

    #[test]
    fn two_sessions_get_distinct_ids_and_csrf_tokens() {
        let store = SessionStore::new(Duration::from_secs(3600), Duration::from_secs(3600));
        let a = store.create(ip());
        let b = store.create(ip());
        assert_ne!(a.id, b.id);
        assert_ne!(a.csrf_token, b.csrf_token);
    }
}
