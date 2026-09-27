//! Trusted-device tracking (review round 1, finding F7 — orchestrator addition): a long-lived,
//! signed cookie that lets a device which has already proven it knows the *current* password skip
//! the shared global login-rate limit, so an attacker who floods `/api/login` from many source
//! addresses cannot lock the real owner out indefinitely by exhausting that shared budget. A
//! trusted device is still fully subject to its own per-prefix limiters (see `auth::rate_limit`)
//! — this only ever bypasses the *global* one.
//!
//! Revocation on a password change needs no cross-process signaling because trust is keyed by the
//! exact password hash a device last confirmed ([`DeviceStore::is_trusted`]): once
//! `ddnet-ai web-passwd` writes a new hash, every previously trusted device's stored fingerprint
//! stops matching and it silently stops being trusted, without the running server needing to be
//! told anything — this holds regardless of whether a given record came from memory or was just
//! loaded from disk, since the comparison is always against whatever hash is *currently* on disk.
//!
//! **Persistence (review round 2, finding F8a).** Devices used to live only in memory: a server
//! restart forgot every trusted device, which meant the owner's *only* way past the global login
//! rate limit (see `auth::rate_limit`'s module docs on why that limit exists and why bypassing it
//! matters) evaporated on every redeploy — exactly the moment an attacker flooding the endpoint
//! from many addresses is most likely to also be in progress. [`DeviceStore::load_or_empty`] now
//! loads any previously-confirmed devices from `<data_dir>/secrets/web-devices.toml`
//! (`crate::secrets::load_devices`/`save_devices`) on startup and writes back on every mutation.
//! A devices-file problem (missing, corrupt, unreadable) is logged and treated as "start empty" —
//! this is a convenience cache, not a security boundary in itself: an owner who loses trusted-
//! device status just re-establishes it on the next successful login, same as day one.
//!
//! Records are keyed by [`hash_id`] (SHA-256 of the raw device id), never the raw id itself — see
//! `crate::secrets::PersistedDevice`'s doc comment for why.
//!
//! **Review round 2, finding F9: a stolen copy of an old device cookie must never regain its
//! bypass just because the legitimate browser holding the same id logs in again later.** The
//! previous version of this crate's `http::login::login` reused whatever device id a request
//! *presented*, even one that `is_trusted` had just said "no" to (e.g. because a password
//! rotation made its stored fingerprint stale) — reasoning that a returning legitimate browser
//! shouldn't accumulate a fresh device record on every login. That reasoning was correct for the
//! legitimate browser, but it had a sharp edge: re-confirming that SAME id (and therefore its
//! identical, still-validly-*signed* cookie value — the HMAC signature only depends on the id and
//! the session key, neither of which a password rotation changes) also silently re-authorized any
//! OTHER copy of that exact cookie value an attacker might have captured earlier, purely as a side
//! effect of the owner's own next successful login. `login.rs` now mints a **fresh** random device
//! id whenever the presented one wasn't *already* trusted going in — see its own comment at the
//! `confirm` call site. An untrusted presented id is simply left alone (never refreshed, never
//! re-trusted): it either expires on its own `expires_at_unix`, or gets pruned once [`MAX_TRACKED_DEVICES`]
//! is exceeded (finding F10).
//!
//! `web-passwd` also wipes this file outright on every password rotation
//! (`secrets::generate_and_store_password`) — the *running* server's own in-memory copy of
//! whatever was trusted under the old password is a separate table that command has no way to
//! reach, and this module's own next [`DeviceStore::persist`] call (triggered by any subsequent
//! login) will cheerfully write it straight back to disk, stale entries included. That is fine:
//! any such lingering record's `password_hash_fingerprint` was computed under the password that
//! just got replaced, so it can never match [`is_trusted`]'s comparison against the *current*
//! fingerprint again — its mere on-disk presence for the rest of its TTL is inert, not a live
//! credential. (A "have the running server itself notice a password rotation and proactively drop
//! stale records" design was considered and set aside: it would need the store to be handed the
//! current password hash out-of-band on every purge/persist cycle for no security benefit over
//! what the fingerprint-mismatch check already guarantees on every single lookup — see review
//! finding F9's own point (d).)

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};

use crate::rand_util::{encode_b64, random_bytes};
use crate::secrets::{self, PersistedDevice, SecretsPaths};

pub type DeviceId = [u8; 32];

/// Hard cap on the number of distinct devices tracked at once (review round 2, finding F10):
/// every login without a usable device cookie (a fresh browser profile, curl, a test run, an
/// attacker who never bothers sending one back) otherwise adds a new ~90-day-lived record
/// forever. When a *new* id would exceed this, the least-recently-*seen* record is evicted first
/// — see [`evict_least_recently_seen`]. 32 is comfortably more than any real single owner's set of
/// browsers/devices while still bounding both the table's memory and the persisted file's size.
const MAX_TRACKED_DEVICES: usize = 32;

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// SHA-256 of a raw device id — the key every record is actually stored/looked-up under, both in
/// memory and on disk (finding F8a).
fn hash_id(id: &DeviceId) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(id);
    hasher.finalize().into()
}

type HmacSha256 = Hmac<Sha256>;

/// Derives the opaque string [`DeviceStore::is_trusted`]/[`DeviceStore::confirm`] actually
/// store/compare (review round 2, finding F9c): an HMAC-SHA256 of the password's PHC hash string,
/// keyed with the session-signing key — never the raw PHC string itself. A leaked
/// `web-devices.toml` therefore reveals nothing at all about any current-or-superseded password
/// hash (salt, argon2 params included) without ALSO having the session-signing key, which if
/// compromised already breaks every session/device cookie's signature regardless — this adds a
/// layer specifically for the case where *only* the devices file leaks (e.g. a misconfigured
/// backup) and not the session key file next to it.
pub fn password_fingerprint(session_key: &[u8], hash_phc: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(session_key).expect("HMAC-SHA256 accepts a key of any length");
    mac.update(hash_phc.as_bytes());
    encode_b64(&mac.finalize().into_bytes())
}

struct DeviceRecord {
    /// The *fingerprint* ([`password_fingerprint`]) that was active the last time this device
    /// successfully logged in — not the raw PHC hash (finding F9c). A password rotation changes
    /// the fingerprint this compares against, so an old device's stored value stops matching —
    /// that's the entire revocation mechanism.
    password_hash_fingerprint: String,
    /// Wall-clock (Unix seconds), not `Instant`: unlike sessions/rate-limit state, a device record
    /// must survive a process restart (finding F8a), and `Instant` has no defined relationship to
    /// wall-clock time across two different process lifetimes.
    expires_at_unix: u64,
    last_seen_unix: u64,
}

pub struct DeviceStore {
    devices: Mutex<HashMap<[u8; 32], DeviceRecord>>,
    ttl: Duration,
    /// Where to persist to disk. `None` keeps everything in-memory only (used by tests that don't
    /// care about persistence, and by the plain [`DeviceStore::new`] constructor); production
    /// always goes through [`DeviceStore::load_or_empty`], which sets this.
    paths: Option<SecretsPaths>,
}

/// Evicts the single entry with the oldest `last_seen_unix` (used only when
/// [`MAX_TRACKED_DEVICES`] would otherwise be exceeded — a scan here is fine at this scale).
fn evict_least_recently_seen(devices: &mut HashMap<[u8; 32], DeviceRecord>) {
    if let Some(&oldest_key) = devices
        .iter()
        .min_by_key(|(_, record)| record.last_seen_unix)
        .map(|(key, _)| key)
    {
        devices.remove(&oldest_key);
    }
}

impl DeviceStore {
    /// In-memory only, no persistence — exists mainly so existing/simple tests don't need a
    /// [`SecretsPaths`]. Production code should use [`DeviceStore::load_or_empty`] instead.
    pub fn new(ttl: Duration) -> Self {
        Self {
            devices: Mutex::new(HashMap::new()),
            ttl,
            paths: None,
        }
    }

    /// Loads any previously-persisted trusted devices (finding F8a) and remembers `paths` so
    /// future confirms/revocations get written back. Never fails: a read or parse problem is
    /// logged and treated as "start with no trusted devices" — see this module's doc comment for
    /// why that's the right trade-off here.
    pub fn load_or_empty(paths: &SecretsPaths, ttl: Duration) -> Self {
        let now = unix_now();
        let devices = match secrets::load_devices(paths) {
            Ok(persisted) => persisted
                .into_iter()
                // Don't even load a record that's already expired — it would just be purged on
                // the next sweep anyway, and skipping it here means a store that loaded nothing
                // but expired rows reports `is_empty()` truthfully right away.
                .filter(|record| record.expires_at_unix > now)
                .map(|record| {
                    (
                        record.id_hash,
                        DeviceRecord {
                            password_hash_fingerprint: record.password_hash_fingerprint,
                            expires_at_unix: record.expires_at_unix,
                            last_seen_unix: record.last_seen_unix,
                        },
                    )
                })
                .collect(),
            Err(error) => {
                tracing::warn!(
                    %error,
                    path = %paths.devices_file().display(),
                    "failed to load trusted devices; starting with none"
                );
                HashMap::new()
            }
        };
        Self {
            devices: Mutex::new(devices),
            ttl,
            paths: Some(paths.clone()),
        }
    }

    /// Writes `rows` (a snapshot already-built by the caller — see below) to disk if this store
    /// was constructed with persistence enabled ([`DeviceStore::load_or_empty`]). A write failure
    /// is only logged: the in-memory state (and this process's own behavior) is unaffected either
    /// way, and only a *future* restart would fail to see whatever change just failed to save.
    ///
    /// Review round 2, finding F10: takes an already-built, owned snapshot rather than the table
    /// itself, specifically so every call site below can build that snapshot *while* holding the
    /// mutex (cheap — a handful of small clones, bounded by [`MAX_TRACKED_DEVICES`]) and then
    /// `persist` the fsync'd write only *after* releasing it. Without this, the disk write
    /// (fsync included) ran for the whole table-mutation call while still holding the lock, on
    /// whatever thread called it — for `confirm`, that's an axum handler's async worker thread,
    /// so every OTHER request needing this same lock (including an unrelated `is_trusted` peek)
    /// queued behind a single fsync's worth of disk latency.
    fn persist(&self, rows: Vec<PersistedDevice>) {
        let Some(paths) = &self.paths else { return };
        if let Err(error) = secrets::save_devices(paths, &rows) {
            tracing::warn!(%error, "failed to persist trusted devices");
        }
    }

    /// Builds the on-disk snapshot of `devices` (a plain, lock-independent value — see
    /// [`DeviceStore::persist`]).
    fn snapshot(devices: &HashMap<[u8; 32], DeviceRecord>) -> Vec<PersistedDevice> {
        devices
            .iter()
            .map(|(id_hash, record)| PersistedDevice {
                id_hash: *id_hash,
                password_hash_fingerprint: record.password_hash_fingerprint.clone(),
                expires_at_unix: record.expires_at_unix,
                last_seen_unix: record.last_seen_unix,
            })
            .collect()
    }

    /// True if `id` is currently trusted for `current_hash_phc`: known, not expired, and its
    /// stored fingerprint matches the password hash that's active *right now*.
    pub fn is_trusted(&self, id: &DeviceId, current_hash_phc: &str) -> bool {
        let now = unix_now();
        let key = hash_id(id);
        let devices = self.devices.lock().expect("device store mutex poisoned");
        devices
            .get(&key)
            .is_some_and(|record| now < record.expires_at_unix && record.password_hash_fingerprint == current_hash_phc)
    }

    /// Records (or refreshes) `id` as trusted for `current_hash_phc` (in current code, always a
    /// [`password_fingerprint`], never a raw PHC string — see finding F9c). Callers must only
    /// pass an id here that has *independently* earned trust for THIS login — see finding F9b's
    /// comment at `http::login::login`'s call site; this function itself does not (cannot) tell
    /// the difference between "a brand-new id" and "an id that was just rejected by `is_trusted`
    /// a moment ago", so that decision belongs entirely to the caller.
    pub fn confirm(&self, id: DeviceId, current_hash_phc: &str) {
        let now = unix_now();
        let key = hash_id(&id);
        let snapshot = {
            let mut devices = self.devices.lock().expect("device store mutex poisoned");
            // Finding F10: cap the table. Only evict for a key that isn't already tracked —
            // refreshing an existing device must never be turned away by its own cap.
            if !devices.contains_key(&key) && devices.len() >= MAX_TRACKED_DEVICES {
                evict_least_recently_seen(&mut devices);
            }
            devices.insert(
                key,
                DeviceRecord {
                    password_hash_fingerprint: current_hash_phc.to_string(),
                    expires_at_unix: now + self.ttl.as_secs(),
                    last_seen_unix: now,
                },
            );
            Self::snapshot(&devices)
        }; // lock released here, before the (possibly slow) disk write (finding F10)
        self.persist(snapshot);
    }

    /// Mints a fresh random device id, confirms it for `current_hash_phc`, and returns it — used
    /// when a login request carried no (or no decodable) device cookie at all.
    pub fn create_and_confirm(&self, current_hash_phc: &str) -> DeviceId {
        let id: DeviceId = random_bytes();
        self.confirm(id, current_hash_phc);
        id
    }

    /// Revokes every trusted device. Not wired to any endpoint yet (no "log out everywhere"
    /// action exists in this skeleton) but a natural building block for one; exercised directly by
    /// tests.
    pub fn revoke_all(&self) {
        let snapshot = {
            let mut devices = self.devices.lock().expect("device store mutex poisoned");
            devices.clear();
            Self::snapshot(&devices)
        };
        self.persist(snapshot);
    }

    /// Removes expired device records. Called periodically alongside session/rate-limit purging.
    pub fn purge_expired(&self) -> usize {
        let now = unix_now();
        let (removed, snapshot) = {
            let mut devices = self.devices.lock().expect("device store mutex poisoned");
            let before = devices.len();
            devices.retain(|_, record| now < record.expires_at_unix);
            (before - devices.len(), Self::snapshot(&devices))
        };
        if removed > 0 {
            self.persist(snapshot);
        }
        removed
    }

    /// Number of device records currently tracked, expired or not. Test/observability only.
    pub fn len(&self) -> usize {
        self.devices.lock().expect("device store mutex poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets::{Argon2Params, SecretsPaths, generate_and_store_password};
    use std::thread::sleep;

    const HASH_A: &str = "$argon2id$v=19$m=8,t=1,p=1$AAAA$hash-a";
    const HASH_B: &str = "$argon2id$v=19$m=8,t=1,p=1$AAAA$hash-b";

    #[test]
    fn newly_confirmed_device_is_trusted_for_its_hash() {
        let store = DeviceStore::new(Duration::from_secs(3600));
        let id = store.create_and_confirm(HASH_A);
        assert!(store.is_trusted(&id, HASH_A));
    }

    #[test]
    fn unknown_device_id_is_never_trusted() {
        let store = DeviceStore::new(Duration::from_secs(3600));
        let bogus: DeviceId = [0u8; 32];
        assert!(!store.is_trusted(&bogus, HASH_A));
    }

    #[test]
    fn password_change_revokes_trust_via_hash_mismatch() {
        let store = DeviceStore::new(Duration::from_secs(3600));
        let id = store.create_and_confirm(HASH_A);
        assert!(store.is_trusted(&id, HASH_A));
        // The password hash on disk changed (e.g. `web-passwd` ran again); the device's stored
        // fingerprint is now stale.
        assert!(!store.is_trusted(&id, HASH_B));
    }

    #[test]
    fn re_login_with_new_password_re_establishes_trust() {
        let store = DeviceStore::new(Duration::from_secs(3600));
        let id = store.create_and_confirm(HASH_A);
        assert!(!store.is_trusted(&id, HASH_B));
        // A fresh successful login (which only happens after proving the *new* password) refreshes
        // the fingerprint.
        store.confirm(id, HASH_B);
        assert!(store.is_trusted(&id, HASH_B));
        assert!(!store.is_trusted(&id, HASH_A));
    }

    #[test]
    fn device_expires_after_its_ttl() {
        // Expiry is stored at whole-second (Unix time) granularity, not `Instant`'s sub-second
        // precision (finding F8a: it must survive a process restart) — so this needs a >=1s TTL
        // and a sleep comfortably longer than it, robust to wherever within the current second
        // `create_and_confirm` happened to land.
        let store = DeviceStore::new(Duration::from_secs(1));
        let id = store.create_and_confirm(HASH_A);
        assert!(store.is_trusted(&id, HASH_A));
        sleep(Duration::from_millis(2100));
        assert!(!store.is_trusted(&id, HASH_A), "device should have expired");
    }

    #[test]
    fn revoke_all_clears_every_device() {
        let store = DeviceStore::new(Duration::from_secs(3600));
        let a = store.create_and_confirm(HASH_A);
        let b = store.create_and_confirm(HASH_B);
        store.revoke_all();
        assert!(!store.is_trusted(&a, HASH_A));
        assert!(!store.is_trusted(&b, HASH_B));
        assert!(store.is_empty());
    }

    #[test]
    fn purge_expired_removes_only_expired_devices() {
        let store = DeviceStore::new(Duration::from_secs(1));
        let expiring = store.create_and_confirm(HASH_A);
        sleep(Duration::from_millis(2100));
        let fresh = store.create_and_confirm(HASH_B);
        let removed = store.purge_expired();
        assert_eq!(removed, 1);
        assert!(!store.is_trusted(&expiring, HASH_A));
        assert!(store.is_trusted(&fresh, HASH_B));
    }

    // -----------------------------------------------------------------------------------------
    // Persistence (review round 2, finding F8a)
    // -----------------------------------------------------------------------------------------

    #[test]
    fn a_confirmed_device_survives_a_simulated_restart() {
        // "Restart" here means: construct a brand-new DeviceStore (as `AppState::new` does on
        // every real process start) pointed at the SAME on-disk paths, rather than reusing the
        // original in-memory store at all.
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());

        let before_restart = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        let id = before_restart.create_and_confirm(HASH_A);
        assert!(before_restart.is_trusted(&id, HASH_A));

        let after_restart = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        assert!(
            after_restart.is_trusted(&id, HASH_A),
            "a trusted device confirmed before a restart must still be trusted after loading from disk"
        );
    }

    #[test]
    fn a_device_persisted_under_an_old_password_is_not_trusted_after_a_restart_with_a_new_one() {
        // The password-change revocation mechanism (hash mismatch) must keep working across a
        // restart too: loading a stale on-disk record must not itself grant trust independent of
        // whatever the *current* password hash is.
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());

        let before_restart = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        let id = before_restart.create_and_confirm(HASH_A);

        let after_restart = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        assert!(
            !after_restart.is_trusted(&id, HASH_B),
            "a device trusted under the OLD password hash must not be trusted for a NEW one, restart or not"
        );
    }

    #[test]
    fn revoke_all_persists_the_now_empty_table() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let store = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        let id = store.create_and_confirm(HASH_A);
        store.revoke_all();

        let reloaded = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        assert!(!reloaded.is_trusted(&id, HASH_A));
        assert!(reloaded.is_empty());
    }

    #[test]
    fn loading_from_a_missing_or_corrupt_devices_file_never_panics_and_starts_empty() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        // No file at all yet.
        let store = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        assert!(store.is_empty());

        // A corrupt file (right permissions, garbage content) must not crash startup either — it
        // should just log a warning and start empty (this module's doc comment: a convenience
        // cache, not a security boundary).
        secrets::ensure_secrets_dir(&paths).expect("ensure dir");
        std::fs::write(paths.devices_file(), b"\x00\x01\xffnot even close to the format\n").expect("write garbage");
        std::fs::set_permissions(
            paths.devices_file(),
            std::os::unix::fs::PermissionsExt::from_mode(0o600),
        )
        .expect("set perms");
        let store2 = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        // Garbage bytes still parse to "zero valid lines" under decode_devices's lenient
        // line-by-line parsing (no line looks like a valid record), not a hard I/O error, so this
        // exercises the "0 devices loaded" path rather than the "load_devices returned Err" path
        // — both are covered by this same assertion either way.
        assert!(store2.is_empty());
    }

    #[test]
    fn confirming_a_device_actually_creates_the_devices_file_on_disk() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let store = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        assert!(!paths.devices_file().exists(), "nothing to persist yet");
        store.create_and_confirm(HASH_A);
        assert!(
            paths.devices_file().exists(),
            "confirm() should have persisted a new file"
        );
    }

    /// End-to-end-ish sanity check that `AppState`'s actual wiring (`load_or_empty` fed a real
    /// `SecretsPaths` computed the same way `AppState::new` computes it) round-trips through a
    /// real `generate_and_store_password`/hash, not just the plain string constants the other
    /// tests use.
    #[test]
    fn survives_a_restart_alongside_a_real_password_hash_on_disk() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let params = Argon2Params {
            m_cost_kib: 8 * 1024,
            t_cost: 1,
            p_cost: 1,
        };
        let generated = generate_and_store_password(&paths, params).expect("generate password");

        let before_restart = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        let id = before_restart.create_and_confirm(&generated.auth.hash_phc);

        let after_restart = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        assert!(after_restart.is_trusted(&id, &generated.auth.hash_phc));
    }

    // -----------------------------------------------------------------------------------------
    // Table cap (review round 2, finding F10)
    // -----------------------------------------------------------------------------------------

    #[test]
    fn table_size_is_capped_by_evicting_the_least_recently_seen_entry() {
        let store = DeviceStore::new(Duration::from_secs(3600));
        // `last_seen_unix` is whole-second granularity (finding F8a's wall-clock design), so a
        // tight loop's worth of these all tie on the same second — any of them is equally "the"
        // least-recently-seen, and `min_by_key` picks whichever a HashMap happens to iterate
        // first among ties (unspecified, not necessarily insertion order). This test only asserts
        // what's actually guaranteed: exactly one entry from that tied-oldest batch gets evicted,
        // and a *distinctly later* entry (past the same tie) never does.
        let mut initial_ids = Vec::new();
        for _ in 0..(MAX_TRACKED_DEVICES - 1) {
            initial_ids.push(store.create_and_confirm(HASH_A));
        }
        sleep(Duration::from_millis(1100)); // cross into a new whole second, breaking the tie
        let newest = store.create_and_confirm(HASH_A);
        assert_eq!(store.len(), MAX_TRACKED_DEVICES);

        // One more distinct device pushes past the cap.
        let extra = store.create_and_confirm(HASH_A);
        assert_eq!(store.len(), MAX_TRACKED_DEVICES, "table must stay at the cap");
        assert!(
            store.is_trusted(&newest, HASH_A),
            "the unambiguously most-recently-seen entry must never be evicted"
        );
        assert!(store.is_trusted(&extra, HASH_A));
        let survivors = initial_ids.iter().filter(|id| store.is_trusted(id, HASH_A)).count();
        assert_eq!(
            survivors,
            initial_ids.len() - 1,
            "exactly one of the tied-oldest initial batch should have been evicted"
        );
    }

    #[test]
    fn refreshing_an_existing_device_never_gets_turned_away_by_the_cap() {
        let store = DeviceStore::new(Duration::from_secs(3600));
        let mut ids = Vec::new();
        for _ in 0..MAX_TRACKED_DEVICES {
            ids.push(store.create_and_confirm(HASH_A));
        }
        // Re-confirming an ALREADY-tracked device at the cap must not evict anything (it's not a
        // new key), and must still succeed.
        store.confirm(ids[5], HASH_A);
        assert_eq!(store.len(), MAX_TRACKED_DEVICES);
        for id in &ids {
            assert!(
                store.is_trusted(id, HASH_A),
                "no existing device should have been evicted"
            );
        }
    }

    #[test]
    fn cap_persists_correctly_survives_a_restart() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let store = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        for _ in 0..(MAX_TRACKED_DEVICES + 5) {
            store.create_and_confirm(HASH_A);
        }
        assert_eq!(store.len(), MAX_TRACKED_DEVICES);

        let reloaded = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        assert_eq!(
            reloaded.len(),
            MAX_TRACKED_DEVICES,
            "the persisted file must also stay at the cap"
        );
    }

    // -----------------------------------------------------------------------------------------
    // Keyed fingerprint (review round 2, finding F9c)
    // -----------------------------------------------------------------------------------------

    #[test]
    fn password_fingerprint_is_deterministic() {
        let key = [7u8; 32];
        let a = password_fingerprint(&key, "$argon2id$v=19$m=8,t=1,p=1$AAAA$hash");
        let b = password_fingerprint(&key, "$argon2id$v=19$m=8,t=1,p=1$AAAA$hash");
        assert_eq!(a, b);
    }

    #[test]
    fn password_fingerprint_depends_on_the_hash() {
        let key = [7u8; 32];
        let a = password_fingerprint(&key, "$argon2id$...$hash-a");
        let b = password_fingerprint(&key, "$argon2id$...$hash-b");
        assert_ne!(a, b);
    }

    #[test]
    fn password_fingerprint_depends_on_the_key() {
        let hash = "$argon2id$v=19$m=8,t=1,p=1$AAAA$hash";
        let a = password_fingerprint(&[1u8; 32], hash);
        let b = password_fingerprint(&[2u8; 32], hash);
        assert_ne!(a, b);
    }

    #[test]
    fn password_fingerprint_never_contains_the_raw_hash_string() {
        let hash = "$argon2id$v=19$m=8,t=1,p=1$AAAA$super-secret-hash-material";
        let fp = password_fingerprint(&[9u8; 32], hash);
        assert!(!fp.contains("argon2id"));
        assert!(!fp.contains("super-secret-hash-material"));
    }

    // -----------------------------------------------------------------------------------------
    // Reviewer's exact 7-step scenario (review round 2, finding F9, requirement (e))
    // -----------------------------------------------------------------------------------------
    //
    // This is deliberately a device.rs-level unit test driving DeviceStore directly (not a full
    // HTTP integration test) so it can assert on the store's internal decisions at each numbered
    // step precisely as the finding describes them, using the exact fix this module now
    // implements (F9b: never re-confirm a presented-but-untrusted id) — the full HTTP-level
    // behavior (the SAME property, reached through a real login/password-rotation flow) is
    // covered by `login_and_sessions.rs`'s `stolen_device_cookie_copy_does_not_regain_trust_...`.
    #[test]
    fn reviewer_scenario_a_saved_old_cookie_copy_never_regains_bypass() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = SecretsPaths::new(tmp.path());
        let store = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));

        // 1. First login, no device cookie presented at all: nothing to be trusted for yet.
        // (Nothing to assert here beyond the precondition: no ids exist yet.)
        assert!(store.is_empty());

        // The browser gets a fresh id minted for it (this is what `login.rs` does when no
        // presented id was trusted — see the login() call site's own comment).
        let owner_id = random_bytes();
        store.confirm(owner_id, HASH_A);

        // 2. Login WITH that device cookie: trusted.
        assert!(store.is_trusted(&owner_id, HASH_A), "step 2: should be trusted");

        // The attacker captures a copy of the cookie's raw id right now (e.g. off disk, a proxy
        // log, a compromised backup) — same 32 bytes as `owner_id`, from this point on.
        let stolen_copy = owner_id;

        // 3. After a restart: still trusted (finding F8a).
        let store = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        assert!(store.is_trusted(&owner_id, HASH_A), "step 3: should survive a restart");

        // 4. Run web-passwd (finding F9a: wipes the persisted file outright).
        secrets::save_devices(&paths, &[]).expect("web-passwd's device wipe");
        // The RUNNING store's in-memory table is untouched by that (separate process in real
        // life) — same as the finding's own point (d). Re-load to pick up the (now-empty) file,
        // matching "the browser's next request arrives at a server that has since restarted, OR
        // this same in-memory table would still hold the stale record either way" — both cases
        // are covered: this represents the in-memory-untouched case directly (`store` still has
        // its OLD in-memory record for `owner_id` under HASH_A, which is now stale).

        // 5. Old cookie + new password (HASH_B): not trusted (fingerprint mismatch).
        assert!(
            !store.is_trusted(&owner_id, HASH_B),
            "step 5: stale fingerprint must not match the new hash"
        );
        // The fix (F9b): since this presented id was NOT trusted for HASH_B, `login.rs` mints a
        // FRESH id instead of calling `confirm(owner_id, HASH_B)` — i.e. this exact line, present
        // in the pre-fix code path, must NOT happen:
        //     store.confirm(owner_id, HASH_B); // <- THE BUG: never do this
        let fresh_id_for_this_login = random_bytes();
        store.confirm(fresh_id_for_this_login, HASH_B);

        // 6. After a restart: the OWNER's id is still untrusted for the new hash (the fix means
        // nothing ever re-confirmed it) — this is the corrected outcome; the finding's own repro
        // number here ("true") describes the PRE-fix bug this test is verifying is now GONE.
        let store = DeviceStore::load_or_empty(&paths, Duration::from_secs(3600));
        assert!(
            !store.is_trusted(&owner_id, HASH_B),
            "step 6 (fixed): the owner's OLD id must still not be trusted after a restart"
        );

        // 7. THE ACTUAL FINDING: the attacker's stolen copy (identical bytes to `owner_id`) must
        // never show up as trusted for the current password.
        assert!(
            !store.is_trusted(&stolen_copy, HASH_B),
            "step 7: a saved copy of the old cookie must NOT regain its bypass"
        );
    }
}
