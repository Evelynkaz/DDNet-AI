//! Trusted-device tracking (review round 1, finding F7 — orchestrator addition): a long-lived,
//! signed cookie that lets a device which has already proven it knows the *current* password skip
//! the shared global login-rate limit, so an attacker who floods `/api/login` from many source
//! addresses cannot lock the real owner out indefinitely by exhausting that shared budget. A
//! trusted device is still fully subject to its own per-IP limiter (see `auth::rate_limit`) — this
//! only ever bypasses the *global* one.
//!
//! Devices live only in memory, exactly like sessions (`auth::session`): a server restart forgets
//! every trusted device, which just means the owner's next login re-establishes trust from
//! scratch. Revocation on a password change needs no cross-process signaling because trust is
//! keyed by the exact password hash a device last confirmed
//! ([`DeviceStore::is_trusted`]): once `ddnet-ai web-passwd` writes a new hash, every previously
//! trusted device's stored fingerprint stops matching and it silently stops being trusted, without
//! the running server needing to be told anything. [`DeviceStore::revoke_all`] exists for a
//! possible future "log out everywhere" action; nothing calls it yet.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::rand_util::random_bytes;

pub type DeviceId = [u8; 32];

struct DeviceRecord {
    /// The PHC hash string (`argon2::secrets::PasswordAuth::hash_phc`) that was active the last
    /// time this device successfully logged in. A password rotation changes this hash, so an old
    /// device's fingerprint stops matching — that's the entire revocation mechanism.
    password_hash_fingerprint: String,
    expires_at: Instant,
}

pub struct DeviceStore {
    devices: Mutex<HashMap<DeviceId, DeviceRecord>>,
    ttl: Duration,
}

impl DeviceStore {
    pub fn new(ttl: Duration) -> Self {
        Self {
            devices: Mutex::new(HashMap::new()),
            ttl,
        }
    }

    /// True if `id` is currently trusted for `current_hash_phc`: known, not expired, and its
    /// stored fingerprint matches the password hash that's active *right now*.
    pub fn is_trusted(&self, id: &DeviceId, current_hash_phc: &str) -> bool {
        let now = Instant::now();
        let devices = self.devices.lock().expect("device store mutex poisoned");
        devices
            .get(id)
            .is_some_and(|record| now < record.expires_at && record.password_hash_fingerprint == current_hash_phc)
    }

    /// Records (or refreshes) `id` as trusted for `current_hash_phc`. Called after a successful
    /// login — reusing the id from a presented (even if not-currently-trusted, e.g.
    /// hash-mismatched after a password change) device cookie means a returning browser doesn't
    /// accumulate a new device record on every login.
    pub fn confirm(&self, id: DeviceId, current_hash_phc: &str) {
        let mut devices = self.devices.lock().expect("device store mutex poisoned");
        devices.insert(
            id,
            DeviceRecord {
                password_hash_fingerprint: current_hash_phc.to_string(),
                expires_at: Instant::now() + self.ttl,
            },
        );
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
        self.devices.lock().expect("device store mutex poisoned").clear();
    }

    /// Removes expired device records. Called periodically alongside session/rate-limit purging.
    pub fn purge_expired(&self) -> usize {
        let now = Instant::now();
        let mut devices = self.devices.lock().expect("device store mutex poisoned");
        let before = devices.len();
        devices.retain(|_, record| now < record.expires_at);
        before - devices.len()
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
        let store = DeviceStore::new(Duration::from_millis(30));
        let id = store.create_and_confirm(HASH_A);
        assert!(store.is_trusted(&id, HASH_A));
        sleep(Duration::from_millis(60));
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
        let store = DeviceStore::new(Duration::from_millis(30));
        let expiring = store.create_and_confirm(HASH_A);
        sleep(Duration::from_millis(60));
        let fresh = store.create_and_confirm(HASH_B);
        let removed = store.purge_expired();
        assert_eq!(removed, 1);
        assert!(!store.is_trusted(&expiring, HASH_A));
        assert!(store.is_trusted(&fresh, HASH_B));
    }
}
