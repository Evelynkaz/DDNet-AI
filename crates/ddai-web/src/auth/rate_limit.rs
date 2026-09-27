//! Login brute-force protection (acceptance criterion 3): a sliding-window counter per client IP
//! with exponential lockout once it's exceeded, plus a separate global sliding-window cap shared
//! across all IPs. Every `POST /api/login` attempt — successful or not — is counted against both
//! windows as it arrives, since the point is to bound the *rate* of attempts an attacker (or a
//! buggy client) can make; [`LoginRateLimiter::record_success`] then clears an IP's backoff state
//! so a legitimate user who mistyped a few times isn't left half-locked-out after finally getting
//! it right.
//!
//! Deliberately not `tower_governor` (see `docs/research/rust-stack.md` §1/§7: sparsely
//! maintained, and its `GovernorConfigBuilder` doesn't give per-key exponential lockout out of
//! the box) — this is small and fully under our control.
//!
//! Three properties earned from review round 1 (findings F3, F7) and round 2 (F8), all
//! load-bearing — `check`'s three-step structure (peek → global → per-IP update) exists
//! specifically to hold all three at once:
//! - **An already-locked-out IP is rejected by a read-only peek at the per-IP table, before the
//!   global counter is ever touched** (review finding F8). Checking the *global* limiter first
//!   (the F3 fix, see below) had an unintended consequence: every repeat attempt from an
//!   already-locked-out IP still consumed a global slot on its way to being per-IP-rejected, so a
//!   *single* attacker who quickly triggers their own lockout and keeps hammering it can drain the
//!   entire global budget alone and lock the real owner out — reproduced in review with one IP
//!   and ~31 requests/min. The peek is read-only (no entry created, no counters touched), so it
//!   doesn't reopen F3's unbounded-growth problem: a *new* IP still can't get a free peek-only
//!   pass, since there's nothing to peek at until step 3 creates its entry, which still only
//!   happens after the global check.
//! - **The global check runs before any per-IP `HashMap` entry is *created*.** A flood of
//!   requests from unique/rotating source addresses (real for IPv6, trivial to spoof via
//!   `X-Forwarded-For` if `--trust-proxy` is misconfigured) can otherwise grow the per-IP table
//!   without bound — 800k entries / 20s were reproduced in review. Checking the IP-agnostic
//!   global limiter before creating a new entry means a flood can create at most `global_limit`
//!   *new* per-IP entries per `global_window`, regardless of how many distinct addresses it uses.
//!   [`purge_expired`] and a hard [`RateLimitConfig::max_tracked_ips`] cap (LRU-ish eviction by
//!   `window_start`) are defense in depth on top of that, not the primary fix. A *known* (already
//!   tracked) IP that isn't currently locked still passes through the global check on every
//!   attempt — only an *already-locked* IP's repeat attempts skip it, per the F8 point above.
//! - **A caller can pass `bypass_global = true`** (used by a verified trusted-device cookie, see
//!   `auth::device`) to skip *only* the global check while still being fully subject to the
//!   per-IP limiter. Without this, an attacker with enough distinct source IPs/addresses can
//!   exhaust the shared global budget and lock the legitimate owner out indefinitely — the global
//!   limit protects against a single attacker hammering the endpoint, not against a determined
//!   owner-lockout attempt, so a device that has already proven it knows the *current* password
//!   gets to skip it.
//!
//! IPv6 addresses are rate-limited by their `/64` (the typical single-customer allocation unit),
//! not the full 128 bits — otherwise an attacker with a routed `/64` can rotate the low 64 bits
//! per request and never repeat an IP.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv6Addr};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct RateLimitConfig {
    pub per_ip_limit: u32,
    pub per_ip_window: Duration,
    pub global_limit: u32,
    pub global_window: Duration,
    pub base_lockout: Duration,
    pub max_lockout: Duration,
    /// Hard cap on the number of distinct per-IP(-prefix) entries tracked at once. When a *new*
    /// key would exceed this, the entry with the oldest `window_start` is evicted first. This is
    /// a defensive backstop — under normal operation the global-first check ordering (see module
    /// docs) keeps the table far below this.
    pub max_tracked_ips: usize,
}

impl Default for RateLimitConfig {
    /// Acceptance criterion 3's suggested numbers: "5 attempts/min/IP with exponential lockout,
    /// 30/min global".
    fn default() -> Self {
        Self {
            per_ip_limit: 5,
            per_ip_window: Duration::from_secs(60),
            global_limit: 30,
            global_window: Duration::from_secs(60),
            base_lockout: Duration::from_secs(30),
            max_lockout: Duration::from_secs(30 * 60),
            max_tracked_ips: 100_000,
        }
    }
}

struct IpState {
    window_start: Instant,
    count_in_window: u32,
    lockout_until: Option<Instant>,
    lockout_streak: u32,
}

struct GlobalState {
    window_start: Instant,
    count_in_window: u32,
}

pub struct LoginRateLimiter {
    config: RateLimitConfig,
    per_ip: Mutex<HashMap<IpAddr, IpState>>,
    global: Mutex<GlobalState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimited {
    pub retry_after: Duration,
}

/// Normalizes the key used for per-address tracking: IPv4 addresses are used as-is, IPv6
/// addresses are truncated to their `/64` routing prefix (see module docs).
fn rate_limit_key(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(_) => ip,
        IpAddr::V6(v6) => {
            let segments = v6.segments();
            IpAddr::V6(Ipv6Addr::new(
                segments[0],
                segments[1],
                segments[2],
                segments[3],
                0,
                0,
                0,
                0,
            ))
        }
    }
}

impl LoginRateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            per_ip: Mutex::new(HashMap::new()),
            global: Mutex::new(GlobalState {
                window_start: Instant::now(),
                count_in_window: 0,
            }),
        }
    }

    /// Records one login attempt from `ip` and decides whether it may proceed. Call this once
    /// per `POST /api/login`, before verifying the password.
    ///
    /// `bypass_global`: skip the shared global limiter for this attempt (the per-IP limiter still
    /// applies in full) — set this only for a request that already carries independent proof it's
    /// not part of a flood, e.g. a verified trusted-device cookie (finding F7).
    pub fn check(&self, ip: IpAddr, bypass_global: bool) -> Result<(), RateLimited> {
        let now = Instant::now();
        let key = rate_limit_key(ip);

        // Step 1 (review finding F8): a read-only peek. If this key is *already* locked out,
        // reject immediately without touching the global counter or creating/mutating any entry
        // — otherwise a single already-locked-out IP could keep consuming global slots forever on
        // its way to being per-IP-rejected anyway, draining the shared budget alone. Read-only on
        // purpose: an IP we've never seen has nothing to peek at, so this can't be used to dodge
        // the F3 "no entry before the global check" guarantee below.
        if let Some(until) = self.peek_lockout(&key)
            && now < until
        {
            return Err(RateLimited {
                retry_after: until - now,
            });
        }

        // Step 2: the global check (finding F3), skippable only by a verified trusted device
        // (finding F7).
        if !bypass_global {
            let mut global = self.global.lock().expect("rate limiter mutex poisoned");
            if now.duration_since(global.window_start) >= self.config.global_window {
                global.window_start = now;
                global.count_in_window = 0;
            }
            global.count_in_window += 1;
            if global.count_in_window > self.config.global_limit {
                let retry_after = self.config.global_window - now.duration_since(global.window_start);
                return Err(RateLimited { retry_after });
            }
        }

        // Step 3: the actual per-IP update. A concurrent request for the same key could have
        // locked it out in the narrow window between step 1 and here — handled below exactly like
        // step 1, just after this attempt already spent a global slot on a very tight race, which
        // is the same order-of-magnitude cost this design already accepts (a per-IP lockout's
        // *triggering* attempt always costs one global slot; see
        // `lockout_triggering_attempt_consumes_a_global_slot_but_locked_out_retries_do_not`).
        let mut per_ip = self.per_ip.lock().expect("rate limiter mutex poisoned");
        if !per_ip.contains_key(&key) && per_ip.len() >= self.config.max_tracked_ips {
            evict_oldest(&mut per_ip);
        }
        let state = per_ip.entry(key).or_insert_with(|| IpState {
            window_start: now,
            count_in_window: 0,
            lockout_until: None,
            lockout_streak: 0,
        });
        if let Some(until) = state.lockout_until {
            if now < until {
                return Err(RateLimited {
                    retry_after: until - now,
                });
            }
            // The lockout just expired: start a fresh window from here, rather than leaving
            // `count_in_window` at its already-over-the-limit value (which would otherwise
            // immediately re-trigger another lockout on this very attempt).
            state.lockout_until = None;
            state.window_start = now;
            state.count_in_window = 0;
        }
        if now.duration_since(state.window_start) >= self.config.per_ip_window {
            state.window_start = now;
            state.count_in_window = 0;
        }
        state.count_in_window += 1;
        if state.count_in_window > self.config.per_ip_limit {
            state.lockout_streak += 1;
            let shift = (state.lockout_streak - 1).min(10);
            let lockout = (self.config.base_lockout * (1u32 << shift)).min(self.config.max_lockout);
            state.lockout_until = Some(now + lockout);
            return Err(RateLimited { retry_after: lockout });
        }

        Ok(())
    }

    /// Read-only: `Some(until)` if `key` is currently tracked and locked out (regardless of
    /// whether `until` has already passed — the caller compares against `now` itself), `None` if
    /// unknown or not locked. Never creates an entry.
    fn peek_lockout(&self, key: &IpAddr) -> Option<Instant> {
        let per_ip = self.per_ip.lock().expect("rate limiter mutex poisoned");
        per_ip.get(key).and_then(|state| state.lockout_until)
    }

    /// Clears `ip`'s backoff state after a successful login.
    pub fn record_success(&self, ip: IpAddr) {
        let key = rate_limit_key(ip);
        let mut per_ip = self.per_ip.lock().expect("rate limiter mutex poisoned");
        if let Some(state) = per_ip.get_mut(&key) {
            state.count_in_window = 0;
            state.lockout_until = None;
            state.lockout_streak = 0;
        }
    }

    /// Removes per-IP entries that are neither within their current window nor still locked out
    /// (i.e. have nothing left to track). Called periodically from the same background task that
    /// purges expired sessions (finding F3: without this the table only ever grows). Returns how
    /// many entries were removed.
    pub fn purge_expired(&self) -> usize {
        let now = Instant::now();
        let mut per_ip = self.per_ip.lock().expect("rate limiter mutex poisoned");
        let before = per_ip.len();
        per_ip.retain(|_, state| {
            let within_window = now.duration_since(state.window_start) < self.config.per_ip_window;
            let locked = state.lockout_until.is_some_and(|until| now < until);
            within_window || locked
        });
        before - per_ip.len()
    }

    /// Number of distinct per-IP(-prefix) entries currently tracked. Test/observability only.
    pub fn tracked_ip_count(&self) -> usize {
        self.per_ip.lock().expect("rate limiter mutex poisoned").len()
    }
}

/// Evicts the single entry with the oldest `window_start` (used only when at
/// [`RateLimitConfig::max_tracked_ips`] capacity — a rare, defensive path, so an O(n) scan here is
/// fine).
fn evict_oldest(map: &mut HashMap<IpAddr, IpState>) {
    if let Some(&oldest_key) = map
        .iter()
        .min_by_key(|(_, state)| state.window_start)
        .map(|(key, _)| key)
    {
        map.remove(&oldest_key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};
    use std::thread::sleep;

    fn ip(last_octet: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, last_octet))
    }

    fn config() -> RateLimitConfig {
        RateLimitConfig {
            per_ip_limit: 5,
            per_ip_window: Duration::from_secs(60),
            global_limit: 30,
            global_window: Duration::from_secs(60),
            base_lockout: Duration::from_millis(50),
            max_lockout: Duration::from_secs(10),
            max_tracked_ips: 100_000,
        }
    }

    #[test]
    fn allows_up_to_the_limit_then_locks_out() {
        let limiter = LoginRateLimiter::new(config());
        let addr = ip(1);
        for _ in 0..5 {
            assert!(limiter.check(addr, false).is_ok());
        }
        let err = limiter
            .check(addr, false)
            .expect_err("6th attempt within the window must be rejected");
        assert!(err.retry_after > Duration::ZERO);
    }

    #[test]
    fn lockout_expires_and_then_allows_again() {
        let limiter = LoginRateLimiter::new(config());
        let addr = ip(2);
        for _ in 0..5 {
            limiter.check(addr, false).expect("under limit");
        }
        limiter.check(addr, false).expect_err("locked out");
        sleep(Duration::from_millis(70));
        assert!(limiter.check(addr, false).is_ok(), "lockout should have expired");
    }

    #[test]
    fn lockout_backs_off_exponentially() {
        let limiter = LoginRateLimiter::new(config());
        let addr = ip(3);
        for _ in 0..5 {
            limiter.check(addr, false).expect("under limit");
        }
        let first_lockout = limiter.check(addr, false).expect_err("locked out").retry_after;
        sleep(first_lockout + Duration::from_millis(5));
        // Immediately exceed the limit again to trigger a second lockout.
        for _ in 0..5 {
            limiter.check(addr, false).expect("under limit after lockout expired");
        }
        let second_lockout = limiter.check(addr, false).expect_err("locked out again").retry_after;
        assert!(
            second_lockout > first_lockout,
            "second lockout ({second_lockout:?}) should be longer than the first ({first_lockout:?})"
        );
    }

    #[test]
    fn lockout_is_capped_at_max_lockout() {
        let mut cfg = config();
        cfg.max_lockout = Duration::from_millis(120);
        // This test is specifically about per-IP lockout capping, driving 36 calls total (6
        // rounds x 6). With the global-first check ordering (review finding F3/F7 — every
        // attempt counts against the global limiter now, even ones that also get per-IP
        // rejected, unlike before), the default global_limit=30 would otherwise start rejecting
        // partway through on the *global* budget instead, which is a real and correct behavior
        // change but not what this test is checking.
        cfg.global_limit = 10_000;
        let limiter = LoginRateLimiter::new(cfg);
        let addr = ip(4);
        // Drive many lockout cycles; the backoff must never exceed max_lockout.
        for _round in 0..6 {
            for _ in 0..5 {
                let _ = limiter.check(addr, false);
            }
            let err = limiter.check(addr, false).expect_err("locked out");
            assert!(err.retry_after <= Duration::from_millis(120));
            sleep(err.retry_after + Duration::from_millis(5));
        }
    }

    #[test]
    fn lockout_triggering_attempt_consumes_a_global_slot_but_locked_out_retries_do_not() {
        // Round 1's F3 fix (global check before any per-IP entry is created) had an unintended
        // consequence, reproduced in review as finding F8: EVERY attempt against an
        // already-locked-out IP kept consuming a global slot too, so a single attacker who
        // quickly triggers their own per-IP lockout and keeps hammering it could drain the whole
        // shared global budget alone — needing only one IP and a normal request rate, not a
        // flood of distinct ones. Only the one attempt that actually *triggers* the lockout
        // should cost a global slot; every retry while still locked must be turned away by the
        // read-only peek before the global counter is ever touched.
        let mut cfg = config();
        cfg.per_ip_limit = 1;
        cfg.global_limit = 3;
        let limiter = LoginRateLimiter::new(cfg);
        let addr = ip(70);

        limiter.check(addr, false).expect("1st attempt: under both limits"); // global slot 1/3
        limiter
            .check(addr, false)
            .expect_err("2nd attempt: triggers the per-IP lockout (costs 1 global slot)"); // 2/3

        // Many more attempts from the SAME now-locked-out IP must not touch the global counter.
        for _ in 0..50 {
            limiter
                .check(addr, false)
                .expect_err("locked out; must be rejected without draining the global budget");
        }

        // One global slot is still free (2 of 3 used) — a *different* IP must be able to use it,
        // proving the locked IP's 50 retries above didn't consume it.
        assert!(
            limiter.check(ip(71), false).is_ok(),
            "a different IP should still find global budget left"
        );
    }

    #[test]
    fn single_locked_out_ip_cannot_lock_out_the_owner_from_a_different_ip() {
        // Direct regression test for review finding F8's repro (scratchpad/single_ip_lockout.py):
        // ~40 wrong-password attempts from one IP, then the real owner logging in correctly from
        // a *different* IP (no trusted-device cookie) must still succeed.
        let limiter = LoginRateLimiter::new(RateLimitConfig::default());
        let attacker = ip(80);
        let owner = ip(81);

        for _ in 0..40 {
            let _ = limiter.check(attacker, false); // password verification would fail too; the
            // rate limiter alone must not be what blocks the owner below.
        }

        assert!(
            limiter.check(owner, false).is_ok(),
            "the owner, from a different IP, must not be caught by the attacker's lockout"
        );
    }

    #[test]
    fn different_ips_are_independent() {
        let limiter = LoginRateLimiter::new(config());
        for _ in 0..5 {
            limiter.check(ip(10), false).expect("under limit");
        }
        limiter.check(ip(10), false).expect_err("ip(10) should be locked out");
        assert!(
            limiter.check(ip(11), false).is_ok(),
            "a different IP must be unaffected"
        );
    }

    #[test]
    fn record_success_clears_backoff() {
        let limiter = LoginRateLimiter::new(config());
        let addr = ip(20);
        for _ in 0..4 {
            limiter.check(addr, false).expect("under limit");
        }
        limiter.record_success(addr);
        // A fresh burst of 5 should be allowed again right away.
        for _ in 0..5 {
            assert!(limiter.check(addr, false).is_ok());
        }
    }

    #[test]
    fn global_limit_rejects_even_distinct_ips() {
        let mut cfg = config();
        cfg.per_ip_limit = 1000; // keep the per-IP limiter out of the way
        cfg.global_limit = 3;
        let limiter = LoginRateLimiter::new(cfg);
        for i in 0..3 {
            limiter.check(ip(30 + i), false).expect("under global limit");
        }
        let err = limiter
            .check(ip(99), false)
            .expect_err("global limit should reject a new IP too");
        assert!(err.retry_after > Duration::ZERO);
    }

    #[test]
    fn bypass_global_skips_only_the_global_check() {
        let mut cfg = config();
        cfg.per_ip_limit = 1000;
        cfg.global_limit = 2;
        let limiter = LoginRateLimiter::new(cfg);
        // Saturate the global limit with two distinct IPs.
        limiter.check(ip(40), false).expect("under global limit");
        limiter.check(ip(41), false).expect("under global limit");
        limiter.check(ip(42), false).expect_err("global limit exhausted");

        // A bypassing caller (e.g. a trusted device) still gets through...
        assert!(
            limiter.check(ip(43), true).is_ok(),
            "bypass_global should skip the global check"
        );
        // ...but is still fully subject to its own per-IP limiter.
        let mut cfg2 = config();
        cfg2.per_ip_limit = 1;
        cfg2.global_limit = 1000;
        let limiter2 = LoginRateLimiter::new(cfg2);
        limiter2.check(ip(50), true).expect("first attempt under per-ip limit");
        limiter2
            .check(ip(50), true)
            .expect_err("bypass_global must not also bypass the per-IP limiter");
    }

    #[test]
    fn ipv6_addresses_are_rate_limited_by_64_bit_prefix() {
        let mut cfg = config();
        cfg.per_ip_limit = 2;
        cfg.global_limit = 100_000;
        let limiter = LoginRateLimiter::new(cfg);
        let addr_of = |low: u16| -> IpAddr { IpAddr::V6(Ipv6Addr::new(0x2001, 0x0db8, 0, 0, 0, 0, 0, low)) };
        // Two different low-64-bit addresses within the same /64 share one bucket.
        limiter.check(addr_of(1), false).expect("under limit");
        limiter.check(addr_of(2), false).expect("under limit, same /64");
        limiter
            .check(addr_of(3), false)
            .expect_err("third address in the same /64 should be rejected");
        assert_eq!(
            limiter.tracked_ip_count(),
            1,
            "same-/64 addresses must collapse to one entry"
        );
    }

    #[test]
    fn ipv6_addresses_in_different_64_prefixes_are_independent() {
        let mut cfg = config();
        cfg.per_ip_limit = 1;
        cfg.global_limit = 100_000;
        let limiter = LoginRateLimiter::new(cfg);
        let a = IpAddr::V6(Ipv6Addr::new(0x2001, 0x0db8, 0, 1, 0, 0, 0, 1));
        let b = IpAddr::V6(Ipv6Addr::new(0x2001, 0x0db8, 0, 2, 0, 0, 0, 1));
        limiter.check(a, false).expect("under limit");
        limiter.check(b, false).expect("a different /64 must be independent");
    }

    #[test]
    fn purge_expired_removes_only_stale_unlocked_entries() {
        let mut cfg = config();
        cfg.per_ip_window = Duration::from_millis(30);
        cfg.per_ip_limit = 1000; // no lockout in play
        let limiter = LoginRateLimiter::new(cfg);
        limiter.check(ip(60), false).expect("attempt 1");
        sleep(Duration::from_millis(60));
        limiter.check(ip(61), false).expect("attempt 2, fresh window");
        let removed = limiter.purge_expired();
        assert_eq!(removed, 1, "only the stale entry (ip 60) should be purged");
        assert_eq!(limiter.tracked_ip_count(), 1);
    }

    #[test]
    fn purge_expired_keeps_still_locked_out_entries() {
        let mut cfg = config();
        cfg.per_ip_limit = 1;
        cfg.per_ip_window = Duration::from_millis(20);
        cfg.base_lockout = Duration::from_secs(3600); // long lockout, must survive a purge
        let limiter = LoginRateLimiter::new(cfg);
        limiter.check(ip(62), false).expect("attempt 1");
        limiter.check(ip(62), false).expect_err("attempt 2 triggers lockout");
        sleep(Duration::from_millis(40)); // window elapses, but the lockout should not
        let removed = limiter.purge_expired();
        assert_eq!(removed, 0, "a still-locked-out entry must not be purged");
    }

    #[test]
    fn table_size_is_capped_by_evicting_the_oldest_entry() {
        let mut cfg = config();
        cfg.per_ip_limit = 1000;
        cfg.global_limit = 1_000_000;
        cfg.max_tracked_ips = 3;
        let limiter = LoginRateLimiter::new(cfg);
        for i in 0..3u8 {
            limiter.check(ip(100 + i), false).expect("under cap");
            sleep(Duration::from_millis(5)); // ensure distinct window_start ordering
        }
        assert_eq!(limiter.tracked_ip_count(), 3);
        // A 4th distinct IP must evict the oldest (ip 100) rather than growing past the cap.
        limiter.check(ip(200), false).expect("still allowed, but should evict");
        assert_eq!(limiter.tracked_ip_count(), 3, "table must stay at the cap");
    }
}
