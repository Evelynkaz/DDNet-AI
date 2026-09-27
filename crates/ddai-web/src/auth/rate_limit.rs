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
//! **Two layers of per-prefix limiter (review round 2, finding F8b).** IPv6 addresses are keyed
//! at TWO granularities at once, checked and updated independently:
//! - `/64` (the typical single-customer allocation unit) — the original, finer-grained limiter.
//! - `/48` — wider. A single actor holding a routed `/48` (trivially/cheaply obtained from many
//!   transit providers, unlike an IPv4 range of comparable size) has 65,536 distinct `/64`s to
//!   round-robin through, each individually staying well under the `/64` limiter's radar, while
//!   in aggregate draining the *global* budget (see below) alone from a single actor —
//!   reproduced in review as ~40+ attempts spread over 1,000 distinct `/64`s inside one `/48`.
//!   The `/48` limiter (its own window/limit/lockout, deliberately wider and looser than the
//!   `/64` one — see [`RateLimitConfig`]'s `per_48_*` fields) catches that aggregate in a way no
//!   amount of tuning the `/64` limiter alone ever could, since by definition it only ever sees
//!   one `/64`'s worth of traffic per key.
//!   IPv4 has no equivalent second tier: an IPv4 range wide enough to pull off the same trick
//!   (a `/16`+, say) is a scarce, expensive, ISP-grade allocation, a fundamentally different
//!   threat model than "free from any IPv6 transit provider" — out of scope here.
//!
//! `check`'s structure (peek both tiers → global → update both tiers) keeps three properties, all
//! load-bearing, earned across review round 1 (findings F3, F7) and round 2 (F8):
//! - **An already-locked-out key (either tier) is rejected by a read-only peek, before the global
//!   counter is ever touched** (round 1 finding F8, extended to the `/48` tier in round 2's F8b).
//!   Checking the *global* limiter first (the F3 fix, see below) had an unintended consequence:
//!   every repeat attempt from an already-locked-out key still consumed a global slot on its way
//!   to being rejected anyway, so a *single* attacker (one `/64`, or now one `/48` round-robining
//!   many `/64`s) who quickly triggers their own lockout and keeps hammering it could drain the
//!   entire global budget alone and lock the real owner out. Both peeks are read-only (no entry
//!   created, no counters touched), so neither reopens F3's unbounded-growth problem: a key with
//!   nothing tracked yet has nothing to peek at, so this can't be used to dodge the "no entry
//!   before the global check" guarantee below.
//! - **The global check runs before any new per-prefix `HashMap` entry is *created*, in either
//!   tier.** A flood of requests from unique/rotating source addresses (real for IPv6, trivial to
//!   spoof via `X-Forwarded-For` if `--trust-proxy` is misconfigured) can otherwise grow either
//!   table without bound — 800k `/64` entries / 20s were reproduced in review round 1.
//!   [`PrefixLimiter::purge_expired`] and each tier's own hard `max_tracked` cap (LRU-ish eviction
//!   by `window_start`) are defense in depth on top of that, not the primary fix.
//! - **A caller can pass `bypass_global = true`** (used by a verified trusted-device cookie, see
//!   `auth::device`) to skip *only* the global check while still being fully subject to BOTH
//!   per-prefix limiters. Without this, an attacker with enough distinct source addresses can
//!   exhaust the shared global budget and lock the legitimate owner out indefinitely — the global
//!   limit protects against a single attacker hammering the endpoint, not against a determined
//!   owner-lockout attempt, so a device that has already proven it knows the *current* password
//!   gets to skip it. Neither prefix limiter is ever bypassable this way — bypassing "am I part of
//!   a flood" checks would defeat their own point.

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
    /// Hard cap on the number of distinct `/64`(-or-v4) entries tracked at once. When a *new*
    /// key would exceed this, the entry with the oldest `window_start` is evicted first. This is
    /// a defensive backstop — under normal operation the global-first check ordering (see module
    /// docs) keeps the table far below this.
    pub max_tracked_ips: usize,
    /// Review round 2, finding F8b: a second, wider IPv6-only limiter keyed by `/48` instead of
    /// `/64`, layered on top of (checked/updated in addition to, not instead of) the `/64` one —
    /// see the module doc comment. Deliberately looser than the `/64` defaults (a *legitimate*
    /// `/48` can easily contain more than one real household/customer's worth of traffic
    /// aggregated together, e.g. an ISP's whole allocation to one PoP), while still being far
    /// tighter than the shared global budget so one abusive `/48` can never consume most of it.
    pub per_48_limit: u32,
    pub per_48_window: Duration,
    pub per_48_base_lockout: Duration,
    pub per_48_max_lockout: Duration,
    pub max_tracked_48s: usize,
}

impl Default for RateLimitConfig {
    /// Acceptance criterion 3's suggested numbers: "5 attempts/min/IP with exponential lockout,
    /// 30/min global". `per_48_limit: 20` is finding F8b's own suggested number — comfortably
    /// below `global_limit`'s 30, so even a `/48` that maxes out its own budget before locking
    /// itself out still leaves headroom in the shared budget for a legitimate login elsewhere.
    fn default() -> Self {
        Self {
            per_ip_limit: 5,
            per_ip_window: Duration::from_secs(60),
            global_limit: 30,
            global_window: Duration::from_secs(60),
            base_lockout: Duration::from_secs(30),
            max_lockout: Duration::from_secs(30 * 60),
            max_tracked_ips: 100_000,
            per_48_limit: 20,
            per_48_window: Duration::from_secs(60),
            per_48_base_lockout: Duration::from_secs(30),
            per_48_max_lockout: Duration::from_secs(30 * 60),
            max_tracked_48s: 100_000,
        }
    }
}

struct PrefixState {
    window_start: Instant,
    count_in_window: u32,
    lockout_until: Option<Instant>,
    lockout_streak: u32,
}

/// One sliding-window-with-exponential-lockout limiter keyed by an already-truncated address (a
/// `/64`-or-v4-address, or a `/48` — see module docs). Generic over nothing but the config it's
/// built with: [`LoginRateLimiter`] holds two independent instances of this, one per tier.
struct PrefixLimiter {
    limit: u32,
    window: Duration,
    base_lockout: Duration,
    max_lockout: Duration,
    max_tracked: usize,
    table: Mutex<HashMap<IpAddr, PrefixState>>,
}

impl PrefixLimiter {
    fn new(limit: u32, window: Duration, base_lockout: Duration, max_lockout: Duration, max_tracked: usize) -> Self {
        Self {
            limit,
            window,
            base_lockout,
            max_lockout,
            max_tracked,
            table: Mutex::new(HashMap::new()),
        }
    }

    /// Read-only: `Some(until)` if `key` is currently tracked and locked out (regardless of
    /// whether `until` has already passed — the caller compares against `now` itself), `None` if
    /// unknown or not locked. Never creates an entry.
    fn peek_lockout(&self, key: &IpAddr) -> Option<Instant> {
        let table = self.table.lock().expect("rate limiter mutex poisoned");
        table.get(key).and_then(|state| state.lockout_until)
    }

    /// Records one attempt against `key` and decides whether it may proceed, exactly like the
    /// crate-level docs' "Step 3" (the actual per-prefix update): sliding window, exponential
    /// lockout on exceeding `limit`. A concurrent request for the same key could have locked it
    /// out in the narrow window between an earlier [`PrefixLimiter::peek_lockout`] and this call —
    /// handled below exactly the same way a peek would have, just after the caller already spent
    /// whatever it spent (e.g. a global slot) on a very tight race, an accepted, bounded cost (see
    /// `lockout_triggering_attempt_consumes_a_global_slot_but_locked_out_retries_do_not`).
    fn record_attempt(&self, key: IpAddr, now: Instant) -> Result<(), RateLimited> {
        let mut table = self.table.lock().expect("rate limiter mutex poisoned");
        if !table.contains_key(&key) && table.len() >= self.max_tracked {
            evict_oldest(&mut table);
        }
        let state = table.entry(key).or_insert_with(|| PrefixState {
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
        if now.duration_since(state.window_start) >= self.window {
            state.window_start = now;
            state.count_in_window = 0;
        }
        state.count_in_window += 1;
        if state.count_in_window > self.limit {
            state.lockout_streak += 1;
            let shift = (state.lockout_streak - 1).min(10);
            let lockout = (self.base_lockout * (1u32 << shift)).min(self.max_lockout);
            state.lockout_until = Some(now + lockout);
            return Err(RateLimited { retry_after: lockout });
        }
        Ok(())
    }

    fn record_success(&self, key: IpAddr) {
        let mut table = self.table.lock().expect("rate limiter mutex poisoned");
        if let Some(state) = table.get_mut(&key) {
            state.count_in_window = 0;
            state.lockout_until = None;
            state.lockout_streak = 0;
        }
    }

    fn purge_expired(&self) -> usize {
        let now = Instant::now();
        let mut table = self.table.lock().expect("rate limiter mutex poisoned");
        let before = table.len();
        table.retain(|_, state| {
            let within_window = now.duration_since(state.window_start) < self.window;
            let locked = state.lockout_until.is_some_and(|until| now < until);
            within_window || locked
        });
        before - table.len()
    }

    fn tracked_count(&self) -> usize {
        self.table.lock().expect("rate limiter mutex poisoned").len()
    }
}

struct GlobalState {
    window_start: Instant,
    count_in_window: u32,
}

pub struct LoginRateLimiter {
    config: RateLimitConfig,
    per_64: PrefixLimiter,
    /// `/48` tier (finding F8b), IPv6 only — see module docs. `rate_limit_key_48` returns `None`
    /// for an IPv4 address, and callers below skip this tier entirely in that case.
    per_48: PrefixLimiter,
    global: Mutex<GlobalState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimited {
    pub retry_after: Duration,
}

/// Normalizes the key used for per-address tracking: IPv4 addresses are used as-is, IPv6
/// addresses are truncated to their `/64` routing prefix (see module docs).
fn rate_limit_key_64(ip: IpAddr) -> IpAddr {
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

/// The `/48` key for the wider IPv6-only tier (finding F8b). `None` for an IPv4 address: there is
/// no second tier for v4 (see the module doc comment on why).
fn rate_limit_key_48(ip: IpAddr) -> Option<IpAddr> {
    match ip {
        IpAddr::V4(_) => None,
        IpAddr::V6(v6) => {
            let segments = v6.segments();
            Some(IpAddr::V6(Ipv6Addr::new(
                segments[0],
                segments[1],
                segments[2],
                0,
                0,
                0,
                0,
                0,
            )))
        }
    }
}

impl LoginRateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            per_64: PrefixLimiter::new(
                config.per_ip_limit,
                config.per_ip_window,
                config.base_lockout,
                config.max_lockout,
                config.max_tracked_ips,
            ),
            per_48: PrefixLimiter::new(
                config.per_48_limit,
                config.per_48_window,
                config.per_48_base_lockout,
                config.per_48_max_lockout,
                config.max_tracked_48s,
            ),
            config,
            global: Mutex::new(GlobalState {
                window_start: Instant::now(),
                count_in_window: 0,
            }),
        }
    }

    /// Records one login attempt from `ip` and decides whether it may proceed. Call this once
    /// per `POST /api/login`, before verifying the password.
    ///
    /// `bypass_global`: skip the shared global limiter for this attempt (both per-prefix limiters
    /// still apply in full) — set this only for a request that already carries independent proof
    /// it's not part of a flood, e.g. a verified trusted-device cookie (finding F7).
    pub fn check(&self, ip: IpAddr, bypass_global: bool) -> Result<(), RateLimited> {
        let now = Instant::now();
        let key64 = rate_limit_key_64(ip);
        let key48 = rate_limit_key_48(ip);

        // Step 1 (finding F8, extended to the /48 tier by F8b): read-only peeks at BOTH tiers
        // before touching anything mutable, so an already-locked key is rejected without ever
        // reaching the global counter.
        if let Some(until) = self.per_64.peek_lockout(&key64)
            && now < until
        {
            return Err(RateLimited {
                retry_after: until - now,
            });
        }
        if let Some(key48) = key48
            && let Some(until) = self.per_48.peek_lockout(&key48)
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

        // Step 3: the actual per-prefix updates, both tiers. Both always run (not short-circuited
        // by one another) — they answer independent questions ("has this one narrow /64
        // misbehaved" vs "has this wide /48 misbehaved in aggregate") and each tier's own count
        // must reflect every attempt regardless of what the other tier decides.
        let result64 = self.per_64.record_attempt(key64, now);
        let result48 = match key48 {
            Some(key48) => self.per_48.record_attempt(key48, now),
            None => Ok(()),
        };
        result64?;
        result48
    }

    /// Clears `ip`'s backoff state (both tiers) after a successful login.
    pub fn record_success(&self, ip: IpAddr) {
        self.per_64.record_success(rate_limit_key_64(ip));
        if let Some(key48) = rate_limit_key_48(ip) {
            self.per_48.record_success(key48);
        }
    }

    /// Removes per-prefix entries (both tiers) that are neither within their current window nor
    /// still locked out (i.e. have nothing left to track). Called periodically from the same
    /// background task that purges expired sessions (finding F3: without this the tables only
    /// ever grow). Returns how many entries were removed in total across both tiers.
    pub fn purge_expired(&self) -> usize {
        self.per_64.purge_expired() + self.per_48.purge_expired()
    }

    /// Number of distinct `/64`(-or-v4) entries currently tracked. Test/observability only.
    pub fn tracked_ip_count(&self) -> usize {
        self.per_64.tracked_count()
    }

    /// Number of distinct `/48` entries currently tracked (finding F8b). Test/observability only.
    pub fn tracked_48_count(&self) -> usize {
        self.per_48.tracked_count()
    }
}

/// Evicts the single entry with the oldest `window_start` (used only when a [`PrefixLimiter`] is
/// at its configured `max_tracked` capacity — a rare, defensive path, so an O(n) scan here is
/// fine).
fn evict_oldest(map: &mut HashMap<IpAddr, PrefixState>) {
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
            per_48_limit: 20,
            per_48_window: Duration::from_secs(60),
            per_48_base_lockout: Duration::from_millis(50),
            per_48_max_lockout: Duration::from_secs(10),
            max_tracked_48s: 100_000,
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
        cfg.per_48_limit = 100_000; // keep the new /48 tier out of the way for this test
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
        cfg.per_48_limit = 100_000;
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

    // -----------------------------------------------------------------------------------------
    // /48 tier (review round 2, finding F8b)
    // -----------------------------------------------------------------------------------------

    fn v6(prefix48: u16, low64: u16, low: u16) -> IpAddr {
        IpAddr::V6(Ipv6Addr::new(0x2001, 0x0db8, prefix48, low64, 0, 0, 0, low))
    }

    #[test]
    fn many_distinct_64s_in_one_48_are_caught_by_the_48_tier_before_the_64_tier_would_notice() {
        let mut cfg = config();
        cfg.per_48_limit = 3;
        cfg.global_limit = 100_000; // isolate the /48 tier itself for this test
        let limiter = LoginRateLimiter::new(cfg);
        // Each of these is a DIFFERENT /64 (varying the 4th segment), all inside the SAME /48
        // (fixed 3rd segment) — the /64 tier alone would treat every single one as brand new.
        for low64 in 0..3u16 {
            limiter.check(v6(0xaaaa, low64, 1), false).expect("under the /48 limit");
        }
        let err = limiter
            .check(v6(0xaaaa, 999, 1), false)
            .expect_err("a 4th distinct /64 in the same /48 must still be caught by the /48 tier");
        assert!(err.retry_after > Duration::ZERO);
        assert_eq!(limiter.tracked_48_count(), 1, "all of these collapse to one /48 entry");
        // Confirm the /64 tier really did treat each as independent (proving the /48 tier, not
        // the /64 one, is what caught the 4th attempt above) — 4 entries: the 3 initial ones plus
        // the 4th /64, which the /64 tier itself was perfectly happy with on its own.
        assert_eq!(limiter.tracked_ip_count(), 4);
    }

    #[test]
    fn different_48_prefixes_are_independent() {
        let mut cfg = config();
        cfg.per_48_limit = 1;
        cfg.global_limit = 100_000;
        let limiter = LoginRateLimiter::new(cfg);
        limiter.check(v6(1, 0, 1), false).expect("under limit");
        assert!(
            limiter.check(v6(2, 0, 1), false).is_ok(),
            "a different /48 must be unaffected by the first one's usage"
        );
    }

    #[test]
    fn ipv4_addresses_never_touch_the_48_tier() {
        let mut cfg = config();
        cfg.per_48_limit = 1;
        cfg.per_ip_limit = 1000; // keep the (irrelevant here) v4/64 tier out of the way too
        cfg.global_limit = 100_000;
        let limiter = LoginRateLimiter::new(cfg);
        for _ in 0..10 {
            limiter.check(ip(1), false).expect("v4 has no /48 tier to trip");
        }
        assert_eq!(
            limiter.tracked_48_count(),
            0,
            "no /48 entries should ever be created for v4"
        );
    }

    #[test]
    fn many_distinct_64s_in_one_48_cannot_drain_the_global_budget_for_a_different_prefix() {
        // Direct regression test for finding F8b's own repro: ~40+ attempts spread over many
        // distinct /64s inside ONE /48 (here, 1000, matching the repro's own scale) must not be
        // able to exhaust the shared global budget — a login from a totally different address
        // (no device-trust bypass involved) must still find room.
        let limiter = LoginRateLimiter::new(RateLimitConfig::default());
        for low64 in 0..1000u16 {
            let _ = limiter.check(v6(0xbeef, low64, 1), false);
        }
        let owner = IpAddr::V4(Ipv4Addr::new(203, 0, 113, 9));
        assert!(
            limiter.check(owner, false).is_ok(),
            "1000 distinct /64s inside one /48 must not drain the global budget for a different address"
        );
    }

    #[test]
    fn bypass_global_does_not_bypass_the_48_tier() {
        let mut cfg = config();
        cfg.per_48_limit = 1;
        cfg.global_limit = 1000;
        let limiter = LoginRateLimiter::new(cfg);
        limiter
            .check(v6(5, 0, 1), true)
            .expect("first attempt under the /48 limit");
        limiter
            .check(v6(5, 1, 1), true) // a different /64, SAME /48
            .expect_err("bypass_global must not also bypass the /48 limiter");
    }

    #[test]
    fn purge_expired_covers_the_48_tier_too() {
        let mut cfg = config();
        cfg.per_48_window = Duration::from_millis(30);
        cfg.per_48_limit = 1000;
        cfg.global_limit = 100_000;
        let limiter = LoginRateLimiter::new(cfg);
        limiter.check(v6(1, 0, 1), false).expect("attempt 1");
        sleep(Duration::from_millis(60));
        limiter.check(v6(2, 0, 1), false).expect("attempt 2, fresh /48 window");
        let removed = limiter.purge_expired();
        assert_eq!(removed, 1, "only the stale /48 entry should be purged");
        assert_eq!(limiter.tracked_48_count(), 1);
    }
}
