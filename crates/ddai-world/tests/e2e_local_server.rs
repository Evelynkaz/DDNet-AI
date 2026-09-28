//! Live accuracy test against the local DDNet server (task spec, acceptance criterion 2:
//! "Accuracy on the local server" / criterion 4's e2e gate). `#[ignore]`d by default; run with:
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddai-world --test e2e_local_server -- --ignored --nocapture
//! ```
//!
//! CLAUDE.md's live-play policy: loopback only (`127.0.0.1:8303`), never chats (D-007 — this
//! crate's inputs never touch chat at all), and this specific accuracy test is explicitly allowed
//! several of our own scripted clients on the local server (two here: `own`, which `LiveWorld`
//! tracks itself with a genuinely *moving* input, and `other`, purely a moving body so there is
//! someone else to predict).
//!
//! Review round 1, finding F5 (confirmed live: without `own_inputs_in_flight` built from the
//! *actual* sent-input history and corrected for late deliveries, own-tee prediction against a
//! real server measured ~0.998 exact — just under the task spec's `>= 0.999` bar — because
//! `NETMSG_INPUTTIMING` occasionally reports an input as having missed its intended tick):
//!
//! - `own`'s `ClientConfig::emit_input_sent = true` turns on `SessionEvent::InputSent`/
//!   `InputTiming`, delivered as `ClientEvent::Session(..)` (review round 3, finding F12: matches
//!   task 8.4a's own shape exactly — see `ddai-client`'s `ClientEvent::is_droppable` doc comment).
//! - `own` now sends a genuinely *varying* input (not `idle`, which trivially can't reveal any
//!   jitter-induced misprediction at all — a constant value is late-delivery-immune by
//!   construction) — see [`own_input`].
//! - The whole sent-input history + each reported `time_left_ms` is collected live, then
//!   retargeted once (review round 3, finding F11: [`ddai_world::retarget_late_inputs`], not the
//!   round-1/2 [`correct_late_inputs`]-only model — see that function's own doc comment for why)
//!   and replayed through `LiveWorld` — see this file's own two-phase structure
//!   (`collect_live_data` then the replay loop below).
//! - Review round 3, finding F12: both events are **droppable** (matching 8.4a's own decision to
//!   avoid unbounded queue growth) — a dropped `InputSent` is tolerated exactly the way
//!   [`retarget_late_inputs`]/[`ddai_world::correct_late_inputs`] already tolerate any other gap
//!   in the sent-input log (hold the previous known input); this test does not need special-case
//!   handling for it beyond that (the live bot already knows what it sent itself, so a dropped
//!   *delivery* of that same fact back to the accuracy harness never produces a wrong prediction,
//!   only a slightly staler one for the affected tick — no different from a gap this model
//!   already had to tolerate).
//! - Review round 3, finding F10: `own`'s own tee is seeded from the real applied input at each
//!   snapshot's base tick (the same retargeted sent-input log, looked up at that exact tick), not
//!   the neutral `fire: 0` guess — see `LiveWorld::on_snapshot`'s `own_input_at_tick` parameter.
//! - Review round 3, finding F13: the own-tee accuracy bar only counts samples where the own tee
//!   is unfrozen at *both* the base and target ticks (`AccuracyTracker::record_prediction`'s
//!   `base_frozen` parameter, `AccuracyTracker::summarize`'s own filtering) and requires a minimum
//!   sample count ([`MIN_OWN_TEE_SAMPLES`]) — the reviewer's own report notes `BlockField` keeps
//!   the own tee unfrozen for the whole run, so this test is meant to be run against it
//!   (`tools/ddnet-server/econ.py change_map BlockField` before the run,
//!   `... change_map "Copy Love Box"` to restore the server's normal default map afterward — an
//!   *operational* step, deliberately kept outside this test itself so the test stays map-agnostic
//!   like every other assertion in this file, per finding F8 below).
//!
//! Review round 1, finding F8: the map is identified from the server's own `MapChanging` event
//! (never hardcoded), and a map change mid-run aborts the test rather than silently computing
//! accuracy against mismatched map data.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ddai_client::{Client, ClientConfig, ClientEvent, LiveWorldSnapshot, SessionEvent};
use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects::PlayerInput as NetPlayerInput;
use ddai_physics::core::PlayerInput;
use ddai_world::accuracy::AccuracyTracker;
use ddai_world::{LiveWorld, player_input_from_net, retarget_late_inputs};

fn e2e_enabled() -> bool {
    std::env::var("DDAI_E2E").as_deref() == Ok("1")
}

fn data_dir() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(home) if !home.is_empty() => PathBuf::from(home).join("aiddnet").join("data"),
        _ => PathBuf::from("data"),
    }
}

fn idle_input() -> NetPlayerInput {
    NetPlayerInput {
        direction: 0,
        target_x: 0,
        target_y: -1,
        jump: 0,
        fire: 0,
        hook: 0,
        player_flags: playerflagflag::PLAYING,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    }
}

/// A small deterministic xorshift64* PRNG — test-harness scaffolding (not physics under test),
/// same pattern `tests/synthetic_replay.rs` already uses for its own scripted input generators.
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
}

/// `own`'s own input — genuinely varying (direction/jump/hook/aim), changing every 40-200ms, so
/// this test can actually exercise (and measure) the late-delivery correction F5 is about; a
/// constant `idle` input can never reveal a misprediction from a late tick at all (holding the
/// same value across a late tick produces the identical result either way).
fn own_input(rng: &mut Rng) -> NetPlayerInput {
    let r = rng.next_u32();
    let direction = (r % 3) as i32 - 1;
    let jump = i32::from((r >> 4).is_multiple_of(4));
    let hook = i32::from((r >> 8).is_multiple_of(3));
    let angle = ((r >> 12) % 360) as f32 * std::f32::consts::PI / 180.0;
    NetPlayerInput {
        direction,
        target_x: (angle.cos() * 200.0) as i32,
        target_y: (angle.sin() * 200.0) as i32,
        jump,
        fire: 0,
        hook,
        player_flags: playerflagflag::PLAYING,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    }
}

/// A short deterministic movement cycle for the `other` bot (same shape as `ddnet-ai play
/// --brain circle`, task 2.3) — walk right, jump, walk left, jump, hook briefly — enough to
/// exercise direction changes, jumps and hook transitions so the "others" accuracy numbers mean
/// something (a purely idle other tee would trivially predict itself perfectly).
fn other_input(elapsed: Duration) -> NetPlayerInput {
    const PERIOD_MS: u128 = 2400;
    let t = elapsed.as_millis() % PERIOD_MS;
    let (direction, jump, hook) = match t {
        0..=799 => (1, 0, 0),
        800..=999 => (1, 1, 0),
        1000..=1399 => (0, 0, 1),
        1400..=2199 => (-1, 0, 0),
        _ => (-1, 1, 0),
    };
    NetPlayerInput {
        direction,
        target_x: if direction == 0 { 32 } else { direction * 32 },
        target_y: -32,
        jump,
        fire: 0,
        hook,
        player_flags: playerflagflag::PLAYING,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    }
}

/// Everything collected live, before any `LiveWorld` replay — see the module doc comment for why
/// this is a separate phase (review round 3, finding F11: the re-target correction needs the
/// *whole* sent-input history, plus every reported `time_left_ms` — not just the late ones, since
/// [`retarget_late_inputs`] needs the actual value to compute how far forward to shift).
#[derive(Default)]
struct LiveData {
    snapshots: Vec<LiveWorldSnapshot>,
    sent: Vec<(i32, PlayerInput)>,
    /// Every tick a `SessionEvent::InputTiming` was ever seen for, not just the late ones (review
    /// round 3, finding F11 — see [`retarget_late_inputs`]'s own doc comment for why the raw
    /// `time_left_ms` value matters, not just its sign).
    timing: BTreeMap<i32, i32>,
    map_name: Option<String>,
    map_changes: u32,
    own_in_game: bool,
    other_in_game: bool,
}

/// Review round 3, finding F12: `SessionEvent::InputSent`/`InputTiming` are now delivered as
/// `ClientEvent::Session(..)` (matching task 8.4a's own shape) and are **droppable** — a caller
/// must tolerate a gap in either stream. This fn's own dispatch already does exactly that: a
/// dropped `InputSent`/`InputTiming` simply never adds an entry to `data.sent`/`data.timing` for
/// that tick, which [`retarget_late_inputs`]/`ddai_world::correct_late_inputs`'s existing
/// gap-filling (hold the previous known input) already covers with no special-casing needed here.
fn dispatch_session_event(ev: SessionEvent, data: &mut LiveData) {
    match ev {
        SessionEvent::InGame => data.own_in_game = true,
        SessionEvent::MapChanging { name, .. } => {
            data.map_changes += 1;
            data.map_name = Some(name);
        }
        SessionEvent::InputSent { tick, input } => {
            data.sent.push((tick, player_input_from_net(input)));
        }
        SessionEvent::InputTiming { tick, time_left } => {
            data.timing.insert(tick, time_left);
        }
        _ => {}
    }
}

fn collect_live_data(own: &Client, other: &Client, duration: Duration) -> LiveData {
    let mut data = LiveData::default();
    let start = Instant::now();
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let mut next_change = Instant::now();
    let mut current_own_input = idle_input();

    while start.elapsed() < duration {
        if let Some(ev) = own.recv_event(Duration::from_millis(5)) {
            match ev {
                ClientEvent::Session(s) => dispatch_session_event(*s, &mut data),
                ClientEvent::LiveWorldSnapshot(snap) => data.snapshots.push(*snap),
                _ => {}
            }
        }
        while let Some(ev) = other.try_recv_event() {
            if let ClientEvent::Session(s) = ev
                && matches!(*s, SessionEvent::InGame)
            {
                data.other_in_game = true;
            }
        }

        if Instant::now() >= next_change {
            if data.own_in_game {
                current_own_input = own_input(&mut rng);
            }
            next_change = Instant::now() + Duration::from_millis(40 + (rng.next_u32() % 160) as u64);
        }
        if data.own_in_game {
            own.set_input(current_own_input);
        }
        if data.other_in_game {
            other.set_input(other_input(start.elapsed()));
        }
    }
    data
}

#[test]
#[ignore]
fn liveworld_accuracy_against_local_server() {
    if !e2e_enabled() {
        eprintln!("skipped: set DDAI_E2E=1 to run against the local ddnet-server (127.0.0.1:8303)");
        return;
    }

    let addr: SocketAddr = "127.0.0.1:8303".parse().expect("valid loopback address");
    let cache_dir = data_dir().join("maps").join("cache");

    let own_config = ClientConfig {
        name: "ddai-world-e2e-own".to_string(),
        cache_dir: cache_dir.clone(),
        emit_input_sent: true,
        ..ClientConfig::default()
    };
    let other_config = ClientConfig {
        name: "ddai-world-e2e-other".to_string(),
        cache_dir,
        ..ClientConfig::default()
    };

    let mut own = Client::connect(addr, own_config);
    let mut other = Client::connect(addr, other_config);

    let mut data = collect_live_data(&own, &other, Duration::from_secs(20));
    // Drain whatever arrived after the loop ended (input/timing events queued from the very last
    // `flush`, plus the final `MarginSummary`) before disconnecting — matches
    // `collect_live_data`'s own event handling exactly, so nothing sent right at the end is lost.
    for ev in own.events() {
        match ev {
            ClientEvent::Session(s) => dispatch_session_event(*s, &mut data),
            ClientEvent::LiveWorldSnapshot(snap) => data.snapshots.push(*snap),
            _ => {}
        }
    }

    own.disconnect();
    other.disconnect();
    own.join();
    other.join();

    assert!(
        data.own_in_game,
        "own bot never reached InGame — is ddnet-local.service running?"
    );
    assert!(
        data.other_in_game,
        "other bot never reached InGame — is ddnet-local.service running?"
    );
    // Review round 1, finding F8: never hardcode the map name; abort rather than silently compute
    // accuracy against a map that changed under us mid-run (the local server is only ever
    // expected to change map between test runs, never during one).
    let map_name = data
        .map_name
        .clone()
        .expect("must have seen at least one MapChanging event");
    assert_eq!(
        data.map_changes, 1,
        "the map changed {} times during this run (expected exactly 1, the initial join) — \
         aborting rather than computing accuracy against mismatched map data",
        data.map_changes
    );

    let map_path = data_dir()
        .join("ddnet-server")
        .join("maps")
        .join(format!("{map_name}.map"));
    let map_bytes = std::fs::read(&map_path).unwrap_or_else(|e| panic!("read {}: {e}", map_path.display()));
    let loaded = ddai_map::load_map(&map_bytes).expect("the server's own active map must load cleanly");
    let map = Arc::new(loaded.data);

    assert!(
        !data.snapshots.is_empty(),
        "must have received at least one LiveWorldSnapshot"
    );
    let own_id = data
        .snapshots
        .iter()
        .find_map(|s| s.own_id)
        .expect("must have learned our own client id");

    // Review round 3, finding F11: retarget late deliveries forward *once*, covering the whole
    // run, before replaying — see this file's module doc comment and `retarget_late_inputs`'s own
    // doc comment for the model (superseding round 1/2's `correct_late_inputs`-only one).
    data.sent.sort_by_key(|&(tick, _)| tick);
    let retargeted: BTreeMap<i32, PlayerInput> = retarget_late_inputs(&data.sent, &data.timing).into_iter().collect();
    eprintln!(
        "sent={} timing={} late(<0)={} retargeted={} map={map_name}",
        data.sent.len(),
        data.timing.len(),
        data.timing.values().filter(|&&t| t < 0).count(),
        retargeted.len()
    );

    let mut live = LiveWorld::new(Arc::clone(&map), own_id, 1);
    let mut tracker = AccuracyTracker::new(own_id);
    // Even-only: `record_actual` here only ever gets *real* snapshots to compare against (unlike
    // `tests/synthetic_replay.rs`, which has perfect per-tick ground truth from its own harness),
    // and the server only ever snapshots on even ticks (`sv_high_bandwidth = 0`, every 2 ticks) —
    // an odd horizon from an even base tick would never land on one, so no prediction at 1 or 5
    // ticks ahead could ever be confirmed here at all. `tests/synthetic_replay.rs` is what proves
    // the task spec's literal 1/5/10-tick criterion; this closest achievable even analogue
    // (2/6/10) is this test's own real-network confirmation of it.
    let horizons = [2i32, 6, 10];

    for snap in &data.snapshots {
        // Review round 3, finding F10: seed the own tee's reconstruction from the input the
        // server actually applied at this exact base tick (looked up from the same retargeted
        // sent-input log `own_inputs_in_flight` below draws from — a gap here, e.g. an event
        // dropped under F12's droppability, simply falls back to `on_snapshot`'s own documented
        // neutral guess for that one snapshot, same as `None` always meant).
        let own_input_at_tick = retargeted.get(&snap.tick).copied();
        live.on_snapshot(
            snap.tick,
            &snap.characters,
            snap.tuning,
            &snap.switch_states,
            snap.teams.as_ref(),
            own_input_at_tick,
        );
        tracker.record_actual(snap.tick, live.base_world());

        for &h in &horizons {
            let in_flight: Vec<(i32, PlayerInput)> = retargeted
                .range((snap.tick + 1)..=(snap.tick + h))
                .map(|(&t, &i)| (t, i))
                .collect();
            let predicted = live.predict(snap.tick + h, &in_flight);
            for cv in &snap.characters {
                if let Some(core) = predicted.cores.get(cv.id as u8) {
                    // Review round 3, finding F13: `cv`'s own frozen-ness *at this exact base
                    // tick* — checked again against the *target* tick's own snapshot by
                    // `AccuracyTracker::summarize` itself once that later snapshot resolves it
                    // (only ever actually consulted for `cv.id == own_id` — see that method's own
                    // doc comment — but computed per-character here for honesty either way).
                    tracker.record_prediction(snap.tick, snap.tick + h, cv.id, core.pos, is_frozen(cv));
                }
            }
        }
    }

    let summary = tracker.summarize();
    assert!(
        !summary.is_empty(),
        "must have resolved at least one prediction against a real snapshot"
    );
    for h in &summary {
        eprintln!(
            "{} horizon {} ticks: n={} exact_fraction={:.4} mean_px={:.3} p50_px={:.3} p90_px={:.3} p99_px={:.3} max_px={:.3}",
            if h.is_own { "own" } else { "other" },
            h.horizon_ticks,
            h.count,
            h.exact_fraction,
            h.mean_error_px,
            h.p50_error_px,
            h.p90_error_px,
            h.p99_error_px,
            h.max_error_px
        );
        assert!(h.count > 0);
        assert!(h.mean_error_px.is_finite() && h.mean_error_px >= 0.0);
        if h.is_own {
            // Review round 3, finding F13: a minimum sample count for the own-tee bar to mean
            // anything — without this, a run whose own tee spent nearly all its time frozen could
            // still "pass" on a handful of unfrozen-at-both-ends samples that happened to be easy.
            assert!(
                h.count >= MIN_OWN_TEE_SAMPLES,
                "only {} unfrozen-at-both-ends own-tee samples at horizon {} (need >= {}) — \
                 run against a map that keeps the own tee unfrozen (e.g. BlockField, per the \
                 review round 3 report: `tools/ddnet-server/econ.py change_map BlockField`)",
                h.count,
                h.horizon_ticks,
                MIN_OWN_TEE_SAMPLES
            );
            // The real task-spec bar (own tee, no interactions — see the module doc comment for
            // why this now holds against a real, jittery UDP connection too, not just
            // `tests/synthetic_replay.rs`'s perfect-information harness).
            assert!(
                h.exact_fraction >= 0.999,
                "own tee's bit-exact fraction ({}) is below the task spec's >= 0.999 bar at horizon {}",
                h.exact_fraction,
                h.horizon_ticks
            );
        }
    }
}

/// Review round 3, finding F13: below this, the own-tee bar's sample count is too small to trust
/// — chosen well under this test's typical yield against `BlockField` at its default 20s duration
/// (a few hundred unfrozen-at-both-ends samples per horizon — see this crate's `BUILD REPORT` for
/// the actual measured counts), so a real regression (not just run-to-run noise) still fails this
/// bound long before it could fail the `>= 0.999` one above.
const MIN_OWN_TEE_SAMPLES: usize = 100;

/// Review round 3, finding F13: DDNet's own freeze signal (`ddai_brain::CharacterObservation`'s
/// `is_frozen` derivation, `world::ddrace_tick`'s own decoding of `CNetObj_DDNetCharacter`) — a
/// character with no `DDNetCharacter` extension at all (a plain, non-DDRace snapshot item) is
/// conservatively treated as frozen (nothing proves it *isn't*), matching `AccuracyTracker`'s own
/// `unwrap_or(false)`-is-the-permissive-default asymmetry the other way around: here, the absence
/// of proof excludes a sample from the bar rather than risking a false pass.
fn is_frozen(cv: &ddai_net::view::CharacterView) -> bool {
    cv.ddnet
        .map(|d| d.freeze_end != 0 || (d.flags & (1 << 21)) != 0)
        .unwrap_or(true)
}
