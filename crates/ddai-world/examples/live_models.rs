//! Review round 3, finding F11: compares three input-correction models' own-tee prediction
//! accuracy against a real captured session from the local DDNet server, at a low
//! `prediction_margin_ms` deliberately chosen to induce late deliveries (so the three models can
//! actually be told apart — a healthy default margin rarely reports anything late at all, see
//! `tests/e2e_local_server.rs`'s own default-margin run).
//!
//! - `RAW`: no correction at all beyond plain gap-filling (hold the previous input across any
//!   tick nothing was ever sent for) — the naive baseline.
//! - `BUILDER`: round 1/2's [`correct_late_inputs`] model — a late tick holds the *previous*
//!   input in place, as if the server had simply ignored the new one for that tick.
//! - `SHIFTN`: round 3's [`retarget_late_inputs`] model — a late tick's input is re-targeted
//!   *forward* by the ceiling of its lateness in ticks, matching DDNet 20.1's own
//!   `IntendedTick = max(IntendedTick, Tick()+1)` (`server.cpp:1921`).
//!
//! Usage (loopback only, per CLAUDE.md's live-play policy — same server/rules as
//! `tests/e2e_local_server.rs`):
//!
//! ```text
//! cargo run -p ddai-world --example live_models -- <seconds> <margin_ms>
//! # e.g. at a deliberately tight margin:
//! cargo run -p ddai-world --example live_models -- 30 5
//! ```
//!
//! Review round 3, finding F14: ground truth for "was the prediction right" is
//! `lw.base_world()`'s own reconstructed core at the confirming tick (the same thing
//! `tests/e2e_local_server.rs`'s `AccuracyTracker::record_actual` compares against) — **not**
//! the raw wire `CNetObj_Character::x/y`. The wire fields are the server's own dead-reckoning
//! copy (`m_SendCore`, taken at `m_ReckoningTick` — see `crate::reckoning`'s module doc comment),
//! which can be up to ~3 seconds stale relative to the tick this exact snapshot message claims to
//! describe; comparing a prediction *for this tick* against that stale copy manufactures a
//! mismatch that has nothing to do with prediction accuracy at all. (An earlier version of this
//! file made exactly that mistake and reported a spurious ~0.19–0.35 "residual" gap it
//! misattributed to an unsimulated map entity — there was no such gap; every one of those misses
//! was simply the wire copy lagging behind `lw.base_world()`, which already had the fully
//! reconstructed, correct current-tick position.)

use ddai_client::{Client, ClientConfig, ClientEvent, LiveWorldSnapshot, SessionEvent};
use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects::PlayerInput as NetInput;
use ddai_physics::core::PlayerInput;
use ddai_world::{LiveWorld, SnapshotInput, correct_late_inputs, player_input_from_net, retarget_late_inputs};
use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One model's own-tee accuracy at each of `HORIZONS`, evaluated the same way
/// `tests/e2e_local_server.rs` does (review round 3, finding F13: skips every base/target tick
/// where the own tee is frozen at either end, matching `AccuracyTracker::summarize`'s own filter —
/// reimplemented directly against `LiveWorld` here rather than via `AccuracyTracker`/
/// `ddai_world::accuracy`, since this is a standalone comparison across three *different*
/// `own_inputs_in_flight` logs, not one accumulating tracker).
const HORIZONS: [i32; 2] = [2, 10];

fn is_frozen(cv: &ddai_net::view::CharacterView) -> bool {
    cv.ddnet
        .map(|d| d.freeze_end != 0 || (d.flags & (1 << 21)) != 0)
        .unwrap_or(true)
}

/// `(horizon, predicted (x, y))` pairs still waiting for their target tick's confirmation.
type PendingByTick = BTreeMap<i32, Vec<(i32, (f32, f32))>>;

fn eval(
    label: &str,
    snaps: &[LiveWorldSnapshot],
    map: &Arc<ddai_physics::map::MapData>,
    own_id: i32,
    sent: &BTreeMap<i32, PlayerInput>,
) {
    let mut lw = LiveWorld::new(Arc::clone(map), own_id, 1);
    let mut pending: PendingByTick = BTreeMap::new();
    let mut stats: BTreeMap<i32, (usize, usize)> = BTreeMap::new(); // horizon -> (n, exact)

    for s in snaps {
        let own_input_at_tick = sent.get(&s.tick).copied();
        lw.on_snapshot(SnapshotInput {
            tick: s.tick,
            characters: &s.characters,
            tuning: s.tuning,
            switch_states: &s.switch_states,
            teams: s.teams.as_ref(),
            own_input_at_tick,
            projectiles: &s.projectiles,
        });

        if let Some(pending_here) = pending.remove(&s.tick)
            && let Some(actual) = s.characters.iter().find(|c| c.id == own_id)
        {
            let base_frozen_at_confirmation = is_frozen(actual);
            for (h, (px, py)) in pending_here {
                let entry = stats.entry(h).or_default();
                if base_frozen_at_confirmation {
                    continue; // F13: frozen at the target tick — excluded, not counted at all.
                }
                entry.0 += 1;
                // Review round 3, finding F14: ground truth is `lw.base_world()`'s own
                // reconstructed core at *this* confirming tick — `on_snapshot` for `s.tick` has
                // already run above, in this same loop iteration — not the raw wire
                // `actual.character.x/y` (the server's own stale `m_SendCore` dead-reckoning
                // copy — see this file's own module doc comment).
                let truth = lw.base_world().cores.get(own_id as u8).map(|c| (c.pos.x, c.pos.y));
                let ok = truth == Some((px, py));
                entry.1 += usize::from(ok);
                if !ok && std::env::var("MISS").is_ok() {
                    eprintln!(
                        "MISS label={label} tick={} h={h} pred=({px},{py}) truth={truth:?} wire=({},{}) wire_tick={}",
                        s.tick, actual.character.x, actual.character.y, actual.character.tick
                    );
                }
            }
        }

        let frozen_now = s
            .characters
            .iter()
            .find(|c| c.id == own_id)
            .map(is_frozen)
            .unwrap_or(true);
        if frozen_now {
            continue; // F13: frozen at the base tick — never even predict from here.
        }
        for &h in &HORIZONS {
            let inflight: Vec<(i32, PlayerInput)> =
                sent.range((s.tick + 1)..=(s.tick + h)).map(|(&t, &i)| (t, i)).collect();
            let predicted = lw.predict(s.tick + h, &inflight);
            if let Some(c) = predicted.cores.get(own_id as u8) {
                pending.entry(s.tick + h).or_default().push((h, (c.pos.x, c.pos.y)));
            }
        }
    }

    let line: Vec<String> = stats
        .iter()
        .map(|(h, &(n, e))| {
            if n == 0 {
                format!("h={h} n=0 (no unfrozen-at-both-ends samples)")
            } else {
                format!("h={h} exact={:.4} (n={n})", e as f64 / n as f64)
            }
        })
        .collect();
    println!("{label:>8}: {}", line.join("  "));
}

fn main() {
    let secs: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let margin: i32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    let data = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("aiddnet/data");
    let addr: SocketAddr = "127.0.0.1:8303".parse().unwrap();
    let cache_dir = data.join("maps").join("cache");
    let own = Client::connect(
        addr,
        ClientConfig {
            name: "ddai-world-live-models".to_string(),
            cache_dir,
            emit_input_sent: true,
            prediction_margin_ms: margin,
            ..ClientConfig::default()
        },
    );

    let mut map_name = None;
    let mut snaps: Vec<LiveWorldSnapshot> = Vec::new();
    let mut sent: Vec<(i32, NetInput)> = Vec::new();
    let mut timing: BTreeMap<i32, i32> = BTreeMap::new();
    let mut in_game = false;
    let start = Instant::now();
    let mut rng: u64 = 0x9e3779b97f4a7c15;
    let mut next = Instant::now();

    while start.elapsed() < Duration::from_secs(secs) {
        while let Some(ev) = own.recv_event(Duration::from_millis(2)) {
            match ev {
                ClientEvent::Session(s) => match *s {
                    SessionEvent::InGame => in_game = true,
                    SessionEvent::MapChanging { name, .. } => map_name = Some(name),
                    SessionEvent::InputSent { tick, input } => sent.push((tick, input)),
                    SessionEvent::InputTiming { tick, time_left } => {
                        timing.insert(tick, time_left);
                    }
                    _ => {}
                },
                ClientEvent::LiveWorldSnapshot(s) => snaps.push(*s),
                _ => {}
            }
        }
        if in_game && Instant::now() >= next {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let r = (rng >> 32) as u32;
            let d = (r % 3) as i32 - 1;
            let a = ((r >> 12) % 360) as f32 * 0.01745;
            own.set_input(NetInput {
                direction: d,
                target_x: (a.cos() * 200.0) as i32,
                target_y: (a.sin() * 200.0) as i32,
                jump: i32::from((r >> 4).is_multiple_of(4)),
                fire: 0,
                hook: i32::from((r >> 8).is_multiple_of(3)),
                player_flags: playerflagflag::PLAYING,
                wanted_weapon: 0,
                next_weapon: 0,
                prev_weapon: 0,
            });
            next = Instant::now() + Duration::from_millis(40 + (r >> 24) as u64 % 200);
        }
    }
    own.disconnect();
    let mut own = own;
    own.join();
    for ev in own.events() {
        if let ClientEvent::Session(s) = ev {
            match *s {
                SessionEvent::InputSent { tick, input } => sent.push((tick, input)),
                SessionEvent::InputTiming { tick, time_left } => {
                    timing.insert(tick, time_left);
                }
                _ => {}
            }
        }
    }

    let map_name = map_name.expect("must have seen at least one MapChanging event");
    let map_path = data.join("ddnet-server").join("maps").join(format!("{map_name}.map"));
    let map = Arc::new(
        ddai_map::load_map(&std::fs::read(&map_path).unwrap_or_else(|e| panic!("read {}: {e}", map_path.display())))
            .unwrap()
            .data,
    );
    let own_id = snaps
        .iter()
        .find_map(|s| s.own_id)
        .expect("must have learned our own client id");

    let sentp: Vec<(i32, PlayerInput)> = sent.iter().map(|&(t, i)| (t, player_input_from_net(i))).collect();
    let late0: BTreeSet<i32> = timing.iter().filter(|e| *e.1 < 0).map(|e| *e.0).collect();
    println!(
        "map={map_name} margin={margin}ms sent={} timing={} late(<0)={}",
        sent.len(),
        timing.len(),
        late0.len()
    );

    let fill = |m: BTreeMap<i32, PlayerInput>| -> BTreeMap<i32, PlayerInput> {
        let v: Vec<(i32, PlayerInput)> = m.into_iter().collect();
        correct_late_inputs(&v, &BTreeSet::new()).into_iter().collect()
    };

    // RAW: plain gap-filling, no lateness handling at all.
    eval("RAW", &snaps, &map, own_id, &fill(sentp.iter().copied().collect()));
    // BUILDER: round 1/2's model — a late tick holds the previous input in place.
    eval(
        "BUILDER",
        &snaps,
        &map,
        own_id,
        &correct_late_inputs(&sentp, &late0).into_iter().collect(),
    );
    // SHIFTN: round 3's model — re-target a late input forward, first-arrival-wins.
    eval(
        "SHIFTN",
        &snaps,
        &map,
        own_id,
        &retarget_late_inputs(&sentp, &timing).into_iter().collect(),
    );
}
