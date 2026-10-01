//! Cannon prediction against the real local 20.1 server (task 2.4b review round 1, F2 — the
//! reviewer's own end-to-end check, carried into task 4.1): BlmapChill has 12 crazy-shotgun cannons,
//! and a huge `Cl_ShowDistance` makes the server snap all of them to us, so this compares
//! `LiveWorld`'s projectile prediction with the server's own item `k` ticks later, bounces included.
//! (The builder's own e2e, `e2e_local_server.rs`, can only see gun bullets with the default
//! `show_distance`; this one sees the cannons.)
//!
//! `#[ignore]`d. Loopback only, never chats. Before the run:
//!
//! ```text
//! tools/ddnet-server/econ.py change_map BlmapChill      # (from the repo root; password from the secrets file)
//! DDAI_E2E=1 cargo test -p ddai-world --test review_cannons_e2e -- --ignored --nocapture
//! tools/ddnet-server/econ.py change_map "Copy Love Box" # restore the server's normal map afterwards
//! ```
//!
//! Reviewer's measurement on this setup: 36 960 comparisons, 7 496 with a server bounce inside the
//! window, 0 `start_tick` mismatches, largest position error 0.0039 px.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ddai_client::{Client, ClientConfig, ClientEvent, LiveWorldSnapshot, SessionEvent};
use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects::PlayerInput as NetPlayerInput;
use ddai_world::projectiles::projectile_from_view;
use ddai_world::{LiveWorld, SnapshotInput};

/// A predicted position further than this from the server's is a mismatch (the reviewer saw 0.0039).
const MAX_POSITION_ERROR_PX: f32 = 0.05;

#[test]
#[ignore = "needs the local DDNet server on BlmapChill; run with DDAI_E2E=1 -- --ignored"]
fn cannon_prediction_matches_the_server_including_bounces() {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        return;
    }
    let home = std::env::var("HOME").expect("HOME");
    let data = std::path::PathBuf::from(&home).join("aiddnet/data");
    let addr: SocketAddr = "127.0.0.1:8303".parse().unwrap();
    let config = ClientConfig {
        name: "ddai-cannons".to_string(),
        cache_dir: data.join("maps").join("cache"),
        show_distance: (200_000, 200_000),
        ..ClientConfig::default()
    };
    let mut client = Client::connect(addr, config);
    let mut snaps: Vec<LiveWorldSnapshot> = Vec::new();
    let mut map_name = None;
    let mut in_game = false;
    let start = Instant::now();
    let idle = NetPlayerInput {
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
    };
    let secs: u64 = std::env::var("SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    while start.elapsed() < Duration::from_secs(secs) {
        if let Some(ev) = client.recv_event(Duration::from_millis(5)) {
            match ev {
                ClientEvent::Session(s) => match *s {
                    SessionEvent::InGame => in_game = true,
                    SessionEvent::MapChanging { name, .. } => map_name = Some(name),
                    _ => {}
                },
                ClientEvent::LiveWorldSnapshot(s) => snaps.push(*s),
                _ => {}
            }
        }
        if in_game {
            client.set_input(idle);
        }
    }
    client.disconnect();
    client.join();

    let map_name = map_name.expect("the server announced a map");
    let bytes = std::fs::read(data.join("ddnet-server/maps").join(format!("{map_name}.map"))).expect("map file");
    let map = Arc::new(ddai_map::load_map(&bytes).expect("map loads").data);
    let own_id = snaps.iter().find_map(|s| s.own_id).expect("own client id");
    let by_tick: BTreeMap<i32, &LiveWorldSnapshot> = snaps.iter().map(|s| (s.tick, s)).collect();
    let max_items = snaps.iter().map(|s| s.projectiles.len()).max().unwrap_or(0);
    eprintln!(
        "map={map_name} snapshots={} max projectile items in one snapshot={max_items}",
        snaps.len()
    );
    assert!(
        max_items > 0,
        "no projectile items seen on {map_name}: switch the server to BlmapChill first"
    );

    let mut live = LiveWorld::new(Arc::clone(&map), own_id, 1);
    let (mut compared, mut start_tick_mismatch, mut position_bad, mut bounces_seen) = (0usize, 0usize, 0usize, 0usize);
    let mut max_err = 0f32;
    let mut per_k: BTreeMap<i32, (usize, usize, f32)> = BTreeMap::new();
    for s in &snaps {
        for k in [2, 4, 6, 8, 10] {
            let Some(s2) = by_tick.get(&(s.tick + k)) else { continue };
            live.on_snapshot(SnapshotInput {
                tick: s.tick,
                characters: &s.characters,
                tuning: s.tuning,
                switch_states: &s.switch_states,
                teams: s.teams.as_ref(),
                own_input_at_tick: None,
                projectiles: &s.projectiles,
            });
            let mut ids: Vec<i32> = s.projectiles.iter().map(|p| p.0).collect();
            ids.sort_unstable();
            if live.base_world().projectiles.len() != ids.len() {
                continue; // an item the world has no model for: no 1:1 mapping
            }
            let predicted = live.predict(s2.tick, &[]).projectiles.clone();
            if predicted.len() != ids.len() {
                continue; // a shot ended (explosion, freeze hit) inside the window
            }
            let w = live.base_world();
            for (id, p) in ids.iter().zip(&predicted) {
                let Some((_, tv)) = s2.projectiles.iter().find(|(i, _)| i == id) else {
                    continue;
                };
                let truth = projectile_from_view(tv, s2.tick, &w.collision, &w.tuning).expect("server item models");
                let before = s.projectiles.iter().find(|(i, _)| i == id).expect("base item");
                let before =
                    projectile_from_view(&before.1, s.tick, &w.collision, &w.tuning).expect("base item models");
                if truth.start_tick != before.start_tick {
                    bounces_seen += 1;
                }
                compared += 1;
                let entry = per_k.entry(k).or_insert((0, 0, 0.0));
                entry.0 += 1;
                if p.start_tick != truth.start_tick {
                    start_tick_mismatch += 1;
                    entry.1 += 1;
                    continue;
                }
                // The position at the target snapshot's tick, evaluated from each item's own origin.
                let at = |q: &ddai_physics::world::Projectile<f32>| {
                    let z = w.tuning.zone(q.tune_zone);
                    let (curvature, speed) = match q.weapon_type {
                        2 => (z.shotgun_curvature::<f32>(), z.shotgun_speed::<f32>()),
                        3 => (z.grenade_curvature::<f32>(), z.grenade_speed::<f32>()),
                        _ => (z.gun_curvature::<f32>(), z.gun_speed::<f32>()),
                    };
                    let t = (s2.tick - q.start_tick) as f32 / 50.0 * speed;
                    (
                        q.pos.x + q.direction.x * t,
                        q.pos.y + q.direction.y * t + curvature / 10000.0 * t * t,
                    )
                };
                let (pc, tc) = (at(p), at(&truth));
                let err = (pc.0 - tc.0).abs().max((pc.1 - tc.1).abs());
                entry.2 = entry.2.max(err);
                max_err = max_err.max(err);
                if err > MAX_POSITION_ERROR_PX {
                    position_bad += 1;
                }
            }
        }
    }
    eprintln!(
        "compared={compared} bounces_in_window={bounces_seen} start_tick_mismatch={start_tick_mismatch} \
         position_err>{MAX_POSITION_ERROR_PX}={position_bad} max_err={max_err}"
    );
    for (k, v) in &per_k {
        eprintln!("  k={k}: n={} start_tick_mismatch={} max_err={}", v.0, v.1, v.2);
    }
    assert!(compared > 1000, "too few comparisons ({compared}) to mean anything");
    assert_eq!(start_tick_mismatch, 0, "a predicted bounce disagrees with the server's");
    assert_eq!(
        position_bad, 0,
        "a predicted position is off by more than {MAX_POSITION_ERROR_PX} px"
    );
}
