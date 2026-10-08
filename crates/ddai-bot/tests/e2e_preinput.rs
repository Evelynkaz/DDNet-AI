//! Task 3.20 e2e (D-112): the server's pre-inputs against a **private** DDNet 20.1 server that this test starts itself (UDP 127.0.0.1:8445, econ
//! 127.0.0.1:8446, `sv_register 0`, `sv_preinput 1`, its own scratch directory and econ password; stopped afterwards), with two real clients: a scripted
//! opponent (walks, jumps, hooks) and an observer that does what the bot does with them -- a `LiveWorld` fed with the snapshots and
//! `LiveWorld::on_pre_input` fed with the `Sv_PreInput` messages. `#[ignore]`d and guarded by `DDAI_E2E=1`:
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddai-bot --test e2e_preinput -- --ignored --nocapture --test-threads=1
//! ```
//!
//! What it shows, with the server's own messages and snapshots as the witnesses:
//! 1. **Pre-inputs arrive** at a client that announces 20010 (the observer), for the opponent's id, ahead of the snapshots that confirm them (the lead
//!    histogram is printed).
//! 2. **The prediction error.** At every snapshot `S` the observer predicts the opponent's position for `S + 2` and `S + 4` twice, without and with
//!    the pre-inputs, and compares both with the position the snapshot of that tick confirms. The error with pre-inputs must not exceed the one without,
//!    and the share of exact (< 1 px) predictions must not fall; both are printed (this is the measurement for the default of D-112).
//! 3. **`sv_preinput 0`** (the control): no message arrives, so the prediction is the old one by construction.

mod support;

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ddai_client::{Client, ClientConfig, ClientEvent, SessionEvent};
use ddai_net::generated::messages::ExGameMsg;
use ddai_physics::core::PlayerInput;
use ddai_world::{LiveWorld, SnapshotInput};
use support::private_server::{MAP, Scratch, home, random_hex, scripted_input, spawn_opponent, start_private_server};

const GAME_PORT: u16 = 8445;
const ECON_PORT: u16 = 8446;
const OBSERVER: &str = "E2ePreObserver";
const OPPONENT: &str = "E2ePreOpponent";

#[derive(Default, Clone)]
struct Stat {
    n: u32,
    sum: f64,
    exact: u32,
}

impl Stat {
    fn add(&mut self, e: f32) {
        self.n += 1;
        self.sum += f64::from(e);
        self.exact += u32::from(e < 1.0);
    }

    fn mean(&self) -> f64 {
        self.sum / f64::from(self.n.max(1))
    }

    fn exact_pct(&self) -> f64 {
        100.0 * f64::from(self.exact) / f64::from(self.n.max(1))
    }
}

/// A prediction waiting for its tick: horizon, opponent id, the predicted positions without and with the pre-inputs.
type Pending = (i32, i32, [[f32; 2]; 2]);

struct Outcome {
    received: u64,
    counts: ddai_world::preinput::PreInputCounts,
    /// `[horizon][without, with]`.
    err: BTreeMap<i32, [Stat; 2]>,
    snapshots: u32,
}

fn observe(server_addr: SocketAddr, scratch: &std::path::Path, secs: u64) -> Outcome {
    let map_bytes = std::fs::read(scratch.join("maps").join(format!("{MAP}.map"))).expect("the map");
    let map = Arc::new(ddai_map::load_map(&map_bytes).expect("map loads").data);
    let mut client = Client::connect(
        server_addr,
        ClientConfig {
            name: OBSERVER.to_string(),
            cache_dir: scratch.join("cache-obs"),
            ..ClientConfig::default()
        },
    );
    let mut live: Option<LiveWorld> = None;
    let mut received = 0u64;
    let mut err: BTreeMap<i32, [Stat; 2]> = BTreeMap::new();
    // Predictions waiting for their tick: tick -> (horizon, opponent id, [without, with] positions).
    let mut pending: BTreeMap<i32, Vec<Pending>> = BTreeMap::new();
    let mut snapshots = 0u32;
    let end = Instant::now() + Duration::from_secs(secs);
    let mut last_snapshot_tick: Option<i32> = None;
    let t0 = Instant::now();
    let mut last_input = Instant::now();
    while Instant::now() < end {
        // The observer moves too (a player that does not is AFK for the server, and an AFK player gets no pre-inputs), on a shifted timeline.
        if last_input.elapsed() >= Duration::from_millis(20) {
            client.set_input(scripted_input(t0.elapsed().as_millis() as i64 + 1000));
            last_input = Instant::now();
        }
        let Some(ev) = client.recv_event(Duration::from_millis(10)) else {
            continue;
        };
        match ev {
            ClientEvent::Session(s) => {
                if let SessionEvent::ExGameMessage(ExGameMsg::SvPreInput(p)) = *s {
                    received += 1;
                    if let Some(l) = live.as_mut() {
                        l.on_pre_input(
                            p.owner,
                            p.intended_tick,
                            PlayerInput {
                                direction: p.direction,
                                target_x: p.target_x,
                                target_y: p.target_y,
                                jump: p.jump,
                                fire: p.fire,
                                hook: p.hook,
                                ..PlayerInput::default()
                            },
                        );
                    }
                }
            }
            ClientEvent::LiveWorldSnapshot(s) => {
                let Some(own) = s.own_id else { continue };
                let l = live.get_or_insert_with(|| {
                    let mut l = LiveWorld::new(Arc::clone(&map), own, 1);
                    l.set_preinput(false);
                    l
                });
                l.on_snapshot(SnapshotInput {
                    tick: s.tick,
                    characters: &s.characters,
                    tuning: s.tuning,
                    switch_states: &s.switch_states,
                    teams: s.teams.as_ref(),
                    own_input_at_tick: None,
                    projectiles: &s.projectiles,
                });
                snapshots += 1;
                last_snapshot_tick = Some(s.tick);
                // Score the predictions made earlier for this tick against what this snapshot confirms.
                if let Some(list) = pending.remove(&s.tick) {
                    for (h, opp, pos) in list {
                        if let Some(c) = l.base_world().cores.get(opp as u8) {
                            let truth = [c.pos.x, c.pos.y];
                            let e = err.entry(h).or_default();
                            for (k, p) in pos.iter().enumerate() {
                                e[k].add(((p[0] - truth[0]).powi(2) + (p[1] - truth[1]).powi(2)).sqrt());
                            }
                        }
                    }
                }
                pending.retain(|t, _| *t >= s.tick);
                // The opponent: the other character, alive, unfrozen (a frozen tee plays no input).
                let Some(opp) = s.characters.iter().map(|c| c.id).find(|&id| id != own) else {
                    continue;
                };
                let frozen = l.base_world().characters[opp as usize]
                    .as_ref()
                    .is_none_or(|c| c.freeze_time > 0 || !c.alive);
                if frozen {
                    continue;
                }
                for h in [2, 4] {
                    let mut pos = [[0.0f32; 2]; 2];
                    for (k, on) in [false, true].into_iter().enumerate() {
                        l.set_preinput(on);
                        let w = l.predict(s.tick + h, &[]);
                        if let Some(c) = w.cores.get(opp as u8) {
                            pos[k] = [c.pos.x, c.pos.y];
                        }
                    }
                    l.set_preinput(false);
                    pending.entry(s.tick + h).or_default().push((h, opp, pos));
                }
            }
            _ => {}
        }
    }
    let _ = last_snapshot_tick;
    client.disconnect();
    client.join();
    let counts = live.map(|l| l.pre_inputs().counts()).unwrap_or_default();
    Outcome {
        received,
        counts,
        err,
        snapshots,
    }
}

fn run(preinput: bool, opponent_margin_ms: Option<i32>) -> Outcome {
    let scratch = Scratch(std::env::temp_dir().join(format!("ddai-e2e-pre-{}-{}", std::process::id(), random_hex())));
    std::fs::create_dir_all(&scratch.0).unwrap();
    let server = start_private_server(
        &scratch.0,
        GAME_PORT,
        ECON_PORT,
        &format!("sv_preinput {}\nsv_max_preinputs_per_tick 0", u8::from(preinput)),
    );
    let server_addr: SocketAddr = format!("127.0.0.1:{GAME_PORT}").parse().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let opp = spawn_opponent(
        server_addr,
        scratch.0.join("cache-opp"),
        Arc::clone(&stop),
        OPPONENT,
        opponent_margin_ms,
    );
    assert!(
        (0..60).any(|_| {
            std::thread::sleep(Duration::from_millis(500));
            server.econ("status").is_some_and(|s| s.contains(OPPONENT))
        }),
        "the scripted opponent never joined"
    );
    let out = observe(server_addr, &scratch.0, 60);
    stop.store(true, Ordering::SeqCst);
    opp.join().unwrap();
    let _ = home();
    drop(server);
    out
}

fn report(tag: &str, o: &Outcome) {
    eprintln!(
        "[{tag}] {} snapshots, {} pre-input messages; counters {:?}",
        o.snapshots,
        o.received,
        (
            o.counts.stored,
            o.counts.ahead,
            o.counts.behind,
            o.counts.stale,
            o.counts.invalid,
            o.counts.used,
            o.counts.distrusted
        )
    );
    eprintln!(
        "[{tag}] trust checks {} (dead-reckoned {}), distrusted {} (dead-reckoned {})",
        o.counts.checked, o.counts.checked_reckoned, o.counts.distrusted, o.counts.distrusted_reckoned
    );
    eprintln!(
        "[{tag}] lead (intended tick - latest snapshot tick, bins {}..={}): {:?}",
        ddai_world::preinput::LEAD_MIN,
        ddai_world::preinput::LEAD_MAX,
        o.counts.lead
    );
    eprintln!(
        "[{tag}] at the snapshots: newest message tick - snapshot tick (bins {}..={}), what a decision from the snapshot can use: {:?}",
        ddai_world::preinput::LEAD_MIN,
        ddai_world::preinput::LEAD_MAX,
        o.counts.known_ahead
    );
    for (h, e) in &o.err {
        eprintln!(
            "[{tag}] horizon {h}: without pre-inputs mean {:.3} px, exact {:.1}% | with {:.3} px, exact {:.1}% ({} predictions)",
            e[0].mean(),
            e[0].exact_pct(),
            e[1].mean(),
            e[1].exact_pct(),
            e[0].n
        );
    }
}

#[test]
#[ignore = "starts a private DDNet server on 127.0.0.1:8445/8446; DDAI_E2E=1 and --ignored"]
fn pre_inputs_arrive_and_make_the_prediction_of_a_real_opponent_better() {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return;
    }
    // 1. The opponent with DDNet's default prediction margin (10 ms): its input is at the server less than a tick before it is used.
    let on = run(true, None);
    report("sv_preinput 1, margin 10 ms", &on);
    assert!(
        on.received > 100,
        "pre-inputs arrive at a client that announces 20010: {}",
        on.received
    );
    assert!(on.counts.stored > 100 && on.counts.invalid == 0, "{:?}", on.counts);
    assert!(
        on.counts.ahead > 0,
        "some of them are ahead of the snapshot that confirms them"
    );
    for (h, e) in &on.err {
        assert!(e[0].n > 100, "horizon {h}: {} predictions", e[0].n);
        assert!(
            e[1].mean() <= e[0].mean() + 0.05,
            "horizon {h}: worse: {} vs {}",
            e[1].mean(),
            e[0].mean()
        );
    }
    // 2. An opponent with a 60 ms margin (3 ticks; what a jittery connection settles at): its pre-inputs reach us before the snapshots that show the
    //    ticks they are for, so the prediction knows its real input past the snapshot.
    let wide = run(true, Some(60));
    report("sv_preinput 1, margin 60 ms", &wide);
    let past_the_snapshot: u64 = wide.counts.known_ahead[(1 - ddai_world::preinput::LEAD_MIN) as usize..]
        .iter()
        .sum();
    assert!(
        past_the_snapshot > 100,
        "the messages reach past the snapshot at a wide margin: {:?}",
        wide.counts.known_ahead
    );
    for (h, e) in &wide.err {
        assert!(
            e[1].mean() <= e[0].mean() + 1e-6,
            "horizon {h}: the pre-inputs made the mean error worse: {} vs {}",
            e[1].mean(),
            e[0].mean()
        );
        assert!(
            e[1].exact_pct() + 0.5 >= e[0].exact_pct(),
            "horizon {h}: fewer exact predictions"
        );
    }
    assert!(
        wide.err.values().any(|e| e[1].mean() < e[0].mean()),
        "the pre-inputs improved the prediction of an opponent whose input they precede: {:?}",
        wide.err
            .iter()
            .map(|(h, e)| (*h, e[0].mean(), e[1].mean()))
            .collect::<Vec<_>>()
    );
    // 3. The control: no messages, so "with" and "without" are the same predictions.
    let off = run(false, None);
    report("sv_preinput 0", &off);
    assert_eq!(off.received, 0, "the server sends none");
    for e in off.err.values() {
        assert_eq!(e[0].sum, e[1].sum);
    }
}
