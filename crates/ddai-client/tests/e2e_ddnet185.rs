//! Task 2.3b: joins a local **DDNet 18.5** server (the version the Swarfey server is based on) with
//! this crate's 20.1 client, as a player (idle) and as an observer (`Cl_SetTeam(-1)`), and checks
//! the properties the Swarfey incident violated: exactly one `Connected` per join, a map, in game,
//! no reconnects, no give-up, clean shutdown.
//!
//! Never part of a plain `cargo test` (`#[ignore]` + the `DDAI_E2E_185=1` env gate, exactly like
//! `e2e_local_server.rs`). Needs a DDNet 18.5 server on loopback — see docs/SETUP.md section 5d for
//! how to build and start one (default `127.0.0.1:8306`, override with `DDAI_E2E_185_ADDR`). The
//! address must be loopback: the crate's live-servers safety switch refuses anything else.

use ddai_client::{Client, ClientConfig, ClientEvent, SessionEvent};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

struct JoinReport {
    connected: u32,
    map_loaded: bool,
    in_game: bool,
    reconnect_attempts: u32,
    server_requested_reconnects: u32,
    redirects: u32,
    gave_up: Option<String>,
    spectating: bool,
}

fn join(addr: SocketAddr, name: &str, observer: bool) -> JoinReport {
    let cache_dir = std::env::temp_dir().join(format!("ddai-e2e-185-{}", std::process::id()));
    let config = ClientConfig {
        name: name.to_string(),
        cache_dir,
        ..ClientConfig::default()
    };
    let mut client = Client::connect(addr, config);
    let mut report = JoinReport {
        connected: 0,
        map_loaded: false,
        in_game: false,
        reconnect_attempts: 0,
        server_requested_reconnects: 0,
        redirects: 0,
        gave_up: None,
        spectating: false,
    };
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut in_game_at: Option<Instant> = None;
    while Instant::now() < deadline && report.gave_up.is_none() {
        match client.recv_event(Duration::from_millis(50)) {
            Some(ClientEvent::Session(ev)) => match *ev {
                SessionEvent::Connected => report.connected += 1,
                SessionEvent::MapLoaded(_) => report.map_loaded = true,
                SessionEvent::InGame => {
                    report.in_game = true;
                    in_game_at = Some(Instant::now());
                    if observer {
                        client.set_team(-1);
                    }
                }
                _ => {}
            },
            Some(ClientEvent::OwnTeam { team, .. }) if observer && team == -1 => report.spectating = true,
            Some(ClientEvent::ReconnectAttempt { .. }) => report.reconnect_attempts += 1,
            Some(ClientEvent::ServerRequestedReconnect { .. }) => report.server_requested_reconnects += 1,
            Some(ClientEvent::RedirectFollowed { .. }) => report.redirects += 1,
            Some(ClientEvent::GaveUp { reason, .. }) => report.gave_up = Some(reason),
            _ => {}
        }
        // Stay in game for a few seconds (long enough for a spurious second Connected or a
        // reconnect to show up), or until the observer switch is confirmed.
        if let Some(t) = in_game_at
            && t.elapsed() > Duration::from_secs(4)
        {
            break;
        }
    }
    client.disconnect();
    client.join();
    // Drain the tail (MarginSummary etc.) so a late GaveUp is not missed.
    for ev in client.events() {
        if let ClientEvent::GaveUp { reason, .. } = ev {
            report.gave_up = Some(reason);
        }
    }
    report
}

fn server_addr() -> SocketAddr {
    std::env::var("DDAI_E2E_185_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8306".to_string())
        .parse()
        .expect("DDAI_E2E_185_ADDR must be a socket address")
}

fn gated() -> bool {
    if std::env::var("DDAI_E2E_185").as_deref() != Ok("1") {
        eprintln!("skipping: set DDAI_E2E_185=1 (needs a DDNet 18.5 server on loopback, see docs/SETUP.md 5d)");
        return false;
    }
    true
}

#[test]
#[ignore = "needs a local DDNet 18.5 server; run with DDAI_E2E_185=1"]
fn joins_ddnet_185_as_player() {
    if !gated() {
        return;
    }
    let r = join(server_addr(), "E185Play", false);
    assert_eq!(r.connected, 1, "Connected must fire exactly once per real connection");
    assert!(r.map_loaded, "no map was loaded");
    assert!(r.in_game, "never reached in game");
    assert_eq!(r.reconnect_attempts + r.server_requested_reconnects + r.redirects, 0);
    assert_eq!(r.gave_up, None);
}

#[test]
#[ignore = "needs a local DDNet 18.5 server; run with DDAI_E2E_185=1"]
fn joins_ddnet_185_as_observer() {
    if !gated() {
        return;
    }
    let r = join(server_addr(), "E185Obs", true);
    assert_eq!(r.connected, 1, "Connected must fire exactly once per real connection");
    assert!(r.map_loaded && r.in_game);
    assert!(r.spectating, "the observer switch (Cl_SetTeam(-1)) was never confirmed");
    assert_eq!(r.reconnect_attempts + r.server_requested_reconnects + r.redirects, 0);
    assert_eq!(r.gave_up, None);
}

/// Needs the *patched* 18.5 server that answers every `NETMSG_INFO` with `reconnect@ddnet.org`
/// (`DDAI_TEST_RECONNECT_FLOOD=1`, docs/SETUP.md 5d) — `DDAI_E2E_185_FLOOD_ADDR` names it. This is
/// the Swarfey incident against a real server binary: the client must follow one reconnect, then
/// stop by itself with `GaveUpCategory::ReconnectLoop`.
#[test]
#[ignore = "needs the patched local DDNet 18.5 flood server; run with DDAI_E2E_185=1 and DDAI_E2E_185_FLOOD_ADDR"]
fn stops_on_a_reconnect_flooding_18_5_server() {
    if !gated() {
        return;
    }
    let Ok(addr) = std::env::var("DDAI_E2E_185_FLOOD_ADDR") else {
        eprintln!("skipping: DDAI_E2E_185_FLOOD_ADDR not set");
        return;
    };
    let addr: SocketAddr = addr.parse().expect("DDAI_E2E_185_FLOOD_ADDR must be a socket address");
    let mut client = Client::connect(
        addr,
        ClientConfig {
            name: "E185Flood".to_string(),
            ..ClientConfig::default()
        },
    );
    let mut connected = 0;
    let mut category = None;
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline && category.is_none() {
        match client.recv_event(Duration::from_millis(50)) {
            Some(ClientEvent::Session(ev)) if matches!(*ev, SessionEvent::Connected) => connected += 1,
            Some(ClientEvent::GaveUp { category: c, .. }) => category = Some(c),
            _ => {}
        }
    }
    client.join();
    assert_eq!(category, Some(ddai_client::GaveUpCategory::ReconnectLoop));
    assert_eq!(
        connected, 2,
        "one initial connection plus exactly one followed reconnect"
    );
}

/// Review F1 against a real full server: `DDAI_E2E_185_FULL_ADDR` names an 18.5 server started with
/// `sv_max_clients 1` (docs/SETUP.md 5d). One client holds the only slot; a second must get
/// "This server is full" and make at most 2 connection attempts before stopping by itself with
/// `GaveUpCategory::TooManyAttempts`, without disturbing the holder.
#[test]
#[ignore = "needs a local DDNet 18.5 server with sv_max_clients 1; run with DDAI_E2E_185=1 and DDAI_E2E_185_FULL_ADDR"]
fn a_full_18_5_server_gets_at_most_two_attempts() {
    if !gated() {
        return;
    }
    let Ok(addr) = std::env::var("DDAI_E2E_185_FULL_ADDR") else {
        eprintln!("skipping: DDAI_E2E_185_FULL_ADDR not set");
        return;
    };
    let addr: SocketAddr = addr.parse().expect("DDAI_E2E_185_FULL_ADDR must be a socket address");
    let config = |name: &str| ClientConfig {
        name: name.to_string(),
        cache_dir: std::env::temp_dir().join(format!("ddai-e2e-185-full-{}", std::process::id())),
        ..ClientConfig::default()
    };

    let mut holder = Client::connect(addr, config("E185Hold"));
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut holder_in_game = false;
    while Instant::now() < deadline && !holder_in_game {
        if let Some(ClientEvent::Session(ev)) = holder.recv_event(Duration::from_millis(50))
            && matches!(*ev, SessionEvent::InGame)
        {
            holder_in_game = true;
        }
    }
    assert!(holder_in_game, "the holder never reached in game");

    let mut second = Client::connect(addr, config("E185Full"));
    let mut attempts_announced = 0;
    let mut category = None;
    let mut ever_in_game = false;
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline && category.is_none() {
        match second.recv_event(Duration::from_millis(50)) {
            Some(ClientEvent::ReconnectAttempt { .. }) => attempts_announced += 1,
            Some(ClientEvent::Session(ev)) if matches!(*ev, SessionEvent::InGame) => ever_in_game = true,
            Some(ClientEvent::GaveUp { category: c, .. }) => category = Some(c),
            _ => {}
        }
    }
    second.join();
    holder.disconnect();
    holder.join();

    assert!(!ever_in_game);
    assert_eq!(category, Some(ddai_client::GaveUpCategory::TooManyAttempts));
    assert_eq!(attempts_announced, 1, "one retry, then stop: 2 attempts in total");
}
