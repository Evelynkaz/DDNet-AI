//! Task 4.10 e2e (D-100): after the proxy drops the SOCKS5 control connection mid-game, the bot reconnects through a new association (a new
//! address for the server) and, with the DDNet timeout code it sent at join, **takes its old tee back** instead of joining as `(1)Name`
//! next to a ghost. Against a **private** DDNet 20.1 server that this test starts itself (UDP 127.0.0.1:8443, econ 127.0.0.1:8444,
//! `sv_register 0`, its own scratch directory, its own random econ password, `conn_timeout 5`, stopped afterwards). `#[ignore]`d and guarded
//! by `DDAI_E2E=1`:
//!
//! ```text
//! DDAI_E2E=1 cargo test -p ddnet-ai --test e2e_timeout_takeover -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Two tests run the same three phases: the real one (with the seed) and **the control without the seed** (the behaviour before task 4.10), so
//! each claim below is also shown to fail without the change.
//!
//! Loopback only. Nothing here touches the shared server (8303), the production units, `~/aiddnet/data/bot`, the allow-list or the secrets:
//! the seed file, the map cache and the allow-list entry live in the scratch directory (the server binary and the map are only read). The
//! proxy is the in-process `TestSocks5Server` with its relay on 127.0.0.2 (public mode, test hook), behind a small TCP gate that can refuse
//! new proxy connections for a while (an outage of the proxy provider).
//!
//! What it proves, with the server's own `status` and log as the witnesses:
//! 1. **Join**: the bot joins through the relay, the server lists one player with the bot's name, and the bot said exactly one chat line,
//!    the `/timeout <code>` (audit label `Cl_Say(/timeout)`, none refused; the server's chat log has nothing from the bot).
//! 2. **Takeover**: the control connection is killed mid-game and the proxy stays unreachable for 5.5 s (longer than the server's
//!    `conn_timeout 5`, so the server has noticed the old connection is dead when the new one asks): the bot reconnects through a **new**
//!    association; the server lists ONE player with the same name, no `(1)` prefix, in the **same slot** (the tee is taken over: the server logs
//!    `Timeout Protection used`); the bot said one more `/timeout` and nothing else.
//! 3. **The limit, and the repeat** (owner's decision of 2026-10-06). The control connection is killed again and the bot reconnects at
//!    once (in game after about 1 s, long before the server's `conn_timeout`): the old connection is not yet timed out, so the join's
//!    `/timeout` takes nothing over; the new join is `(1)Name` (`CServer::SetClientName`, `"(%d)%s"`) and the old tee stays as a ghost,
//!    beyond `conn_timeout`, because it sent `/timeout` and is timeout-protected. The bot sees a same-name player that is not itself and
//!    repeats the same `/timeout <code>` 30 s after the join's send; the ghost's connection is in the error state by then, so the repeat
//!    **takes it over**: `status` lists ONE player with the original name in the ghost's slot, the server drops `(1)Name` with
//!    "Timeout Protection used", and no further repeat follows (the cadence and the cap of 35 are unit-tested).
//!
//! The control test (no seed) shows the old tee dropped by the timeout and a ghost that times out by itself.

// Task 5.5a: an ignored end-to-end test against a local DDNet server driven by python3/bash/POSIX tools: Linux only.
#![cfg(unix)]

mod owner_chat_rig;

use std::collections::BTreeMap;
use std::io::Read;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ddai_client::live_servers::{LiveServerEntry, LiveServers};
use ddai_client::proxy::{ProxyConfig, RelayMode};
use ddai_client::session::TIMEOUT_CODE_LABEL;
use ddai_client::socks5_testserver::{Auth, Config, TestSocks5Server};
use ddai_client::{Client, ClientConfig, ClientEvent, SessionEvent};
use owner_chat_rig::{PrivateServer, Scratch, chat_lines, random_hex, read_log, start_private_server};

const GAME_PORT: u16 = 8443;
const ECON_PORT: u16 = 8444;
const NAME: &str = "E2eTimeout";
const USER: &str = "e2e-user";
const PASS: &str = "e2e-pass";

/// A TCP front for the proxy that can refuse new connections (the provider's outage) while the test is running.
struct Gate {
    addr: SocketAddr,
    open: Arc<AtomicBool>,
}

impl Gate {
    fn start(upstream: SocketAddr) -> Gate {
        let listener = TcpListener::bind("127.0.0.1:0").expect("gate port");
        let addr = listener.local_addr().unwrap();
        let open = Arc::new(AtomicBool::new(true));
        let flag = open.clone();
        std::thread::spawn(move || {
            for client in listener.incoming().flatten() {
                if !flag.load(Ordering::SeqCst) {
                    continue; // dropped at once: connection reset/closed
                }
                let Ok(up) = TcpStream::connect(upstream) else { continue };
                for (mut from, mut to) in [(client.try_clone().unwrap(), up.try_clone().unwrap()), (up, client)] {
                    std::thread::spawn(move || {
                        let mut buf = [0u8; 2048];
                        while let Ok(n) = from.read(&mut buf) {
                            if n == 0 || std::io::Write::write_all(&mut to, &buf[..n]).is_err() {
                                break;
                            }
                        }
                        let _ = from.shutdown(std::net::Shutdown::Both);
                        let _ = to.shutdown(std::net::Shutdown::Both);
                    });
                }
            }
        });
        Gate { addr, open }
    }

    fn set_open(&self, open: bool) {
        self.open.store(open, Ordering::SeqCst);
    }
}

/// What the bot's event stream showed so far.
#[derive(Default)]
struct Seen {
    in_game: u32,
    in_game_at: Option<Instant>,
    /// `OutgoingGame` per label: (accepted, refused).
    outgoing: BTreeMap<&'static str, (u32, u32)>,
    own_id: Option<i32>,
    snapshots_since_in_game: u32,
    reconnect_attempts: u32,
}

impl Seen {
    fn count(&self, label: &str) -> u32 {
        self.outgoing.get(label).map_or(0, |c| c.0)
    }
}

/// Reads the bot's events for up to `limit` or until `done` holds.
fn pump(client: &mut Client, seen: &mut Seen, limit: Duration, done: impl Fn(&Seen) -> bool) {
    let end = Instant::now() + limit;
    while Instant::now() < end && !done(seen) {
        match client.recv_event(Duration::from_millis(50)) {
            Some(ClientEvent::Session(ev)) => match *ev {
                SessionEvent::InGame => {
                    seen.in_game += 1;
                    seen.in_game_at = Some(Instant::now());
                    seen.snapshots_since_in_game = 0;
                }
                SessionEvent::OutgoingGame { label, accepted } => {
                    let c = seen.outgoing.entry(label).or_default();
                    if accepted {
                        c.0 += 1;
                    } else {
                        c.1 += 1;
                    }
                }
                _ => {}
            },
            Some(ClientEvent::LiveWorldSnapshot(s)) => {
                seen.own_id = s.own_id;
                seen.snapshots_since_in_game += 1;
            }
            Some(ClientEvent::ReconnectAttempt { .. }) => seen.reconnect_attempts += 1,
            Some(ClientEvent::GaveUp { reason, .. }) => panic!("the bot gave up: {reason}"),
            _ => {}
        }
    }
}

/// `(slot, name, address)` of every player of the server's `status`.
fn players(server: &PrivateServer) -> Vec<(i32, String, String)> {
    let out = server.econ("status").expect("econ status");
    out.lines()
        .filter_map(|l| {
            let id = l.split("id=").nth(1)?.split_whitespace().next()?.parse().ok()?;
            let name = l.split("name='").nth(1)?.split('\'').next()?.to_string();
            let addr = l.split("addr=<{").nth(1)?.split("}>").next()?.to_string();
            Some((id, name, addr))
        })
        .collect()
}

fn no_chat_but_the_timeout_code(seen: &Seen) {
    for (label, (_accepted, refused)) in &seen.outgoing {
        assert!(
            *label == TIMEOUT_CODE_LABEL || (!label.contains("Say") && !label.contains("Chat")),
            "chat on the wire besides the timeout code: {label}"
        );
        assert_eq!(*refused, 0, "the allow-list refused {label}");
    }
}

fn seed_for(scratch: &Path) -> ddai_net::timeout_code::TimeoutSeed {
    let path = ddai_client::timeout_seed::path_in(scratch);
    let seed = ddai_client::timeout_seed::load_or_create(&path).expect("the seed file");
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "the seed file is private");
    seed
}

#[test]
#[ignore = "starts a private DDNet server on 127.0.0.1:8443/8444; DDAI_E2E=1 and --ignored"]
fn a_reconnect_after_a_proxy_drop_takes_the_old_tee_back_with_the_timeout_code() {
    scenario(true);
}

/// The control: the very same outages with no seed (no `/timeout`, the behaviour before task 4.10). The old tee is dropped by the server's
/// `conn_timeout` and the bot enters as a new player, and a ghost of a fast reconnect times out after `conn_timeout` instead of being held.
#[test]
#[ignore = "starts a private DDNet server on 127.0.0.1:8443/8444; DDAI_E2E=1 and --ignored"]
fn without_the_timeout_code_the_same_outage_drops_the_old_tee() {
    scenario(false);
}

fn scenario(with_code: bool) {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return;
    }
    // `/timeout`s on the wire after `joins` joins.
    let codes = |joins: u32| if with_code { joins } else { 0 };
    let scratch = Scratch(std::env::temp_dir().join(format!("ddai-e2e-timeout-{}", random_hex())));
    std::fs::create_dir_all(&scratch.0).unwrap();
    let server = start_private_server(
        &scratch.0,
        GAME_PORT,
        ECON_PORT,
        "aiddnet e2e 4.10 (private, 127.0.0.1 only)",
        "conn_timeout 5\n",
    );
    let server_addr: SocketAddr = format!("127.0.0.1:{GAME_PORT}").parse().unwrap();

    let proxy = TestSocks5Server::start(Config {
        auth: Auth::UserPass(USER.into(), PASS.into()),
        relay_ip: Some("127.0.0.2".parse().unwrap()),
        ..Default::default()
    });
    let gate = Gate::start(proxy.addr());
    let seed = seed_for(&scratch.0);
    let cfg = ClientConfig {
        name: NAME.to_string(),
        cache_dir: scratch.0.join("cache"),
        emit_outgoing_audit: true,
        timeout_seed: with_code.then_some(seed),
        live_servers: LiveServers {
            servers: vec![LiveServerEntry {
                address: server_addr.to_string(),
                nick: NAME.to_string(),
                purpose: "e2e timeout takeover".to_string(),
                ready: true,
                proxy: Some("e2e".to_string()),
            }],
        },
        proxy: Some(
            ProxyConfig::new(
                "e2e",
                gate.addr.ip().to_string(),
                gate.addr.port(),
                Some((USER.to_string(), PASS.to_string())),
            )
            .unwrap()
            .with_relay(RelayMode::Public)
            .with_test_loopback_relay()
            .with_for_server(server_addr.to_string()),
        ),
        ..ClientConfig::default()
    };
    let mut client = Client::connect(server_addr, cfg);
    let mut seen = Seen::default();

    // ---- 1. join ------------------------------------------------------------------------------------------------
    pump(&mut client, &mut seen, Duration::from_secs(40), |s| {
        s.in_game >= 1 && s.snapshots_since_in_game > 60
    });
    assert_eq!(
        seen.count(TIMEOUT_CODE_LABEL),
        codes(1),
        "the bot sent its timeout code once at the join (and none without a seed)"
    );
    let first = players(&server);
    eprintln!("[1] after the join, the server lists: {first:?}");
    assert_eq!(first.len(), 1, "one player: {first:?}");
    assert_eq!(first[0].1, NAME);
    let slot = first[0].0;
    assert_eq!(seen.own_id, Some(slot), "the bot's own id is the server's slot");
    assert!(
        first[0].2.starts_with("127.0.0.2:"),
        "the traffic comes out of the relay: {first:?}"
    );
    let old_addr = first[0].2.clone();
    no_chat_but_the_timeout_code(&seen);
    assert!(
        chat_lines(&read_log(&scratch.0), NAME).is_empty(),
        "the server's chat log has nothing from the bot (the command is consumed, not said)"
    );

    // ---- 2. proxy outage longer than the server's conn_timeout ------------------------------------------------------
    eprintln!("[2] killing the SOCKS5 control connection; the proxy stays down for 5.5 s");
    let accepts = proxy.tcp_accepts();
    gate.set_open(false);
    proxy.drop_control_connections();
    let killed = Instant::now();
    pump(&mut client, &mut seen, Duration::from_millis(5500), |_| false);
    gate.set_open(true);
    pump(&mut client, &mut seen, Duration::from_secs(40), |s| {
        s.in_game >= 2 && s.snapshots_since_in_game > 60
    });
    eprintln!(
        "[2] back in game {:.1} s after the kill, {} reconnect attempt(s), {} proxy connection(s) since",
        killed.elapsed().as_secs_f32(),
        seen.reconnect_attempts,
        proxy.tcp_accepts() - accepts
    );
    assert_eq!(seen.in_game, 2, "the bot reconnected");
    assert_eq!(proxy.associations(), 2, "through a new association");
    assert_eq!(
        seen.count(TIMEOUT_CODE_LABEL),
        codes(2),
        "one `/timeout` per join, no more"
    );
    let second = players(&server);
    eprintln!("[2] the server lists: {second:?}");
    assert_eq!(second.len(), 1, "ONE player, no ghost: {second:?}");
    assert_eq!(second[0].1, NAME, "the same name, no `(1)` prefix");
    assert_ne!(second[0].2, old_addr, "from a new address");
    assert_eq!(seen.own_id, Some(second[0].0), "the bot sees itself in its slot");
    let log = read_log(&scratch.0);
    let takeovers: Vec<&str> = log.lines().filter(|l| l.contains("Timeout Protection used")).collect();
    let own_joins = log.matches(&format!("'{NAME}' entered and joined the game")).count();
    let own_drops: Vec<&str> = log
        .lines()
        .filter(|l| l.contains(&format!("'{NAME}' has left the game")))
        .collect();
    for l in takeovers.iter().chain(own_drops.iter()) {
        eprintln!("[2] server log: {l}");
    }
    if with_code {
        assert_eq!(second[0].0, slot, "the same slot: the tee is the old one");
        assert_eq!(takeovers.len(), 1, "exactly one takeover: {takeovers:?}");
        // The old tee never left: the server did not drop it for the timeout and the bot did not enter again as that player.
        assert_eq!(own_joins, 1, "the original tee joined once and was not replaced");
        assert!(
            own_drops.is_empty(),
            "the original tee was never dropped: {own_drops:?}"
        );
        assert!(
            log.contains(&format!(
                "'{NAME}' would have timed out, but can use timeout protection now"
            )),
            "the server held the old connection under timeout protection"
        );
    } else {
        // The control: the server dropped the old tee for the timeout, and the bot entered as a new player.
        assert!(takeovers.is_empty(), "no takeover without the code: {takeovers:?}");
        assert_eq!(own_joins, 2, "the bot entered again as a new player");
        assert_eq!(own_drops.len(), 1, "the old tee timed out: {own_drops:?}");
        assert!(own_drops[0].contains("(Timeout)"), "{own_drops:?}");
    }
    no_chat_but_the_timeout_code(&seen);
    assert!(chat_lines(&log, NAME).is_empty(), "still nothing in the chat log");

    // ---- 3. immediate reconnect: nothing to take over yet -----------------------------------------------------------
    eprintln!("[3] killing the control connection again; the proxy is up, the bot reconnects at once");
    proxy.drop_control_connections();
    let killed = Instant::now();
    pump(&mut client, &mut seen, Duration::from_secs(40), |s| {
        s.in_game >= 3 && s.snapshots_since_in_game > 60
    });
    let back_after = seen.in_game_at.expect("in game again").duration_since(killed);
    eprintln!("[3] in game again {:.1} s after the kill", back_after.as_secs_f32());
    assert_eq!(seen.in_game, 3);
    assert!(
        back_after < Duration::from_secs(5),
        "the reconnect is immediate (about 1 s), before the server's conn_timeout: {back_after:?}"
    );
    let third = players(&server);
    eprintln!("[3] the server lists: {third:?}");
    // The new join is `(1)Name`: the old connection was still there when it joined, so nothing could be taken over. (The old tee is
    // still listed as a ghost only with the code: without it the ghost times out after `conn_timeout`, 5 s here, which can be over by
    // the time `status` is asked.)
    assert!(third.iter().any(|p| p.1 == format!("(1){NAME}")), "{third:?}");
    if with_code {
        assert_eq!(third.len(), 2, "the old tee is still there as a ghost: {third:?}");
        assert!(third.iter().any(|p| p.1 == NAME), "{third:?}");
    }
    no_chat_but_the_timeout_code(&seen);
    assert_eq!(seen.count(TIMEOUT_CODE_LABEL), codes(3));
    // What happens to the ghost: a connection that sent `/timeout` is held under timeout protection (up to `conn_timeout_protection`,
    // 1000 s), so it stays; a ghost without the code is dropped after `conn_timeout` (5 s here, 100 s by default).
    pump(&mut client, &mut seen, Duration::from_secs(8), |_| false);
    let later = players(&server);
    eprintln!("[3] 8 s later (longer than conn_timeout) the server lists: {later:?}");
    if with_code {
        assert_eq!(later.len(), 2, "the protected ghost is still there: {later:?}");
        // The owner's decision of 2026-10-06: the same `/timeout <code>` is repeated every 30 s while a same-name ghost is in the
        // snapshots. The ghost's connection reached the error state long ago (5 s), so the first repeat takes it over: our slot becomes the
        // ghost's, `(1)Name` is dropped with "Timeout Protection used", and the repeats stop.
        let ghost_slot = later.iter().find(|p| p.1 == NAME).expect("the ghost").0;
        let before = seen.count(TIMEOUT_CODE_LABEL);
        let waited = Instant::now();
        pump(&mut client, &mut seen, Duration::from_secs(45), |s| {
            s.count(TIMEOUT_CODE_LABEL) > before
        });
        assert_eq!(
            seen.count(TIMEOUT_CODE_LABEL),
            before + 1,
            "one repeat of the same code"
        );
        pump(&mut client, &mut seen, Duration::from_secs(3), |_| false);
        let after = players(&server);
        eprintln!(
            "[3] the repeat went out {:.0} s after the 8 s check (30 s after the join's send); the server lists: {after:?}",
            waited.elapsed().as_secs_f32()
        );
        assert_eq!(after.len(), 1, "ONE player, the ghost is taken over: {after:?}");
        assert_eq!(after[0].1, NAME, "with the original name");
        assert_eq!(after[0].0, ghost_slot, "in the ghost's slot");
        assert_eq!(seen.own_id, Some(ghost_slot), "the bot sees itself there");
        let log = read_log(&scratch.0);
        let takeovers = log.lines().filter(|l| l.contains("Timeout Protection used")).count();
        assert_eq!(takeovers, 2, "the second takeover is the ghost's");
        assert!(
            log.lines()
                .any(|l| l.contains("'(1)E2eTimeout' has left the game (Timeout Protection used)")),
            "the `(1)` player was dropped by the takeover"
        );
        // The ghost is gone: no more repeats (the cadence is unit-tested; here a further 35 s show none)
        let total = seen.count(TIMEOUT_CODE_LABEL);
        pump(&mut client, &mut seen, Duration::from_secs(35), |_| false);
        assert_eq!(seen.count(TIMEOUT_CODE_LABEL), total, "no repeat without a ghost");
        no_chat_but_the_timeout_code(&seen);
        assert!(chat_lines(&read_log(&scratch.0), NAME).is_empty());
    } else {
        assert_eq!(later.len(), 1, "the ghost timed out: {later:?}");
    }

    client.disconnect();
    client.join();
}
