//! Task 3.11: the live-timing harness. `#[ignore]`d; run with
//!
//! ```text
//! DDAI_E2E=1 DDAI_T_LABEL=base DDAI_T_SERVER=127.0.0.1:8453 \
//!   cargo test --release -p ddai-bot --test e2e_live_timing -- --ignored --nocapture
//! ```
//!
//! (`tools/e2e/live_timing.sh` wraps it: reloads a private server's map around the run so the server's teehistorian of the run is
//! closed, then joins it with the bot's input trace — the server-side truth of when every input took effect.)
//!
//! One focal bot plays against `DDAI_T_OPPONENTS` scripted bots on a **private** server (loopback only; never port 8303, which
//! belongs to the shared dev server). The focal bot's datagrams go through a userspace UDP relay that delays both directions by
//! `DDAI_T_DELAY_US` (one way, default 12 500 = 25 ms RTT) plus a uniform `+-DDAI_T_JITTER_US` (default 2 500), order preserved.
//! `DDAI_T_BURN=N` adds N busy threads to the load the machine already has (the report prints the load average before and after).
//!
//! Output: `$DDAI_T_OUT/<label>.json` (everything the bot measured about itself: percentiles of the decision's wall time and of
//! its phases, the slot statistics, the input margin, the driver's send lag) and `$DDAI_T_OUT/<label>.trace.jsonl` (see
//! `ddai_bot::trace`). The default output directory is `~/aiddnet/data/scratch/task-3.11/runs`.
//!
//! Knobs for the fixes of task 3.11 are in `fixes_from_env` below (all off unless named).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ddai_bot::brains::{BrainKind, BrainOptions};
use ddai_bot::runner::{RunReport, RunnerConfig, run};
use ddai_bot::{BotConfig, Mode, Relations};
use ddai_client::ClientConfig;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("aiddnet/data")
}

fn out_dir() -> PathBuf {
    std::env::var_os("DDAI_T_OUT").map_or_else(|| data_dir().join("scratch/task-3.11/runs"), PathBuf::from)
}

/// CPU seconds (user + system) of every thread of this process, by thread id: `(comm, seconds)`, from `/proc/self/task/*/stat`.
fn thread_cpu() -> std::collections::BTreeMap<u32, (String, f64)> {
    let mut out = std::collections::BTreeMap::new();
    let Ok(rd) = std::fs::read_dir("/proc/self/task") else {
        return out;
    };
    for e in rd.flatten() {
        let Some(tid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(e.path().join("stat")) else {
            continue;
        };
        // `pid (comm) state ...`: the comm may contain spaces; fields 14 and 15 (1-based) are utime and stime in clock ticks.
        let (Some(a), Some(b)) = (stat.find('('), stat.rfind(')')) else {
            continue;
        };
        let comm = stat[a + 1..b].to_string();
        let f: Vec<&str> = stat[b + 2..].split_whitespace().collect();
        let (Some(u), Some(s)) = (f.get(11), f.get(12)) else {
            continue;
        };
        let ticks = u.parse::<f64>().unwrap_or(0.0) + s.parse::<f64>().unwrap_or(0.0);
        out.insert(tid, (comm, ticks / 100.0));
    }
    out
}

fn load1() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(f64::NAN)
}

mod relay {
    use std::net::{SocketAddr, UdpSocket};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::{Duration, Instant};

    /// A userspace UDP delay relay between one client and one **loopback** server. Each direction is a reader thread that stamps
    /// a release time on every datagram (`base` + uniform `+-jitter`, never before the datagram ahead of it) and a sender thread
    /// that sleeps until it. Blocking reads and channel waits only: no polling, so the relay adds no timer-granularity jitter of
    /// its own beyond a thread wake-up (tens of microseconds).
    pub struct Relay {
        pub addr: SocketAddr,
        stop: Arc<AtomicBool>,
        handles: Vec<std::thread::JoinHandle<()>>,
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn release_time(rng: &mut u64, now: Instant, base: Duration, jitter: Duration, last: &mut Instant) -> Instant {
        let span = 2 * jitter.as_micros() as u64 + 1;
        let j = (xorshift(rng) % span) as i64 - jitter.as_micros() as i64;
        let d = (base.as_micros() as i64 + j).max(0) as u64;
        let release = (now + Duration::from_micros(d)).max(*last);
        *last = release;
        release
    }

    impl Relay {
        pub fn start(server: SocketAddr, base: Duration, jitter: Duration, seed: u64) -> Relay {
            assert!(
                server.ip().is_loopback(),
                "the relay only ever talks to a loopback server"
            );
            let front = UdpSocket::bind("127.0.0.1:0").expect("bind the relay");
            let back = UdpSocket::bind("127.0.0.1:0").expect("bind the relay's server side");
            back.connect(server).expect("connect to the server");
            for s in [&front, &back] {
                s.set_read_timeout(Some(Duration::from_millis(100))).unwrap();
            }
            let addr = front.local_addr().unwrap();
            let stop = Arc::new(AtomicBool::new(false));
            let client: Arc<Mutex<Option<SocketAddr>>> = Arc::new(Mutex::new(None));
            let (up_tx, up_rx) = mpsc::channel::<(Instant, Vec<u8>)>();
            let (down_tx, down_rx) = mpsc::channel::<(Instant, Vec<u8>)>();
            let mut handles = Vec::new();
            // client -> server: reader
            {
                let (front, stop, client) = (front.try_clone().unwrap(), Arc::clone(&stop), Arc::clone(&client));
                handles.push(std::thread::spawn(move || {
                    let (mut rng, mut last, mut buf) = (seed | 1, Instant::now(), [0u8; 2048]);
                    while !stop.load(Ordering::Relaxed) {
                        if let Ok((n, from)) = front.recv_from(&mut buf) {
                            *client.lock().unwrap() = Some(from);
                            let r = release_time(&mut rng, Instant::now(), base, jitter, &mut last);
                            let _ = up_tx.send((r, buf[..n].to_vec()));
                        }
                    }
                }));
            }
            // client -> server: sender
            {
                let back = back.try_clone().unwrap();
                handles.push(std::thread::spawn(move || {
                    while let Ok((r, d)) = up_rx.recv() {
                        let now = Instant::now();
                        if r > now {
                            std::thread::sleep(r - now);
                        }
                        let _ = back.send(&d);
                    }
                }));
            }
            // server -> client: reader
            {
                let (back, stop) = (back.try_clone().unwrap(), Arc::clone(&stop));
                handles.push(std::thread::spawn(move || {
                    let (mut rng, mut last, mut buf) = (seed.wrapping_mul(7) | 1, Instant::now(), [0u8; 2048]);
                    while !stop.load(Ordering::Relaxed) {
                        if let Ok(n) = back.recv(&mut buf) {
                            let r = release_time(&mut rng, Instant::now(), base, jitter, &mut last);
                            let _ = down_tx.send((r, buf[..n].to_vec()));
                        }
                    }
                }));
            }
            // server -> client: sender
            {
                let (front, client) = (front, Arc::clone(&client));
                handles.push(std::thread::spawn(move || {
                    while let Ok((r, d)) = down_rx.recv() {
                        let now = Instant::now();
                        if r > now {
                            std::thread::sleep(r - now);
                        }
                        if let Some(c) = *client.lock().unwrap() {
                            let _ = front.send_to(&d, c);
                        }
                    }
                }));
            }
            Relay { addr, stop, handles }
        }
    }

    impl Drop for Relay {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            // The readers notice within their 100 ms read timeout, drop the channels, and the senders end.
            for h in self.handles.drain(..) {
                let _ = h.join();
            }
        }
    }
}

/// What the task-3.11 fixes switch on, from `DDAI_T_FIX` (comma separated names; see the arms). Nothing is on by default.
#[derive(Debug, Default, Clone)]
struct Fixes {
    names: Vec<String>,
}

fn fixes_from_env() -> Fixes {
    Fixes {
        names: std::env::var("DDAI_T_FIX")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
    }
}

fn config(
    name: &str,
    server: SocketAddr,
    kind: BrainKind,
    seed: u64,
    duration: Duration,
    trace: Option<PathBuf>,
    fixes: &Fixes,
) -> RunnerConfig {
    let mut client = ClientConfig {
        name: name.to_string(),
        cache_dir: data_dir().join("scratch/task-3.11/mapcache"),
        adaptive_margin: true,
        ..ClientConfig::default()
    };
    let trace_is_focal = trace.is_some();
    let mut bot = BotConfig {
        brain: kind,
        mode: Mode::Fight,
        seed,
        input_trace: trace,
        ..BotConfig::default()
    };
    apply_fixes(fixes, &mut client, &mut bot);
    let wb_auto = trace_is_focal && std::env::var("DDAI_T_WB").as_deref() == Ok("auto");
    // `DDAI_T_PROD=1`: what the production unit also has: the clip recorder with the autoclip, the navigation memory files, the bridge.
    let prod = trace_is_focal && std::env::var("DDAI_T_PROD").as_deref() == Ok("1");
    let scratch = data_dir().join("scratch/task-3.11");
    if prod {
        bot.clips = ddai_bot::clipper::ClipConfig {
            dir: Some(scratch.join("clips")),
            autoclip: true,
            async_save: true,
        };
    }
    RunnerConfig {
        server,
        client,
        bot,
        brain: BrainOptions {
            seed,
            ..BrainOptions::default()
        },
        relations: Relations::new(),
        duration: Some(duration),
        bridge_path: prod.then(|| scratch.join("bridge.sock")),
        web_names: false,
        debug_names_log: None,
        audit_outgoing: false,
        shutdown: Arc::new(AtomicBool::new(false)),
        nav: ddai_bot::nav_hooks::NavConfig {
            memory_dir: prod.then(|| scratch.join("memory")),
            // `DDAI_T_WB=auto` is the production rule (the way-block holds its place and the brain only runs when somebody comes);
            // off (default) keeps the focal bot after its target all the time, so every decision is a brain decision.
            wb_mode: if wb_auto {
                ddai_bot::nav_hooks::WbMode::Auto
            } else {
                ddai_bot::nav_hooks::WbMode::Off
            },
            ..ddai_bot::nav_hooks::NavConfig::default()
        },
        nav_handle: ddai_bot::nav_hooks::NavHandle::new(),
        commands: None,
        console_out: None,
    }
}

fn apply_fixes(fixes: &Fixes, client: &mut ClientConfig, bot: &mut BotConfig) {
    for n in &fixes.names {
        match n.as_str() {
            // The driver loop sleeps on the input channel with a precise timeout instead of blocking in `recv`.
            "precise" => client.precise_wakeups = true,
            // Aim brain decisions by the brain decisions' own time estimate.
            "kind" => bot.kind_estimate = true,
            other => panic!("unknown fix {other}"),
        }
    }
}

fn spawn_bot(cfg: RunnerConfig) -> std::thread::JoinHandle<RunReport> {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || run(cfg).expect("the bot starts"))
        .expect("thread")
}

fn report_json(r: &RunReport, load_before: f64, load_after: f64, label: &str, fixes: &Fixes) -> serde_json::Value {
    let l = &r.latency;
    let s = &l.slots;
    let m = r.margin.clone();
    serde_json::json!({
        "label": label,
        "fixes": fixes.names,
        "elapsed_s": r.elapsed.as_secs_f64(),
        "load1_before": load_before,
        "load1_after": load_after,
        "exit_code": r.exit_code,
        "gave_up": r.gave_up.as_ref().map(|g| g.0.clone()),
        "stats": {
            "snapshots": r.stats.snapshots, "decisions": r.stats.decisions, "brain_decisions": r.stats.brain_decisions,
            "wander_decisions": r.stats.wander_decisions, "idle_decisions": r.stats.idle_decisions,
            "collapsed": r.stats.collapsed, "hooks_fired": r.stats.hooks_fired, "deaths": r.stats.deaths,
        },
        "latency_us": l.json(),
        "slots": {
            "decisions": s.decisions, "first_slot": s.in_first_slot, "missed_first_slot": s.missed_first_slot,
            "missed_first_slot_share": s.missed_first_slot as f64 / s.decisions.max(1) as f64,
            "as_predicted": s.as_predicted, "as_predicted_share": s.as_predicted as f64 / s.decisions.max(1) as f64,
            "later_than_predicted": s.later_than_predicted, "earlier_than_predicted": s.earlier_than_predicted,
        },
        "input_margin": m.map(|m| serde_json::json!({
            "count": m.count, "late": m.late_count, "stalls": m.stall_count, "late_fraction": m.late_fraction,
            "margin_ms": m.margin_ms, "margin_changes": m.margin_changes,
            "time_left_ms": {"min": m.min_ms, "p50": m.p50_ms, "p90": m.p90_ms, "p99": m.p99_ms},
            "superseded_decisions": m.superseded_decisions, "late_presses_dropped": m.late_presses_dropped,
            "send_lag_us": m.send_lag_us.map(|v| serde_json::json!({"n": v.count, "p50": v.p50_us, "p90": v.p90_us, "p99": v.p99_us, "max": v.max_us})),
        })),
    })
}

#[test]
#[ignore = "needs a private DDNet server (DDAI_T_SERVER); DDAI_E2E=1 and --ignored"]
fn live_timing_run() {
    if std::env::var("DDAI_E2E").as_deref() != Ok("1") {
        eprintln!("skipped: set DDAI_E2E=1");
        return;
    }
    let server: SocketAddr = std::env::var("DDAI_T_SERVER")
        .unwrap_or_else(|_| "127.0.0.1:8453".to_string())
        .parse()
        .expect("DDAI_T_SERVER");
    assert!(server.ip().is_loopback(), "private servers only");
    assert_ne!(server.port(), 8303, "8303 is the shared dev server");
    let label = std::env::var("DDAI_T_LABEL").unwrap_or_else(|_| "run".to_string());
    let secs = env_u64("DDAI_T_SECS", 90);
    let delay = Duration::from_micros(env_u64("DDAI_T_DELAY_US", 12_500));
    let jitter = Duration::from_micros(env_u64("DDAI_T_JITTER_US", 2_500));
    let opponents = env_u64("DDAI_T_OPPONENTS", 1);
    let burn = env_u64("DDAI_T_BURN", 0);
    let seed = env_u64("DDAI_T_SEED", 21);
    let brain = std::env::var("DDAI_T_BRAIN").unwrap_or_else(|_| "hybrid".to_string());
    let kind = BrainKind::parse(&brain).unwrap_or_else(|| panic!("unknown brain {brain}"));
    let fixes = fixes_from_env();
    let out = out_dir();
    std::fs::create_dir_all(&out).unwrap();
    let dur = Duration::from_secs(secs);

    let stop_burn = Arc::new(AtomicBool::new(false));
    let burners: Vec<_> = (0..burn)
        .map(|_| {
            let stop = Arc::clone(&stop_burn);
            std::thread::spawn(move || {
                let mut x = 1u64;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..10_000 {
                        x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407));
                    }
                }
                x
            })
        })
        .collect();

    let relay = relay::Relay::start(server, delay, jitter, 0x4b1d ^ seed);
    let load_before = load1();
    // Threads come and go (the focal bot's end before the run's last look): a sampler keeps the last CPU time seen per thread id.
    let cpu_seen = Arc::new(std::sync::Mutex::new(
        std::collections::BTreeMap::<u32, (String, f64)>::new(),
    ));
    let cpu_stop = Arc::new(AtomicBool::new(false));
    let cpu_sampler = {
        let (seen, stop) = (Arc::clone(&cpu_seen), Arc::clone(&cpu_stop));
        std::thread::Builder::new()
            .name("t311-cpu".into())
            .spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    for (tid, v) in thread_cpu() {
                        seen.lock().unwrap().insert(tid, v);
                    }
                    std::thread::sleep(Duration::from_millis(250));
                }
            })
            .unwrap()
    };
    let trace = out.join(format!("{label}.trace.jsonl"));
    let mut focal_cfg = config("t311-focal", relay.addr, kind, seed, dur, Some(trace), &fixes);
    // `DDAI_T_DUTY=F,P`: the focal bot alternates F seconds of `fight` (brain decisions) with P seconds of `passive` (cheap wandering
    // decisions), like the production mix (about a quarter brain decisions): the first brain decisions of every fight follow a
    // wander phase, which is what the decision-time estimate of a mixed window gets wrong.
    let duty_stop = Arc::new(AtomicBool::new(false));
    let duty = std::env::var("DDAI_T_DUTY").ok().and_then(|v| {
        let (f, p) = v.split_once(',')?;
        Some((f.trim().parse::<u64>().ok()?, p.trim().parse::<u64>().ok()?))
    });
    let duty_thread = duty.map(|(fight, passive)| {
        let (sender, inbox) = ddai_bot::command::CommandBus::open();
        focal_cfg.commands = Some(inbox);
        let stop = Arc::clone(&duty_stop);
        std::thread::spawn(move || {
            let to = Duration::from_secs(2);
            let nap = |s: u64| {
                let end = std::time::Instant::now() + Duration::from_secs(s);
                while std::time::Instant::now() < end && !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(100));
                }
            };
            // The first seconds are the join; start with a wander phase.
            while !stop.load(Ordering::Relaxed) {
                let _ = sender.send(ddai_bot::command::BotCommand::Mode(Some(Mode::Passive)), to);
                nap(passive);
                let _ = sender.send(ddai_bot::command::BotCommand::Mode(Some(Mode::Fight)), to);
                nap(fight);
            }
        })
    });
    let focal = spawn_bot(focal_cfg);
    let mut others = Vec::new();
    for i in 0..opponents {
        std::thread::sleep(Duration::from_millis(700));
        others.push(spawn_bot(config(
            &format!("t311-opp{i}"),
            server,
            BrainKind::Scripted,
            150 + i,
            dur,
            None,
            &Fixes::default(),
        )));
    }
    let r = focal.join().expect("the focal bot");
    let load_after = load1();
    duty_stop.store(true, Ordering::Relaxed);
    if let Some(h) = duty_thread {
        let _ = h.join();
    }
    cpu_stop.store(true, Ordering::Relaxed);
    let _ = cpu_sampler.join();
    for h in others {
        let _ = h.join();
    }
    stop_burn.store(true, Ordering::Relaxed);
    for b in burners {
        let _ = b.join();
    }
    drop(relay);
    let mut j = report_json(&r, load_before, load_after, &label, &fixes);
    // Whole-run CPU seconds by thread name (the focal bot's driver thread shares its name with the opponents', which are the same in
    // every variant; the test process also runs the relay and the opponents).
    let mut by_name = std::collections::BTreeMap::<String, f64>::new();
    for (name, secs) in cpu_seen.lock().unwrap().values() {
        *by_name.entry(name.clone()).or_insert(0.0) += secs;
    }
    j["thread_cpu_s"] = serde_json::json!(by_name);
    std::fs::write(
        out.join(format!("{label}.json")),
        serde_json::to_vec_pretty(&j).unwrap(),
    )
    .unwrap();
    eprintln!("{}\n{}", serde_json::to_string_pretty(&j).unwrap(), r.latency_text());
    assert_eq!(r.exit_code, 0, "{:?}", r.gave_up);
}
