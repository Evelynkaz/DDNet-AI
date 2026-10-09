//! Task 3.24 / E-039, step 2: physics validation of the real inputs (`Sv_PreInput`) of demos.
//!
//! For every demo the whole interval pipeline is run twice over the same frames: once with the
//! inputs reconstructed from snapshots (the pre-3.24 way) and once with the real inputs wherever a
//! player's `Sv_PreInput` track has one in force. Each interval is replayed with `World<f32>` from
//! the state at the first snapshot and compared with the state at the next one. The tool reports,
//! per demo, how often the replay reproduces the next snapshot exactly (quantised position and
//! velocity identical, same hook), within 1 px or not at all - for the real inputs, and for the
//! reconstructed inputs on *the same* samples - and how the reconstructed action differs from the
//! real one. Demos are named by a sha256 prefix, never by file name.
//!
//! `human_validate [--shift N] [--threads N] [--json out.json] <demo|dir>...`
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::Serialize;
use sha2::{Digest, Sha256};

use ddai_dataset::config::Config;
use ddai_dataset::humaninput::true_table;
use ddai_dataset::ingest::FrameSource;
use ddai_dataset::pipeline::StepInfo;
use ddai_dataset::pipeline::{build_stream_real, recon_table};
use ddai_dataset::types::{ActionRec, CharRec, ReplayClass, char_flags};

#[derive(Default, Serialize, Clone)]
struct Classes {
    n: u64,
    exact: u64,
    within1px: u64,
    off: u64,
    unavailable: u64,
}

impl Classes {
    fn add(&mut self, c: ReplayClass) {
        self.n += 1;
        match c {
            ReplayClass::Exact => self.exact += 1,
            ReplayClass::Within1px => self.within1px += 1,
            ReplayClass::Off => self.off += 1,
            ReplayClass::Unavailable => self.unavailable += 1,
        }
    }
}

/// A sample split by what the sample is: all / active (something happened) / fresh (the wire core of
/// the *next* snapshot was fresh, so the target is a true server state and not an extrapolation) /
/// active and fresh.
#[derive(Default, Serialize, Clone)]
struct Split {
    all: Classes,
    active: Classes,
    fresh: Classes,
    active_fresh: Classes,
}

impl Split {
    fn add(&mut self, c: ReplayClass, active: bool, fresh: bool) {
        self.all.add(c);
        if active {
            self.active.add(c);
        }
        if fresh {
            self.fresh.add(c);
        }
        if active && fresh {
            self.active_fresh.add(c);
        }
    }
}

#[derive(Default, Serialize, Clone)]
struct Agree {
    n: u64,
    direction: u64,
    jump: u64,
    hook: u64,
    fire: u64,
    /// Fire confusion on the real samples: a swing in the real counter and an `attack_tick`
    /// advance in the snapshots in the same interval / only the former / only the latter.
    fire_both: u64,
    fire_real_only: u64,
    fire_snapshot_only: u64,
}

/// Exactness by the age of the newest real input event at the start of the interval (ticks):
/// `<2, <4, <10, <25, <100, more`.
#[derive(Default, Serialize, Clone)]
struct ByAge {
    n: [u64; 6],
    exact: [u64; 6],
    active_n: [u64; 6],
    active_exact: [u64; 6],
}

fn age_bucket(a: i32) -> usize {
    match a {
        ..=1 => 0,
        2..=3 => 1,
        4..=9 => 2,
        10..=24 => 3,
        25..=99 => 4,
        _ => 5,
    }
}

#[derive(Default, Serialize)]
struct DemoReport {
    by_age: ByAge,
    id: String,
    bytes: u64,
    status: String,
    frames: u64,
    preinput_messages: u64,
    labelled_messages: u64,
    tracks: u64,
    out_of_order: u64,
    /// Samples (character, interval) replayed with the real input.
    real: Split,
    /// The same samples replayed with the reconstructed input.
    real_baseline: Split,
    /// Samples without a real input (reconstruction in both runs).
    other: Split,
    /// Real vs reconstructed action on the real samples.
    agree: Agree,
    /// Samples where a swing was expected: fire events by the real counter, and the replayed weapon
    /// fired on exactly the tick of the demo's `attack_tick`.
    fire_events: u64,
    fire_tick_match: u64,
    base_fire_events: u64,
    base_fire_tick_match: u64,
    decode_error: bool,
}

fn files(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut stack = vec![p.clone()];
            while let Some(d) = stack.pop() {
                let Ok(rd) = std::fs::read_dir(&d) else { continue };
                for e in rd.flatten() {
                    let q = e.path();
                    if q.is_dir() && !q.is_symlink() {
                        stack.push(q);
                    } else if q.extension().is_some_and(|x| x == "demo") && !q.is_symlink() {
                        out.push(q);
                    }
                }
            }
        } else {
            out.push(p.clone());
        }
    }
    out.sort();
    out
}

fn one(path: &Path, shift: i32) -> DemoReport {
    let mut rep = DemoReport::default();
    let Ok(bytes) = std::fs::read(path) else {
        rep.status = "unreadable".into();
        return rep;
    };
    rep.bytes = bytes.len() as u64;
    let sha = Sha256::digest(&bytes);
    rep.id = sha.iter().take(6).map(|b| format!("{b:02x}")).collect();
    let Ok(demo) = ddai_demo::Demo::parse(&bytes) else {
        rep.status = "bad header".into();
        return rep;
    };
    let Ok(map) = ddai_map::load_map(demo.map_bytes()) else {
        rep.status = "no embedded map".into();
        return rep;
    };
    let map = std::sync::Arc::new(map.data);
    let cfg = Config {
        min_frame_spacing: 2,
        max_frame_gap: 4,
        ..Config::default()
    };
    let mut table = true_table(&demo);
    table.tick_shift = shift;
    rep.preinput_messages = table.stats.messages;
    rep.labelled_messages = table.stats.labelled;
    rep.tracks = table.tracks.len() as u64;
    rep.out_of_order = table.stats.out_of_order;
    let recon = recon_table(FrameSource::new(&demo).min_spacing(cfg.min_frame_spacing));

    // Run 1: reconstructed inputs only. Keep a compact record of every sample.
    type Rec = (ReplayClass, bool, ActionRec);
    let mut base: Vec<Vec<Option<Rec>>> = Vec::new();
    let s1 = build_stream_real::<std::convert::Infallible>(
        &cfg,
        &map,
        &recon,
        None,
        FrameSource::new(&demo).min_spacing(cfg.min_frame_spacing),
        |fo| {
            base.push(
                fo.steps
                    .iter()
                    .map(|s| s.as_ref().map(|s| (s.replay, s.active, s.action)))
                    .collect(),
            );
            Ok(())
        },
    )
    .unwrap_or_else(|e| match e {});
    // Run 2: real inputs. The steps of frame k concern the interval to frame k + 1, whose wire-core
    // freshness says whether the target state is a true server state.
    let mut k = 0usize;
    type Pending = (i32, usize, Vec<Option<StepInfo>>, Vec<CharRec>);
    let mut pending: Option<Pending> = None;
    let settle = |rep: &mut DemoReport,
                  tick: i32,
                  base: &[Vec<Option<Rec>>],
                  idx: usize,
                  steps: &[Option<StepInfo>],
                  chars: &[CharRec],
                  next: &[CharRec]| {
        let b = &base[idx];
        for (slot, st) in steps.iter().enumerate() {
            let (Some(st), Some(Some(bs))) = (st, b.get(slot)) else {
                continue;
            };
            let id = chars[slot].id;
            let fresh = next
                .iter()
                .find(|c| c.id == id)
                .is_some_and(|c| c.has(char_flags::FRESH));
            if st.real {
                if let Some(e) = table.track(chars[slot].player).and_then(|t| t.at(tick + 1)) {
                    let b = age_bucket(tick + 1 - e.intended);
                    rep.by_age.n[b] += 1;
                    rep.by_age.exact[b] += u64::from(st.replay == ReplayClass::Exact);
                    if st.active {
                        rep.by_age.active_n[b] += 1;
                        rep.by_age.active_exact[b] += u64::from(st.replay == ReplayClass::Exact);
                    }
                }
                rep.real.add(st.replay, st.active, fresh);
                rep.real_baseline.add(bs.0, bs.1, fresh);
                let (a, r) = (&bs.2, &st.action);
                rep.agree.n += 1;
                rep.agree.direction += u64::from(a.direction == r.direction);
                rep.agree.jump += u64::from(a.jump == r.jump);
                rep.agree.hook += u64::from(a.hook == r.hook);
                rep.agree.fire += u64::from(a.fire == r.fire);
                match (r.fire, a.fire) {
                    (true, true) => rep.agree.fire_both += 1,
                    (true, false) => rep.agree.fire_real_only += 1,
                    (false, true) => rep.agree.fire_snapshot_only += 1,
                    _ => {}
                }
            } else {
                rep.other.add(st.replay, st.active, fresh);
            }
        }
    };
    let s2 = build_stream_real::<std::convert::Infallible>(
        &cfg,
        &map,
        &recon,
        Some(&table),
        FrameSource::new(&demo).min_spacing(cfg.min_frame_spacing),
        |fo| {
            if let Some((tick, idx, steps, chars)) = pending.take() {
                settle(&mut rep, tick, &base, idx, &steps, &chars, &fo.frame.chars);
            }
            pending = Some((fo.frame.tick, k, fo.steps, fo.frame.chars));
            k += 1;
            Ok(())
        },
    )
    .unwrap_or_else(|e| match e {});
    rep.frames = s2.frames as u64;
    for r in s2.replay.values() {
        rep.fire_events += r.fire_events;
        rep.fire_tick_match += r.fire_tick_match;
    }
    for r in s1.replay.values() {
        rep.base_fire_events += r.fire_events;
        rep.base_fire_tick_match += r.fire_tick_match;
    }
    rep.status = "ok".into();
    rep
}

fn pct(a: u64, n: u64) -> String {
    if n == 0 {
        "-".into()
    } else {
        format!("{:.1}", 100.0 * a as f64 / n as f64)
    }
}

fn main() {
    let mut shift = 0;
    let mut threads = 3usize;
    let mut json: Option<PathBuf> = None;
    let mut paths = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--shift" => shift = args.next().and_then(|v| v.parse().ok()).expect("--shift N"),
            "--threads" => threads = args.next().and_then(|v| v.parse().ok()).expect("--threads N"),
            "--json" => json = args.next().map(PathBuf::from),
            _ => paths.push(PathBuf::from(a)),
        }
    }
    let list = files(&paths);
    let next = AtomicUsize::new(0);
    let reports: Mutex<BTreeMap<usize, DemoReport>> = Mutex::new(BTreeMap::new());
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, 3) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(p) = list.get(i) else { break };
                    let r = one(p, shift);
                    eprintln!("done {} ({})", r.id, r.status);
                    reports.lock().unwrap().insert(i, r);
                }
            });
        }
    });
    let reports: Vec<DemoReport> = reports.into_inner().unwrap().into_values().collect();
    println!(
        "shift={shift}  (exact% / within1px% / off% of real samples; baseline = reconstructed input on the same samples)"
    );
    for r in &reports {
        println!(
            "{} {} frames={} msgs={} tracks={} real_samples={} | real: exact {} w1 {} off {} | base: exact {} w1 {} off {} | fresh-target real: exact {} (n={}) base {} | active+fresh real: exact {} (n={}) base {} | agree dir {} jump {} hook {} fire {}",
            r.id,
            r.status,
            r.frames,
            r.preinput_messages,
            r.tracks,
            r.real.all.n,
            pct(r.real.all.exact, r.real.all.n),
            pct(r.real.all.within1px, r.real.all.n),
            pct(r.real.all.off, r.real.all.n),
            pct(r.real_baseline.all.exact, r.real_baseline.all.n),
            pct(r.real_baseline.all.within1px, r.real_baseline.all.n),
            pct(r.real_baseline.all.off, r.real_baseline.all.n),
            pct(r.real.fresh.exact, r.real.fresh.n),
            r.real.fresh.n,
            pct(r.real_baseline.fresh.exact, r.real_baseline.fresh.n),
            pct(r.real.active_fresh.exact, r.real.active_fresh.n),
            r.real.active_fresh.n,
            pct(r.real_baseline.active_fresh.exact, r.real_baseline.active_fresh.n),
            pct(r.agree.direction, r.agree.n),
            pct(r.agree.jump, r.agree.n),
            pct(r.agree.hook, r.agree.n),
            pct(r.agree.fire, r.agree.n),
        );
    }
    if let Some(j) = json {
        std::fs::write(&j, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
    }
}
