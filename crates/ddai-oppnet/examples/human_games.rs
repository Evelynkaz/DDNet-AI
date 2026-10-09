//! Task 3.24 (E-039): DDNet demos -> [`HumanGame`]s, the training and evaluation data of the opponent-input predictor with the players' **real** inputs.
//!
//! For every demo with `Sv_PreInput` messages the worlds are rebuilt through `LiveWorld` exactly as the live bot builds them (same views as the dataset
//! pipeline, snapshots thinned to two ticks), and for every character and its nearest other character the pair `[us, opponent]` is kept as long as the live
//! regime gate admits it (a duel: the target within 480 px, nobody else within 1000 px of either). Both humans play both roles. The real inputs of both
//! come from the demo's pre-input table tick by tick. No names: games are tagged with the demo number and anonymous per-demo labels.
//!
//! ```text
//! cargo run --release -p ddai-oppnet --example human_games -- --out DIR [--min-frames 8] [--threads 3] <demo|dir>...
//! ```
//! Writes `DIR/h<demo number>.humangames` (one file per demo, in the order of the arguments; the number is the split unit) and prints one line per demo.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ddai_dataset::config::Config;
use ddai_dataset::humaninput::{InputEvent, Track, TrueTable, true_table};
use ddai_dataset::ingest::{FrameSource, LabelTracker};
use ddai_dataset::pipeline::ViewBuilder;
use ddai_oppnet::blob::write_blob;
use ddai_oppnet::clipdata::ClipTick;
use ddai_oppnet::frame::{InputRec, N_RAYS, TeeFrame, rays};
use ddai_oppnet::humandata::HumanGame;
use ddai_oppnet::live::RegimeGate;
use ddai_recorder::format::Frame;
use ddai_world::{LiveWorld, SnapshotInput};

fn rec_of(e: &InputEvent) -> InputRec {
    InputRec {
        direction: e.direction,
        jump: e.jump,
        hook: e.hook,
        fire: e.fire,
        target_x: e.aim[0].clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
        target_y: e.aim[1].clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
    }
}

fn input_of(table: &TrueTable, label: u16, tick: i32) -> Option<InputRec> {
    table.track(label).and_then(|t: &Track| t.at(tick)).map(rec_of)
}

/// A message of `label` is applied at tick `t` (its input changed there; the aim is fresh only then).
fn is_msg(table: &TrueTable, label: u16, t: i32) -> bool {
    table
        .track(label)
        .is_some_and(|tr| tr.events.binary_search_by_key(&t, |e| e.intended).is_ok())
}

#[derive(Default)]
struct Run {
    ticks: Vec<ClipTick>,
}

#[derive(Default)]
struct Stats {
    frames: u64,
    pair_frames: u64,
    duel_frames: u64,
    games: usize,
    game_frames: u64,
    known_both: u64,
}

/// What `finish` needs of the demo besides the run itself.
struct Demo<'a> {
    no: u8,
    table: &'a TrueTable,
    min_frames: usize,
}

fn finish(d: &Demo<'_>, run: Run, key: (u16, u16), part: usize, out: &mut Vec<HumanGame>, st: &mut Stats) {
    let (demo_no, table, min_frames) = (d.no, d.table, d.min_frames);
    if run.ticks.len() < min_frames {
        return;
    }
    let (t_first, t_last) = (run.ticks[0].tick, run.ticks.last().map_or(0, |t| t.tick));
    let first = t_first - 3;
    let inputs: Vec<[Option<InputRec>; 2]> = (first..=t_last + 6)
        .map(|t| [input_of(table, key.0, t), input_of(table, key.1, t)])
        .collect();
    let msg: Vec<[bool; 2]> = (first..=t_last + 6)
        .map(|t| [is_msg(table, key.0, t), is_msg(table, key.1, t)])
        .collect();
    st.known_both += inputs.iter().filter(|i| i[0].is_some() && i[1].is_some()).count() as u64;
    let mut ticks = run.ticks;
    for t in &mut ticks {
        t.sent = [input_of(table, key.0, t.tick - 1), input_of(table, key.0, t.tick)];
    }
    st.games += 1;
    st.game_frames += ticks.len() as u64;
    out.push(HumanGame {
        source: format!("h{demo_no}-{}-{}p{part}", key.0, key.1),
        session: demo_no,
        ticks,
        first,
        inputs,
        msg,
    });
}

fn convert(path: &Path, demo_no: u8, min_frames: usize) -> Result<(Vec<HumanGame>, Stats, String), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    let id: String = {
        use sha2::{Digest, Sha256};
        Sha256::digest(&bytes)
            .iter()
            .take(6)
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    let demo = ddai_demo::Demo::parse(&bytes).map_err(|e| format!("{id}: header {e}"))?;
    let map = ddai_map::load_map(demo.map_bytes()).map_err(|_| format!("{id}: no embedded map"))?;
    let map = Arc::new(map.data);
    let table = true_table(&demo);
    if table.stats.messages == 0 {
        return Err(format!("{id}: no pre-inputs"));
    }
    let cfg = Config {
        min_frame_spacing: 2,
        max_frame_gap: 4,
        ..Config::default()
    };
    let dm = Demo {
        no: demo_no,
        table: &table,
        min_frames,
    };
    let gate = RegimeGate::default();
    let mut live = LiveWorld::new(Arc::clone(&map), -1, 0);
    let mut tracker = LabelTracker::new();
    let mut vb = ViewBuilder::default();
    let mut runs: BTreeMap<(u16, u16), Run> = BTreeMap::new();
    let mut parts: BTreeMap<(u16, u16), usize> = BTreeMap::new();
    let mut games = Vec::new();
    let mut st = Stats::default();
    let src = FrameSource::new(&demo).min_spacing(cfg.min_frame_spacing);
    for (frame, tune) in src {
        let Frame::Snapshot { tick, characters, .. } = &frame else {
            continue;
        };
        let tick = *tick;
        let row = tracker.push(&frame);
        let views = vb.views(&cfg, tick, characters, &row);
        live.on_snapshot(SnapshotInput::new(tick, &views, tune));
        let w = live.base_world();
        st.frames += 1;
        let mut active: HashSet<(u16, u16)> = HashSet::new();
        for (ai, a) in characters.iter().enumerate() {
            let la = row[ai].1;
            // the nearest other character in the world
            let Some(core_a) = w.cores.get(a.id as u8) else {
                continue;
            };
            let nearest = characters
                .iter()
                .enumerate()
                .filter(|&(bi, _)| bi != ai)
                .filter_map(|(bi, b)| {
                    w.cores
                        .get(b.id as u8)
                        .map(|c| (bi, b.id, (c.pos.x - core_a.pos.x).hypot(c.pos.y - core_a.pos.y)))
                })
                .min_by(|x, y| x.2.total_cmp(&y.2));
            let Some((bi, bid, _)) = nearest else { continue };
            let lb = row[bi].1;
            let (Some(me), Some(op)) = (TeeFrame::from_world(w, a.id, bid), TeeFrame::from_world(w, bid, a.id)) else {
                continue;
            };
            st.pair_frames += 1;
            let duel = me.alive && op.alive && gate.admits(w, a.id, bid, me.pos, op.pos);
            if !duel {
                continue;
            }
            st.duel_frames += 1;
            let key = (la, lb);
            active.insert(key);
            // a break in the run: the previous frame of this pair was not two ticks before
            if runs
                .get(&key)
                .and_then(|r| r.ticks.last())
                .is_some_and(|t| t.tick != tick - 2)
                && let Some(r) = runs.remove(&key)
            {
                let part = parts.entry(key).or_default();
                *part += 1;
                finish(&dm, r, key, *part, &mut games, &mut st);
            }
            let (mut r_us, mut r_opp) = ([1.0f32; N_RAYS], [1.0f32; N_RAYS]);
            rays(w, me.pos, &mut r_us);
            rays(w, op.pos, &mut r_opp);
            runs.entry(key).or_default().ticks.push(ClipTick {
                tick,
                frames: [me, op],
                rays: [r_us, r_opp],
                sent: [None, None],
                opp_attack_tick: 0,
                opp_weapon: characters[bi].character.weapon.clamp(-1, 8) as i8,
                duel: true,
            });
        }
        // runs whose pair was not a duel in this frame end here
        let ended: Vec<(u16, u16)> = runs.keys().filter(|k| !active.contains(k)).copied().collect();
        for key in ended {
            if let Some(r) = runs.remove(&key) {
                let part = parts.entry(key).or_default();
                *part += 1;
                finish(&dm, r, key, *part, &mut games, &mut st);
            }
        }
    }
    let rest: Vec<(u16, u16)> = runs.keys().copied().collect();
    for key in rest {
        if let Some(r) = runs.remove(&key) {
            let part = parts.entry(key).or_default();
            *part += 1;
            finish(&dm, r, key, *part, &mut games, &mut st);
        }
    }
    Ok((games, st, id))
}

fn main() -> Result<(), String> {
    let (mut out, mut min_frames, mut threads) = (PathBuf::new(), 8usize, 3usize);
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        match k.as_str() {
            "--out" => out = PathBuf::from(it.next().ok_or("--out needs a value")?),
            "--min-frames" => min_frames = it.next().ok_or("--min-frames N")?.parse().map_err(|e| format!("{e}"))?,
            "--threads" => threads = it.next().ok_or("--threads N")?.parse().map_err(|e| format!("{e}"))?,
            other => paths.push(PathBuf::from(other)),
        }
    }
    if out.as_os_str().is_empty() || paths.is_empty() {
        return Err("usage: human_games --out DIR [--min-frames N] [--threads N] <demo>...".into());
    }
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let next = AtomicUsize::new(0);
    let lines = std::sync::Mutex::new(BTreeMap::<usize, String>::new());
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, 3) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(p) = paths.get(i) else { break };
                    let line = match convert(p, i as u8, min_frames) {
                        Ok((games, st, id)) => {
                            let file = out.join(format!("h{i}.humangames"));
                            match write_blob(&file, &games, 3) {
                                Ok(()) => format!(
                                    "h{i} = {id}: {} frames, {} pair-frames, {} duel pair-frames, {} games, {} game frames, real inputs of both for {} ticks",
                                    st.frames, st.pair_frames, st.duel_frames, st.games, st.game_frames, st.known_both
                                ),
                                Err(e) => format!("h{i}: {e}"),
                            }
                        }
                        Err(e) => format!("h{i}: skipped: {e}"),
                    };
                    lines.lock().unwrap().insert(i, line);
                }
            });
        }
    });
    for l in lines.into_inner().unwrap().values() {
        println!("{l}");
    }
    Ok(())
}
