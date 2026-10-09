//! Task 3.24 (E-039), the imitation pilot, step 1: demos -> behaviour-cloning samples of what a human does after freezing an opponent.
//!
//! For every credited freeze of a demo (the actor's hook or hammer touched the victim within the attribution window before it froze, D-030) the frames from the
//! freeze until the victim thaws (at most 3 s) in which the actor is free and the victim frozen are samples: the head's input is built from the worlds
//! `LiveWorld` rebuilds out of the snapshots (as the live bot builds them), the target is the actor's **real** input of the next two ticks (the demo's
//! `Sv_PreInput` messages). Windows where the real input of the actor is not known are skipped. No names.
//!
//! ```text
//! cargo run --release -p ddai-oppnet --example human_bc -- --out DIR [--threads 3] <demo>...
//! ```
//! Writes `DIR/b<demo number>.bcsamples` (one file per demo; the number is the split unit) and prints one line per demo.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ddai_dataset::analysis::{self, Timeline};
use ddai_dataset::config::Config;
use ddai_dataset::demo::process_source_real;
use ddai_dataset::humaninput::true_table;
use ddai_dataset::ingest::{FrameSource, LabelTracker};
use ddai_dataset::pipeline::ViewBuilder;
use ddai_oppnet::bc::{BC_IN, BcLabel, BcSample, aim_rel, features};
use ddai_oppnet::blob::write_blob;
use ddai_oppnet::frame::{InputRec, N_RAYS, TeeFrame, rays};
use ddai_physics::world::count_input_presses;
use ddai_recorder::format::Frame;
use ddai_world::{LiveWorld, SnapshotInput};

/// The window after a credited freeze: 3 s.
const WINDOW: i32 = 150;

struct Window {
    actor: u16,
    victim: u16,
    from: i32,
    to: i32,
}

fn rec_of(e: &ddai_dataset::humaninput::InputEvent) -> InputRec {
    InputRec {
        direction: e.direction,
        jump: e.jump,
        hook: e.hook,
        fire: e.fire,
        target_x: e.aim[0].clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
        target_y: e.aim[1].clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
    }
}

fn convert(path: &Path, demo_no: u8) -> Result<(Vec<BcSample>, String), String> {
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
    let cfg = Config {
        real_inputs: true,
        min_frame_spacing: 2,
        max_frame_gap: 4,
        ..Config::default()
    };
    let table = true_table(&demo);
    if table.stats.messages == 0 {
        return Err(format!("{id}: no pre-inputs"));
    }
    // 1. the credited freezes
    let out = process_source_real(&cfg, &map, &demo, None, Some(&table)).map_err(|e| format!("{id}: {e}"))?;
    let mut src = FrameSource::new(&demo);
    for _ in &mut src {}
    let kills = std::mem::take(&mut src.kills);
    let tl = Timeline::new(&cfg, &out.frames, &kills, &map);
    let entries = analysis::freeze_entries(&tl);
    let hooks = analysis::hook_episodes(&tl);
    let hits = analysis::hammer_hits(&tl);
    let attr = analysis::attribute(&tl, &entries, &hooks, &hits);
    let mut windows: Vec<Window> = attr
        .iter()
        .filter_map(|a| {
            let (actor, _) = a.toucher?;
            let t0 = a.entry.tick;
            let thaw = a.entry.exit_k.map_or(i32::MAX, |k| tl.tick(k));
            Some(Window {
                actor,
                victim: a.entry.player,
                from: t0,
                to: (t0 + WINDOW).min(thaw),
            })
        })
        .collect();
    windows.sort_by_key(|w| w.from);
    drop(tl);
    // 2. the worlds, and the samples inside the windows
    let mut live = LiveWorld::new(Arc::clone(&map), -1, 0);
    let mut tracker = LabelTracker::new();
    let mut vb = ViewBuilder::default();
    let mut samples: Vec<BcSample> = Vec::new();
    let mut skipped_unknown = 0u64;
    let src = FrameSource::new(&demo).min_spacing(cfg.min_frame_spacing);
    let mut next_w = 0usize;
    let mut active: Vec<usize> = Vec::new();
    for (frame, tune) in src {
        let Frame::Snapshot { tick, characters, .. } = &frame else {
            continue;
        };
        let tick = *tick;
        let row = tracker.push(&frame);
        while next_w < windows.len() && windows[next_w].from <= tick {
            active.push(next_w);
            next_w += 1;
        }
        active.retain(|&i| windows[i].to >= tick);
        let views = vb.views(&cfg, tick, characters, &row);
        live.on_snapshot(SnapshotInput::new(tick, &views, tune));
        if active.is_empty() {
            continue;
        }
        let w = live.base_world();
        for &wi in &active {
            let win = &windows[wi];
            let find = |label: u16| row.iter().position(|&(_, l)| l == label);
            let (Some(ai), Some(vi)) = (find(win.actor), find(win.victim)) else {
                continue;
            };
            let (aid, vid) = (characters[ai].id, characters[vi].id);
            let (Some(me), Some(victim)) = (TeeFrame::from_world(w, aid, vid), TeeFrame::from_world(w, vid, aid))
            else {
                continue;
            };
            if !me.alive || me.freeze_left > 0 || !victim.alive || victim.freeze_left == 0 {
                continue;
            }
            let Some(track) = table.track(win.actor) else { continue };
            let (Some(p), Some(e1), Some(e2)) = (track.at(tick), track.at(tick + 1), track.at(tick + 2)) else {
                skipped_unknown += 1;
                continue;
            };
            let hammer = characters[ai].character.weapon == 0;
            let fire = hammer && count_input_presses(p.fire, e2.fire) != 0;
            let (mut r_me, mut r_victim) = ([1.0f32; N_RAYS], [1.0f32; N_RAYS]);
            rays(w, me.pos, &mut r_me);
            rays(w, victim.pos, &mut r_victim);
            let mut x = [0.0f32; BC_IN];
            features(&me, &victim, &r_me, &r_victim, &rec_of(p), &mut x);
            let last = rec_of(e2);
            samples.push(BcSample {
                session: demo_no,
                x: x.to_vec(),
                y: BcLabel {
                    dir: (i32::from(e2.direction) + 1).clamp(0, 2) as u8,
                    jump: e1.jump || e2.jump,
                    hook: e2.hook,
                    fire,
                    aim_rel: aim_rel(&last, &me, &victim),
                },
            });
        }
    }
    let line = format!(
        "{id}: {} credited freezes, {} samples ({} skipped: real input unknown)",
        windows.len(),
        samples.len(),
        skipped_unknown
    );
    Ok((samples, line))
}

fn main() -> Result<(), String> {
    let (mut out, mut threads) = (PathBuf::new(), 3usize);
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        match k.as_str() {
            "--out" => out = PathBuf::from(it.next().ok_or("--out needs a value")?),
            "--threads" => threads = it.next().ok_or("--threads N")?.parse().map_err(|e| format!("{e}"))?,
            other => paths.push(PathBuf::from(other)),
        }
    }
    if out.as_os_str().is_empty() || paths.is_empty() {
        return Err("usage: human_bc --out DIR [--threads N] <demo>...".into());
    }
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let next = AtomicUsize::new(0);
    let lines = std::sync::Mutex::new(std::collections::BTreeMap::<usize, String>::new());
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, 3) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(p) = paths.get(i) else { break };
                    let line = match convert(p, i as u8) {
                        Ok((samples, l)) => match write_blob(&out.join(format!("b{i}.bcsamples")), &samples, 3) {
                            Ok(()) => format!("b{i} = {l}"),
                            Err(e) => format!("b{i}: {e}"),
                        },
                        Err(e) => format!("b{i}: skipped: {e}"),
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
