//! Task 3.24 / E-039, step 3: the human behaviour study on demos with real inputs (`Sv_PreInput`).
//!
//! `human_study run --out DIR [--threads N] <demo|dir>...` studies every demo (records per demo go to
//! `DIR/study-<sha prefix>.bin`; no names anywhere) and `human_study report DIR` aggregates the
//! records into the tables of `docs/research/human-demos.md` (printed as markdown, also written to
//! `DIR/report.md`). See `ddai_dataset::study` for the definitions.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use ddai_dataset::analysis::{self, Timeline};
use ddai_dataset::config::Config;
use ddai_dataset::demo::process_source_real;
use ddai_dataset::humaninput::true_table;
use ddai_dataset::ingest::FrameSource;
use ddai_dataset::study::{
    self, AfterBlock, DemoStudy, Fight, HookRec, VictimEnd, after_rows, fight_row, hook_cells, summarize_after,
};

#[derive(Serialize, Deserialize)]
struct DemoRecord {
    id: String,
    /// The file name says "a vs b": a duel recording (the name itself is never stored).
    duel_named: bool,
    bytes: u64,
    minutes: f32,
    frames: u64,
    preinput_messages: u64,
    tracks: u64,
    study: DemoStudy,
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
                    if q.is_symlink() {
                        continue;
                    }
                    if q.is_dir() {
                        stack.push(q);
                    } else if q.extension().is_some_and(|x| x == "demo") {
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

fn run_one(path: &Path, idx: u16, out_dir: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    let id: String = Sha256::digest(&bytes)
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect();
    let duel_named = path.file_name().is_some_and(|n| n.to_string_lossy().contains(" vs "));
    let demo = ddai_demo::Demo::parse(&bytes).map_err(|e| format!("{id}: header {e}"))?;
    let map = ddai_map::load_map(demo.map_bytes()).map_err(|_| format!("{id}: no embedded map"))?;
    let map = std::sync::Arc::new(map.data);
    let cfg = Config {
        real_inputs: true,
        min_frame_spacing: 2,
        max_frame_gap: 4,
        ..Config::default()
    };
    let table = true_table(&demo);
    if table.stats.messages == 0 {
        return Ok(format!("{id}: no pre-inputs, skipped"));
    }
    let out = process_source_real(&cfg, &map, &demo, None, Some(&table)).map_err(|e| format!("{id}: {e}"))?;
    let mut src = FrameSource::new(&demo);
    for _ in &mut src {}
    let kills = std::mem::take(&mut src.kills);
    let tl = Timeline::new(&cfg, &out.frames, &kills, &map);
    let entries = analysis::freeze_entries(&tl);
    let hooks = analysis::hook_episodes(&tl);
    let hits = analysis::hammer_hits(&tl);
    let attr = analysis::attribute(&tl, &entries, &hooks, &hits);
    let st = study::study_demo(idx, &tl, &table, &hooks, &hits, &attr);
    let rec = DemoRecord {
        id: id.clone(),
        duel_named,
        bytes: bytes.len() as u64,
        minutes: (out.last_tick - out.first_tick) as f32 / 50.0 / 60.0,
        frames: out.frame_count as u64,
        preinput_messages: table.stats.messages,
        tracks: table.tracks.len() as u64,
        study: st,
    };
    let line = format!(
        "{id}: {:.0} min, {} frames, {} after-block, {} hook episodes, fight2 {} player-frames",
        rec.minutes,
        rec.frames,
        rec.study.after.len(),
        rec.study.hooks.len(),
        rec.study.fight2.player_frames
    );
    // postcard (JSON has no NaN, and the records hold some)
    std::fs::write(
        out_dir.join(format!("study-{id}.bin")),
        postcard::to_stdvec(&rec).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(line)
}

fn load(dir: &Path) -> Vec<DemoRecord> {
    let mut out = Vec::new();
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("report dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("study-") && n.to_string_lossy().ends_with(".bin"))
        })
        .collect();
    names.sort();
    for p in names {
        let rec: DemoRecord = postcard::from_bytes(&std::fs::read(&p).expect("read")).expect("parse");
        out.push(rec);
    }
    out
}

fn report(dir: &Path) -> String {
    use std::fmt::Write;
    let recs = load(dir);
    let mut md = String::new();
    type Group = (&'static str, Box<dyn Fn(&DemoRecord) -> bool>);
    let groups: Vec<Group> = vec![
        ("all demos with pre-inputs", Box::new(|_| true)),
        ("duel-named demos", Box::new(|r| r.duel_named)),
        ("other demos", Box::new(|r| !r.duel_named)),
    ];
    writeln!(md, "## Demos\n").unwrap();
    writeln!(md, "| demo | duel-named | min | frames | pre-inputs | tracks | freeze entries / credited / blocks | hook episodes | frames with a player hooked % | frames with 1 / 2 / 3-4 / 5+ chars % |\n|---|---|---:|---:|---:|---:|---|---:|---:|---|").unwrap();
    for r in &recs {
        let h = &r.study.chars_hist;
        writeln!(
            md,
            "| {} | {} | {:.0} | {} | {} | {} | {} / {} / {} | {} | {:.0} | {:.0} / {:.0} / {:.0} / {:.0} |",
            r.id,
            r.duel_named,
            r.minutes,
            r.frames,
            r.preinput_messages,
            r.tracks,
            r.study.freeze_entries,
            r.study.credited,
            r.study.blocks,
            r.study.hooks.len(),
            100.0 * r.study.hook_frames as f64 / r.frames.max(1) as f64,
            100.0 * h[1] as f64 / r.frames.max(1) as f64,
            100.0 * h[2] as f64 / r.frames.max(1) as f64,
            100.0 * (h[3] + h[4]) as f64 / r.frames.max(1) as f64,
            100.0 * h[5..].iter().sum::<u64>() as f64 / r.frames.max(1) as f64
        )
        .unwrap();
    }
    let head = "| group | credited freezes | blocks | victim killed / thawed / still frozen % | frozen median ticks | hooked the victim % / first hook tick | hook-on-victim share median % | hold median / taps % / holds per freeze | swung % / swings / swung at frozen % / hits | idle % / neutral input % | victim \\|dx\\| / dy at +150 | nearer freeze % | freeze dist start / end px |\n|---|---:|---:|---|---:|---|---:|---|---|---|---|---:|---|";
    writeln!(md, "\n## After the block (3 s after a credited freeze)\n").unwrap();
    writeln!(md, "{head}").unwrap();
    for (gname, pred) in &groups {
        let all: Vec<&AfterBlock> = recs
            .iter()
            .filter(|r| pred(r))
            .flat_map(|r| r.study.after.iter())
            .collect();
        let slices: Vec<(String, Vec<&AfterBlock>)> = vec![
            (format!("{gname}: all credited"), all.clone()),
            (
                format!("{gname}: actor input known"),
                all.iter().copied().filter(|a| a.known_input > 0.5).collect(),
            ),
            (
                format!("{gname}: blocks"),
                all.iter().copied().filter(|a| a.block).collect(),
            ),
            (
                format!("{gname}: blocks, actor input known"),
                all.iter().copied().filter(|a| a.block && a.known_input > 0.5).collect(),
            ),
            (
                format!("{gname}: by hook"),
                all.iter().copied().filter(|a| a.hook).collect(),
            ),
            (
                format!("{gname}: by hammer"),
                all.iter().copied().filter(|a| !a.hook).collect(),
            ),
            (
                format!("{gname}: 2 chars in view"),
                all.iter().copied().filter(|a| a.n_chars == 2).collect(),
            ),
            (
                format!("{gname}: 3-4 chars"),
                all.iter().copied().filter(|a| (3..=4).contains(&a.n_chars)).collect(),
            ),
            (
                format!("{gname}: 5+ chars"),
                all.iter().copied().filter(|a| a.n_chars >= 5).collect(),
            ),
        ];
        for (name, v) in slices {
            if v.is_empty() {
                continue;
            }
            writeln!(md, "{}", after_rows(&name, &summarize_after(&v))).unwrap();
        }
    }

    writeln!(
        md,
        "\n## Hook episodes: height of the hooker over the victim at the start, and what became of them\n"
    )
    .unwrap();
    writeln!(md, "| group | actor | hook length | episodes | froze the victim (credited) | % | blocks | release to freeze median ticks |\n|---|---|---|---:|---:|---:|---:|---:|").unwrap();
    for (gname, pred) in &groups {
        let all: Vec<&HookRec> = recs
            .iter()
            .filter(|r| pred(r))
            .flat_map(|r| r.study.hooks.iter())
            .collect();
        for (min_chars, max_chars, tag) in [(0u8, 255u8, "any"), (2, 2, "2 chars"), (3, 255, "3+ chars")] {
            let v: Vec<&HookRec> = all
                .iter()
                .copied()
                .filter(|h| h.n_chars >= min_chars && h.n_chars <= max_chars)
                .collect();
            if v.is_empty() {
                continue;
            }
            let cells = hook_cells(&v);
            for (hi, hname) in ["above (> 16 px)", "level", "below"].into_iter().enumerate() {
                for (li, lname) in ["< 10 ticks", ">= 10 ticks"].into_iter().enumerate() {
                    let c = &cells[hi][li];
                    writeln!(
                        md,
                        "| {gname} / {tag} | {hname} | {lname} | {} | {} | {:.1} | {} | {} |",
                        c.n,
                        c.froze,
                        100.0 * c.froze as f64 / c.n.max(1) as f64,
                        c.blocks,
                        if c.release_to_freeze_median.is_nan() {
                            "-".into()
                        } else {
                            format!("{:.0}", c.release_to_freeze_median)
                        }
                    )
                    .unwrap();
                }
            }
        }
    }
    // The pattern of the duel post-mortem: hooked from above, released a few ticks before the freeze.
    writeln!(md, "\n### Credited hook freezes: the pattern before the block\n").unwrap();
    writeln!(md, "| group | credited hook freezes | actor above at start % | actor below the victim at the release % | release to freeze median / p25 / p75 ticks | hold median ticks | victim vy at release median | victim jumped in the last 30 ticks % | actor jumped % | victim freeze dist at release median px |\n|---|---:|---:|---:|---|---:|---:|---:|---:|---:|").unwrap();
    for (gname, pred) in &groups {
        let v: Vec<&HookRec> = recs
            .iter()
            .filter(|r| pred(r))
            .flat_map(|r| r.study.hooks.iter())
            .filter(|h| h.froze)
            .collect();
        if v.is_empty() {
            continue;
        }
        let pc = |f: &dyn Fn(&HookRec) -> bool| 100.0 * v.iter().filter(|h| f(h)).count() as f64 / v.len() as f64;
        let rel: Vec<f32> = v.iter().map(|h| h.release_to_freeze as f32).collect();
        let hold: Vec<f32> = v.iter().map(|h| h.ticks as f32).collect();
        let vy: Vec<f32> = v.iter().map(|h| h.victim_vy_end).collect();
        let fd: Vec<f32> = v.iter().map(|h| h.freeze_dist_end).collect();
        writeln!(
            md,
            "| {gname} | {} | {:.0} | {:.0} | {:.0} / {:.0} / {:.0} | {:.0} | {:.1} | {:.0} | {:.0} | {:.0} |",
            v.len(),
            pc(&|h| h.dy_start < -study::ABOVE_PX),
            pc(&|h| h.dy_end > study::ABOVE_PX),
            study::median(&rel),
            study::quantile(&rel, 0.25),
            study::quantile(&rel, 0.75),
            study::median(&hold),
            study::median(&vy),
            pc(&|h| h.victim_jumps > 0),
            pc(&|h| h.actor_jumps > 0),
            study::median(&fd),
        )
        .unwrap();
    }

    writeln!(
        md,
        "\n## Fights (two free players within 400 px), rates per 30 s per player\n"
    )
    .unwrap();
    writeln!(md, "| group | 30 s units of fight per player | hits | swings | fire presses | jumps | eagerness % (swung / ready and close) | hook on the other % of frames | hold p25 / p50 / p75 / p90 ticks | taps % | holds | other above % |\n|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|").unwrap();
    for (gname, pred) in &groups {
        for (which, tag) in [(0, "2 chars"), (1, "<= 4 chars")] {
            let mut f = Fight::default();
            for r in recs.iter().filter(|r| pred(r)) {
                f.merge(if which == 0 { &r.study.fight2 } else { &r.study.fight4 });
            }
            writeln!(md, "{}", fight_row(&format!("{gname} / {tag}"), &f)).unwrap();
        }
    }
    // victim end by group for the quick read
    let mut ends: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    for r in &recs {
        let e = ends
            .entry(if r.duel_named {
                "duel-named".into()
            } else {
                "other".into()
            })
            .or_default();
        for a in &r.study.after {
            e[match a.end {
                VictimEnd::StillFrozen => 0,
                VictimEnd::Thawed => 1,
                VictimEnd::Killed => 2,
                VictimEnd::Vanished => 3,
            }] += 1;
        }
    }
    writeln!(md, "\nVictim end (still frozen / thawed / killed / vanished): {ends:?}").unwrap();
    md
}

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("run") => {
            let mut out_dir = PathBuf::from(".");
            let mut threads = 3usize;
            let mut paths = Vec::new();
            while let Some(a) = args.next() {
                match a.as_str() {
                    "--out" => out_dir = PathBuf::from(args.next().expect("--out DIR")),
                    "--threads" => threads = args.next().and_then(|v| v.parse().ok()).expect("--threads N"),
                    _ => paths.push(PathBuf::from(a)),
                }
            }
            std::fs::create_dir_all(&out_dir).expect("out dir");
            let list = files(&paths);
            let next = AtomicUsize::new(0);
            let log = Mutex::new(());
            std::thread::scope(|s| {
                for _ in 0..threads.clamp(1, 3) {
                    s.spawn(|| {
                        loop {
                            let i = next.fetch_add(1, Ordering::SeqCst);
                            let Some(p) = list.get(i) else { break };
                            let r = run_one(p, i as u16, &out_dir);
                            let _g = log.lock().unwrap();
                            match r {
                                Ok(l) => eprintln!("{l}"),
                                Err(e) => eprintln!("error: {e}"),
                            }
                        }
                    });
                }
            });
        }
        Some("dump") => {
            // debugging aid: the first N after-block records of one demo (anonymous labels and ticks only)
            let dir = PathBuf::from(args.next().expect("dump DIR"));
            let id = args.next().expect("dump DIR ID");
            let n: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(10);
            for r in load(&dir).iter().filter(|r| r.id.starts_with(&id)) {
                for a in r.study.after.iter().take(n) {
                    println!(
                        "t={} L{}->L{} hook={} block={} killed={} frozen_ticks={} end={:?} observed={} dy={:.0} dist0={:.0}",
                        a.tick,
                        a.actor,
                        a.victim,
                        a.hook,
                        a.block,
                        a.killed,
                        a.frozen_ticks,
                        a.end,
                        a.observed,
                        a.dy,
                        a.dist[0]
                    );
                }
            }
        }
        Some("report") => {
            let dir = PathBuf::from(args.next().expect("report DIR"));
            let md = report(&dir);
            std::fs::write(dir.join("report.md"), &md).expect("write report");
            println!("{md}");
        }
        _ => eprintln!("usage: human_study run --out DIR [--threads N] <demo|dir>... | human_study report DIR"),
    }
}
