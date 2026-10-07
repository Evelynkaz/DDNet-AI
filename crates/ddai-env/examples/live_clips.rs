//! Task 3.17 (D-111): the live window pipeline (`LiveOpp`, default guard) over a directory of live clips: what the guard does per clip and what the
//! model costs against hold with and without the regime gate. Read-only. Window 3 (`--lag`).
//!
//! ```text
//! cargo run --release -p ddai-env --example live_clips -- --clips ~/aiddnet/data/bot/clips --model m1.oppnet [--lag 3] [--map-contains JoniTee]
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_clip::format::Clip;
use ddai_oppnet::OppPredictor;
use ddai_oppnet::live::analyze::Report;
use ddai_oppnet::live::guard::{GuardConfig, GuardState};
use ddai_oppnet::live::{LiveOpp, Pair, RegimeGate};
use ddai_physics::core::PlayerInput as Wire;
use ddai_world::{LiveWorld, player_input_from_net};

fn find_map(dir: &Path, sha: &[u8; 32]) -> Option<PathBuf> {
    let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .find(|e| e.file_name().to_string_lossy().ends_with(&format!("{hex}.map")))
        .map(|e| e.path())
}

struct Run {
    report: Report,
    guard: String,
    fell_back: bool,
    windows: u64,
    skipped: (u64, u64, u64),
}

fn run(clip: &Clip, map: Arc<ddai_physics::map::MapData>, model: &Path, lag: usize, gate: RegimeGate) -> Run {
    let mut l = LiveOpp::new(
        OppPredictor::load(model).expect("model"),
        GuardConfig::default(),
        [0; 32],
    )
    .unwrap()
    .with_gate(gate);
    let own = clip.header.own_id;
    let mut lw = LiveWorld::new(map, own, clip.header.world_seed);
    let mut victim = Vec::new();
    let mut fell_back = false;
    let mut windows = 0u64;
    for (i, f) in clip.frames.iter().enumerate() {
        ddai_clip::replay::feed(&mut lw, clip, f);
        if !f.own_alive {
            continue;
        }
        // The target the bot would have: the nearest other tee.
        let Some(me) = f.tee(own) else { continue };
        let Some(opp) = f
            .tees
            .iter()
            .filter(|t| t.id != own)
            .min_by_key(|t| i64::from(t.ch.x - me.ch.x).pow(2) + i64::from(t.ch.y - me.ch.y).pow(2))
            .map(|t| t.id)
        else {
            continue;
        };
        let tag = format!("c{opp}-clip");
        let sent_at = |t: i32| {
            clip.frames[i + 1..(i + 3).min(clip.frames.len())]
                .iter()
                .flat_map(|f| f.sent.iter())
                .find(|s| s.tick == t)
                .map(|s| player_input_from_net(s.input.to_net()))
        };
        let pair = Pair {
            world: lw.base_world(),
            self_id: own,
            target: opp,
            tag: &tag,
        };
        let ours: Option<Vec<Wire>> = (1..=lag as i32).map(|k| sent_at(f.tick + k)).collect();
        match ours {
            Some(o) => {
                windows += 1;
                let hold = lw.held_input_of(opp).unwrap_or_default();
                l.window(&pair, &o, &hold, &mut victim);
            }
            None => l.observe(&pair),
        }
        fell_back |= l.guard_status().state == GuardState::Fallback;
    }
    let mut report = Report::new();
    for line in String::from_utf8(l.take_log()).unwrap().lines() {
        report.add_line(line);
    }
    let g = l.guard_status();
    let c = l.counts();
    Run {
        report,
        guard: format!(
            "{:?} model {:.3} hold {:.3} ratio {:.3}",
            g.state,
            g.model_cost,
            g.hold_cost,
            g.model_cost / g.hold_cost.max(1e-9)
        ),
        fell_back,
        windows,
        skipped: (c.skipped_own_frozen, c.skipped_frozen, c.skipped_regime),
    }
}

fn gate_from_env() -> RegimeGate {
    let get = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let d = RegimeGate::default();
    RegimeGate {
        enabled: true,
        max_target_dist_px: get("GATE_DIST", d.max_target_dist_px),
        others_radius_px: get("GATE_RADIUS", d.others_radius_px),
        max_others: get("GATE_OTHERS", d.max_others as f32) as u32,
    }
}

fn main() {
    let (mut clips, mut model, mut lag, mut contains) =
        (PathBuf::new(), PathBuf::new(), 3usize, String::from("JoniTee"));
    let maps = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("aiddnet/data/maps/cache");
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().expect("a value");
        match k.as_str() {
            "--clips" => clips = PathBuf::from(v),
            "--model" => model = PathBuf::from(v),
            "--lag" => lag = v.parse().unwrap(),
            "--map-contains" => contains = v,
            other => panic!("unknown {other}"),
        }
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&clips)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "clip"))
        .collect();
    files.sort();
    let (mut tot_off, mut tot_on) = (Report::new(), Report::new());
    let _ = (&mut tot_off, &mut tot_on);
    let mut fell = [0u32; 2];
    let mut n_duel = 0;
    for f in &files {
        let clip = Clip::read(f).unwrap();
        if !clip.header.map_name.contains(&contains) {
            continue;
        }
        let two = clip
            .frames
            .iter()
            .filter(|f| f.tees.len() == 2 && f.tees_dropped == 0)
            .count();
        let duel = two >= 300;
        let Some(mp) = find_map(&maps, &clip.header.map_sha256) else {
            continue;
        };
        let map = Arc::new(ddai_map::load_map(&std::fs::read(mp).unwrap()).unwrap().data);
        let off = run(&clip, Arc::clone(&map), &model, lag, RegimeGate::off());
        let on = run(&clip, map, &model, lag, gate_from_env());
        let k1 = |r: &Run| r.report.pooled(None, Some(1));
        let (a, b) = (k1(&off), k1(&on));
        println!(
            "{} duel={duel} two-tee frames {two}\n   no gate : windows {} samples {} | guard {} | fell back {} | k=1 dir {:.1} -> {:.1}\n   gate    : samples {} | guard {} | fell back {} | skipped own-frozen/opp-frozen/regime {:?} | k=1 dir {:.1} -> {:.1}",
            f.file_name().unwrap().to_string_lossy(),
            off.windows,
            off.report.samples(),
            off.guard,
            off.fell_back,
            100.0 * a.dir_hold as f64 / a.n.max(1) as f64,
            100.0 * a.dir_model as f64 / a.n.max(1) as f64,
            on.report.samples(),
            on.guard,
            on.fell_back,
            on.skipped,
            100.0 * b.dir_hold as f64 / b.n.max(1) as f64,
            100.0 * b.dir_model as f64 / b.n.max(1) as f64,
        );
        if duel {
            n_duel += 1;
            fell[0] += u32::from(off.fell_back);
            fell[1] += u32::from(on.fell_back);
        }
    }
    println!(
        "\nduel clips: {n_duel}; the guard fell back on {} without the gate, {} with it",
        fell[0], fell[1]
    );
}
