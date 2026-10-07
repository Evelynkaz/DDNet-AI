//! Task 3.15 (E-028): the offline sanity check of an opponent-input model on **live clips** (`ddai-clip`; read-only copies).
//!
//! A clip is the live bot's 30 s ring: snapshot frames every 2 ticks with the wire state of the tees and the inputs we sent. The opponent's raw inputs are not
//! in it, so the check uses what the snapshots show. Every frame is fed to a `LiveWorld` as the bot did; at every frame `T` (where the next four frames follow
//! at 2-tick steps) the model is asked for the window, and
//!
//! * the **direction** the next snapshots show (the direction of the last applied step: ticks `T + 1, T + 3, T + 5, T + 7`) is compared with the model's
//!   prediction and with the snapshot's own ("hold");
//! * the exact physics is rolled from the reconstructed world at `T` with our sent inputs and the opponent holding what the snapshot shows (`snap`) or playing
//!   the model's inputs (`model`), and the opponent's position after 2, 4, 6, 8 ticks is compared with the reconstructed position at that frame.
//!
//! The window length the model is asked for is `--lag` (default 3): the live bot's direct-connection lag.
//!
//! ```text
//! cargo run --release -p ddai-env --example opp_clips -- --clips ~/aiddnet/data/scratch/t315/clips --model m1.oppnet [--lag 3] [--map-contains JoniTee]
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_clip::format::Clip;
use ddai_clip::replay::feed;
use ddai_oppnet::OppPredictor;
use ddai_physics::core::PlayerInput as Wire;
use ddai_physics::map::MapData;
use ddai_planner::brains::enemy_input_from_tee;
use ddai_planner::hybrid::window::{PredictedInput, WindowCtx, WindowModel, input_from_prediction};
use ddai_planner::physics_adapter::{PhysicsWorld, from_ddnet_input};
use ddai_planner::plan_world::PlanWorld;
use ddai_world::{LiveWorld, player_input_from_net};

const K: usize = 8;

#[derive(Default)]
struct Acc {
    /// Direction hits at `k = 1, 3, 5, 7` for hold and model, and the count.
    dir_n: [u64; 4],
    dir_hold: [u64; 4],
    dir_model: [u64; 4],
    /// Position errors after 2, 4, 6, 8 ticks: `[snap, model][m]`.
    err: [[Vec<f32>; 4]; 2],
    /// Sum and count of the per-frame features of the pair, to compare with the arena's (`opp_train --feat-means`).
    feat_sum: Vec<f64>,
    feat_n: u64,
}

impl Acc {
    fn merge(&mut self, o: &Acc) {
        for m in 0..4 {
            self.dir_n[m] += o.dir_n[m];
            self.dir_hold[m] += o.dir_hold[m];
            self.dir_model[m] += o.dir_model[m];
            for w in 0..2 {
                self.err[w][m].extend_from_slice(&o.err[w][m]);
            }
        }
        if self.feat_sum.len() < o.feat_sum.len() {
            self.feat_sum.resize(o.feat_sum.len(), 0.0);
        }
        for (a, b) in self.feat_sum.iter_mut().zip(&o.feat_sum) {
            *a += b;
        }
        self.feat_n += o.feat_n;
    }
}

fn find_map(dir: &Path, sha: &[u8; 32]) -> Result<PathBuf, String> {
    let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
    for e in std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .flatten()
    {
        if e.file_name().to_string_lossy().ends_with(&format!("{hex}.map")) {
            return Ok(e.path());
        }
    }
    Err(format!("no map {hex} in {}", dir.display()))
}

fn pctl(v: &mut [f32], p: f64) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

/// The 1vs1 box of the joni-duel arena in pixels (`configs/arenas/joni-duel.toml`: x 166..190, y 43..59 tiles, a tile of margin).
const DUEL_BOX_PX: [i32; 4] = [163 * 32, 42 * 32, 193 * 32, 61 * 32];

/// A frame of a duel: exactly two tees in the snapshot (ours and one other, none dropped) and both inside `box_px`.
fn is_duel_frame(f: &ddai_clip::format::Frame, box_px: [i32; 4]) -> bool {
    f.tees.len() == 2
        && f.tees_dropped == 0
        && f.tees
            .iter()
            .all(|t| (box_px[0]..=box_px[2]).contains(&t.ch.x) && (box_px[1]..=box_px[3]).contains(&t.ch.y))
}

fn check(clip: &Clip, map: Arc<MapData>, model: &mut OppPredictor, lag: usize, duel_box: Option<[i32; 4]>) -> Acc {
    let mut acc = Acc::default();
    let own = clip.header.own_id;
    let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
    for f in &clip.frames {
        for t in &f.tees {
            if t.id != own {
                *counts.entry(t.id).or_default() += 1;
            }
        }
    }
    let Some((&opp, _)) = counts.iter().max_by_key(|(_, c)| **c) else {
        return acc;
    };
    let mut lw = LiveWorld::new(Arc::clone(&map), own, clip.header.world_seed);
    // Per frame: the reconstructed world at the frame (when both tees are alive in it).
    let mut worlds: Vec<Option<ddai_physics::world::World<f32>>> = Vec::with_capacity(clip.frames.len());
    for f in &clip.frames {
        feed(&mut lw, clip, f);
        let ok =
            f.own_alive && f.tee(own).is_some() && f.tee(opp).is_some() && duel_box.is_none_or(|b| is_duel_frame(f, b));
        if std::env::var("OPP_DEBUG").is_ok() && worlds.len() < 6 {
            eprintln!("frame tick {} base world tick {}", f.tick, lw.base_world().tick);
        }
        worlds.push(ok.then(|| lw.base_world().clone()));
    }
    let mut pw = PhysicsWorld::from_world(
        worlds
            .iter()
            .flatten()
            .next()
            .cloned()
            .unwrap_or_else(|| lw.base_world().clone()),
        map,
    );
    let n = clip.frames.len();
    for i in 0..n.saturating_sub(4) {
        let t = clip.frames[i].tick;
        let consecutive = (1..=4).all(|m| clip.frames[i + m].tick == t + 2 * m as i32);
        let Some(w) = worlds[i].as_ref() else { continue };
        if !consecutive || (1..=4).any(|m| worlds[i + m].is_none()) {
            continue;
        }
        // The arena's games end at the first freeze: windows with a frozen tee anywhere in them are outside what the model was trained on.
        let frozen = |wd: &ddai_physics::world::World<f32>| {
            [own, opp].iter().any(|&id| {
                wd.characters[id as usize]
                    .as_ref()
                    .is_none_or(|c| c.freeze_time > 0 || !c.alive)
            })
        };
        if (0..=4).any(|m| worlds[i + m].as_ref().is_none_or(frozen)) {
            continue;
        }
        // Our inputs for the steps from T: those tagged T + 1 ..= T + 8.
        let sent_at = |tag: i32| {
            clip.frames[i + 1..=i + 4]
                .iter()
                .flat_map(|f| f.sent.iter())
                .find(|s| s.tick == tag)
                .map(|s| player_input_from_net(s.input.to_net()))
        };
        let ours: Option<Vec<Wire>> = (0..K as i32).map(|k| sent_at(t + 1 + k)).collect();
        let Some(ours) = ours else { continue };
        if let (Some(me), Some(op)) = (
            ddai_oppnet::frame::TeeFrame::from_world(w, own, opp),
            ddai_oppnet::frame::TeeFrame::from_world(w, opp, own),
        ) {
            let mut f = [0.0f32; ddai_oppnet::feature::FD];
            ddai_oppnet::feature::frame_features(&me, &op, &mut f);
            acc.feat_sum.resize(f.len(), 0.0);
            for (a, b) in acc.feat_sum.iter_mut().zip(f) {
                *a += f64::from(b);
            }
            acc.feat_n += 1;
        }
        let mut out: [Option<PredictedInput>; K] = [None; K];
        model.predict(
            &WindowCtx {
                world: w,
                self_id: own,
                victim_id: opp,
                in_flight: &ours[..lag],
            },
            &mut out,
        );
        // Direction: the snapshot at T + 2m shows the direction of the step T + 2m - 1 (k = 2m - 1).
        let dir_of = |wd: &ddai_physics::world::World<f32>| wd.cores.get(opp as u8).map(|c| c.direction.clamp(-1, 1));
        let Some(hold_dir) = dir_of(w) else { continue };
        for m in 1..=4usize {
            let Some(truth) = worlds[i + m].as_ref().and_then(dir_of) else {
                continue;
            };
            let k = 2 * m - 1;
            acc.dir_n[m - 1] += 1;
            acc.dir_hold[m - 1] += u64::from(hold_dir == truth);
            if let Some(p) = out[k] {
                acc.dir_model[m - 1] += u64::from(p.direction == truth);
            }
        }
        // Position: roll the physics with the opponent holding the snapshot (`snap`) or playing the prediction (`model`).
        for (wi, use_model) in [(0usize, false), (1, true)] {
            pw.sync_from(w);
            let Some(tee) = pw.get_tee(opp) else { continue };
            let snap_input = enemy_input_from_tee(&tee);
            let mut fire = w.cores.get(opp as u8).map_or(0, |c| c.input.fire);
            for (k, wire) in ours.iter().enumerate().take(K) {
                pw.set_input(own, from_ddnet_input(wire));
                let x = match (use_model, out[k]) {
                    (true, Some(p)) => input_from_prediction(&p, &snap_input, &mut fire),
                    _ => snap_input,
                };
                pw.set_input(opp, x);
                pw.step();
                if (k + 1) % 2 == 0 {
                    let m = (k + 1).div_ceil(2);
                    let (Some(sim), Some(truth)) = (
                        pw.get_tee(opp),
                        worlds[i + m].as_ref().and_then(|wd| wd.cores.get(opp as u8)),
                    ) else {
                        continue;
                    };
                    let e =
                        ((sim.pos.x as f32 - truth.pos.x).powi(2) + (sim.pos.y as f32 - truth.pos.y).powi(2)).sqrt();
                    acc.err[wi][m - 1].push(e);
                }
            }
        }
    }
    acc
}

fn main() -> Result<(), String> {
    let mut clips = PathBuf::new();
    let mut model_path = None;
    let mut maps = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("aiddnet/data/maps/cache");
    let mut lag = 3usize;
    let mut gate = 0.0f32;
    let mut contains = String::from("JoniTee");
    let mut duel_box = Some(DUEL_BOX_PX);
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--clips" => clips = PathBuf::from(v()?),
            "--model" => model_path = Some(PathBuf::from(v()?)),
            "--maps" => maps = PathBuf::from(v()?),
            "--gate" => gate = v()?.parse().map_err(|e| format!("--gate: {e}"))?,
            "--lag" => lag = v()?.parse().map_err(|e| format!("--lag: {e}"))?,
            "--map-contains" => contains = v()?,
            // Without the filter every frame of the map counts (crowd clips included): not a duel, only for comparison.
            "--no-duel-filter" => duel_box = None,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let model_path = model_path.ok_or(
        "usage: opp_clips --clips <dir> --model <bundle> [--lag 3] [--gate <logit margin>] [--map-contains JoniTee] [--no-duel-filter]",
    )?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(&clips)
        .map_err(|e| format!("{}: {e}", clips.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "clip"))
        .collect();
    files.sort();
    let mut pooled = Acc::default();
    let mut used = 0;
    for f in &files {
        let clip = Clip::read(f).map_err(|e| format!("{}: {e}", f.display()))?;
        if !clip.header.map_name.contains(&contains) {
            continue;
        }
        let map_file = find_map(&maps, &clip.header.map_sha256)?;
        let bytes = std::fs::read(&map_file).map_err(|e| e.to_string())?;
        let map = Arc::new(ddai_map::load_map(&bytes).map_err(|e| format!("{e}"))?.data);
        let mut model = OppPredictor::load(&model_path)?.with_gate(gate);
        model.reset();
        let acc = check(&clip, map, &mut model, lag, duel_box);
        let pc = |a: u64, n: u64| 100.0 * a as f64 / n.max(1) as f64;
        println!(
            "{}: {} frames, {} duel frames, {} windows; direction k = 3 / 5 / 7, hold -> model: {:.1} -> {:.1} / {:.1} -> {:.1} / {:.1} -> {:.1}; < 1 px after 4 ticks, snap -> model: {:.1} -> {:.1}",
            f.file_name().unwrap_or_default().to_string_lossy(),
            clip.frames.len(),
            clip.frames.iter().filter(|f| is_duel_frame(f, DUEL_BOX_PX)).count(),
            acc.dir_n[0],
            pc(acc.dir_hold[1], acc.dir_n[1]),
            pc(acc.dir_model[1], acc.dir_n[1]),
            pc(acc.dir_hold[2], acc.dir_n[2]),
            pc(acc.dir_model[2], acc.dir_n[2]),
            pc(acc.dir_hold[3], acc.dir_n[3]),
            pc(acc.dir_model[3], acc.dir_n[3]),
            pc(
                acc.err[0][1].iter().filter(|&&e| e < 1.0).count() as u64,
                acc.err[0][1].len() as u64
            ),
            pc(
                acc.err[1][1].iter().filter(|&&e| e < 1.0).count() as u64,
                acc.err[1][1].len() as u64
            ),
        );
        pooled.merge(&acc);
        used += 1;
    }
    println!(
        "\n## live clips: {used} clips of maps containing {contains:?}{}, window {lag} ticks, {} windows\n",
        if duel_box.is_some() {
            ", duel frames only (two tees, both in the 1vs1 box)"
        } else {
            ", all frames"
        },
        pooled.dir_n[0]
    );
    println!("| k (ticks after the snapshot) | n | direction: hold % | direction: model % |\n|---:|---:|---:|---:|");
    for m in 0..4 {
        let n = pooled.dir_n[m].max(1) as f64;
        println!(
            "| {} | {} | {:.1} | {:.1} |",
            2 * m + 1,
            pooled.dir_n[m],
            100.0 * pooled.dir_hold[m] as f64 / n,
            100.0 * pooled.dir_model[m] as f64 / n
        );
    }
    println!(
        "\nmean frame features of the clips: {:?}",
        pooled
            .feat_sum
            .iter()
            .map(|s| (s / pooled.feat_n.max(1) as f64 * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    println!(
        "\n| ticks | method | n | exact (<1 px) % | median px | p90 px | mean px |\n|---:|---|---:|---:|---:|---:|---:|"
    );
    for m in 0..4 {
        for (wi, name) in ["snap", "model"].iter().enumerate() {
            let v = &mut pooled.err[wi][m];
            if v.is_empty() {
                continue;
            }
            let exact = 100.0 * v.iter().filter(|&&x| x < 1.0).count() as f64 / v.len() as f64;
            let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
            println!(
                "| {} | {name} | {} | {exact:.1} | {:.2} | {:.2} | {mean:.2} |",
                2 * (m + 1),
                v.len(),
                pctl(v, 0.5),
                pctl(v, 0.9)
            );
        }
    }
    Ok(())
}
