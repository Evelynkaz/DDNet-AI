#![allow(dead_code)] // shared by two examples, each uses a part
//! Clips -> [`ClipGame`]s, through `LiveWorld` exactly as the live bot builds its worlds (task 3.21, E-036). Shared by `live_data` and `live_eval`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_clip::format::{Clip, Frame};
use ddai_clip::replay::feed;
use ddai_oppnet::clipdata::{ClipGame, ClipTick};
use ddai_oppnet::frame::{InputRec, N_RAYS, TeeFrame, rays};
use ddai_oppnet::live::RegimeGate;
use ddai_physics::map::MapData;
use ddai_world::LiveWorld;

pub fn find_map(dir: &Path, sha: &[u8; 32]) -> Result<PathBuf, String> {
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

pub fn load_map(dir: &Path, clip: &Clip) -> Result<Arc<MapData>, String> {
    let file = find_map(dir, &clip.header.map_sha256)?;
    let bytes = std::fs::read(&file).map_err(|e| e.to_string())?;
    Ok(Arc::new(ddai_map::load_map(&bytes).map_err(|e| format!("{e}"))?.data))
}

/// The target the bot would pick: the nearest other tee of the frame.
fn target_of(f: &Frame, own: i32) -> Option<i32> {
    let me = f.tee(own)?;
    f.tees
        .iter()
        .filter(|t| t.id != own)
        .min_by_key(|t| i64::from(t.ch.x - me.ch.x).pow(2) + i64::from(t.ch.y - me.ch.y).pow(2))
        .map(|t| t.id)
}

fn sent_input(f: &Frame, tick: i32) -> Option<InputRec> {
    f.sent
        .iter()
        .find(|s| s.tick == tick)
        .map(|s| InputRec::from_wire(&ddai_world::player_input_from_net(s.input.to_net())))
}

/// What a per-frame callback of [`convert_with`] sees: the `LiveWorld` right after the frame was fed (as the live bot has it at the decision), the clip, the
/// frame's index, the pair's ids and the regime flag.
pub struct FrameCtx<'a> {
    pub lw: &'a mut LiveWorld,
    pub clip: &'a Clip,
    pub idx: usize,
    pub own: i32,
    pub opp: i32,
    pub duel: bool,
}

/// Splits a clip into runs of consecutive frames with the same target (a death, a gap or another target starts a new run). Every frame of a run
/// carries the pair's [`TeeFrame`]s and the rays; `duel` says whether the regime gate admits it.
pub fn convert(clip: &Clip, map: Arc<MapData>, gate: &RegimeGate, source: &str, session: u8) -> Vec<ClipGame> {
    convert_with(clip, map, gate, source, session, &mut |_| ()).0
}

/// [`convert`] that also calls `cb` for every kept frame (in order, with the world fed up to that frame) and returns what it returned, per run and frame. The
/// callback also sees a new run start: `start` is true for the first frame of a run (a model's history must be reset there).
pub fn convert_with<T>(
    clip: &Clip,
    map: Arc<MapData>,
    gate: &RegimeGate,
    source: &str,
    session: u8,
    cb: &mut dyn FnMut(&mut FrameCtx<'_>) -> T,
) -> (Vec<ClipGame>, Vec<Vec<T>>) {
    let own = clip.header.own_id;
    let mut outs: Vec<Vec<T>> = Vec::new();
    let mut cur_out: Vec<T> = Vec::new();
    let mut lw = LiveWorld::new(map, own, clip.header.world_seed);
    let mut games: Vec<ClipGame> = Vec::new();
    let mut cur: Option<(i32, ClipGame)> = None;
    let mut part = 0;
    for (fi, f) in clip.frames.iter().enumerate() {
        feed(&mut lw, clip, f);
        let opp = if f.own_alive { target_of(f, own) } else { None };
        let w = lw.base_world();
        let pair = opp.and_then(|o| {
            Some((
                o,
                TeeFrame::from_world(w, own, o)?,
                TeeFrame::from_world(w, o, own)?,
                f.tee(o)?,
            ))
        });
        let Some((opp, me, op, rec)) = pair else {
            if let Some((_, g)) = cur.take() {
                games.push(g);
                outs.push(std::mem::take(&mut cur_out));
            }
            continue;
        };
        let gap = cur
            .as_ref()
            .is_some_and(|(o, g)| *o != opp || g.ticks.last().is_none_or(|t| f.tick != t.tick + 2));
        if gap && let Some((_, g)) = cur.take() {
            games.push(g);
            outs.push(std::mem::take(&mut cur_out));
        }
        let (_, g) = cur.get_or_insert_with(|| {
            part += 1;
            (
                opp,
                ClipGame {
                    source: format!("{source}p{part}"),
                    session,
                    ticks: Vec::new(),
                },
            )
        });
        let mut r_us = [1.0f32; N_RAYS];
        let mut r_opp = [1.0f32; N_RAYS];
        if me.alive {
            rays(w, me.pos, &mut r_us);
        }
        if op.alive {
            rays(w, op.pos, &mut r_opp);
        }
        let duel = me.alive && op.alive && gate.admits(w, own, opp, me.pos, op.pos);
        g.ticks.push(ClipTick {
            tick: f.tick,
            frames: [me, op],
            rays: [r_us, r_opp],
            sent: [sent_input(f, f.tick - 1), sent_input(f, f.tick)],
            opp_attack_tick: rec.ch.attack_tick,
            opp_weapon: rec.ch.weapon.clamp(-1, 8) as i8,
            duel,
        });
        cur_out.push(cb(&mut FrameCtx {
            lw: &mut lw,
            clip,
            idx: fi,
            own,
            opp,
            duel,
        }));
    }
    if let Some((_, g)) = cur.take() {
        games.push(g);
        outs.push(cur_out);
    }
    (games, outs)
}

/// The clips of `dirs` (`*.clip`, sorted per directory), identical ones (same map, own id, first and last tick) once, with their session
/// (see `live_data`); `manual` rehearsal clips are skipped.
pub fn collect_clips(dirs: &[PathBuf]) -> Result<Vec<(PathBuf, Clip, u8)>, String> {
    let mut files: Vec<PathBuf> = Vec::new();
    for d in dirs {
        let mut v: Vec<PathBuf> = std::fs::read_dir(d)
            .map_err(|e| format!("{}: {e}", d.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "clip"))
            .collect();
        v.sort();
        files.extend(v);
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for f in files {
        let clip = Clip::read(&f).map_err(|e| format!("{}: {e}", f.display()))?;
        let Some(session) = session_of(&clip) else { continue };
        let key = (
            clip.header.map_sha256,
            clip.header.own_id,
            clip.frames.first().map(|f| f.tick),
            clip.frames.last().map(|f| f.tick),
        );
        if seen.insert(key) {
            out.push((f, clip, session));
        }
    }
    Ok(out)
}

/// Session `0`: the 06.10 duels on the JoniTee map; `1`: the 07.10 test duel (ticks below 200 000); `2`: everything else (public servers).
pub fn session_of(clip: &Clip) -> Option<u8> {
    if clip.header.reason.kind == "manual" {
        return None;
    }
    let first = clip.frames.first().map_or(0, |f| f.tick);
    Some(if clip.header.map_name.contains("JoniTee") {
        0
    } else if first < 200_000 {
        1
    } else {
        2
    })
}
