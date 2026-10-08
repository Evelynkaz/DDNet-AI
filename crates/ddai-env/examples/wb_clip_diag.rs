//! Task 3.18: what happened to every block of a live clip at the wayblock. For each block (`BotRec::blocks` goes up) it prints the victim's track every
//! few frames with the tile kind under it, who hooks it, the hammer hits on it, and where we are and what we do.
//!
//! ```text
//! cargo run --release -p ddai-env --example wb_clip_diag -- <map file> <clip> [<clip>...] [--step N] [--until TICKS]
//! ```

use ddai_clip::format::{Clip, ClipEvent, Frame};
use ddai_clip::held::block_fates;
use ddai_physics::map::{
    TILE_DEATH, TILE_DFREEZE, TILE_DUNFREEZE, TILE_FREEZE, TILE_NOHOOK, TILE_SOLID, TILE_UNFREEZE,
};

fn kind(map: &ddai_physics::map::MapData, tx: i32, ty: i32) -> char {
    let (w, h) = (map.width as i32, map.height as i32);
    if tx < 0 || ty < 0 || tx >= w || ty >= h {
        return '?';
    }
    let i = (ty * w + tx) as usize;
    let pick = |idx: u8| match idx {
        TILE_SOLID => '#',
        TILE_DEATH => 'X',
        TILE_FREEZE => 'F',
        TILE_UNFREEZE => 'U',
        TILE_DFREEZE => 'f',
        TILE_DUNFREEZE => 'u',
        TILE_NOHOOK => 'n',
        _ => '.',
    };
    let g = pick(map.game[i].index);
    if g != '.' {
        return g;
    }
    map.front.as_ref().map_or('.', |f| pick(f[i].index))
}

fn main() -> Result<(), String> {
    let mut files = Vec::new();
    let mut step = 5usize;
    let mut until = 260i32;
    let mut it = std::env::args().skip(1);
    let map_file = it
        .next()
        .ok_or("usage: wb_clip_diag <map> <clip>... [--step N] [--until T]")?;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--step" => step = it.next().ok_or("--step N")?.parse().map_err(|e| format!("{e}"))?,
            "--until" => until = it.next().ok_or("--until T")?.parse().map_err(|e| format!("{e}"))?,
            _ => files.push(a),
        }
    }
    let bytes = std::fs::read(&map_file).map_err(|e| e.to_string())?;
    let map = ddai_map::load_map(&bytes).map_err(|e| e.to_string())?.data;
    for f in files {
        let clip = Clip::read(std::path::Path::new(&f)).map_err(|e| e.to_string())?;
        let own_id = clip.header.own_id;
        println!("== {f}: own id {own_id}, ticks {:?}", clip.tick_range());
        for fate in block_fates(&clip.frames, own_id) {
            println!(
                "-- block tick {} victim {} fate {:?} out {} target_at_block {} off_target {} touches {}",
                fate.tick,
                fate.victim,
                fate.fate,
                fate.out_ticks,
                fate.target_at_block,
                fate.frames_off_target,
                fate.touches
            );
            let t0 = fate.tick;
            let frames: Vec<&Frame> = clip.frames[fate.frame.saturating_sub(5)..]
                .iter()
                .take_while(|f| f.tick - t0 <= until)
                .collect();
            for (k, fr) in frames.iter().enumerate() {
                let hooks_on_victim: Vec<i32> = fr
                    .tees
                    .iter()
                    .filter(|t| t.ch.hooked_player == fate.victim && t.id != fate.victim)
                    .map(|t| t.id)
                    .collect();
                let hits: Vec<String> = fr
                    .events
                    .iter()
                    .filter_map(|e| match *e {
                        ClipEvent::HammerHit { from, to } if to == fate.victim => Some(format!("hammer {from}->v")),
                        ClipEvent::HookAttach { id, target } if target == fate.victim => Some(format!("hook {id}->v")),
                        ClipEvent::HookRelease { id, target, held } if target == fate.victim => {
                            Some(format!("release {id}->v after {held}"))
                        }
                        ClipEvent::FreezeOnset { id } if id == fate.victim || id == own_id => {
                            Some(format!("FREEZE {id}"))
                        }
                        ClipEvent::Kill { killer, victim, weapon } => {
                            Some(format!("kill {killer}->{victim} w{weapon}"))
                        }
                        _ => None,
                    })
                    .collect();
                if k % step != 0 && hits.is_empty() {
                    continue;
                }
                let v = fr.tee(fate.victim);
                let me = fr.tee(own_id);
                let vs = match v {
                    Some(v) => {
                        let (x, y) = v.pos();
                        let (vx, vy) = v.vel();
                        format!(
                            "v ({:.1},{:.1}) tile {} vel ({:.1},{:.1}) fr {} left {} hookedby {:?} vhook {}/{}",
                            x / 32.0,
                            y / 32.0,
                            kind(&map, (x / 32.0).floor() as i32, (y / 32.0).floor() as i32),
                            vx,
                            vy,
                            v.frozen as u8,
                            v.freeze_left,
                            hooks_on_victim,
                            v.ch.hook_state,
                            v.ch.hooked_player
                        )
                    }
                    None => "v (not in view)".to_string(),
                };
                let ms = match me {
                    Some(m) => {
                        let (x, y) = m.pos();
                        let (vx, vy) = m.vel();
                        format!(
                            "me ({:.1},{:.1}) vel ({:.1},{:.1}) fr {} hook {}/{} tgt {} bits {:#x}",
                            x / 32.0,
                            y / 32.0,
                            vx,
                            vy,
                            m.frozen as u8,
                            m.ch.hook_state,
                            m.ch.hooked_player,
                            fr.bot.target,
                            fr.bot.flags
                        )
                    }
                    None => "me (dead)".to_string(),
                };
                println!("{:+4} {} | {} | {}", fr.tick - t0, vs, ms, hits.join(", "));
            }
        }
    }
    Ok(())
}
