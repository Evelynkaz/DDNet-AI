//! Task 3.10: an ASCII picture of a window of a real map's tiles (game layer, then front layer, tele and speedup where the game layer is air), to see
//! the ground an incident of a clip happened on. `map_ascii <map file> <x0> <y0> <x1> <y1>` in tiles (the clip tools print pixels: divide by 32).
//!
//! `#` solid, `X` death, `F` freeze, `U` unfreeze, `f`/`u` deep freeze/unfreeze, `n` no-hook, `T` tele, `>` speedup, `.` air.

use ddai_physics::map::{
    TILE_DEATH, TILE_DFREEZE, TILE_DUNFREEZE, TILE_FREEZE, TILE_NOHOOK, TILE_SOLID, TILE_UNFREEZE,
};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 5 {
        return Err("usage: map_ascii <map file> <x0> <y0> <x1> <y1>".into());
    }
    let n = |i: usize| a[i].parse::<i32>().map_err(|e| format!("{}: {e}", a[i]));
    let (x0, y0, x1, y1) = (n(1)?, n(2)?, n(3)?, n(4)?);
    let bytes = std::fs::read(&a[0]).map_err(|e| e.to_string())?;
    let map = ddai_map::load_map(&bytes).map_err(|e| e.to_string())?.data;
    println!(
        "legend: # solid, X death, F freeze, U unfreeze, f deep freeze, u deep unfreeze, n no-hook, T tele, > speedup, . air; lower row = front layer shown as the same letters in brackets only where the game layer is air"
    );
    for y in y0..=y1 {
        let mut line = format!("{y:>4} ");
        for x in x0..=x1 {
            let (w, h) = (map.width as i32, map.height as i32);
            if x < 0 || y < 0 || x >= w || y >= h {
                line.push(' ');
                continue;
            }
            let i = (y * w + x) as usize;
            let g = map.game[i].index;
            let f = map.front.as_ref().map_or(0, |f| f[i].index);
            let t = map.tele.as_ref().map_or(0, |t| t[i].kind);
            let s = map.speedup.as_ref().map_or(0, |s| s[i].force);
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
            let mut c = pick(g);
            if c == '.' {
                c = pick(f);
            }
            if c == '.' && t != 0 {
                c = 'T';
            }
            if c == '.' && s != 0 {
                c = '>';
            }
            line.push(c);
        }
        println!("{line}");
    }
    println!("      x from {x0} to {x1}");
    Ok(())
}
