//! `gridOf` (`route.ts:55-161`): the per-map boolean grids the route search runs on.

use ddai_planner::plan_world::PlanCollision;
use ddai_planner::tuning::{TILE_TELEIN, TILE_TELEINEVIL};
use std::collections::HashMap;

use crate::TILE_PX;

/// `RAY_DX`/`RAY_DY` (`route.ts:475-476`): the 8 rope directions, clockwise from up.
pub const RAY_DX: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];
pub const RAY_DY: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];

/// The grids (`type Grid`).
#[derive(Debug, Clone)]
pub struct NavGrid {
    pub width: i32,
    pub height: i32,
    /// A tee can stand/fly here: not solid, not freeze, not death (a tele entrance counts as free).
    pub free: Vec<u8>,
    /// A rope can attach: solid and not no-hook.
    pub hookable: Vec<u8>,
    pub solid: Vec<u8>,
    pub death: Vec<u8>,
    pub unfreeze: Vec<u8>,
    /// Any of the 3x3 neighbourhood is freeze or death.
    pub danger: Vec<u8>,
    /// For a tele entrance: the tile index of its (first) exit, else -1.
    pub tele_out: Vec<i32>,
    /// For an exit tile: the entrance tiles leading to it.
    pub tele_in: HashMap<i32, Vec<i32>>,
    /// For each of the 8 directions and tile: the index of the first solid tile along the ray, else -1.
    pub first_solid: Vec<i32>,
}

impl NavGrid {
    /// `gridOf(collision)`.
    pub fn new(col: &impl PlanCollision) -> NavGrid {
        let (width, height) = (col.width(), col.height());
        let n = (width * height) as usize;
        let mut free = vec![0u8; n];
        let mut hookable = vec![0u8; n];
        let mut solid = vec![0u8; n];
        let mut death = vec![0u8; n];
        let mut unfreeze = vec![0u8; n];
        let half = f64::from(TILE_PX) / 2.0;
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) as usize;
                let px = f64::from(x * TILE_PX) + half;
                let py = f64::from(y * TILE_PX) + half;
                let is_solid = col.is_solid(px, py);
                solid[i] = u8::from(is_solid);
                death[i] = u8::from(col.is_death(px, py));
                unfreeze[i] = u8::from(col.is_un_freeze(px, py));
                hookable[i] = u8::from(is_solid && !col.is_no_hook(px, py));
                free[i] = u8::from(!is_solid && !col.is_freeze(px, py) && !col.is_death(px, py));
                if free[i] == 0 && !is_solid {
                    let t = col.tele_at(px, py).0;
                    if t == i32::from(TILE_TELEIN) || t == i32::from(TILE_TELEINEVIL) {
                        free[i] = 1;
                    }
                }
            }
        }
        let mut danger = vec![0u8; n];
        for y in 0..height {
            for x in 0..width {
                let mut bad = 0u8;
                'scan: for oy in -1..=1 {
                    for ox in -1..=1 {
                        let (nx, ny) = (x + ox, y + oy);
                        if nx < 0 || ny < 0 || nx >= width || ny >= height {
                            continue;
                        }
                        let px2 = f64::from(nx * TILE_PX) + half;
                        let py2 = f64::from(ny * TILE_PX) + half;
                        if col.is_freeze(px2, py2) || col.is_death(px2, py2) {
                            bad = 1;
                            break 'scan;
                        }
                    }
                }
                danger[(y * width + x) as usize] = bad;
            }
        }
        let mut tele_out = vec![-1i32; n];
        let mut tele_in: HashMap<i32, Vec<i32>> = HashMap::new();
        if col.has_tele() {
            for i in 0..width * height {
                let (x, y) = (i % width, i / width);
                let (t, number) = col.tele_at(f64::from(x * TILE_PX) + half, f64::from(y * TILE_PX) + half);
                if t != i32::from(TILE_TELEIN) && t != i32::from(TILE_TELEINEVIL) {
                    continue;
                }
                let outs = col.tele_outs_for(number);
                let Some(first) = outs.first() else { continue };
                // `Math.trunc(outs[0].y / TILE_PX) * width + Math.trunc(outs[0].x / TILE_PX)`
                let oi = (first.y / f64::from(TILE_PX)).trunc() as i32 * width
                    + (first.x / f64::from(TILE_PX)).trunc() as i32;
                tele_out[i as usize] = oi;
                tele_in.entry(oi).or_default().push(i);
            }
        }
        let area = (width * height) as usize;
        let mut first_solid = vec![-1i32; 8 * area];
        for d in 0..8usize {
            let (dx, dy) = (RAY_DX[d], RAY_DY[d]);
            let xs: Vec<i32> = if dx > 0 {
                (0..width).rev().collect()
            } else {
                (0..width).collect()
            };
            let ys: Vec<i32> = if dy > 0 {
                (0..height).rev().collect()
            } else {
                (0..height).collect()
            };
            for &y in &ys {
                for &x in &xs {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= width || ny >= height {
                        continue;
                    }
                    let ni = (ny * width + nx) as usize;
                    first_solid[d * area + (y * width + x) as usize] = if solid[ni] == 1 {
                        ni as i32
                    } else {
                        first_solid[d * area + ni]
                    };
                }
            }
        }
        NavGrid {
            width,
            height,
            free,
            hookable,
            solid,
            death,
            unfreeze,
            danger,
            tele_out,
            tele_in,
            first_solid,
        }
    }

    pub fn idx(&self, x: i32, y: i32) -> usize {
        (y * self.width + x) as usize
    }

    /// `supported(g, x, y)`: the tile below is solid (or off the map).
    pub fn supported(&self, x: i32, y: i32) -> bool {
        y + 1 >= self.height || self.solid[((y + 1) * self.width + x) as usize] == 1
    }

    /// `clearLine` (`route.ts:165-189`): Bresenham over tiles; `for_tee` demands free tiles, otherwise
    /// non-solid ones. The two end tiles are not tested.
    pub fn clear_line(&self, x0: i32, y0: i32, x1: i32, y1: i32, for_tee: bool) -> bool {
        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;
        let (mut x, mut y) = (x0, y0);
        loop {
            if !(x == x0 && y == y0) && !(x == x1 && y == y1) {
                let i = (y * self.width + x) as usize;
                let blocked = if for_tee { self.free[i] == 0 } else { self.solid[i] == 1 };
                if blocked {
                    return false;
                }
            }
            if x == x1 && y == y1 {
                return true;
            }
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }
}
