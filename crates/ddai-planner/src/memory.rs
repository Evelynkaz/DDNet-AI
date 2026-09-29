//! `FreezeMemory` (`src/plan/memory.ts`) — a per-tile "here we froze / here we passed safely" map,
//! feeding `selfHazard` (via `memoryTrust`) and the `memoryWeight` risk penalty in `scoreTick`.
//! `Float32Array` in TS (`cells`/`passes`) -- kept `f32` here too (not widened to `f64`): every
//! read (`safety`/`risk`) immediately promotes to `f64` for the score, matching TS's own implicit
//! widening when a `Float32Array` element is read into a JS `number`.
//!
//! **Descoped:** `FreezeMemory::save`/`::load` (`memory.ts:72-129`, JSON-file persistence) are not
//! ported — `docs/research/orig-run.md` §2.1 notes the harness always starts with
//! `FreezeMemory = null`/a fresh, all-zero memory (a brand-new map has no history), and no
//! acceptance criterion here exercises loading a saved one. A fresh, all-zero
//! [`FreezeMemory::new`] is bit-identical to a freshly-constructed TS `FreezeMemory` for every
//! query this crate's tests exercise.

const TILE_PX: f64 = 32.0;
const SPREAD: f32 = 0.4;

/// `class FreezeMemory` (`memory.ts:10-130`).
#[derive(Debug, Clone)]
pub struct FreezeMemory {
    pub width: i32,
    pub height: i32,
    cells: Vec<f32>,
    passes: Vec<f32>,
    events: i64,
}

impl FreezeMemory {
    pub fn new(width: i32, height: i32) -> Self {
        let n = (width * height).max(0) as usize;
        FreezeMemory {
            width,
            height,
            cells: vec![0.0; n],
            passes: vec![0.0; n],
            events: 0,
        }
    }

    fn tile_index(&self, x: f64, y: f64) -> Option<usize> {
        let tx = ddai_jsmath::trunc(x / TILE_PX) as i32;
        let ty = ddai_jsmath::trunc(y / TILE_PX) as i32;
        if tx < 0 || ty < 0 || tx >= self.width || ty >= self.height {
            return None;
        }
        Some((ty * self.width + tx) as usize)
    }

    /// `notePass(x, y)` (`memory.ts:25-30`).
    pub fn note_pass(&mut self, x: f64, y: f64) {
        if let Some(i) = self.tile_index(x, y) {
            self.passes[i] += 1.0;
        }
    }

    /// `safety(x, y)` (`memory.ts:32-43`).
    pub fn safety(&self, x: f64, y: f64) -> f64 {
        let Some(i) = self.tile_index(x, y) else { return 0.0 };
        let good = f64::from(self.passes[i]);
        let bad = f64::from(self.cells[i]);
        if good <= 0.0 {
            return 0.0;
        }
        let clean = good / (good + 15.0 * bad);
        clean * (good / (good + 10.0))
    }

    pub fn noted(&self) -> i64 {
        self.events
    }

    /// `note(x, y)` (`memory.ts:49-62`): +1 at the center tile, `+SPREAD` at the 8 neighbours.
    pub fn note(&mut self, x: f64, y: f64) {
        let tx = ddai_jsmath::trunc(x / TILE_PX) as i32;
        let ty = ddai_jsmath::trunc(y / TILE_PX) as i32;
        if tx < 0 || ty < 0 || tx >= self.width || ty >= self.height {
            return;
        }
        self.events += 1;
        for oy in -1..=1 {
            for ox in -1..=1 {
                let nx = tx + ox;
                let ny = ty + oy;
                if nx < 0 || ny < 0 || nx >= self.width || ny >= self.height {
                    continue;
                }
                let idx = (ny * self.width + nx) as usize;
                self.cells[idx] += if ox == 0 && oy == 0 { 1.0 } else { SPREAD };
            }
        }
    }

    /// `risk(x, y)` (`memory.ts:64-70`).
    pub fn risk(&self, x: f64, y: f64) -> f64 {
        let Some(i) = self.tile_index(x, y) else { return 0.0 };
        let v = f64::from(self.cells[i]);
        if v <= 0.0 { 0.0 } else { v / (1.0 + v) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_memory_is_neutral_everywhere() {
        let m = FreezeMemory::new(10, 10);
        assert_eq!(m.safety(50.0, 50.0), 0.0);
        assert_eq!(m.risk(50.0, 50.0), 0.0);
    }

    #[test]
    fn note_raises_risk_at_center_more_than_neighbours() {
        let mut m = FreezeMemory::new(10, 10);
        m.note(5.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0);
        let center = m.risk(5.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0);
        let neighbour = m.risk(6.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0);
        assert!(center > neighbour);
        assert!(neighbour > 0.0);
        assert_eq!(m.noted(), 1);
    }

    #[test]
    fn note_pass_then_safety_is_positive_without_any_bad_note() {
        let mut m = FreezeMemory::new(10, 10);
        m.note_pass(16.0, 16.0);
        assert!(m.safety(16.0, 16.0) > 0.0);
    }

    #[test]
    fn out_of_bounds_queries_are_neutral_and_note_is_a_no_op() {
        let mut m = FreezeMemory::new(4, 4);
        m.note(-100.0, -100.0);
        assert_eq!(m.noted(), 0);
        assert_eq!(m.safety(-100.0, -100.0), 0.0);
        assert_eq!(m.risk(1000.0, 1000.0), 0.0);
    }
}
