//! The handful of `TUNING`/tile constants the planner's own geometry heuristics need
//! (`src/core/tuning.ts`), duplicated here (rather than pulled from `ddai-tsworld`) so this crate
//! does not have to depend on `ddai-tsworld` outside the `ts-parity` feature. Every value goes
//! through the same `tune(v) = trunc(v * 100) / 100` step TS applies at module load
//! (`docs/research/orig-plan.md` §3; `ddai-tsworld/src/tuning.rs`'s doc comment) — computed once,
//! not hardcoded as a decimal literal, so a transcription slip can't silently diverge from what
//! `js::trunc` actually produces.

use ddai_jsmath as js;
use std::sync::LazyLock;

fn tune(v: f64) -> f64 {
    js::trunc(v * 100.0) / 100.0
}

pub const PHYSICAL_SIZE: f64 = 28.0;
pub const SERVER_TICK_SPEED: f64 = 50.0;

pub static GRAVITY: LazyLock<f64> = LazyLock::new(|| tune(0.5));
pub static GROUND_FRICTION: LazyLock<f64> = LazyLock::new(|| tune(0.5));
pub static AIR_FRICTION: LazyLock<f64> = LazyLock::new(|| tune(0.95));
pub static HOOK_LENGTH: LazyLock<f64> = LazyLock::new(|| tune(380.0));
pub static HAMMER_STRENGTH: LazyLock<f64> = LazyLock::new(|| tune(1.0));
pub static VELRAMP_START: LazyLock<f64> = LazyLock::new(|| tune(550.0));
pub static VELRAMP_RANGE: LazyLock<f64> = LazyLock::new(|| tune(2000.0));
pub static VELRAMP_CURVATURE: LazyLock<f64> = LazyLock::new(|| tune(1.4));
pub static GROUND_CONTROL_SPEED: LazyLock<f64> = LazyLock::new(|| tune(10.0));

// --- Tile ids (`tuning.ts:68-98`), game layer numeric convention every backend shares ----------

pub const TILE_AIR: u8 = 0;
pub const TILE_SOLID: u8 = 1;
pub const TILE_DEATH: u8 = 2;
pub const TILE_NOHOOK: u8 = 3;
pub const TILE_FREEZE: u8 = 9;
pub const TILE_TELEINEVIL: u8 = 10;
pub const TILE_UNFREEZE: u8 = 11;
pub const TILE_TELEIN: u8 = 26;
pub const TILE_TELECHECK: u8 = 29;
pub const TILE_TELECHECKOUT: u8 = 30;
pub const TILE_TELECHECKIN: u8 = 31;
pub const TILE_TELECHECKINEVIL: u8 = 63;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tune_matches_ts_module_load_constants() {
        // Spot-checked against `ddai-tsworld`'s own `tuning()` (task 1.9, itself proven bit-exact
        // against real V8) -- see that crate's README for the `trunc(v*100)/100` quirk.
        assert_eq!(*GRAVITY, 0.5);
        assert_eq!(*AIR_FRICTION, 0.95);
        assert_eq!(*HOOK_LENGTH, 380.0);
        assert_eq!(*VELRAMP_CURVATURE, 1.4);
    }
}
