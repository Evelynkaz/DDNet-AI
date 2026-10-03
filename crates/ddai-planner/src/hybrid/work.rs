//! The work clock (task 3.5, D-045): a [`Clock`] that counts *work* instead of wall time -- the
//! physics ticks the search has simulated plus the hook-anchor ray casts, at a fixed
//! microseconds-per-tick rate. On this VM host stalls of ~10 ms and neighbouring jobs make a wall
//! deadline give a different search on every run (a loaded machine turns a "4 ms" search into a
//! 1 ms one); with a work clock the deadline mode is **reproducible** and independent of load, so
//! budget-versus-strength questions (D-042/D-044) can be answered in the arena without the noise.
//! It needs no thread other than the deciding one (`workers = 1`): helper threads read wall time.
//!
//! The unit is a **tee-tick**: one physics tick of one tee (a tick's cost grows with the number of
//! tees in the world). The rate is a calibration, not a truth: [`WORK_US_PER_TEE_TICK`] is what a whole
//! decision costs per tee-tick (rollouts with their scoring, shield, search bookkeeping) on an unloaded
//! core. Task 3.6 (D-076, `docs/EXPERIMENTS.md` E-010) sped the search up by about 1.75x and moved it from
//! the `2.2 us` of E-003/E-007 to `1.25 us`, so a work budget of 4 ms is now 3 200 tee-ticks: 1 600 ticks
//! with 2 tees, 800 with 4, 530 with 6 -- what the wall clock gives an idle machine.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::clock::Clock;

/// Microseconds one tee-tick of a hybrid decision costs on an unloaded core: the work clock's default rate
/// (D-076, task 3.6; it was `2.2` in E-003/E-007, still pinned by `step_ms = 0.0022` in those experiments'
/// configs). The unit of the 4 ms search budget and of the "<= 5 ms work" bound of D-042.
///
/// **Provisional:** derived from the measured old/new speed ratio (1.73x) applied to the old 2.2 us, not from a
/// direct idle measurement (D-076); a pinned-core, training-paused measurement is scheduled. It only decides how
/// much search the *arena* gives a "4 ms" hybrid (comparability); the live bot reads the wall clock.
pub const WORK_US_PER_TEE_TICK: f64 = 1.25;

/// The shared tick counter the search increments and a [`WorkClock`] reads.
#[derive(Debug, Default)]
pub struct WorkMeter {
    done: AtomicU64,
    /// Tees in the world of the current decision: what a tick is multiplied by.
    scale: AtomicU64,
    /// While the shield runs: the shield's step counter (`prof`) when it began; `u64::MAX` = idle.
    shield_base: AtomicU64,
}

impl WorkMeter {
    pub fn new() -> Arc<WorkMeter> {
        Arc::new(WorkMeter {
            done: AtomicU64::new(0),
            scale: AtomicU64::new(1),
            shield_base: AtomicU64::new(u64::MAX),
        })
    }

    /// The number of tees a physics tick simulates from now on (set once per decision).
    pub fn set_scale(&self, tees: usize) {
        self.scale.store(tees.max(1) as u64, Ordering::Relaxed);
    }

    /// Adds finished physics ticks (each is `scale` tee-ticks).
    pub fn add(&self, ticks: u64) {
        self.done
            .fetch_add(ticks * self.scale.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    /// Adds finished work already in tee-ticks (an anchor ray cast counts as two).
    pub fn add_units(&self, units: u64) {
        self.done.fetch_add(units, Ordering::Relaxed);
    }

    /// The shield starts stepping the world; its steps count as they happen (the shield polls the
    /// clock inside its rollouts).
    pub fn begin_shield(&self) {
        self.shield_base.store(crate::prof::counters().0, Ordering::Relaxed);
    }

    /// The shield is done: its steps become finished work.
    pub fn end_shield(&self) {
        let base = self.shield_base.swap(u64::MAX, Ordering::Relaxed);
        if base != u64::MAX {
            self.add(crate::prof::counters().0.saturating_sub(base));
        }
    }

    pub fn ticks(&self) -> u64 {
        let base = self.shield_base.load(Ordering::Relaxed);
        let live = if base == u64::MAX {
            0
        } else {
            crate::prof::counters().0.saturating_sub(base) * self.scale.load(Ordering::Relaxed)
        };
        self.done.load(Ordering::Relaxed) + live
    }
}

/// `now_ms` = tee-ticks so far * `us_per_tee_tick` / 1000.
#[derive(Debug, Clone)]
pub struct WorkClock {
    meter: Arc<WorkMeter>,
    us_per_tee_tick: f64,
}

impl WorkClock {
    pub fn new(meter: Arc<WorkMeter>, us_per_tee_tick: f64) -> WorkClock {
        WorkClock { meter, us_per_tee_tick }
    }
}

impl Clock for WorkClock {
    fn now_ms(&self) -> f64 {
        self.meter.ticks() as f64 * self.us_per_tee_tick / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_advances_with_added_work_only() {
        let m = WorkMeter::new();
        let c = WorkClock::new(Arc::clone(&m), 2.0);
        assert_eq!(c.now_ms(), 0.0);
        assert_eq!(c.now_ms(), 0.0, "reading does not advance it");
        m.set_scale(4);
        m.add(500); // 500 ticks of 4 tees = 2000 tee-ticks
        assert_eq!(c.now_ms(), 4.0);
        m.add_units(500);
        assert_eq!(c.now_ms(), 5.0);
    }

    #[test]
    fn shield_steps_count_live_and_are_kept_when_it_ends() {
        crate::prof::disable();
        crate::prof::enable();
        let m = WorkMeter::new();
        let c = WorkClock::new(Arc::clone(&m), 10.0);
        m.set_scale(1);
        m.begin_shield();
        for _ in 0..100 {
            crate::prof::inc_step();
        }
        assert_eq!(c.now_ms(), 1.0);
        m.end_shield();
        assert_eq!(c.now_ms(), 1.0);
        m.end_shield();
        assert_eq!(c.now_ms(), 1.0, "ending twice adds nothing");
        crate::prof::disable();
    }
}
