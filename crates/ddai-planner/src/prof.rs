//! Review round 2, F12: work counters (physics steps done by the shield's own rollouts,
//! `escape_exists` call counts) plus four wall-clock marks inside
//! [`crate::planner::Planner::decide_production`] -- the stall-proof latency measure this VM's
//! `/proc/thread-self/schedstat`-based thread-CPU accounting cannot provide (round 2 review: on
//! this VM, schedstat counts a scheduler stall as run time, so a search that did 48 physics steps
//! -- about half a microsecond of real work each -- reads back as ~11 ms of "CPU"; work counters
//! don't have that failure mode because they only increment on an actual completed `world.step()`
//! call, never while merely waiting for the scheduler).
//!
//! Layout matches the reviewer's own scratch instrumentation
//! (`.../scratchpad/3.2/r2/crates/ddai-planner/tests/zz_r2_phase.rs`,
//! `.../planner-r2-instr.rs`) so that harness can run against this crate unmodified: four marks
//! per `decide_production` call (0 = search start, 1 = search end, 2 = after `stepToInput`/before
//! the shield, 3 = after the shield), each a `(wall_time, cpu_ns, steps, escape_exists_calls)`
//! tuple. `cpu_ns` is always `0` here -- deliberately never populated, not merely unimplemented:
//! per-thread CPU is not a reliable signal on this machine (see above), so nothing should read it
//! as one. `steps`/`escape_exists_calls` are the load-bearing fields.
//!
//! **Review round 3, F16: opt-in, off by default, fixed-size storage.** Before this
//! fix, every one of the four marks per `decide_production` call unconditionally pushed onto a
//! thread-local `Vec` that only a test ever drained (via [`take`]) -- on the live path nothing
//! ever calls `take`, so the `Vec` grew forever: the reviewer's `zz_r3_leak.rs` measured 12,000
//! retained marks (480 B) after 3,000 undrained `decide_production` calls, i.e. 160 B/decision,
//! ~430 MB over a 15-hour live session, plus periodic `Vec`-doubling copies landing as latency
//! spikes on whichever decision triggers them. Fixed by:
//! - a thread-local `enabled` flag, off by default -- [`inc_step`]/[`inc_escape`]/[`mark`] are a
//!   single relaxed boolean check and an early return when it is `false`, so the disabled cost is
//!   one branch, no allocation, no counter writes, ever;
//! - callers that want measurements call [`enable`] first (every test that reads marks does this;
//!   [`disable`] undoes it, though letting a test process just exit works too since this is
//!   thread-local, not global, state);
//! - storage is a **fixed 4-slot array** (`[Option<(Instant, u64, u64, u64)>; 4]`), never a
//!   growing `Vec` -- `mark(slot)` writes directly to `slot`, so it is impossible for storage to
//!   exceed 4 entries regardless of how many `decide_production` calls happen without a `take()`
//!   in between (unlike the old `Vec`, which only stopped growing because tests happened to call
//!   `take()` every decision -- nothing enforced that). [`take`] still returns a `Vec<...>` (a
//!   fresh, small, transient allocation, at most 4 elements, discarded by the caller -- not
//!   retained state), for source compatibility with every existing reader.
use std::cell::Cell;
use std::time::Instant;

/// `(wall_time, cpu_ns, steps_so_far, escapes_so_far)` -- see the module doc comment.
type Mark = (Instant, u64, u64, u64);

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static STEPS: Cell<u64> = const { Cell::new(0) };
    static ESCAPES: Cell<u64> = const { Cell::new(0) };
    static MARKS: Cell<[Option<Mark>; 4]> = const { Cell::new([None, None, None, None]) };
}

/// Turns on this thread's instrumentation (counters increment, marks are recorded). Off by
/// default -- callers (tests, or a future opt-in debug mode) must call this before any of
/// [`inc_step`]/[`inc_escape`]/[`mark`] will do anything.
pub fn enable() {
    ENABLED.with(|e| e.set(true));
}

/// Turns this thread's instrumentation back off. Does not clear already-recorded marks/counters
/// (a following [`take`] still drains whatever was last written); it only stops new writes.
pub fn disable() {
    ENABLED.with(|e| e.set(false));
}

pub fn is_enabled() -> bool {
    ENABLED.with(Cell::get)
}

/// Called once per `world.step()` made by `shield::escape_exists_inner`/`settles_safe` --
/// backend-agnostic (works the same whether the caller is on `ts_adapter` or `physics_adapter`),
/// and identical whether the call came from the bounded (`decide_production`) or unbounded
/// (`decide_once`) entry point, since both route through the same inner function. A no-op (one
/// boolean check, no write) unless [`enable`] was called on this thread.
pub fn inc_step() {
    if !is_enabled() {
        return;
    }
    STEPS.with(|s| s.set(s.get() + 1));
}

/// Called once per `escape_exists`/`escape_exists_bounded` invocation (including the ones
/// `safer_input`/`safer_input_bounded` make internally for each alternative it tries). A no-op
/// unless [`enable`] was called on this thread.
pub fn inc_escape() {
    if !is_enabled() {
        return;
    }
    ESCAPES.with(|e| e.set(e.get() + 1));
}

/// Records `(Instant::now(), 0, steps_so_far, escapes_so_far)` into the fixed slot `slot` (must be
/// `0..4`; out-of-range silently does nothing rather than panicking, since this is diagnostic
/// instrumentation, never load-bearing for correctness). Overwrites whatever was in that slot from
/// a previous `decide_production` call -- callers only ever care about the most recent one. A
/// no-op unless [`enable`] was called on this thread.
pub fn mark(slot: usize) {
    if !is_enabled() || slot >= 4 {
        return;
    }
    let steps = STEPS.with(Cell::get);
    let escapes = ESCAPES.with(Cell::get);
    MARKS.with(|m| {
        let mut arr = m.get();
        arr[slot] = Some((Instant::now(), 0, steps, escapes));
        m.set(arr);
    });
}

/// Drains and returns whatever marks are currently recorded (at most 4, in slot order 0..4, only
/// the slots that have been written at least once since the last `take()`). The returned `Vec` is
/// a fresh, small, transient allocation -- not retained state; nothing here keeps growing no
/// matter how many `decide_production` calls happen between `take()` calls, disabled or not.
pub fn take() -> Vec<Mark> {
    MARKS.with(|m| {
        let arr = m.replace([None, None, None, None]);
        arr.into_iter().flatten().collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review round 3, F16's own acceptance test: with `prof` left at its default (disabled)
    /// state, 10k `decide_production`-shaped call patterns (well, just `mark`/`inc_step` calls
    /// directly here -- exercising the full `Planner` needs a `PlanWorld`, which this module
    /// doesn't have; `zz_r3_leak.rs`-equivalent coverage through the real `Planner` is
    /// `prof_stays_disabled_by_default_across_many_decisions` in
    /// `tests/phase_breakdown.rs`/a dedicated unit test below) must leave no retained marks and
    /// the counters at zero, proving the disabled path never writes anything.
    #[test]
    fn disabled_by_default_10k_calls_leave_no_marks_or_counters() {
        disable(); // this thread may have been left `enabled` by an earlier test in the same binary
        for _ in 0..10_000 {
            inc_step();
            inc_escape();
            mark(0);
            mark(1);
            mark(2);
            mark(3);
        }
        assert_eq!(take().len(), 0, "disabled prof must never record a mark");
        assert_eq!(
            STEPS.with(Cell::get),
            0,
            "disabled prof must never increment the step counter"
        );
        assert_eq!(
            ESCAPES.with(Cell::get),
            0,
            "disabled prof must never increment the escape counter"
        );
    }

    #[test]
    fn enabled_records_exactly_the_last_four_marks_fixed_size() {
        disable();
        enable();
        for _ in 0..1000 {
            mark(0);
            inc_step();
            mark(1);
        }
        // Only slots 0 and 1 were ever written on this pass -- `take()` must return exactly 2, not
        // 1000 * 2, proving storage is fixed-size (overwritten in place), not an accumulating Vec.
        let marks = take();
        assert_eq!(
            marks.len(),
            2,
            "storage must be fixed-size (overwrite in place), not accumulate"
        );
        // A second `take()` with nothing written in between is empty (drained).
        assert_eq!(take().len(), 0);
        disable();
    }

    #[test]
    fn out_of_range_slot_is_a_silent_no_op_not_a_panic() {
        disable();
        enable();
        mark(4);
        mark(100);
        assert_eq!(take().len(), 0);
        disable();
    }
}
