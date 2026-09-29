//! `src/plan/throwLines.ts` — fixed hook "throw" seed plans for the `freezeThrow`/`frozenThrow`
//! opening-book extensions (`docs/research/orig-plan.md` §1.6).

use crate::planner::PlanStep;
use ddai_jsmath as js;

const THROW_RANGE_NEARNESS: f64 = 0.4;

#[derive(Debug, Clone, Copy)]
pub struct ThrowSituation {
    pub separation: f64,
    pub enemy_hazard_nearness: f64,
    pub me_frozen: bool,
    pub enemy_frozen: bool,
    pub enemy_alive: bool,
}

/// `throwWorthTrying(s)` (`throwLines.ts:16-20`).
pub fn throw_worth_trying(s: &ThrowSituation) -> bool {
    if s.me_frozen || s.enemy_frozen || !s.enemy_alive {
        return false;
    }
    if s.separation > *crate::tuning::HOOK_LENGTH {
        return false;
    }
    s.enemy_hazard_nearness >= THROW_RANGE_NEARNESS
}

fn mk(steps: i32, fn_: impl Fn(i32) -> PlanStep) -> Vec<PlanStep> {
    (0..steps).map(fn_).collect()
}

/// `throwLines(steps, at)` (`throwLines.ts:22-39`).
pub fn throw_lines(steps: i32, at: f64) -> Vec<Vec<PlanStep>> {
    let releases = [
        2.0,
        js::max(3.0, js::round(f64::from(steps) / 3.0)),
        js::max(5.0, js::round(2.0 * f64::from(steps) / 3.0)),
        f64::from(steps),
    ];
    let mid = js::max(3.0, js::round(f64::from(steps) / 3.0));
    let mut lines = Vec::new();
    for dir in [-1i32, 1] {
        for &r in &releases {
            lines.push(mk(steps, |s| PlanStep {
                dir,
                jump: 0,
                hook: i32::from(f64::from(s) < r),
                fire: 0,
                aim: at,
            }));
        }
        lines.push(mk(steps, |s| PlanStep {
            dir,
            jump: i32::from(s == 1),
            hook: i32::from(f64::from(s) < mid + 1.0),
            fire: 0,
            aim: at,
        }));
        lines.push(mk(steps, |s| PlanStep {
            dir,
            jump: 0,
            hook: i32::from(f64::from(s) < mid),
            fire: i32::from(f64::from(s) >= mid),
            aim: at,
        }));
    }
    lines
}

/// `frozenThrowWorthTrying(s)` (`throwLines.ts:41-44`).
pub fn frozen_throw_worth_trying(s: &ThrowSituation) -> bool {
    if s.me_frozen || !s.enemy_frozen || !s.enemy_alive {
        return false;
    }
    s.separation <= *crate::tuning::HOOK_LENGTH
}

/// `frozenThrowLines(steps, at)` (`throwLines.ts:46-60`).
pub fn frozen_throw_lines(steps: i32, at: f64) -> Vec<Vec<PlanStep>> {
    let mut lines = throw_lines(steps, at);
    let third = js::max(3.0, js::round(f64::from(steps) / 3.0));
    for dir in [-1i32, 0, 1] {
        for h in [2.0, 3.0, 5.0] {
            if h + 1.0 >= f64::from(steps) {
                continue;
            }
            lines.push(mk(steps, |s| PlanStep {
                dir,
                jump: i32::from(f64::from(s) == h - 1.0),
                hook: i32::from(f64::from(s) < h),
                fire: i32::from(f64::from(s) == h || f64::from(s) == h + 1.0),
                aim: at,
            }));
        }
        lines.push(mk(steps, |s| PlanStep {
            dir,
            jump: i32::from(s == 0 || f64::from(s) == third),
            hook: i32::from(f64::from(s) < third + 1.0),
            fire: 0,
            aim: at,
        }));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throw_lines_produces_12_lines_for_the_default_step_count() {
        let lines = throw_lines(9, 0.0);
        assert_eq!(lines.len(), 12);
        assert!(lines.iter().all(|l| l.len() == 9));
    }

    #[test]
    fn frozen_throw_lines_adds_the_frozen_specific_lines() {
        let lines = frozen_throw_lines(9, 0.0);
        // 12 (throwLines) + 3 dirs * (3 h-lines (h+1<9 -> h in {2,3,5}, 5+1=6<9 ok) + 1 third-line).
        assert_eq!(lines.len(), 12 + 3 * (3 + 1));
    }

    #[test]
    fn throw_worth_trying_requires_both_alive_unfrozen_and_in_range() {
        let s = ThrowSituation {
            separation: 100.0,
            enemy_hazard_nearness: 0.5,
            me_frozen: false,
            enemy_frozen: false,
            enemy_alive: true,
        };
        assert!(throw_worth_trying(&s));
        assert!(!throw_worth_trying(&ThrowSituation { me_frozen: true, ..s }));
        assert!(!throw_worth_trying(&ThrowSituation {
            enemy_hazard_nearness: 0.1,
            ..s
        }));
    }
}
