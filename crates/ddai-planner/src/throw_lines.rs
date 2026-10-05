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

const WALL_SWING_JUMPS: [f64; 2] = [4.0, 6.0];
const WALL_SWING_FLIP: f64 = 11.0;
const WALL_SWING_RELEASES: [f64; 3] = [18.0, 22.0, 26.0];

/// The tick at which each of `steps` plan steps starts (`start` of `wallSwingLines`/`airChainLines`): step `s` lasts
/// `step_ticks[min(s, len - 1)]` ticks.
fn step_starts(steps: i32, step_ticks: &[i32]) -> Vec<f64> {
    let mut start = Vec::with_capacity(steps.max(0) as usize);
    let mut t = 0.0;
    for s in 0..steps {
        start.push(t);
        t += f64::from(step_ticks[(s as usize).min(step_ticks.len() - 1)]);
    }
    start
}

/// `wallSwingLines(steps, stepTicks, at, wallDir)` (af49dfb `throwLines.ts`): hook the wall and swing off it toward the
/// victim -- the wayblock guard's throw into a freeze at the hall wall. `wall_dir` is the side of the wall (`-1`/`1`);
/// `0` (no wall), no steps or no step ticks give no lines.
pub fn wall_swing_lines(steps: i32, step_ticks: &[i32], at: f64, wall_dir: i32) -> Vec<Vec<PlanStep>> {
    if wall_dir == 0 || steps <= 0 || step_ticks.is_empty() {
        return Vec::new();
    }
    let start = step_starts(steps, step_ticks);
    let n = steps as usize;
    let covers = |s: usize, tick: f64| start[s] <= tick && tick < if s + 1 < n { start[s + 1] } else { f64::INFINITY };
    let mut jump_steps: Vec<usize> = Vec::new();
    for tick in WALL_SWING_JUMPS {
        if let Some(s) = (0..n).find(|&i| covers(i, tick))
            && !jump_steps.contains(&s)
        {
            jump_steps.push(s);
        }
    }
    let d = wall_dir.signum();
    let mut lines = Vec::new();
    for &j in &jump_steps {
        for r in WALL_SWING_RELEASES {
            lines.push(
                start
                    .iter()
                    .enumerate()
                    .map(|(s, &t)| PlanStep {
                        dir: if t < WALL_SWING_FLIP { d } else { -d },
                        jump: i32::from(s == j),
                        hook: i32::from(t < r),
                        fire: 0,
                        aim: at,
                    })
                    .collect(),
            );
        }
    }
    lines
}

const AIR_CHAIN_PLANS: [(f64, f64); 4] = [(6.0, 10.0), (8.0, 12.0), (8.0, 14.0), (11.0, 16.0)];
const AIR_CHAIN_JUMP_PLANS: [usize; 2] = [2, 3];

/// `airChainLines(steps, stepTicks, at, wallDir, airJump)` (af49dfb `throwLines.ts`): hook chains in the air along the wall
/// (a hook, a flip, a release, optionally one air jump at step 1).
pub fn air_chain_lines(steps: i32, step_ticks: &[i32], at: f64, wall_dir: i32, air_jump: bool) -> Vec<Vec<PlanStep>> {
    if wall_dir == 0 || steps <= 0 || step_ticks.is_empty() {
        return Vec::new();
    }
    let start = step_starts(steps, step_ticks);
    let d = wall_dir.signum();
    let line = |flip: f64, release: f64, jump_at: i64| -> Vec<PlanStep> {
        start
            .iter()
            .enumerate()
            .map(|(s, &t)| PlanStep {
                dir: if t < flip { d } else { -d },
                jump: i32::from(s as i64 == jump_at),
                hook: i32::from(t < release),
                fire: 0,
                aim: at,
            })
            .collect()
    };
    let mut lines: Vec<Vec<PlanStep>> = AIR_CHAIN_PLANS
        .iter()
        .map(|&(flip, release)| line(flip, release, -1))
        .collect();
    if air_jump && steps > 1 {
        for i in AIR_CHAIN_JUMP_PLANS {
            lines.push(line(AIR_CHAIN_PLANS[i].0, AIR_CHAIN_PLANS[i].1, 1));
        }
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
    fn wall_swing_lines_cover_the_jump_steps_and_releases() {
        // 9 steps of 3 ticks: step s starts at 3 * s; tick 4 is in step 1, tick 6 in step 2.
        let ticks = [3; 9];
        let lines = wall_swing_lines(9, &ticks, 0.5, -1);
        assert_eq!(lines.len(), 2 * 3);
        assert!(lines.iter().all(|l| l.len() == 9));
        // First line: jump at step 1, hook released at tick 18 (steps 0..=5 hold it), flip to +1 at tick 11 (step 4).
        let first = &lines[0];
        assert_eq!(first.iter().position(|s| s.jump == 1), Some(1));
        assert_eq!(first.iter().filter(|s| s.hook == 1).count(), 6);
        assert_eq!(first[0].dir, -1);
        assert_eq!(first[3].dir, -1);
        assert_eq!(first[4].dir, 1);
        assert!(first.iter().all(|s| s.fire == 0 && s.aim == 0.5));
        // No wall / no steps / no step ticks: nothing.
        assert!(wall_swing_lines(9, &ticks, 0.0, 0).is_empty());
        assert!(wall_swing_lines(0, &ticks, 0.0, 1).is_empty());
        assert!(wall_swing_lines(9, &[], 0.0, 1).is_empty());
        // The mirrored wall mirrors the directions.
        let right = wall_swing_lines(9, &ticks, 0.5, 1);
        assert_eq!(right[0][0].dir, 1);
        assert_eq!(right[0][4].dir, -1);
    }

    #[test]
    fn air_chain_lines_offer_four_chains_and_two_air_jump_variants() {
        let ticks = [3; 9];
        assert_eq!(air_chain_lines(9, &ticks, 0.0, 1, true).len(), 4 + 2);
        assert_eq!(air_chain_lines(9, &ticks, 0.0, 1, false).len(), 4);
        assert_eq!(air_chain_lines(1, &ticks, 0.0, 1, true).len(), 4);
        assert!(air_chain_lines(9, &ticks, 0.0, 0, true).is_empty());
        let with_jump = &air_chain_lines(9, &ticks, 0.0, 1, true)[4];
        assert_eq!(with_jump.iter().position(|s| s.jump == 1), Some(1));
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
