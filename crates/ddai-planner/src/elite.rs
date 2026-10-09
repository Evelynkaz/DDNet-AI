//! The final CEM elite set's first plan step, as a soft imitation target (task 8.2).
//!
//! After the last CEM iteration of a fixed-iteration decision ([`crate::planner::Planner::decide`])
//! the `elite` best-scoring plans are the planner's own idea of "what a good first move looks
//! like"; the chosen action is a single point of that set (and passes through post-processing:
//! stay-hysteresis, hook polish, escape bias, shield). Their *frequencies* are a richer teaching
//! signal than the argmax: a 50/50 split between "run left" and "run right" is a very different
//! fact from "run left, everybody agrees".
//!
//! Pure data plus one pure function, so the planner (and its TS-parity path) is unaffected: the
//! only hook is one assignment in `decide_once` that copies a summary into
//! [`crate::planner::Planner::last_elite`]; nothing reads it back into the search.

use crate::planner::PlanStep;

/// Raw (un-smoothed) statistics of the first step of the elite plans of the last CEM iteration.
///
/// Fractions are plain counts over the elites, *not* the smoothed `0.1 + 0.8 * f` / `0.05 + 0.9 * f`
/// probabilities the CEM itself samples from (`Planner::refit`). Angles are in the planner's
/// screen convention (`atan2(dy, dx)`, `y` grows downward).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EliteFirstStep {
    /// How many elite plans the statistics are over.
    pub elites: u32,
    /// Fraction of the elites whose first step runs left / stops / runs right (sums to 1).
    pub left: f32,
    pub stop: f32,
    pub right: f32,
    /// Fraction of the elites holding jump / hook / fire in their first step.
    pub jump: f32,
    pub hook: f32,
    pub fire: f32,
    /// Circular mean of the *absolute* aim angle of the first step (screen convention).
    pub aim_mean: f32,
    /// Circular standard deviation of that aim (`sqrt(-2 ln R)`, `R` the mean resultant length),
    /// radians; `0` when every elite aims the same way.
    pub aim_spread: f32,
}

/// Summarises the first step of `elites` (each plan's `[0]`). `track_aim` says whether a plan
/// step's `aim` is an offset from the bearing to the enemy (`aim_at`, the planner's `trackAim`)
/// or an absolute angle. Returns `None` for an empty set or plans without a first step.
pub fn summarize_first_step(elites: &[Vec<PlanStep>], track_aim: bool, aim_at: f64) -> Option<EliteFirstStep> {
    let steps: Vec<PlanStep> = elites.iter().filter_map(|p| p.first().copied()).collect();
    if steps.is_empty() {
        return None;
    }
    let n = steps.len() as f64;
    let (mut left, mut right, mut stop) = (0.0f64, 0.0f64, 0.0f64);
    let (mut jump, mut hook, mut fire) = (0.0f64, 0.0f64, 0.0f64);
    let (mut ax, mut ay) = (0.0f64, 0.0f64);
    for s in &steps {
        match s.dir {
            -1 => left += 1.0,
            1 => right += 1.0,
            _ => stop += 1.0,
        }
        jump += f64::from(s.jump != 0);
        hook += f64::from(s.hook != 0);
        fire += f64::from(s.fire != 0);
        let abs = if track_aim { aim_at + s.aim } else { s.aim };
        // libm-census: f64 sin/cos have no ddai-libm port yet (the elite's mean direction is a soft training target for the fly, not a live
        // decision; D-127 lists it)
        ax += abs.cos();
        ay += abs.sin();
    }
    let (mx, my) = (ax / n, ay / n);
    let resultant = (mx * mx + my * my).sqrt().clamp(1e-12, 1.0);
    Some(EliteFirstStep {
        elites: steps.len() as u32,
        left: (left / n) as f32,
        stop: (stop / n) as f32,
        right: (right / n) as f32,
        jump: (jump / n) as f32,
        hook: (hook / n) as f32,
        fire: (fire / n) as f32,
        aim_mean: ddai_libm::atan2(my, mx) as f32,
        aim_spread: (-2.0 * ddai_libm::log(resultant)).max(0.0).sqrt() as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(dir: i32, jump: i32, hook: i32, fire: i32, aim: f64) -> PlanStep {
        PlanStep {
            dir,
            jump,
            hook,
            fire,
            aim,
        }
    }

    #[test]
    fn frequencies_are_plain_counts_over_the_elites() {
        let elites = vec![
            vec![step(-1, 1, 0, 0, 0.0), step(1, 0, 0, 0, 0.0)],
            vec![step(-1, 0, 1, 0, 0.0)],
            vec![step(0, 0, 1, 1, 0.0)],
            vec![step(1, 1, 1, 0, 0.0)],
        ];
        let e = summarize_first_step(&elites, false, 0.0).unwrap();
        assert_eq!(e.elites, 4);
        assert_eq!((e.left, e.stop, e.right), (0.5, 0.25, 0.25));
        assert_eq!((e.jump, e.hook, e.fire), (0.5, 0.75, 0.25));
    }

    #[test]
    fn aim_is_the_circular_mean_and_spread_is_zero_when_all_agree() {
        let agree = vec![vec![step(0, 0, 0, 0, 1.0)]; 5];
        let e = summarize_first_step(&agree, false, 0.0).unwrap();
        assert!((e.aim_mean - 1.0).abs() < 1e-6);
        assert!(e.aim_spread < 1e-3, "spread {}", e.aim_spread);

        // +-0.4 rad around 3.0 wraps across pi: the mean must stay near 3.0, not near 0.
        let wrap = vec![vec![step(0, 0, 0, 0, 3.0 + 0.4)], vec![step(0, 0, 0, 0, 3.0 - 0.4)]];
        let e = summarize_first_step(&wrap, false, 0.0).unwrap();
        assert!((e.aim_mean - 3.0).abs() < 1e-5, "mean {}", e.aim_mean);
        assert!(e.aim_spread > 0.3 && e.aim_spread < 0.5, "spread {}", e.aim_spread);
    }

    #[test]
    fn track_aim_adds_the_bearing_to_the_enemy() {
        let elites = vec![vec![step(0, 0, 0, 0, 0.25)]];
        let e = summarize_first_step(&elites, true, 1.0).unwrap();
        assert!((e.aim_mean - 1.25).abs() < 1e-6);
        let e = summarize_first_step(&elites, false, 1.0).unwrap();
        assert!((e.aim_mean - 0.25).abs() < 1e-6);
    }

    #[test]
    fn empty_input_gives_none() {
        assert!(summarize_first_step(&[], false, 0.0).is_none());
        assert!(summarize_first_step(&[vec![]], false, 0.0).is_none());
    }
}
