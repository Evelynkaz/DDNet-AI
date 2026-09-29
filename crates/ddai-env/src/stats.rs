//! Statistics used by every arena report: Wilson score interval, nearest-rank percentiles,
//! W:L:D:T tallies.

use serde::{Deserialize, Serialize};

/// z for a two-sided 95% interval (`Phi^-1(0.975)`).
pub const Z95: f64 = 1.959_963_984_540_054;

/// 95% Wilson score interval `(low, high)` for `successes` out of `n` Bernoulli trials.
/// `n == 0` gives the vacuous `(0, 1)`.
pub fn wilson95(successes: f64, n: f64) -> (f64, f64) {
    if n <= 0.0 {
        return (0.0, 1.0);
    }
    let p = successes / n;
    let z2 = Z95 * Z95;
    let denom = 1.0 + z2 / n;
    let center = p + z2 / (2.0 * n);
    let margin = Z95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    (
        ((center - margin) / denom).max(0.0),
        ((center + margin) / denom).min(1.0),
    )
}

/// Nearest-rank percentile of an *ascending-sorted* slice (`p` in `0..=100`), the same rule as
/// the phase-0 harness's `pct` (`ceil(p/100 * n) - 1`). Empty input gives `None`.
pub fn percentile_sorted<T: Copy>(sorted: &[T], p: f64) -> Option<T> {
    if sorted.is_empty() {
        return None;
    }
    let idx = ((p / 100.0 * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1;
    Some(sorted[idx])
}

/// Sorts a copy and takes the percentile; see [`percentile_sorted`].
pub fn percentile_u32(values: &[u32], p: f64) -> Option<u32> {
    let mut v = values.to_vec();
    v.sort_unstable();
    percentile_sorted(&v, p)
}

/// Game outcome from the focal player's (slot 0) point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameResult {
    /// The focal player won.
    W,
    /// The focal player lost.
    L,
    /// Both went out on the same tick.
    D,
    /// Nobody went out within the tick limit.
    T,
}

/// W:L:D:T counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tally {
    pub w: u32,
    pub l: u32,
    pub d: u32,
    pub t: u32,
}

impl Tally {
    pub fn record(&mut self, r: GameResult) {
        match r {
            GameResult::W => self.w += 1,
            GameResult::L => self.l += 1,
            GameResult::D => self.d += 1,
            GameResult::T => self.t += 1,
        }
    }

    pub fn total(&self) -> u32 {
        self.w + self.l + self.d + self.t
    }

    /// Games with a decisive-or-draw outcome (everything but timeouts): the denominator of the
    /// headline win rate (`PLAN.md` §0: "W / (W + L + D)").
    pub fn decided(&self) -> u32 {
        self.w + self.l + self.d
    }

    /// `W / (W + L + D)` with its 95% Wilson interval; `None` when no game was decided.
    pub fn win_rate(&self) -> Option<(f64, f64, f64)> {
        let n = f64::from(self.decided());
        (n > 0.0).then(|| {
            let (lo, hi) = wilson95(f64::from(self.w), n);
            (f64::from(self.w) / n, lo, hi)
        })
    }

    /// `W / (W + L + D + T)`: the stricter rate that counts timeouts against the player.
    pub fn win_rate_all(&self) -> Option<(f64, f64, f64)> {
        let n = f64::from(self.total());
        (n > 0.0).then(|| {
            let (lo, hi) = wilson95(f64::from(self.w), n);
            (f64::from(self.w) / n, lo, hi)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_matches_reference_values() {
        // Reference values: the closed form evaluated independently (Python), agreeing to 4 digits
        // with the published Wilson bounds 0.8389 (20/20), 0.1611 (0/20), 0.4038/0.5962 (50/100).
        let (lo, hi) = wilson95(20.0, 20.0);
        assert!((lo - 0.838_874_8).abs() < 1e-6, "{lo}");
        assert!((hi - 1.0).abs() < 1e-12);
        let (lo, hi) = wilson95(0.0, 20.0);
        assert!(lo.abs() < 1e-12);
        assert!((hi - 0.161_125_2).abs() < 1e-6, "{hi}");
        let (lo, hi) = wilson95(50.0, 100.0);
        assert!((lo - 0.403_831_5).abs() < 1e-6, "{lo}");
        assert!((hi - 0.596_168_5).abs() < 1e-6, "{hi}");
        assert_eq!(wilson95(0.0, 0.0), (0.0, 1.0));
    }

    #[test]
    fn percentile_is_nearest_rank_like_the_harness() {
        let v: Vec<u32> = (1..=100).collect();
        assert_eq!(percentile_u32(&v, 50.0), Some(50));
        assert_eq!(percentile_u32(&v, 99.0), Some(99));
        assert_eq!(percentile_u32(&v, 100.0), Some(100));
        assert_eq!(percentile_u32(&v, 0.0), Some(1));
        assert_eq!(percentile_u32(&[], 50.0), None);
        assert_eq!(percentile_u32(&[7], 99.0), Some(7));
    }

    #[test]
    fn tally_rates() {
        let mut t = Tally::default();
        for r in [GameResult::W, GameResult::W, GameResult::L, GameResult::T] {
            t.record(r);
        }
        assert_eq!(t.total(), 4);
        assert_eq!(t.decided(), 3);
        let (p, lo, hi) = t.win_rate().unwrap();
        assert!((p - 2.0 / 3.0).abs() < 1e-12 && lo < p && p < hi);
        let (p_all, ..) = t.win_rate_all().unwrap();
        assert!((p_all - 0.5).abs() < 1e-12);
        assert!(Tally::default().win_rate().is_none());
    }
}
