//! The statistics of the ES pilot: rank shaping, the exact McNemar test for paired outcomes, Wilson intervals.

pub use ddai_env::stats::wilson95;

/// Centred rank transform (Salimans et al. 2017, §3.1): the `n` values get ranks `0..n-1` (ties share the mean rank, so a tie
/// never turns into an arbitrary order), mapped linearly onto `[-0.5, 0.5]`. A single value maps to `0`.
pub fn centered_ranks(fitness: &[f32]) -> Vec<f32> {
    let n = fitness.len();
    if n <= 1 {
        return vec![0.0; n];
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| fitness[a].total_cmp(&fitness[b]).then(a.cmp(&b)));
    let mut rank = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && fitness[order[j + 1]] == fitness[order[i]] {
            j += 1;
        }
        let mean = (i + j) as f32 / 2.0;
        for &k in &order[i..=j] {
            rank[k] = mean;
        }
        i = j + 1;
    }
    rank.iter().map(|r| r / (n - 1) as f32 - 0.5).collect()
}

/// Exact two-sided McNemar test on the discordant pairs: `b` pairs where only the first condition succeeded, `c` where only the
/// second did. The p-value is `min(1, 2 * P(X <= min(b, c)))` for `X ~ Binomial(b + c, 1/2)`.
pub fn mcnemar_exact(b: u32, c: u32) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    // P(X <= k) by the stable recurrence of the binomial pmf in log space.
    let ln_half_n = f64::from(n) * 0.5f64.ln();
    let mut ln_coeff = 0.0f64; // ln C(n, 0)
    let mut tail = 0.0f64;
    for i in 0..=k {
        if i > 0 {
            ln_coeff += (f64::from(n - i + 1)).ln() - f64::from(i).ln();
        }
        tail += (ln_coeff + ln_half_n).exp();
    }
    (2.0 * tail).min(1.0)
}

/// The paired comparison of two boolean outcomes over the same items: `(only a, only b, McNemar p)`.
pub fn paired(a: &[bool], b: &[bool]) -> (u32, u32, f64) {
    assert_eq!(a.len(), b.len(), "paired outcomes need the same items");
    let only_a = a.iter().zip(b).filter(|(x, y)| **x && !**y).count() as u32;
    let only_b = a.iter().zip(b).filter(|(x, y)| !**x && **y).count() as u32;
    (only_a, only_b, mcnemar_exact(only_a, only_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_ranks_span_minus_half_to_half_and_share_ties() {
        let r = centered_ranks(&[3.0, 1.0, 2.0]);
        assert_eq!(r, vec![0.5, -0.5, 0.0]);
        // Ties get the mean of their ranks, whatever their order in the input.
        let r = centered_ranks(&[1.0, 1.0, 5.0, 1.0]);
        assert!((r[0] - r[1]).abs() < 1e-7 && (r[1] - r[3]).abs() < 1e-7);
        assert!((r[0] - (1.0 / 3.0 - 0.5)).abs() < 1e-6, "{r:?}");
        assert_eq!(r[2], 0.5);
        assert!((r.iter().sum::<f32>()).abs() < 1e-6, "centred");
        assert_eq!(centered_ranks(&[7.0]), vec![0.0]);
        assert!(centered_ranks(&[]).is_empty());
        // A monotone transform of the fitness changes nothing.
        assert_eq!(
            centered_ranks(&[0.1, 5.0, -3.0, 2.0]),
            centered_ranks(&[1.0, 500.0, -300.0, 20.0])
        );
    }

    #[test]
    fn mcnemar_matches_reference_values() {
        // Reference: scipy.stats.binomtest(k, n, 0.5).pvalue (two-sided).
        assert!((mcnemar_exact(0, 0) - 1.0).abs() < 1e-12);
        assert!((mcnemar_exact(5, 5) - 1.0).abs() < 1e-12);
        assert!((mcnemar_exact(0, 10) - 2.0 / 1024.0).abs() < 1e-12);
        assert!(
            (mcnemar_exact(3, 12) - 0.035_156_25).abs() < 1e-9,
            "{}",
            mcnemar_exact(3, 12)
        );
        assert!((mcnemar_exact(12, 3) - mcnemar_exact(3, 12)).abs() < 1e-15, "symmetric");
        // Large counts do not overflow.
        let p = mcnemar_exact(400, 520);
        assert!(p > 0.0 && p < 0.001, "{p}");
    }

    #[test]
    fn paired_counts_the_discordant_items() {
        let a = [true, true, false, false, true];
        let b = [true, false, false, true, false];
        let (oa, ob, p) = paired(&a, &b);
        assert_eq!((oa, ob), (2, 1));
        assert!((p - 1.0).abs() < 1e-12);
    }
}
