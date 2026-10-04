"""Small statistics helpers shared by the E-017 scripts (Wilson interval, exact McNemar)."""
import math

Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """95% Wilson score interval (low, high) for k successes of n; (0, 1) for n == 0."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    d = 1 + z2 / n
    c = p + z2 / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def pct(k, n):
    """'12.3% [10.1; 14.8] (k/n)' formatted with the Wilson interval."""
    lo, hi = wilson(k, n)
    p = 100.0 * k / n if n else float("nan")
    return f"{p:.1f}% [{100*lo:.1f}; {100*hi:.1f}] ({k}/{n})"


def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar p-value for the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)
