"""Exact multinomial bootstrap and percentiles, standard library only.

Both bootstraps in analysis.py resample counts from a fixed three-category
multinomial: in poststrat_test the paired difference d takes values in
{-1, 0, +1}, and in deficit_ci every question is delivered-correct, wrong-but-
recoverable, or absent from the candidate set. Neither needs general array
machinery, and neither should be unrunnable on a machine without numpy -- a
bootstrap that cannot be re-run is not a reproducible statistic.

Method. A multinomial draw is a chain of conditional binomials,

    x0 ~ Bin(n, p0),   x1 ~ Bin(n - x0, p1 / (1 - p0)),   x2 = n - x0 - x1,

and each binomial is drawn by inverse transform against a tabulated CDF. This
is exact, not a normal approximation. Two details keep it affordable: the
tables are truncated to +-8 standard deviations about the mean, where the
discarded tail mass is below 1e-15 and is renormalised away, and they are
cached by (m, p), which matters because the second stage's m changes with every
draw while its success probability does not.

Reproducibility. Draws depend on the `random.Random` stream, so the intervals
differ in the last decimal or two from a numpy-generated run of the same data.
That difference is Monte-Carlo error of the bootstrap itself, not disagreement
about the estimate; at 20,000 resamples it is a few hundredths of a point.
"""
import bisect
import math

__all__ = ["multinomial", "percentile", "mean", "var"]


def _binom_table(m, p):
    """Truncated CDF of Bin(m, p).

    Returns (lo, cdf) with cdf[i] = P(X <= lo + i), renormalised to 1 over the
    retained range.
    """
    if m <= 0 or p <= 0.0:
        return 0, [1.0]
    if p >= 1.0:
        return m, [1.0]

    mean_ = m * p
    sd = math.sqrt(m * p * (1.0 - p))
    lo = max(0, int(mean_ - 8.0 * sd) - 1)
    hi = min(m, int(mean_ + 8.0 * sd) + 1)

    # Start from the log pmf so that large m does not overflow, then walk
    # forward with pmf(k+1) = pmf(k) * (m-k)/(k+1) * p/(1-p).
    logp = (math.lgamma(m + 1) - math.lgamma(lo + 1) - math.lgamma(m - lo + 1)
            + lo * math.log(p) + (m - lo) * math.log1p(-p))
    v = math.exp(logp)
    ratio = p / (1.0 - p)

    cdf, s = [], 0.0
    for k in range(lo, hi + 1):
        s += v
        cdf.append(s)
        v *= (m - k) * ratio / (k + 1)

    total = cdf[-1]
    if total <= 0.0:                      # degenerate; fall back to the mean
        return int(round(mean_)), [1.0]
    return lo, [x / total for x in cdf]


def multinomial(rng, n, probs, size, cache=None):
    """`size` exact draws from Multinomial(n, probs) over three categories.

    Returns a list of (x0, x1, x2) tuples. `rng` is a random.Random.
    """
    if len(probs) != 3:
        raise ValueError("three categories expected, got %d" % len(probs))
    if cache is None:
        cache = {}

    p0, p1 = probs[0], probs[1]

    def table(m, p):
        key = (m, round(p, 12))
        t = cache.get(key)
        if t is None:
            t = _binom_table(m, p)
            cache[key] = t
        return t

    lo0, cdf0 = table(n, p0)
    q = p1 / (1.0 - p0) if p0 < 1.0 else 0.0
    q = min(1.0, max(0.0, q))

    out = []
    rand = rng.random
    for _ in range(size):
        x0 = lo0 + bisect.bisect_left(cdf0, rand())
        if x0 > n:
            x0 = n
        m = n - x0
        lo1, cdf1 = table(m, q)
        x1 = lo1 + bisect.bisect_left(cdf1, rand())
        if x1 > m:
            x1 = m
        out.append((x0, x1, m - x1))
    return out


def percentile(xs, qs):
    """Linear-interpolated percentiles, matching numpy's default method."""
    a = sorted(xs)
    n = len(a)
    if n == 0:
        return [float("nan")] * len(qs)
    out = []
    for q in qs:
        if n == 1:
            out.append(a[0])
            continue
        pos = (q / 100.0) * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        out.append(a[lo] + (pos - lo) * (a[hi] - a[lo]))
    return out


def mean(xs):
    return sum(xs) / len(xs)


def var(xs, ddof=1):
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


def _selftest():
    """Sampler must reproduce the multinomial's mean and covariance."""
    import random as _r
    rng = _r.Random(12345)
    n, p = 400, [0.55, 0.30, 0.15]
    draws = multinomial(rng, n, p, 40000)
    for i in range(3):
        col = [d[i] for d in draws]
        want_m = n * p[i]
        want_v = n * p[i] * (1 - p[i])
        got_m, got_v = mean(col), var(col)
        assert abs(got_m - want_m) < 0.06 * math.sqrt(want_v), \
            f"cat {i}: mean {got_m:.3f} vs {want_m:.3f}"
        assert abs(got_v - want_v) / want_v < 0.05, \
            f"cat {i}: var {got_v:.2f} vs {want_v:.2f}"
        assert all(0 <= x <= n for x in col)
    assert all(sum(d) == n for d in draws), "counts must sum to n"

    # covariance between categories is -n p_i p_j
    c0 = [d[0] for d in draws]
    c1 = [d[1] for d in draws]
    m0, m1 = mean(c0), mean(c1)
    cov = sum((a - m0) * (b - m1) for a, b in zip(c0, c1)) / (len(c0) - 1)
    want = -n * p[0] * p[1]
    assert abs(cov - want) / abs(want) < 0.08, f"cov {cov:.2f} vs {want:.2f}"

    # percentile must agree with the textbook definition on a known list
    assert percentile([1, 2, 3, 4], [0, 50, 100]) == [1, 2.5, 4]
    assert abs(percentile(list(range(101)), [2.5])[0] - 2.5) < 1e-9

    # degenerate inputs must not raise
    assert multinomial(rng, 0, [0.5, 0.3, 0.2], 3) == [(0, 0, 0)] * 3
    assert all(d == (5, 0, 0) for d in multinomial(rng, 5, [1.0, 0.0, 0.0], 3))
    print("stdlib_boot: all self-checks passed")


if __name__ == "__main__":
    _selftest()
