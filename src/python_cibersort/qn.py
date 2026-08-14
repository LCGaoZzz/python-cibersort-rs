"""Bit-exact reimplementation of preprocessCore::normalize.quantiles.

Verified bit-exact against the CRAN Windows binary of preprocessCore
1.72.0 (a) by direct ctypes calls into preprocessCore.dll and (b) by
in-memory comparison inside R (0/333 target rows differ).  Both the
classic (qnorm_c) and NA-handling (qnorm_c_handleNA) entries reduce to
the same arithmetic for NA-free, equal-length input:

1. Target: for each column j (ascending), sort ascending, accumulate
   ``row_mean[i] += sorted[i] / cols`` in plain f64 (divsd/addsd).
2. Distribute: average ranks with tie handling
   (``get_ranks``: tied group [i..k] gets (i+k+2)*0.5), map back to
   original positions; fractional part > 0.4 interpolates as
   ``(target[fl-1] + target[fl]) * 0.5``, else ``target[fl-1]``.

NOTE: comparisons against decimal TSV reference files show ~1-ULP
scatter — that is a decimal round-trip artifact of the file, not the
algorithm; in-memory comparison is exact.

No NA handling (matches qnorm_c, which errors on NA upstream).
"""

from __future__ import annotations

import numpy as np


def quantile_normalize(Y: np.ndarray) -> np.ndarray:
    """normalize.quantiles(Y) — column-wise quantile normalisation."""
    Y = np.asarray(Y, dtype=np.float64)
    rows, cols = Y.shape
    # 1) target distribution, j ascending, divide-then-accumulate (f64)
    row_mean = np.zeros(rows, dtype=np.float64)
    for j in range(cols):
        row_mean += np.sort(Y[:, j]) / cols

    # 2) distribute target back with averaged ranks for ties
    out = np.empty_like(Y)
    for j in range(cols):
        col = Y[:, j]
        order = np.argsort(col, kind="quicksort")  # tie order irrelevant
        sv = col[order]
        ranks = np.empty(rows, dtype=np.float64)
        i = 0
        while i < rows:
            k = i
            while k < rows - 1 and sv[k] == sv[k + 1]:
                k += 1
            if k != i:
                ranks[i : k + 1] = (i + k + 2) * 0.5
            else:
                ranks[i] = i + 1
            i = k + 1
        fl = np.floor(ranks).astype(np.int64)  # floor(rank), 1-based
        frac = ranks - fl
        lo = row_mean[fl - 1]
        hi = row_mean[np.minimum(fl, rows - 1)]  # only used when frac > 0.4
        vals = np.where(frac > 0.4, (lo + hi) * 0.5, lo)
        out[order, j] = vals
    return out
