"""Rust engine bridge: batch nu-SVR fits via the bundled native extension.

All preprocessing (R RNG permutation draws, z-scores) happens in Python so
the solver inputs are bit-identical across engines; only the fits run
natively (rayon-parallel, deterministic for any thread count).  The nu
selection / normalisation / P-value logic stays in core.py, shared with the
pure-Python engines.
"""

from __future__ import annotations

import numpy as np


def run_fits_rust(X: np.ndarray, targets: np.ndarray, n_perm: int,
                  n_samples: int, threads: int) -> dict:
    """Fit all (target, nu) combos natively; return the fits dict keyed like
    the Python path: (0, i, ni) for permutation i, (1, j, ni) for sample j."""
    from . import _native

    X = np.ascontiguousarray(X, dtype=np.float64)
    targets = np.ascontiguousarray(targets, dtype=np.float64)
    d = X.shape[1]
    n_targets = targets.shape[0]
    flat = _native.run_fits(X, targets, max(int(threads), 1))
    W = np.asarray(flat, dtype=np.float64).reshape(n_targets, 3, d)
    fits: dict = {}
    for t in range(n_targets):
        kind = 0 if t < n_perm else 1
        idx = t if t < n_perm else t - n_perm
        for ni in range(3):
            fits[(kind, idx, ni)] = W[t, ni]
    return fits
