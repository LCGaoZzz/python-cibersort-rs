"""Shared fixtures and paths for the regression suite.

The repository ships a deterministic synthetic fixture and numeric oracle
outputs. It does not redistribute the original CIBERSORT R script or LM22.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(HERE, "data")
HEX_DIR = os.path.join(HERE, "data_hex")
FIXTURE_DIR = os.path.join(HERE, "fixtures", "permutation_safe")

SIG = os.path.join(FIXTURE_DIR, "signature_matrix.tsv")
MIX_EXACT = os.path.join(FIXTURE_DIR, "mixture.tsv")
MIX_NOISY = os.path.join(FIXTURE_DIR, "mixture_noisy.tsv")

# tag -> (mixture, perm, QN, absolute, abs_method, seed)
ORACLES = {
    "exact_rel_perm0":               (MIX_EXACT, 0,   False, False, "sig.score", 42),
    "noisy_rel_perm100_seed1":       (MIX_NOISY, 100, False, False, "sig.score", 1),
    "noisy_rel_perm100_seed42":      (MIX_NOISY, 100, False, False, "sig.score", 42),
    "noisy_rel_perm100_seed20260814": (MIX_NOISY, 100, False, False, "sig.score", 20260814),
    "noisy_qn_rel_perm100":          (MIX_NOISY, 100, True,  False, "sig.score", 42),
    "noisy_abs_perm100":             (MIX_NOISY, 100, False, True,  "sig.score", 42),
    "noisy_absnosum_perm100":        (MIX_NOISY, 100, False, True,  "no.sumto1", 42),
}

GATE = 1e-6  # max-abs-error gate for proportions/Correlation/RMSE/Absolute score


def read_hex_matrix(path: str) -> np.ndarray:
    """Read a headerless matrix of R %a hex floats (exact round-trip)."""
    df = pd.read_csv(path, sep="\t", header=None)
    return df.map(lambda v: float.fromhex(str(v).strip())).to_numpy(np.float64)


def read_oracle(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0, float_precision="round_trip")
    return df.astype(np.float64)


def compare_to_oracle(table: pd.DataFrame, oracle: pd.DataFrame):
    """Return (max_err_excluding_P, p_value_identical, per_column_max_err)."""
    assert list(table.columns) == list(oracle.columns), (
        f"column mismatch:\n{list(table.columns)}\nvs\n{list(oracle.columns)}")
    assert list(table.index) == list(oracle.index)
    got = table.to_numpy(np.float64)
    exp = oracle.to_numpy(np.float64)
    err = np.abs(got - exp)
    pcol = list(table.columns).index("P-value")
    p_identical = bool(np.array_equal(got[:, pcol], exp[:, pcol]))
    max_err = float(np.max(np.delete(err, pcol, axis=1)))
    per_col = {c: float(err[:, i].max()) for i, c in enumerate(table.columns)}
    return max_err, p_identical, per_col
