"""CoreAlg regression: 3-nu SVR sweep + RMSE selection + weight normalisation."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from python_cibersort.core import _CoreAlg

from conftest import DATA_DIR, HEX_DIR, read_hex_matrix

MODES = (  # reference tag -> (absolute, abs_method)
    ("rel", (False, "sig.score")),
    ("sig_score", (True, "sig.score")),
    ("no_sumto1", (True, "no.sumto1")),
)


def _load_xy():
    X = read_hex_matrix(os.path.join(HEX_DIR, "corealg_X_hex.tsv"))
    y = read_hex_matrix(os.path.join(HEX_DIR, "corealg_y1_hex.tsv")).ravel()
    return X, y


def _load_ref(tag):
    df = pd.read_csv(os.path.join(DATA_DIR, f"corealg_ref_{tag}.tsv"), sep="\t")
    stats = dict(zip(df["stat"], df["value"].astype(float)))
    w = np.array([stats[f"w_CellType_{i:02d}"] for i in range(1, 23)])
    return w, stats["mix_rmse"], stats["mix_r"]


class TestCoreAlg:
    @pytest.mark.parametrize("engine", ["libsvm"])
    @pytest.mark.parametrize("tag,mode", MODES)
    def test_matches_r_corealg(self, engine, tag, mode):
        X, y = _load_xy()
        absolute, abs_method = mode
        alg = _CoreAlg(X, engine=engine)
        raws = [alg.fit_one(y.copy(), nu) for nu in (0.25, 0.5, 0.75)]
        w, mix_rmse, mix_r, best = alg.evaluate(y, raws, absolute, abs_method)
        w_ref, rmse_ref, r_ref = _load_ref(tag)
        assert np.max(np.abs(w - w_ref)) < 1e-12
        assert abs(mix_rmse - rmse_ref) < 1e-12
        assert abs(mix_r - r_ref) < 1e-12

    def test_best_nu_selection_by_rmse(self):
        """The RMSE-minimising nu must match R's (which.min over nusvm)."""
        X, y = _load_xy()
        alg = _CoreAlg(X, engine="libsvm")
        raws = [alg.fit_one(y.copy(), nu) for nu in (0.25, 0.5, 0.75)]
        _, rmse, _, best = alg.evaluate(y, raws, False, "sig.score")
        rmses = []
        for wr in raws:
            wt = np.where(wr < 0, 0.0, wr)
            k = X @ (wt / wt.sum())
            rmses.append(float(np.sqrt(np.mean((k - y) ** 2))))
        assert best == int(np.argmin(rmses))
