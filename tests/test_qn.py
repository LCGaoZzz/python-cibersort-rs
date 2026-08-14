"""Quantile-normalisation regression vs preprocessCore::normalize.quantiles."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from python_cibersort.qn import quantile_normalize

from conftest import DATA_DIR, HEX_DIR, read_hex_matrix

CASES = (("1", (120, 5)), ("2", (333, 7)), ("3", (200, 4)))


class TestQuantileNormalize:
    @pytest.mark.parametrize("tag,shape", CASES)
    def test_bitexact_vs_hex_reference(self, tag, shape):
        qin = read_hex_matrix(os.path.join(HEX_DIR, f"qn_in{tag}_hex.tsv"))
        qref = read_hex_matrix(os.path.join(HEX_DIR, f"qn_out{tag}_hex.tsv"))
        assert qin.shape == shape
        out = quantile_normalize(qin)
        np.testing.assert_array_equal(out, qref)  # bit-exact

    @pytest.mark.parametrize("tag,shape", CASES)
    def test_decimal_17g_reference(self, tag, shape):
        qin = pd.read_csv(os.path.join(DATA_DIR, f"qn_in{tag}.tsv"),
                          sep="\t", header=None,
                          float_precision="round_trip").to_numpy(np.float64)
        qref = pd.read_csv(os.path.join(DATA_DIR, f"qn_out{tag}.tsv"),
                           sep="\t", header=None,
                           float_precision="round_trip").to_numpy(np.float64)
        assert qin.shape == shape
        out = quantile_normalize(qin)
        np.testing.assert_array_equal(out, qref)  # %.17g round-trips exactly

    def test_ties_averaged_like_get_ranks(self):
        # preprocessCore get_ranks: tied group [i..k] -> rank (i+k+2)/2 (1-based)
        Y = np.array([[5.0, 1.0], [1.0, 3.0], [1.0, 2.0], [3.0, 4.0]])
        out = quantile_normalize(Y)
        # column 0: values 1,1 tie at ranks 1,2 -> 1.5 -> frac .5 > .4 ->
        # (target[0]+target[1])/2; target rows from sorted columns / 2
        t = (np.sort(Y[:, 0]) + np.sort(Y[:, 1])) / 2.0
        assert out[1, 0] == out[2, 0] == pytest.approx((t[0] + t[1]) / 2.0, rel=0, abs=0)
        assert out[0, 0] == t[3]
        assert out[3, 0] == t[2]


@pytest.mark.skipif(__import__("importlib").util.find_spec("python_cibersort._native") is None,
                    reason="native extension not built")
class TestRustQuantileNormalize:
    @pytest.mark.parametrize("tag,shape", CASES)
    def test_bitexact_vs_hex_reference(self, tag, shape):
        from python_cibersort import _native
        qin = read_hex_matrix(os.path.join(HEX_DIR, f"qn_in{tag}_hex.tsv"))
        qref = read_hex_matrix(os.path.join(HEX_DIR, f"qn_out{tag}_hex.tsv"))
        rows, cols = qin.shape
        out = np.asarray(_native.quantile_normalize(
            qin.ravel().tolist(), rows, cols), dtype=np.float64).reshape(rows, cols)
        np.testing.assert_array_equal(out, qref)  # bit-exact
