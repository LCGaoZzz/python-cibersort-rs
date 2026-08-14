"""RNG regression: R's Mersenne-Twister + set.seed + sample() + runif()."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from python_cibersort.rrnd import RRng

from conftest import DATA_DIR, HEX_DIR


class TestSample:
    ref = pd.read_csv(os.path.join(DATA_DIR, "rng_reference.tsv"), sep="\t")

    @pytest.mark.parametrize("case", sorted(ref["case"].unique()))
    def test_sample_no_replace_matches_r(self, case):
        grp = self.ref[self.ref["case"] == case].sort_values("pos")
        seed = int(grp["seed"].iloc[0])
        n = int(grp["n"].iloc[0])
        k = int(grp["k"].iloc[0])
        rng = RRng(seed)
        draws = rng.sample_no_replace(n, k)
        expected = grp["value"].to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(draws, expected)

    def test_sequential_samples_match_r_stream(self):
        """R: set.seed(42); sample(5940,330); sample(5940,330) — second draw
        must continue the SAME stream (doPerm semantics)."""
        rng = RRng(42)
        first = rng.sample_no_replace(5940, 330)
        second = rng.sample_no_replace(5940, 330)
        rng2 = RRng(42)
        assert np.array_equal(first, rng2.sample_no_replace(5940, 330))
        assert not np.array_equal(first, second)


class TestRunif:
    def test_runif_bitexact_hex(self):
        ref = pd.read_csv(os.path.join(HEX_DIR, "unif_reference_hex.tsv"), sep="\t")
        for seed, grp in ref.groupby("seed"):
            rng = RRng(int(seed))
            vals = rng.runif(len(grp))
            exp = np.array([float.fromhex(v)
                            for v in grp.sort_values("pos")["value"]])
            np.testing.assert_array_equal(vals, exp)  # bit-exact

    def test_runif_decimal_17g(self):
        ref = pd.read_csv(os.path.join(DATA_DIR, "unif_reference.tsv"), sep="\t",
                          float_precision="round_trip")
        for seed, grp in ref.groupby("seed"):
            rng = RRng(int(seed))
            vals = rng.runif(len(grp))
            exp = grp.sort_values("pos")["value"].to_numpy(np.float64)
            np.testing.assert_array_equal(vals, exp)  # %.17g round-trips exactly


@pytest.mark.skipif(__import__("importlib").util.find_spec("python_cibersort._native") is None,
                    reason="native extension not built")
class TestRustRng:
    """The Rust port of R's RNG must be bit-exact too (used for fast draws)."""

    ref = pd.read_csv(os.path.join(DATA_DIR, "rng_reference.tsv"), sep="\t")

    @pytest.mark.parametrize("case", sorted(ref["case"].unique()))
    def test_sample_no_replace_matches_r(self, case):
        from python_cibersort import _native
        grp = self.ref[self.ref["case"] == case].sort_values("pos")
        draws = np.asarray(
            _native.sample_no_replace(int(grp["seed"].iloc[0]),
                                           int(grp["n"].iloc[0]),
                                           int(grp["k"].iloc[0])), dtype=np.int64)
        np.testing.assert_array_equal(draws, grp["value"].to_numpy(np.int64))

    def test_runif_bitexact(self):
        from python_cibersort import _native
        ref = pd.read_csv(os.path.join(HEX_DIR, "unif_reference_hex.tsv"), sep="\t")
        for seed, grp in ref.groupby("seed"):
            vals = np.asarray(_native.runif(int(seed), len(grp)))
            exp = np.array([float.fromhex(v)
                            for v in grp.sort_values("pos")["value"]])
            np.testing.assert_array_equal(vals, exp)

    def test_perm_indices_matches_python_stream(self):
        """perm_indices(seed,n,k,p) == p sequential Python RRng draws (one stream)."""
        from python_cibersort import _native
        seed, n, k, p = 42, 5940, 330, 5
        native = np.asarray(_native.perm_indices(seed, n, k, p),
                            dtype=np.int64).reshape(p, k)
        rng = RRng(seed)
        for i in range(p):
            np.testing.assert_array_equal(native[i], rng.sample_no_replace(n, k))
