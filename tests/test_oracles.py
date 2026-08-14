"""End-to-end oracle regression: python-cibersort vs unmodified R CIBERSORT v1.04.

Gate (task requirement):
  * cell proportions, Correlation, RMSE, Absolute score: max abs error <= 1e-6
  * perm=100 P-value: exactly equal to the oracle for the matching seed
  * fixed seed => bit-identical results, for any thread count
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from python_cibersort import cibersort

from conftest import (FIXTURE_DIR, GATE, ORACLES, SIG,
                      compare_to_oracle, read_oracle)

ENGINES = ["libsvm", "rust"]


def _oracle_path(tag):
    return os.path.join(FIXTURE_DIR, f"oracle_{tag}.tsv")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("tag", list(ORACLES))
@pytest.mark.slow
class TestOracleRegression:
    def test_oracle(self, engine, tag):
        if engine == "rust":
            pytest.importorskip("python_cibersort._native")
        mixture, perm, qn, absolute, abs_method, seed = ORACLES[tag]
        res = cibersort(SIG, mixture, perm=perm, QN=qn, absolute=absolute,
                        abs_method=abs_method, seed=seed, threads=4,
                        engine=engine)
        oracle = read_oracle(_oracle_path(tag))
        max_err, p_identical, per_col = compare_to_oracle(res.table, oracle)
        assert max_err <= GATE, f"{engine}/{tag}: max abs err {max_err:g} > {GATE}; per-col {per_col}"
        if perm > 0:
            assert p_identical, f"{engine}/{tag}: P-value column differs from oracle"


@pytest.mark.slow
class TestDeterminism:
    def test_same_seed_same_result(self):
        kwargs = dict(perm=100, QN=False, absolute=False, seed=42, engine="libsvm")
        r1 = cibersort(SIG, ORACLES["noisy_rel_perm100_seed42"][0], threads=1, **kwargs)
        r2 = cibersort(SIG, ORACLES["noisy_rel_perm100_seed42"][0], threads=1, **kwargs)
        np.testing.assert_array_equal(r1.table.to_numpy(float),
                                      r2.table.to_numpy(float))

    @pytest.mark.parametrize("engine", ENGINES)
    def test_thread_count_invariant(self, engine):
        if engine == "rust":
            pytest.importorskip("python_cibersort._native")
        mix = ORACLES["noisy_rel_perm100_seed42"][0]
        kwargs = dict(perm=100, QN=False, absolute=False, seed=42, engine=engine)
        r1 = cibersort(SIG, mix, threads=1, **kwargs)
        r8 = cibersort(SIG, mix, threads=8, **kwargs)
        np.testing.assert_array_equal(r1.table.to_numpy(float),
                                      r8.table.to_numpy(float))
