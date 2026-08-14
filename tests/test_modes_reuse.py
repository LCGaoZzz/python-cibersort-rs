"""One-fit reuse: cibersort_all must equal three separate cibersort() calls."""

from __future__ import annotations

import numpy as np
import pytest

from python_cibersort import cibersort, cibersort_all

from conftest import ORACLES, SIG

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("engine", ["libsvm", "rust"])
def test_cibersort_all_matches_separate_runs(engine):
    if engine == "rust":
        pytest.importorskip("python_cibersort._native")
    mixture = ORACLES["noisy_rel_perm100_seed42"][0]
    common = dict(perm=10, QN=False, seed=42, threads=4, engine=engine)
    bundle = cibersort_all(SIG, mixture, **common)
    rel = cibersort(SIG, mixture, absolute=False, **common)
    sig = cibersort(SIG, mixture, absolute=True, abs_method="sig.score", **common)
    nosum = cibersort(SIG, mixture, absolute=True, abs_method="no.sumto1", **common)
    for got, exp in ((bundle["relative"], rel),
                     (bundle["sig.score"], sig),
                     (bundle["no.sumto1"], nosum)):
        assert list(got.table.columns) == list(exp.table.columns)
        np.testing.assert_array_equal(got.table.to_numpy(float),
                                      exp.table.to_numpy(float))
    # P-values / Correlation / RMSE are shared across modes
    np.testing.assert_array_equal(bundle["relative"].table["P-value"],
                                  bundle["sig.score"].table["P-value"])
    np.testing.assert_array_equal(bundle["relative"].null_distribution,
                                  bundle["no.sumto1"].null_distribution)
