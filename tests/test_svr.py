"""nu-SVR engine regression vs e1071/libsvm references (hex = bit-exact)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from conftest import DATA_DIR, HEX_DIR, read_hex_matrix

X_HEX = None
Y_HEX = None


def _xy():
    global X_HEX, Y_HEX
    if X_HEX is None:
        X_HEX = read_hex_matrix(os.path.join(HEX_DIR, "corealg_X_hex.tsv"))
        Y_HEX = read_hex_matrix(os.path.join(HEX_DIR, "corealg_y1_hex.tsv")).ravel()
    return X_HEX, Y_HEX


def _w_ref(nu_tag: str) -> np.ndarray:
    return read_hex_matrix(os.path.join(HEX_DIR, f"svm_w_nu{nu_tag}_hex.tsv")).ravel()


NUS = ((0.25, "0.25"), (0.5, "0.50"), (0.75, "0.75"))


class TestLibsvmEngine:
    """The official libsvm bindings must reproduce e1071 bit-for-bit."""

    @pytest.mark.parametrize("nu,tag", NUS)
    def test_w_bitexact(self, nu, tag):
        from python_cibersort.svr_libsvm import LibsvmEngine
        X, y = _xy()
        eng = LibsvmEngine(X)
        w = eng.fit(y.copy(), nu)
        np.testing.assert_array_equal(w, _w_ref(tag))

    @pytest.mark.parametrize("nu,tag", NUS)
    def test_nsv_and_rho(self, nu, tag):
        """Structural agreement: same support-vector count and rho as e1071."""
        from libsvm.svmutil import svm_problem, svm_parameter, svm_train
        X, y = _xy()
        prob = svm_problem(y, X)
        param = svm_parameter(f"-s 4 -t 0 -c 1 -n {nu} -p 0.1 -e 0.001 -h 1 -b 0 -q")
        model = svm_train(prob, param)
        diag = pd.read_csv(os.path.join(DATA_DIR, f"svm_diag_nu{tag}.tsv"), sep="\t")
        ref = dict(zip(diag["name"], diag["value"]))
        assert model.get_nr_sv() == int(ref["tot.nSV"])
        # e1071 reports libsvm's rho[0] verbatim for nu-regression
        assert abs(model.rho[0] - float(ref["rho_1"])) < 1e-12

    def test_problem_cache_reuse_is_consistent(self):
        """Swapping y into the cached problem must give identical fits as fresh."""
        from python_cibersort.svr_libsvm import LibsvmEngine, fit_nusvr_libsvm
        X, y = _xy()
        eng = LibsvmEngine(X)
        y2 = y[::-1].copy()  # different target
        w1a = eng.fit(y.copy(), 0.5)
        w2a = eng.fit(y2.copy(), 0.5)
        w1b = fit_nusvr_libsvm(X, y.copy(), 0.5)
        w2b = fit_nusvr_libsvm(X, y2.copy(), 0.5)
        np.testing.assert_array_equal(w1a, w1b)
        np.testing.assert_array_equal(w2a, w2b)


class TestRustEngine:
    pytestmark = pytest.mark.skipif(
        __import__("importlib").util.find_spec("python_cibersort._native") is None,
        reason="native extension not built")

    @pytest.mark.parametrize("nu,tag", NUS)
    def test_w_bitexact(self, nu, tag):
        from python_cibersort import _native
        X, y = _xy()
        w = np.asarray(_native.nusvr_w(X, y, nu), dtype=np.float64)
        np.testing.assert_array_equal(w, _w_ref(tag))


class TestNumpyEngine:
    """Pure-NumPy port (performance baseline). Shrinking is disabled, so the
    solution can differ from libsvm at the eps=1e-3 KKT tolerance level; the
    agreement check below uses a loose gate appropriate to that role."""

    def test_small_problem_agrees_with_libsvm(self):
        from python_cibersort.svr_libsvm import fit_nusvr_libsvm
        from python_cibersort.svr_numpy import fit_nusvr_numpy
        rng = np.random.default_rng(7)
        Xs = rng.normal(size=(40, 6))
        ys = rng.normal(size=40)
        for nu in (0.25, 0.5, 0.75):
            wn = fit_nusvr_numpy(Xs, ys, nu)
            wl = fit_nusvr_libsvm(Xs, ys, nu)
            err = float(np.max(np.abs(wn - wl)))
            assert err < 2e-3, f"nu={nu}: numpy vs libsvm max abs w error {err:g}"
