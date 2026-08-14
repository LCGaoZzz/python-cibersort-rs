"""nu-SVR engine backed by the official libsvm Python bindings.

Matches e1071::svm(type="nu-regression", kernel="linear", scale=FALSE)
semantics exactly: libsvm NU_SVR, linear kernel, C=1, p=0.1, eps=1e-3,
shrinking on.

Performance notes:
* The libsvm problem (X block in libsvm sparse-node format) is built once
  per thread and the target vector y is swapped in via ctypes.memmove,
  avoiding the ~12 ms/problem Python-side rebuild on every fit.
* svm_parameter objects are parsed once per nu value and shared.
* libsvm's C entry point releases the GIL (ctypes), so fits parallelise
  across Python threads.
"""

from __future__ import annotations

import ctypes
import threading

import numpy as np
from libsvm.svmutil import svm_problem, svm_parameter, svm_train

_NUS = (0.25, 0.5, 0.75)
_PARAM_TEMPLATE = "-s 4 -t 0 -c 1 -n {nu} -p 0.1 -e 0.001 -h 1 -b 0 -q"


class LibsvmEngine:
    """Thread-safe nu-SVR trainer for a fixed design matrix X."""

    name = "libsvm"

    def __init__(self, X: np.ndarray):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.n, self.d = self.X.shape
        self._params = {nu: svm_parameter(_PARAM_TEMPLATE.format(nu=nu)) for nu in _NUS}
        self._tls = threading.local()
        self._lock = threading.Lock()
        self._all_probs = []  # keep refs alive

    def _problem(self, y: np.ndarray):
        """Thread-local svm_problem whose X nodes are built once; y memmoved."""
        prob = getattr(self._tls, "prob", None)
        if prob is None:
            prob = svm_problem(y, self.X)
            self._tls.prob = prob
            with self._lock:
                self._all_probs.append(prob)
        else:
            ctypes.memmove(prob.y, y.ctypes.data, 8 * self.n)
        return prob

    def fit(self, y: np.ndarray, nu: float) -> np.ndarray:
        """Fit nu-SVR and return w = coefs @ SV (dense, length d)."""
        y = np.ascontiguousarray(y, dtype=np.float64)
        prob = self._problem(y)
        model = svm_train(prob, self._params[nu])
        coefs = model.get_sv_coef()
        svs = model.get_SV()
        w = np.zeros(self.d, dtype=np.float64)
        for c, sv in zip(coefs, svs):
            cv = c[0]
            for idx, val in sv.items():
                w[idx - 1] += cv * val
        return w


def fit_nusvr_libsvm(X: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray:
    """Convenience one-shot fit (no caching)."""
    return LibsvmEngine(X).fit(y, nu)
