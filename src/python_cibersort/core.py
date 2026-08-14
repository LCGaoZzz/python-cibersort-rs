"""CIBERSORT v1.04 pipeline — exact Python re-implementation of CIBERSORT.R.

Reproduces, in R's exact order of operations:
read TSV -> sort by gene symbol -> anti-log if max(Y)<50 -> optional
quantile normalisation (preprocessCore) -> store Yorig/Ymedian -> intersect
genes (Y filtered by X, then X filtered by Y) -> standardise X with R's
(n-1) sd over all elements -> empirical null via doPerm (R RNG + sample)
-> per-sample z-scored nu-SVR deconvolution -> optional absolute scaling.

Determinism: all randomness (the permutation draws) is consumed
sequentially *before* any parallel fitting, so results are bit-identical
for any thread count. Fits themselves are deterministic.

One-fit reuse: the ν-SVR fits do not depend on `absolute`/`abs_method`
(the R CoreAlg applies those only when normalising the final weights), so
`cibersort_all` computes relative, sig.score and no.sumto1 results from a
single batch of fits.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .qn import quantile_normalize
from .rrnd import RRng
from .svr_libsvm import LibsvmEngine
from .svr_numpy import fit_nusvr_numpy

NUS = (0.25, 0.5, 0.75)
ABS_METHODS = ("sig.score", "no.sumto1")


# --------------------------------------------------------------------- I/O
def read_table(path: str) -> pd.DataFrame:
    """read.table(header=T, sep='\\t', row.names=1, check.names=F)."""
    df = pd.read_csv(path, sep="\t", index_col=0, float_precision="round_trip")
    df = df.apply(pd.to_numeric, errors="raise")
    df = df.astype(np.float64)
    return df


def _r_sort(df: pd.DataFrame) -> pd.DataFrame:
    """order(rownames(df)) — lexicographic (C-locale) like R for ASCII."""
    order = np.argsort(df.index.to_numpy(), kind="stable")
    return df.iloc[order]


# ----------------------------------------------------------------- helpers
def _zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std(ddof=1)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ac = a - a.mean()
    bc = b - b.mean()
    denom = np.sqrt((ac * ac).sum() * (bc * bc).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return float((ac * bc).sum() / denom)


# ------------------------------------------------------------------ result
@dataclass
class CibersortResult:
    table: pd.DataFrame          # the R `obj` equivalent
    absolute: bool
    abs_method: str
    perm: int
    engine: str
    threads: int
    null_distribution: np.ndarray | None = None

    def write(self, path: str) -> None:
        """write.table(obj, sep='\\t', quote=F, row.names=T, col.names=NA)."""
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("\t" + "\t".join(self.table.columns) + "\n")
            for name, row in self.table.iterrows():
                cells = [format(v, ".17g") for v in row.to_numpy()]
                fh.write(str(name) + "\t" + "\t".join(cells) + "\n")


# --------------------------------------------------------------- CoreAlg
class _CoreAlg:
    """CoreAlg over a fixed standardised signature matrix X."""

    def __init__(self, X: np.ndarray, engine: str = "libsvm"):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.d = X.shape[1]
        self.engine_name = engine
        self._libsvm = LibsvmEngine(self.X) if engine == "libsvm" else None

    def fit_one(self, y: np.ndarray, nu: float) -> np.ndarray:
        if self._libsvm is not None:
            return self._libsvm.fit(y, nu)
        return fit_nusvr_numpy(self.X, y, nu)

    def evaluate(self, y: np.ndarray, w_raws: list[np.ndarray],
                 absolute: bool, abs_method: str):
        """Given the 3 raw weight vectors (per nu), return (w, mix_rmse, mix_r)."""
        best_rmse = np.inf
        best = 0
        rmses = []
        corrs = []
        for i, w_raw in enumerate(w_raws):
            wt = np.where(w_raw < 0, 0.0, w_raw)
            s = wt.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                w = wt / s
            k = self.X @ w
            rmse = float(np.sqrt(np.mean((k - y) ** 2)))
            rmses.append(rmse)
            corrs.append(_pearson(k, y))
            if rmse < best_rmse:
                best_rmse = rmse
                best = i
        q = np.where(w_raws[best] < 0, 0.0, w_raws[best])
        if (not absolute) or abs_method == "sig.score":
            with np.errstate(invalid="ignore", divide="ignore"):
                w = q / q.sum()
        else:  # absolute & no.sumto1
            w = q
        return w, rmses[best], corrs[best], best


# ---------------------------------------------------------------- pipeline
class _Run:
    """Everything computed once per (sig, mixture, perm, QN, seed) regardless
    of absolute/abs_method: preprocessed matrices, fits, and the pieces needed
    to assemble any output mode."""

    def __init__(self, sig_matrix, mixture_file, perm, QN, seed, threads, engine):
        if perm > 0 and seed is None:
            raise ValueError("perm > 0 requires an explicit `seed` for reproducible P-values")
        if engine not in {"rust", "libsvm", "numpy"}:
            raise ValueError("engine must be one of: 'rust', 'libsvm', 'numpy'")
        if engine == "rust":
            try:
                from . import _native  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "engine='rust' requires the bundled native extension; "
                    "install a platform wheel or build the project with maturin") from exc
        self.perm = perm
        self.threads = threads
        self.engine = engine

        Xdf = read_table(sig_matrix) if isinstance(sig_matrix, str) else sig_matrix.copy()
        Ydf = read_table(mixture_file) if isinstance(mixture_file, str) else mixture_file.copy()

        Xdf = _r_sort(Xdf)
        Ydf = _r_sort(Ydf)

        X = Xdf.to_numpy(dtype=np.float64)
        Y = Ydf.to_numpy(dtype=np.float64)

        # anti-log if max < 50 in mixture file
        if Y.max() < 50:
            Y = np.exp2(Y)

        # quantile normalisation of mixture file
        if QN:
            Y = quantile_normalize(Y)

        # store original mixtures
        Yorig = Y
        self.Ymedian = max(float(np.median(Yorig)), 1.0)

        # intersect genes (Y by X, then X by the filtered Y) — order preserved
        xgns = Xdf.index.to_numpy()
        ygns = Ydf.index.to_numpy()
        ymask = np.isin(ygns, xgns)
        Ydf = Ydf.iloc[ymask]
        Y = Y[ymask]
        xmask = np.isin(xgns, Ydf.index.to_numpy())
        Xdf = Xdf.iloc[xmask]
        X = X[xmask]

        # standardise sig matrix (R sd = n-1 over all elements, as.vector order)
        X = (X - X.mean()) / X.std(ddof=1)
        self.X = np.ascontiguousarray(X)
        self.Y = Y
        self.Ydf = Ydf
        self.n_samples = Y.shape[1]
        self.n_genes = X.shape[0]
        self.cell_types = list(Xdf.columns)

        self.alg = _CoreAlg(self.X, engine=("libsvm" if engine == "rust" else engine))
        # NB: for engine="rust" the fits run natively; `alg` is then only used
        # for evaluate() (nu selection / normalisation), which is engine-free.

        # ------------------------------------------------ permutation draws
        self.perm_targets: list[np.ndarray] = []
        if perm > 0:
            flat = Y.ravel(order="F")  # as.list(data.matrix(Y)): column-major
            n = flat.size
            if engine == "rust":
                # native draw stream (bit-exact port of R's RNG, ~50x faster)
                from . import _native
                idx_all = np.asarray(
                    _native.perm_indices(seed, n, self.n_genes, perm),
                    dtype=np.int64).reshape(perm, self.n_genes)
                for i in range(perm):
                    self.perm_targets.append(_zscore(flat[idx_all[i] - 1]))
            else:
                rng = RRng(seed)
                for _ in range(perm):
                    idx = rng.sample_no_replace(n, self.n_genes)  # 1-based like R
                    yr = flat[idx - 1]
                    self.perm_targets.append(_zscore(yr))

        # ------------------------------------------------ parallel fitting
        self.sample_targets = [_zscore(Y[:, j].copy()) for j in range(self.n_samples)]

        if engine == "rust":
            from ._rust_engine import run_fits_rust
            targets = np.vstack(self.perm_targets + self.sample_targets) \
                if (self.perm_targets or self.sample_targets) \
                else np.empty((0, self.n_genes))
            self.fits = run_fits_rust(self.X, targets, len(self.perm_targets),
                                      self.n_samples, threads)
        else:
            self.fits = self._run_fits_threaded()

    def _run_fits_threaded(self) -> dict:
        # task list: (kind, idx, nu_idx); kind 0=perm, 1=sample
        out = {}
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            futs = {}
            for i, yr in enumerate(self.perm_targets):
                for ni, nu in enumerate(NUS):
                    futs[pool.submit(self.alg.fit_one, yr, nu)] = (0, i, ni)
            for j, ys in enumerate(self.sample_targets):
                for ni, nu in enumerate(NUS):
                    futs[pool.submit(self.alg.fit_one, ys, nu)] = (1, j, ni)
            for fut, key in futs.items():
                out[key] = fut.result()
        return out

    # ------------------------------------------------ empirical null
    def null_distribution(self) -> np.ndarray | None:
        """Sorted per-permutation correlations (mode-independent: the null
        uses mix_r, which does not depend on absolute/abs_method)."""
        if self.perm == 0:
            return None
        mix_rs = np.empty(self.perm, dtype=np.float64)
        for i, yr in enumerate(self.perm_targets):
            w_raws = [self.fits[(0, i, ni)] for ni in range(3)]
            _, _, mix_r, _ = self.alg.evaluate(yr, w_raws, False, "sig.score")
            mix_rs[i] = mix_r
        return np.sort(mix_rs)

    # ------------------------------------------------ per-sample results
    def assemble(self, absolute: bool, abs_method: str) -> CibersortResult:
        if absolute and abs_method not in ABS_METHODS:
            raise ValueError("abs_method must be set to either 'sig.score' or 'no.sumto1'")
        nulldist = self.null_distribution()
        rows = []
        index = []
        for j in range(self.n_samples):
            y = self.sample_targets[j]
            w_raws = [self.fits[(1, j, ni)] for ni in range(3)]
            w, mix_rmse, mix_r, _ = self.alg.evaluate(y, w_raws, absolute, abs_method)
            if absolute and abs_method == "sig.score":
                w = w * (float(np.median(self.Y[:, j])) / self.Ymedian)
            if self.perm > 0:
                pval = 1.0 - (int(np.argmin(np.abs(nulldist - mix_r))) + 1) / len(nulldist)
            else:
                pval = 9999.0
            row = list(w) + [pval, mix_r, mix_rmse]
            if absolute:
                row.append(float(w.sum()))
            rows.append(row)
            index.append(str(self.Ydf.columns[j]))

        cols = self.cell_types + ["P-value", "Correlation", "RMSE"]
        if absolute:
            cols.append(f"Absolute score ({abs_method})")
        table = pd.DataFrame(rows, index=index, columns=cols)

        return CibersortResult(table=table, absolute=absolute, abs_method=abs_method,
                               perm=self.perm, engine=self.engine, threads=self.threads,
                               null_distribution=nulldist)


def cibersort(sig_matrix: str | pd.DataFrame,
              mixture_file: str | pd.DataFrame,
              perm: int = 0,
              QN: bool = True,
              absolute: bool = False,
              abs_method: str = "sig.score",
              seed: int | None = None,
              threads: int = 1,
              engine: str = "rust") -> CibersortResult:
    """Python port of CIBERSORT(sig_matrix, mixture_file, perm, QN, absolute, abs_method).

    Additional kwargs: `seed` (required when perm > 0 for reproducible
    P-values; consumed by an exact replica of R's RNG), `threads`
    (parallel workers over samples x permutations x nu), `engine`
    ('libsvm' | 'numpy' | 'rust').
    """
    run = _Run(sig_matrix, mixture_file, perm, QN, seed, threads, engine)
    return run.assemble(absolute, abs_method)


def cibersort_all(sig_matrix: str | pd.DataFrame,
                  mixture_file: str | pd.DataFrame,
                  perm: int = 0,
                  QN: bool = True,
                  seed: int | None = None,
                  threads: int = 1,
                  engine: str = "rust") -> dict[str, CibersortResult]:
    """Run the fits ONCE and return all three output modes.

    Returns {'relative': ..., 'sig.score': ..., 'no.sumto1': ...}.  The
    ν-SVR fits, the empirical null and the P-values are shared; only the
    final weight normalisation differs between modes.
    """
    run = _Run(sig_matrix, mixture_file, perm, QN, seed, threads, engine)
    return {
        "relative": run.assemble(False, "sig.score"),
        "sig.score": run.assemble(True, "sig.score"),
        "no.sumto1": run.assemble(True, "no.sumto1"),
    }


# R-style alias
CIBERSORT = cibersort
