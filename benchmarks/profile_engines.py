"""Profile the three SVR engines on identical input.

  * stage-level timing (read/sort -> anti-log -> QN -> RNG draws -> fits -> eval)
    for the Python engines (numpy / libsvm) via an instrumented replica of the
    cibersort() pipeline;
  * total wall time for all three engines (rust collapses fits+eval into one
    native call; its stage split is reported as preprocessing vs native);
  * cProfile top functions per engine.

Writes benchmarks/profile_results.txt with all numbers used in PROFILE_REPORT.md.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import cProfile
import io
import pstats
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from python_cibersort.core import NUS, _CoreAlg, _r_sort, _zscore, read_table
from python_cibersort.qn import quantile_normalize
from python_cibersort.rrnd import RRng


def staged_pipeline(sig_path: str, mix_path: str, perm: int, seed: int,
                    threads: int, engine: str) -> dict:
    """Instrumented replica of core.cibersort for Python engines."""
    t = {}
    t0 = time.perf_counter()
    Xdf = _r_sort(read_table(sig_path))
    Ydf = _r_sort(read_table(mix_path))
    X = Xdf.to_numpy(np.float64)
    Y = Ydf.to_numpy(np.float64)
    t["read_sort"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if Y.max() < 50:
        Y = np.exp2(Y)
    Y = quantile_normalize(Y)
    t["antilog_qn"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ymask = np.isin(Ydf.index.to_numpy(), Xdf.index.to_numpy())
    Ydf = Ydf.iloc[ymask]
    Y = Y[ymask]
    xmask = np.isin(Xdf.index.to_numpy(), Ydf.index.to_numpy())
    Xdf = Xdf.iloc[xmask]
    X = np.ascontiguousarray((X[xmask] - X[xmask].mean()) / X[xmask].std(ddof=1))
    t["intersect_standardise"] = time.perf_counter() - t0

    n_genes, n_samples = X.shape[0], Y.shape[1]

    t0 = time.perf_counter()
    rng = RRng(seed)
    flat = Y.ravel(order="F")
    perm_targets = []
    for _ in range(perm):
        idx = rng.sample_no_replace(flat.size, n_genes)
        perm_targets.append(_zscore(flat[idx - 1]))
    t["rng_draws"] = time.perf_counter() - t0

    alg = _CoreAlg(X, engine=engine)
    targets = perm_targets + [_zscore(Y[:, j].copy()) for j in range(n_samples)]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = [pool.submit(alg.fit_one, y, nu) for y in targets for nu in NUS]
        raws = [f.result() for f in futs]
    t["fits"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i, y in enumerate(targets):
        alg.evaluate(y, raws[3 * i:3 * i + 3], False, "sig.score")
    t["evaluate"] = time.perf_counter() - t0
    t["total"] = sum(t.values())
    return t


def total_only(sig_path: str, mix_path: str, perm: int, seed: int,
               threads: int, engine: str) -> float:
    from python_cibersort import cibersort
    t0 = time.perf_counter()
    cibersort(sig_path, mix_path, perm=perm, QN=True, absolute=False,
              seed=seed, threads=threads, engine=engine)
    return time.perf_counter() - t0


def cprofile_engine(sig_path: str, mix_path: str, perm: int, seed: int,
                    threads: int, engine: str) -> str:
    pr = cProfile.Profile()
    pr.enable()
    total_only(sig_path, mix_path, perm, seed, threads, engine)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(15)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--perm", default="100")
    ap.add_argument("--numpy-perm", default="10",
                    help="reduced perm for the pure-Python engine (it is ~100x slower)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sig = os.path.join(here, "data", args.size, "signature_matrix.tsv")
    mix = os.path.join(here, "data", args.size, "mixture.tsv")
    out_lines = []

    def log(s=""):
        print(s, flush=True)
        out_lines.append(s)

    log(f"# profile run: size={args.size} threads={args.threads} seed={args.seed} "
        f"perm={args.perm} (numpy perm={args.numpy_perm})")
    for engine, perm in (("numpy", int(args.numpy_perm)),
                         ("libsvm", int(args.perm)),
                         ("rust", int(args.perm))):
        if engine == "rust":
            tot = total_only(sig, mix, perm, args.seed, args.threads, "rust")
            log(f"\n## engine=rust perm={perm}: total={tot:.3f}s "
                f"(preprocessing + single native fits+eval call)")
        else:
            stages = staged_pipeline(sig, mix, perm, args.seed, args.threads, engine)
            log(f"\n## engine={engine} perm={perm}: total={stages['total']:.3f}s")
            for k, v in stages.items():
                if k != "total":
                    log(f"   {k:22s} {v:9.4f}s  ({100 * v / stages['total']:5.1f}%)")
        log(f"\n### cProfile top-15 (engine={engine})")
        log(cprofile_engine(sig, mix, perm, args.seed, args.threads, engine))

    out = os.path.join(here, "profile_results.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(out_lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
