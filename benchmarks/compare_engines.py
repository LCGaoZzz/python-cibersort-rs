"""Abbreviated engine-comparison benchmark.

libsvm: sizes {small,medium,large} x threads {1,8,16} x 3 reps.
numpy : small x threads {1} x 1 rep (GIL-bound; perm=100 extrapolated from
        profiling otherwise). Rust numbers live in results_summary.csv.

Writes results_engines.csv.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import csv, gc, time, tracemalloc
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
proc = psutil.Process(os.getpid())

def peak_rss_mb(p):
    try:
        return p.memory_info().peak_wset / 1e6
    except AttributeError:
        return p.memory_info().rss / 1e6

from python_cibersort import cibersort

def one(size, engine, threads, rep, fh, w):
    gc.collect(); tracemalloc.start()
    t0 = time.perf_counter()
    cibersort(os.path.join(HERE, "data", size, "signature_matrix.tsv"),
              os.path.join(HERE, "data", size, "mixture.tsv"),
              perm=100, QN=False, absolute=False,
              seed=42, threads=threads, engine=engine)
    dt = time.perf_counter() - t0
    _, tm = tracemalloc.get_traced_memory(); tracemalloc.stop()
    w.writerow([size, engine, threads, rep, f"{dt:.6f}",
                f"{peak_rss_mb(proc):.1f}", f"{tm / 1e6:.1f}"])
    fh.flush()
    print(f"{size:6s} {engine:6s} t={threads:2d} rep={rep} {dt:9.3f}s", flush=True)

out = os.path.join(HERE, "results_engines.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["size", "engine", "threads", "repeat", "seconds",
                "rss_peak_mb", "tracemalloc_peak_mb"])
    for size in ("small", "medium", "large"):
        for t in (1, 8, 16):
            for rep in range(3):
                one(size, "libsvm", t, rep, fh, w)
    one("small", "numpy", 1, 0, fh, w)
print("done", flush=True)
