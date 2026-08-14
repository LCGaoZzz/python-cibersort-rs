"""Thread-scaling benchmark: sizes x threads x repeats for the default engine.

BLAS pools are pinned to 1 thread BEFORE numpy import so the package-level
thread pool is the only parallelism (no oversubscription).

Outputs:
  benchmarks/results_scaling.csv   (raw per-repeat rows)
  benchmarks/results_summary.csv   (mean +/- SD, peak memory, speedup)
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import gc
import time
import tracemalloc

import psutil

REPEATS = 5
THREADS = (1, 4, 8, 16)
SEED = 42
PERM = 100


def peak_rss_mb(proc) -> float:
    try:
        return proc.memory_info().peak_wset / 1e6  # Windows peak working set
    except AttributeError:
        return proc.memory_info().rss / 1e6


def run_one(sig: str, mix: str, engine: str, threads: int, proc):
    from python_cibersort import cibersort
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    res = cibersort(sig, mix, perm=PERM, QN=False, absolute=False,
                    seed=SEED, threads=threads, engine=engine)
    dt = time.perf_counter() - t0
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dt, peak_rss_mb(proc), tm_peak / 1e6, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="rust")
    ap.add_argument("--sizes", default="small,medium,large")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--threads", default=",".join(map(str, THREADS)))
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sizes = args.sizes.split(",")
    thread_list = [int(t) for t in args.threads.split(",")]
    proc = psutil.Process(os.getpid())

    raw_path = os.path.join(here, "results_scaling.csv")
    write_header = not os.path.exists(raw_path)
    rows = []
    with open(raw_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["size", "engine", "threads", "repeat", "seconds",
                        "rss_peak_mb", "tracemalloc_peak_mb"])
        for size in sizes:
            sig = os.path.join(here, "data", size, "signature_matrix.tsv")
            mix = os.path.join(here, "data", size, "mixture.tsv")
            for threads in thread_list:
                for rep in range(args.repeats):
                    dt, rss, tm, _ = run_one(sig, mix, args.engine, threads, proc)
                    row = [size, args.engine, threads, rep, f"{dt:.6f}",
                           f"{rss:.1f}", f"{tm:.1f}"]
                    w.writerow(row)
                    rows.append(row)
                    fh.flush()
                    print(f"{size:6s} engine={args.engine:6s} threads={threads:2d} "
                          f"rep={rep}  {dt:8.3f}s  rss_peak={rss:8.1f}MB", flush=True)

    # ---- summary
    import statistics as st
    agg = {}
    for size, engine, threads, rep, secs, rss, tm in rows:
        agg.setdefault((size, int(threads)), []).append((float(secs), float(rss), float(tm)))
    sum_path = os.path.join(here, "results_summary.csv")
    with open(sum_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["size", "engine", "threads", "n", "time_mean_s", "time_sd_s",
                    "rss_peak_max_mb", "tracemalloc_peak_max_mb", "speedup_vs_1t"])
        for (size, threads), vals in sorted(agg.items()):
            times = [v[0] for v in vals]
            mean = st.mean(times)
            sd = st.stdev(times) if len(times) > 1 else 0.0
            base = st.mean([v[0] for v in agg[(size, 1)]])
            w.writerow([size, args.engine, threads, len(times), f"{mean:.4f}",
                        f"{sd:.4f}", f"{max(v[1] for v in vals):.1f}",
                        f"{max(v[2] for v in vals):.1f}", f"{base / mean:.3f}"])
    print("wrote", raw_path, "and", sum_path)


if __name__ == "__main__":
    main()
