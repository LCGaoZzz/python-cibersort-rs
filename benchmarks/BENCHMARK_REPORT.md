# Benchmark report

Date: 2026-08-14

System: shared Windows development workstation, 16 CPU cores, 33.8 GB RAM

Runtime: Python 3.11.15, NumPy 2.2.6, pandas 2.3.3,
libsvm-official 3.37.0, release-mode Rust extension

## Method

All benchmark inputs are deterministic and synthetic. Unless stated otherwise,
runs use `perm=100`, `QN=False`, `absolute=False`, `seed=42`, and BLAS pools
pinned to one thread. The CIBERSORT null distribution is shared by all samples,
so the number of SVR fits is `(permutations + samples) x 3 candidate nu values`.

| Size | Signature matrix | Mixture samples | SVR fits per run |
|---|---:|---:|---:|
| Small | 100 genes x 8 cell types | 4 | 312 |
| Medium | 300 genes x 12 cell types | 20 | 360 |
| Large | 547 genes x 22 cell types | 50 | 450 |

The machine carried variable background load. Raw rows and standard deviations
are retained so the variability is visible rather than hidden.

## Rust thread scaling

Five repetitions per cell. Source: `results_scaling.csv`; aggregation:
`results_summary.csv`.

| Size | Threads | Mean +/- SD (s) | Peak RSS (MB) | Speedup vs 1 thread |
|---|---:|---:|---:|---:|
| Small | 1 | 0.541 +/- 0.034 | 120.4 | 1.00x |
| Small | 4 | 0.171 +/- 0.006 | 121.8 | 3.17x |
| Small | 8 | 0.136 +/- 0.014 | 122.9 | 3.99x |
| Small | 16 | 0.112 +/- 0.007 | 124.6 | 4.85x |
| Medium | 1 | 8.605 +/- 0.552 | 127.8 | 1.00x |
| Medium | 4 | 4.699 +/- 0.471 | 130.4 | 1.83x |
| Medium | 8 | 3.406 +/- 0.110 | 133.9 | 2.53x |
| Medium | 16 | 3.019 +/- 0.360 | 135.6 | 2.85x |
| Large | 1 | 210.25 +/- 25.82 | 132.1 | 1.00x |
| Large | 4 | 55.32 +/- 5.08 | 142.4 | 3.80x |
| Large | 8 | 38.58 +/- 5.02 | 145.5 | 5.45x |
| Large | 16 | 32.64 +/- 0.74 | 148.4 | 6.44x |

Small and medium jobs saturate early because TSV I/O, preprocessing, and thread
pool setup become material at short runtimes. The large workload retains useful
scaling through 16 threads. Large single-thread timings were re-run under a
consistent load after the first collection showed a bimodal 88/220-second
distribution.

## Engine comparison

LIBSVM values are means of three repetitions. NumPy was run only for the small
single-thread configuration because it is a correctness/performance baseline,
not a recommended production engine. Rust values come from the five-repetition
scaling benchmark.

| Size | Threads | NumPy (s) | LIBSVM mean (s) | Rust mean (s) | Rust vs LIBSVM |
|---|---:|---:|---:|---:|---:|
| Small | 1 | 1167.6 | 3.19 | 0.541 | 5.9x |
| Small | 8 | - | 2.67 | 0.136 | 19.6x |
| Small | 16 | - | 2.47 | 0.112 | 22.1x |
| Medium | 1 | - | 17.52 | 8.60 | 2.0x |
| Medium | 8 | - | 11.53 | 3.41 | 3.4x |
| Medium | 16 | - | 10.92 | 3.02 | 3.6x |
| Large | 1 | - | 185.13 | 210.25 | 0.88x |
| Large | 8 | - | 49.40 | 38.58 | 1.3x |
| Large | 16 | - | 51.60 | 32.64 | 1.6x |

The Rust engine is not universally faster per core: at large/one-thread it was
slower than LIBSVM in this collection. Its main benefits are one native batch
call, a shared Gram matrix, native permutation draws, and Rayon parallelism.
The LIBSVM Python engine incurs per-fit ctypes marshalling and stops scaling
after eight threads on the large fixture.

## Original R comparison

The same 330-gene x 22-cell-type x 18-sample synthetic fixture was evaluated
with the unmodified CIBERSORT v1.04 script and with the Rust engine at eight
threads. The original R script and its license-restricted source are not
distributed in this repository.

| Oracle configuration | R v1.04 (s) | Rust (s) | Speedup |
|---|---:|---:|---:|
| Relative, seed 1 | 37.52 | 14.9 | 2.5x |
| Relative, seed 42 | 38.81 | 9.1 | 4.3x |
| Relative, seed 20260814 | 39.52 | 10.1 | 3.9x |
| Relative + QN, seed 42 | 25.31 | 6.6 | 3.8x |
| Absolute `sig.score`, seed 42 | 38.99 | 11.3 | 3.5x |
| Absolute `no.sumto1`, seed 42 | 40.10 | 10.1 | 4.0x |

The `perm=100` speedup ranged from 2.5x to 4.3x, with a median of about 3.8x.
These validation timings were collected while other benchmark processes shared
the machine, so they should be interpreted as indicative rather than as a
controlled hardware comparison.

## Numerical validation

- Maximum absolute error excluding P-values: `1.41e-11`.
- Acceptance threshold: `1e-6`.
- Permutation P-values: exactly equal for matching seeds.
- Output invariant across tested thread counts.

See `tests/validation_summary.md` for the oracle matrix.

## Reproduction

```bash
python benchmarks/generate_benchmark_data.py
python benchmarks/run_benchmarks.py --engine rust
python benchmarks/compare_engines.py
```

Benchmark scripts overwrite their corresponding CSV output. Run them on an
otherwise idle machine and record CPU model, power policy, operating system,
and package versions when publishing new results.
