# python-cibersort-rs

[![CI](https://github.com/LCGaoZzz/python-cibersort-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/LCGaoZzz/python-cibersort-rs/actions/workflows/ci.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

An installable Python implementation of the CIBERSORT v1.04 deconvolution
pipeline with a bundled Rust accelerator. It runs without R and exposes both
a Python API and a command-line interface.

The implementation reproduces the original preprocessing, R-compatible random
draws, three linear nu-SVR fits, model selection, relative output, and both
absolute modes. On the included synthetic fixture, the maximum absolute error
against R-generated oracle output is `1.41e-11`, and permutation P-values are
identical for matching seeds.

> [!IMPORTANT]
> This is an independent research implementation. It is not affiliated with,
> endorsed by, or distributed by Stanford University. The repository does not
> contain the original `CIBERSORT.R` script, LM22, or any proprietary signature
> matrix. Users are responsible for complying with the applicable
> [CIBERSORT/CIBERSORTx terms](https://cibersortx.stanford.edu/) and for obtaining
> any required data or commercial-use rights. It is not intended for clinical
> diagnosis or treatment.

## Features

- One package containing the Python API, CLI, and PyO3/Rayon Rust extension.
- R-compatible Mersenne-Twister and `sample()` sequence for reproducible
  permutation P-values.
- Relative, `sig.score`, and `no.sumto1` output modes.
- `cibersort_all()` reuses one fit batch for all three output modes.
- Rust, official LIBSVM, and NumPy engines for performance/reference testing.
- Deterministic output across thread counts.
- Synthetic fixtures, oracle tests, raw benchmark CSV files, and reproducible
  benchmark scripts.

## Installation

### Prebuilt wheel

Download the wheel for your Python version and operating system from
[GitHub Releases](https://github.com/LCGaoZzz/python-cibersort-rs/releases),
then install it:

```bash
python -m pip install python_cibersort_rs-0.1.0-*.whl
```

### Build from source

Building from source requires Python 3.10 or later and a stable Rust toolchain:

```bash
git clone https://github.com/LCGaoZzz/python-cibersort-rs.git
cd python-cibersort-rs
python -m pip install .
```

For an editable development installation:

```bash
python -m pip install "maturin>=1.7,<2" pytest psutil
maturin develop --release
```

## Python API

```python
from python_cibersort import cibersort

result = cibersort(
    "signature_matrix.tsv",  # genes x cell types
    "mixture.tsv",           # genes x samples
    perm=100,
    QN=True,
    absolute=False,
    abs_method="sig.score",  # or "no.sumto1"
    seed=42,                  # required when perm > 0
    threads=8,
    engine="rust",           # "rust", "libsvm", or "numpy"
)

print(result.table)           # pandas.DataFrame
result.write("CIBERSORT-Results.txt")
```

`CIBERSORT` is exported as an alias for users migrating from the R entry point.
Input paths may also be replaced with pandas DataFrames.

To calculate every output mode from one shared batch of nu-SVR fits:

```python
from python_cibersort import cibersort_all

outputs = cibersort_all(
    "signature_matrix.tsv",
    "mixture.tsv",
    perm=100,
    QN=True,
    seed=42,
    threads=8,
    engine="rust",
)

relative = outputs["relative"].table
absolute_sig_score = outputs["sig.score"].table
absolute_no_sum = outputs["no.sumto1"].table
```

## Command line

```bash
cibersort signature_matrix.tsv mixture.tsv \
  --out CIBERSORT-Results.txt \
  --perm 100 \
  --seed 42 \
  --threads 8 \
  --engine rust
```

Use `cibersort --help` for all options, including quantile normalization and
absolute mode. The CLI defaults to the Rust engine and falls back to LIBSVM if
the native extension is unavailable.

## Input format

Both inputs are tab-separated matrices with identifiers in the first column:

- Signature matrix: rows are genes; remaining columns are cell types.
- Mixture matrix: rows are genes; remaining columns are bulk samples.
- Gene identifiers are intersected and ordered internally.
- Values must be numeric and finite.

The package does not ship LM22 or another biological signature matrix.

## Correctness

The public fixture contains 330 synthetic genes, 22 synthetic cell types, and
18 mixtures. Seven oracle configurations cover relative output, both absolute
modes, quantile normalization on/off, three random seeds, and `perm=100`.

| Check | Result |
|---|---:|
| Maximum absolute error excluding P-value | `1.41e-11` |
| Required numerical tolerance | `1e-6` |
| Permutation P-values | exactly identical |
| Thread-count determinism | identical |

See [the validation summary](tests/validation_summary.md) and the tests under
[`tests/`](tests/).

## Benchmarks

Measurements were collected on a shared Windows workstation with 16 CPU cores
and 33.8 GB RAM. Every timing uses synthetic data, `perm=100`, `QN=False`,
`seed=42`, and three to five repetitions for the Rust scaling table
(re-measured 2026-08-14 after the solver optimization below).

### Rust thread scaling

| Dataset | Matrix and samples | 1 thread | 4 threads | 8 threads | 16 threads |
|---|---|---:|---:|---:|---:|
| Small | 100 genes x 8 types, 4 samples | 0.308 s | 0.169 s (1.8x) | 0.126 s (2.4x) | 0.105 s (2.9x) |
| Medium | 300 genes x 12 types, 20 samples | 5.64 s | 2.25 s (2.5x) | 1.62 s (3.5x) | 1.37 s (4.1x) |
| Large | 547 genes x 22 types, 50 samples | 44.5 s | 14.6 s (3.0x) | 14.7 s (3.0x) | 7.32 s (6.1x) |

### Solver data-movement optimization (2026-08)

PR #1 rewrote the nu-SVR solver's hot loop with pure data-movement changes: a
signed permuted Gram matrix with zero-copy `get_Q` row access, plus
auto-vectorized (zip-form) gradient updates. No floating-point operation order
changed, so outputs — including permutation P-values — are **bit-identical**
to the previous implementation. In a controlled same-session comparison on
this workstation the large fixture improved from 118.0 s to 37.9 s at one
thread (3.1x) and from 8.7 s to 3.6 s at sixteen threads (2.4x); quiet-window
single-thread runs reached 29.0 s. Pre-optimization engine comparisons
(Rust vs LIBSVM: 0.136/2.67 s, 3.41/11.53 s, 38.58/49.40 s for small,
medium, large at eight threads) and the older scaling table (large 1 thread:
210.2 s) remain archived in the raw CSV files below.

These are workload-specific measurements, not universal guarantees. The
single-thread large runs were sensitive to background load; raw measurements
and standard deviations are retained for transparency.

See the full [benchmark report](benchmarks/BENCHMARK_REPORT.md),
[`results_scaling.csv`](benchmarks/results_scaling.csv), and
[`results_engines.csv`](benchmarks/results_engines.csv).

## Reproduce tests and benchmarks

```bash
# Fast unit tests
pytest -m "not slow"

# Complete oracle suite (several minutes)
pytest

# Regenerate synthetic benchmark inputs
python benchmarks/generate_benchmark_data.py

# Rust thread scaling and engine comparison
python benchmarks/run_benchmarks.py --engine rust
python benchmarks/compare_engines.py
```

## Project layout

```text
src/python_cibersort/    Python API, CLI, preprocessing, and fallback engines
rust/src/                PyO3/Rayon native RNG, QN, and nu-SVR implementation
tests/                   Unit tests, synthetic fixture, and numeric oracles
benchmarks/              Reproduction scripts, raw CSV results, and report
.github/workflows/       CI and release-wheel automation
```

## Citation

If you use this software, cite both this repository and the original CIBERSORT
publication:

> Newman AM, Liu CL, Green MR, et al. Robust enumeration of cell subsets from
> tissue expression profiles. *Nature Methods*. 2015;12:453-457.
> https://doi.org/10.1038/nmeth.3337

The native SVR implementation follows LIBSVM; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution.

## License

The repository is distributed under GPL-3.0-or-later. Third-party components,
research-use considerations, and upstream attributions are documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The project license does
not grant rights to CIBERSORT/CIBERSORTx, LM22, trademarks, patents, or other
materials owned by third parties.
