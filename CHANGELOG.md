# Changelog

## 0.1.1 - 2026-08-14

- Performance: the Rust nu-SVR solver now maintains a signed permuted Gram
  matrix with zero-copy `get_Q` row access and auto-vectorized gradient
  updates — pure data-movement changes with bit-identical outputs, including
  permutation P-values ([#1](https://github.com/LCGaoZzz/python-cibersort-rs/pull/1)).
  Controlled same-session measurement: 3.1x faster at one thread and 2.4x at
  sixteen threads on the large synthetic fixture; the full R-oracle test suite
  passes unchanged.
- Docs: benchmark tables re-measured after the solver optimization.

## 0.1.0 - 2026-08-14

- Initial public release.
- Python API and CLI for relative and absolute CIBERSORT v1.04 modes.
- Bundled PyO3/Rayon Rust accelerator.
- R-compatible random stream and permutation P-values.
- Synthetic oracle fixtures, regression tests, and benchmark suite.
