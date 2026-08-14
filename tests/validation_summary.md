# Validation summary

Comparison: `python-cibersort-rs` versus numeric oracle output generated with
the original CIBERSORT v1.04 implementation. Numerical gate: maximum absolute
error no greater than `1e-6`; P-values must be exactly equal for `perm=100`.
Python/Rust runs used eight threads.

| Oracle | Engine | Max absolute error excluding P | P-value identical | Wall time (s) |
|---|---|---:|---:|---:|
| exact_rel_perm0 | libsvm | 1.28e-15 | n/a | 0.1 |
| exact_rel_perm0 | rust | 1.28e-15 | n/a | <0.1 |
| noisy_rel_perm100_seed1 | libsvm | 6.71e-13 | yes | 9.7 |
| noisy_rel_perm100_seed1 | rust | 6.71e-13 | yes | 14.9 |
| noisy_rel_perm100_seed42 | libsvm | 6.71e-13 | yes | 17.8 |
| noisy_rel_perm100_seed42 | rust | 6.71e-13 | yes | 9.1 |
| noisy_rel_perm100_seed20260814 | libsvm | 6.71e-13 | yes | 14.0 |
| noisy_rel_perm100_seed20260814 | rust | 6.71e-13 | yes | 10.1 |
| noisy_qn_rel_perm100 | libsvm | 7.39e-16 | yes | 8.5 |
| noisy_qn_rel_perm100 | rust | 7.39e-16 | yes | 6.6 |
| noisy_abs_perm100 | libsvm | 7.39e-13 | yes | 13.2 |
| noisy_abs_perm100 | rust | 7.39e-13 | yes | 11.3 |
| noisy_absnosum_perm100 | libsvm | 1.41e-11 | yes | 14.7 |
| noisy_absnosum_perm100 | rust | 1.41e-11 | yes | 10.1 |

Worst observed maximum absolute error: **1.41e-11** - PASS.

All tested `perm=100` P-value columns: **exactly identical** - PASS.

Wall times were collected on a shared development machine and are retained as
diagnostic data, not as a controlled performance claim. See the benchmark
report for the dedicated benchmark suite.
