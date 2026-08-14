# Permutation-safe synthetic fixture

This deterministic fixture contains 330 genes, 22 synthetic cell types, and 18
synthetic mixtures. It is designed to exercise the `perm=100` path without the
all-negative coefficient degeneracy seen in very small signature matrices.

- `generate_fixture.R` generates the signature, fractions, and mixtures.
- `signature_matrix.tsv` is the synthetic reference matrix.
- `mixture.tsv` contains exact linear mixtures.
- `mixture_noisy.tsv` adds deterministic 1.5% log-normal noise.
- `ground_truth_fractions.tsv` contains the generating fractions.
- `oracle_*.tsv` contains numeric outputs generated with CIBERSORT v1.04.

The original CIBERSORT script and LM22 are not included. Oracle files are
numeric regression fixtures only. See the repository README and third-party
notices for attribution and usage considerations.
