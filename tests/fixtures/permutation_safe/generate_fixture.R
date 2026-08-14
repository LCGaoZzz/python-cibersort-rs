# Generate a deterministic fixture suitable for exercising CIBERSORT v1.04
# with permutations. The small root-level fixture has only four cell types and
# is intentionally retained as a fast perm=0 smoke test.

set.seed(20260814)
options(digits = 17)

n_cell_types <- 22L
markers_per_type <- 12L
n_housekeeping <- 66L
n_genes <- n_cell_types * markers_per_type + n_housekeeping
n_samples <- 18L

cell_types <- sprintf("CellType_%02d", seq_len(n_cell_types))
genes <- sprintf("Gene_%04d", seq_len(n_genes))
samples <- sprintf("Sample_%02d", seq_len(n_samples))

# Positive, heterogeneous background expression plus disjoint strong marker
# blocks. Mild secondary signals keep the matrix realistic without making the
# cell-type columns collinear.
signature <- matrix(
  runif(n_genes * n_cell_types, min = 25, max = 85),
  nrow = n_genes,
  ncol = n_cell_types,
  dimnames = list(genes, cell_types)
)

for (ct in seq_len(n_cell_types)) {
  first <- (ct - 1L) * markers_per_type + 1L
  marker_rows <- first:(first + markers_per_type - 1L)
  signature[marker_rows, ct] <- signature[marker_rows, ct] +
    runif(markers_per_type, min = 500, max = 900)

  secondary <- (ct %% n_cell_types) + 1L
  signature[marker_rows[seq(2L, markers_per_type, by = 3L)], secondary] <-
    signature[marker_rows[seq(2L, markers_per_type, by = 3L)], secondary] +
    runif(length(seq(2L, markers_per_type, by = 3L)), min = 60, max = 120)
}

# Build diverse but strictly positive cell fractions. Six samples contain a
# dominant cell type; the remaining samples are broad mixtures.
fractions <- matrix(
  rgamma(n_cell_types * n_samples, shape = 1.8, rate = 1),
  nrow = n_cell_types,
  ncol = n_samples,
  dimnames = list(cell_types, samples)
)
for (s in seq_len(min(6L, n_samples))) {
  fractions[, s] <- fractions[, s] / sum(fractions[, s]) * 0.4
  fractions[s, s] <- fractions[s, s] + 0.6
}
fractions <- sweep(fractions, 2L, colSums(fractions), "/")

mixture <- signature %*% fractions

# Deterministic mild multiplicative noise for a more realistic benchmark.
noise <- matrix(
  rnorm(length(mixture), mean = 0, sd = 0.015),
  nrow = nrow(mixture),
  ncol = ncol(mixture)
)
mixture_noisy <- pmax(mixture * exp(noise), .Machine$double.eps)
dimnames(mixture_noisy) <- dimnames(mixture)

write_tsv <- function(x, path) {
  write.table(
    x,
    file = path,
    sep = "\t",
    quote = FALSE,
    row.names = TRUE,
    col.names = NA
  )
}

write_tsv(signature, "signature_matrix.tsv")
write_tsv(mixture, "mixture.tsv")
write_tsv(mixture_noisy, "mixture_noisy.tsv")
write_tsv(t(fractions), "ground_truth_fractions.tsv")

metadata <- c(
  "fixture_version=2",
  "generator_seed=20260814",
  sprintf("genes=%d", n_genes),
  sprintf("cell_types=%d", n_cell_types),
  sprintf("samples=%d", n_samples),
  "noise_model=multiplicative_lognormal_sd_0.015",
  "intended_use=CIBERSORT_v1.04_perm0_and_perm100_correctness_benchmark"
)
writeLines(metadata, "fixture_metadata.txt")

cat("Generated permutation-safe CIBERSORT fixture\n")
cat(sprintf("signature: %d genes x %d cell types\n", n_genes, n_cell_types))
cat(sprintf("mixture: %d genes x %d samples\n", n_genes, n_samples))
