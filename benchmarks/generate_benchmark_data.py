"""Generate small/medium/large CIBERSORT benchmark datasets (deterministic).

Layout per size tag: benchmarks/data/<tag>/signature_matrix.tsv, mixture.tsv,
ground_truth_fractions.tsv, meta.json.

The generative model mirrors test_data/permutation_safe: log-normal-ish
signature matrix, mixtures = signature %*% fractions + multiplicative noise,
values kept in log2 space (< 50) so CIBERSORT's anti-log path is exercised.

Sizes (genes x cell_types x samples, perm):
  small  100 x 8  x 4   perm=100
  medium 300 x 12 x 20  perm=100
  large  547 x 22 x 50  perm=100   (LM22-scale)
"""

from __future__ import annotations

import json
import os
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

SIZES = {
    "small": dict(genes=100, cell_types=8, samples=4, perm=100),
    "medium": dict(genes=300, cell_types=12, samples=20, perm=100),
    "large": dict(genes=547, cell_types=22, samples=50, perm=100),
}

SEED = 20260813


def make_size(tag: str, genes: int, cell_types: int, samples: int) -> None:
    rng = np.random.default_rng(SEED + zlib.crc32(tag.encode()) % 1000)
    out_dir = os.path.join(DATA, tag)
    os.makedirs(out_dir, exist_ok=True)

    gene_names = [f"BGene_{i:04d}" for i in range(1, genes + 1)]
    ct_names = [f"BCellType_{i:02d}" for i in range(1, cell_types + 1)]

    # signature matrix: cell-type specific markers over a log-scale baseline
    base = rng.gamma(shape=2.0, scale=1.5, size=(genes, 1))
    sig = np.repeat(base, cell_types, axis=1)
    marker_strength = rng.uniform(0.5, 4.0, size=(genes, cell_types))
    # each cell type up-marks a distinct gene block
    blocks = np.array_split(np.arange(genes), cell_types)
    for c, blk in enumerate(blocks):
        sig[blk, c] += marker_strength[blk, c]
    sig += rng.normal(0, 0.15, size=sig.shape)  # small per-cell jitter
    sig = np.clip(sig, 0.0, None)

    # ground-truth fractions (rows sum to 1)
    fr = rng.dirichlet(alpha=np.full(cell_types, 0.7), size=samples)
    # exact mixtures in linear space, then back to log2 with noise
    mix = sig @ fr.T
    mix = np.clip(mix, 1e-6, None)
    noise = rng.lognormal(mean=0.0, sigma=0.015, size=mix.shape)
    mix_noisy = np.log2(mix * noise)

    sig_df = pd.DataFrame(sig, index=gene_names, columns=ct_names)
    mix_df = pd.DataFrame(mix_noisy, index=gene_names,
                          columns=[f"BSample_{i:02d}" for i in range(1, samples + 1)])
    fr_df = pd.DataFrame(fr, index=[f"BSample_{i:02d}" for i in range(1, samples + 1)],
                         columns=ct_names)

    sig_df.to_csv(os.path.join(out_dir, "signature_matrix.tsv"), sep="\t")
    mix_df.to_csv(os.path.join(out_dir, "mixture.tsv"), sep="\t")
    fr_df.to_csv(os.path.join(out_dir, "ground_truth_fractions.tsv"), sep="\t")
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(dict(tag=tag, genes=genes, cell_types=cell_types,
                       samples=samples, perm=SIZES[tag]["perm"],
                       seed=SEED + zlib.crc32(tag.encode()) % 1000), fh, indent=2)
    print(f"{tag}: genes={genes} cell_types={cell_types} samples={samples} -> {out_dir}")


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    for tag, kw in SIZES.items():
        make_size(tag, **{k: v for k, v in kw.items() if k != "perm"})


if __name__ == "__main__":
    main()
