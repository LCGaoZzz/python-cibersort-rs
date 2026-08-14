"""Command-line interface:  cibersort SIG_MATRIX MIXTURE_FILE [options]

BLAS thread pools are pinned to 1 *before* numpy loads, so parallel worker
threads never oversubscribe BLAS.
"""

from __future__ import annotations

import argparse
import os
import sys


def _pin_blas_threads() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cibersort",
        description="High-performance Python re-implementation of CIBERSORT R v1.04 "
                    "(no R dependency). Output matches the original R script.")
    p.add_argument("sig_matrix", help="signature matrix TSV (genes x cell types, first column = gene names)")
    p.add_argument("mixture_file", help="mixture TSV (genes x samples, first column = gene names)")
    p.add_argument("-o", "--out", default="CIBERSORT-Results.txt",
                   help="output TSV path (default: CIBERSORT-Results.txt)")
    p.add_argument("--perm", type=int, default=0,
                   help="number of permutations for P-values (default: 0)")
    p.add_argument("--qn", dest="qn", action="store_true", default=True,
                   help="quantile-normalise the mixture (default)")
    p.add_argument("--no-qn", dest="qn", action="store_false",
                   help="disable quantile normalisation")
    p.add_argument("--absolute", action="store_true",
                   help="run in absolute mode")
    p.add_argument("--abs-method", choices=["sig.score", "no.sumto1"], default="sig.score",
                   help="absolute-mode method (default: sig.score)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (required when --perm > 0; matches R set.seed)")
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("CIBERSORT_THREADS", "1")),
                   help="parallel worker threads over samples/permutations/nu "
                        "(default: 1, or CIBERSORT_THREADS)")
    p.add_argument("--engine", choices=["auto", "libsvm", "numpy", "rust"], default="auto",
                   help="SVR engine (default: auto = rust if built, else libsvm)")
    return p


def main(argv: list[str] | None = None) -> int:
    _pin_blas_threads()
    args = build_parser().parse_args(argv)

    engine = args.engine
    if engine == "auto":
        try:
            from . import _native  # noqa: F401
            engine = "rust"
        except Exception:
            engine = "libsvm"
    if engine == "rust":
        try:
            from . import _native  # noqa: F401
        except Exception:
            print("error: engine 'rust' requested but the bundled native extension "
                  "is unavailable; install a platform wheel or build with maturin",
                  file=sys.stderr)
            return 2

    from .core import cibersort  # deferred: after BLAS pinning

    res = cibersort(
        args.sig_matrix,
        args.mixture_file,
        perm=args.perm,
        QN=args.qn,
        absolute=args.absolute,
        abs_method=args.abs_method,
        seed=args.seed,
        threads=args.threads,
        engine=engine,
    )
    res.write(args.out)
    print(f"cibersort: wrote {args.out}  ({len(res.table)} samples, "
          f"perm={args.perm}, QN={args.qn}, absolute={args.absolute}, "
          f"abs_method={args.abs_method}, engine={engine}, threads={args.threads})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
