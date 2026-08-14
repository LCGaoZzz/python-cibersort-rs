"""Run every oracle configuration with both engines and write a validation
summary (max abs error vs oracle per column group, P-value identity).

Usage:  python tests/run_validation.py [threads]
Writes: tests/validation_summary.md
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from python_cibersort import cibersort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conftest import FIXTURE_DIR, ORACLES, SIG, compare_to_oracle, read_oracle  # noqa: E402

ENGINES = ("libsvm", "rust")


def main() -> None:
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    lines = []
    lines.append("# Validation summary - python-cibersort-rs vs original R CIBERSORT v1.04\n")
    lines.append(f"Gate: proportions / Correlation / RMSE / Absolute score max abs error <= 1e-6; "
                 f"P-value exactly equal for perm=100. threads={threads}.\n")
    lines.append("| oracle | engine | max |err| (excl P) | P-value identical | wall (s) |")
    lines.append("|---|---|---|---|---|")
    worst = 0.0
    all_p = True
    for tag, (mixture, perm, qn, absolute, abs_method, seed) in ORACLES.items():
        oracle = read_oracle(os.path.join(FIXTURE_DIR, f"oracle_{tag}.tsv"))
        for engine in ENGINES:
            t0 = time.perf_counter()
            res = cibersort(SIG, mixture, perm=perm, QN=qn, absolute=absolute,
                            abs_method=abs_method, seed=seed, threads=threads,
                            engine=engine)
            dt = time.perf_counter() - t0
            max_err, p_id, _ = compare_to_oracle(res.table, oracle)
            worst = max(worst, max_err)
            all_p = all_p and (p_id or perm == 0)
            lines.append(f"| {tag} | {engine} | {max_err:.3g} | "
                         f"{'n/a (perm=0)' if perm == 0 else str(p_id)} | {dt:.1f} |")
            print(lines[-1], flush=True)
    lines.append(f"\nWorst max-abs-error across all runs: **{worst:.3g}** "
                 f"(gate 1e-6) - {'PASS' if worst <= 1e-6 else 'FAIL'}; "
                 f"P-values all identical: {'PASS' if all_p else 'FAIL'}.\n")
    out = os.path.join(HERE, "validation_summary.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
