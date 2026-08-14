"""python-cibersort: high-performance Python re-implementation of CIBERSORT R v1.04.

Public API mirrors the R entry point:

    from python_cibersort import cibersort
    res = cibersort("sig_matrix.txt", "mixture.txt", perm=100, QN=True,
                    absolute=False, abs_method="sig.score", seed=42, threads=8)
    res.table            # pandas DataFrame identical to R's return object
    res.write("CIBERSORT-Results.txt")

`cibersort_all` returns relative + sig.score + no.sumto1 results from a
single batch of ν-SVR fits.
"""

from .core import (CIBERSORT, CibersortResult, cibersort, cibersort_all,
                   read_table)
from .qn import quantile_normalize
from .rrnd import RRng

__version__ = "0.1.0"
__all__ = ["cibersort", "cibersort_all", "CIBERSORT", "CibersortResult",
           "read_table", "quantile_normalize", "RRng", "__version__"]
