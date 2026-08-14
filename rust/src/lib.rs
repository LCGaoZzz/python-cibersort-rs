//! Native Rust accelerator for python-cibersort-rs.
//!
//! Exact ports of: R's RNG/sample (rng.rs), preprocessCore QN (qn.rs),
//! libsvm nu-SVR (svr.rs), plus a rayon-parallel batch pipeline that
//! reproduces CIBERSORT.R's doPerm + per-sample loop semantics.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

mod qn;
mod rng;
mod svr;

use rng::RRng;
use svr::{Gram, NuSvr};

const NUS: [f64; 3] = [0.25, 0.5, 0.75];

/// R sample() without replacement, 1-based indices.
#[pyfunction]
fn sample_no_replace(seed: i64, n: u64, k: usize) -> Vec<u64> {
    let mut r = RRng::new(seed);
    r.sample_no_replace(n, k)
}

/// `perm` sequential draws of R sample(n, k) after one set.seed(seed):
/// returns perm*k 1-based indices, row-major (draw0, draw1, ...).
/// Bit-exact vs R (the doPerm draw stream).
#[pyfunction]
#[pyo3(signature = (seed, n, k, perm))]
fn perm_indices(seed: i64, n: u64, k: usize, perm: usize) -> Vec<u64> {
    let mut r = RRng::new(seed);
    let mut out = Vec::with_capacity(perm * k);
    for _ in 0..perm {
        out.extend(r.sample_no_replace(n, k));
    }
    out
}

/// R runif(m) after set.seed(seed).
#[pyfunction]
fn runif(seed: i64, m: usize) -> Vec<f64> {
    let mut r = RRng::new(seed);
    (0..m).map(|_| r.unif_rand()).collect()
}

/// preprocessCore::normalize.quantiles on a row-major (rows, cols) matrix.
#[pyfunction]
fn quantile_normalize(y: Vec<f64>, rows: usize, cols: usize) -> Vec<f64> {
    qn::quantile_normalize(&y, rows, cols)
}

/// One nu-SVR fit -> dense w (for testing/parity checks).
#[pyfunction]
fn nusvr_w(x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, nu: f64) -> Vec<f64> {
    let xa = x.as_array();
    let ya = y.as_array();
    let l = xa.nrows();
    let d = xa.ncols();
    let xs: Vec<f64> = xa.iter().copied().collect();
    let ys: Vec<f64> = ya.iter().copied().collect();
    svr::nusvr_w(&xs, l, d, &ys, nu)
}

/// Batch nu-SVR fits over many z-scored targets, parallelised with rayon.
/// Deterministic for any thread count.
///
/// * `x_std`: standardised signature matrix (genes x cell types), row-major.
/// * `targets`: z-scored targets (n_targets x genes), row-major — computed by
///   Python (R RNG draws + z-scores), so the solver inputs are bit-identical
///   across engines.
/// Returns flat w: [target0_nu0, target0_nu1, target0_nu2, target1_nu0, ...],
/// each of length d (cell types), w = alpha . X for that fit.
#[pyfunction]
#[pyo3(signature = (x_std, targets, threads))]
fn run_fits(
    py: Python<'_>,
    x_std: PyReadonlyArray2<f64>,
    targets: PyReadonlyArray2<f64>,
    threads: usize,
) -> PyResult<Vec<f64>> {
    let xa = x_std.as_array();
    let ta = targets.as_array();
    let g = xa.nrows();
    let d = xa.ncols();
    let n_targets = ta.nrows();
    let x: Vec<f64> = xa.iter().copied().collect();
    let tvec: Vec<f64> = ta.iter().copied().collect();

    py.allow_threads(move || {
        let gram = Gram::new(&x, g, d);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .build()
            .expect("rayon pool");
        let out: Vec<f64> = pool.install(|| {
            let mut tasks: Vec<(usize, usize)> = Vec::with_capacity(n_targets * 3);
            for t in 0..n_targets {
                for ni in 0..3 {
                    tasks.push((t, ni));
                }
            }
            let fits: Vec<((usize, usize), Vec<f64>)> = tasks
                .par_iter()
                .map(|&(t, ni)| {
                    let target = &tvec[t * g..(t + 1) * g];
                    let mut solver = NuSvr::new(&gram, g);
                    let alpha = solver.solve(target, NUS[ni], 1.0, 1e-3, true);
                    let mut w = vec![0.0f64; d];
                    for i in 0..g {
                        let a = alpha[i];
                        if a != 0.0 {
                            let xi = &x[i * d..(i + 1) * d];
                            for k in 0..d {
                                w[k] += a * xi[k];
                            }
                        }
                    }
                    ((t, ni), w)
                })
                .collect();
            let mut flat = vec![0.0f64; n_targets * 3 * d];
            for ((t, ni), w) in fits {
                flat[(t * 3 + ni) * d..(t * 3 + ni + 1) * d].copy_from_slice(&w);
            }
            flat
        });
        Ok(out)
    })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_no_replace, m)?)?;
    m.add_function(wrap_pyfunction!(perm_indices, m)?)?;
    m.add_function(wrap_pyfunction!(runif, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(nusvr_w, m)?)?;
    m.add_function(wrap_pyfunction!(run_fits, m)?)?;
    Ok(())
}
