//! Exact port of preprocessCore::normalize.quantiles (qnorm_c / handleNA path).
//!
//! Verified bit-exact against the CRAN Windows binary of preprocessCore
//! 1.72.0 via direct ctypes calls into preprocessCore.dll and via in-memory
//! comparison inside R (0/333 rows differ): the binary computes
//!   row_mean[i] += sorted_col_j[i] / cols      (f64 divsd+addsd, j ascending)
//! then assigns by averaged ranks; ties with fractional part > 0.4 get
//! 0.5*(target[fl-1] + target[fl]) computed as (a+b)*0.5.

/// Quantile-normalise columns of `y` (row-major, rows x cols).
pub fn quantile_normalize(y: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    // 1) target: row_mean[i] += sorted_col_j[i] / cols, column by column
    let inv_cols = cols as f64;
    let mut row_mean = vec![0.0f64; rows];
    let mut col_buf = vec![0.0f64; rows];
    for j in 0..cols {
        for i in 0..rows {
            col_buf[i] = y[i * cols + j];
        }
        col_buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..rows {
            row_mean[i] += col_buf[i] / inv_cols;
        }
    }

    // 2) distribute with averaged ranks for ties (get_ranks: (i+k+2)*0.5)
    let mut out = vec![0.0f64; rows * cols];
    let mut order: Vec<usize> = (0..rows).collect();
    let mut ranks = vec![0.0f64; rows];
    for j in 0..cols {
        for i in 0..rows {
            col_buf[i] = y[i * cols + j];
        }
        order.sort_by(|&a, &b| col_buf[a].partial_cmp(&col_buf[b]).unwrap());
        let sv: Vec<f64> = order.iter().map(|&i| col_buf[i]).collect();
        let mut i = 0;
        while i < rows {
            let mut k = i;
            while k < rows - 1 && sv[k] == sv[k + 1] {
                k += 1;
            }
            if k != i {
                let r = (i + k + 2) as f64 * 0.5;
                for t in i..=k {
                    ranks[t] = r;
                }
            } else {
                ranks[i] = (i + 1) as f64;
            }
            i = k + 1;
        }
        for t in 0..rows {
            let fl = ranks[t].floor() as usize; // 1-based floor
            let frac = ranks[t] - fl as f64;
            let lo = row_mean[fl - 1];
            let val = if frac > 0.4 {
                (lo + row_mean[fl.min(rows - 1)]) * 0.5
            } else {
                lo
            };
            out[order[t] * cols + j] = val;
        }
    }
    out
}
