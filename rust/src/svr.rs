//! Faithful Rust port of libsvm's nu-SVR solver (linear kernel), from svm.cpp
//! solve_nu_svr + Solver_NU (with shrinking, matching e1071 defaults).
//!
//! Bit-faithful choices:
//! * Q rows cached as f32 (libsvm's Qfloat), computed from f64 sequential
//!   dot products (same order as libsvm's sparse dot over feature index).
//! * Same working-set selection, tie semantics (>= / <=, last wins), update
//!   branches, shrinking, unshrink, and gradient reconstruction as svm.cpp.
//! * The Gram depends only on X, so it is computed once and shared by all
//!   fits of a CIBERSORT run.

const TAU: f64 = 1e-12;

const LOWER: u8 = 0;
const UPPER: u8 = 1;
const FREE: u8 = 2;

/// Sequential f64 dot product (libsvm order: ascending feature index).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for k in 0..a.len() {
        s += a[k] * b[k];
    }
    s
}

/// Shared Q cache for one signature matrix X (row-major l x d).
pub struct Gram {
    pub data: Vec<f32>, // l x l, data[i*l+j] = f32(dot(x_i, x_j))
    pub qd1: Vec<f64>,  // l, kernel(k, k)
}

impl Gram {
    pub fn new(x: &[f64], l: usize, d: usize) -> Self {
        let mut data = vec![0f32; l * l];
        for i in 0..l {
            let xi = &x[i * d..(i + 1) * d];
            for j in i..l {
                let v = dot(xi, &x[j * d..(j + 1) * d]) as f32;
                data[i * l + j] = v;
                data[j * l + i] = v;
            }
        }
        let mut qd1 = vec![0.0; l];
        for k in 0..l {
            qd1[k] = dot(&x[k * d..(k + 1) * d], &x[k * d..(k + 1) * d]);
        }
        Gram { data, qd1 }
    }
}

pub struct NuSvr<'a> {
    gram: &'a Gram,
    l: usize,
    qd: Vec<f64>,           // 2l
    sign: Vec<i8>,          // 2l
    index: Vec<usize>,      // 2l
    y: Vec<i8>,             // 2l
    p: Vec<f64>,            // 2l
    g: Vec<f64>,            // 2l
    g_bar: Vec<f64>,        // 2l
    alpha: Vec<f64>,        // 2l
    alpha_status: Vec<u8>,  // 2l
    active_set: Vec<usize>, // 2l
    active_size: usize,
    eps: f64,
    c: f64,
    unshrink: bool,
    // signed permuted Gram: qs[i*n+j] = sign[i]*sign[j]*gram[index[i]][index[j]]
    // (n = 2l), kept in sync by swap_index so get_Q is a zero-copy row slice;
    // the f32 values are identical to the on-the-fly gather.
    qs: Vec<f32>,
}

impl<'a> NuSvr<'a> {
    pub fn new(gram: &'a Gram, l: usize) -> Self {
        let mut qd = vec![0.0; 2 * l];
        for k in 0..l {
            qd[k] = gram.qd1[k];
            qd[k + l] = gram.qd1[k];
        }
        let n = 2 * l;
        NuSvr {
            gram,
            l,
            qd,
            sign: vec![0; 2 * l],
            index: vec![0; 2 * l],
            y: vec![0; 2 * l],
            p: vec![0.0; 2 * l],
            g: vec![0.0; 2 * l],
            g_bar: vec![0.0; 2 * l],
            alpha: vec![0.0; 2 * l],
            alpha_status: vec![0; 2 * l],
            active_set: vec![0; 2 * l],
            active_size: 0,
            eps: 0.0,
            c: 0.0,
            unshrink: false,
            qs: vec![0.0; n * n],
        }
    }

    /// SVR_Q.get_Q(i, len) as a zero-copy slice of the signed permuted Gram.
    #[inline]
    fn get_q(&self, i: usize, len: usize) -> &[f32] {
        let n = 2 * self.l;
        &self.qs[i * n..i * n + len]
    }

    #[inline]
    fn is_upper(&self, i: usize) -> bool {
        self.alpha_status[i] == UPPER
    }
    #[inline]
    fn is_lower(&self, i: usize) -> bool {
        self.alpha_status[i] == LOWER
    }
    #[inline]
    fn update_status(&mut self, i: usize) {
        self.alpha_status[i] = if self.alpha[i] >= self.c {
            UPPER
        } else if self.alpha[i] <= 0.0 {
            LOWER
        } else {
            FREE
        };
    }

    fn swap_index(&mut self, i: usize, j: usize) {
        self.sign.swap(i, j);
        self.index.swap(i, j);
        self.qd.swap(i, j);
        self.y.swap(i, j);
        self.g.swap(i, j);
        self.alpha_status.swap(i, j);
        self.alpha.swap(i, j);
        self.p.swap(i, j);
        self.active_set.swap(i, j);
        self.g_bar.swap(i, j);
        // keep qs = sign-permuted gram in sync: swap rows i,j then cols i,j
        let n = 2 * self.l;
        for c in 0..n {
            self.qs.swap(i * n + c, j * n + c);
        }
        for r in 0..n {
            self.qs.swap(r * n + i, r * n + j);
        }
    }

    fn reconstruct_gradient(&mut self) {
        if self.active_size == 2 * self.l {
            return;
        }
        let n = 2 * self.l;
        for j in self.active_size..n {
            self.g[j] = self.g_bar[j] + self.p[j];
        }
        let nr_free = (0..self.active_size)
            .filter(|&j| self.alpha_status[j] == FREE)
            .count();
        if nr_free * n > 2 * self.active_size * (n - self.active_size) {
            for i in self.active_size..n {
                let qi = &self.qs[i * n..i * n + self.active_size];
                for j in 0..self.active_size {
                    if self.alpha_status[j] == FREE {
                        self.g[i] += self.alpha[j] * qi[j] as f64;
                    }
                }
            }
        } else {
            for i in 0..self.active_size {
                if self.alpha_status[i] == FREE {
                    let qi = &self.qs[i * n..(i + 1) * n];
                    let alpha_i = self.alpha[i];
                    for j in self.active_size..n {
                        self.g[j] += alpha_i * qi[j] as f64;
                    }
                }
            }
        }
    }

    /// Solver_NU::select_working_set -> Ok((i,j)) or Err(()) when optimal
    fn select_working_set(&mut self) -> Result<(usize, usize), ()> {
        let mut gmaxp = f64::NEG_INFINITY;
        let mut gmaxp2 = f64::NEG_INFINITY;
        let mut gmaxp_idx: i64 = -1;
        let mut gmaxn = f64::NEG_INFINITY;
        let mut gmaxn2 = f64::NEG_INFINITY;
        let mut gmaxn_idx: i64 = -1;
        let mut gmin_idx: i64 = -1;
        let mut obj_diff_min = f64::INFINITY;

        for t in 0..self.active_size {
            if self.y[t] == 1 {
                if !self.is_upper(t) && -self.g[t] >= gmaxp {
                    gmaxp = -self.g[t];
                    gmaxp_idx = t as i64;
                }
            } else if !self.is_lower(t) && self.g[t] >= gmaxn {
                gmaxn = self.g[t];
                gmaxn_idx = t as i64;
            }
        }

        let ip = gmaxp_idx;
        let inx = gmaxn_idx;
        let nn = 2 * self.l;
        let act = self.active_size;
        let qip: Option<&[f32]> = if ip != -1 {
            Some(&self.qs[ip as usize * nn..ip as usize * nn + act])
        } else {
            None
        };
        let qin: Option<&[f32]> = if inx != -1 {
            Some(&self.qs[inx as usize * nn..inx as usize * nn + act])
        } else {
            None
        };

        for j in 0..self.active_size {
            if self.y[j] == 1 {
                if !self.is_lower(j) {
                    let grad_diff = gmaxp + self.g[j];
                    if self.g[j] >= gmaxp2 {
                        gmaxp2 = self.g[j];
                    }
                    if grad_diff > 0.0 {
                        let qc = self.qd[ip as usize] + self.qd[j]
                            - 2.0 * qip.as_ref().unwrap()[j] as f64;
                        let od = -(grad_diff * grad_diff) / if qc > 0.0 { qc } else { TAU };
                        if od <= obj_diff_min {
                            gmin_idx = j as i64;
                            obj_diff_min = od;
                        }
                    }
                }
            } else if !self.is_upper(j) {
                let grad_diff = gmaxn - self.g[j];
                if -self.g[j] >= gmaxn2 {
                    gmaxn2 = -self.g[j];
                }
                if grad_diff > 0.0 {
                    let qc =
                        self.qd[inx as usize] + self.qd[j] - 2.0 * qin.as_ref().unwrap()[j] as f64;
                    let od = -(grad_diff * grad_diff) / if qc > 0.0 { qc } else { TAU };
                    if od <= obj_diff_min {
                        gmin_idx = j as i64;
                        obj_diff_min = od;
                    }
                }
            }
        }

        if (gmaxp + gmaxp2).max(gmaxn + gmaxn2) < self.eps || gmin_idx == -1 {
            return Err(());
        }
        let j = gmin_idx as usize;
        let i = if self.y[j] == 1 {
            gmaxp_idx as usize
        } else {
            gmaxn_idx as usize
        };
        Ok((i, j))
    }

    fn be_shrunk(&self, i: usize, gmax1: f64, gmax2: f64, gmax3: f64, gmax4: f64) -> bool {
        if self.is_upper(i) {
            if self.y[i] == 1 {
                -self.g[i] > gmax1
            } else {
                -self.g[i] > gmax4
            }
        } else if self.is_lower(i) {
            if self.y[i] == 1 {
                self.g[i] > gmax2
            } else {
                self.g[i] > gmax3
            }
        } else {
            false
        }
    }

    fn do_shrinking(&mut self) {
        let mut gmax1 = f64::NEG_INFINITY;
        let mut gmax2 = f64::NEG_INFINITY;
        let mut gmax3 = f64::NEG_INFINITY;
        let mut gmax4 = f64::NEG_INFINITY;
        for i in 0..self.active_size {
            if !self.is_upper(i) {
                if self.y[i] == 1 {
                    if -self.g[i] > gmax1 {
                        gmax1 = -self.g[i];
                    }
                } else if -self.g[i] > gmax4 {
                    gmax4 = -self.g[i];
                }
            }
            if !self.is_lower(i) {
                if self.y[i] == 1 {
                    if self.g[i] > gmax2 {
                        gmax2 = self.g[i];
                    }
                } else if self.g[i] > gmax3 {
                    gmax3 = self.g[i];
                }
            }
        }
        if !self.unshrink && (gmax1 + gmax2).max(gmax3 + gmax4) <= self.eps * 10.0 {
            self.unshrink = true;
            self.reconstruct_gradient();
            self.active_size = 2 * self.l;
        }
        let mut i = 0;
        while i < self.active_size {
            if self.be_shrunk(i, gmax1, gmax2, gmax3, gmax4) {
                self.active_size -= 1;
                while self.active_size > i {
                    if !self.be_shrunk(self.active_size, gmax1, gmax2, gmax3, gmax4) {
                        self.swap_index(i, self.active_size);
                        break;
                    }
                    self.active_size -= 1;
                }
            }
            i += 1;
        }
    }

    /// solve_nu_svr + Solver_NU::Solve; returns alpha (length l).
    pub fn solve(
        &mut self,
        y_target: &[f64],
        nu: f64,
        c: f64,
        eps: f64,
        shrinking: bool,
    ) -> Vec<f64> {
        let l = self.l;
        let n = 2 * l;
        self.c = c;
        self.eps = eps;
        self.unshrink = false;

        // solve_nu_svr init
        let mut sum = c * nu * l as f64 / 2.0;
        for i in 0..l {
            let a = sum.min(c);
            self.alpha[i] = a;
            self.alpha[i + l] = a;
            sum -= a;
            self.p[i] = -y_target[i];
            self.y[i] = 1;
            self.p[i + l] = y_target[i];
            self.y[i + l] = -1;
        }
        for k in 0..l {
            self.sign[k] = 1;
            self.sign[k + l] = -1;
            self.index[k] = k;
            self.index[k + l] = k;
        }
        // build the signed permuted Gram once per fit
        {
            let NuSvr { gram, index, sign, qs, l, .. } = self;
            let li = *l;
            for i in 0..n {
                let real_i = index[i];
                let si = sign[i];
                let src = &gram.data[real_i * li..(real_i + 1) * li];
                for j in 0..n {
                    let v = src[index[j]];
                    qs[i * n + j] = if si * sign[j] == 1 { v } else { -v };
                }
            }
        }
        for i in 0..n {
            self.update_status(i);
            self.active_set[i] = i;
            self.g[i] = self.p[i];
            self.g_bar[i] = 0.0;
        }
        self.active_size = n;

        // initial gradient
        for i in 0..n {
            if !self.is_lower(i) {
                let qi = &self.qs[i * n..(i + 1) * n];
                let alpha_i = self.alpha[i];
                for j in 0..n {
                    self.g[j] += alpha_i * qi[j] as f64;
                }
                if self.is_upper(i) {
                    for j in 0..n {
                        self.g_bar[j] += c * qi[j] as f64;
                    }
                }
            }
        }

        let max_iter = 10_000_000usize.max(100 * n);
        let mut counter = n.min(1000) + 1;
        let mut iter = 0usize;

        while iter < max_iter {
            counter -= 1;
            if counter == 0 {
                counter = n.min(1000);
                if shrinking {
                    self.do_shrinking();
                }
            }
            let (i, j) = match self.select_working_set() {
                Ok(pair) => pair,
                Err(()) => {
                    self.reconstruct_gradient();
                    self.active_size = n;
                    match self.select_working_set() {
                        Err(()) => break,
                        Ok(pair) => {
                            counter = 1;
                            pair
                        }
                    }
                }
            };
            iter += 1;

            let active = self.active_size;
            let qi: &[f32] = &self.qs[i * n..i * n + active];
            let qj: &[f32] = &self.qs[j * n..j * n + active];

            let old_ai = self.alpha[i];
            let old_aj = self.alpha[j];

            if self.y[i] != self.y[j] {
                let mut qc = self.qd[i] + self.qd[j] + 2.0 * qi[j] as f64;
                if qc <= 0.0 {
                    qc = TAU;
                }
                let delta = (-self.g[i] - self.g[j]) / qc;
                let diff = self.alpha[i] - self.alpha[j];
                self.alpha[i] += delta;
                self.alpha[j] += delta;
                if diff > 0.0 {
                    if self.alpha[j] < 0.0 {
                        self.alpha[j] = 0.0;
                        self.alpha[i] = diff;
                    }
                } else if self.alpha[i] < 0.0 {
                    self.alpha[i] = 0.0;
                    self.alpha[j] = -diff;
                }
                if diff > 0.0 {
                    // C_i - C_j == 0 (equal C)
                    if self.alpha[i] > self.c {
                        self.alpha[i] = self.c;
                        self.alpha[j] = self.c - diff;
                    }
                } else if self.alpha[j] > self.c {
                    self.alpha[j] = self.c;
                    self.alpha[i] = self.c + diff;
                }
            } else {
                let mut qc = self.qd[i] + self.qd[j] - 2.0 * qi[j] as f64;
                if qc <= 0.0 {
                    qc = TAU;
                }
                let delta = (self.g[i] - self.g[j]) / qc;
                let sm = self.alpha[i] + self.alpha[j];
                self.alpha[i] -= delta;
                self.alpha[j] += delta;
                if sm > self.c {
                    if self.alpha[i] > self.c {
                        self.alpha[i] = self.c;
                        self.alpha[j] = sm - self.c;
                    }
                } else if self.alpha[j] < 0.0 {
                    self.alpha[j] = 0.0;
                    self.alpha[i] = sm;
                }
                if sm > self.c {
                    if self.alpha[j] > self.c {
                        self.alpha[j] = self.c;
                        self.alpha[i] = sm - self.c;
                    }
                } else if self.alpha[i] < 0.0 {
                    self.alpha[i] = 0.0;
                    self.alpha[j] = sm;
                }
            }

            let dai = self.alpha[i] - old_ai;
            let daj = self.alpha[j] - old_aj;
            let active = self.active_size;
            for (gk, (qik, qjk)) in self.g[..active]
                .iter_mut()
                .zip(qi[..active].iter().zip(qj[..active].iter()))
            {
                *gk += *qik as f64 * dai + *qjk as f64 * daj;
            }

            let ui = self.is_upper(i);
            let uj = self.is_upper(j);
            self.update_status(i);
            self.update_status(j);
            if ui != self.is_upper(i) {
                let qif = &self.qs[i * n..(i + 1) * n];
                let sgn = if ui { -self.c } else { self.c };
                for (gbk, qk) in self.g_bar.iter_mut().zip(qif.iter()) {
                    *gbk += sgn * *qk as f64;
                }
            }
            if uj != self.is_upper(j) {
                let qjf = &self.qs[j * n..(j + 1) * n];
                let sgn = if uj { -self.c } else { self.c };
                for (gbk, qk) in self.g_bar.iter_mut().zip(qjf.iter()) {
                    *gbk += sgn * *qk as f64;
                }
            }
        }

        // put back the solution
        let mut out = vec![0.0f64; n];
        for i in 0..n {
            out[self.active_set[i]] = self.alpha[i];
        }
        let mut alpha = vec![0.0f64; l];
        for i in 0..l {
            alpha[i] = out[i] - out[i + l];
        }
        alpha
    }
}

/// Fit nu-SVR (libsvm defaults: C=1, eps=1e-3, shrinking on) and return
/// w = coefs . SV (dense, length d), like e1071's t(model$coefs) %*% model$SV.
pub fn nusvr_w_with_gram(
    x: &[f64],
    l: usize,
    d: usize,
    gram: &Gram,
    y: &[f64],
    nu: f64,
) -> Vec<f64> {
    let mut solver = NuSvr::new(gram, l);
    let alpha = solver.solve(y, nu, 1.0, 1e-3, true);
    let mut w = vec![0.0f64; d];
    for i in 0..l {
        let a = alpha[i];
        if a != 0.0 {
            let xi = &x[i * d..(i + 1) * d];
            for k in 0..d {
                w[k] += a * xi[k];
            }
        }
    }
    w
}

pub fn nusvr_w(x: &[f64], l: usize, d: usize, y: &[f64], nu: f64) -> Vec<f64> {
    let gram = Gram::new(x, l, d);
    nusvr_w_with_gram(x, l, d, &gram, y, nu)
}
