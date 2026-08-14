"""Pure-NumPy port of libsvm's nu-SVR solver (linear kernel).

Faithful port of ``solve_nu_svr`` + ``Solver_NU`` from libsvm svm.cpp
(shrinking disabled, i.e. the ``-h 0`` path; the optimum reached satisfies
the same eps-KKT criterion). libsvm stores cached Q rows as ``Qfloat``
(float32); we do the same. The Gram matrix is built with BLAS (float64)
instead of libsvm's sequential sparse dot products, so Q values can differ
from libsvm's by ~1 ulp in float64 before the float32 cast — in practice
solutions agree with libsvm to well below the 1e-6 regression gate.

This engine exists as the *pure-Python* performance baseline. It is exact
(no approximation) but much slower than native libsvm / Rust.
"""

from __future__ import annotations

import numpy as np

_LOWER, _UPPER, _FREE = 0, 1, 2
_TAU = 1e-12
_INF = np.inf


def solve_nu_svr_numpy(X: np.ndarray, y: np.ndarray, nu: float,
                       C: float = 1.0, eps: float = 1e-3,
                       p_loss: float = 0.1) -> np.ndarray:
    """Return the alpha vector (length l) of libsvm's solve_nu_svr."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    l = y.shape[0]

    # ---- SVR_Q -----------------------------------------------------------
    Gf = (X @ X.T).astype(np.float32)          # cached Q rows (Qfloat)
    QD1 = np.einsum("ij,ij->i", X, X)          # kernel(k, k), float64
    QD = np.concatenate([QD1, QD1])
    index = np.concatenate([np.arange(l), np.arange(l)])
    sign = np.concatenate([np.ones(l, np.int8), -np.ones(l, np.int8)])

    # ---- solve_nu_svr ----------------------------------------------------
    n = 2 * l
    alpha = np.empty(n, dtype=np.float64)
    p = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.int8)
    s = C * nu * l / 2.0
    for i in range(l):
        alpha[i] = alpha[i + l] = min(s, C)
        s -= alpha[i]
        p[i] = -y[i]
        ys[i] = 1
        p[i + l] = y[i]
        ys[i + l] = -1

    # ---- Solver state ----------------------------------------------------
    alpha_status = np.where(alpha <= 0, _LOWER, np.where(alpha >= C, _UPPER, _FREE))
    active_set = np.arange(n)
    active_size = n

    def get_Q(i: int, length: int) -> np.ndarray:
        """SVR_Q.get_Q: buf[j] = sign[i]*sign[j]*Qrow[index[j]] (float32)."""
        row = Gf[index[i]]                      # float32, length l
        sgn = (sign[i] * sign[:length]).astype(np.float32)
        return sgn * row[index[:length]]

    G = p.copy()
    G_bar = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if alpha_status[i] != _LOWER:
            qi = get_Q(i, n).astype(np.float64)
            G += alpha[i] * qi
            if alpha_status[i] == _UPPER:
                G_bar += C * qi

    max_iter = max(10_000_000, 100 * n)
    it = 0
    while it < max_iter:
        # --- Solver_NU::select_working_set -------------------------------
        a = slice(0, active_size)
        Ga = G[a]; ya = ys[a]; sta = alpha_status[a]
        # positive part: y==+1 & not upper -> -G ; y==-1 & not lower -> G
        cand_p = np.where((ya == 1) & (sta != _UPPER), -Ga, -np.inf)
        ip = int(np.flatnonzero(cand_p == cand_p.max())[-1]) if np.isfinite(cand_p.max()) else -1
        Gmaxp = cand_p[ip] if ip >= 0 else -np.inf
        cand_n = np.where((ya == -1) & (sta != _LOWER), Ga, -np.inf)
        inx = int(np.flatnonzero(cand_n == cand_n.max())[-1]) if np.isfinite(cand_n.max()) else -1
        Gmaxn = cand_n[inx] if inx >= 0 else -np.inf

        Q_ip = get_Q(ip, active_size).astype(np.float64) if ip != -1 else None
        Q_in = get_Q(inx, active_size).astype(np.float64) if inx != -1 else None

        # j scan (vectorised; identical arithmetic and tie semantics to the
        # scalar libsvm loop: per-branch masks, last exact-tie wins).
        QD_a = QD[a]
        jp = (ya == 1) & (sta != _LOWER)
        jn = (ya == -1) & (sta != _UPPER)
        Gmaxp2 = Ga[jp].max() if jp.any() else -np.inf
        Gmaxn2 = (-Ga[jn]).max() if jn.any() else -np.inf

        od = np.full(active_size, np.inf)
        if ip != -1 and jp.any():
            gd = Gmaxp + Ga[jp]
            qc = QD[ip] + QD_a[jp] - 2.0 * Q_ip[jp]
            pos = gd > 0
            od_j = np.full(int(jp.sum()), np.inf)
            od_j[pos] = -(gd[pos] * gd[pos]) / np.where(qc[pos] > 0, qc[pos], _TAU)
            od[jp] = od_j
        if inx != -1 and jn.any():
            gd = Gmaxn - Ga[jn]
            qc = QD[inx] + QD_a[jn] - 2.0 * Q_in[jn]
            pos = gd > 0
            od_j = np.full(int(jn.sum()), np.inf)
            od_j[pos] = -(gd[pos] * gd[pos]) / np.where(qc[pos] > 0, qc[pos], _TAU)
            od[jn] = od_j

        if np.isinf(od).all():
            Gmin_idx = -1
        else:
            od_min = od.min()
            Gmin_idx = int(np.flatnonzero(od == od_min)[-1])  # last tie wins

        if max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < eps or Gmin_idx == -1:
            break

        i_idx = ip if ya[Gmin_idx] == 1 else inx
        j_idx = Gmin_idx

        # --- update alpha[i], alpha[j] (y[i] != y[j] branch always) ------
        Q_i = get_Q(i_idx, active_size).astype(np.float64)
        Q_j = get_Q(j_idx, active_size).astype(np.float64)
        old_ai = alpha[i_idx]
        old_aj = alpha[j_idx]

        if ys[i_idx] != ys[j_idx]:
            qc = QD[i_idx] + QD[j_idx] + 2.0 * Q_i[j_idx]
            if qc <= 0:
                qc = _TAU
            delta = (-G[i_idx] - G[j_idx]) / qc
            diff = alpha[i_idx] - alpha[j_idx]
            alpha[i_idx] += delta
            alpha[j_idx] += delta
            if diff > 0:
                if alpha[j_idx] < 0:
                    alpha[j_idx] = 0.0
                    alpha[i_idx] = diff
            else:
                if alpha[i_idx] < 0:
                    alpha[i_idx] = 0.0
                    alpha[j_idx] = -diff
            if diff > 0.0:  # C_i - C_j == 0 (equal C)
                if alpha[i_idx] > C:
                    alpha[i_idx] = C
                    alpha[j_idx] = C - diff
            else:
                if alpha[j_idx] > C:
                    alpha[j_idx] = C
                    alpha[i_idx] = C + diff
        else:
            qc = QD[i_idx] + QD[j_idx] - 2.0 * Q_i[j_idx]
            if qc <= 0:
                qc = _TAU
            delta = (G[i_idx] - G[j_idx]) / qc
            sm = alpha[i_idx] + alpha[j_idx]
            alpha[i_idx] -= delta
            alpha[j_idx] += delta
            if sm > C:
                if alpha[i_idx] > C:
                    alpha[i_idx] = C
                    alpha[j_idx] = sm - C
            else:
                if alpha[j_idx] < 0:
                    alpha[j_idx] = 0.0
                    alpha[i_idx] = sm
            if sm > C:
                if alpha[j_idx] > C:
                    alpha[j_idx] = C
                    alpha[i_idx] = sm - C
            else:
                if alpha[i_idx] < 0:
                    alpha[i_idx] = 0.0
                    alpha[j_idx] = sm

        dai = alpha[i_idx] - old_ai
        daj = alpha[j_idx] - old_aj
        G[a] += Q_i * dai + Q_j * daj

        # alpha_status / G_bar updates
        ui = alpha_status[i_idx] == _UPPER
        uj = alpha_status[j_idx] == _UPPER
        for idx in (i_idx, j_idx):
            alpha_status[idx] = _LOWER if alpha[idx] <= 0 else (_UPPER if alpha[idx] >= C else _FREE)
        if ui != (alpha_status[i_idx] == _UPPER):
            qi_full = get_Q(i_idx, n).astype(np.float64)
            G_bar += (-C if ui else C) * qi_full
        if uj != (alpha_status[j_idx] == _UPPER):
            qj_full = get_Q(j_idx, n).astype(np.float64)
            G_bar += (-C if uj else C) * qj_full
        it += 1

    # ---- solution ---------------------------------------------------------
    out = np.empty(n, dtype=np.float64)
    out[active_set] = alpha
    return out[:l] - out[l:]


def fit_nusvr_numpy(X: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray:
    """nu-SVR fit -> dense weight vector w = alpha @ X (like e1071 t(coefs)%*%SV)."""
    alpha = solve_nu_svr_numpy(X, y, nu)
    nz = alpha != 0
    return alpha[nz] @ X[nz]
