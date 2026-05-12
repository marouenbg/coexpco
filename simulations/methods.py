"""
Three implementations of the leave-one-out Pearson correlation matrix
C^{alpha - q} that LIONESS-style single-sample networks consume, plus
a LIONESS combiner that turns any (population, leave-one-out)
correlation pair into single-sample edges.

Notation
--------
For N samples and p genes:
  X       : (N, p) data matrix
  m_a     : column means of X (the "alpha" / population mean)
  C_a     : population (Pearson) correlation matrix, p x p
  C_aq[q] : leave-one-out Pearson correlation excluding sample q, p x p
            (the implementations below all return an (N, p, p) tensor)
  e_q     : LIONESS single-sample edge matrix for sample q,
            defined as e_q = N * C_a - (N - 1) * C_aq[q].

Methods
-------
Reference -- brute-force per-q recomputation in float64 with explicit
             centered residuals (no shortcut). Spot-checked against an
             mpmath calculation at 50 decimal digits.
M1        -- gpuLIONESS derivation: orthogonal Welford rank-1 downdate
             S_aq = S_a - (N/(N-1)) * r_q r_q^T with r_q = x_q - m_a.
             Float64.
M2        -- ISN-tractor's `incremental_pearson` path -- a port of the
             function `dense_isn` in
             https://github.com/GiadaLalli/ISN-tractor/blob/main/isn_tractor/ibisn.py
             which implements ISN-tractor Eq. 4 in closed form
             (sufficient-statistics rearrangement). Forced float64 by
             the upstream code.

The LIONESS combiner is shared and uses the population correlation
returned by `population_correlation(X)`, which is itself just the
centered-residual computation (the same algorithm `np.corrcoef`
implements).
"""

from __future__ import annotations

import numpy as np

try:
    import mpmath as _mp
    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False


# ---------------------------------------------------------------------------
# Shared population-correlation implementation
# ---------------------------------------------------------------------------

def population_correlation(X, dtype=np.float64):
    """
    Population Pearson correlation via centered residuals, float64.

    All three methods agree on the population correlation -- the
    algorithmic differences are confined to the leave-one-out step --
    so we share a single implementation for the LIONESS combiner.
    """
    X = np.asarray(X, dtype=dtype)
    m = X.mean(axis=0)
    R = X - m
    S = R.T @ R
    d = np.diag(S)
    return (S / np.sqrt(np.outer(d, d))).astype(np.float64)


# ---------------------------------------------------------------------------
# Reference: exact Pearson, brute-force per leave-one-out subset
# ---------------------------------------------------------------------------

def reference_loo_correlation(X):
    """
    For each held-out sample q, recompute the (N-1)-row Pearson
    correlation matrix from raw data using explicit centered residuals
    in float64. No online update, no sufficient-statistics shortcut.

    Spot-checked against mpmath at 50 decimal digits at small N, p.
    """
    X = np.asarray(X, dtype=np.float64)
    N, p = X.shape
    out = np.empty((N, p, p), dtype=np.float64)
    for q in range(N):
        Y = np.concatenate([X[:q], X[q + 1:]], axis=0)
        m = Y.mean(axis=0)
        R = Y - m
        S = R.T @ R
        d = np.diag(S)
        out[q] = S / np.sqrt(np.outer(d, d))
    return out


def reference_loo_correlation_mpmath(X, prec_digits=50):
    """
    Bit-exact reference at `prec_digits` decimal precision via mpmath.
    Slow; use for spot-checks at small N, p.
    """
    if not _HAS_MPMATH:
        raise RuntimeError("mpmath required.")
    _mp.mp.dps = prec_digits
    Xm = [[_mp.mpf(float(v)) for v in row] for row in X]
    N, p = len(Xm), len(Xm[0])

    def _corr_mp(Y):
        n = len(Y)
        m = [sum(Y[i][j] for i in range(n)) / n for j in range(p)]
        S = [[_mp.mpf(0)] * p for _ in range(p)]
        for i in range(n):
            r = [Y[i][j] - m[j] for j in range(p)]
            for a in range(p):
                ra = r[a]
                for b in range(a, p):
                    S[a][b] += ra * r[b]
        for a in range(p):
            for b in range(a):
                S[a][b] = S[b][a]
        d = [S[a][a] for a in range(p)]
        return [[S[a][b] / _mp.sqrt(d[a] * d[b]) for b in range(p)] for a in range(p)]

    out = np.empty((N, p, p), dtype=np.float64)
    for q in range(N):
        Y = Xm[:q] + Xm[q + 1:]
        C_aq = _corr_mp(Y)
        for a in range(p):
            for b in range(p):
                out[q, a, b] = float(C_aq[a][b])
    return out


# ---------------------------------------------------------------------------
# M1: gpuLIONESS / orthogonal Welford-LOO downdate
# ---------------------------------------------------------------------------

def m1_loo_correlation(X, dtype=np.float64):
    """
    Compute C_{alpha - q} for every q via the orthogonal Welford
    rank-1 downdate.

    Population sum of squares about the population mean is computed
    once. For each q, the orthogonality identity Sum_k (G_k - m_a) = 0
    eliminates the cross-term in the rank-1 update and gives

        S_{a-q}  =  S_a  -  (N / (N-1)) * (x_q - m_a)(x_q - m_a)^T,

    after which C_{a-q} = S_{a-q} normalised by the outer product of
    its diagonal square roots. (The 1/(N-2) factor cancels.)
    """
    X = np.asarray(X, dtype=dtype)
    N, p = X.shape
    m_a = X.mean(axis=0)
    R = X - m_a                           # centered residuals
    S_a = R.T @ R                         # population sum-of-squares
    factor = N / (N - 1.0)

    out = np.empty((N, p, p), dtype=dtype)
    for q in range(N):
        r_q = R[q]
        S_aq = S_a - factor * np.outer(r_q, r_q)
        d = np.diag(S_aq)
        out[q] = S_aq / np.sqrt(np.outer(d, d))
    return out.astype(np.float64)


# ---------------------------------------------------------------------------
# M2: ISN-tractor's `incremental_pearson` / `dense_isn`
# ---------------------------------------------------------------------------

def m2_loo_correlation(X):
    """
    Faithful port of the leave-one-out step inside ISN-tractor's
    `dense_isn` function (selected by metric="incremental_pearson").
    Forced float64 by the upstream code.

    Uses sufficient statistics A = X^T X, x_hat_j = sum_i X_{ij},
    a_hat_j = sum_i X_{ij}^2 and forms the (N-1)-sample correlation
    by direct subtraction:

        C_{a-q}_{j,k} =
          [ (N-1)(A_jk - x_qj x_qk) - (x_hat_j - x_qj)(x_hat_k - x_qk) ]
        / sqrt{ [(N-1)(a_hat_j - x_qj^2) - (x_hat_j - x_qj)^2]
                * [(N-1)(a_hat_k - x_qk^2) - (x_hat_k - x_qk)^2] }

    which is ISN-tractor Eq. 4. The denominator's expression
    (N-1) a_hat_loo - x_hat_loo^2 is the difference of two quantities
    of size N^2 mu^2 when the data has mean offset mu, so it is
    susceptible to catastrophic cancellation -- but the precision and
    the formula are exactly what `dense_isn` runs.
    """
    X = np.asarray(X, dtype=np.float64)
    N, p = X.shape

    x_hat = X.sum(axis=0)
    a_hat = (X * X).sum(axis=0)
    A = X.T @ X

    out = np.empty((N, p, p), dtype=np.float64)
    for q in range(N):
        x_q = X[q]
        x_loo = x_hat - x_q
        var_loo_diag = (N - 1) * (a_hat - x_q * x_q) - x_loo * x_loo
        # Square root may receive negative inputs once cancellation
        # destroys all significant digits; the float64 path then
        # produces NaN. We let it propagate.
        d_q = np.sqrt(var_loo_diag)
        nom = (N - 1) * (A - np.outer(x_q, x_q)) - np.outer(x_loo, x_loo)
        out[q] = nom / np.outer(d_q, d_q)
    return out


# ---------------------------------------------------------------------------
# LIONESS combiner
# ---------------------------------------------------------------------------

def lioness_from_correlations(C_a, C_loo, N):
    """
    e_q = N * C_a - (N - 1) * C_aq[q]    for q = 0, ..., N-1.

    Inputs:
      C_a   : (p, p) population correlation
      C_loo : (N, p, p) leave-one-out correlations
      N     : sample count
    Returns:
      e     : (N, p, p) LIONESS edge tensor
    """
    return N * C_a[None, :, :] - (N - 1) * C_loo


# ---------------------------------------------------------------------------
# Synthetic data + small helpers
# ---------------------------------------------------------------------------

def gen_block_data(N, p, n_blocks=4, condition=1.0, mean=0.0, seed=0):
    """
    Multivariate Gaussian samples with a block-correlation structure.
    `mean` adds a constant offset to all genes (the cancellation
    stress); `condition` re-scales eigenvalues to set kappa(Sigma).
    """
    rng = np.random.default_rng(seed)
    block_size = max(2, p // n_blocks)
    Sigma = np.eye(p)
    for b in range(n_blocks):
        i0, i1 = b * block_size, (b + 1) * block_size
        rho = 0.4 + 0.5 * rng.random()
        Sigma[i0:i1, i0:i1] = rho
        np.fill_diagonal(Sigma[i0:i1, i0:i1], 1.0)
    if condition != 1.0:
        w, V = np.linalg.eigh(Sigma)
        w_new = np.geomspace(w.max(), w.max() / condition, p)[::-1]
        Sigma = (V @ np.diag(w_new) @ V.T)
        Sigma = (Sigma + Sigma.T) / 2.0
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
    return rng.standard_normal((N, p)) @ L.T + mean


def max_err(A, B):
    diff = np.abs(A - B)
    if not np.all(np.isfinite(diff)):
        return float("inf")
    return float(np.max(diff))


def frob_err(A, B):
    e = np.linalg.norm(A - B, axis=(1, 2))
    e = np.where(np.isfinite(e), e, np.inf)
    return e


# Method registry used by experiments.
LOO_METHODS = {
    "Reference": ("exact Pearson, centered residuals",
                  reference_loo_correlation),
    "M1":        ("gpuLIONESS, orthogonal Welford-LOO",
                  m1_loo_correlation),
    "M2":        ("ISN-tractor incremental_pearson (dense_isn)",
                  m2_loo_correlation),
}
