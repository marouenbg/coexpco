"""
Exp 3 -- cluster discrimination: does method noise overwhelm a fixed
biological signal in two-cluster LIONESS comparisons?

Setup
-----
Two clusters of N_per samples each. Cluster A has block-correlated
covariance with within-block correlation rho_A = 0.7; cluster B has
rho_B = 0. p genes split into 4 equal blocks. We sweep the data mean
offset mu over a log range and, at each mu, measure

    biological_signal(mu)  =  || cluster_mean_e^{Ref}_A
                                - cluster_mean_e^{Ref}_B ||_F
    method_noise(M, mu)    =  mean_q || e_q^M - e_q^{Ref} ||_F

For each method we plot signal and noise versus mu. The biological
signal is correlation-driven and (in expectation) does not depend on
mu; the method noise is what each implementation injects.

We do not predict the curves -- the script draws them.
"""

from __future__ import annotations

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

THIS = pathlib.Path(__file__).parent
sys.path.insert(0, str(THIS))
ROOT = THIS.parent
FIG_DIR = ROOT / "manuscript" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from methods import (LOO_METHODS, population_correlation,
                     lioness_from_correlations)

EVAL_METHODS = ["M1", "M2"]
STYLE = {
    "M1": dict(color="C0", marker="o", lw=1.6),
    "M2": dict(color="C3", marker="^", lw=1.6),
}


def block_cov(p, n_blocks, rho):
    block_size = p // n_blocks
    M = np.eye(p)
    for b in range(n_blocks):
        i0, i1 = b * block_size, (b + 1) * block_size
        M[i0:i1, i0:i1] = rho
    np.fill_diagonal(M, 1.0)
    return M


def synth(N_per, p, mu, rho_A=0.7, rho_B=0.0, n_blocks=4, seed=0):
    rng = np.random.default_rng(seed)
    Sigma_A = block_cov(p, n_blocks, rho_A)
    Sigma_B = block_cov(p, n_blocks, rho_B)
    L_A = np.linalg.cholesky(Sigma_A + 1e-10 * np.eye(p))
    L_B = np.linalg.cholesky(Sigma_B + 1e-10 * np.eye(p))
    XA = rng.standard_normal((N_per, p)) @ L_A.T + mu
    XB = rng.standard_normal((N_per, p)) @ L_B.T + mu
    return np.concatenate([XA, XB], axis=0), N_per


def isns_for(method, X):
    C_a = population_correlation(X)
    C_loo = LOO_METHODS[method][1](X)
    return lioness_from_correlations(C_a, C_loo, X.shape[0])


def run():
    p = 100
    N_per = 100
    mus = np.logspace(-1.0, 4.0, 11)
    n_seeds = 3

    signal = []
    noise = {m: [] for m in EVAL_METHODS}

    for mu in mus:
        per_seed_sig = []
        per_seed_noise = {m: [] for m in EVAL_METHODS}
        for seed in range(n_seeds):
            X, N1 = synth(N_per, p, mu=mu, seed=seed)
            e_ref = isns_for("Reference", X)
            mean_A = e_ref[:N1].mean(axis=0)
            mean_B = e_ref[N1:].mean(axis=0)
            per_seed_sig.append(float(np.linalg.norm(mean_A - mean_B)))

            for m in EVAL_METHODS:
                e_m = isns_for(m, X)
                if not np.all(np.isfinite(e_m)):
                    per_seed_noise[m].append(np.inf)
                else:
                    diff_norms = np.linalg.norm(e_m - e_ref, axis=(1, 2))
                    per_seed_noise[m].append(float(diff_norms.mean()))

        signal.append(np.mean(per_seed_sig))
        for m in EVAL_METHODS:
            noise[m].append(np.mean(per_seed_noise[m]))
        print(f"  mu={mu:>9.3g}  signal={signal[-1]:.3e}   "
              + "  ".join(f"{m}_noise={noise[m][-1]:.2e}" for m in EVAL_METHODS))

    signal = np.array(signal)

    print("\nCrossover mu* (smallest mu where method noise reaches signal):")
    crossovers = {}
    for m in EVAL_METHODS:
        n_arr = np.array(noise[m])
        idx = np.where(n_arr >= signal)[0]
        crossovers[m] = float(mus[idx[0]]) if len(idx) > 0 else None
        print(f"  {m}: mu* = {crossovers[m]}")

    # ---- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(mus, signal, color="k", lw=2.0, marker="D", markersize=5,
            label=r"biological signal $\|\overline{e}_A - \overline{e}_B\|_F$")
    cap = signal.max() * 10.0
    for m in EVAL_METHODS:
        ys = np.array(noise[m], dtype=float)
        ys_plot = np.where(np.isfinite(ys), ys, cap)
        ax.plot(mus, ys_plot, label=f"{m} per-sample noise",
                markersize=4.5, **STYLE[m])
        nan_idx = np.where(~np.isfinite(ys))[0]
        if len(nan_idx) > 0:
            ax.scatter(mus[nan_idx], [cap] * len(nan_idx), marker="x",
                       s=60, color=STYLE[m]["color"], zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"data mean offset $\mu$")
    ax.set_ylabel(r"$\|\,\cdot\,\|_F$")
    ax.set_title(r"Two-cluster comparison ($\rho_A=0.7,\,\rho_B=0$, "
                 r"100 samples per cluster, $p=100$)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()

    out = FIG_DIR / "exp3"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}.pdf / .png")
    return {"mus": mus, "signal": signal, "noise": noise,
            "crossovers": crossovers}


if __name__ == "__main__":
    run()
