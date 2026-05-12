"""
Exp 2 -- effect of method choice on the LIONESS single-sample edges.

Here we plug each method's leave-one-out correlation tensor into the
LIONESS combination

    e_q  =  N * C_a  -  (N-1) * C_{a-q}[q]

and measure the per-sample deviation from the Reference-derived
e_q^{Reference}. Same data sweeps as Exp 1; the population correlation
C_a is shared (centered residuals, float64) since all three methods
use that algorithm for the population.

We do not predict the magnitude of the deviation -- the script
measures what propagates into LIONESS edges.

Output: figures/exp2.{pdf,png}.
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
                     lioness_from_correlations, max_err, frob_err,
                     gen_block_data)

EVAL_METHODS = ["M1", "M2"]
STYLE = {
    "M1": dict(color="C0", marker="o", lw=1.6),
    "M2": dict(color="C3", marker="^", lw=1.6),
}


def isns_for(method, X):
    """Compose the LIONESS edge tensor for a given method."""
    C_a = population_correlation(X)
    C_loo = LOO_METHODS[method][1](X)
    return lioness_from_correlations(C_a, C_loo, X.shape[0])


def run():
    Ns = [50, 200, 500]
    mus = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    p = 100
    n_seeds = 3

    results = {m: {} for m in EVAL_METHODS}
    for N in Ns:
        for mu in mus:
            errs = {m: [] for m in EVAL_METHODS}
            for seed in range(n_seeds):
                X = gen_block_data(N, p, n_blocks=4, mean=mu, seed=seed)
                e_ref = isns_for("Reference", X)
                for m in EVAL_METHODS:
                    e_m = isns_for(m, X)
                    errs[m].append(max_err(e_m, e_ref))
            for m in EVAL_METHODS:
                results[m][(N, mu)] = errs[m]
                print(f"  N={N:>4} mu={mu:>7.0f}  {m}: "
                      f"mean={np.mean(errs[m]):.3e}  "
                      f"max={np.max(errs[m]):.3e}")

    # ---- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(1, len(Ns), figsize=(11.0, 3.4), sharey=True)
    for ax, N in zip(axes, Ns):
        for m in EVAL_METHODS:
            ys = [np.mean(results[m][(N, mu)]) for mu in mus]
            ax.plot(np.maximum(mus, 1e-1), ys, label=m,
                    markersize=4.5, **STYLE[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(np.finfo(np.float64).eps, color="0.5", lw=0.6, ls=":",
                   label=r"$\epsilon_{\mathrm{f64}}$")
        ax.set_title(f"N = {N}")
        ax.set_xlabel(r"data mean offset $\mu$")
    axes[0].set_ylabel(r"$\max\,|e^{\mathrm{method}}_q - e^{\mathrm{Reference}}_q|$")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(r"Effect of method choice on LIONESS edges $e_q = N\,C_\alpha - (N-1)\,C_{\alpha-q}$",
                 fontsize=11, y=1.02)
    fig.tight_layout()

    out = FIG_DIR / "exp2"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}.pdf / .png")
    return {"Ns": Ns, "mus": mus, "results": results}


if __name__ == "__main__":
    run()
