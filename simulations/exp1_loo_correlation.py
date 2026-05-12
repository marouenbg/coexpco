"""
Exp 1 -- benchmark on the leave-one-out Pearson correlation.

Three methods (Reference, M1, M2) are run on synthetic block-correlated
Gaussian data. For each (N, mean offset mu) cell we measure

    max_{q, j, k}  | C_{a-q,j,k}^{method}  -  C_{a-q,j,k}^{Reference} |.

Reference is the bit-exact baseline; M1 (gpuLIONESS, orthogonal
Welford-LOO) and M2 (ISN-tractor incremental_pearson / dense_isn) are
the methods under test. We do not predict outcomes -- the script
records what each method actually produces.

Output: figures/exp1.{pdf,png}, plus stdout table.
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

from methods import LOO_METHODS, max_err, gen_block_data, reference_loo_correlation_mpmath

EVAL_METHODS = ["M1", "M2"]
STYLE = {
    "M1": dict(color="C0", marker="o", lw=1.6),
    "M2": dict(color="C3", marker="^", lw=1.6),
}


def run():
    Ns = [50, 200, 500]
    mus = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    p = 100
    n_seeds = 3

    # results[method][(N, mu)] = list of max-errors across seeds
    results = {m: {} for m in EVAL_METHODS}

    for N in Ns:
        for mu in mus:
            errs = {m: [] for m in EVAL_METHODS}
            for seed in range(n_seeds):
                X = gen_block_data(N, p, n_blocks=4, mean=mu, seed=seed)
                ref = LOO_METHODS["Reference"][1](X)
                for m in EVAL_METHODS:
                    M = LOO_METHODS[m][1](X)
                    errs[m].append(max_err(M, ref))
            for m in EVAL_METHODS:
                results[m][(N, mu)] = errs[m]
                print(f"  N={N:>4} mu={mu:>7.0f}  {m}: "
                      f"mean={np.mean(errs[m]):.3e}  "
                      f"min={np.min(errs[m]):.3e}  max={np.max(errs[m]):.3e}")

    # mpmath spot-check at the highest stress cell.
    print("\nmpmath spot-check (N=30, p=10, mu=1000)...")
    Xmp = gen_block_data(30, 10, n_blocks=2, mean=1000.0, seed=0)
    ref_f64 = LOO_METHODS["Reference"][1](Xmp)
    ref_mp = reference_loo_correlation_mpmath(Xmp, prec_digits=50)
    err_mp = max_err(ref_f64, ref_mp)
    print(f"  max|Reference_f64 - mpmath_50dps| = {err_mp:.3e}")

    # ---- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(1, len(Ns), figsize=(11.0, 3.4),
                             sharey=True)
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
    axes[0].set_ylabel(r"$\max\,|C^{\mathrm{method}}_{\alpha-q} - C^{\mathrm{Reference}}_{\alpha-q}|$")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(r"Benchmark on the leave-one-out Pearson correlation $C^{\alpha-q}$",
                 fontsize=11, y=1.02)
    fig.tight_layout()

    out = FIG_DIR / "exp1"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}.pdf / .png")
    return {"Ns": Ns, "mus": mus, "results": results, "mpmath_check": err_mp}


if __name__ == "__main__":
    run()
