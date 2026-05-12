"""
Exp 4 -- runtime comparison: M1 (orthogonal Welford-LOO) vs the
brute-force per-q recompute (Reference).

The leave-one-out Pearson correlation tensor C^{alpha - q} is the
inner-loop quantity for LIONESS. Two ways to produce it:

  brute-force:   for each q, recompute corrcoef(X_{-q}) from raw data
                 -- O(N) full Pearson computations, each O(N p^2),
                    total O(N^2 p^2)

  M1 (on-line):  compute S_alpha = R^T R once (O(N p^2)),
                 then for each q apply the rank-1 downdate
                 S_{alpha-q} = S_alpha - (N/(N-1)) r_q r_q^T (O(p^2))
                 -- total O(N p^2 + N p^2) = O(N p^2)

Asymptotic speedup ratio is N / 2; the constant depends on how much
of the brute-force cost is dominated by re-centering and re-forming
the dot product, vs how much is amortised by NumPy vectorisation.

We measure wall time across an (N, p) grid and report speedup as
brute_time / m1_time.
"""

from __future__ import annotations

import sys
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt

THIS = pathlib.Path(__file__).parent
sys.path.insert(0, str(THIS))
ROOT = THIS.parent
FIG_DIR = ROOT / "manuscript" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from methods import (m1_loo_correlation, reference_loo_correlation,
                     gen_block_data)


def _time(fn, X, n_repeats=3):
    """Wall time best-of-n_repeats."""
    best = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = fn(X)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best, out


def run():
    Ns = [50, 100, 200, 500, 1000]
    ps = [50, 100, 500]
    n_repeats = 3

    # results[(p, N)] = (t_bruteforce, t_m1)
    results = {}
    for p in ps:
        for N in Ns:
            X = gen_block_data(N, p, n_blocks=4, mean=10.0, seed=0)
            t_bf, _ = _time(reference_loo_correlation, X, n_repeats)
            t_m1, _ = _time(m1_loo_correlation, X, n_repeats)
            results[(p, N)] = (t_bf, t_m1)
            speedup = t_bf / t_m1
            print(f"  p={p:>4}  N={N:>4}  brute={t_bf*1000:>8.1f}ms  "
                  f"M1={t_m1*1000:>8.1f}ms   speedup={speedup:>5.1f}x")

    # ---- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for p in ps:
        ratios = [Ns[i] / p for i in range(len(Ns))]
        speedups = [results[(p, N)][0] / results[(p, N)][1] for N in Ns]
        ax.plot(ratios, speedups, marker="o", lw=1.6, label=f"p = {p}")

    # Reference line: N/2 scaling
    rs = np.linspace(0.05, max(Ns) / min(ps) * 1.05, 50)
    ax.plot(rs, np.maximum(rs * (min(ps) / 2.0), 1), color="0.6", lw=0.8,
            ls="--", label=r"$\propto N/2$ (asymptote)")
    # Mark the break-even line
    ax.axhline(1.0, color="k", lw=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"sample-to-gene ratio $N/p$")
    ax.set_ylabel(r"speedup  $t_{\mathrm{brute}}/t_{\mathrm{M1}}$")
    ax.set_title("Wall-time speedup of M1 over brute-force per-$q$ recompute")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()

    out = FIG_DIR / "exp4"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}.pdf / .png")
    return results


if __name__ == "__main__":
    run()
