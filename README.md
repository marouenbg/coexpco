# coexpco — on-line co-expression benchmarks

Reference implementations and synthetic benchmarks for three ways of
computing a leave-one-out Pearson correlation matrix $C^{(i)}$ over
$N{-}1$ samples (sample $i$ held out), plus the LIONESS single-sample
edge combiner $e_i = N C - (N-1)\,C^{(i)}$ that consumes it.

## What it benchmarks

| Key       | Algorithm                                              |
|-----------|--------------------------------------------------------|
| Reference | Brute-force per-sample recompute via centered residuals, float64. Spot-checked against `mpmath` at 50 decimal digits. |
| M1        | Orthogonal Welford-LOO rank-1 downdate: $S^{(i)} = S - \frac{N}{N-1}\,r_i r_i^\top$ with $r_i = X_i - \bar m$. Float64. |
| M2        | Closed-form "incremental Pearson" rearrangement of the Pearson formula. Faithful port of ISN-tractor's `dense_isn` (`metric="incremental_pearson"`). Float64. |

All three operate at float64 and produce the same $\{C^{(i)}\}$ tensor
in infinite precision. The benchmarks show where they disagree in
finite precision and what propagates into LIONESS edges.

## Layout

```
simulations/
  methods.py                       -- Reference, M1, M2, and the LIONESS combiner
  exp1_loo_correlation.py          -- max deviation in C^{(i)} vs (N, mu)
  exp2_lioness_application.py      -- same, propagated to LIONESS edges e_i
  exp3_two_isn.py                  -- two-cluster signal vs noise (extra)
  exp4_timing.py                   -- wall-time speedup of M1 over brute-force
  make_all.py                      -- run every experiment in order
```

`exp3_two_isn.py` is an extra cluster-discrimination experiment that
is not required by the main benchmarks; it explores whether method
noise overtakes a fixed biological signal as the data mean offset
grows.

## Reproducing

```
pip install numpy scipy matplotlib mpmath
cd simulations
python make_all.py
```

Runs in about 30 s on a laptop CPU. Each experiment writes its figure
to a `figs/` subdirectory (created on first run).

## Notes

* The reference is the slow-but-correct baseline. M1 is the
  recommended implementation: same time complexity as the other
  fast methods, exact at the float64 mean-subtraction floor across
  all tested $\mu$.
* M2's deviation grows roughly as $\mu^2$ for data with mean offset
  $\mu$, consistent with the catastrophic-cancellation count
  $\log_{10}(N^2\mu^2/\sigma^2)$ for the sufficient-statistics
  rearrangement. The deviation is bounded but real, and propagates
  by a factor of $(N{-}1)$ into the LIONESS edge.
* PyTorch's `torch.corrcoef` at the default float32 dispatch is a
  separate code path with its own failure mode (NaN at large $\mu$)
  and is not benchmarked here.

## References

* Ben Guebila, M. *et al.* (2022). gpuZoo: cost-effective estimation
  of gene regulatory networks using the Graphics Processing Unit.
  *NAR Genomics and Bioinformatics* 4(1), lqac002.
* Lalli, G. *et al.* (2024). A python library for the fast and scalable
  computation of biologically meaningful individual specific networks.
  *Scientific Reports* 14, 18568.
* Kuijjer, M.L. *et al.* (2019). Estimating sample-specific regulatory
  networks. *iScience* 14, 226–240.
* Welford, B.P. (1962). Note on a method for calculating corrected sums
  of squares and products. *Technometrics* 4(3), 419–420.
* Chan, T.F., Golub, G.H., LeVeque, R.J. (1983). Algorithms for
  computing the sample variance. *The American Statistician* 37(3),
  242–247.
