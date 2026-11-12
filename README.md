# A benchmark of on-line gene co-expression methods

This project is a benchmark of three methods to derive gene co-expression networks on-line in terms of performance and accuracy.
Downstream methods that use these networks as input to derive gene regulatory networks are further tested for propagation/stabilization of errors.

## What it benchmarks

| Key       | Algorithm                                              |
|-----------|--------------------------------------------------------|
| Reference | Direct computing of co-expression (Brute force). |
| M1        | A rank-1 downdate method. |
| M2        | A closed-form incremental Pearson method. |

All three operate at float64.

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

The layout of the benchmark aims at exploring whether
noise supersedes a fixed biological signal in a simulation study.

## Reproducing

```
pip install numpy scipy matplotlib mpmath
cd simulations
python make_all.py
```
