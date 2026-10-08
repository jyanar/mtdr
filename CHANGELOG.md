# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-08

First public release: a Python port of the MATLAB mTDR demo
([`pillowlab/mTDRdemo`](https://github.com/pillowlab/mTDRdemo)) of Aoi & Pillow
(NeurIPS 2018) and Aoi, Mante & Pillow (*Nature Neuroscience* 2020), checked against
outputs recorded by running the MATLAB code.

### Added

- The `MTDR` estimator: `MTDR(...).fit(Y, X, mask)` with fixed ranks or a greedy AIC
  rank search; the marginal-likelihood estimator (`estimator="mmle"`, the paper's) or
  reduced-rank regression (`estimator="svd"`); `predict`, `score` / `log_likelihood`,
  `aic`, `project` (trajectories, the paper's formula or a mask-robust default),
  `decode` (the paper's model-based decoder, with log-likelihood ratios for discrete
  regressors), `explained_variance`, `orthogonalize`, `subspace_angles` and
  `to_xarray`. Fitted coefficient matrices, bases and weights are keyed by regressor
  name; `W_` and `S_` are in the paper's PC orientation by default.
- Neurons recorded in different sessions through a per-trial observation mask (or
  `NaN` in `Y`), with validation that rejects neurons whose likelihood is unbounded.
- The functional building blocks: `sufficient_statistics`, `fit_svd`, the marginal
  likelihood and its gradients, ECME, the coordinate-ascent refinement and the
  posterior weights of `mtdr.mmle` (`fit_mmle`), the parameter counts of `mtdr.aic`,
  and `greedy_aic` with its `RankSearchHistory`.
- Speed options for the rank search: warm-started candidates
  (`rank_search_warm_start=True`, the default), candidate fits in worker processes
  (`n_jobs`), and a preconditioned basis step for cold fits
  (`basis_preconditioning`, off by default).
- Data utilities in `mtdr.data`: `validate_inputs`, `check_design`,
  `condition_average`, `stack_sessions`, `split_trials`; `canonical_factors`.
- Plots in `mtdr.plot` (optional `plot` extra): `bases`, `weights`,
  `coefficient_norms`, `rank_search`, `trajectories` and `recovery`.
- `mtdr.simulate`: the demo's ground-truth simulator, also for a supplied design,
  bases or weights.
- Documentation: getting started, the model and its estimation (every equation the
  code implements), differences from the MATLAB code and the API reference.
- Tests: unit, property and doctest suites, MATLAB parity at the demos' and the
  paper's scale (`pytest -m slow`), and an independent line-by-line reference
  implementation of the MMLE stage.
