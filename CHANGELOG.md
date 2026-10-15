# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Regressor groups, `MTDR(groups={"sa": ["sa_1", "sa_2", "sa_3", "sa_4"]})`: the listed
  columns share one neural basis and one rank, each keeping its own temporal basis
  ($B_p=W_gS_p^\top$). Latent quantities are keyed by group, column quantities by column
  (new `groups_`, `group_names_`); every method, plot, `to_xarray` and the rank search
  support groups. The model and its derivations are section 11 of "The model and its
  estimation".
- `groups=` (column indices) in `mtdr.mmle`, `fit_svd` and `mtdr.aic`; `MMLEFit` and
  `SVDFit` record their groups.
- `mtdr.resolve_groups`, and `mtdr.categorical_design` / `CategoricalDesign` for a
  categorical regressor as one group of orthonormal zero-sum columns.
- `simulate(groups=...)` and `greedy_aic(max_rank=...)` with one cap per regressor.

### Changed

- The `ValidationError` for a rank-deficient pooled design names a categorical level
  absent from the fitted trials as one cause.

### Fixed

- `MTDR.project(..., basis=...)` raises `ParameterError`, not NumPy's `ValueError`, for
  a ragged basis.

Without `groups` every number is unchanged, bit for bit. Models, fits and simulations
pickled by 0.1.0 load as ungrouped ones.

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
