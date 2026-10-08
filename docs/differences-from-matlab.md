# Differences from the MATLAB code

`mtdr` is a port of the authors' MATLAB demo,
[`pillowlab/mTDRdemo`](https://github.com/pillowlab/mTDRdemo). It fits the same model by
the same procedure (SVD initialisation, ECME, then coordinate ascent; a greedy AIC rank
search), and the test suite checks it against fits the MATLAB code recorded. It does not
reproduce the MATLAB code iterate by iterate: some defaults differ, a few defects of the
reference are fixed, and some options are new. This page lists what differs and, where one
exists, the option that restores the reference's behaviour. The equations cited are those
of [The model and its estimation](model.md).

## How the port was checked

The scripts in
[`tests/fixtures/matlab/`](https://github.com/jyanar/mtdr/tree/main/tests/fixtures/matlab)
record the unmodified MATLAB reference on both shipped demos and simulated data,
including 800 neurons in 100 sessions, 15 bins and six correlated regressors.
This checks the implementation at the paper's scale, not on its experimental data.

- **Recorded quantities:** sufficient statistics, SVD coefficients, marginal
  likelihoods, AICs and posterior means are compared at reference parameters.
  Corrected quantities are checked separately from replicas of the MATLAB defects.
- **Reference paths:** `ecme(matlab_compat=True)` reproduces recorded ECME iterates
  to about $10^{-11}$ relative. Compatibility helpers replay the demo's rank paths;
  candidate marginal likelihoods agree within $10^{-8}$ relative. These checks
  do not imply that the default `MTDR()` follows the same rank path.
- **Final fits:** likelihoods from reference starting points agree within
  $10^{-11}$ relative. From the package's own start they are at least as good as
  the reference's to $10^{-7}$ relative. Tight parameter comparisons ($10^{-4}$)
  refine both MATLAB- and Python-seeded endpoints in Python; they test where the
  Python optimiser ends, separately from the recorded-quantity checks.

An independent, line-by-line translation of the MATLAB MMLE stage in
[`tests/nversion/`](https://github.com/jyanar/mtdr/tree/main/tests/nversion)
is checked against both implementations. Decoder identities are checked in Python;
the MATLAB demo does not decode. The paper-scale fits run with `pytest -m slow`.

## Defaults that differ

These change what `MTDR()` does compared with `mTDRdemo.m`. Each can be set back.

| What | MATLAB | `mtdr` default | Reference behaviour |
|---|---|---|---|
| Where the MMLE rank search starts | the ranks of an SVD AIC search (`mTDRdemo.m`, `EstRankGreedily.m`) | the ranks of a *precision-weighted* SVD AIC search (M18w), which over-selects less when neurons' noise levels differ | `rank_search_init="svd"` |
| How each rank-search candidate starts | from its own SVD fit (`ECMEregress_wrapper.m`) | from the accepted fit, plus one new column for the raised regressor: about 4-6x fewer basis-step evaluations per search at the paper's scale, and no selected ranks changed in the 600 simulated searches compared | `rank_search_warm_start=False` |
| Basis step's iteration cap | `minFunc`'s default `MaxIter` 500 | `optimizer_max_iter=2000` (the first basis step at the paper's scale needs 582-909 iterations) | `optimizer_max_iter=500` |
| Basis step's gradient test | `minFunc`'s `optTol` 1e-5 | `optimizer_tol=1e-6`, also the precision step's | `optimizer_tol=1e-5` |
| Orientation of `W_` and `S_` | the estimator's raw frame | rotated to the paper's PC orientation: `W_[p]` orthonormal, columns ordered by singular value; `B_`, the likelihoods and the AIC are unchanged | `canonicalize=False` |

## Optimisers: replaced, not reproducible exactly

The MATLAB code uses `minFunc` and `fminunc`; `mtdr` uses SciPy. The fit comparisons
above assess endpoints rather than inner optimiser iterates: `minFunc`'s stopping
tests (gradient, absolute function change, step size) and its strong-Wolfe line
search have no exact SciPy counterpart.

| What | MATLAB (`Estpars_CoordAscent_lambi_S_b.m`) | `mtdr` |
|---|---|---|
| Basis step | `minFunc` L-BFGS (memory 100, strong-Wolfe cubic line search), stopping on `max(abs(g)) <= optTol`, on an *absolute* function change `progTol = 1e-9`, or on a small step | SciPy L-BFGS-B (memory 100). SciPy's function test is relative, so `optimizer_progtol=1e-9` is turned into `ftol = progtol / max(abs(f0), 1)` at each call's start; there is no step-size test, and an end at the rounding level counts as converged, as `minFunc`'s `progTol` end does ([model § E.3](model.md#e3-rounding-level-ends-of-the-inner-optimisers)) |
| Noise-precision step | `fminunc`, trust-region-reflective with a finite-difference diagonal Hessian, unconstrained in $\lambda$ (a step can make $\lambda_i\le0$) | L-BFGS-B on $\log\lambda$ inside a box of $\pm23$ around the current value, then up to 3 diagonal Newton steps to polish a line-search end |
| Basis preconditioning | none | off by default; `basis_preconditioning=True` rescales the basis step's variables in cold fits (about 7x fewer basis-step evaluations at the paper's scale, same optimum to the optimiser's tolerance). `False` is the reference's step |

## Reference defects that are fixed

The MATLAB code has bugs that its defaults mostly hide. `mtdr` fixes them, so these
differences have no option to switch back. Where parity needs the reference's
behaviour, it is reproduced in the tests (`tests/matlab_compat.py`) or behind a flag of the
functional API, noted below.

| What | MATLAB | `mtdr` |
|---|---|---|
| SVD-stage AIC | `SVDRegB_AIC.m` scrambles neurons when reshaping, misaligns the residual, uses the regressor count for the observation count and flips a sign, so its "AIC" can fall as the fit worsens (M20a)-(M20c) | the Gaussian AIC with the textbook count (M21a)-(M21b). So the SVD search, and therefore the MMLE search's start, can differ from the demo's even with `rank_search_init="svd"`; `mtdr.aic.n_parameters_svd(formula="reference")` gives the reference's count |
| SVD-stage noise precision | `SVDRegress_S_Vdata.m` computes it from the same misaligned residual, about 30x too small at demo scale (M19a) | the aligned residual with the intercept refitted to the truncated blocks (M19b). ECME's first step absorbs the difference |
| ECME updates | the precision is overwritten mid-iteration, and the S- and b-steps mix old and new quantities (`ECMEtdr.m`) | each step uses the current iterate consistently (M30)-(M33), the authors' written algorithm ([NeurIPS supplement, Algorithm 1](https://proceedings.neurips.cc/paper/2018/hash/8a1ee9f2b7abe6e88d1a479ab6a42c5e-Abstract.html)); same fixed point, different path. `mtdr.mmle.ecme(matlab_compat=True)` reproduces the reference's steps |
| Convergence test | pure relative change (M34), `Inf`/`NaN` when a parameter is exactly zero | relative change with a floor, `convergence_eps=1e-12`. The thresholds are the reference's: `ecme_tol=1.0`, `ecme_max_iter=100`, `refine_tol=1e-4`, `refine_max_iter=10` |
| Accepting a rank-search step | accepted when the AIC does not increase (`<=`), and the history records the wrong candidate when the candidate set is not a prefix (`EstRankGreedily.m`) | accepted only when the AIC drops by more than `rank_search_threshold` (default 0, so a tie is rejected); correct bookkeeping; ties go to the lowest regressor index |
| Ridge on the bases | the value and the gradient penalise different entries (`neglogLikBTDR_IncompObs_uneqvar_Sonly.m`); the precision step's value never includes it | $\tfrac g2\lVert S\rVert^2$ in both, kept out of the reported likelihood. Inert at the default `basis_ridge=0`, which is the reference's |
| Degenerate neurons | a never-observed neuron, or one with too few trials for its design, gives `0/0` or a singular solve without an error | a `ValidationError` naming the neurons (`min_observations`, and the MMLE's residual-degrees-of-freedom check, [model § 10.3](model.md#103-degenerate-neurons)) |
| The intercept | correct only if the constant regressor is the **last** column of `X` | an explicit per-neuron, per-bin intercept by default; with `condition_independent=True`, a constant column in `X` is a `ValidationError` |
| `mTDRdemo.m`'s final unpacking | unpacks the estimated parameters with the *true* ranks' total and passes the uncentred statistics to the posterior (lines 152-155), which puts $\hat B_p$ off by 7-23 % on demo-scale data | uses the fitted ranks and the centred statistics (M36), as `demoLearning.m` does |
| Simulated noise | `d` is documented as a variance or standard deviation but used as a precision (`SimPopData.m`) | named `noise_precision` in `mtdr.simulate` |

## Reported numbers

Even at the same fitted parameters, some reported numbers use different conventions.

| What | MATLAB | `mtdr` |
|---|---|---|
| Log-likelihood | omits the $\tfrac12\sum_in_iT\log2\pi$ constant | `log_likelihood_` includes it, so absolute values and AICs are shifted by a rank-independent constant; differences between fits are unchanged |
| MMLE AIC | one count, $n+T\sum_pr_p+nT$ (M38) (`BTDR_AIC_S_lamb_b_wrapper.m`) | the search *selects* with that count, including the Gaussian likelihood constant above; `n_parameters_` uses the identifiable count (M38a), smaller by $\sum_pr_p(r_p-1)/2$, so `aic_` is below the selection score by $\sum_pr_p(r_p-1)$ |

## Not in the MATLAB code

- `MTDR(n_jobs=)` fits each round's rank-search candidates in worker processes, with the
  same result as fitting them in turn.
- `MTDR(basis_preconditioning=)` (above).
- `project`, `decode`, `explained_variance`, `orthogonalize` and `subspace_angles`
  implement quantities of the 2020 paper and its Nature Neuroscience supplement
  that the demo does not compute ([model § 9](model.md#9-decoding-and-projection)). `project` defaults
  to a projection that stays on one scale when neurons are missing on a trial;
  `method="paper"` is the paper's formula.
- Input validation and diagnostics: [`validate_inputs`][mtdr.data.validate_inputs] and
  [`check_design`][mtdr.data.check_design].

## Function map

Where each MATLAB function's work happens in `mtdr`.

| MATLAB (`mTDRdemo/functionFiles/`) | `mtdr` |
|---|---|
| `SimWeights`, `SimConditions`, `SimPopData`, the mask step of `mTDRdemo.m` | [`simulate`][mtdr.simulation.simulate] |
| `MkSuffStatsBTDR_IncompObs_uneqvar_S_fast`, `ECMEsuffstat`, `MkSuffStats_BilinReg_Sims`, the `xbari` / `Ybar` loop of the demos | [`sufficient_statistics`][mtdr.stats.sufficient_statistics] (the centred statistics are formed inside `mtdr.mmle`) |
| `SVDRegressB`, `SVDRegress_S_Vdata` | [`fit_svd`][mtdr.svd_fit.fit_svd] |
| `SVDRegB_AIC` | `SVDFit.aic`, with the count of [`n_parameters_svd`][mtdr.aic.n_parameters_svd] |
| `neglogLikBTDR_IncompObs_uneqvar_S_nllonly` | [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood] |
| `neglogLikBTDR_IncompObs_uneqvar_Sonly` | [`marginal_nll_grad_S`][mtdr.mmle.marginal_nll_grad_S] |
| `neglogLikBTDR_IncompObs_uneqvar_lambonly` | [`marginal_nll_grad_noise`][mtdr.mmle.marginal_nll_grad_noise] |
| `ECMEregress_wrapper` | the start of [`fit_mmle`][mtdr.mmle.fit_mmle] (M29) |
| `ECMEtdr`, `keepActive_S` | [`ecme`][mtdr.mmle.ecme] |
| `MMLE_b` | [`update_intercept`][mtdr.mmle.update_intercept] |
| `Q_TDR` | not ported (a diagnostic the reference evaluates but does not use) |
| `Estpars_CoordAscent_lambi_S_b` | [`refine`][mtdr.mmle.refine] |
| `MMLE_CoordAscentWrapper` | [`fit_mmle`][mtdr.mmle.fit_mmle] |
| `EBpost_W_uneqvar`, `MakeBhat_data` | [`posterior_weights`][mtdr.mmle.posterior_weights]; `MTDR.W_`, `MTDR.B_` |
| `BTDR_AIC_S_lamb_b_wrapper` | `MMLEFit.aic`, with the count of [`n_parameters_mmle`][mtdr.aic.n_parameters_mmle] |
| `EstRankGreedily` | [`greedy_aic`][mtdr.rank_search.greedy_aic] and [`RankSearchHistory`][mtdr.rank_search.RankSearchHistory] |
| `mTDRdemo.m`, `demoLearning.m` | [`MTDR.fit`][mtdr.model.MTDR.fit]; see [Getting started](getting-started.md) |
| `minFunc` (L-BFGS), `fminunc` | `scipy.optimize.minimize(method="L-BFGS-B")` |
| `mmx_mkl_single`, `slowMult`, `slowBackslash`, `slowChol`, `kronmult` | batched NumPy linear algebra; no Kronecker product is formed |

Parameter vectors are packed as in the MATLAB, $\theta=(\lambda;s;\mathrm{vec}\,b)$ (M28),
only where the tests read the reference's fixtures; the package keeps the blocks
separate (`MMLEFit.noise_precision`, `.S`, `.intercept`).
