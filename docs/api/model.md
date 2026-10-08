# The MTDR estimator

## `converged_=False`: what it means and what to do

For MMLE, `converged_=False` means that at least one optimisation step did not pass
the package's convergence checks. The fitted parameters are still returned, but
inspect the `ConvergenceWarning` and check the fit before interpreting the results.
The flag stays `False` if an earlier step failed, even when later steps finish
normally. `True` means the checks passed; it does not guarantee a global optimum.

The warning identifies the step and the reason. Start with the corresponding action:

| Warning mentions | Meaning and action |
|---|---|
| An iteration cap | Raise the named limit: `ecme_max_iter`, `refine_max_iter` or `optimizer_max_iter`, then refit. |
| `ABNORMAL` or L-BFGS-B status 2 | The optimiser could not find an acceptable next step. Check input conditioning with `check_design`, then assess a refit as below. |
| A result was not finite or raised the objective | That inner optimiser result was discarded and the previous parameters kept. Check input scales and design conditioning before refitting. |
| A neuron ended on the precision step bound | Its estimated noise precision reached the step limit. Inspect the named neurons for very small residual variation. |
| ECME raised the marginal NLL | An ECME update worsened the fit beyond the rounding allowance. Assess a refit; this warning is not covered by the inner optimiser's rule for discarding steps. |

Line searches that stop because further improvement is below numerical precision
can pass the convergence checks without a warning. The criteria are in
[model §E.3](../model.md#e3-rounding-level-ends-of-the-inner-optimisers).

In a rank search, `converged_` describes the **selected fit**. The warning also lists
other candidate fits that failed. Their failures can affect rank selection even
when the selected fit converged.

To assess the result, fit again at the selected ranks with a tighter stopping
tolerance and more refinement iterations. Use the same data and mask, and preserve
the original settings, including regularisation and preconditioning:

```python
params = model.get_params()
params.update(
    ranks=model.ranks_,
    refine_tol=min(model.refine_tol, 1e-8),
    refine_max_iter=max(model.refine_max_iter, 50),
)
refit = MTDR(**params).fit(Y, X, mask, regressor_names=model.regressor_names_)
print(model.objective_, refit.objective_)  # higher is better
```

Also raise any limit named in the warning. Compare `objective_` and the coefficient
matrices `B_`, and inspect any new warnings. At `basis_ridge=0`, `objective_` equals
the marginal training `log_likelihood_`; otherwise it includes the penalty.
A substantial objective gain shows that the original fit could be improved.
Similar objectives and coefficients provide evidence of stability, rather than
proof of an optimum.

This is a new fit, not a continuation. A fixed-rank fit starts from its own SVD
estimate, whereas the default rank search starts candidates from the preceding
accepted fit. The refit can therefore reach a different result, including a lower
objective. A substantial difference warrants checking sensitivity to initialisation;
`rank_search_warm_start=False` makes each search candidate start from its own SVD
estimate too. A fixed-rank check assesses the fit at those ranks; it does not check
whether failed candidates changed the rank-search outcome.

## Plain per-neuron least squares

The natural baseline, ordinary least squares per neuron with the intercept, is the
`"svd"` estimator at full rank, where the truncation removes nothing:

```python
P = X.shape[1]
ols = MTDR(ranks=[min(n_neurons, n_bins)] * P, estimator="svd").fit(Y, X, mask)
ols.B_  # each neuron's least-squares coefficients, regressor by regressor
```

The full rank is `min(n_neurons, n_bins)`, not `n_bins`: a rank above it is a
`ParameterError`, so `[n_bins] * P` fails when there are fewer neurons than bins. Under
a mask each neuron's fit uses only its observed trials; with `ridge > 0` it is ridge
regression.

## Reference

::: mtdr.model
