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

## Regressor groups

`MTDR(groups={"sa": ["sa_1", "sa_2", "sa_3", "sa_4"]})` makes the listed columns share
one neural basis and one rank; columns not listed are groups of their own, named after
themselves ([model §11](../model.md#11-regressor-groups); a worked example is in
[Getting started](../getting-started.md#a-categorical-stimulus-with-one-shared-subspace)).
[`resolve_groups`][mtdr.data.resolve_groups] returns the groups `fit` will use, in their
order (by smallest member column). With `groups`, `ranks` and `rank_search_init` are
`"aic"` (or a string) or mappings naming every group; a sequence is refused, and so is a
key that is a column of a group of several columns.

Everything that belongs to a latent weight is keyed by **group**; everything that belongs
to a design column by **column**:

| Keyed by group | Keyed by column |
|---|---|
| `groups_`, `group_names_`, `ranks_`, `W_`, the blocks of `weight_posterior_cov_`, `rank_search_history_` names | `regressor_names_`, `S_`, `B_` |
| `project(Y, name)` and its `basis=`, `orthogonalize(order)`, `subspace_angles(a, b)`, `explained_variance` keys | `decode` values and `known=`, `predict`, `log_likelihood` |
| `plot.weights`, `plot.trajectories`, `plot.rank_search`, `plot.recovery` | `plot.bases`, `plot.coefficient_norms` |

A column of a group of several columns passed where a group is expected raises a
`ParameterError` that names its group. `decode(levels={"sa": rows})` takes candidate rows
of one value per column of the group (`CategoricalDesign.rows()`), enumerated as one
discrete unit for the cap of 64 combinations and for `return_llr`; a column of such a
group cannot have levels of its own. `to_xarray()` puts `W` and `rank` on a `group`
dimension and records each column's group in a `group_of` coordinate. With every column
alone (`groups=None`, or explicit single-column groups named after their columns) the
fit, the attributes and every output are those without groups, bit for bit.

Each group's rank is at most `min(n_neurons, |g| n_bins, max_rank)`, so a group's rank
can exceed `n_bins`; `max_rank` itself may not exceed the largest group's
`min(n_neurons, |g| n_bins)`. `total_rank_` counts a shared basis once, but the parameter
counts (`n_parameters_`, model (M59)-(M60)) grow with `|g|`: compare groupings by
held-out likelihood.

## Reference

::: mtdr.model
