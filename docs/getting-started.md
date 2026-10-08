# Getting started

A complete session on simulated data: choose the ranks by AIC, compare held-out
likelihoods, decode a binary regressor, and plot. The test suite runs this page's code
as written.

Install the plotting extra for this example:

```bash
pip install "mtdr[plot]"
```

## Fit and evaluate

```python
import numpy as np
import matplotlib.pyplot as plt
from mtdr import MTDR, simulate, split_trials, plot

sim = simulate(n_neurons=100, n_bins=15, n_trials=400, ranks={"sa": 2, "sb": 1, "choice": 3},
               levels=[[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-1, 1]], drop_prob=0.3, seed=0)
train, test = split_trials(sim.X.shape[0], test_fraction=0.25, random_state=0,
                           stratify=sim.condition_of_trial)
print(len(train), "training trials,", len(test), "test trials")

model = MTDR(ranks="aic", estimator="mmle", max_rank=6, verbose=1)
model.fit(sim.Y[train], sim.X[train], mask=sim.mask[train], regressor_names=sim.regressor_names)
print("ranks:", model.ranks_, "| true:", sim.ranks)
print("AIC:", round(model.aic_, 1), "| stop:", model.rank_search_history_.stop_reason)
print("held-out log-likelihood:", round(model.log_likelihood(sim.Y[test], sim.X[test], sim.mask[test]), 1))
print("explained variance:", model.explained_variance(sim.Y[test], sim.X[test], sim.mask[test]))

fixed = MTDR(ranks=sim.ranks, estimator="mmle").fit(sim.Y[train], sim.X[train], mask=sim.mask[train],
                                                   regressor_names=sim.regressor_names)
for p in sim.regressor_names:
    r = np.corrcoef(fixed.B_[p].ravel(), sim.B[p].ravel())[0, 1]
    print(f"{p}: corr(B_hat, B_true) = {r:.3f}")
```

## Decode and plot

Continue in the same session. The binary choice is decoded by profile likelihood:
the continuous stimuli are re-estimated under each choice hypothesis.

```python
# Nature Neuroscience supplement, §6.3: re-decode stimuli under each choice hypothesis
out, llr = model.decode(sim.Y[test], ["sa", "sb", "choice"], mask=sim.mask[test],
                        levels={"choice": [-1, 1]}, return_llr=True)
print("choice decoding accuracy:", np.mean(out["choice"] == sim.X[test, 2]))

fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
plot.rank_search(model, axes=axes[0, :])                       # SVD stage left, MMLE stage right
plot.bases(model, regressors=["sa"], axes=axes[1, 0])
plot.trajectories(model, sim.Y[test], sim.X[test], "sa", mask=sim.mask[test], by=["sa"], ax=axes[1, 1])
plot.coefficient_norms(model)                                  # its own figure
plot.recovery(fixed, sim)                                      # its own figure, five panels
```

What the outputs mean:

- `verbose=1` prints one line per accepted step of the rank search: first the
  precision-weighted SVD stage that seeds the search, then the marginal-likelihood
  stage. The marginal stage's scores are the selection AIC, with the reference
  parameter count $n+T\sum_pr_p+nT$ (M38); `model.aic_` reports the identifiable
  count instead, so it is lower by $\sum_pr_p(r_p-1)$.
- `ranks:` the chosen ranks; rank recovery is likely at this scale, not
  guaranteed.
- `held-out log-likelihood:` the plug-in Gaussian log-likelihood of the test
  trials, the number to compare between models fitted to different codings of
  the same trials. This is `model.log_likelihood(...)`; under MMLE,
  `model.log_likelihood_` instead holds the marginal training likelihood, with
  neuron weights integrated out. The two are different quantities.
- `explained variance:` the fraction of held-out variance each term explains
  alone, around each neuron's flat mean; `intercept` is the condition-independent
  time course.
- `choice decoding accuracy:` the profile-likelihood decoder of the
  [Nature Neuroscience supplement, §6.3](https://doi.org/10.1038/s41593-020-0696-5);
  `llr` holds its per-trial evidence for
  `choice = +1`.

Missing data are marked per neuron and trial (`mask`): without a mask, one `NaN`
bin makes that neuron unobserved on that trial, its finite bins included. To fit
neurons recorded in different sessions, see
[the session recipe](api/data.md#simulating-non-simultaneously-recorded-sessions).

If a fit warns with a `ConvergenceWarning` or reports `converged_=False`, see
[what it means and what to do](api/model.md#converged_false-what-it-means-and-what-to-do):
the fit is returned and usable, and the warning names the cap to raise or how to check
the fit.
