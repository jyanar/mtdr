# Data utilities

## Simulating non-simultaneously recorded sessions

To test a pipeline on sessions recorded one after another, simulate **one**
population and partition its neurons and trials into sessions, then stack them with
[`stack_sessions`][mtdr.data.stack_sessions]. Separate `simulate` calls draw separate
temporal bases, so their sessions would not share one mTDR model.

```python
from mtdr import MTDR, simulate, split_trials, stack_sessions

# One population of 120 neurons and 360 trials, cut into three sessions of
# 40 neurons and 120 trials each.
sim = simulate(
    n_neurons=120, n_bins=8, n_trials=360, ranks=[2, 1, 2], drop_prob=0.2, seed=42
)
sessions = []
for s in range(3):
    trials, neurons = slice(120 * s, 120 * (s + 1)), slice(40 * s, 40 * (s + 1))
    sessions.append((sim.Y[trials, neurons], sim.X[trials], sim.mask[trials, neurons]))
stacked = stack_sessions(sessions)  # Y (360, 120, 8), a block-diagonal mask
train, test = split_trials(
    stacked.Y.shape[0], random_state=0, stratify=stacked.session_of_trial
)
model = MTDR(ranks=[2, 1, 2]).fit(
    stacked.Y[train], stacked.X[train], mask=stacked.mask[train]
)
print(model.ranks_, model.n_neurons_, stacked.mask.sum(axis=1).max())
```

Each neuron is then observed only on its own session's trials, and every trial sees
only its session's neurons, as in the recorded data.

## Categorical regressors and regressor groups

[`categorical_design`][mtdr.data.categorical_design] codes $L$ levels as $L-1$
orthonormal, zero-sum columns and returns a
[`CategoricalDesign`][mtdr.data.CategoricalDesign]: its `X` for the design, its `groups`
for `MTDR(groups=...)`, `encode` for held-out trials, `rows()` for decoding the level, and
`level_responses(model)` for each level's coefficient matrix, from a grouped or a
separate-term fit ([model §11.7](../model.md#117-categorical-regressors)). Any orthonormal
zero-sum basis gives the same fitted model. A coding that is not orthonormal (indicators
with a level dropped, say) can still be grouped as plain columns, at the cost of the
invariances of [model §11.9](../model.md#119-invariances-and-identifiability).
[`resolve_groups`][mtdr.data.resolve_groups] applies `MTDR`'s naming and ordering rules
without fitting, for example to order rank tuples before a search.

## Reference

::: mtdr.data
