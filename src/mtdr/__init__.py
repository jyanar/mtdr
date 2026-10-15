"""Model-based targeted dimensionality reduction (mTDR) for neural population data.

A Python port of mTDR (Aoi, Mante & Pillow, *Nat Neurosci* 2020; estimator from
Aoi & Pillow, NeurIPS 2018). Pre-alpha. The estimator is
[`MTDR`][mtdr.model.MTDR] (fit, predict, score, project, decode), with
[`canonical_factors`][mtdr.model.canonical_factors] for its PC frame; the data
utilities are [`validate_inputs`][mtdr.data.validate_inputs],
[`check_design`][mtdr.data.check_design],
[`condition_average`][mtdr.data.condition_average],
[`stack_sessions`][mtdr.data.stack_sessions],
[`split_trials`][mtdr.data.split_trials], and, for regressor groups,
[`resolve_groups`][mtdr.data.resolve_groups] and
[`categorical_design`][mtdr.data.categorical_design];
[`simulate`][mtdr.simulation.simulate]
draws ground truth; `mtdr.plot` (matplotlib) and `MTDR.to_xarray` (xarray) use
optional extras. The functional building blocks are the per-neuron sufficient
statistics (`mtdr.stats`), the SVD fit (`mtdr.svd_fit`), the marginal
likelihood, ECME, coordinate ascent and the posterior weights (`mtdr.mmle`),
parameter counts and AIC (`mtdr.aic`) and the greedy AIC rank search
(`mtdr.rank_search`).

Equation numbers such as (M24) in the package's docstrings refer to
`docs/model.md` ("The model and its estimation" page of the documentation).

Examples
--------
>>> import mtdr
>>> isinstance(mtdr.__version__, str)
True
"""

from mtdr import aic, mmle, plot, rank_search, stats, svd_fit
from mtdr._version import __version__
from mtdr.data import (
    CategoricalDesign,
    ConditionAverage,
    DesignReport,
    StackedSessions,
    categorical_design,
    check_design,
    condition_average,
    resolve_groups,
    split_trials,
    stack_sessions,
    validate_inputs,
)
from mtdr.errors import (
    ConvergenceWarning,
    DecodingWarning,
    DesignWarning,
    MTDRError,
    MTDRWarning,
    NotFittedError,
    ParameterError,
    ProjectionWarning,
    SingularDesignError,
    ValidationError,
)
from mtdr.mmle import MMLEFit
from mtdr.model import MTDR, canonical_factors
from mtdr.rank_search import RankSearchHistory
from mtdr.simulation import SimulatedData, simulate
from mtdr.stats import SufficientStats
from mtdr.svd_fit import SVDFit

__all__ = [
    "MTDR",
    "CategoricalDesign",
    "ConditionAverage",
    "ConvergenceWarning",
    "DecodingWarning",
    "DesignReport",
    "DesignWarning",
    "MMLEFit",
    "MTDRError",
    "MTDRWarning",
    "NotFittedError",
    "ParameterError",
    "ProjectionWarning",
    "RankSearchHistory",
    "SVDFit",
    "SimulatedData",
    "SingularDesignError",
    "StackedSessions",
    "SufficientStats",
    "ValidationError",
    "__version__",
    "aic",
    "canonical_factors",
    "categorical_design",
    "check_design",
    "condition_average",
    "mmle",
    "plot",
    "rank_search",
    "resolve_groups",
    "simulate",
    "split_trials",
    "stack_sessions",
    "stats",
    "svd_fit",
    "validate_inputs",
]
