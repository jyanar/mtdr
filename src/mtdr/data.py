r"""Input validation and data utilities.

Everything the [`MTDR`][mtdr.model.MTDR] estimator reads goes through
[`validate_inputs`][mtdr.data.validate_inputs] first: the layout
`Y[k, i, t]`, `X[k, p]`, `mask[k, i]` (`docs/model.md` § 1.2), `NaN`-to-mask
inference, and the fit-only checks on neurons and designs that the
estimators cannot fit. The other functions here prepare data for the
estimator and never fit anything:

- [`check_design`][mtdr.data.check_design] reports, without raising, what
  `fit` would reject or warn about in a design;
- [`condition_average`][mtdr.data.condition_average] averages trials within
  conditions under the mask (a masked mean; for pseudo-trial resampling see the
  [Nature Neuroscience supplement, §6.1](https://doi.org/10.1038/s41593-020-0696-5));
- [`stack_sessions`][mtdr.data.stack_sessions] assembles sequentially recorded
  sessions into one union-trial tensor with a block-diagonal mask, the
  observation model (M3), (M9) the estimators are written for;
- [`split_trials`][mtdr.data.split_trials] draws a trial split for held-out
  comparisons.

The package never centres or rescales `X` or `Y`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mtdr._args import (
    as_int,
    as_list,
    as_real,
    require_names,
    require_non_negative_real,
)
from mtdr._frozen import freeze, restore_frozen
from mtdr.errors import DesignWarning, ParameterError, ValidationError
from mtdr.stats import sufficient_statistics
from mtdr.svd_fit import RANK_TOLERANCE, _least_squares, _normal_equations

__all__ = [
    "ConditionAverage",
    "DesignReport",
    "StackedSessions",
    "check_design",
    "condition_average",
    "split_trials",
    "stack_sessions",
    "validate_inputs",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

#: The estimators whose observation floor `check_design` can apply.
_ESTIMATORS = ("mmle", "svd")
#: `|corr| >` this between two columns of `X` is a near-collinearity warning.
_CORRELATION_WARNING = 0.95
#: A largest-over-smallest column standard deviation above this is a scale
#: disparity (a warning only when `basis_ridge > 0`).
_SCALE_RATIO = 10.0
#: A continuous column (more than two distinct values) whose mean exceeds this
#: fraction of its standard deviation is reported as uncentred (a note).
_CENTRING_NOTE = 0.1
#: Under "mmle", `MTDR.fit` warns about a continuous column whose mean exceeds
#: this many standard deviations.
_CENTRING_WARNING = 1.0
_MAX_LISTED = 20


# =========================================================================== validate


def validate_inputs(
    Y: ArrayLike,
    X: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    *,
    for_fit: bool = False,
    condition_independent: bool = True,
    min_observations: int = 2,
) -> tuple[FloatArray, FloatArray | None, BoolArray]:
    r"""Validate `(Y, X, mask)` and return them as `float64`, `float64`, `bool`.

    What [`MTDR.fit`][mtdr.model.MTDR.fit] (with `for_fit=True`) and every
    evaluation method run first. It never drops or reorders trials: the
    returned arrays have the input's trial count, and the returned mask is the
    observation pattern $h_{ki}$ of (M3).

    - `Y` must be 3-D `(n_trials, n_neurons, n_bins)` with non-empty axes and
      a real dtype; it is converted to `float64` (a copy only if needed).
    - Without a `mask`, a neuron-trial with any `NaN` bin is unobserved:
      `mask[k, i] = not np.isnan(Y[k, i, :]).any()`. With a `mask` (boolean or
      exactly 0/1), a `NaN` or `Inf` under `True` is an error; values under
      `False` are ignored.
    - `X`, when given, must be 2-D `(n_trials, n_regressors)`, real, finite,
      with at least one column.

    With `for_fit=True` it also applies the fit-only checks: a neuron
    observed on fewer than `min_observations` trials (never observed
    included), a neuron whose observed entries are all equal, a pooled `X`
    whose rank is below its column count, and, with
    `condition_independent=True`, a constant column of `X` or a combination
    of columns that is constant (collinear with the intercept), are
    `ValidationError`s; a trial with no observed neuron is kept with one
    [`DesignWarning`][mtdr.errors.DesignWarning]. The pooled design checks use
    the trials with at least one observed neuron, and every rank test is
    scale- and shift-invariant: a set of columns is rank-deficient when an
    eigenvalue of its unit-diagonal Gram is at most `svd_fit.RANK_TOLERANCE`.

    Parameters
    ----------
    Y : array_like
        `(n_trials, n_neurons, n_bins)` responses; `NaN` allowed as above.
    X : array_like, optional
        `(n_trials, n_regressors)` design, no constant column; `None` where a
        method has no design (`project`).
    mask : array_like, optional
        `(n_trials, n_neurons)` observation mask; `None` infers it from `NaN`.
    for_fit : bool
        Apply the fit-only checks.
    condition_independent : bool
        Whether the model has the intercept; with `False` constant columns
        are allowed.
    min_observations : int
        The per-neuron observation floor, positive (`MTDR` passes
        `max(min_observations, n_regressors + 2)` for `"mmle"`).

    Returns
    -------
    Y : numpy.ndarray
        `float64`, the input's shape.
    X : numpy.ndarray or None
        `float64`, or `None` when not given.
    mask : numpy.ndarray
        `bool`, `(n_trials, n_neurons)`.

    Raises
    ------
    ValidationError
        For any of the data errors above (the message names the offending
        shape, neurons or columns and states the expected layout).
    ParameterError
        For a bad `for_fit`, `condition_independent` or `min_observations`.

    Warns
    -----
    DesignWarning
        With `for_fit=True`, once, listing the trials with no observed neuron.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import validate_inputs
    >>> Y = np.zeros((3, 2, 4))
    >>> Y[1, 0, 2] = np.nan                 # neuron 0 unobserved on trial 1
    >>> Y_out, X_out, mask = validate_inputs(Y, np.array([[1.0], [-1.0], [0.0]]))
    >>> mask
    array([[ True,  True],
           [False,  True],
           [ True,  True]])
    """
    fit_checks = _require_bool("for_fit", for_fit)
    ci = _require_bool("condition_independent", condition_independent)
    floor = as_int(min_observations)
    if floor is None or floor < 1:
        raise ParameterError(
            f"min_observations must be a positive integer; got {min_observations!r}"
        )
    return _validated(Y, X, mask, fit_checks, ci, floor)


def _observation_floor(
    min_observations: int, estimator: str | None, n_regressors: int, ci: bool
) -> tuple[int, str]:
    """Return the per-neuron observation floor and a note on where it comes from.

    `MTDR.fit` and `check_design(estimator=...)` share this one calculation:
    `min_observations`, raised under `"mmle"` to `n_regressors + 1 + ci`,
    below which a neuron's marginal likelihood is unbounded. The note is empty
    unless the floor was raised.
    """
    floor, note = min_observations, ""
    if estimator == "mmle" and n_regressors and n_regressors + 1 + int(ci) > floor:
        floor = n_regressors + 1 + int(ci)
        note = (
            f', the "mmle" floor n_regressors + {1 + int(ci)} (above '
            f"min_observations = {min_observations})"
        )
    return floor, note


def _validated(
    Y: ArrayLike,
    X: ArrayLike | None,
    mask: ArrayLike | None,
    fit_checks: bool,
    ci: bool,
    floor: int,
    floor_note: str = "",
) -> tuple[FloatArray, FloatArray | None, BoolArray]:
    """Run `validate_inputs` on checked arguments.

    `floor_note` explains a floor `MTDR.fit` raised.
    """
    Y_arr = _real_array("Y", Y, 3, "(n_trials, n_neurons, n_bins)")
    n_trials, n_neurons, _ = Y_arr.shape
    X_arr: FloatArray | None = None
    if X is not None:
        X_arr = _real_array("X", X, 2, "(n_trials, n_regressors)")
        if X_arr.shape[0] != n_trials:
            raise ValidationError(
                f"Y has {n_trials} trials (axis 0) but X has {X_arr.shape[0]} rows; "
                "Y must be laid out (n_trials, n_neurons, n_bins) and X (n_trials, "
                "n_regressors), not MATLAB's (n_neurons, n_bins, n_trials)"
            )
        if not np.isfinite(X_arr).all():
            rows = np.flatnonzero(~np.isfinite(X_arr).all(axis=1))
            raise ValidationError(f"X must be finite; non-finite rows {_listed(rows)}")
    if mask is None:
        mask_arr: BoolArray = ~np.isnan(Y_arr).any(axis=2)
        bad = mask_arr & ~np.isfinite(Y_arr).all(axis=2)
        what = "Inf (NaN marks an unobserved neuron-trial when no mask is given)"
    else:
        mask_arr = _mask_array(mask, (n_trials, n_neurons))
        bad = mask_arr & ~np.isfinite(Y_arr).all(axis=2)
        what = "NaN or Inf where mask is True"
    if bad.any():
        pairs = [(int(k), int(i)) for k, i in np.argwhere(bad)]
        raise ValidationError(
            f"Y has {what} at (trial, neuron) {_listed(pairs)}; impute them or mark "
            "those neuron-trials unobserved"
        )
    if fit_checks:
        _fit_checks(Y_arr, X_arr, mask_arr, ci, floor, floor_note)
    return Y_arr, X_arr, mask_arr


def _fit_checks(
    Y: FloatArray,
    X: FloatArray | None,
    mask: BoolArray,
    ci: bool,
    floor: int,
    floor_note: str = "",
) -> None:
    """Apply the fit-only checks that `validate_inputs` documents."""
    n_obs = mask.sum(axis=0)
    few = np.flatnonzero(n_obs < floor)
    if few.size:
        raise ValidationError(
            f"neurons {_listed(few)} are observed on fewer than {floor} trials"
            f"{floor_note} (observed on {_listed(n_obs[few])}); drop those columns "
            "before fitting"
        )
    with np.errstate(invalid="ignore"):
        observed = np.where(mask[:, :, None], Y, np.nan)
        spread = np.nanmax(observed, axis=(0, 2)) - np.nanmin(observed, axis=(0, 2))
    flat = np.flatnonzero(~(spread > 0))
    if flat.size:
        raise ValidationError(
            f"neurons {_listed(flat)} take one value on every observed entry (zero "
            "variance), so their noise precision is unbounded: drop them"
        )
    empty = np.flatnonzero(~mask.any(axis=1))
    if X is not None:
        problems = _design_problems(X, mask, None, ci)
        if problems:
            raise ValidationError("; ".join(problems))
    if empty.size:
        warnings.warn(
            f"trials {_listed(empty)} have no observed neuron; they are kept, "
            "contribute nothing to the fit, and give NaN per-trial outputs",
            DesignWarning,
            stacklevel=3,
        )


# =========================================================================== design


@dataclass(frozen=True, eq=False)
class DesignReport:
    r"""What [`check_design`][mtdr.data.check_design] found.

    Attributes
    ----------
    n_trials : int
        Rows of `X`.
    n_regressors : int
        Columns of `X`.
    regressor_names : tuple of str
        Names of the columns.
    means : numpy.ndarray
        `(P,)` column means over the design rows (the trials with an observed
        neuron when a mask is given).
    stds : numpy.ndarray
        `(P,)` population standard deviations of the columns over the same
        rows.
    constant : numpy.ndarray
        `(P,)` bool, columns that take one value.
    rank : int
        Rank of `X` by the scale-invariant test of
        [`validate_inputs`][mtdr.data.validate_inputs].
    augmented_rank : int
        Rank of `[X, 1]` (shift-invariant: one plus the rank of the centred
        columns).
    duplicate_columns : list of tuple of int
        Pairs `(a, b)`, `a < b`, of exactly collinear columns.
    condition_number : float
        2-norm condition number of the design the per-neuron solves see: the
        columns centred when `condition_independent=True`, each scaled to unit
        norm (`inf` when singular). Scale-free, so it measures collinearity,
        not column scale.
    max_abs_correlation : float
        Largest absolute correlation between two non-constant columns (0 with
        fewer than two).
    few_trial_neurons : numpy.ndarray
        `(n_flagged,)` int, neurons observed on fewer than `min_observations`
        trials (empty without a mask).
    rank_deficient_neurons : numpy.ndarray
        `(n_flagged,)` int, observed neurons whose own design `[X_i, 1]`
        (`X_i` without the intercept) is rank-deficient by `fit_svd`'s test,
        at the given `ridge` (empty without a mask).
    empty_trials : numpy.ndarray
        `(n_flagged,)` int, trials with no observed neuron (empty without a
        mask).
    problems : list of str
        What `fit` would raise on.
    warnings : list of str
        What `fit` would warn about with the same `basis_ridge` and `ridge`.
    notes : list of str
        What `fit` stays silent about: uncentred continuous columns and, at
        `basis_ridge == 0`, scale disparities.
    """

    n_trials: int
    n_regressors: int
    regressor_names: tuple[str, ...]
    means: FloatArray
    stds: FloatArray
    constant: BoolArray
    rank: int
    augmented_rank: int
    duplicate_columns: list[tuple[int, int]]
    condition_number: float
    max_abs_correlation: float
    few_trial_neurons: IntArray
    rank_deficient_neurons: IntArray
    empty_trials: IntArray
    problems: list[str]
    warnings: list[str]
    notes: list[str]

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        freeze(
            [
                self.means,
                self.stds,
                self.constant,
                self.few_trial_neurons,
                self.rank_deficient_neurons,
                self.empty_trials,
            ]
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled report with its arrays read-only again."""
        restore_frozen(self, state)

    def __str__(self) -> str:
        """Return a readable table: one row per column, then the findings."""
        width = max(8, *(len(name) for name in self.regressor_names))
        lines = [
            f"DesignReport: {self.n_trials} trials, {self.n_regressors} regressors, "
            f"rank {self.rank}, rank of [X, 1] {self.augmented_rank}, condition "
            f"number {self.condition_number:.3g}, max |corr| "
            f"{self.max_abs_correlation:.3g}",
            f"{'column':<{width}}  {'mean':>10}  {'std':>10}  constant",
        ]
        for p, name in enumerate(self.regressor_names):
            lines.append(
                f"{name:<{width}}  {self.means[p]:>10.4g}  {self.stds[p]:>10.4g}  "
                f"{bool(self.constant[p])}"
            )
        for title, items in (
            ("problems", self.problems),
            ("warnings", self.warnings),
            ("notes", self.notes),
        ):
            lines.append(f"{title}: " + ("none" if not items else ""))
            lines.extend(f"  - {item}" for item in items)
        return "\n".join(lines)


def check_design(
    X: ArrayLike,
    mask: ArrayLike | None = None,
    regressor_names: Sequence[str] | None = None,
    condition_independent: bool = True,
    basis_ridge: float = 0.0,
    *,
    min_observations: int = 2,
    ridge: float = 0.0,
    estimator: Literal["mmle", "svd"] | None = None,
) -> DesignReport:
    r"""Report what `fit` would reject or warn about in a design, without raising.

    Runs the hard checks `fit` applies to the pooled design (a pooled `X` of
    rank below its column count; with `condition_independent=True` a constant
    column, or a combination of columns that is constant, collinear with the
    intercept of `docs/model.md` § 2) and adds soft diagnostics. With a
    `mask` it also runs the per-neuron checks: neurons below the observation
    floor, neurons whose observed design `[X_i, 1]` is rank-deficient (the
    per-neuron Gram $A_i$ of (M6) by `fit_svd`'s scale-invariant test, at the
    given `ridge`), and trials with no observed neuron. `warnings` lists what
    `fit` would warn about with the same `basis_ridge` and `ridge`; `notes`
    what it stays silent about, so the two never disagree with `fit`.
    Zero-variance neurons need `Y` and are reported by
    [`validate_inputs`][mtdr.data.validate_inputs].

    The observation floor is `min_observations` when `estimator` is `None`.
    With `estimator`, it is the floor [`MTDR.fit`][mtdr.model.MTDR.fit]
    applies for that estimator, by the same calculation: `min_observations`
    for `"svd"`, and
    $\max(\texttt{min\_observations},\,P+1+\mathbf 1_{\rm intercept})$ for
    `"mmle"`, below which a neuron keeps no residual degree of freedom and its
    marginal likelihood (M24) is unbounded. Pass the same
    `condition_independent` and `min_observations` as to `MTDR`.

    Parameters
    ----------
    X : array_like
        `(n_trials, n_regressors)` design, finite.
    mask : array_like, optional
        `(n_trials, n_neurons)` observation mask, boolean or 0/1.
    regressor_names : sequence of str, optional
        Column names, `("x0", "x1", ...)` by default.
    condition_independent : bool
        Whether the model has the intercept.
    basis_ridge : float
        `>= 0`; scale disparities are warnings only when positive.
    min_observations : int
        The per-neuron observation floor (keyword-only), raised as `fit`
        raises it when `estimator="mmle"`.
    ridge : float
        `>= 0`, the least-squares ridge, which removes per-neuron rank
        deficiency (keyword-only).
    estimator : {"mmle", "svd"} or None
        The estimator whose observation floor to apply (keyword-only); `None`
        applies `min_observations` as given.

    Returns
    -------
    DesignReport
        The findings; `str(report)` prints a table.

    Raises
    ------
    ValidationError
        If `X` is not a finite 2-D real array, or the mask has the wrong
        shape or values.
    ParameterError
        For bad names, `basis_ridge`, `ridge`, `min_observations` or
        `estimator`.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import check_design
    >>> rng = np.random.default_rng(0)
    >>> X = np.column_stack([rng.normal(size=50), np.ones(50)])
    >>> report = check_design(X, regressor_names=["coherence", "bias"])
    >>> report.constant.tolist(), report.rank, report.augmented_rank
    ([False, True], 2, 2)
    >>> print(report.problems[0])  # doctest: +NORMALIZE_WHITESPACE
    column(s) ['bias'] constant: collinear with the intercept; drop it or pass
    condition_independent=False

    The `"mmle"` floor, with three regressors and the intercept, is 5 trials:

    >>> mask = np.ones((50, 3), dtype=bool)
    >>> mask[4:, 0] = False                       # neuron 0 seen on 4 trials
    >>> X3 = rng.normal(size=(50, 3))
    >>> check_design(X3, mask).few_trial_neurons.tolist()
    []
    >>> report = check_design(X3, mask, estimator="mmle")
    >>> report.few_trial_neurons.tolist()
    [0]
    >>> print(report.problems[0])  # doctest: +NORMALIZE_WHITESPACE
    neurons [0] are observed on fewer than 5 trials, the "mmle" floor
    n_regressors + 2 (above min_observations = 2)
    """
    X_arr = _real_array("X", X, 2, "(n_trials, n_regressors)")
    if not np.isfinite(X_arr).all():
        rows = np.flatnonzero(~np.isfinite(X_arr).all(axis=1))
        raise ValidationError(f"X must be finite; non-finite rows {_listed(rows)}")
    n_trials, P = X_arr.shape
    names = require_names(regressor_names, P)
    ci = _require_bool("condition_independent", condition_independent)
    g = require_non_negative_real("basis_ridge", basis_ridge)
    gamma = require_non_negative_real("ridge", ridge)
    floor = as_int(min_observations)
    if floor is None or floor < 1:
        raise ParameterError(
            f"min_observations must be a positive integer; got {min_observations!r}"
        )
    if estimator is not None and estimator not in _ESTIMATORS:
        raise ParameterError(
            f"estimator must be one of {_ESTIMATORS} or None; got {estimator!r}"
        )
    floor, floor_note = _observation_floor(floor, estimator, P, ci)
    mask_arr = None if mask is None else _mask_array(mask, (n_trials, -1))
    rows = _design_rows(X_arr, mask_arr)
    Xr = X_arr[rows]
    means = Xr.mean(axis=0)
    stds = Xr.std(axis=0)
    constant = np.ptp(Xr, axis=0) == 0
    rank = _rank(Xr)
    augmented = _augmented_rank(Xr)
    duplicates = [
        (a, b) for a in range(P) for b in range(a + 1, P) if _rank(Xr[:, [a, b]]) < 2
    ]
    cond = _condition_number(_centred(Xr) if ci else Xr)
    corr = _max_abs_correlation(Xr, constant)

    problems = _design_problems(X_arr, mask_arr, names, ci)
    warn = _soft_warnings(X_arr, mask_arr, names, g)
    notes: list[str] = []
    few = np.zeros(0, dtype=np.int64)
    deficient = np.zeros(0, dtype=np.int64)
    empty = np.zeros(0, dtype=np.int64)
    if mask_arr is not None:
        n_obs = mask_arr.sum(axis=0)
        few = np.flatnonzero(n_obs < floor).astype(np.int64)
        if few.size:
            problems.append(
                f"neurons {_listed(few)} are observed on fewer than {floor} trials"
                f"{floor_note}"
            )
        deficient = _rank_deficient_neurons(X_arr, mask_arr, ci, gamma)
        if deficient.size:
            warn.insert(0, _deficient_message(deficient))
        empty = np.flatnonzero(~mask_arr.any(axis=1)).astype(np.int64)
        if empty.size:
            warn.insert(0, _empty_message(empty))
    distinct = np.array([np.unique(Xr[:, p]).size for p in range(P)])
    uncentred = [
        names[p]
        for p in range(P)
        if distinct[p] > 2 and abs(means[p]) > _CENTRING_NOTE * stds[p]
    ]
    if uncentred:
        notes.append(
            f"continuous column(s) {uncentred} are not centred; the fit does not "
            "need it, but under 'mmle' the marginal likelihood depends on where a "
            "regressor's zero is; centre continuous regressors and code binary "
            "ones as +-1"
        )
    if not g:
        ratio = _scale_ratio(stds, constant)
        if ratio > _SCALE_RATIO:
            notes.append(_scale_message(ratio, stds, constant, names, warning=False))
    return DesignReport(
        n_trials=n_trials,
        n_regressors=P,
        regressor_names=names,
        means=means,
        stds=stds,
        constant=constant,
        rank=rank,
        augmented_rank=augmented,
        duplicate_columns=duplicates,
        condition_number=cond,
        max_abs_correlation=corr,
        few_trial_neurons=few,
        rank_deficient_neurons=deficient,
        empty_trials=empty,
        problems=problems,
        warnings=warn,
        notes=notes,
    )


def _design_rows(X: FloatArray, mask: BoolArray | None) -> NDArray[np.intp]:
    """Return the design rows the pooled checks use: trials with a neuron."""
    if mask is None:
        return np.arange(X.shape[0])
    rows = np.flatnonzero(mask.any(axis=1))
    return rows if rows.size else np.arange(X.shape[0])


def _gram_eigenvalues(M: FloatArray) -> FloatArray:
    """Eigenvalues of the unit-diagonal Gram of `M`'s columns, for the rank test."""
    norms = np.sqrt(np.einsum("kp,kp->p", M, M))
    scaled = M / np.where(norms > 0, norms, 1.0)
    out: FloatArray = np.linalg.eigvalsh(scaled.T @ scaled)
    return out


def _rank(M: FloatArray) -> int:
    """Rank: the eigenvalues of the unit-diagonal Gram above the tolerance."""
    return int(np.sum(_gram_eigenvalues(M) > RANK_TOLERANCE))


def _centred(M: FloatArray) -> FloatArray:
    """Centre the columns: shift by the first row, then by the mean of the result.

    A constant column centres to exactly zero.
    """
    shifted = M - M[0]
    out: FloatArray = shifted - shifted.mean(axis=0)
    return out


def _augmented_rank(M: FloatArray) -> int:
    """Rank of `[M, 1]`, shift-invariantly: one plus the rank of the centred columns."""
    return 1 + _rank(_centred(M))


def _condition_number(M: FloatArray) -> float:
    norms = np.sqrt(np.einsum("kp,kp->p", M, M))
    if not (norms > 0).all():
        return float("inf")
    s = np.linalg.svd(M / norms, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def _max_abs_correlation(M: FloatArray, constant: BoolArray) -> float:
    keep = M[:, ~constant]
    if keep.shape[1] < 2:
        return 0.0
    corr = np.asarray(np.corrcoef(keep, rowvar=False))
    off = np.abs(corr[~np.eye(keep.shape[1], dtype=bool)])
    return float(off.max())


def _scale_ratio(stds: FloatArray, constant: BoolArray) -> float:
    live = stds[~constant]
    if live.size < 2:
        return 1.0
    return float(live.max() / live.min())


def _scale_message(
    ratio: float,
    stds: FloatArray,
    constant: BoolArray,
    names: Sequence[str],
    *,
    warning: bool,
) -> str:
    live = np.flatnonzero(~constant)
    big = names[int(live[np.argmax(stds[live])])]
    small = names[int(live[np.argmin(stds[live])])]
    head = (
        f"column scales differ by a factor {ratio:.3g} (std of {big!r} over std "
        f"of {small!r})"
    )
    if warning:
        return (
            head + "; with basis_ridge > 0 the penalty on the bases makes the fit "
            "depend on column scale: put the columns on comparable scales"
        )
    return (
        head + "; at basis_ridge = 0 the fit is invariant to column scale, "
        "but comparable scales condition the solves better"
    )


def _design_problems(
    X: FloatArray, mask: BoolArray | None, names: Sequence[str] | None, ci: bool
) -> list[str]:
    """Return the pooled design errors as messages."""
    P = X.shape[1]
    labels = list(names) if names is not None else [f"x{p}" for p in range(P)]
    Xr = X[_design_rows(X, mask)]
    problems: list[str] = []
    rank = _rank(Xr)
    if rank < P:
        problems.append(
            f"X has rank {rank} < {P} columns (exactly collinear columns); drop the "
            "redundant ones"
        )
    if ci:
        constant = np.ptp(Xr, axis=0) == 0
        if constant.any():
            problems.append(
                f"column(s) {[labels[p] for p in np.flatnonzero(constant)]} constant: "
                "collinear with the intercept; drop it or pass "
                "condition_independent=False"
            )
        elif rank == P and _augmented_rank(Xr) < P + 1:
            problems.append(
                "a combination of the columns of X is constant, so it is collinear "
                "with the intercept: with condition_independent=True an indicator "
                "coding must drop one level"
            )
    return problems


def _uncentred_message(
    X: FloatArray, mask: BoolArray | None, names: Sequence[str]
) -> str | None:
    """Return `fit`'s "mmle" warning about far-off-centre columns, if any.

    A continuous column (more than two distinct values on the design rows) whose
    mean is more than `_CENTRING_WARNING` standard deviations from zero. Binary
    codings (0/1, +-1) and constant columns are never flagged.
    """
    Xr = X[_design_rows(X, mask)]
    means, stds = Xr.mean(axis=0), Xr.std(axis=0)
    flagged = [
        names[p]
        for p in range(X.shape[1])
        if np.unique(Xr[:, p]).size > 2 and abs(means[p]) > _CENTRING_WARNING * stds[p]
    ]
    if not flagged:
        return None
    return (
        f"continuous column(s) {flagged} have a mean more than one standard "
        "deviation from zero; under 'mmle' the marginal likelihood, and so the "
        "selected ranks, depend on where a regressor's zero is: centre continuous "
        "regressors before fitting"
    )


def _soft_warnings(
    X: FloatArray, mask: BoolArray | None, names: Sequence[str], basis_ridge: float
) -> list[str]:
    """Return `fit`'s pooled-design warnings.

    Near-collinearity and, with a basis ridge, column scale.
    """
    Xr = X[_design_rows(X, mask)]
    constant = np.ptp(Xr, axis=0) == 0
    out: list[str] = []
    keep = np.flatnonzero(~constant)
    if keep.size >= 2:
        corr = np.abs(np.corrcoef(Xr[:, keep], rowvar=False))
        pairs = [
            (names[int(keep[a])], names[int(keep[b])], float(corr[a, b]))
            for a in range(keep.size)
            for b in range(a + 1, keep.size)
            if corr[a, b] > _CORRELATION_WARNING
        ]
        if pairs:
            listed = ", ".join(f"{a}/{b} ({c:.3f})" for a, b, c in pairs)
            out.append(
                f"near-collinear columns of X, |corr| > {_CORRELATION_WARNING}: "
                f"{listed}; their coefficients are poorly determined"
            )
    if basis_ridge > 0:
        stds = Xr.std(axis=0)
        ratio = _scale_ratio(stds, constant)
        if ratio > _SCALE_RATIO:
            out.append(_scale_message(ratio, stds, constant, names, warning=True))
    return out


def _rank_deficient_neurons(
    X: FloatArray, mask: BoolArray, ci: bool, ridge: float
) -> IntArray:
    """Observed neurons `fit_svd` would flag, from the design alone."""
    observed = np.flatnonzero(mask.any(axis=0))
    if not observed.size:
        return np.zeros(0, dtype=np.int64)
    zeros = np.zeros((X.shape[0], observed.size, 1))
    stats = sufficient_statistics(zeros, X, mask[:, observed])
    G, rhs = _normal_equations(stats, ci, ridge)
    _, deficient = _least_squares(G, rhs)
    out: IntArray = observed[deficient].astype(np.int64)
    return out


def _deficient_message(neurons: IntArray) -> str:
    return (
        f"neurons {_listed(neurons)} have a rank-deficient observed design (fewer "
        "observed trials than coefficients, collinear regressors, or a regressor "
        "constant over their trials); the least-squares stage gives them the "
        "minimum-norm solution; ridge > 0 removes this"
    )


def _empty_message(trials: IntArray) -> str:
    return (
        f"trials {_listed(trials)} have no observed neuron; they are kept, contribute "
        "nothing to the fit, and give NaN per-trial outputs"
    )


# =========================================================================== averages


@dataclass(frozen=True, eq=False)
class ConditionAverage:
    """Condition-averaged data from [`condition_average`][mtdr.data.condition_average].

    Attributes
    ----------
    Y : numpy.ndarray
        `(n_conditions, n_neurons, n_bins)` masked means; `NaN` where the
        neuron has fewer than `min_trials` trials in the condition.
    X : numpy.ndarray
        `(n_conditions, n_regressors)`: each condition's regressor values (the
        `by` columns), and the unweighted mean over its trials for the others.
    mask : numpy.ndarray
        `(n_conditions, n_neurons)` bool, the neuron has at least `min_trials`
        trials in the condition.
    counts : numpy.ndarray
        `(n_conditions, n_neurons)` int, observed trials per neuron and
        condition (informational; `project` does not weight by it).
    condition_of_trial : numpy.ndarray
        `(n_trials,)` int, the row of each input trial.
    """

    Y: FloatArray
    X: FloatArray
    mask: BoolArray
    counts: IntArray
    condition_of_trial: IntArray

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        freeze([self.Y, self.X, self.mask, self.counts, self.condition_of_trial])

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled object with its arrays read-only again."""
        restore_frozen(self, state)

    def __repr__(self) -> str:
        """Summarise shapes instead of printing the arrays."""
        n_cond, n_neurons, n_bins = self.Y.shape
        return (
            f"ConditionAverage(n_conditions={n_cond}, n_neurons={n_neurons}, "
            f"n_bins={n_bins}, observed={float(self.mask.mean()):.3g})"
        )


def condition_average(
    Y: ArrayLike,
    X: ArrayLike,
    mask: ArrayLike | None = None,
    by: Sequence[int] | Sequence[str] | None = None,
    regressor_names: Sequence[str] | None = None,
    min_trials: int = 1,
) -> ConditionAverage:
    r"""Average trials within conditions, per neuron, over its observed trials.

    Conditions are the unique rows of `X[:, by]`, ordered as
    `numpy.unique(X[:, by], axis=0)` orders them (lexicographic); for neuron
    $i$ and condition $c$ the average is
    $\bar Y_{ci}=\frac1{n_{ci}}\sum_{k\in c}h_{ki}Y_{ki\cdot}$ with
    $n_{ci}=\sum_{k\in c}h_{ki}$ its observed trials there, $h$ the mask of
    (M3). A neuron with fewer than `min_trials` trials in a condition has
    `mask` `False` and `Y` `NaN` there; every condition row is kept, so
    conditions line up across neurons and calls. The result plugs into
    [`MTDR.project`][mtdr.model.MTDR.project] with its mask. This is a masked
    mean; pseudo-trial resampling is described in the
    [Nature Neuroscience supplement, §6.1](https://doi.org/10.1038/s41593-020-0696-5).

    Parameters
    ----------
    Y : array_like
        `(n_trials, n_neurons, n_bins)`, as
        [`validate_inputs`][mtdr.data.validate_inputs] (`NaN` allowed).
    X : array_like
        `(n_trials, n_regressors)`.
    mask : array_like, optional
        `(n_trials, n_neurons)`; `None` infers it from `NaN`.
    by : sequence of int or of str, optional
        Columns of `X` that define a condition, by index or by name; `None`
        uses every column. Continuous regressors should be binned first.
    regressor_names : sequence of str, optional
        Names of the columns of `X`, needed when `by` holds names
        (`("x0", ...)` by default).
    min_trials : int
        Positive; the per-neuron, per-condition trial floor.

    Returns
    -------
    ConditionAverage
        The averaged arrays, read-only.

    Raises
    ------
    ValidationError
        As `validate_inputs`.
    ParameterError
        For unknown or repeated `by` columns, bad names or `min_trials`.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import condition_average
    >>> Y = np.arange(4.0).reshape(4, 1, 1)          # 4 trials, 1 neuron, 1 bin
    >>> X = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    >>> avg = condition_average(Y, X)
    >>> avg.X.ravel().tolist(), avg.Y.ravel().tolist(), avg.condition_of_trial.tolist()
    ([-1.0, 1.0], [2.0, 1.0], [1, 0, 1, 0])
    """
    Y_arr, X_arr, mask_arr = validate_inputs(Y, X, mask)
    assert X_arr is not None
    P = X_arr.shape[1]
    names = require_names(regressor_names, P)
    columns = _columns("by", by, names)
    floor = as_int(min_trials)
    if floor is None or floor < 1:
        raise ParameterError(
            f"min_trials must be a positive integer; got {min_trials!r}"
        )
    _, condition = np.unique(X_arr[:, columns], axis=0, return_inverse=True)
    condition = condition.reshape(-1).astype(np.int64)
    n_cond = int(condition.max()) + 1
    onehot = np.zeros((n_cond, X_arr.shape[0]))
    onehot[condition, np.arange(X_arr.shape[0])] = 1.0
    counts = (onehot @ mask_arr.astype(np.float64)).astype(np.int64)
    Y0 = np.where(mask_arr[:, :, None], Y_arr, 0.0)
    sums = np.einsum("ck,kit->cit", onehot, Y0)
    keep = counts >= floor
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts[:, :, None]
    Y_out = np.where(keep[:, :, None], means, np.nan)
    X_out = (onehot @ X_arr) / onehot.sum(axis=1)[:, None]
    return ConditionAverage(
        Y=Y_out, X=X_out, mask=keep, counts=counts, condition_of_trial=condition
    )


# =========================================================================== sessions


@dataclass(frozen=True, eq=False)
class StackedSessions:
    """Sessions assembled by [`stack_sessions`][mtdr.data.stack_sessions].

    Attributes
    ----------
    Y : numpy.ndarray
        `(sum K_s, sum n_s, T)`; `NaN` off the block diagonal.
    X : numpy.ndarray
        `(sum K_s, P)`, the sessions' designs stacked.
    mask : numpy.ndarray
        `(sum K_s, sum n_s)` bool, block-diagonal (each block AND-ed with the
        session's own mask).
    session_of_trial : numpy.ndarray
        `(sum K_s,)` int, the session of each trial (row).
    session_of_neuron : numpy.ndarray
        `(sum n_s,)` int, the session of each neuron (column).
    trial_index_in_session : numpy.ndarray
        `(sum K_s,)` int, each trial's position within its session.
    neuron_index_in_session : numpy.ndarray
        `(sum n_s,)` int, each neuron's position within its session.
    """

    Y: FloatArray
    X: FloatArray
    mask: BoolArray
    session_of_trial: IntArray
    session_of_neuron: IntArray
    trial_index_in_session: IntArray
    neuron_index_in_session: IntArray

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        freeze(
            [
                self.Y,
                self.X,
                self.mask,
                self.session_of_trial,
                self.session_of_neuron,
                self.trial_index_in_session,
                self.neuron_index_in_session,
            ]
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled object with its arrays read-only again."""
        restore_frozen(self, state)

    def __repr__(self) -> str:
        """Summarise shapes instead of printing the arrays."""
        n_trials, n_neurons, n_bins = self.Y.shape
        n_sessions = int(self.session_of_trial.max()) + 1
        return (
            f"StackedSessions(n_sessions={n_sessions}, n_trials={n_trials}, "
            f"n_neurons={n_neurons}, n_bins={n_bins})"
        )


def stack_sessions(sessions: Sequence[Sequence[ArrayLike]]) -> StackedSessions:
    r"""Stack sequentially recorded sessions into one block-diagonal dataset.

    Session $s$ contributes its trials and its neurons; a neuron of session
    $s$ is observed only on session $s$'s trials, so the mask is
    block-diagonal and every neuron has its own trial set and its own
    $X_i^\top X_i$: exactly the observation model (M3), (M9) the estimators
    are written for. Each session's mask is the given one or inferred from
    `NaN`. The trial axis of the result is the union of the sessions' trials,
    so memory grows as $8\,\sum_sK_s\,\sum_sn_s\,T$ bytes. The regressors must
    mean the same thing in every session.

    Parameters
    ----------
    sessions : sequence of tuples
        Each `(Y_s, X_s)` or `(Y_s, X_s, mask_s)`, `Y_s` of shape
        `(K_s, n_s, T)` and `X_s` `(K_s, P)`, with the same `T` and `P`.

    Returns
    -------
    StackedSessions
        The stacked arrays and the index maps back to the sessions.

    Raises
    ------
    ParameterError
        If `sessions` is not a non-empty sequence of 2- or 3-tuples.
    ValidationError
        If a session's arrays are invalid (as `validate_inputs`) or the
        sessions disagree on `T` or `P`.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import stack_sessions
    >>> a = (np.ones((3, 2, 5)), np.array([[1.0], [-1.0], [1.0]]))
    >>> b = (np.zeros((2, 1, 5)), np.array([[-1.0], [1.0]]))
    >>> st = stack_sessions([a, b])
    >>> st.Y.shape, st.mask.astype(int).tolist()
    ((5, 3, 5), [[1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]])
    >>> st.session_of_neuron.tolist(), st.trial_index_in_session.tolist()
    ([0, 0, 1], [0, 1, 2, 0, 1])
    """
    entries = as_list(sessions)
    if not entries:
        raise ParameterError(
            "sessions must be a non-empty sequence of (Y_s, X_s) or (Y_s, X_s, "
            "mask_s) tuples"
        )
    parts: list[tuple[FloatArray, FloatArray, BoolArray]] = []
    for s, entry in enumerate(entries):
        items = as_list(entry)
        if items is None or len(items) not in (2, 3):
            raise ParameterError(
                f"sessions[{s}] must be a (Y_s, X_s) or (Y_s, X_s, mask_s) tuple"
            )
        mask_s = items[2] if len(items) == 3 else None
        try:
            Y_s, X_s, m_s = validate_inputs(items[0], items[1], mask_s)  # type: ignore[arg-type]
        except ValidationError as err:
            raise ValidationError(f"sessions[{s}]: {err}") from err
        assert X_s is not None
        parts.append((Y_s, X_s, m_s))
    T, P = parts[0][0].shape[2], parts[0][1].shape[1]
    for s, (Y_s, X_s, _) in enumerate(parts):
        if Y_s.shape[2] != T or X_s.shape[1] != P:
            raise ValidationError(
                f"sessions[{s}] has {Y_s.shape[2]} bins and {X_s.shape[1]} regressors; "
                f"session 0 has {T} and {P}: every session needs the same time bins "
                "and regressors"
            )
    K = [Y_s.shape[0] for Y_s, _, _ in parts]
    n = [Y_s.shape[1] for Y_s, _, _ in parts]
    Y = np.full((sum(K), sum(n), T), np.nan)
    mask = np.zeros((sum(K), sum(n)), dtype=bool)
    k0 = i0 = 0
    for (Y_s, _, m_s), k, i in zip(parts, K, n, strict=True):
        Y[k0 : k0 + k, i0 : i0 + i] = np.where(m_s[:, :, None], Y_s, np.nan)
        mask[k0 : k0 + k, i0 : i0 + i] = m_s
        k0, i0 = k0 + k, i0 + i
    return StackedSessions(
        Y=Y,
        X=np.concatenate([X_s for _, X_s, _ in parts], axis=0),
        mask=mask,
        session_of_trial=np.repeat(np.arange(len(K)), K).astype(np.int64),
        session_of_neuron=np.repeat(np.arange(len(n)), n).astype(np.int64),
        trial_index_in_session=np.concatenate([np.arange(k) for k in K]).astype(
            np.int64
        ),
        neuron_index_in_session=np.concatenate([np.arange(i) for i in n]).astype(
            np.int64
        ),
    )


# =========================================================================== split


def split_trials(
    n_trials: int,
    test_fraction: float = 0.2,
    random_state: int | np.random.Generator | None = None,
    stratify: ArrayLike | None = None,
) -> tuple[IntArray, IntArray]:
    r"""Split trial indices into a training and a test set.

    Without `stratify`, `round(test_fraction * n_trials)` trials (at least one
    in each half) go to the test set at random. With `stratify`, each label's
    trials are split the same way, so each label's proportion is
    approximately preserved in both halves: a label with a single trial goes
    to the training half (one [`DesignWarning`][mtdr.errors.DesignWarning]
    lists such labels), a label with two trials goes one each way. The
    function sees no mask, so it cannot promise that every neuron is observed
    in both halves; check `mask[train].any(0)` and `mask[test].any(0)`.

    Parameters
    ----------
    n_trials : int
        Number of trials, at least 2.
    test_fraction : float
        In `(0, 1)`.
    random_state : int, numpy.random.Generator or None
        Seed (non-negative) or generator (`numpy.random.default_rng`).
    stratify : array_like, optional
        `(n_trials,)` labels, e.g. `condition_of_trial` or `session_of_trial`
        (for a joint stratification pass a combined label).

    Returns
    -------
    train : numpy.ndarray
        Sorted `int64` training-trial indices.
    test : numpy.ndarray
        Sorted `int64` test-trial indices; with `train` a partition of
        `range(n_trials)`.

    Raises
    ------
    ParameterError
        For a bad `n_trials`, `test_fraction`, `random_state` or `stratify`.

    Warns
    -----
    DesignWarning
        When a stratification label has a single trial.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import split_trials
    >>> train, test = split_trials(10, test_fraction=0.3, random_state=0)
    >>> both = np.concatenate([train, test])
    >>> len(train), len(test), sorted(both.tolist()) == list(range(10))
    (7, 3, True)
    """
    n = as_int(n_trials)
    if n is None or n < 2:
        raise ParameterError(f"n_trials must be an integer >= 2; got {n_trials!r}")
    fraction = as_real(test_fraction)
    if fraction is None or not 0.0 < fraction < 1.0:
        raise ParameterError(f"test_fraction must be in (0, 1); got {test_fraction!r}")
    seed = as_int(random_state)
    if (
        random_state is not None
        and not isinstance(random_state, np.random.Generator)
        and (seed is None or seed < 0)
    ):
        raise ParameterError(
            "random_state must be a non-negative int, a numpy.random.Generator or "
            f"None; got {random_state!r}"
        )
    rng = np.random.default_rng(random_state)
    if stratify is None:
        groups = [np.arange(n)]
        labels: list[object] = [None]
    else:
        lab = np.asarray(stratify)
        if lab.shape != (n,):
            raise ParameterError(
                f"stratify must have shape (n_trials,) = ({n},); got {lab.shape}"
            )
        uniq, inverse = np.unique(lab, return_inverse=True)
        groups = [np.flatnonzero(inverse == g) for g in range(uniq.size)]
        labels = list(uniq)
    test: list[int] = []
    singles: list[object] = []
    for label, members in zip(labels, groups, strict=True):
        shuffled = rng.permutation(members)
        m = shuffled.size
        if m == 1:
            singles.append(label)
            continue
        k = int(np.clip(round(fraction * m), 1, m - 1))
        test.extend(int(t) for t in shuffled[:k])
    if singles:
        shown = ", ".join(repr(_plain(s)) for s in singles[:_MAX_LISTED])
        more = (
            f" and {len(singles) - _MAX_LISTED} more"
            if len(singles) > _MAX_LISTED
            else ""
        )
        warnings.warn(
            f"stratification labels [{shown}{more}] have a single trial; it goes to "
            "the training half",
            DesignWarning,
            stacklevel=2,
        )
    test_idx = np.array(sorted(test), dtype=np.int64)
    train_idx = np.setdiff1d(np.arange(n), test_idx).astype(np.int64)
    return train_idx, test_idx


# =========================================================================== helpers


def _plain(value: object) -> object:
    """Return a NumPy scalar as its Python value, for messages."""
    return value.item() if isinstance(value, np.generic) else value


def _require_bool(name: str, value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    raise ParameterError(f"{name} must be a bool; got {value!r}")


def _columns(name: str, value: object, names: tuple[str, ...]) -> list[int]:
    """Resolve a `by` argument (indices or names) to column indices."""
    P = len(names)
    if value is None:
        return list(range(P))
    entries = as_list(value)
    if not entries:
        raise ParameterError(
            f"{name} must be a non-empty sequence of column indices or names"
        )
    out: list[int] = []
    for entry in entries:
        if isinstance(entry, str):
            if entry not in names:
                raise ParameterError(
                    f"{name}: unknown regressor {entry!r}; names {names}"
                )
            out.append(names.index(entry))
            continue
        index = as_int(entry)
        if index is None or not 0 <= index < P:
            raise ParameterError(
                f"{name}: {entry!r} is not a column index in [0, {P}) or a name"
            )
        out.append(index)
    if len(set(out)) != len(out):
        raise ParameterError(f"{name} repeats a column: {entries!r}")
    return out


def _as_array(name: str, value: object) -> NDArray[Any]:
    """Return `np.asarray(value)`, a `ValidationError` if NumPy cannot."""
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as err:
        raise ValidationError(
            f"{name} is ragged or not an array (nested sequences of unequal "
            f"lengths?): {err}"
        ) from err


def _real_array(name: str, value: object, ndim: int, layout: str) -> FloatArray:
    """`value` as a `float64` array with `ndim` non-empty axes, else raise."""
    arr = _as_array(name, value)
    if arr.dtype.kind not in "biuf":
        recipe = (
            "; encode factors numerically: centre continuous regressors, code "
            "binary ones +-1, use indicators with one level dropped"
            if name == "X"
            else ""
        )
        raise ValidationError(
            f"{name} must be real numeric; got dtype {arr.dtype}{recipe}"
        )
    if arr.ndim != ndim:
        raise ValidationError(
            f"{name} must be {ndim}-D {layout}; got shape {arr.shape}"
        )
    if 0 in arr.shape:
        raise ValidationError(
            f"{name} must have non-empty axes {layout}; got shape {arr.shape}"
        )
    return arr.astype(np.float64, copy=False)


def _mask_array(value: object, shape: tuple[int, int]) -> BoolArray:
    """Return the mask as `bool`, checked for `shape` (`-1`: any width), 0/1."""
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as err:
        raise ValidationError(f"mask is not an array: {err}") from err
    expected = shape if shape[1] >= 0 else (shape[0], arr.shape[-1] if arr.ndim else -1)
    if arr.ndim != 2 or arr.shape != expected or 0 in arr.shape:
        raise ValidationError(
            f"mask must have shape (n_trials, n_neurons) = {expected}; got {arr.shape}"
        )
    if arr.dtype == np.bool_:
        return arr
    if arr.dtype.kind not in "iuf" or not np.isin(arr, (0, 1)).all():
        raise ValidationError("mask must be boolean or hold only 0 and 1")
    out: BoolArray = arr == 1
    return out


def _listed(items: Any) -> str:
    values = [_plain(item) for item in items]
    head = ", ".join(str(v) for v in values[:_MAX_LISTED])
    more = f" and {len(values) - _MAX_LISTED} more" if len(values) > _MAX_LISTED else ""
    return f"[{head}{more}]"
