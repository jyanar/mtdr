"""Exceptions and warnings raised by `mtdr`.

Every class is importable from here and, as an alias, from the top-level package
(`from mtdr import ValidationError`).

Each exception also subclasses the built-in (or NumPy) exception a caller would
naturally catch, so code written without knowledge of `mtdr` still handles it:

- [`ValidationError`][mtdr.errors.ValidationError] — a [`ValueError`][]; invalid
  or degenerate data.
- [`ParameterError`][mtdr.errors.ParameterError] — a [`ValueError`][]; an invalid
  argument that is not data.
- [`SingularDesignError`][mtdr.errors.SingularDesignError] — a
  [`numpy.linalg.LinAlgError`][] (and so a [`ValueError`][]); a structurally
  singular joint `project`.
- [`NotFittedError`][mtdr.errors.NotFittedError] — an [`AttributeError`][]; a
  fitted attribute used before `fit`.

All warnings derive from [`MTDRWarning`][mtdr.errors.MTDRWarning], a
[`UserWarning`][], so one filter silences or escalates every warning the package
emits: `warnings.simplefilter("error", mtdr.MTDRWarning)`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ConvergenceWarning",
    "DecodingWarning",
    "DesignWarning",
    "MTDRError",
    "MTDRWarning",
    "NotFittedError",
    "ParameterError",
    "ProjectionWarning",
    "SingularDesignError",
    "ValidationError",
]


# --------------------------------------------------------------------------- errors


class MTDRError(Exception):
    """Root of every exception class defined by `mtdr`.

    Never raised directly; catch it to handle every error `mtdr` raises
    deliberately: invalid or degenerate data, invalid arguments, an unfitted
    model, a structurally singular projection. Two exceptions are not
    subclasses, by design: the [`numpy.linalg.LinAlgError`][] raised when a
    per-neuron Cholesky factorisation fails, a numerical failure of NumPy's
    routine, and [`ImportError`][] for a missing optional extra.

    Examples
    --------
    >>> from mtdr.errors import MTDRError, ValidationError
    >>> issubclass(ValidationError, MTDRError)
    True
    """


class ValidationError(MTDRError, ValueError):
    """Input data are invalid or make the fit degenerate.

    Raised for wrong shapes or layouts (`Y[k, i, t]`, `X[k, p]`, `mask[k, i]`),
    empty axes, non-finite values in observed entries of `Y` (a `NaN` or `Inf`
    where the mask is `True`) or anywhere in `X`, and a mask that is neither
    boolean nor exactly 0/1. In `fit` only: a rank-deficient pooled `X`, a
    constant column when `condition_independent=True`, neurons never observed or
    observed on fewer than `min_observations` trials, neurons whose observed
    entries are all equal, and neurons fitted with zero residual (an unbounded
    noise precision). Also raised by `explained_variance` when a variance it
    divides by is zero. The message names the offending neurons, trials or bins.

    A [`ValueError`][] subclass, so `except ValueError` catches it.

    Examples
    --------
    >>> from mtdr.errors import ValidationError
    >>> try:
    ...     raise ValidationError("Y must be 3-D (n_trials, n_neurons, n_bins)")
    ... except ValueError as err:
    ...     print(err)
    Y must be 3-D (n_trials, n_neurons, n_bins)
    """


class SingularDesignError(MTDRError, np.linalg.LinAlgError):
    """A joint projection is structurally ill-posed.

    Raised by `MTDR.project(..., regressor=None)` when the concatenated weight
    basis of all regressors is rank-deficient (two regressors share a direction,
    or `total_rank_ > n_neurons`), so joint coordinates are undefined on every
    trial. The message names the regressors involved. Per-neuron singular designs
    in `fit` do **not** raise this error; they get the minimum-norm solution and a
    [`DesignWarning`][mtdr.errors.DesignWarning].

    A [`numpy.linalg.LinAlgError`][] subclass, and therefore also a
    [`ValueError`][].

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr.errors import SingularDesignError
    >>> issubclass(SingularDesignError, np.linalg.LinAlgError)
    True
    >>> issubclass(SingularDesignError, ValueError)
    True
    """


class ParameterError(MTDRError, ValueError):
    """An argument that is not data is invalid.

    Raised for invalid constructor hyper-parameters, regressor names, `ranks`
    specifications, `max_rank` against the data, basis shapes or ranks,
    `method`/`regressor` combinations, conflicting or missing `decode`
    arguments, and level sets. Invalid data raise
    [`ValidationError`][mtdr.errors.ValidationError] instead.

    A [`ValueError`][] subclass, so `except ValueError` catches it.

    Examples
    --------
    >>> from mtdr.errors import MTDRError, ParameterError
    >>> issubclass(ParameterError, ValueError) and issubclass(ParameterError, MTDRError)
    True
    """


class NotFittedError(MTDRError, AttributeError):
    """A fitted attribute or method was used before `fit`.

    An [`AttributeError`][] subclass, so `hasattr(model, "ranks_")` returns
    `False` on an unfitted model instead of raising.

    Examples
    --------
    >>> from mtdr.errors import NotFittedError
    >>> issubclass(NotFittedError, AttributeError)
    True
    """


# --------------------------------------------------------------------------- warnings


class MTDRWarning(UserWarning):
    """Root of every warning emitted by `mtdr`.

    Examples
    --------
    >>> import warnings
    >>> from mtdr.errors import DesignWarning, MTDRWarning
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always", MTDRWarning)
    ...     warnings.warn("neuron 3 has a rank-deficient design", DesignWarning)
    >>> caught[0].category.__name__
    'DesignWarning'
    """


class ConvergenceWarning(MTDRWarning):
    """An iterative fit stopped before meeting its tolerance.

    Emitted when an ECME or refinement loop reaches its iteration cap, or when an
    inner optimiser reports failure. The fit is still returned.
    """


class DesignWarning(MTDRWarning):
    """A soft problem with the design or the observation pattern.

    Emitted, for example, at `fit` for neurons whose observed design is
    rank-deficient (they get the minimum-norm solution; `ridge > 0` removes the
    warning), for near-collinear regressors, for trials on which no neuron is
    observed, for column-scale disparities when `basis_ridge > 0` (the
    unpenalised fit is scale-invariant), and by `split_trials` for a
    stratification label with a single trial.
    """


class DecodingWarning(MTDRWarning):
    """Some trials could not be decoded.

    Emitted once per `decode` call when per-trial systems are singular or
    ill-conditioned; those trials' outputs are `NaN` and the message gives the
    count and the bins.
    """


class ProjectionWarning(MTDRWarning):
    """Some trials could not be projected.

    Emitted once per `project` call when, on some trials, the observed rows of
    the basis lack full column rank; those trials' coordinates are `NaN` and the
    message gives the count.
    """
