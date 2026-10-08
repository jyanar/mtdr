r"""Per-neuron sufficient statistics under the observation mask.

Every estimator and likelihood in `mtdr` reads the data only through the
statistics (M6), (M10)-(M12): for neuron $i$, with
$\mathcal K_i$ the trials on which it was observed and $n_i=|\mathcal K_i|$,
the means

$$
\bar x_i=\frac1{n_i}\sum_{k\in\mathcal K_i}x_k,\qquad
\bar y_i=\frac1{n_i}\sum_{k\in\mathcal K_i}Y_{ki\cdot},
$$

and the moments about them,

$$
\tilde A_i=\sum_{k\in\mathcal K_i}(x_k-\bar x_i)(x_k-\bar x_i)^\top,\quad
\tilde\xi_i=\sum_{k\in\mathcal K_i}(x_k-\bar x_i)(Y_{ki\cdot}-\bar y_i)^\top,\quad
\tilde\upsilon_i=\sum_{k\in\mathcal K_i}\lVert Y_{ki\cdot}-\bar y_i\rVert^2 .
$$

The raw moments of (M6), (M10)-(M11) follow from them without cancellation,
$A_i=\tilde A_i+n_i\bar x_i\bar x_i^\top$, $\xi_i=\tilde\xi_i+n_i\bar x_i\bar y_i^\top$
and $\upsilon_i=\tilde\upsilon_i+n_i\lVert\bar y_i\rVert^2$, and are kept as
read-only views, `XtX`, `XtY_raw` and `YtY_raw`.

[`sufficient_statistics`][mtdr.stats.sufficient_statistics] computes them in
one pass over `Y`; [`SufficientStats.centered`][mtdr.stats.SufficientStats.centered]
derives the intercept-centred statistics (M13)-(M14) from them without touching
`Y` again. The reference builds the raw quantities neuron by neuron
(`MkSuffStatsBTDR_IncompObs_uneqvar_S_fast`, `MkSuffStats_BilinReg_Sims`,
`ECMEsuffstat`, the inline `xbari`/`Ybar` loop of `mTDRdemo.m`); the port
computes them batched over neurons with the mask applied explicitly, so values
of `Y` at unobserved entries never enter (the reference's sums over all trials
are right only because it zeroes those entries first).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mtdr._frozen import restore_frozen
from mtdr.errors import ParameterError, ValidationError

__all__ = ["SufficientStats", "sufficient_statistics"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: How many offending (trial, neuron) pairs an error message lists.
_MAX_LISTED = 10


@dataclass(frozen=True, eq=False)
class SufficientStats:
    r"""Per-neuron sufficient statistics, (M6), (M10)-(M12).

    Returned by [`sufficient_statistics`][mtdr.stats.sufficient_statistics].
    The stored fields are the moments about each neuron's means and the means
    themselves; the raw moments (`XtX`, `XtY_raw`, `YtY_raw`) are derived from
    them once, on construction, and exposed as read-only properties. The
    intercept-centred views that the likelihoods need come from
    [`centered`][mtdr.stats.SufficientStats.centered]. Arrays are `float64`
    (`n_obs` is `int64`) and read-only, also after unpickling; the constructor
    copies and checks them (shapes; finite values except the `NaN` means of a
    neuron never observed; `XtX_c` symmetric to $10^{-12}$ of its largest
    entry, then symmetrised exactly; zero moments for a neuron with
    `n_obs == 0`), so a `SufficientStats` built by hand obeys the same
    contract. Statistics accumulated elsewhere as raw sums convert by
    $\tilde A_i=A_i-n_i\bar x_i\bar x_i^\top$,
    $\tilde\xi_i=\xi_i-n_i\bar x_i\bar y_i^\top$ and
    $\tilde\upsilon_i=\upsilon_i-n_i\lVert\bar y_i\rVert^2$, which carries the
    cancellation that the centred accumulation avoids.

    Attributes
    ----------
    XtX_c : numpy.ndarray
        `(n_neurons, n_regressors, n_regressors)`, $\tilde A_i$, the Gram matrix
        of the neuron's observed regressors about their mean $\bar x_i$.
    XtY_c : numpy.ndarray
        `(n_neurons, n_regressors, n_bins)`, $\tilde\xi_i[p,t]=\sum_{k\in\mathcal K_i}
        (X_{kp}-\bar x_{ip})(Y_{kit}-\bar y_{it})$.
    YtY_c : numpy.ndarray
        `(n_neurons,)`, $\tilde\upsilon_i=\sum_{k\in\mathcal K_i}
        \lVert Y_{ki\cdot}-\bar y_i\rVert^2$, the energy about the mean.
    n_obs : numpy.ndarray
        `(n_neurons,)` int, the number of observed trials $n_i$, (M11).
    X_mean : numpy.ndarray
        `(n_neurons, n_regressors)`, the mean regressor vector over the
        neuron's observed trials, (M12); `NaN` for a neuron never observed.
    Y_mean : numpy.ndarray
        `(n_neurons, n_bins)`, the mean response over the neuron's observed
        trials, (M12); `NaN` for a neuron never observed.
    n_regressors : int
        Number of columns of `X` (the intercept is not one).
    n_bins : int
        Number of time bins.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import simulate
    >>> from mtdr.stats import sufficient_statistics
    >>> sim = simulate(n_neurons=4, n_bins=3, n_trials=12, ranks=[1, 1],
    ...                drop_prob=0.2, seed=0)
    >>> stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    >>> stats
    SufficientStats(n_neurons=4, n_regressors=2, n_bins=3, n_obs=[...])
    >>> stats.XtX.shape, stats.XtY_raw.shape, stats.Y_mean.shape
    ((4, 2, 2), (4, 2, 3), (4, 3))
    >>> raw = stats.XtX_c + stats.n_obs[:, None, None] * (
    ...     stats.X_mean[:, :, None] * stats.X_mean[:, None, :])
    >>> bool(np.allclose(raw, stats.XtX))
    True
    """

    XtX_c: FloatArray
    XtY_c: FloatArray
    YtY_c: FloatArray
    n_obs: IntArray
    X_mean: FloatArray
    Y_mean: FloatArray
    n_regressors: int
    n_bins: int

    def __post_init__(self) -> None:
        """Check shapes, convert to `float64`/`int64` copies, derive and freeze."""
        P = _positive_count("n_regressors", self.n_regressors)
        T = _positive_count("n_bins", self.n_bins)
        n_obs = np.asarray(self.n_obs)
        if n_obs.ndim != 1 or n_obs.size == 0:
            raise ValidationError(
                f"n_obs must be a non-empty 1-D array; got shape {n_obs.shape}"
            )
        if n_obs.dtype.kind not in "iu" or (n_obs < 0).any():
            raise ValidationError("n_obs must hold non-negative integers")
        n = n_obs.size
        expected = {
            "XtX_c": (n, P, P),
            "XtY_c": (n, P, T),
            "YtY_c": (n,),
            "X_mean": (n, P),
            "Y_mean": (n, T),
        }
        values: dict[str, FloatArray] = {}
        for name, shape in expected.items():
            try:
                value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            except (TypeError, ValueError) as err:
                raise ValidationError(f"{name} is not a real array: {err}") from err
            if value.shape != shape:
                raise ValidationError(
                    f"{name} must have shape {shape} for {n} neurons, "
                    f"{P} regressors and {T} bins; got {value.shape}"
                )
            # Means are NaN, legitimately, for a neuron never observed.
            finite = np.isfinite(value)
            if name in ("X_mean", "Y_mean"):
                finite |= (n_obs == 0).reshape(-1, 1)
            if not finite.all():
                raise ValidationError(f"{name} must be finite")
            values[name] = value
        n_obs = n_obs.astype(np.int64, copy=True)
        A = values["XtX_c"]
        scale = np.abs(A).max(axis=(1, 2))
        asymmetric = np.abs(A - A.transpose(0, 2, 1)).max(axis=(1, 2)) > 1e-12 * scale
        if asymmetric.any():
            raise ValidationError(
                f"XtX_c must be symmetric; it is not for neurons "
                f"{_listed(np.flatnonzero(asymmetric))}"
            )
        values["XtX_c"] = 0.5 * (A + A.transpose(0, 2, 1))
        unobserved = n_obs == 0
        nonzero = unobserved & (
            (values["XtX_c"] != 0).any(axis=(1, 2))
            | (values["XtY_c"] != 0).any(axis=(1, 2))
            | (values["YtY_c"] != 0)
        )
        if nonzero.any():
            raise ValidationError(
                "a neuron with n_obs == 0 must have zero moments; neurons "
                f"{_listed(np.flatnonzero(nonzero))} do not"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            raw = _raw_moments(
                values["XtX_c"],
                values["XtY_c"],
                values["YtY_c"],
                n_obs,
                values["X_mean"],
                values["Y_mean"],
            )
        for name, value in zip(("XtX", "XtY_raw", "YtY_raw"), raw, strict=True):
            if not np.isfinite(value).all():
                raise ValidationError(
                    f"the raw moment {name} derived from the centred statistics "
                    "and the means overflows float64"
                )
            values[f"_{name}"] = value
        n_obs.setflags(write=False)
        object.__setattr__(self, "n_obs", n_obs)
        for name, value in values.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "n_regressors", P)
        object.__setattr__(self, "n_bins", T)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled object with its arrays read-only again."""
        restore_frozen(self, state)

    @property
    def XtX(self) -> FloatArray:  # noqa: N802 - the API name
        r"""The raw Gram, (M6): $A_i=\tilde A_i+n_i\bar x_i\bar x_i^\top$.

        `(n_neurons, n_regressors, n_regressors)`, read-only, derived once.
        """
        out: FloatArray = self.__dict__["_XtX"]
        return out

    @property
    def XtY_raw(self) -> FloatArray:  # noqa: N802 - the API name
        r"""The raw cross-moment, (M10): $\xi_i=\tilde\xi_i+n_i\bar x_i\bar y_i^\top$.

        `(n_neurons, n_regressors, n_bins)`, read-only, derived once.
        """
        out: FloatArray = self.__dict__["_XtY_raw"]
        return out

    @property
    def YtY_raw(self) -> FloatArray:  # noqa: N802 - the API name
        r"""The raw energy, (M11): $\upsilon_i=\lVert Y_i\rVert^2$.

        `(n_neurons,)`, $\tilde\upsilon_i+n_i\lVert\bar y_i\rVert^2$, read-only,
        derived once.
        """
        out: FloatArray = self.__dict__["_YtY_raw"]
        return out

    @property
    def n_neurons(self) -> int:
        """Number of neurons, `n_obs.size`."""
        return int(self.n_obs.size)

    def centered(self, intercept: ArrayLike | None) -> tuple[FloatArray, FloatArray]:
        r"""Statistics of the responses with an intercept subtracted, (M13)-(M14).

        With $\tilde Y_{ki\cdot}=Y_{ki\cdot}-b_i$ on the observed trials,

        $$
        \xi_i(b)=\tilde\xi_i+n_i\,\bar x_i(\bar y_i-b_i)^\top,\qquad
        \upsilon_i(b)=\tilde\upsilon_i+n_i\lVert\bar y_i-b_i\rVert^2,
        $$

        which are (M13)-(M14) written about the means, in
        $O(n\,P\,T)$ and without the data. They replace the reference's
        `ECMEsuffstat`, which re-reads every response. A neuron never observed
        contributes zeros.

        Parameters
        ----------
        intercept : array_like of shape (n_neurons, n_bins), or None
            The condition-independent term $b$ to subtract. `None` returns the
            raw statistics themselves (read-only).

        Returns
        -------
        XtY : numpy.ndarray
            `(n_neurons, n_regressors, n_bins)`, $\xi_i(b)$.
        YtY : numpy.ndarray
            `(n_neurons,)`, $\upsilon_i(b)$.

        Raises
        ------
        ParameterError
            If `intercept` is not a finite real array of shape
            `(n_neurons, n_bins)`.

        Notes
        -----
        $\upsilon_i(b)$ is a sum of non-negative terms, so there is no
        cancellation: a baseline far from zero costs no accuracy (the closed
        form (M14) on the raw moments loses about
        $\log_{10}(\upsilon_i/\upsilon_i(b))$ digits).

        Examples
        --------
        >>> import numpy as np
        >>> from mtdr.stats import sufficient_statistics
        >>> rng = np.random.default_rng(0)
        >>> Y = rng.normal(size=(8, 3, 4)); X = rng.normal(size=(8, 2))
        >>> mask = np.ones((8, 3), dtype=bool)
        >>> b = rng.normal(size=(3, 4))
        >>> XtY, YtY = sufficient_statistics(Y, X, mask).centered(b)
        >>> direct = sufficient_statistics(Y - b, X, mask)
        >>> bool(np.allclose(XtY, direct.XtY_raw) and np.allclose(YtY, direct.YtY_raw))
        True
        """
        if intercept is None:
            return self.XtY_raw, self.YtY_raw
        b = _intercept_array(intercept, (self.n_neurons, self.n_bins))
        observed = (self.n_obs > 0)[:, None]
        n_obs = self.n_obs.astype(np.float64)
        x_mean = np.where(observed, self.X_mean, 0.0)
        gap = np.where(observed, self.Y_mean, 0.0) - b
        XtY = self.XtY_c + n_obs[:, None, None] * (x_mean[:, :, None] * gap[:, None, :])
        YtY = self.YtY_c + n_obs * np.einsum("it,it->i", gap, gap)
        return XtY, YtY

    def __repr__(self) -> str:
        """Summarise the sizes instead of printing the arrays."""
        return (
            f"SufficientStats(n_neurons={self.n_neurons}, "
            f"n_regressors={self.n_regressors}, n_bins={self.n_bins}, "
            f"n_obs={np.array2string(self.n_obs, threshold=8)})"
        )


def sufficient_statistics(
    Y: ArrayLike, X: ArrayLike, mask: ArrayLike
) -> SufficientStats:
    r"""Per-neuron sufficient statistics of `Y` given `X` under `mask`.

    Computes, for every neuron $i$ over the trials $\mathcal K_i$ on which
    `mask[:, i]` is `True`, the trial count $n_i$ (M11), the means $\bar x_i$,
    $\bar y_i$ (M12), and the moments about the means $\tilde A_i$,
    $\tilde\xi_i$, $\tilde\upsilon_i$, from which the raw Gram $A_i$ (M6),
    cross-moment $\xi_i$ (M10) and energy $\upsilon_i$ (M11) follow. The
    reference's `MkSuffStatsBTDR_IncompObs_uneqvar_S_fast` plus the
    `xbari`/`Ybar` loop of `mTDRdemo.m`; the full-design statistics of
    `MkSuffStats_BilinReg_Sims` (with the constant column) are the same
    moments with the intercept appended.

    Parameters
    ----------
    Y : array_like of shape (n_trials, n_neurons, n_bins)
        Responses `Y[k, i, t]`. Real numeric (bool and integer counts are
        converted to `float64`). Entries where `mask` is `False` are ignored
        entirely and may hold anything, including `NaN`.
    X : array_like of shape (n_trials, n_regressors)
        Design `X[k, p]`, finite, at least one column, no column of ones (the
        intercept is a separate term).
    mask : array_like of shape (n_trials, n_neurons)
        `True` (or `1`) where neuron `i` was observed on trial `k`; boolean or
        exactly 0/1. Required: this function does not infer the mask from
        `NaN` (that is `validate_inputs`).

    Returns
    -------
    SufficientStats
        The statistics. A neuron that is never observed is kept, with zero
        statistics and `NaN` means (evaluation contexts allow it; `fit`
        rejects it).

    Raises
    ------
    ValidationError
        For a wrong shape or layout (the message states the expected
        `(n_trials, n_neurons, n_bins)` layout), an empty axis, a non-real
        dtype, a mask that is not boolean or 0/1, a non-finite entry of `X`, a
        `NaN`/`Inf` in `Y` where `mask` is `True` (the message lists the
        trial-neuron pairs), or finite values of `X` or `Y` too large to
        square in `float64` (above about $10^{154}$).

    Notes
    -----
    The moments are accumulated about the means: each neuron's responses and
    regressors are first shifted by their values on the neuron's first
    observed trial, then by the mean of the shifted values, so a regressor or
    response that is constant over the neuron's trials centres to exactly
    zero, and a large baseline or offset costs no accuracy.

    Cost: one pass over `Y`, $O(N\,n\,(P\,T+P^2))$ in batched matrix products,
    and one neuron-major working copy of `Y`. Never-observed or degenerate
    neurons, the `min_observations` floor, zero-variance neurons and the design
    checks are fit-time checks of `validate_inputs` and are not applied here.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr.stats import sufficient_statistics
    >>> Y = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)   # 2 trials, 2 neurons
    >>> X = np.array([[1.0], [-1.0]])
    >>> mask = np.array([[True, True], [True, False]])
    >>> Y[1, 1] = np.nan                       # unobserved: ignored
    >>> stats = sufficient_statistics(Y, X, mask)
    >>> stats.n_obs
    array([2, 1])
    >>> stats.XtY_raw[:, 0]                    # sum_k x_k * Y[k, i, :]
    array([[-6., -6., -6.],
           [ 3.,  4.,  5.]])
    >>> stats.Y_mean
    array([[3., 4., 5.],
           [3., 4., 5.]])
    >>> stats.YtY_c                            # about the mean: 2 * 3 * 3**2
    array([54.,  0.])
    """
    Y_arr = _real_array("Y", Y, 3, "(n_trials, n_neurons, n_bins)")
    X_arr = _real_array("X", X, 2, "(n_trials, n_regressors)")
    n_trials, n_neurons, n_bins = Y_arr.shape
    if X_arr.shape[0] != n_trials:
        raise ValidationError(
            f"Y has {n_trials} trials (axis 0) but X has {X_arr.shape[0]} rows; Y "
            "must be laid out (n_trials, n_neurons, n_bins) and X (n_trials, "
            "n_regressors)"
        )
    if not np.isfinite(X_arr).all():
        rows = np.flatnonzero(~np.isfinite(X_arr).all(axis=1))
        raise ValidationError(f"X must be finite; non-finite rows {_listed(rows)}")
    mask_arr = _mask_array(mask, (n_trials, n_neurons))
    bad = mask_arr & ~np.isfinite(Y_arr).all(axis=2)
    if bad.any():
        pairs = [(int(k), int(i)) for k, i in np.argwhere(bad)]
        raise ValidationError(
            "Y has NaN or Inf where mask is True, at (trial, neuron) "
            f"{_listed(pairs)}; impute them or mark those neuron-trials unobserved"
        )

    n_obs = mask_arr.sum(axis=0).astype(np.int64)
    observed = n_obs > 0
    count = np.maximum(n_obs, 1).astype(np.float64)[:, None]
    # Neuron-major working layout (n, N, .), so the per-neuron products are
    # batched matrix products.
    m = mask_arr.T[:, :, None]
    first = np.argmax(mask_arr, axis=0)  # first observed trial; 0 if never observed
    with np.errstate(over="ignore", invalid="ignore"):
        # Shift by the first observed value, then by the mean of the shifted
        # values: a column constant over the neuron's trials becomes exactly 0.
        X_ref = np.where(observed[:, None], X_arr[first], 0.0)
        Y_ref = np.where(observed[:, None], Y_arr[first, np.arange(n_neurons)], 0.0)
        Xc = np.where(m, X_arr[None, :, :] - X_ref[:, None, :], 0.0)
        Yc = np.where(m, Y_arr.transpose(1, 0, 2) - Y_ref[:, None, :], 0.0)
        dx = Xc.sum(axis=1) / count
        dy = Yc.sum(axis=1) / count
        Xc -= dx[:, None, :]
        Yc -= dy[:, None, :]
        Xc *= m
        Yc *= m
        XtX_c = np.matmul(Xc.transpose(0, 2, 1), Xc)
        XtX_c = 0.5 * (XtX_c + XtX_c.transpose(0, 2, 1))  # exact symmetry
        XtY_c = np.matmul(Xc.transpose(0, 2, 1), Yc)
        YtY_c = np.einsum("ikt,ikt->i", Yc, Yc)
        X_mean = np.where(observed[:, None], X_ref + dx, np.nan)
        Y_mean = np.where(observed[:, None], Y_ref + dy, np.nan)
        XtX, XtY, YtY = _raw_moments(XtX_c, XtY_c, YtY_c, n_obs, X_mean, Y_mean)
    big_x = ~(np.isfinite(XtX_c).all(axis=(1, 2)) & np.isfinite(XtX).all(axis=(1, 2)))
    if big_x.any():
        raise ValidationError(
            "X is too large to square in float64 (|X| above about 1e154) on the "
            f"trials of neurons {_listed(np.flatnonzero(big_x))}; rescale X"
        )
    big_y = ~(
        np.isfinite(YtY_c)
        & np.isfinite(YtY)
        & np.isfinite(XtY_c).all(axis=(1, 2))
        & np.isfinite(XtY).all(axis=(1, 2))
    )
    if big_y.any():
        raise ValidationError(
            "Y is too large to square in float64 (|Y| above about 1e154) for "
            f"neurons {_listed(np.flatnonzero(big_y))}; rescale Y"
        )
    return SufficientStats(
        XtX_c=XtX_c,
        XtY_c=XtY_c,
        YtY_c=YtY_c,
        n_obs=n_obs,
        X_mean=X_mean,
        Y_mean=Y_mean,
        n_regressors=X_arr.shape[1],
        n_bins=n_bins,
    )


def _raw_moments(
    XtX_c: FloatArray,
    XtY_c: FloatArray,
    YtY_c: FloatArray,
    n_obs: IntArray,
    X_mean: FloatArray,
    Y_mean: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the raw moments (M6), (M10), (M11) from the centred ones and the means."""
    observed = (n_obs > 0)[:, None]
    count = n_obs.astype(np.float64)
    x = np.where(observed, X_mean, 0.0)
    y = np.where(observed, Y_mean, 0.0)
    XtX = XtX_c + count[:, None, None] * (x[:, :, None] * x[:, None, :])
    XtX = 0.5 * (XtX + XtX.transpose(0, 2, 1))  # exact symmetry
    XtY = XtY_c + count[:, None, None] * (x[:, :, None] * y[:, None, :])
    YtY = YtY_c + count * np.einsum("it,it->i", y, y)
    return XtX, XtY, YtY


# --------------------------------------------------------------------------- validation


def _real_array(name: str, value: object, ndim: int, layout: str) -> FloatArray:
    """`value` as a `float64` array with `ndim` non-empty axes, else raise."""
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as err:
        raise ValidationError(f"{name} is not a numeric array: {err}") from err
    if arr.dtype.kind not in "biuf":
        raise ValidationError(f"{name} must be real numeric; got dtype {arr.dtype}")
    if arr.ndim != ndim:
        raise ValidationError(
            f"{name} must be {ndim}-D {layout}; got shape {arr.shape}"
        )
    if 0 in arr.shape:
        raise ValidationError(
            f"{name} must have non-empty axes {layout}; got shape {arr.shape}"
        )
    return arr.astype(np.float64, copy=False)


def _mask_array(value: object, shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Return the observation mask as `bool`, checked for `shape` and 0/1 values."""
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as err:
        raise ValidationError(f"mask is not an array: {err}") from err
    if arr.shape != shape:
        raise ValidationError(
            f"mask must have shape (n_trials, n_neurons) = {shape}; got {arr.shape}"
        )
    if arr.dtype == np.bool_:
        return arr
    if arr.dtype.kind not in "iuf":
        raise ValidationError(f"mask must be boolean or 0/1; got dtype {arr.dtype}")
    if not np.isin(arr, (0, 1)).all():
        raise ValidationError("mask must be boolean or hold only 0 and 1")
    out: NDArray[np.bool_] = arr == 1
    return out


def _intercept_array(value: object, shape: tuple[int, int]) -> FloatArray:
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as err:
        raise ParameterError(f"intercept is not an array: {err}") from err
    if arr.dtype.kind not in "iuf" or arr.shape != shape:
        raise ParameterError(
            f"intercept must be a real array of shape (n_neurons, n_bins) = {shape}; "
            f"got dtype {arr.dtype}, shape {arr.shape}"
        )
    out = arr.astype(np.float64, copy=False)
    if not np.isfinite(out).all():
        raise ParameterError("intercept must be finite")
    return out


def _positive_count(name: str, value: Any) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValidationError(f"{name} must be a positive int; got {value!r}")
    if value < 1:
        raise ValidationError(f"{name} must be a positive int; got {value!r}")
    return int(value)


def _listed(items: Any) -> str:
    items = list(items)
    head = ", ".join(str(item) for item in items[:_MAX_LISTED])
    more = f" and {len(items) - _MAX_LISTED} more" if len(items) > _MAX_LISTED else ""
    return f"{head}{more}"
