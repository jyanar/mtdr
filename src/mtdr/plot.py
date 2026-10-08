r"""Plotting helpers (optional `plot` extra, matplotlib).

`from mtdr import plot` always works; matplotlib is imported when a function
is first called, and its absence then raises `ImportError` with the install
hint `pip install 'mtdr[plot]'`. Every function returns the `Axes` (or the
array of `Axes`) it drew on and never calls `plt.show()`.

Axes handling: a one-panel function takes `ax` (an `Axes`, or `None` for a new
figure); a several-panel function takes `axes`: `None` (a new figure, one row
of panels), an array of `Axes` with at least as many entries as panels (used
in order, extras untouched), or a single `Axes` when exactly one panel is
drawn. `time` is an optional `(n_bins,)` array of bin centres for the x-axis.

The quantities drawn are the fitted objects of the model (`docs/model.md`): the
temporal bases $S_p$ and weights $W_p$ of (M1), the coefficient norms
$\lVert B_p[:,t]\rVert$ of (M37), the projections (M49)-(M49a) of condition
averages, and the AIC trace of the greedy search (M39).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mtdr._args import as_int, as_list
from mtdr.data import condition_average
from mtdr.errors import ParameterError
from mtdr.rank_search import RankSearchHistory

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from mtdr.model import MTDR
    from mtdr.simulation import SimulatedData

__all__ = [
    "bases",
    "coefficient_norms",
    "rank_search",
    "recovery",
    "trajectories",
    "weights",
]

AxesArray = NDArray[np.object_]
#: Most levels of the colouring regressor `trajectories` keys by a legend; more
#: get a colourbar.
_MAX_LEGEND_LEVELS = 8


def _pyplot() -> Any:
    """Import `matplotlib.pyplot`, or raise `ImportError` with the install hint."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "mtdr.plot needs matplotlib: pip install 'mtdr[plot]' (from a clone "
            "of the repository: pip install '.[plot]' in it, or pip install "
            "matplotlib)"
        ) from err
    return plt


def _axes(axes: object, n_panels: int) -> AxesArray:
    """Return the `n_panels` axes to draw on (module docstring's convention)."""
    plt = _pyplot()
    from matplotlib.axes import Axes as AxesClass

    if axes is None:
        _, grid = plt.subplots(
            1,
            n_panels,
            figsize=(4.0 * n_panels, 3.2),
            squeeze=False,
            layout="constrained",
        )
        out: AxesArray = np.asarray(grid[0], dtype=object)
        return out
    if isinstance(axes, AxesClass):
        if n_panels != 1:
            raise ParameterError(
                f"a single Axes was given but {n_panels} panels are drawn; pass an "
                "array of Axes or None"
            )
        return np.array([axes], dtype=object)
    arr = np.asarray(axes, dtype=object).ravel()
    if arr.size < n_panels or not all(isinstance(a, AxesClass) for a in arr):
        raise ParameterError(
            f"axes must be None, an Axes, or an array of at least {n_panels} Axes"
        )
    return arr[:n_panels]


def _ax(ax: object) -> Axes:
    return _axes(ax, 1)[0]  # type: ignore[no-any-return]


def _time(time: ArrayLike | None, n_bins: int) -> NDArray[np.float64]:
    if time is None:
        return np.arange(n_bins, dtype=np.float64)
    arr = np.asarray(time, dtype=np.float64)
    if arr.shape != (n_bins,):
        raise ParameterError(
            f"time must have shape (n_bins,) = ({n_bins},); got {arr.shape}"
        )
    return arr


def _regressors(model: MTDR, regressors: object, positive: bool) -> list[str]:
    model._check_fitted()
    names = list(model.regressor_names_)
    if regressors is None:
        if not positive:
            return names
        live = [n for n in names if model.ranks_[n] > 0]
        if not live:
            raise ParameterError(
                "no regressor has positive rank: there are no bases or weights to draw"
            )
        return live
    entries = [regressors] if isinstance(regressors, str) else as_list(regressors)
    if not entries:
        raise ParameterError(
            f"regressors must be a name or a non-empty sequence of names; got "
            f"{regressors!r}"
        )
    for name in entries:
        if name not in names:
            raise ParameterError(f"unknown regressor {name!r}; names {names}")
    return [str(n) for n in entries]


# =========================================================================== bases


def bases(
    model: MTDR,
    regressors: Sequence[str] | None = None,
    time: ArrayLike | None = None,
    axes: object = None,
) -> AxesArray:
    r"""One panel per regressor, one line per component of `S_[p]`.

    The temporal bases $S_p$ of the model (M1), $B_p=W_pS_p^\top$. Answers
    "when in the trial does this regressor's code live". In the canonical
    frame column $j$ of $S_p$ is $\sigma_jv_j$ (its norm the singular value
    of $B_p$) and is labelled by $\sigma_j$ and its share of
    $\lVert B_p\rVert_F^2$; in the raw frame by
    $\lVert W_p[:,j]\rVert\,\lVert S_p[:,j]\rVert$.

    Parameters
    ----------
    model : MTDR
        A fitted model.
    regressors : sequence of str, optional
        Which; default every regressor of positive rank.
    time : array_like, optional
        `(n_bins,)` bin centres.
    axes : Axes, array of Axes or None
        Where to draw.

    Returns
    -------
    numpy.ndarray
        The `Axes`, one per regressor.

    Examples
    --------
    >>> from mtdr import MTDR, plot, simulate
    >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=100, ranks=[2, 1], seed=0)
    >>> model = MTDR(ranks=[2, 1], estimator="svd").fit(sim.Y, sim.X)
    >>> plot.bases(model).shape
    (2,)
    """
    names = _regressors(model, regressors, positive=True)
    t = _time(time, model.n_bins_)
    out = _axes(axes, len(names))
    canonical = bool(model._canonical_frame)
    for ax, name in zip(out, names, strict=True):
        S = model.S_[name]
        W = model.W_[name]
        total = float(np.sum(model.B_[name] ** 2))
        for j in range(S.shape[1]):
            if canonical:
                sigma = float(np.linalg.norm(S[:, j]))
                share = sigma**2 / total if total > 0 else 0.0
                label = f"$\\sigma_{j + 1}$ = {sigma:.3g} ({share:.0%})"
            else:
                scale = float(np.linalg.norm(W[:, j]) * np.linalg.norm(S[:, j]))
                label = f"comp. {j + 1}: |W||S| = {scale:.3g}"
            ax.plot(t, S[:, j], marker="o", ms=3, label=label)
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("time" if time is not None else "time bin")
        ax.set_ylabel("$S_p$")
        if S.shape[1]:
            ax.legend(fontsize="small")
    return out


def weights(
    model: MTDR,
    regressors: Sequence[str] | None = None,
    kind: Literal["hist", "bar"] = "hist",
    component: int = 0,
    axes: object = None,
) -> AxesArray:
    """Per regressor, the distribution of `W_[p][:, component]` over neurons.

    A histogram (`kind="hist"`) or a bar per neuron (`"bar"`, the reference
    demo's view; useful below about 200 neurons). Answers "is the code carried
    by a few neurons or by many" ($W_p$ of (M1)).

    Parameters
    ----------
    model : MTDR
        A fitted model.
    regressors : sequence of str, optional
        Which; default every regressor of positive rank.
    kind : {"hist", "bar"}
        The view.
    component : int
        Which column of `W_[p]`; must exist for every regressor drawn.
    axes : Axes, array of Axes or None
        Where to draw.

    Returns
    -------
    numpy.ndarray
        The `Axes`.

    Examples
    --------
    >>> from mtdr import MTDR, plot, simulate
    >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=100, ranks=[2, 1], seed=0)
    >>> model = MTDR(ranks=[2, 1], estimator="svd").fit(sim.Y, sim.X)
    >>> plot.weights(model, kind="bar", component=0).shape
    (2,)
    """
    if kind not in ("hist", "bar"):
        raise ParameterError(f"kind must be 'hist' or 'bar'; got {kind!r}")
    names = _regressors(model, regressors, positive=True)
    j = as_int(component)
    if j is None or j < 0:
        raise ParameterError(f"component must be a non-negative int; got {component!r}")
    for name in names:
        if j >= model.ranks_[name]:
            raise ParameterError(
                f"component {j} does not exist for {name!r} (rank {model.ranks_[name]})"
            )
    out = _axes(axes, len(names))
    for ax, name in zip(out, names, strict=True):
        values = model.W_[name][:, j]
        if kind == "hist":
            ax.hist(values, bins="auto", color="C0")
            ax.set_xlabel(f"W[{name}][:, {j}]")
            ax.set_ylabel("neurons")
        else:
            ax.bar(np.arange(values.size), values, color="C0")
            ax.set_xlabel("neuron")
            ax.set_ylabel(f"W[{name}][:, {j}]")
        ax.set_title(name)
    return out


# =========================================================================== search


def rank_search(history: RankSearchHistory | MTDR, axes: object = None) -> AxesArray:
    """Plot the AIC trace of a greedy rank search, every candidate included (M39).

    AIC against search step for the accepted path (the accepted regressor
    annotated at each step), and every candidate tried from a step as a marker
    at the next step, coloured by the regressor it raised (legend), including
    the terminal rejected round. The AIC axis has no offset and the steps are
    integers. A search seeded by an SVD stage (`rank_search_init="svd_weighted"`
    or `"svd"`) gets a second panel, the SVD stage first, each on its own
    y-axis: the two stages' scores are on different scales. Answers "how
    decisive was the rank choice"; a flat tail means near ties
    (`history.near_ties()`).

    Parameters
    ----------
    history : RankSearchHistory or MTDR
        A history, or a model fitted with `ranks="aic"`.
    axes : Axes, array of Axes or None
        One or two panels.

    Returns
    -------
    numpy.ndarray
        The `Axes`, one per stage.

    Raises
    ------
    ParameterError
        For a model fitted with fixed ranks, or another object.

    Examples
    --------
    >>> from mtdr import MTDR, plot, simulate
    >>> sim = simulate(n_neurons=30, n_bins=6, n_trials=150, ranks=[2, 1], seed=0)
    >>> model = MTDR(ranks="aic", estimator="svd").fit(sim.Y, sim.X)
    >>> plot.rank_search(model).shape
    (1,)
    """
    if not isinstance(history, RankSearchHistory):
        from mtdr.model import MTDR

        if not isinstance(history, MTDR):
            raise ParameterError(
                f"history must be a RankSearchHistory or a fitted MTDR; got "
                f"{type(history).__name__}"
            )
        history._check_fitted()
        found = history.rank_search_history_
        if found is None:
            raise ParameterError(
                "the model was fitted with fixed ranks: there is no rank search to plot"
            )
        history = found
    _pyplot()
    from matplotlib.ticker import MaxNLocator

    stages = [history] if history.svd_stage is None else [history.svd_stage, history]
    out = _axes(axes, len(stages))
    for ax, stage in zip(out, stages, strict=True):
        steps = np.arange(stage.aic.size)
        # Every candidate, coloured and labelled by the regressor it raised.
        for j, name in enumerate(stage.regressor_names):
            tried = [
                (k + 1, round_[name])
                for k, round_ in enumerate(stage.candidates)
                if name in round_
            ]
            if tried:
                at, values = zip(*tried, strict=True)
                ax.plot(at, values, "o", color=f"C{j + 1}", alpha=0.5, label=f"+{name}")
        ax.plot(steps, stage.aic, "o-", color="C0")
        for k, name in enumerate(stage.accepted):
            ax.annotate(
                f"+{name}",
                (k + 1, stage.aic[k + 1]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize="small",
            )
        ax.set_xlabel("accepted steps")
        ax.set_ylabel("AIC" if stage.estimator != "custom" else "objective")
        ax.set_title(f"{stage.estimator} stage: {stage.stop_reason}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        if any(stage.candidates):
            ax.legend(title="candidate", fontsize="small")
    return out


def coefficient_norms(
    model: MTDR,
    regressors: Sequence[str] | None = None,
    time: ArrayLike | None = None,
    ax: object = None,
) -> Axes:
    r"""$\lVert B_p[:,t]\rVert$ over time bins, one line per regressor.

    The population-level strength of each regressor's coefficient matrix
    (M37) over the trial; frame-independent.

    Parameters
    ----------
    model : MTDR
        A fitted model.
    regressors : sequence of str, optional
        Which; default every regressor.
    time : array_like, optional
        `(n_bins,)` bin centres.
    ax : Axes or None
        Where to draw.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.

    Examples
    --------
    >>> from mtdr import MTDR, plot, simulate
    >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=100, ranks=[2, 1], seed=0)
    >>> model = MTDR(ranks=[2, 1], estimator="svd").fit(sim.Y, sim.X)
    >>> len(plot.coefficient_norms(model).lines)
    2
    """
    names = _regressors(model, regressors, positive=False)
    t = _time(time, model.n_bins_)
    out = _ax(ax)
    for name in names:
        out.plot(
            t, np.linalg.norm(model.B_[name], axis=0), marker="o", ms=3, label=name
        )
    out.set_xlabel("time" if time is not None else "time bin")
    out.set_ylabel(r"$\|B_p[:, t]\|$")
    out.legend(fontsize="small")
    return out


# =========================================================================== project


def trajectories(
    model: MTDR,
    Y: ArrayLike,
    X: ArrayLike,
    regressor: str,
    mask: ArrayLike | None = None,
    by: Sequence[int] | Sequence[str] | None = None,
    components: tuple[int, int] | int | None = None,
    method: Literal["gls", "paper"] = "gls",
    weighted: bool = True,
    basis: Mapping[str, ArrayLike] | None = None,
    min_trials: int = 1,
    time: ArrayLike | None = None,
    cmap: str = "viridis",
    ax: object = None,
) -> Axes:
    """Condition-averaged trajectories in a regressor's subspace.

    Calls [`condition_average`][mtdr.data.condition_average]`(Y, X, mask,
    by=by, regressor_names=model.regressor_names_, min_trials=min_trials)`,
    then [`MTDR.project`][mtdr.model.MTDR.project] on the averages with their
    mask and the given `method`, `weighted` and `basis` ((M49)-(M49a)). One
    trajectory per condition in the plane of two components (or one component
    against time, when `components` is an int or the width is 1), coloured by
    the condition's value of `regressor`, with a marker at the first bin. The
    colours are keyed by a legend with one entry per level of `regressor` (not
    per condition, when `by` spans several regressors) at up to eight levels,
    and by a colourbar ticked at the levels beyond. Conditions whose
    projection is all `NaN` are skipped and counted in the title.

    Parameters
    ----------
    model : MTDR
        A fitted model.
    Y : array_like
        `(n_trials, n_neurons, n_bins)` responses to average, `NaN` allowed
        (as `fit`).
    X : array_like
        `(n_trials, n_regressors)` design that defines the conditions.
    mask : array_like, optional
        `(n_trials, n_neurons)`; `None` infers it from `NaN`, as `fit`.
    regressor : str
        The subspace to project onto, and the colouring variable.
    by : sequence of int or str, optional
        Condition columns; default every column.
    components : (int, int), int or None
        `None` is `(0, 1)` at width >= 2, else `0` against time.
    method : {"gls", "paper"}
        The projection, as [`MTDR.project`][mtdr.model.MTDR.project].
    weighted : bool
        Weight neurons by their noise precision, as `project`.
    basis : mapping of str to array_like, optional
        `(n_neurons, r)` bases replacing `W_[name]`, as `project` (e.g. from
        [`orthogonalize`][mtdr.model.MTDR.orthogonalize]).
    min_trials : int
        As `condition_average`.
    time : array_like, optional
        `(n_bins,)` bin centres, for the against-time view.
    cmap : str
        Matplotlib colormap name.
    ax : Axes or None
        Where to draw.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.

    Raises
    ------
    ParameterError
        For components beyond the projected width, or as `project`.

    Examples
    --------
    >>> from mtdr import MTDR, plot, simulate
    >>> sim = simulate(n_neurons=30, n_bins=6, n_trials=200, ranks=[2, 1], seed=0)
    >>> model = MTDR(ranks=[2, 1], estimator="svd").fit(sim.Y, sim.X)
    >>> ax = plot.trajectories(model, sim.Y, sim.X, "x0", by=["x0"])
    >>> len(ax.lines)   # five levels of x0, a line and a start marker each
    10
    """
    plt = _pyplot()
    model._check_fitted()
    avg = condition_average(
        Y, X, mask, by=by, regressor_names=model.regressor_names_, min_trials=min_trials
    )
    z = model.project(
        avg.Y, regressor, mask=avg.mask, method=method, weighted=weighted, basis=basis
    )
    assert isinstance(z, np.ndarray)
    width = z.shape[2]
    if components is None:
        comps: tuple[int, ...] = (0, 1) if width >= 2 else (0,)
    elif isinstance(components, tuple):
        pair = [as_int(c) for c in components]
        if len(pair) != 2 or None in pair:  # never truncate 0.5 to 0
            raise ParameterError(
                f"components must be an int or a pair of ints; got {components!r}"
            )
        comps = tuple(c for c in pair if c is not None)
    else:
        single = as_int(components)
        if single is None:
            raise ParameterError(
                f"components must be an int or a pair; got {components!r}"
            )
        comps = (single,)
    if width == 0 or any(not 0 <= c < width for c in comps):
        raise ParameterError(
            f"components {comps} are beyond the projected width {width} of "
            f"{regressor!r}"
        )
    t = _time(time, model.n_bins_)
    column = list(model.regressor_names_).index(regressor)
    values = avg.X[:, column]
    span = float(values.max() - values.min())
    norm = (values - values.min()) / span if span > 0 else np.zeros_like(values)
    colours = plt.get_cmap(cmap)(norm)
    out = _ax(ax)
    drawn = ~np.isnan(z).all(axis=(1, 2))
    skipped = int(np.sum(~drawn))
    levels = np.unique(values[drawn])
    first: dict[float, Any] = {}  # level -> its first line, for the legend
    for c in np.flatnonzero(drawn):
        path = z[c]
        if len(comps) == 2:
            xs, ys = path[:, comps[0]], path[:, comps[1]]
        else:
            xs, ys = t, path[:, comps[0]]
        (line,) = out.plot(xs, ys, color=colours[c])
        first.setdefault(float(values[c]), line)
        out.plot(xs[:1], ys[:1], "o", color=colours[c])
    if 0 < levels.size <= _MAX_LEGEND_LEVELS:
        out.legend(
            [first[float(v)] for v in levels],
            [f"{regressor} = {v:.3g}" for v in levels],
            fontsize="small",
        )
    elif levels.size:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        mappable = ScalarMappable(
            norm=Normalize(float(values.min()), float(values.max())),
            cmap=plt.get_cmap(cmap),
        )
        bar = out.figure.colorbar(mappable, ax=out, ticks=levels)
        bar.set_label(regressor)
    if len(comps) == 2:
        out.set_xlabel(f"component {comps[0] + 1}")
        out.set_ylabel(f"component {comps[1] + 1}")
    else:
        out.set_xlabel("time" if time is not None else "time bin")
        out.set_ylabel(f"component {comps[0] + 1}")
    title = f"{regressor} subspace ({method})"
    if skipped:
        title += f"; {skipped} conditions skipped"
    out.set_title(title)
    return out


def recovery(model: MTDR, sim: SimulatedData, axes: object = None) -> AxesArray:
    r"""Estimated against true `B_p`, intercept and noise precision.

    One scatter per regressor's coefficient matrix $B_p=W_pS_p^\top$ of (M1)
    (frame-invariant, so no alignment is needed), one for the intercept
    when both the model and the simulation have one, and one for the noise
    precisions on log-log axes, each with the identity line and the
    correlation over all entries in its
    title ("r undefined" when either side is constant, e.g. a scalar
    `noise_precision`): the reference demo's final figures.

    Parameters
    ----------
    model : MTDR
        A model fitted to `sim` (same neurons, regressors in the same order).
    sim : SimulatedData
        The ground truth.
    axes : array of Axes or None
        `n_regressors + 2` panels (`+ 1` without an intercept).

    Returns
    -------
    numpy.ndarray
        The `Axes`.

    Raises
    ------
    ParameterError
        If the model and the simulation do not match.

    Examples
    --------
    >>> from mtdr import MTDR, plot, simulate
    >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=100, ranks=[2, 1], seed=0)
    >>> model = MTDR(ranks=[2, 1], estimator="svd").fit(sim.Y, sim.X)
    >>> plot.recovery(model, sim).shape
    (4,)
    """
    model._check_fitted()
    names = list(model.regressor_names_)
    shape = (model.n_neurons_, model.n_bins_)
    if (
        len(sim.regressor_names) != len(names)
        or sim.noise_precision.shape != (model.n_neurons_,)
        or any(np.shape(sim.B[n]) != shape for n in sim.regressor_names)
    ):
        raise ParameterError(
            "the model was not fitted to this simulation's design: it has "
            f"{model.n_neurons_} neurons, {model.n_bins_} bins and {len(names)} "
            f"regressors, the simulation {sim.noise_precision.shape[0]}, "
            f"{sim.Y.shape[2]} and {len(sim.regressor_names)}"
        )
    pairs: list[tuple[str, NDArray[np.float64], NDArray[np.float64], bool]] = [
        (name, model.B_[name], sim.B[true], False)
        for name, true in zip(names, sim.regressor_names, strict=True)
    ]
    if model.intercept_ is not None and sim.intercept is not None:
        pairs.append(("intercept", model.intercept_, sim.intercept, False))
    pairs.append(("noise precision", model.noise_precision_, sim.noise_precision, True))
    out = _axes(axes, len(pairs))
    for ax, (label, est, true, log) in zip(out, pairs, strict=True):
        e, t = np.ravel(est), np.ravel(true)
        ax.plot(t, e, ".", ms=3, alpha=0.5)
        lo, hi = float(min(t.min(), e.min())), float(max(t.max(), e.max()))
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=0.8)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        # A constant side has no correlation; its float64 std is not 0.
        if np.ptp(t) > 0 and np.ptp(e) > 0:
            ax.set_title(f"{label}: r = {float(np.corrcoef(t, e)[0, 1]):.3f}")
        else:
            ax.set_title(f"{label}: r undefined")
        ax.set_xlabel("true")
        ax.set_ylabel("estimated")
    return out
