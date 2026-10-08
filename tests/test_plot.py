"""`mtdr.plot`: what each helper draws, the axes conventions, the errors, and the
lazy matplotlib import."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import mtdr
from mtdr import MTDR, plot, simulate
from mtdr.errors import ParameterError

plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def _close() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def sim() -> mtdr.SimulatedData:
    return simulate(
        n_neurons=30,
        n_bins=6,
        n_trials=200,
        ranks={"sa": 2, "choice": 1},
        levels=[[-2, -1, 0, 1, 2], [-1, 1]],
        seed=0,
    )


@pytest.fixture(scope="module")
def model(sim: mtdr.SimulatedData) -> MTDR:
    return MTDR(ranks="aic", max_rank=3).fit(
        sim.Y, sim.X, regressor_names=sim.regressor_names
    )


@pytest.fixture(scope="module")
def fixed(sim: mtdr.SimulatedData) -> MTDR:
    return MTDR(ranks=sim.ranks, estimator="svd").fit(
        sim.Y, sim.X, regressor_names=sim.regressor_names
    )


def test_import_does_not_need_matplotlib(
    monkeypatch: pytest.MonkeyPatch, fixed: MTDR
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    reloaded = importlib.reload(plot)
    with pytest.raises(ImportError, match=r"pip install 'mtdr\[plot\]' \(from a clone"):
        reloaded.bases(fixed)
    monkeypatch.undo()
    importlib.reload(plot)


def test_bases(model: MTDR, sim: mtdr.SimulatedData) -> None:
    axes = plot.bases(model, time=np.linspace(0, 1, 6))
    assert axes.shape == (2,)
    sa = axes[0]
    assert sa.get_title() == "sa"
    assert len(sa.get_lines()) == 2 + 1  # two components and the zero line
    sigma = np.linalg.norm(model.S_["sa"][:, 0])
    labels = [t.get_text() for t in sa.get_legend().get_texts()]
    assert labels[0].startswith("$\\sigma_1$ = " + f"{sigma:.3g}")
    np.testing.assert_allclose(sa.get_lines()[0].get_ydata(), model.S_["sa"][:, 0])
    one = plot.bases(model, regressors="choice", axes=plt.subplots()[1])
    assert one.shape == (1,)


def test_bases_in_the_raw_frame(sim: mtdr.SimulatedData) -> None:
    raw = MTDR(ranks=sim.ranks, canonicalize=False, estimator="svd").fit(
        sim.Y, sim.X, regressor_names=sim.regressor_names
    )
    ax = plot.bases(raw, ["sa"])[0]
    assert ax.get_legend().get_texts()[0].get_text().startswith("comp. 1: |W||S|")


def test_weights(fixed: MTDR) -> None:
    hist = plot.weights(fixed)
    assert hist.shape == (2,)
    assert hist[0].get_ylabel() == "neurons"
    _, grid = plt.subplots(1, 3)
    bars = plot.weights(fixed, ["sa"], kind="bar", component=1, axes=grid)
    np.testing.assert_allclose(
        [p.get_height() for p in bars[0].patches], fixed.W_["sa"][:, 1]
    )
    assert not grid[1].has_data()  # extras untouched


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"kind": "violin"}, "kind"),
        ({"component": -1}, "component"),
        ({"component": 1}, "does not exist for 'choice'"),
        ({"regressors": ["zz"]}, "unknown regressor"),
        ({"regressors": []}, "non-empty"),
    ],
)
def test_weights_errors(fixed: MTDR, kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ParameterError, match=match):
        plot.weights(fixed, **kw)


def test_rank_search(model: MTDR) -> None:
    h = model.rank_search_history_
    assert h is not None
    assert h.svd_stage is not None
    axes = plot.rank_search(model)
    assert axes.shape == (2,)
    assert axes[0].get_title().startswith("svd stage")
    assert axes[1].get_title().startswith("mmle stage")
    accepted = axes[1].get_lines()[-1]
    np.testing.assert_allclose(accepted.get_ydata(), h.aic)
    assert len(axes[0].texts) == len(h.svd_stage.accepted)
    custom = mtdr.rank_search.greedy_aic(
        lambda r: None, lambda f, r: (r[0] - 2.0) ** 2, [1], 2
    )[1]
    single = plot.rank_search(custom, axes=plt.subplots()[1])
    assert single[0].get_ylabel() == "objective"
    assert single[0].get_title() == "custom stage: max_rank"


def test_rank_search_errors(fixed: MTDR) -> None:
    with pytest.raises(ParameterError, match="fixed ranks"):
        plot.rank_search(fixed)
    with pytest.raises(ParameterError, match="RankSearchHistory or a fitted MTDR"):
        plot.rank_search("history")  # type: ignore[arg-type]
    with pytest.raises(mtdr.NotFittedError):
        plot.rank_search(MTDR())


def test_axes_conventions(fixed: MTDR) -> None:
    _, ax = plt.subplots()
    with pytest.raises(ParameterError, match="single Axes"):
        plot.bases(fixed, axes=ax)
    with pytest.raises(ParameterError, match="at least 2 Axes"):
        plot.bases(fixed, axes=[ax])
    with pytest.raises(ParameterError, match="at least 2 Axes"):
        plot.bases(fixed, axes=[ax, "not an axes"])
    with pytest.raises(ParameterError, match="time must have shape"):
        plot.bases(fixed, time=np.arange(5))


def test_coefficient_norms(fixed: MTDR) -> None:
    ax = plot.coefficient_norms(fixed, time=np.arange(6) * 0.05)
    assert len(ax.get_lines()) == 2
    np.testing.assert_allclose(
        np.asarray(ax.get_lines()[0].get_ydata(), dtype=float),
        np.linalg.norm(fixed.B_["sa"], axis=0),
    )
    assert ax.get_xlabel() == "time"
    _, mine = plt.subplots()
    assert plot.coefficient_norms(fixed, ["choice"], ax=mine) is mine


def test_trajectories(fixed: MTDR, sim: mtdr.SimulatedData) -> None:
    ax = plot.trajectories(fixed, sim.Y, sim.X, "sa", by=["sa"])
    assert len(ax.get_lines()) == 10  # five levels, a line and a start marker each
    avg = mtdr.condition_average(
        sim.Y, sim.X, by=["sa"], regressor_names=("sa", "choice")
    )
    z = fixed.project(avg.Y, "sa", mask=avg.mask)
    np.testing.assert_allclose(
        np.asarray(ax.get_lines()[0].get_xdata(), dtype=float), z[0, :, 0]
    )
    np.testing.assert_allclose(
        np.asarray(ax.get_lines()[0].get_ydata(), dtype=float), z[0, :, 1]
    )
    against_time = plot.trajectories(
        fixed, sim.Y, sim.X, "sa", by=["sa"], components=1, time=np.arange(6) / 10
    )
    assert against_time.get_xlabel() == "time"
    width_one = plot.trajectories(fixed, sim.Y, sim.X, "choice", by=["choice"])
    assert width_one.get_xlabel() == "time bin"
    swapped = plot.trajectories(fixed, sim.Y, sim.X, "sa", components=(1, 0))
    assert swapped.get_xlabel() == "component 2"


def test_trajectories_skips_empty_conditions(
    fixed: MTDR, sim: mtdr.SimulatedData
) -> None:
    ax = plot.trajectories(fixed, sim.Y, sim.X, "sa", by=["sa"], min_trials=10_000)
    assert "5 conditions skipped" in ax.get_title()
    constant = np.array(sim.X)
    constant[:, 0] = 1.0  # one condition: the colour scale has no span
    ax = plot.trajectories(fixed, sim.Y, constant, "sa", by=["sa"])
    assert len(ax.get_lines()) == 2


@pytest.mark.parametrize(
    ("components", "match"),
    [
        ((0, 1, 2), "an int or a pair"),
        ("a", "an int or a pair"),
        (2, "beyond"),
        ((0.5, 1.5), "an int or a pair"),  # not truncated to (0, 1)
        ((0, "1"), "an int or a pair"),
    ],
)
def test_trajectories_errors(
    fixed: MTDR, sim: mtdr.SimulatedData, components: Any, match: str
) -> None:
    with pytest.raises(ParameterError, match=match):
        plot.trajectories(fixed, sim.Y, sim.X, "sa", components=components)


def test_recovery(fixed: MTDR, sim: mtdr.SimulatedData) -> None:
    axes = plot.recovery(fixed, sim)
    assert axes.shape == (4,)
    r = np.corrcoef(fixed.B_["sa"].ravel(), np.ravel(sim.B["sa"]))[0, 1]
    assert axes[0].get_title() == f"sa: r = {r:.3f}"
    assert axes[3].get_xscale() == "log"
    assert axes[2].get_title().startswith("intercept")


def test_recovery_without_an_intercept_and_errors(sim: mtdr.SimulatedData) -> None:
    Y = np.array(sim.Y) - np.array(sim.intercept)[None]
    model = MTDR(ranks=[2, 1], estimator="svd", condition_independent=False).fit(
        Y, sim.X
    )
    assert plot.recovery(model, sim).shape == (3,)
    other = simulate(n_neurons=5, n_bins=6, n_trials=40, ranks=[1, 1], seed=1)
    with pytest.raises(ParameterError, match="not fitted to this simulation"):
        plot.recovery(model, other)
    flat = MTDR(ranks=[0, 1], estimator="svd").fit(sim.Y, sim.X)
    axes = plot.recovery(flat, sim)
    assert axes[0].get_title() == "x0: r undefined"


def test_bases_of_a_rank_zero_regressor(sim: mtdr.SimulatedData) -> None:
    model = MTDR(ranks=[0, 1], estimator="svd").fit(sim.Y, sim.X)
    ax = plot.bases(model, ["x0"])[0]
    assert ax.get_legend() is None


def test_no_positive_rank_is_a_parameter_error(sim: mtdr.SimulatedData) -> None:
    # An all-rank-0 model is a ParameterError, not matplotlib's ValueError.
    model = MTDR(ranks=[0, 0], estimator="svd").fit(sim.Y, sim.X)
    for func in (plot.bases, plot.weights):
        with pytest.raises(ParameterError, match="no regressor has positive rank"):
            func(model)


def test_recovery_checks_bins_first(fixed: MTDR) -> None:
    # A simulation with other bins is a ParameterError, not matplotlib's
    # ValueError.
    other = simulate(
        n_neurons=30, n_bins=7, n_trials=60, ranks={"sa": 2, "choice": 1}, seed=0
    )
    with pytest.raises(ParameterError, match="not fitted to this simulation"):
        plot.recovery(fixed, other)


def test_recovery_with_a_constant_truth_says_undefined() -> None:
    # A scalar noise precision is constant, but its float64 standard deviation
    # is 1e-16, not 0, so a test `std > 0` would title it "r = 0.000".
    s = simulate(
        n_neurons=15, n_bins=6, n_trials=100, ranks=[1], noise_precision=2 / 3, seed=0
    )
    model = MTDR(ranks=[1], estimator="svd").fit(s.Y, s.X)
    axes = plot.recovery(model, s)
    assert axes[-1].get_title() == "noise precision: r undefined"


def test_trajectories_has_a_key(fixed: MTDR, sim: mtdr.SimulatedData) -> None:
    # One legend entry per level at <= 8 levels, also when `by` spans several
    # regressors (ten conditions, five levels of sa).
    ax = plot.trajectories(fixed, sim.Y, sim.X, "sa")
    legend = ax.get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    assert texts == [f"sa = {v:g}" for v in (-2, -1, 0, 1, 2)]
    many = simulate(
        n_neurons=30,
        n_bins=6,
        n_trials=300,
        ranks=[2, 1],
        levels=[np.linspace(-2, 2, 9), [-1, 1]],
        seed=0,
    )
    model = MTDR(ranks=[2, 1], estimator="svd").fit(many.Y, many.X)
    ax = plot.trajectories(model, many.Y, many.X, "x0", by=["x0"])
    assert ax.get_legend() is None
    (bar,) = [a for a in ax.figure.axes if a is not ax]
    assert bar.get_ylabel() == "x0"
    np.testing.assert_allclose(bar.get_yticks(), np.linspace(-2, 2, 9))


def test_rank_search_with_capped_regressors() -> None:
    # A regressor that starts at max_rank is never a candidate, so it gets no
    # legend entry; a search that starts at max_rank has no candidate at all.
    _, capped = mtdr.rank_search.greedy_aic(
        lambda r: None, lambda f, r: -float(sum(r)), [1, 3], 3
    )
    ax = plot.rank_search(capped)[0]
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["+x0"]
    _, none = mtdr.rank_search.greedy_aic(lambda r: None, lambda f, r: 0.0, [2, 2], 2)
    assert plot.rank_search(none)[0].get_legend() is None


def test_rank_search_is_readable(model: MTDR) -> None:
    # Candidate markers labelled by regressor, no offset on the AIC axis,
    # integer steps, constrained layout.
    axes = plot.rank_search(model)
    for ax in axes:
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == [f"+{name}" for name in model.regressor_names_]
        assert ax.yaxis.get_major_formatter().get_useOffset() is False
        ax.figure.canvas.draw()
        ticks = ax.get_xticks()
        np.testing.assert_array_equal(ticks, np.round(ticks))
    engine = axes[0].figure.get_layout_engine()
    assert type(engine).__name__ == "ConstrainedLayoutEngine"
