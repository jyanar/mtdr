"""`simulate(groups=...)` and the per-regressor caps of `greedy_aic`."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest

import mtdr
from mtdr import ParameterError, simulate
from mtdr.rank_search import greedy_aic

COLS = ["a", "b", "c", "d"]


def test_a_group_shares_its_weight_draw() -> None:
    sim = simulate(
        n_neurons=12,
        n_bins=5,
        n_trials=60,
        ranks={"g": 2, "d": 1},
        regressor_names=COLS,
        groups={"g": ["c", "a", "b"]},
        seed=3,
    )
    assert dict(sim.groups) == {"g": ("a", "b", "c"), "d": ("d",)}  # type: ignore[arg-type]
    assert dict(sim.ranks) == {"g": 2, "d": 1}
    assert list(sim.W) == ["g", "d"]
    assert list(sim.S) == COLS
    assert sim.W["g"].shape == (12, 2)
    assert sim.S["b"].shape == (5, 2)
    assert sim.S["d"].shape == (5, 1)
    for c in ("a", "b", "c"):
        np.testing.assert_array_equal(sim.B[c], sim.W["g"] @ sim.S[c].T)
    assert "groups={'g': ('a', 'b', 'c'), 'd': ('d',)}" in repr(sim)
    again = pickle.loads(pickle.dumps(sim))
    assert dict(again.groups) == dict(sim.groups)  # type: ignore[arg-type]


def test_single_column_groups_draw_exactly_as_without_groups() -> None:
    plain = simulate(
        n_neurons=10, n_bins=4, n_trials=50, ranks={"a": 1, "b": 2}, seed=5
    )
    grouped = simulate(
        n_neurons=10,
        n_bins=4,
        n_trials=50,
        ranks={"a": 1, "b": 2},
        regressor_names=["a", "b"],
        groups={},
        seed=5,
    )
    for name in ("Y", "X", "mask", "intercept", "noise_precision"):
        np.testing.assert_array_equal(getattr(plain, name), getattr(grouped, name))
    for key in ("a", "b"):
        np.testing.assert_array_equal(plain.W[key], grouped.W[key])
        np.testing.assert_array_equal(plain.S[key], grouped.S[key])
    assert dict(plain.groups) == {"a": ("a",), "b": ("b",)}  # type: ignore[arg-type]
    assert "groups=" not in repr(plain)


def test_supplied_factors_and_amplitudes_with_groups() -> None:
    W = np.ones((8, 2))
    S = {"a": np.ones((4, 2)), "d": np.ones(4)}
    sim = simulate(
        n_neurons=8,
        n_bins=4,
        n_trials=40,
        ranks={"g": 2, "d": 1},
        regressor_names=COLS,
        groups={"g": ["a", "b", "c"]},
        W={"g": W},
        S=S,
        amplitude=[1.0, 0.0],
        seed=0,
    )
    np.testing.assert_array_equal(sim.W["g"], W)
    np.testing.assert_array_equal(sim.S["a"], np.ones((4, 2)))
    assert not sim.W["d"].any()  # amplitude 0 for group "d"
    seq = simulate(
        n_neurons=8,
        n_bins=4,
        n_trials=40,
        ranks={"g": 2, "d": 1},
        regressor_names=COLS,
        groups={"g": ["a", "b", "c"]},
        W=[W, None],  # type: ignore[list-item]
        seed=0,
    )
    np.testing.assert_array_equal(seq.W["g"], W)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"regressor_names": None}, "pass the column names as regressor_names"),
        ({"regressor_names": "abcd"}, "must be a sequence of str"),
        ({"regressor_names": []}, r"which are not columns of X; the columns are \[\]"),
        ({"ranks": [2, 1]}, "ranks as a mapping of group name"),
        ({"regressor_names": [], "groups": {}, "ranks": {}}, "at least one regressor"),
        ({"ranks": {"g": 2}}, "name exactly the groups"),
        ({"ranks": {"g": -1, "d": 1}}, "rank of 'g' must be a non-negative"),
        ({"ranks": {"g": 16, "d": 1}}, r"min\(n_neurons, 3 \* n_bins\) = 15"),
        ({"groups": {"g": ["a", "z"]}}, "which are not columns"),
        ({"W": {"a": np.ones((8, 2))}}, "which are not groups; the groups are"),
        ({"W": [np.ones((8, 2))]}, "1 entries but ranks has 2 groups"),
        ({"W": {"g": np.ones((8, 3))}}, r"group's rank as columns"),
        ({"S": {"a": np.ones((5, 1))}}, r"S\['a'\] must have shape \(5, 2\)"),
        ({"amplitude": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, "amplitude has 6 entries"),
    ],
)
def test_grouped_simulation_rejects(kw: dict[str, Any], match: str) -> None:
    args: dict[str, Any] = {
        "n_neurons": 8 if "ranks" not in kw else 20,
        "n_bins": 5,
        "n_trials": 40,
        "ranks": {"g": 2, "d": 1},
        "regressor_names": COLS,
        "groups": {"g": ["a", "b", "c"]},
        "seed": 0,
    }
    args.update(kw)
    with pytest.raises(ParameterError, match=match):
        simulate(**args)


def test_an_old_pickle_records_single_column_groups() -> None:
    sim = simulate(n_neurons=4, n_bins=3, n_trials=20, ranks=[1, 1], seed=0)
    state = dict(sim.__dict__)
    del state["groups"]
    old = mtdr.SimulatedData.__new__(mtdr.SimulatedData)
    old.__setstate__(state)
    assert dict(old.groups) == {"x0": ("x0",), "x1": ("x1",)}  # type: ignore[arg-type]


# ------------------------------------------------------------------ greedy_aic caps


def test_greedy_aic_takes_a_cap_per_regressor() -> None:
    # Each regressor stops at its own cap; the search ends when all are capped.
    _, history = greedy_aic(
        lambda r: None,
        lambda f, r: -float(np.sum(r)),
        init_ranks=[1, 1, 1],
        max_rank=[3, 1, 2],
        regressor_names=["a", "b", "c"],
    )
    assert history.final_ranks() == {"a": 3, "b": 1, "c": 2}
    assert history.stop_reason == "max_rank"
    assert all("b" not in round_ for round_ in history.candidates)
    _, same = greedy_aic(lambda r: None, lambda f, r: -float(r[0]), [1], np.array([2]))
    assert same.final_ranks() == {"x0": 2}


@pytest.mark.parametrize(
    ("init", "caps", "match"),
    [
        ([1, 3], [2, 2], r"init_ranks\[1\] is 3, above max_rank\[1\] = 2"),
        ([1, 3], 2, r"init_ranks\[1\] is 3, above max_rank = 2"),
        ([1, 1], [2], "max_rank has 1 entries; expected one per regressor"),
        ([1, 1], [2, 0], r"max_rank\[1\] must be a positive integer"),
    ],
)
def test_greedy_aic_checks_its_caps(init: list[int], caps: Any, match: str) -> None:
    with pytest.raises(ParameterError, match=match):
        greedy_aic(lambda r: None, lambda f, r: 0.0, init, caps)
