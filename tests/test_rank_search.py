"""Tests for `mtdr.rank_search`: greedy semantics, history, recovery on simulations."""

from __future__ import annotations

import builtins
import functools
import itertools
import pickle
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

import matlab_compat as mc
import mtdr
from mtdr.errors import ConvergenceWarning, ParameterError
from mtdr.mmle import fit_mmle
from mtdr.rank_search import RankSearchHistory, greedy_aic
from mtdr.stats import sufficient_statistics
from mtdr.svd_fit import fit_svd

IntArray = NDArray[np.int64]


@dataclass
class Fake:
    """A fit object that remembers the ranks it was fitted at."""

    ranks: tuple[int, ...]
    converged: Any = True


def fake_fit(ranks: IntArray) -> Fake:
    return Fake(tuple(int(r) for r in ranks))


def table_objective(table: dict[Any, float]) -> Callable[[Fake, IntArray], float]:
    def objective(fit: Fake, ranks: IntArray) -> float:
        # The objective sees the same ranks as the fit, and the fit object
        # that `fit` returned for them (no stale or misindexed fit).
        assert fit.ranks == tuple(int(r) for r in ranks)
        return table[fit.ranks]

    return objective


def bowl(
    target: tuple[int, ...], weights: tuple[float, ...]
) -> Callable[[Fake, IntArray], float]:
    def objective(fit: Fake, ranks: IntArray) -> float:
        return float(
            sum(
                w * (r - t) ** 2 for r, t, w in zip(ranks, target, weights, strict=True)
            )
        )

    return objective


# ------------------------------------------------------------------ semantics


def test_descends_a_bowl_to_its_minimum() -> None:
    best, h = greedy_aic(fake_fit, bowl((3, 1, 5), (1.0, 2.0, 0.5)), [1, 1, 1], 8)
    assert best.ranks == (3, 1, 5)
    assert h.final_ranks() == {"x0": 3, "x1": 1, "x2": 5}
    assert h.stop_reason == "no_improvement"
    steps = np.diff(h.ranks, axis=0)
    assert ((steps == 0) | (steps == 1)).all()
    assert (steps.sum(axis=1) == 1).all()
    assert (np.diff(h.aic) < 0).all()
    assert len(h.candidates[-1]) == 3
    assert min(h.candidates[-1].values()) >= h.aic[-1]
    for k, name in enumerate(h.accepted):
        p = h.regressor_names.index(name)
        assert h.ranks[k + 1, p] == h.ranks[k, p] + 1


def test_ties_go_to_the_lowest_index() -> None:
    # Raising either regressor lowers the score equally: always regressor 0.
    def objective(fit: Fake, ranks: IntArray) -> float:
        return -float(min(sum(ranks), 4))

    _, h = greedy_aic(fake_fit, objective, [1, 1], 5)
    assert h.accepted == ("x0", "x0")
    assert h.ranks[-1].tolist() == [3, 1]


def test_an_equal_score_is_rejected_unlike_the_reference() -> None:
    # EstRankGreedily accepts dObj <= 0; the port requires a strict decrease.
    table = {(1,): 5.0, (2,): 5.0, (3,): 4.0}
    best, h = greedy_aic(fake_fit, table_objective(table), [1], 3)
    assert best.ranks == (1,)
    assert h.stop_reason == "no_improvement"
    assert list(h.candidates) == [{"x0": 5.0}]
    rest, rhist, funhist, _ = mc.est_rank_greedily_reference(
        table_objective(table), fake_fit, [1], 3
    )
    assert rest.tolist() == [3]
    assert rhist[:, 0].tolist() == [1, 2, 3]
    assert funhist.tolist() == [5.0, 5.0, 4.0]


@pytest.mark.parametrize(
    ("threshold", "accepted"),
    [(0.0, True), (2.5, False), (2.4999999, True), (3.0, False)],
)
def test_threshold_boundary(threshold: float, accepted: bool) -> None:
    # Accept iff the decrease exceeds the threshold: exactly 2.5 is rejected.
    table = {(1,): 10.0, (2,): 7.5, (3,): 7.5}
    _, h = greedy_aic(fake_fit, table_objective(table), [1], 3, threshold=threshold)
    assert (len(h.accepted) == 1) is accepted
    assert h.threshold == threshold


def test_capped_regressors_leave_the_candidate_set_and_max_rank_stops() -> None:
    def objective(fit: Fake, ranks: IntArray) -> float:
        return -float(sum(ranks))  # always improves

    calls: list[tuple[int, ...]] = []

    def counting_fit(ranks: IntArray) -> Fake:
        calls.append(tuple(int(r) for r in ranks))
        return fake_fit(ranks)

    best, h = greedy_aic(counting_fit, objective, [1, 3], 3)
    assert best.ranks == (3, 3)
    assert h.stop_reason == "max_rank"
    assert h.candidates[-1] == {}
    # Regressor 1 starts capped: it is never a candidate.
    assert all("x1" not in c for c in h.candidates)
    assert len(calls) == 1 + sum(len(c) for c in h.candidates)
    assert h.ranks.shape[0] == h.aic.shape[0] == len(h.candidates) == len(h.converged)
    assert len(h.accepted) + 1 == h.ranks.shape[0]


def test_start_at_max_rank_runs_no_round() -> None:
    calls: list[Any] = []

    def counting_fit(ranks: IntArray) -> Fake:
        calls.append(ranks)
        return fake_fit(ranks)

    best, h = greedy_aic(counting_fit, bowl((0, 0), (1.0, 1.0)), [2, 2], 2)
    assert len(calls) == 1
    assert best.ranks == (2, 2)
    assert list(h.candidates) == [{}]
    assert h.accepted == ()
    assert h.stop_reason == "max_rank"


def test_stored_fit_is_the_accepted_one() -> None:
    # EstRankGreedily.m:67 stores parhat{indmin} where it means
    # parhat{indmove(indmin)}. From (1, 1) with max_rank 2, the search accepts
    # (2, 1) and then (2, 2). In round 2 only regressor 1 is movable, and the
    # reference stores parhat{indmin} = parhat{1}, the stale fit of (2, 1)
    # from round 1, as the fit of (2, 2).
    table = {(1, 1): 10.0, (2, 1): 8.0, (1, 2): 9.0, (2, 2): 7.0}
    best, h = greedy_aic(fake_fit, table_objective(table), [1, 1], 2)
    assert h.ranks.tolist() == [[1, 1], [2, 1], [2, 2]]
    assert best.ranks == (2, 2)
    _, rhist, _, parhist = mc.est_rank_greedily_reference(
        table_objective(table), fake_fit, [1, 1], 2
    )
    assert rhist.tolist() == h.ranks.tolist()
    assert parhist[-1].ranks == (2, 1)  # the reference's defect, not reproduced


def test_six_regressor_search_returns_the_accepted_fit() -> None:
    # With six regressors and a last accepted step that is not regressor 0,
    # the returned fit is the accepted candidate's (its ranks, its parameters
    # and its score), not another candidate's.
    sim = mtdr.simulate(
        n_neurons=40,
        n_bins=8,
        n_trials=400,
        ranks=[1, 1, 1, 1, 1, 3],
        noise_precision=4.0,
        seed=3,
    )
    stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    fits: list[Any] = []

    def fit(ranks: IntArray) -> Any:
        out = fit_svd(stats, ranks)
        fits.append(out)
        return out

    best, h = greedy_aic(fit, lambda f, r: f.aic, [1] * 6, 8)
    assert h.accepted, "the search must take a step"
    assert h.accepted[-1] != h.regressor_names[0]
    assert best.ranks == tuple(h.ranks[-1])
    assert best.aic == h.aic[-1]
    # The candidate object fitted at the accepted ranks in the last round.
    accepted = [f for f in fits if f.ranks == best.ranks]
    assert accepted[-1] is best
    again = fit_svd(stats, h.ranks[-1])
    assert best.intercept is not None
    assert again.intercept is not None
    for a, b in zip(best.B, again.B, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(best.intercept, again.intercept)
    assert best.log_likelihood == again.log_likelihood


def test_fit_receives_fresh_arrays() -> None:
    seen: list[IntArray] = []

    def vandal(ranks: IntArray) -> Fake:
        assert ranks.dtype == np.int64
        assert ranks.shape == (2,)
        seen.append(ranks)
        out = fake_fit(ranks)
        ranks[:] = 99  # must not leak into the search
        return out

    def objective(fit: Fake, ranks: IntArray) -> float:
        ranks[:] = -5
        return bowl((2, 2), (1.0, 1.0))(fit, np.array(fit.ranks))

    best, h = greedy_aic(vandal, objective, [1, 1], 4)
    assert best.ranks == (2, 2)
    assert h.ranks[-1].tolist() == [2, 2]
    assert len({id(r) for r in seen}) == len(seen)


def test_callback_and_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    events: list[tuple[int, list[int], float]] = []

    def callback(step: int, ranks: IntArray, score: float) -> None:
        events.append((step, ranks.tolist(), score))
        ranks[:] = 0  # a copy

    _, h = greedy_aic(
        fake_fit,
        bowl((2, 3), (1.0, 1.0)),
        [1, 1],
        5,
        verbose=1,
        callback=callback,
        regressor_names=["sa", "choice"],
    )
    assert [e[0] for e in events] == list(range(1, len(h.accepted) + 1))
    assert [e[1] for e in events] == h.ranks[1:].tolist()
    assert [e[2] for e in events] == h.aic[1:].tolist()
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(h.accepted) + 1
    assert lines[0].startswith("rank search (custom): step 0: {sa: 1, choice: 1}")
    assert lines[-1].startswith(
        f"rank search (custom): step {len(h.accepted)}: {{sa: 2, choice: 3}}"
    )
    greedy_aic(fake_fit, bowl((2, 3), (1.0, 1.0)), [1, 1], 5, verbose=2)
    out = capsys.readouterr().out
    assert out.count("candidates") == len(h.candidates)


def test_fit_many_fits_each_round() -> None:
    # `fit_many` receives a round's candidates in regressor order, fresh
    # arrays, and the search is the same as with `fit` alone.
    rounds: list[list[list[int]]] = []

    def fit_many(trials: list[IntArray]) -> list[Fake]:
        rounds.append([t.tolist() for t in trials])
        out = [fake_fit(t) for t in trials]
        for t in trials:
            t[:] = 99  # must not leak into the search
        return out

    objective = bowl((2, 3, 1), (1.0, 2.0, 0.5))
    plain_best, plain = greedy_aic(fake_fit, objective, [1, 1, 1], 4)
    best, h = greedy_aic(fake_fit, objective, [1, 1, 1], 4, fit_many=fit_many)
    assert best == plain_best
    np.testing.assert_array_equal(h.ranks, plain.ranks)
    assert h.candidates == plain.candidates
    assert h.accepted == plain.accepted
    assert len(rounds) == len(h.candidates)
    for row, trials in zip(h.ranks, rounds, strict=False):
        assert trials == [
            (row + np.eye(3, dtype=np.int64)[p]).tolist() for p in range(3)
        ]


def test_fit_many_must_return_one_fit_per_candidate() -> None:
    with pytest.raises(ParameterError, match="fit_many returned 1 fits for 2"):
        greedy_aic(
            fake_fit,
            bowl((2, 2), (1.0, 1.0)),
            [1, 1],
            3,
            fit_many=lambda trials: [fake_fit(trials[0])],
        )


def test_converged_and_estimator_are_recorded() -> None:
    def fit(ranks: IntArray) -> Fake:
        return Fake(tuple(int(r) for r in ranks), converged=np.bool_(ranks[0] != 2))

    _, h = greedy_aic(fit, bowl((3,), (1.0,)), [1], 4)
    assert h.converged == (True, False, True)
    assert h.estimator == "custom"
    with pytest.raises(ParameterError, match="converged attribute must be a bool"):
        greedy_aic(lambda r: Fake((1,), converged="yes"), bowl((1,), (1.0,)), [1], 2)

    class NoFlag:
        pass

    _, h = greedy_aic(lambda r: NoFlag(), lambda f, r: 1.0, [1], 2)
    assert h.converged == (True,)
    # With P >= 2, each entry is the accepted candidate's flag, not that of
    # another candidate of the same round. Only fits with x1 == 2 are
    # unconverged; the search raises x0, x1 (beating the converged x0
    # candidate (3, 1)), then x0.

    def fit2(ranks: IntArray) -> Fake:
        return Fake(tuple(int(r) for r in ranks), converged=bool(ranks[1] != 2))

    table = {
        (1, 1): 10.0,
        (2, 1): 8.0,
        (1, 2): 9.0,
        (3, 1): 7.5,
        (2, 2): 7.0,
        (3, 2): 6.0,
        (2, 3): 6.5,
        (3, 3): 6.2,
    }
    _, h = greedy_aic(fit2, table_objective(table), [1, 1], 3)
    assert h.accepted == ("x0", "x1", "x0")
    assert h.converged == (True, True, False, False)


def test_infinite_scores() -> None:
    # +inf marks an infeasible candidate; from an infinite start any finite
    # candidate is an improvement.
    table = {(1,): np.inf, (2,): 3.0, (3,): np.inf}
    best, h = greedy_aic(fake_fit, table_objective(table), [1], 3)
    assert best.ranks == (2,)
    assert h.candidates[-1] == {"x0": np.inf}


@pytest.mark.parametrize("value", [np.nan, True, "1.0", None, [1.0], -np.inf])
def test_objective_must_return_a_number(value: Any) -> None:
    with pytest.raises(ParameterError, match="objective must return a real number"):
        greedy_aic(fake_fit, lambda f, r: value, [1], 2)


def test_a_minus_inf_candidate_is_an_error() -> None:
    # A -inf score (an unbounded likelihood) would win every round and freeze
    # the search, so it is rejected like NaN; +inf stays an infeasible
    # candidate.
    table = {(1,): 0.0, (2,): -np.inf, (3,): -np.inf}
    with pytest.raises(ParameterError, match=r"or -inf; got -inf at ranks \[2\]"):
        greedy_aic(fake_fit, table_objective(table), [1], 3)


def test_errors_from_fit_propagate() -> None:
    def broken(ranks: IntArray) -> Fake:
        if ranks[0] > 1:
            raise RuntimeError("boom")
        return fake_fit(ranks)

    with pytest.raises(RuntimeError, match="boom"):
        greedy_aic(broken, bowl((3,), (1.0,)), [1], 3)


# ------------------------------------------------------------------ properties


@st.composite
def rank_tables(
    draw: st.DrawFn,
) -> tuple[list[int], int, dict[tuple[int, ...], float], float]:
    P = draw(st.integers(1, 3))
    cap = draw(st.integers(1, 4))
    init = draw(st.lists(st.integers(1, cap), min_size=P, max_size=P))
    grid = itertools.product(*(range(1, cap + 1) for _ in range(P)))
    # Small integer scores make ties and exact-threshold cases frequent.
    table = {g: float(draw(st.integers(-6, 6))) for g in grid}
    threshold = float(draw(st.sampled_from([0.0, 0.0, 1.0, 2.5])))
    return init, cap, table, threshold


@settings(deadline=None)
@given(case=rank_tables())
def test_greedy_invariants(
    case: tuple[list[int], int, dict[tuple[int, ...], float], float],
) -> None:
    init, cap, table, threshold = case
    best, h = greedy_aic(
        fake_fit, table_objective(table), init, cap, threshold=threshold
    )
    rows = len(h.accepted) + 1
    assert h.ranks.shape == (rows, len(init))
    assert h.aic.shape == (rows,)
    assert len(h.candidates) == rows == len(h.converged)
    assert h.ranks[0].tolist() == init == h.init_ranks.tolist()
    assert best.ranks == tuple(h.ranks[-1])
    for k in range(rows):
        assert h.aic[k] == table[tuple(h.ranks[k])]
        movable = [p for p in range(len(init)) if h.ranks[k, p] < cap]
        expected = {}
        for p in movable:
            trial = h.ranks[k].copy()
            trial[p] += 1
            expected[h.regressor_names[p]] = table[tuple(trial)]
        assert h.candidates[k] == expected
        if k < rows - 1:
            # Accepted: the first minimiser, and it improves by more than δ
            # (strictly, at δ = 0: the AIC strictly improves at each step).
            p = h.regressor_names.index(h.accepted[k])
            scores = [expected[h.regressor_names[q]] for q in movable]
            assert movable[int(np.argmin(scores))] == p
            assert h.aic[k] - h.aic[k + 1] > threshold
    last = h.candidates[-1]
    if h.stop_reason == "max_rank":
        assert last == {}
        assert (h.ranks[-1] == cap).all()
    else:
        assert last
        assert h.aic[-1] - min(last.values()) <= threshold


@settings(deadline=None)
@given(data=st.data())
def test_matches_the_reference_bookkeeping_away_from_ties(data: st.DataObject) -> None:
    # With distinct scores (no dObj == 0) and threshold 0 the port's path equals
    # EstRankGreedily's (the replica's), rank vectors and scores.
    P = data.draw(st.integers(1, 3))
    cap = data.draw(st.integers(1, 4))
    grid = list(itertools.product(*(range(1, cap + 1) for _ in range(P))))
    values = data.draw(
        st.lists(
            st.integers(-1000, 1000),
            min_size=len(grid),
            max_size=len(grid),
            unique=True,
        )
    )
    table = {g: float(v) for g, v in zip(grid, values, strict=True)}
    init = data.draw(st.lists(st.integers(1, cap), min_size=P, max_size=P))
    _, h = greedy_aic(fake_fit, table_objective(table), init, cap)
    _rest, rhist, funhist, _ = mc.est_rank_greedily_reference(
        table_objective(table), fake_fit, init, cap
    )
    assert h.ranks.tolist() == rhist.tolist()
    assert h.aic.tolist() == funhist.tolist()


# ------------------------------------------------------------------ history


def _history() -> RankSearchHistory:
    return greedy_aic(
        fake_fit, bowl((2, 1), (1.0, 1.0)), [1, 1], 3, regressor_names=["sa", "choice"]
    )[1]


def test_history_is_read_only_and_pickles() -> None:
    h = _history()
    for a in (h.init_ranks, h.ranks, h.aic):
        assert not a.flags.writeable
    with pytest.raises(AttributeError):
        h.stop_reason = "max_rank"  # type: ignore[misc]
    # Immutable containers: no append, no item assignment.
    assert isinstance(h.candidates, tuple)
    assert isinstance(h.accepted, tuple)
    assert isinstance(h.converged, tuple)
    with pytest.raises(TypeError):
        h.candidates[0]["sa"] = 0.0  # type: ignore[index]
    again = pickle.loads(pickle.dumps(h))
    assert again.final_ranks() == h.final_ranks() == {"sa": 2, "choice": 1}
    # Unpickling restores read-only arrays.
    for a in (again.init_ranks, again.ranks, again.aic):
        assert not a.flags.writeable
    assert list(again.candidates) == list(h.candidates)
    assert h.svd_stage is None
    assert repr(h) == (
        "RankSearchHistory(estimator='custom', init_ranks=[1, 1], "
        "final_ranks={'sa': 2, 'choice': 1}, n_accepted=1, aic=0, "
        "stop_reason='no_improvement')"
    )


def test_history_checks_its_invariant() -> None:
    h = _history()
    fields: dict[str, Any] = {
        "regressor_names": h.regressor_names,
        "estimator": "svd",
        "init_ranks": h.init_ranks,
        "ranks": h.ranks,
        "aic": h.aic,
        "candidates": h.candidates,
        "accepted": h.accepted,
        "converged": h.converged,
        "threshold": 0.0,
        "stop_reason": "no_improvement",
    }
    nested = RankSearchHistory(**{**fields, "svd_stage": h})
    assert nested.svd_stage is h
    with pytest.raises(ParameterError, match="len\\(accepted\\) \\+ 1"):
        RankSearchHistory(**{**fields, "accepted": []})
    with pytest.raises(ParameterError, match="len\\(accepted\\) \\+ 1"):
        RankSearchHistory(**{**fields, "init_ranks": [1, 1, 1]})
    with pytest.raises(ParameterError, match="stop_reason must be one of"):
        RankSearchHistory(**{**fields, "stop_reason": "threshold"})


def _fields() -> dict[str, Any]:
    h = _history()  # sa 1 -> 2, then the terminal round
    return {
        "regressor_names": ["sa", "choice"],
        "estimator": "custom",
        "init_ranks": [1, 1],
        "ranks": [[1, 1], [2, 1]],
        "aic": h.aic.tolist(),
        "candidates": [dict(c) for c in h.candidates],
        "accepted": ["sa"],
        "converged": [True, True],
        "threshold": 0.0,
        "stop_reason": "no_improvement",
    }


BAD_HISTORIES: list[tuple[dict[str, Any], str]] = [
    ({"init_ranks": [2, 1]}, r"ranks\[0\] must equal init_ranks"),
    ({"ranks": [[1.2, 1], [2.9, 1]]}, "ranks must hold non-negative integers"),
    ({"init_ranks": [1.0, 1.0]}, "init_ranks must hold non-negative integers"),
    ({"init_ranks": [True, True]}, "init_ranks must hold non-negative integers"),
    (
        {"init_ranks": [-1, 1], "ranks": [[-1, 1], [0, 1]]},
        "init_ranks must hold non-negative integers",
    ),
    ({"ranks": [[1, 1], [3, 1]]}, r"ranks\[1\] must be ranks\[0\] with 'sa' raised"),
    ({"ranks": [[1, 1], [1, 2]]}, r"ranks\[1\] must be ranks\[0\] with 'sa' raised"),
    ({"accepted": ["nope"]}, r"accepted\[0\] is 'nope', not a regressor name"),
    ({"accepted": [3]}, "accepted must be a sequence of str"),
    ({"aic": [0.0, 99.0]}, r"aic\[1\] must equal candidates\[0\]\['sa'\]"),
    (
        {"candidates": [{"zz": 1.0, "sa": -1.0}, {"sa": 1.0, "choice": 1.0}]},
        r"candidates\[0\] has keys \['zz'\]",
    ),
    ({"candidates": [{}, 3]}, "candidates must be a sequence of mappings"),
    ({"candidates": "ab"}, "candidates must be a sequence of mappings"),
    ({"candidates": [{"sa": "x"}, {}]}, "candidates must map str to a real number"),
    ({"converged": [True, "x"]}, "converged must hold bools"),
    ({"threshold": -3.0}, "threshold must be a finite real number >= 0"),
    ({"stop_reason": "max_rank"}, "empty exactly when stop_reason is 'max_rank'"),
    ({"regressor_names": ["sa", "sa"]}, "unique"),
    ({"regressor_names": 3}, "regressor_names must be a sequence of str"),
    ({"svd_stage": "svd"}, "svd_stage must be a RankSearchHistory or None"),
]


@pytest.mark.parametrize(("change", "match"), BAD_HISTORIES)
def test_history_rejects_an_incoherent_record(
    change: dict[str, Any], match: str
) -> None:
    # The constructor checks that the record is a coherent search.
    fields = _fields()
    RankSearchHistory(**fields)  # the unchanged record is accepted
    with pytest.raises(ParameterError, match=match):
        RankSearchHistory(**{**fields, **change})


def test_history_copies_the_callers_containers() -> None:
    # Mutating the lists passed in, or the history's own containers, cannot
    # break the invariant after construction.
    fields = _fields()
    h = RankSearchHistory(**fields)
    fields["accepted"].clear()
    fields["candidates"][0]["sa"] = 99.0
    fields["candidates"].append({})
    fields["converged"][0] = "x"
    assert h.accepted == ("sa",)
    assert len(h.candidates) == len(h.aic) == len(h.converged) == 2
    assert h.candidates[0]["sa"] == h.aic[1]
    assert h.converged == (True, True)
    with pytest.raises(AttributeError):
        h.accepted.append("x")  # type: ignore[attr-defined]


def test_max_rank_history_with_an_empty_last_round_is_accepted() -> None:
    _, h = greedy_aic(fake_fit, lambda f, r: -float(sum(r)), [1], 2)
    assert h.stop_reason == "max_rank"
    again = RankSearchHistory(
        regressor_names=h.regressor_names,
        estimator=h.estimator,
        init_ranks=h.init_ranks,
        ranks=h.ranks,
        aic=h.aic,
        candidates=h.candidates,
        accepted=h.accepted,
        converged=h.converged,
        threshold=h.threshold,
        stop_reason="max_rank",
    )
    assert list(again.candidates) == [{"x0": -2.0}, {}]


def test_n_fits_counts_the_start_and_every_candidate() -> None:
    # n_fits = 1 + the candidates of every round, the fit calls greedy_aic
    # made; this stage only (svd_stage counts its own).
    calls: list[list[int]] = []

    def fit(r: np.ndarray) -> None:
        calls.append(r.tolist())

    _, h = greedy_aic(
        fit, lambda f, r: float((r[0] - 3) ** 2 + (r[1] - 2) ** 2), [1, 1], 5
    )
    assert h.n_fits == len(calls) == 1 + sum(len(c) for c in h.candidates)
    _, capped = greedy_aic(fit, lambda f, r: -float(r.sum()), [4, 5], 5)
    assert capped.stop_reason == "max_rank"
    assert capped.n_fits == 2  # the start and the one move left
    with pytest.raises(AttributeError):
        h.n_fits = 3  # type: ignore[misc]


def test_n_fits_of_a_two_stage_search() -> None:
    sim = mtdr.simulate(n_neurons=30, n_bins=6, n_trials=200, ranks=[2, 1], seed=0)
    model = mtdr.MTDR(max_rank=4).fit(sim.Y, sim.X)
    h = model.rank_search_history_
    assert h is not None
    assert h.svd_stage is not None
    assert h.n_fits == 1 + sum(len(c) for c in h.candidates)
    assert h.svd_stage.n_fits == 1 + sum(len(c) for c in h.svd_stage.candidates)


def test_to_frame() -> None:
    import pandas as pd  # in the test extra; to_frame imports it lazily

    h = _history()
    df = h.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "rank[sa]",
        "rank[choice]",
        "aic",
        "converged",
        "candidate[sa]",
        "candidate[choice]",
        "accepted",
        "improvement",
        "decision",
        "runner_up",
    ]
    assert df.index.name == "step"
    assert len(df) == 2
    assert df["rank[sa]"].tolist() == [1, 2]
    assert df["accepted"].iloc[0] == "sa"
    assert pd.isna(df["accepted"].iloc[-1])
    assert df["candidate[choice]"].tolist() == [
        h.candidates[0]["choice"],
        h.candidates[1]["choice"],
    ]


def test_to_frame_without_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def no_pandas(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pandas", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_pandas)
    with pytest.raises(
        ImportError, match=r"needs pandas: pip install 'mtdr\[xarray\]' \(from a clone"
    ):
        _history().to_frame()


# ------------------------------------------------------------------ arguments


BAD: list[tuple[dict[str, Any], str]] = [
    ({"fit": 3}, "fit must be callable"),
    ({"objective": None}, "objective must be callable"),
    ({"callback": "f"}, "callback must be callable or None"),
    ({"fit_many": 1}, "fit_many must be callable or None"),
    ({"init_ranks": [0, 1]}, r"init_ranks\[0\] must be a positive integer"),
    ({"init_ranks": [-1]}, "positive integer"),
    ({"init_ranks": [True]}, "positive integer"),
    ({"init_ranks": [1.0]}, "positive integer"),
    ({"init_ranks": []}, "at least one entry"),
    ({"init_ranks": np.ones((1, 2), dtype=int)}, "init_ranks must be a sequence"),
    ({"init_ranks": 1}, "init_ranks must be a sequence"),
    ({"init_ranks": [4]}, r"init_ranks\[0\] is 4, above max_rank = 3"),
    ({"max_rank": 0}, "max_rank must be a positive integer"),
    ({"max_rank": True}, "max_rank must be a positive integer"),
    ({"max_rank": 2.0}, "max_rank must be a positive integer"),
    ({"threshold": -1.0}, "threshold must be a finite real number >= 0"),
    ({"threshold": np.inf}, "threshold"),
    ({"threshold": np.nan}, "threshold"),
    ({"threshold": True}, "threshold"),
    ({"regressor_names": ["a", "b"]}, "regressor_names has 2 entries; expected 1"),
    ({"regressor_names": "a"}, "regressor_names must be a sequence of str"),
    ({"regressor_names": [3]}, "regressor names must be str"),
    ({"regressor_names": ["total"]}, "is reserved"),
    ({"verbose": 3}, "verbose must be 0, 1 or 2"),
    ({"verbose": True}, "verbose must be 0, 1 or 2"),
    ({"verbose": "1"}, "verbose must be 0, 1 or 2"),
]


@pytest.mark.parametrize(("change", "match"), BAD)
def test_invalid_arguments_raise_parameter_error(
    change: dict[str, Any], match: str
) -> None:
    kwargs: dict[str, Any] = {
        "fit": fake_fit,
        "objective": bowl((1,), (1.0,)),
        "init_ranks": [1],
        "max_rank": 3,
        **change,
    }
    with pytest.raises(ParameterError, match=match):
        greedy_aic(**kwargs)


def test_duplicate_names_and_accepted_containers() -> None:
    with pytest.raises(ParameterError, match="unique"):
        greedy_aic(
            fake_fit, bowl((1, 1), (1, 1)), [1, 1], 2, regressor_names=["a", "a"]
        )
    # 1-D arrays are sequences, NumPy scalars and 0-d arrays are scalars.
    numpy_kwargs: dict[str, Any] = {
        "init_ranks": np.array([1]),
        "max_rank": np.int64(3),
        "threshold": np.array(0.0),
        "regressor_names": np.array(["sa"]),
        "verbose": np.int8(0),
    }
    _, h = greedy_aic(fake_fit, bowl((2,), (1.0,)), **numpy_kwargs)
    assert h.regressor_names == ("sa",)
    assert h.final_ranks() == {"sa": 2}


# ------------------------------------------------------------ SVD-AIC searches


def _svd_search(
    sim: mtdr.SimulatedData,
    start: list[int],
    X: NDArray[np.float64] | None = None,
    **kwargs: Any,
) -> tuple[Any, RankSearchHistory]:
    design = sim.X if X is None else X
    stats = sufficient_statistics(sim.Y_masked, design, sim.mask)
    n, T = sim.Y.shape[1:]
    return greedy_aic(
        functools.partial(fit_svd, stats, **kwargs),
        lambda f, r: f.aic,
        start,
        min(n, T),
        regressor_names=sim.regressor_names,
    )


@pytest.mark.parametrize("seed", range(100))
def test_svd_aic_recovers_ranks_with_equal_noise(seed: int) -> None:
    # Success criterion: the exact true ranks, with margins. Configuration:
    # n = 200 neurons, T = 10 bins, N = 400 trials, 30 % of neuron-trials
    # dropped, equal noise precisions. Over 1000 seeds (0-999) recovery was
    # exact every time, every accepted step lowered the AIC by at least 2661,
    # and the best rejected final candidate was at least 68.9 above the chosen
    # fit; the bounds below are 26x and 6.9x looser.
    sim = mtdr.simulate(
        n_neurons=200,
        n_bins=10,
        n_trials=400,
        ranks=[2, 1, 3],
        drop_prob=0.3,
        noise_precision=1.25,
        seed=seed,
    )
    best, h = _svd_search(sim, [1, 1, 1])
    assert h.final_ranks() == dict(sim.ranks)
    assert best.ranks == (2, 1, 3)
    assert h.estimator == "svd"
    assert (-np.diff(h.aic) > 100.0).all()
    assert min(h.candidates[-1].values()) - h.aic[-1] > 10.0
    assert h.stop_reason == "no_improvement"


def test_svd_aic_overselects_under_heavy_tailed_noise() -> None:
    # Under the demo's Exponential precisions the unweighted SVD truncation is
    # not the rank-r plug-in optimum, so the corrected SVD AIC keeps finding
    # improvements past the true rank. Over 1000 seeds of this configuration
    # (n=100, T=15, N=400, drop 0.3, the demo's precision draw) it never
    # under-selected and over-selected 43.7 % of the time. At small N or low
    # SNR it can also under-select, which this configuration does not reach.
    # The SVD stage is a seed for the MMLE search, not a rank estimate.
    over = 0
    for seed in range(40):
        sim = mtdr.simulate(
            n_neurons=100,
            n_bins=15,
            n_trials=400,
            ranks=[2, 1, 3],
            drop_prob=0.3,
            seed=seed,
        )
        _, h = _svd_search(sim, [1, 1, 1])
        chosen = h.ranks[-1]
        assert (chosen >= [2, 1, 3]).all()  # never below the truth
        over += int((chosen > [2, 1, 3]).any())
        assert (np.diff(h.aic) < 0).all()
    # Binomial(40, 0.44): mean 17.6, sd 3.1; 8 is 3.1 sd below.
    assert over >= 8


def test_svd_aic_search_does_not_depend_on_regressor_coding() -> None:
    # The SVD path refits the intercept given the truncated blocks. Keeping
    # the joint least-squares intercept next to them instead makes the search
    # depend on the coding: on this simulation (equal precisions, n=200,
    # T=10, N=400, seed 0) recoding the binary regressor from +-1 to 0/1
    # moved its chosen rank from 3 to 7, and with x0 also shifted to 0..4 the
    # search chose [10, 1, 7]. Shifting and rescaling a regressor is the same
    # model (`docs/model.md` § 10.2): the search and every score are
    # unchanged.
    sim = mtdr.simulate(
        n_neurons=200,
        n_bins=10,
        n_trials=400,
        ranks=[2, 1, 3],
        drop_prob=0.3,
        noise_precision=1.25,
        seed=0,
    )
    recoded = sim.X.copy()
    recoded[:, 2] = (recoded[:, 2] + 1) / 2  # +-1 to 0/1
    recoded[:, 0] += 2.0  # levels -2..2 to 0..4
    _, h1 = _svd_search(sim, [1, 1, 1])
    _, h2 = _svd_search(sim, [1, 1, 1], X=recoded)
    assert h1.final_ranks() == h2.final_ranks() == dict(sim.ranks)
    np.testing.assert_allclose(h2.aic, h1.aic, rtol=1e-12)
    for c1, c2 in zip(h1.candidates, h2.candidates, strict=True):
        assert c1.keys() == c2.keys()
        np.testing.assert_allclose(list(c2.values()), list(c1.values()), rtol=1e-12)


def _demo_noise(seed: int) -> mtdr.SimulatedData:
    """Heavy-tailed noise: the demo's Exponential precisions, n=100, T=15, N=400."""
    return mtdr.simulate(
        n_neurons=100,
        n_bins=15,
        n_trials=400,
        ranks=[2, 1, 3],
        drop_prob=0.3,
        seed=seed,
    )


def test_precision_weighted_svd_search_recovers_ranks_under_heavy_tailed_noise() -> (
    None
):
    # The SVD AIC search on the precision-weighted truncation (M18w). Over
    # seeds 2000-2099 of `_demo_noise` it chose the true ranks in 100/100
    # (exact / over / under 100 / 0 / 0) against 59 / 41 / 0 for the
    # unweighted truncation; on the 20 seeds below, 20 and 12 exact. The
    # bounds leave two misses and half the measured gap.
    exact_weighted = exact_plain = 0
    for seed in range(2000, 2020):
        sim = _demo_noise(seed)
        _, weighted = _svd_search(sim, [1, 1, 1], precision_weighted=True)
        _, plain = _svd_search(sim, [1, 1, 1])
        chosen = weighted.ranks[-1]
        assert (chosen >= [2, 1, 3]).all()
        assert weighted.estimator == "svd"
        exact_weighted += int((chosen == [2, 1, 3]).all())
        exact_plain += int((plain.ranks[-1] == [2, 1, 3]).all())
    assert exact_weighted >= 18
    assert exact_weighted - exact_plain >= 4


def test_mmle_search_seeded_by_the_weighted_svd_search() -> None:
    # The MMLE AIC search (M38a) started from the ranks the weighted SVD
    # search chose. Over seeds 2000-2099: 89 / 11 / 0, the same final ranks
    # as a search from ones on every seed (89 / 11 / 0) at the cost of the
    # unweighted seed, whose search ends 52 / 48 / 0; the 11 are the MMLE
    # AIC's own over-selections (they recur from the true ranks with (M38a);
    # 4 with the reference count (M38)). On the 10 seeds below, 9 exact and
    # 1 over; never under.
    exact = 0
    for seed in range(2000, 2010):
        sim = _demo_noise(seed)
        stats = sufficient_statistics(sim.Y_masked, sim.X, sim.mask)
        _, seed_search = _svd_search(sim, [1, 1, 1], precision_weighted=True)
        with warnings.catch_warnings():
            # A candidate may hit the refinement's 10-iteration cap: 1 of 436
            # offline.
            warnings.simplefilter("ignore", ConvergenceWarning)
            _, h = greedy_aic(
                functools.partial(fit_mmle, stats),
                lambda f, r: f.aic,
                seed_search.ranks[-1],
                15,
            )
        assert h.estimator == "mmle"
        chosen = h.ranks[-1]
        assert (chosen >= [2, 1, 3]).all()
        exact += int((chosen == [2, 1, 3]).all())
    assert exact >= 7


def test_margins_and_near_ties() -> None:
    # The decision margin of every round, and the near ties.
    _, capped = greedy_aic(lambda r: None, lambda f, r: -float(r[0]), [1], 3)
    m = capped.margins()
    assert m["improvement"][:2].tolist() == [1.0, 1.0]
    assert np.isnan(m["improvement"][-1])
    assert capped.near_ties() == [
        "round 0 (accepted x0+1): gained 1 beyond the threshold",
        "round 1 (accepted x0+1): gained 1 beyond the threshold",
    ]
    scores = {(2, 1): 9.0, (1, 2): 9.5, (1, 1): 10.0, (3, 1): 9.4, (2, 2): 9.45}
    _, h = greedy_aic(
        lambda r: None,
        lambda f, r: scores.get((int(r[0]), int(r[1])), 20.0),
        [1, 1],
        5,
        threshold=0.1,
    )
    m = h.margins()
    np.testing.assert_allclose(m["decision"], [0.9, -0.5])
    np.testing.assert_allclose(m["runner_up"], [0.5, 0.05])
    lines = h.near_ties(within=1.0)
    assert lines == [
        "round 0 (accepted x0+1): gained 0.9 beyond the threshold",
        "round 0: x0+1 beat the runner-up by 0.5",
        "round 1 (rejected): best move x0+1 missed acceptance by 0.5",
        # A rejected round says no move was accepted.
        "round 1 (no move accepted): best move x0+1 beat the runner-up by 0.05 "
        "and missed acceptance by 0.5",
    ]
    assert h.near_ties(within=0.0) == []
    with pytest.raises(ParameterError, match="within"):
        h.near_ties(within=-1.0)
