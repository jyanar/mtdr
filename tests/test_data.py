"""Data utilities (`mtdr.data`): validate_inputs, check_design, condition_average,
stack_sessions and split_trials.

Each function gets a recovery-style test on simulated data (it reproduces a
quantity computed independently), property tests with hypothesis (invariances
of the layout and of the mask), and tests of every documented error.
"""

from __future__ import annotations

import pickle
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import mtdr
from mtdr import (
    check_design,
    condition_average,
    simulate,
    split_trials,
    stack_sessions,
    validate_inputs,
)
from mtdr.errors import DesignWarning, ParameterError, ValidationError
from mtdr.svd_fit import fit_svd

# ----------------------------------------------------------------- validate_inputs


def test_validate_inputs_infers_the_mask_from_nan() -> None:
    sim = simulate(n_neurons=8, n_bins=4, n_trials=30, ranks=[1], drop_prob=0.3, seed=1)
    Y, X, mask = validate_inputs(sim.Y_masked, sim.X)
    np.testing.assert_array_equal(mask, sim.mask)
    assert Y.dtype == np.float64
    assert X is not None
    assert X.dtype == np.float64
    assert Y.shape == sim.Y.shape  # never drops trials


def test_validate_inputs_keeps_float64_without_a_copy() -> None:
    Y = np.ones((3, 2, 2))
    Y_out, _, _ = validate_inputs(Y)
    assert Y_out is Y


def test_integers_and_bools_are_converted() -> None:
    Y = np.ones((3, 2, 2), dtype=np.int32)
    X = np.array([[True], [False], [True]])
    Y_out, X_out, mask = validate_inputs(Y, X, np.ones((3, 2), dtype=int))
    assert Y_out.dtype == np.float64
    assert X_out is not None
    assert X_out.tolist() == [[1.0], [0.0], [1.0]]
    assert mask.dtype == np.bool_
    assert mask.all()


def test_a_given_mask_wins_over_finite_data_but_not_over_nan() -> None:
    Y = np.ones((3, 2, 2))
    mask = np.array([[True, False], [True, True], [False, True]])
    _, _, out = validate_inputs(Y, None, mask)
    np.testing.assert_array_equal(out, mask)
    Y[0, 1, 0] = np.nan  # unobserved: ignored
    validate_inputs(Y, None, mask)
    Y[1, 0, 1] = np.nan  # observed: an error
    with pytest.raises(ValidationError, match=r"NaN or Inf where mask is True"):
        validate_inputs(Y, None, mask)


def test_inf_is_an_error_without_a_mask() -> None:
    Y = np.ones((2, 2, 2))
    Y[0, 0, 0] = np.inf
    with pytest.raises(ValidationError, match="Inf"):
        validate_inputs(Y)


@pytest.mark.parametrize(
    ("Y", "match"),
    [
        (np.ones((3, 2)), "3-D"),
        (np.ones((0, 2, 2)), "non-empty"),
        (np.array([[["a"]]]), "real numeric"),
        (object(), "real numeric"),
    ],
)
def test_bad_y(Y: Any, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        validate_inputs(Y)


def test_a_textual_x_gets_a_coding_recipe() -> None:
    # The message says how to encode a factor.
    X = np.array([["left"], ["right"], ["left"]])
    with pytest.raises(ValidationError, match=r"got dtype <U5; encode factors") as err:
        validate_inputs(np.ones((3, 2, 2)), X)
    assert "indicators with one level dropped" in str(err.value)
    with pytest.raises(ValidationError, match=r"dtype <U1$"):
        validate_inputs(np.array([[["a"]]]))  # no recipe for Y


def test_the_docs_session_recipe_runs(capsys: pytest.CaptureFixture[str]) -> None:
    # docs/api/data.md's recipe for non-simultaneous sessions that share one
    # temporal model runs as written.
    page = Path(__file__).resolve().parents[1] / "docs" / "api" / "data.md"
    if not page.is_file():  # the wheel job copies tests/ alone
        pytest.skip("docs/ is not next to the tests")
    match = re.search(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.S)
    assert match is not None
    namespace: dict[str, Any] = {}
    with warnings.catch_warnings():
        # A rounding-level inner end is not this recipe's subject.
        warnings.simplefilter("ignore", mtdr.ConvergenceWarning)
        exec(compile(match.group(1), str(page), "exec"), namespace)
    stacked = namespace["stacked"]
    assert stacked.Y.shape == (360, 120, 8)
    assert int(stacked.mask.sum(axis=1).max()) <= 40
    assert namespace["model"].n_neurons_ == 120
    assert capsys.readouterr().out.startswith("{'x0': 2, 'x1': 1, 'x2': 2} 120")


def test_bad_y_that_is_not_an_array() -> None:
    with pytest.raises(ValidationError, match="ragged or not an array"):
        validate_inputs([[[1.0]], [[1.0, 2.0]]])


def test_the_matlab_layout_is_named() -> None:
    with pytest.raises(ValidationError, match="MATLAB"):
        validate_inputs(np.ones((5, 4, 3)), np.ones((3, 1)))


@pytest.mark.parametrize(
    ("X", "match"),
    [
        (np.ones(3), "2-D"),
        (np.ones((3, 0)), "non-empty"),
        (np.array([[1.0], [np.nan], [0.0]]), "finite"),
    ],
)
def test_bad_x(X: Any, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        validate_inputs(np.ones((3, 2, 2)), X)


@pytest.mark.parametrize(
    ("mask", "match"),
    [
        (np.ones((3, 3)), "shape"),
        (np.full((3, 2), 2), "0 and 1"),
        (np.full((3, 2), "a"), "0 and 1"),
        (np.ones(6), "shape"),
    ],
)
def test_bad_mask(mask: Any, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        validate_inputs(np.ones((3, 2, 2)), None, mask)


def test_mask_that_is_not_an_array() -> None:
    with pytest.raises(ValidationError, match="not an array"):
        validate_inputs(np.ones((2, 2, 2)), None, [[1, 0], [1]])


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"for_fit": 1}, "for_fit"),
        ({"condition_independent": "yes"}, "condition_independent"),
        ({"min_observations": 0}, "min_observations"),
        ({"min_observations": 2.5}, "min_observations"),
    ],
)
def test_bad_arguments(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ParameterError, match=match):
        validate_inputs(np.ones((2, 2, 2)), **kw)


def _fit_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim = simulate(n_neurons=10, n_bins=4, n_trials=60, ranks=[1, 1], seed=seed)
    return np.array(sim.Y), np.array(sim.X), np.array(sim.mask)


def test_fit_checks_reject_few_trial_neurons() -> None:
    Y, X, mask = _fit_data()
    mask[:, 3] = False
    mask[:2, 3] = True
    validate_inputs(Y, X, mask, for_fit=True)  # two trials: the default floor
    with pytest.raises(ValidationError, match=r"neurons \[3\] are observed on fewer"):
        validate_inputs(Y, X, mask, for_fit=True, min_observations=3)
    mask[:, 3] = False
    with pytest.raises(ValidationError, match=r"neurons \[3\]"):
        validate_inputs(Y, X, mask, for_fit=True)
    validate_inputs(Y, X, mask)  # allowed at evaluation


def test_fit_checks_reject_zero_variance_neurons() -> None:
    Y, X, mask = _fit_data()
    Y[:, 4, :] = 3.0
    with pytest.raises(ValidationError, match=r"neurons \[4\] take one value"):
        validate_inputs(Y, X, mask, for_fit=True)


def test_fit_checks_reject_constant_and_collinear_columns() -> None:
    Y, X, mask = _fit_data()
    const = np.column_stack([X[:, 0], np.full(len(X), 2.0)])
    with pytest.raises(ValidationError, match="constant: collinear with the intercept"):
        validate_inputs(Y, const, mask, for_fit=True)
    validate_inputs(Y, const, mask, for_fit=True, condition_independent=False)
    dup = np.column_stack([X[:, 0], 3 * X[:, 0]])
    with pytest.raises(ValidationError, match="rank 1 < 2"):
        validate_inputs(Y, dup, mask, for_fit=True)
    # Indicators of every level of a binary variable sum to one.
    ind = np.column_stack([X[:, 1] > 0, X[:, 1] <= 0]).astype(float)
    with pytest.raises(ValidationError, match="drop one level"):
        validate_inputs(Y, ind, mask, for_fit=True)
    validate_inputs(Y, ind, mask, for_fit=True, condition_independent=False)


def test_fit_checks_warn_about_empty_trials() -> None:
    Y, X, mask = _fit_data()
    mask[5] = False
    with pytest.warns(DesignWarning, match=r"trials \[5\] have no observed neuron"):
        validate_inputs(Y, X, mask, for_fit=True)


def test_rank_tests_are_scale_and_shift_invariant() -> None:
    Y, X, mask = _fit_data()
    validate_inputs(Y, X * [1e-8, 1e8] + [1e6, -1e5], mask, for_fit=True)


@settings(deadline=None)
@given(
    data=hnp.arrays(np.float64, (4, 3, 2), elements=st.floats(-1e6, 1e6)),
    holes=hnp.arrays(np.bool_, (4, 3)),
)
def test_property_nan_holes_are_the_mask(data: np.ndarray, holes: np.ndarray) -> None:
    Y = data.copy()
    Y[holes, 0] = np.nan
    _, _, mask = validate_inputs(Y)
    np.testing.assert_array_equal(mask, ~holes)
    # The returned Y keeps the holes; a given equal mask gives the same result.
    _, _, again = validate_inputs(Y, None, ~holes)
    np.testing.assert_array_equal(again, mask)


# ----------------------------------------------------------------- check_design


def test_check_design_without_a_mask() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = np.column_stack([a, 2 * a + 1e-9 * rng.normal(size=200), a + 5, np.ones(200)])
    report = check_design(X, regressor_names=["a", "b", "c", "one"])
    assert report.n_trials == 200
    assert report.n_regressors == 4
    assert report.constant.tolist() == [False, False, False, True]
    # a, 2a (+ 1e-9 noise, below svd_fit.RANK_TOLERANCE's test), a + 5 vs 1
    assert report.rank == 2
    assert report.augmented_rank == 2
    assert (0, 1) in report.duplicate_columns
    assert report.condition_number == np.inf
    assert report.max_abs_correlation == pytest.approx(1.0)
    assert any("rank" in p for p in report.problems)
    assert any("'one'" in p for p in report.problems)
    assert any("near-collinear" in w for w in report.warnings)
    assert any("not centred" in n for n in report.notes)
    assert report.few_trial_neurons.size == 0
    assert report.empty_trials.size == 0
    text = str(report)
    assert "DesignReport: 200 trials" in text
    assert "problems:" in text
    with pytest.raises(ValueError, match="read-only"):
        report.means[0] = 1.0
    again = pickle.loads(pickle.dumps(report))
    assert not again.means.flags.writeable


def test_check_design_clean() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    report = check_design(X)
    assert report.problems == []
    assert report.warnings == []
    assert report.rank == 3
    assert report.augmented_rank == 4
    assert report.duplicate_columns == []
    assert 1.0 <= report.condition_number < 10
    assert "none" in str(report)


def test_check_design_scale_is_a_note_or_a_warning() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 2)) * [1.0, 1000.0]
    assert any("invariant" in n for n in check_design(X).notes)
    report = check_design(X, basis_ridge=0.1)
    assert any("basis_ridge > 0" in w for w in report.warnings)
    assert not any("factor" in n for n in report.notes)


def test_check_design_mirrors_fit_svd_per_neuron() -> None:
    # A regressor constant over one neuron's trials (a per-session constant):
    # fit_svd's minimum-norm neurons are exactly check_design's.
    sim = simulate(n_neurons=6, n_bins=4, n_trials=80, ranks=[1, 1], seed=4)
    X = np.array(sim.X)
    mask = np.array(sim.mask)
    mask[:, 2] = X[:, 0] == 2  # neuron 2 sees x0 at one level only
    mask[0, :] = False  # an empty trial
    stats = mtdr.stats.sufficient_statistics(sim.Y, X, mask)
    with pytest.warns(DesignWarning):
        svd = fit_svd(stats, [1, 1])
    report = check_design(X, mask, min_observations=50)
    np.testing.assert_array_equal(
        report.rank_deficient_neurons, svd.rank_deficient_neurons
    )
    assert 2 in report.rank_deficient_neurons.tolist()
    assert report.empty_trials.tolist() == [0]
    assert report.few_trial_neurons.tolist() == [2]
    assert any("observed on fewer than 50" in p for p in report.problems)
    assert report.warnings[0].startswith("trials [0]")
    assert check_design(X, mask, ridge=1.0).rank_deficient_neurons.size == 0


def _floor_mask(counts: list[int], n_trials: int = 40) -> np.ndarray:
    # Neuron i observed on its first counts[i] trials; one extra neuron on all,
    # so every trial has an observed neuron.
    mask = np.zeros((n_trials, len(counts) + 1), dtype=bool)
    for i, c in enumerate(counts):
        mask[:c, i] = True
    mask[:, -1] = True
    return mask


@pytest.mark.parametrize(
    ("estimator", "ci", "min_obs", "floor"),
    [
        (None, True, 2, 2),  # no estimator: the min_observations given
        ("svd", True, 2, 2),
        ("mmle", True, 2, 5),  # P + 2 with three regressors and the intercept
        ("mmle", False, 2, 4),  # P + 1 without the intercept
        ("mmle", True, 7, 7),  # a larger min_observations wins
        ("svd", False, 6, 6),
    ],
)
def test_check_design_applies_the_estimators_floor(
    estimator: Any, ci: bool, min_obs: int, floor: int
) -> None:
    # check_design(estimator=...) applies the observation floor MTDR.fit
    # applies, max(min_observations, P + 1 + 1_intercept) under "mmle", below
    # which a neuron's marginal likelihood is unbounded; without an estimator
    # it applies min_observations.
    X = np.random.default_rng(0).normal(size=(40, 3))
    counts = [1, 2, 3, 4, 5, 6, 7, 8]
    report = check_design(
        X,
        _floor_mask(counts),
        condition_independent=ci,
        min_observations=min_obs,
        estimator=estimator,
    )
    expected = [i for i, c in enumerate(counts) if c < floor]
    assert report.few_trial_neurons.tolist() == expected
    assert any(f"fewer than {floor} trials" in p for p in report.problems)


@pytest.mark.parametrize("ci", [True, False])
@pytest.mark.parametrize("n_obs", [3, 4, 5, 6])
def test_check_design_floor_agrees_with_fit(ci: bool, n_obs: int) -> None:
    # The floor is one calculation shared with MTDR.fit: check_design flags a
    # neuron exactly when fit rejects it, with the same wording.
    sim = simulate(
        n_neurons=6,
        n_bins=4,
        n_trials=60,
        ranks=[1, 1, 1],
        condition_independent=ci,
        noise_precision=1.0,
        seed=2,
    )
    mask = np.ones((60, 6), dtype=bool)
    mask[n_obs:, 0] = False
    report = check_design(sim.X, mask, condition_independent=ci, estimator="mmle")
    model = mtdr.MTDR(ranks=[1, 1, 1], condition_independent=ci)
    if report.few_trial_neurons.size:
        assert report.few_trial_neurons.tolist() == [0]
        with pytest.raises(ValidationError) as err:
            model.fit(sim.Y, sim.X, mask)
        # The fit error is the report's problem, then the observed counts.
        assert str(err.value).startswith(report.problems[-1])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DesignWarning)
            model.fit(sim.Y, sim.X, mask)
    assert (report.few_trial_neurons.size > 0) == (n_obs < 4 + int(ci))


def test_check_design_estimator_errors() -> None:
    with pytest.raises(ParameterError, match="estimator"):
        check_design(np.ones((3, 1)), estimator="ols")  # type: ignore[arg-type]
    with pytest.raises(ParameterError, match="estimator"):
        check_design(np.ones((3, 1)), estimator=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "kw", "error"),
    [
        ((np.ones(3),), {}, ValidationError),
        ((np.array([[np.inf]]),), {}, ValidationError),
        ((np.ones((3, 1)),), {"regressor_names": ["a", "b"]}, ParameterError),
        ((np.ones((3, 1)),), {"basis_ridge": -1.0}, ParameterError),
        ((np.ones((3, 1)),), {"ridge": np.nan}, ParameterError),
        ((np.ones((3, 1)),), {"min_observations": 0}, ParameterError),
        ((np.ones((3, 1)),), {"condition_independent": None}, ParameterError),
        ((np.ones((3, 1)), np.ones((2, 2))), {}, ValidationError),
    ],
)
def test_check_design_errors(
    args: tuple[Any, ...], kw: dict[str, Any], error: type
) -> None:
    with pytest.raises(error):
        check_design(*args, **kw)


def test_check_design_one_column_has_no_correlation() -> None:
    report = check_design(np.arange(5.0)[:, None])
    assert report.max_abs_correlation == 0.0
    assert report.condition_number == 1.0


def test_check_design_uses_the_observed_trials() -> None:
    X = np.array([[1.0], [2.0], [5.0]])
    mask = np.array([[True], [True], [False]])
    report = check_design(X, mask)
    assert report.means.tolist() == [1.5]
    # With no observed trial at all, every row is used.
    assert check_design(X, np.zeros((3, 1))).means.tolist() == [pytest.approx(8 / 3)]


@settings(deadline=None)
@given(
    X=hnp.arrays(np.float64, (12, 3), elements=st.integers(-10, 10).map(float)),
    scale=st.integers(-10, 10).map(lambda e: 2.0**e),
    shift=st.integers(-1000, 1000).map(float),
)
def test_property_check_design_is_scale_and_shift_invariant(
    X: np.ndarray, scale: float, shift: float
) -> None:
    # Exact on a grid of integers and powers of two (no rounding to blame).
    a = check_design(X, condition_independent=False)
    b = check_design(X * scale + shift, condition_independent=False)
    assert b.augmented_rank == a.augmented_rank
    assert b.constant.tolist() == a.constant.tolist()


# ----------------------------------------------------------------- condition_average


def test_condition_average_against_a_loop() -> None:
    sim = simulate(
        n_neurons=7,
        n_bins=3,
        n_trials=90,
        ranks=[1, 1],
        drop_prob=0.4,
        seed=5,
        levels=[[-1, 0, 1], [-1, 1]],
    )
    avg = condition_average(sim.Y_masked, sim.X, by=["x0"], min_trials=2)
    keys = np.unique(sim.X[:, 0])
    assert avg.Y.shape == (keys.size, 7, 3)
    for c, key in enumerate(keys):
        rows = sim.X[:, 0] == key
        np.testing.assert_allclose(avg.X[c], sim.X[rows].mean(axis=0))
        for i in range(7):
            obs = rows & sim.mask[:, i]
            assert avg.counts[c, i] == obs.sum()
            if obs.sum() >= 2:
                assert avg.mask[c, i]
                np.testing.assert_allclose(avg.Y[c, i], sim.Y[obs, i].mean(axis=0))
            else:
                assert not avg.mask[c, i]
                assert np.isnan(avg.Y[c, i]).all()
    np.testing.assert_array_equal(keys[avg.condition_of_trial], sim.X[:, 0])
    assert "ConditionAverage(n_conditions=3" in repr(avg)
    assert not pickle.loads(pickle.dumps(avg)).Y.flags.writeable


def test_condition_average_by_index_and_all_columns() -> None:
    X = np.array([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])
    Y = np.arange(4.0).reshape(4, 1, 1)
    full = condition_average(Y, X)
    assert full.X.tolist() == [[-1.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert full.Y.ravel().tolist() == [2.0, 1.5, 1.0]
    by0 = condition_average(Y, X, by=[0])
    np.testing.assert_allclose(by0.X, [[-1.0, 0.0], [1.0, 1 / 3]])


def test_condition_average_keeps_rows_below_the_floor() -> None:
    X = np.array([[1.0], [-1.0], [-1.0]])
    Y = np.ones((3, 2, 1))
    mask = np.array([[True, False], [True, True], [True, True]])
    avg = condition_average(Y, X, mask, min_trials=2)
    assert avg.mask.tolist() == [[True, True], [False, False]]
    assert np.isnan(avg.Y[1]).all()


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"by": ["nope"]}, "unknown regressor"),
        ({"by": [3]}, "not a column index"),
        ({"by": [0, 0]}, "repeats"),
        ({"by": []}, "non-empty"),
        ({"by": [1.5]}, "not a column index"),
        ({"min_trials": 0}, "min_trials"),
    ],
)
def test_condition_average_errors(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ParameterError, match=match):
        condition_average(np.ones((3, 1, 1)), np.ones((3, 2)) * [[1], [2], [3]], **kw)


@settings(deadline=None)
@given(perm=st.permutations(list(range(8))))
def test_property_condition_average_ignores_trial_order(perm: list[int]) -> None:
    sim = simulate(n_neurons=3, n_bins=2, n_trials=8, ranks=[1], drop_prob=0.3, seed=6)
    a = condition_average(sim.Y_masked, sim.X)
    b = condition_average(sim.Y_masked[perm], sim.X[perm])
    np.testing.assert_allclose(a.Y, b.Y, equal_nan=True)
    np.testing.assert_array_equal(a.counts, b.counts)


# ----------------------------------------------------------------- stack_sessions


def test_stack_sessions_round_trip() -> None:
    sims = [
        simulate(n_neurons=n, n_bins=4, n_trials=k, ranks=[1, 1], seed=s, drop_prob=0.2)
        for s, (n, k) in enumerate([(3, 20), (5, 15), (2, 30)])
    ]
    st_ = stack_sessions(
        [(s.Y, s.X, s.mask) for s in sims[:2]] + [(sims[2].Y_masked, sims[2].X)]
    )
    assert st_.Y.shape == (65, 10, 4)
    assert st_.X.shape == (65, 2)
    for s, sim in enumerate(sims):
        rows = st_.session_of_trial == s
        cols = st_.session_of_neuron == s
        np.testing.assert_array_equal(st_.mask[np.ix_(rows, cols)], sim.mask)
        assert not st_.mask[np.ix_(rows, ~cols)].any()
        assert np.isnan(st_.Y[np.ix_(rows, ~cols)]).all()
        block = st_.Y[np.ix_(rows, cols)]
        np.testing.assert_array_equal(block[sim.mask], np.array(sim.Y)[sim.mask])
        np.testing.assert_array_equal(st_.X[rows], sim.X)
        assert st_.trial_index_in_session[rows].tolist() == list(range(rows.sum()))
        assert st_.neuron_index_in_session[cols].tolist() == list(range(cols.sum()))
    assert "StackedSessions(n_sessions=3" in repr(st_)
    assert not pickle.loads(pickle.dumps(st_)).mask.flags.writeable
    # The stacked data fit: every neuron has its own trials (M3).
    _, _, mask = validate_inputs(st_.Y, st_.X, st_.mask, for_fit=True)
    assert mask.sum() == sum(int(s.mask.sum()) for s in sims)


@pytest.mark.parametrize(
    ("sessions", "error", "match"),
    [
        ([], ParameterError, "non-empty"),
        ("abc", ParameterError, "non-empty"),
        ([(np.ones((2, 1, 1)),)], ParameterError, "tuple"),
        ([(np.ones((2, 1, 1)), np.ones((3, 1)))], ValidationError, r"sessions\[0\]"),
        (
            [
                (np.ones((2, 1, 1)), np.ones((2, 1))),
                (np.ones((2, 1, 2)), np.ones((2, 1))),
            ],
            ValidationError,
            "same time bins",
        ),
    ],
)
def test_stack_sessions_errors(sessions: Any, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        stack_sessions(sessions)


# ----------------------------------------------------------------- split_trials


def test_split_trials_partition() -> None:
    train, test = split_trials(50, test_fraction=0.2, random_state=1)
    assert len(test) == 10
    assert len(train) == 40
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(50))
    assert train.dtype == np.int64
    assert (np.diff(train) > 0).all()
    again = split_trials(50, test_fraction=0.2, random_state=np.random.default_rng(1))
    np.testing.assert_array_equal(again[1], test)


def test_split_trials_stratified() -> None:
    labels = np.repeat([0, 1, 2, 3], [20, 10, 2, 1])
    with pytest.warns(DesignWarning, match=r"labels \[3\] have a single trial"):
        train, test = split_trials(
            33, test_fraction=0.3, random_state=0, stratify=labels
        )
    counts = np.bincount(labels[test], minlength=4)
    assert counts.tolist() == [6, 3, 1, 0]
    assert 32 in train.tolist()


def test_split_trials_many_singletons_are_summarised() -> None:
    labels = np.arange(25)
    with pytest.warns(DesignWarning, match="and 5 more"):
        train, test = split_trials(25, stratify=labels, random_state=0)
    assert test.size == 0
    assert train.size == 25


@pytest.mark.parametrize(
    ("args", "kw", "match"),
    [
        ((1,), {}, "n_trials"),
        ((10,), {"test_fraction": 1.0}, "test_fraction"),
        ((10,), {"test_fraction": "a"}, "test_fraction"),
        ((10,), {"random_state": 1.5}, "random_state"),
        ((10,), {"random_state": -1}, "random_state"),  # not NumPy's ValueError
        ((10,), {"stratify": np.zeros(9)}, "stratify"),
    ],
)
def test_split_trials_errors(
    args: tuple[Any, ...], kw: dict[str, Any], match: str
) -> None:
    with pytest.raises(ParameterError, match=match):
        split_trials(*args, **kw)


@settings(deadline=None)
@given(
    n=st.integers(2, 200),
    fraction=st.floats(0.01, 0.99),
    seed=st.integers(0, 2**32 - 1),
)
def test_property_split_is_a_partition(n: int, fraction: float, seed: int) -> None:
    train, test = split_trials(n, test_fraction=fraction, random_state=seed)
    assert len(train) >= 1
    assert len(test) >= 1
    assert np.intersect1d(train, test).size == 0
    assert np.union1d(train, test).tolist() == list(range(n))


@settings(deadline=None)
@given(
    labels=hnp.arrays(np.int64, st.integers(2, 60), elements=st.integers(0, 4)),
    seed=st.integers(0, 1000),
)
def test_property_stratified_split_partitions(labels: np.ndarray, seed: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DesignWarning)
        train, test = split_trials(labels.size, stratify=labels, random_state=seed)
    assert np.union1d(train, test).tolist() == list(range(labels.size))
    for value in np.unique(labels):
        members = np.flatnonzero(labels == value)
        in_test = np.isin(members, test).sum()
        assert in_test <= max(members.size - 1, 0)
        if members.size >= 2:
            assert in_test >= 1


def test_fit_checks_without_a_design() -> None:
    Y, _, mask = _fit_data()
    out, X, _ = validate_inputs(Y, None, mask, for_fit=True)
    assert X is None
    assert out.shape == Y.shape


def test_check_design_with_a_clean_mask() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 2))
    report = check_design(X, np.ones((60, 4), dtype=bool), basis_ridge=0.1)
    assert report.warnings == []
    assert report.rank_deficient_neurons.size == 0
    assert report.empty_trials.size == 0
