"""Tests for `mtdr.stats`: definitions, mask handling, properties, validation."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

import mtdr
from mtdr.errors import ParameterError, ValidationError
from mtdr.stats import SufficientStats, sufficient_statistics

FloatArray = NDArray[np.float64]


def _random_problem(
    seed: int, N: int, n: int, T: int, P: int, p_obs: float = 0.7
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(N, n, T)) * rng.uniform(0.5, 3.0, size=(1, n, 1)) + 2.0
    X = rng.normal(size=(N, P))
    mask = rng.random((N, n)) < p_obs
    return Y, X, mask


def _naive(Y: FloatArray, X: FloatArray, mask: NDArray[np.bool_]) -> dict[str, Any]:
    """Per-neuron loop over the observed trials: the definitions (M6), (M10)-(M12)."""
    n = Y.shape[1]
    keys = ("XtX", "XtY", "YtY", "XtX_c", "XtY_c", "YtY_c", "n", "xm", "ym")
    out: dict[str, list[Any]] = {k: [] for k in keys}
    for i in range(n):
        rows = mask[:, i]
        Xi, Yi = X[rows], Y[rows, i, :]
        out["XtX"].append(Xi.T @ Xi)
        out["XtY"].append(Xi.T @ Yi)
        out["YtY"].append(float((Yi**2).sum()))
        Xc = Xi - Xi.mean(axis=0) if rows.any() else Xi
        Yc = Yi - Yi.mean(axis=0) if rows.any() else Yi
        out["XtX_c"].append(Xc.T @ Xc)
        out["XtY_c"].append(Xc.T @ Yc)
        out["YtY_c"].append(float((Yc**2).sum()))
        out["n"].append(int(rows.sum()))
        out["xm"].append(Xi.mean(axis=0) if rows.any() else np.full(X.shape[1], np.nan))
        out["ym"].append(Yi.mean(axis=0) if rows.any() else np.full(Y.shape[2], np.nan))
    return {k: np.array(v) for k, v in out.items()}


FIELDS = ("XtX_c", "XtY_c", "YtY_c", "n_obs", "X_mean", "Y_mean")
VIEWS = ("XtX", "XtY_raw", "YtY_raw")


def _assert_stats_equal(a: SufficientStats, b: SufficientStats) -> None:
    for name in ("XtX_c", "XtY_c", "YtY_c", *VIEWS, "X_mean", "Y_mean"):
        np.testing.assert_allclose(
            getattr(a, name), getattr(b, name), rtol=1e-12, atol=1e-12
        )
    np.testing.assert_array_equal(a.n_obs, b.n_obs)


sizes = st.tuples(
    st.integers(1, 7), st.integers(1, 4), st.integers(1, 4), st.integers(1, 3)
)


# ------------------------------------------------------------------ definitions


@given(sizes=sizes, seed=st.integers(0, 2**32 - 1))
def test_matches_a_per_neuron_loop(sizes: tuple[int, int, int, int], seed: int) -> None:
    N, n, T, P = sizes
    Y, X, mask = _random_problem(seed, N, n, T, P)
    stats = sufficient_statistics(Y, X, mask)
    ref = _naive(Y, X, mask)
    np.testing.assert_allclose(stats.XtX, ref["XtX"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stats.XtY_raw, ref["XtY"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stats.YtY_raw, ref["YtY"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(stats.n_obs, ref["n"])
    np.testing.assert_allclose(stats.X_mean, ref["xm"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stats.Y_mean, ref["ym"], rtol=1e-12, atol=1e-12)
    for name in ("XtX_c", "XtY_c", "YtY_c"):
        np.testing.assert_allclose(getattr(stats, name), ref[name], atol=1e-11)
    assert (stats.n_regressors, stats.n_bins, stats.n_neurons) == (P, T, n)
    np.testing.assert_array_equal(stats.XtX, stats.XtX.transpose(0, 2, 1))
    np.testing.assert_array_equal(stats.XtX_c, stats.XtX_c.transpose(0, 2, 1))


@given(sizes=sizes, seed=st.integers(0, 2**32 - 1))
def test_unobserved_entries_are_ignored(
    sizes: tuple[int, int, int, int], seed: int
) -> None:
    # Whatever sits under a False mask (NaN, Inf, huge values) never enters a
    # statistic.
    N, n, T, P = sizes
    Y, X, mask = _random_problem(seed, N, n, T, P)
    garbage = Y.copy()
    rng = np.random.default_rng(seed)
    fill = rng.choice([np.nan, np.inf, -np.inf, 1e300, 0.0], size=Y.shape)
    garbage[~mask] = fill[~mask]
    _assert_stats_equal(
        sufficient_statistics(garbage, X, mask), sufficient_statistics(Y, X, mask)
    )


@given(sizes=sizes, seed=st.integers(0, 2**32 - 1))
def test_all_ones_mask_equals_the_unmasked_formulas(
    sizes: tuple[int, int, int, int], seed: int
) -> None:
    N, n, T, P = sizes
    Y, X, _ = _random_problem(seed, N, n, T, P)
    stats = sufficient_statistics(Y, X, np.ones((N, n), dtype=bool))
    np.testing.assert_allclose(
        stats.XtX, np.broadcast_to(X.T @ X, (n, P, P)), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        stats.XtY_raw, np.einsum("kp,kit->ipt", X, Y), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(stats.YtY_raw, (Y**2).sum(axis=(0, 2)), rtol=1e-12)
    np.testing.assert_array_equal(stats.n_obs, N)
    np.testing.assert_allclose(stats.X_mean, np.broadcast_to(X.mean(0), (n, P)))
    np.testing.assert_allclose(stats.Y_mean, Y.mean(axis=0))


@given(sizes=sizes, seed=st.integers(0, 2**32 - 1), data=st.data())
def test_neuron_permutation_equivariance(
    sizes: tuple[int, int, int, int], seed: int, data: st.DataObject
) -> None:
    N, n, T, P = sizes
    Y, X, mask = _random_problem(seed, N, n, T, P)
    perm = np.array(data.draw(st.permutations(range(n))))
    a = sufficient_statistics(Y, X, mask)
    b = sufficient_statistics(Y[:, perm], X, mask[:, perm])
    for name in (*FIELDS, *VIEWS):
        np.testing.assert_allclose(getattr(b, name), getattr(a, name)[perm], rtol=1e-12)


@given(sizes=sizes, seed=st.integers(0, 2**32 - 1), data=st.data())
def test_trial_permutation_invariance(
    sizes: tuple[int, int, int, int], seed: int, data: st.DataObject
) -> None:
    N, n, T, P = sizes
    Y, X, mask = _random_problem(seed, N, n, T, P)
    perm = np.array(data.draw(st.permutations(range(N))))
    _assert_stats_equal(
        sufficient_statistics(Y[perm], X[perm], mask[perm]),
        sufficient_statistics(Y, X, mask),
    )


# ------------------------------------------------------------------ centred views


@given(sizes=sizes, seed=st.integers(0, 2**32 - 1))
def test_centered_equals_statistics_of_shifted_data(
    sizes: tuple[int, int, int, int], seed: int
) -> None:
    # centered(b) must equal the statistics of Y - b.
    N, n, T, P = sizes
    Y, X, mask = _random_problem(seed, N, n, T, P)
    b = np.random.default_rng(seed + 1).normal(size=(n, T)) * 3.0
    XtY, YtY = sufficient_statistics(Y, X, mask).centered(b)
    direct = sufficient_statistics(Y - b[None], X, mask)
    np.testing.assert_allclose(XtY, direct.XtY_raw, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(YtY, direct.YtY_raw, rtol=1e-10, atol=1e-10)


def test_centered_none_returns_the_raw_arrays() -> None:
    Y, X, mask = _random_problem(0, 6, 3, 4, 2)
    stats = sufficient_statistics(Y, X, mask)
    XtY, YtY = stats.centered(None)
    assert XtY is stats.XtY_raw
    assert YtY is stats.YtY_raw


def test_centered_at_the_mean_is_the_sum_of_squares_about_it() -> None:
    Y, X, mask = _random_problem(1, 30, 4, 5, 2, p_obs=0.8)
    stats = sufficient_statistics(Y, X, mask)
    _, YtY = stats.centered(stats.Y_mean)
    for i in range(4):
        Yi = Y[mask[:, i], i, :]
        np.testing.assert_allclose(YtY[i], ((Yi - Yi.mean(0)) ** 2).sum(), rtol=1e-10)


@pytest.mark.parametrize("baseline", [1e3, 1e7, 1e10])
def test_centred_moments_are_free_of_cancellation(baseline: float) -> None:
    # Accumulated about the means, the statistics of Y + c equal those of Y to
    # rounding, and centered(b) loses nothing to the baseline; the raw closed
    # form (M14) loses about log10(upsilon / upsilon(b)) digits (six at
    # c = 1e3). `back` holds exactly the values the lifted data shift.
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(200, 3, 5))
    X = rng.normal(size=(200, 1))
    mask = rng.random((200, 3)) < 0.8
    lifted = Y + baseline
    back = lifted - baseline
    a, b = sufficient_statistics(back, X, mask), sufficient_statistics(lifted, X, mask)
    np.testing.assert_allclose(b.YtY_c, a.YtY_c, rtol=1e-12)
    np.testing.assert_allclose(b.XtY_c, a.XtY_c, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(b.XtX_c, a.XtX_c)
    _, YtY = b.centered(b.Y_mean)
    np.testing.assert_allclose(YtY, a.YtY_c, rtol=1e-12)
    XtY, YtY = b.centered(b.Y_mean + 0.5)
    XtY_a, YtY_a = a.centered(a.Y_mean + 0.5)
    np.testing.assert_allclose(YtY, YtY_a, rtol=1e-12)
    np.testing.assert_allclose(XtY, XtY_a, rtol=1e-10, atol=1e-10)


def test_constant_columns_centre_to_exactly_zero() -> None:
    # A regressor or a response constant over a neuron's trials has exactly
    # zero centred moments, whatever its level: the mean of n copies of 0.1
    # need not be 0.1 in floating point, but the moments are taken about the
    # neuron's first observed value first.
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(30, 2, 4))
    X = rng.normal(size=(30, 2))
    mask = np.ones((30, 2), dtype=bool)
    mask[15:, 0] = False
    X[:15, 1] = 0.1
    Y[:, 1] = 1e7 + 0.3
    stats = sufficient_statistics(Y, X, mask)
    np.testing.assert_array_equal(stats.XtX_c[0, 1], 0.0)
    np.testing.assert_array_equal(stats.XtY_c[0, 1], 0.0)
    assert stats.YtY_c[1] == 0.0
    np.testing.assert_array_equal(stats.XtY_c[1], 0.0)
    assert stats.X_mean[0, 1] == 0.1
    np.testing.assert_array_equal(stats.Y_mean[1], 1e7 + 0.3)


def test_never_observed_neuron_is_kept_with_nan_means() -> None:
    Y, X, mask = _random_problem(3, 8, 3, 4, 2)
    mask[:, 1] = False
    stats = sufficient_statistics(Y, X, mask)
    assert stats.n_obs[1] == 0
    np.testing.assert_array_equal(stats.XtX[1], 0.0)
    np.testing.assert_array_equal(stats.XtY_raw[1], 0.0)
    assert stats.YtY_raw[1] == 0.0
    assert np.isnan(stats.X_mean[1]).all()
    assert np.isnan(stats.Y_mean[1]).all()
    XtY, YtY = stats.centered(np.ones((3, 4)))
    np.testing.assert_array_equal(XtY[1], 0.0)
    assert YtY[1] == 0.0
    assert np.isfinite(XtY).all()
    assert np.isfinite(YtY).all()


# ------------------------------------------------------------------ inputs


def test_dtypes_are_converted() -> None:
    rng = np.random.default_rng(4)
    counts = rng.poisson(3.0, size=(10, 3, 4))
    X = rng.integers(-2, 3, size=(10, 2))
    mask = (rng.random((10, 3)) < 0.8).astype(np.int8)
    stats = sufficient_statistics(counts, X, mask)
    ref = sufficient_statistics(counts.astype(float), X.astype(float), mask == 1)
    _assert_stats_equal(stats, ref)
    assert stats.XtX.dtype == stats.XtY_raw.dtype == np.float64
    assert stats.n_obs.dtype == np.int64
    # Boolean data and a float 0/1 mask are accepted too.
    spikes = counts > 2
    _assert_stats_equal(
        sufficient_statistics(spikes, X.astype(bool), mask.astype(float)),
        sufficient_statistics(
            spikes.astype(float), X.astype(bool).astype(float), mask == 1
        ),
    )


def test_nested_lists_are_accepted() -> None:
    Y = [[[1.0, 2.0]], [[3.0, 5.0]]]
    stats = sufficient_statistics(Y, [[1.0], [2.0]], [[True], [True]])
    np.testing.assert_allclose(stats.XtY_raw, [[[7.0, 12.0]]])


def test_result_is_read_only_pickles_and_summarises() -> None:
    Y, X, mask = _random_problem(5, 6, 3, 4, 2)
    stats = sufficient_statistics(Y, X, mask)
    for name in (*FIELDS, *VIEWS):
        with pytest.raises(ValueError, match="read-only"):
            getattr(stats, name)[...] = 0
    with pytest.raises(AttributeError):
        stats.n_bins = 3  # type: ignore[misc]
    again = pickle.loads(pickle.dumps(stats))
    _assert_stats_equal(again, stats)
    # Unpickling restores read-only arrays.
    for name in (*FIELDS, *VIEWS):
        assert not getattr(again, name).flags.writeable
    assert repr(stats).startswith(
        "SufficientStats(n_neurons=3, n_regressors=2, n_bins=4"
    )


def test_constructor_copies_and_checks() -> None:
    Y, X, mask = _random_problem(6, 6, 3, 4, 2)
    stats = sufficient_statistics(Y, X, mask)
    fields: dict[str, Any] = {name: np.array(getattr(stats, name)) for name in FIELDS}
    fields.update(n_regressors=2, n_bins=4)
    built = SufficientStats(**fields)
    fields["XtX_c"][...] = 99.0  # the constructor copied
    _assert_stats_equal(built, stats)
    assert SufficientStats(**{**fields, "n_regressors": np.int64(2)}).n_regressors == 2
    # NaN means are allowed for a neuron with no observed trial.
    unobserved = {**fields, "n_obs": np.array([0, *fields["n_obs"][1:]])}
    unobserved["Y_mean"] = np.array(fields["Y_mean"])
    unobserved["Y_mean"][0] = np.nan
    for name in ("XtX_c", "XtY_c", "YtY_c"):
        unobserved[name] = np.array(fields[name])
        unobserved[name][0] = 0.0
    hand_built = SufficientStats(**unobserved)
    assert np.isnan(hand_built.Y_mean[0]).all()
    assert hand_built.YtY_raw[0] == 0.0  # the raw views ignore the NaN means
    np.testing.assert_array_equal(hand_built.XtY_raw[0], 0.0)
    # Raw moments convert by subtracting the mean terms (the class docstring).
    n_obs = stats.n_obs[:, None, None]
    np.testing.assert_allclose(
        stats.XtX - n_obs * stats.X_mean[:, :, None] * stats.X_mean[:, None, :],
        stats.XtX_c,
        atol=1e-12,
    )


BAD_STATS: list[tuple[dict[str, Any], str]] = [
    ({"n_regressors": 0}, "n_regressors must be a positive int"),
    ({"n_regressors": True}, "n_regressors must be a positive int"),
    ({"n_bins": 2.0}, "n_bins must be a positive int"),
    ({"n_obs": np.zeros((3, 1), dtype=int)}, "non-empty 1-D"),
    ({"n_obs": np.array([], dtype=int)}, "non-empty 1-D"),
    ({"n_obs": np.array([1.0, 2.0, 3.0])}, "non-negative integers"),
    ({"n_obs": np.array([1, -2, 3])}, "non-negative integers"),
    ({"XtX_c": np.zeros((3, 2, 3))}, r"XtX_c must have shape \(3, 2, 2\)"),
    ({"Y_mean": np.zeros((3, 5))}, r"Y_mean must have shape \(3, 4\)"),
    ({"YtY_c": [[1, 2]]}, "YtY_c must have shape"),
    ({"X_mean": "abc"}, "X_mean is not a real array"),
    ({"XtX_c": np.full((3, 2, 2), np.nan)}, "XtX_c must be finite"),
    ({"YtY_c": np.array([1.0, np.inf, 1.0])}, "YtY_c must be finite"),
    ({"Y_mean": np.full((3, 4), np.nan)}, "Y_mean must be finite"),
    ({"Y_mean": np.full((3, 4), 1e200)}, "the raw moment YtY_raw .* overflows"),
]


def test_constructor_rejects_an_asymmetric_gram_and_sums_without_trials() -> None:
    # eigh reads one triangle of the Gram while the residual uses all of it,
    # so an asymmetric hand-built XtX_c would be silently inconsistent; a
    # neuron with no observed trial has zero moments.
    Y, X, mask = _random_problem(7, 6, 3, 4, 2)
    stats = sufficient_statistics(Y, X, mask)
    fields: dict[str, Any] = {name: np.array(getattr(stats, name)) for name in FIELDS}
    fields.update(n_regressors=2, n_bins=4)
    skew = np.array(fields["XtX_c"])
    skew[1, 0, 1] += 5.0
    with pytest.raises(ValidationError, match=r"XtX_c must be symmetric.* 1$"):
        SufficientStats(**{**fields, "XtX_c": skew})
    # Rounding-level asymmetry is accepted and symmetrised exactly.
    tiny = np.array(fields["XtX_c"])
    tiny[1, 0, 1] *= 1 + 1e-14
    built = SufficientStats(**{**fields, "XtX_c": tiny})
    np.testing.assert_array_equal(built.XtX_c, built.XtX_c.transpose(0, 2, 1))
    n_obs = np.array(fields["n_obs"])
    n_obs[2] = 0
    with pytest.raises(ValidationError, match=r"n_obs == 0 must have zero.* 2 do"):
        SufficientStats(**{**fields, "n_obs": n_obs})


@pytest.mark.parametrize("which", ["Y", "X"])
def test_values_too_large_to_square_are_named(which: str) -> None:
    # Finite values whose squares overflow get an error that names the cause,
    # not a generic "YtY_raw must be finite". (Unobserved entries may still
    # hold anything: test_unobserved_entries_are_ignored.)
    Y, X, mask = _random_problem(10, 8, 3, 4, 2, p_obs=1.0)
    if which == "Y":
        Y[:, 1] *= 1e160
        match = r"Y is too large to square .* neurons 1; rescale Y"
    else:
        X[:, 0] *= 1e160
        match = r"X is too large to square .* neurons 0, 1, 2; rescale X"
    with pytest.raises(ValidationError, match=match):
        sufficient_statistics(Y, X, mask)


@pytest.mark.parametrize(("change", "match"), BAD_STATS)
def test_constructor_rejects_inconsistent_fields(
    change: dict[str, Any], match: str
) -> None:
    Y, X, mask = _random_problem(7, 6, 3, 4, 2)
    stats = sufficient_statistics(Y, X, mask)
    fields: dict[str, Any] = {name: getattr(stats, name) for name in FIELDS}
    fields.update(n_regressors=2, n_bins=4)
    fields.update(change)
    with pytest.raises(ValidationError, match=match):
        SufficientStats(**fields)


def _bad_data_cases() -> list[tuple[Any, Any, Any, str]]:
    Y, X, mask = _random_problem(8, 5, 3, 4, 2)
    nan_obs = Y.copy()
    nan_obs[1, 2, 3] = np.nan
    inf_obs = Y.copy()
    inf_obs[0, 0, 0] = np.inf
    many = np.full_like(Y, np.nan)
    X_nan = X.copy()
    X_nan[3, 1] = np.nan
    ones = np.ones_like(mask)
    return [
        (Y[0], X, mask, r"Y must be 3-D \(n_trials, n_neurons, n_bins\)"),
        (Y[:, :, :0], X, mask, "non-empty axes"),
        (Y.astype(complex), X, mask, "real numeric"),
        (np.array([["a"]]), X, mask, "real numeric"),
        ([[[1.0], [1.0, 2.0]]], X, mask, "not a numeric array"),
        (Y, X[0], mask, r"X must be 2-D \(n_trials, n_regressors\)"),
        (Y, X[:, :0], mask, "non-empty axes"),
        (Y, X[:4], mask, "Y has 5 trials .* but X has 4 rows"),
        (np.moveaxis(Y, 0, 2), X, mask, "laid out \\(n_trials, n_neurons, n_bins\\)"),
        (Y, X_nan, mask, r"X must be finite; non-finite rows 3"),
        (Y, X, mask[:, :2], r"mask must have shape \(n_trials, n_neurons\)"),
        (Y, X, mask.astype(float) * 2, "only 0 and 1"),
        (Y, X, np.full(mask.shape, 0.5), "only 0 and 1"),
        (Y, X, np.where(mask, 1.0, np.nan), "only 0 and 1"),
        (Y, X, mask.astype(str), "boolean or 0/1"),
        (Y, X, [[1, 0], [1]], "mask is not an array"),
        (
            nan_obs,
            X,
            ones,
            r"NaN or Inf where mask is True, at \(trial, neuron\) \(1, 2\)",
        ),
        (inf_obs, X, ones, r"\(0, 0\)"),
        (many, X, ones, "and 5 more"),
    ]


@pytest.mark.parametrize(("Y", "X", "mask", "match"), _bad_data_cases())
def test_invalid_data_raise_validation_error(
    Y: Any, X: Any, mask: Any, match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        sufficient_statistics(Y, X, mask)


@pytest.mark.parametrize(
    ("intercept", "match"),
    [
        (np.zeros((3, 5)), r"shape \(n_neurons, n_bins\) = \(3, 4\)"),
        (np.zeros((3, 4), dtype=bool), "real array"),
        (np.full((3, 4), np.nan), "finite"),
        ([[1.0, 2.0], [1.0]], "not an array"),
        ("ab", "real array"),
    ],
)
def test_centered_rejects_bad_intercepts(intercept: Any, match: str) -> None:
    Y, X, mask = _random_problem(9, 6, 3, 4, 2)
    with pytest.raises(ParameterError, match=match):
        sufficient_statistics(Y, X, mask).centered(intercept)


def test_validation_errors_are_value_errors() -> None:
    with pytest.raises(ValueError, match="3-D"):
        sufficient_statistics(np.zeros((2, 2)), np.zeros((2, 1)), np.ones((2, 2)))


def test_simulated_data_pass_through() -> None:
    sim = mtdr.simulate(
        n_neurons=5, n_bins=4, n_trials=20, ranks=[1], drop_prob=0.3, seed=0
    )
    _assert_stats_equal(
        sufficient_statistics(sim.Y_masked, sim.X, sim.mask),
        sufficient_statistics(sim.Y, sim.X, sim.mask),
    )
