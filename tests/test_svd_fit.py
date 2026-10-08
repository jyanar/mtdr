"""Tests for `mtdr.svd_fit.fit_svd`: recovery, identities, properties, edge cases."""

from __future__ import annotations

import pickle
import warnings
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

import mtdr
from mtdr import aic as aic_mod
from mtdr.errors import DesignWarning, ParameterError, ValidationError
from mtdr.stats import sufficient_statistics
from mtdr.svd_fit import RANK_TOLERANCE, SVDFit, fit_svd

FloatArray = NDArray[np.float64]


def _problem(
    seed: int,
    N: int = 40,
    n: int = 6,
    T: int = 5,
    P: int = 2,
    p_obs: float = 0.8,
    offset: float = 1.0,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Random data whose every neuron sees at least P + 2 trials."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, P))
    Y = offset + rng.normal(size=(N, n, T)) * rng.uniform(0.5, 2.0, size=(1, n, 1))
    mask = rng.random((N, n)) < p_obs
    mask[: P + 2] = True
    return Y, X, mask


def _fit(
    Y: FloatArray, X: FloatArray, mask: NDArray[np.bool_], ranks: Any, **kw: Any
) -> SVDFit:
    return fit_svd(sufficient_statistics(Y, X, mask), ranks, **kw)


def _b0(fit: SVDFit) -> FloatArray:
    """The fitted intercept, asserted present."""
    assert fit.intercept is not None
    return fit.intercept


def _ols(
    Y: FloatArray, X: FloatArray, mask: NDArray[np.bool_], intercept: bool = True
) -> FloatArray:
    """Independent per-neuron least squares, (n, P', T), intercept last."""
    out = []
    for i in range(Y.shape[1]):
        rows = mask[:, i]
        D = np.column_stack([X[rows], np.ones(rows.sum())]) if intercept else X[rows]
        out.append(np.linalg.lstsq(D, Y[rows, i, :], rcond=None)[0])
    return np.stack(out)


def _rss_direct(
    Y: FloatArray, X: FloatArray, mask: NDArray[np.bool_], fit: SVDFit
) -> FloatArray:
    pred = np.einsum("kp,pit->kit", X, np.stack(fit.B))
    if fit.intercept is not None:
        pred = pred + fit.intercept
    resid = np.where(mask[:, :, None], Y - pred, 0.0)
    return np.asarray((resid**2).sum(axis=(0, 2)))


# ------------------------------------------------------------------ recovery


@pytest.mark.parametrize("seed", range(10))
def test_recovers_the_simulated_coefficients(seed: int) -> None:
    # Bounded precisions (uniform on [0.5, 2]): the unweighted SVD truncation
    # is not robust to the demo's heavy-tailed Exponential draws. Over
    # 300 seeds of this configuration the worst relative errors were 0.106
    # (B_p), 0.032 (intercept) and 0.107 (|log| precision ratio); the bounds
    # below are 2.4-3x those.
    rng = np.random.default_rng(1000 + seed)
    sim = mtdr.simulate(
        n_neurons=60,
        n_bins=10,
        n_trials=400,
        ranks=[2, 1, 3],
        drop_prob=0.3,
        noise_precision=rng.uniform(0.5, 2.0, 60),
        seed=seed,
    )
    fit = _fit(sim.Y_masked, sim.X, sim.mask, [2, 1, 3])
    for p, name in enumerate(sim.regressor_names):
        err = np.linalg.norm(fit.B[p] - sim.B[name]) / np.linalg.norm(sim.B[name])
        assert err < 0.25, (name, err)
    assert sim.intercept is not None
    assert fit.intercept is not None
    err = np.linalg.norm(fit.intercept - sim.intercept) / np.linalg.norm(sim.intercept)
    assert err < 0.1
    assert np.abs(np.log(fit.noise_precision / sim.noise_precision)).max() < 0.3


def test_recovers_without_the_intercept() -> None:
    sim = mtdr.simulate(
        n_neurons=30,
        n_bins=8,
        n_trials=300,
        ranks=[2, 1],
        noise_precision=1.0,
        condition_independent=False,
        seed=4,
    )
    fit = _fit(sim.Y, sim.X, sim.mask, [2, 1], condition_independent=False)
    assert fit.intercept is None
    for p, name in enumerate(sim.regressor_names):
        err = np.linalg.norm(fit.B[p] - sim.B[name]) / np.linalg.norm(sim.B[name])
        assert err < 0.1
    assert fit.n_parameters == aic_mod.n_parameters_svd(
        [2, 1], 30, 8, condition_independent=False
    )


# ------------------------------------------------------------------ identities


@pytest.mark.parametrize("intercept", [True, False])
def test_full_rank_equals_per_neuron_least_squares(intercept: bool) -> None:
    # Full rank (here with unequal variances and a mask, which the SVD
    # estimator ignores) is ordinary least squares.
    Y, X, mask = _problem(0, n=6, T=5, P=3)
    fit = _fit(Y, X, mask, [5, 5, 5], condition_independent=intercept)
    beta = _ols(Y, X, mask, intercept)
    for p in range(3):
        np.testing.assert_allclose(fit.B_full[p], beta[:, p], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(fit.B[p], fit.B_full[p], rtol=1e-10, atol=1e-12)
    if intercept:
        assert fit.intercept is not None
        np.testing.assert_allclose(fit.intercept, beta[:, 3], rtol=1e-10, atol=1e-12)
    assert fit.rank_deficient_neurons.size == 0


@given(
    seed=st.integers(0, 2**32 - 1),
    ranks=st.lists(st.integers(0, 4), min_size=2, max_size=2),
)
def test_precision_and_log_likelihood_follow_m19b_and_the_gaussian_density(
    seed: int, ranks: list[int]
) -> None:
    Y, X, mask = _problem(seed, n=5, T=4, P=2)
    fit = _fit(Y, X, mask, ranks)
    rss = _rss_direct(Y, X, mask, fit)
    m = mask.sum(axis=0) * 4
    np.testing.assert_allclose(fit.noise_precision, m / rss, rtol=1e-9)
    # The sum over observed entries of the Gaussian log density, constant
    # included.
    lam = fit.noise_precision[None, :, None]
    pred = np.einsum("kp,pit->kit", X, np.stack(fit.B)) + fit.intercept
    dens = 0.5 * np.log(lam) - 0.5 * np.log(2 * np.pi) - 0.5 * lam * (Y - pred) ** 2
    ll = float(np.where(mask[:, :, None], dens, 0.0).sum())
    np.testing.assert_allclose(fit.log_likelihood, ll, rtol=1e-10)
    assert fit.n_parameters == aic_mod.n_parameters_svd(ranks, 5, 4)
    assert fit.aic == pytest.approx(2 * fit.n_parameters - 2 * fit.log_likelihood)


@given(
    seed=st.integers(0, 2**32 - 1),
    ranks=st.lists(st.integers(0, 4), min_size=2, max_size=2),
)
def test_factors_are_the_truncated_svd(seed: int, ranks: list[int]) -> None:
    Y, X, mask = _problem(seed, n=5, T=4, P=2)
    fit = _fit(Y, X, mask, ranks)
    for p, r in enumerate(ranks):
        W, S, B = fit.W[p], fit.S[p], fit.B[p]
        assert W.shape == (5, r)
        assert S.shape == (4, r)
        np.testing.assert_allclose(W @ S.T, B, atol=1e-12)
        U, s, Vt = np.linalg.svd(fit.B_full[p], full_matrices=False)
        np.testing.assert_allclose(B, (U[:, :r] * s[:r]) @ Vt[:r], atol=1e-10)
        # (M17): W = U sqrt(s), S = V sqrt(s): orthogonal columns, equal norms.
        np.testing.assert_allclose(W.T @ W, np.diag(s[:r]), atol=1e-10)
        np.testing.assert_allclose(S.T @ S, np.diag(s[:r]), atol=1e-10)
        # The largest-magnitude entry of each column of W is positive.
        if r:
            lead = W[np.argmax(np.abs(W), axis=0), np.arange(r)]
            assert (lead >= 0).all()


def test_all_ones_mask_equals_the_unmasked_pooled_regression() -> None:
    Y, X, _ = _problem(1, N=30, n=4, T=3, P=2)
    mask = np.ones((30, 4), dtype=bool)
    fit = _fit(Y, X, mask, [3, 3])
    D = np.column_stack([X, np.ones(30)])
    beta = np.linalg.lstsq(D, Y.reshape(30, -1), rcond=None)[0].reshape(3, 4, 3)
    for p in range(2):
        np.testing.assert_allclose(fit.B[p], beta[p], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(_b0(fit), beta[2], rtol=1e-10, atol=1e-12)


def test_unobserved_entries_do_not_change_the_fit() -> None:
    Y, X, mask = _problem(2)
    garbage = np.where(mask[:, :, None], Y, np.nan)
    a, b = _fit(Y, X, mask, [1, 2]), _fit(garbage, X, mask, [1, 2])
    for p in range(2):
        np.testing.assert_array_equal(a.B[p], b.B[p])
    np.testing.assert_array_equal(a.noise_precision, b.noise_precision)


# ------------------------------------------------------------------ properties


@settings(deadline=None)
@given(seed=st.integers(0, 2**32 - 1), data=st.data())
def test_neuron_permutation_equivariance(seed: int, data: st.DataObject) -> None:
    Y, X, mask = _problem(seed, n=6, T=4, P=2)
    perm = np.array(data.draw(st.permutations(range(6))))
    a = _fit(Y, X, mask, [2, 1])
    b = _fit(Y[:, perm], X, mask[:, perm], [2, 1])
    for p in range(2):
        np.testing.assert_allclose(b.B[p], a.B[p][perm], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(
            b.B_full[p], a.B_full[p][perm], rtol=1e-9, atol=1e-12
        )
    np.testing.assert_allclose(_b0(b), _b0(a)[perm], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(b.noise_precision, a.noise_precision[perm], rtol=1e-9)
    assert b.log_likelihood == pytest.approx(a.log_likelihood, rel=1e-12)


@given(seed=st.integers(0, 2**32 - 1), log_c=st.floats(-3, 3), negative=st.booleans())
def test_scaling_y_scales_b_by_c_and_precision_by_c_to_minus_two(
    seed: int, log_c: float, negative: bool
) -> None:
    c = (-1.0 if negative else 1.0) * 10.0**log_c
    Y, X, mask = _problem(seed, n=5, T=4, P=2)
    a = _fit(Y, X, mask, [2, 1])
    b = _fit(c * Y, X, mask, [2, 1])
    for p in range(2):
        np.testing.assert_allclose(b.B[p], c * a.B[p], rtol=1e-8, atol=1e-12 * abs(c))
        # The factors split |c| evenly; the sign convention keeps W's sign.
        np.testing.assert_allclose(
            b.W[p], np.sqrt(abs(c)) * a.W[p], rtol=1e-7, atol=1e-10
        )
        np.testing.assert_allclose(
            b.S[p], np.sign(c) * np.sqrt(abs(c)) * a.S[p], rtol=1e-7, atol=1e-10
        )
    np.testing.assert_allclose(_b0(b), c * _b0(a), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(b.noise_precision, a.noise_precision / c**2, rtol=1e-8)
    n_entries = mask.sum() * 4
    assert b.log_likelihood == pytest.approx(
        a.log_likelihood - n_entries * np.log(abs(c)), rel=1e-9, abs=1e-9
    )


@given(seed=st.integers(0, 2**32 - 1), log_a=st.floats(-6, 6), p=st.integers(0, 1))
def test_scaling_a_column_of_x_rescales_only_its_coefficients(
    seed: int, log_a: float, p: int
) -> None:
    # The unpenalised SVD fit is invariant to rescaling a column of X; also
    # checks that the rank-deficiency test is scale invariant: no neuron
    # becomes "deficient" at a 1e6 scale ratio.
    a_scale = 10.0**log_a
    Y, X, mask = _problem(seed, n=5, T=4, P=2)
    X2 = X.copy()
    X2[:, p] *= a_scale
    f1 = _fit(Y, X, mask, [2, 1])
    f2 = _fit(Y, X2, mask, [2, 1])
    assert f2.rank_deficient_neurons.size == 0
    for q in range(2):
        factor = a_scale if q == p else 1.0
        np.testing.assert_allclose(f2.B[q] * factor, f1.B[q], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(_b0(f2), _b0(f1), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(f2.noise_precision, f1.noise_precision, rtol=1e-6)


@pytest.mark.parametrize("shift", [7.0, 50.0, -1e3])
def test_shifting_a_regressor_leaves_the_score_unchanged(shift: float) -> None:
    # With the intercept, shifting X[:, p] by c leaves B_full, the truncated B,
    # the precisions, the log-likelihood and the AIC unchanged; only the
    # intercepts move, by -c times the regressor's coefficient (the
    # unconstrained one for intercept_full, the truncated one for intercept).
    # Keeping the joint-OLS intercept next to the truncated B (the reference's
    # (M19b)) is not shift invariant: on a 50-neuron simulation its AIC moves
    # from 215827.5 to 339165.3 under a +7 shift, while the refitted intercept
    # gives 215578.475375939 both ways.
    Y, X, mask = _problem(3, n=6, T=5, P=2)
    shifted = X.copy()
    shifted[:, 0] += shift
    X_back = shifted.copy()
    X_back[:, 0] -= shift  # exact: the design that `shifted` shifts by `shift`
    f1, f2 = _fit(Y, X_back, mask, [1, 1]), _fit(Y, shifted, mask, [1, 1])
    for p in range(2):
        np.testing.assert_allclose(f2.B_full[p], f1.B_full[p], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(f2.B[p], f1.B[p], rtol=1e-9, atol=1e-12)
    assert f1.intercept_full is not None
    assert f2.intercept_full is not None
    np.testing.assert_allclose(
        f2.intercept_full, f1.intercept_full - shift * f1.B_full[0], atol=1e-9
    )
    np.testing.assert_allclose(_b0(f2), _b0(f1) - shift * f1.B[0], atol=1e-9)
    np.testing.assert_allclose(f2.noise_precision, f1.noise_precision, rtol=1e-11)
    assert f2.log_likelihood == pytest.approx(f1.log_likelihood, rel=1e-13)
    assert f2.aic == pytest.approx(f1.aic, rel=1e-13)


@settings(deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    ranks=st.lists(st.integers(0, 4), min_size=2, max_size=2),
    ridge=st.sampled_from([0.0, 0.5, 30.0]),
)
def test_intercept_is_the_conditional_refit_given_the_truncated_blocks(
    seed: int, ranks: list[int], ridge: float
) -> None:
    # b_i = n_i / (n_i + ridge) * (ybar_i - sum_p xbar_ip B_p[i, :]), the
    # maximiser over b of the (ridge-penalised) residual at the truncated B;
    # at full rank it is the joint least-squares intercept.
    Y, X, mask = _problem(seed, n=5, T=4, P=2)
    fit = _fit(Y, X, mask, ranks, ridge=ridge)
    n_obs = mask.sum(axis=0)
    ybar = np.stack([Y[mask[:, i], i].mean(axis=0) for i in range(5)])
    xbar = np.stack([X[mask[:, i]].mean(axis=0) for i in range(5)])
    shrink = (n_obs / (n_obs + ridge))[:, None]
    level = ybar - np.einsum("ip,pit->it", xbar, np.stack(fit.B))
    np.testing.assert_allclose(_b0(fit), shrink * level, rtol=1e-10, atol=1e-12)
    full = ybar - np.einsum("ip,pit->it", xbar, np.stack(fit.B_full))
    assert fit.intercept_full is not None
    np.testing.assert_allclose(
        fit.intercept_full, shrink * full, rtol=1e-10, atol=1e-12
    )
    if all(r == 4 for r in ranks):
        np.testing.assert_allclose(_b0(fit), fit.intercept_full, rtol=1e-10, atol=1e-12)
    # A maximiser: moving the intercept never lowers the penalised residual.
    pred = np.einsum("kp,pit->kit", X, np.stack(fit.B))

    def objective(b: FloatArray) -> float:
        resid = np.where(mask[:, :, None], Y - pred - b, 0.0)
        return float((resid**2).sum() + ridge * (b**2).sum())

    best = objective(_b0(fit))
    step = np.random.default_rng(seed).normal(size=(5, 4)) * 1e-3
    assert objective(_b0(fit) + step) > best
    assert objective(_b0(fit) - step) > best


@pytest.mark.parametrize("baseline", [1e3, 1e5, 1e7, 1e8])
def test_a_large_baseline_costs_no_accuracy(baseline: float) -> None:
    # Moments not taken about the means would give the precisions relative
    # errors of 8.8e-10 at a baseline of 1e3, 1.5e-5 at 1e5 and 9.8e-4 at 1e6,
    # and a false "zero residual" ValidationError at 1e7 or 1e8.
    Y, X, mask = _problem(20, N=200, n=5, T=10, P=2, offset=0.0)
    lifted = Y + baseline
    back = lifted - baseline  # exact: the responses `lifted` shifts by `baseline`
    f1, f2 = _fit(back, X, mask, [1, 1]), _fit(lifted, X, mask, [1, 1])
    np.testing.assert_allclose(f2.noise_precision, f1.noise_precision, rtol=1e-9)
    for p in range(2):
        err = np.abs(f2.B_full[p] - f1.B_full[p]).max() / np.abs(f1.B_full[p]).max()
        assert err < 1e-9
    np.testing.assert_allclose(_b0(f2) - baseline, _b0(f1), atol=1e-9 * baseline)
    # Against the data without the baseline: the rounding of Y + baseline itself.
    f0 = _fit(Y, X, mask, [1, 1])
    np.testing.assert_allclose(f2.noise_precision, f0.noise_precision, rtol=1e-6)


@pytest.mark.parametrize("offset", [1e2, 1e5, 1e7])
def test_a_large_regressor_offset_costs_no_accuracy(offset: float) -> None:
    # With moments not taken about the means, an offset of 1e5 standard
    # deviations in a column would give B_full[0] a relative error of 1.0 and
    # flag every neuron as rank-deficient ("too few trials").
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    Y = rng.normal(size=(200, 5, 6))
    mask = rng.random((200, 5)) < 0.8
    shifted = X.copy()
    shifted[:, 0] += offset
    X_back = shifted.copy()
    X_back[:, 0] -= offset  # exact
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f2 = _fit(Y, shifted, mask, [5, 5])
    assert f2.rank_deficient_neurons.size == 0
    beta = _ols(Y, X_back, mask)
    for p in range(2):
        err = np.abs(f2.B_full[p] - beta[:, p]).max() / np.abs(beta[:, p]).max()
        assert err < 1e-9
    f1 = _fit(Y, X_back, mask, [5, 5])
    np.testing.assert_allclose(f2.noise_precision, f1.noise_precision, rtol=1e-9)


def test_deterministic() -> None:
    Y, X, mask = _problem(4)
    a, b = _fit(Y, X, mask, [2, 2]), _fit(Y, X, mask, [2, 2])
    for p in range(2):
        np.testing.assert_array_equal(a.W[p], b.W[p])
        np.testing.assert_array_equal(a.S[p], b.S[p])
    assert a.aic == b.aic


# ------------------------------------------------------------------ edge cases


@pytest.mark.parametrize(("n", "T"), [(1, 5), (6, 1)])
def test_singleton_shapes(n: int, T: int) -> None:
    # The reference fails here (`squeeze` returns the wrong orientation at
    # T = 1, its `slow*` fallbacks misread n = 1); the port does not.
    sim = mtdr.simulate(n_neurons=n, n_bins=T, n_trials=12, ranks=[1, 0], seed=0)
    fit = _fit(sim.Y, sim.X, sim.mask, [1, 0])
    assert fit.W[0].shape == (n, 1)
    assert fit.S[0].shape == (T, 1)
    assert fit.W[1].shape == (n, 0)
    assert fit.S[1].shape == (T, 0)
    np.testing.assert_array_equal(fit.B[1], 0.0)
    assert fit.intercept is not None
    assert fit.intercept.shape == (n, T)
    assert np.isfinite(fit.log_likelihood)


def test_one_regressor_and_one_bin() -> None:
    Y, X, mask = _problem(5, n=4, T=1, P=1)
    fit = _fit(Y, X, mask, [1])
    np.testing.assert_allclose(fit.B[0], _ols(Y, X, mask)[:, 0], rtol=1e-10)


def test_rank_zero_removes_the_term() -> None:
    Y, X, mask = _problem(6, n=6, T=5, P=2)
    fit = _fit(Y, X, mask, [0, 2])
    assert fit.W[0].shape == (6, 0)
    assert fit.S[0].shape == (5, 0)
    np.testing.assert_array_equal(fit.B[0], 0.0)
    assert np.abs(fit.B_full[0]).max() > 0  # still estimated
    assert fit.n_parameters == aic_mod.n_parameters_svd([0, 2], 6, 5)
    np.testing.assert_allclose(
        _rss_direct(Y, X, mask, fit), mask.sum(0) * 5 / fit.noise_precision, rtol=1e-9
    )


def test_identical_regressors_get_the_minimum_norm_split() -> None:
    Y, X1, mask = _problem(7, n=5, T=4, P=1)
    X = np.column_stack([X1, X1])
    with pytest.warns(DesignWarning, match=r"neurons \[0, 1, 2, 3, 4\]"):
        fit = _fit(Y, X, mask, [2, 2])
    np.testing.assert_array_equal(fit.rank_deficient_neurons, np.arange(5))
    single = _ols(Y, X1, mask)
    # Minimum norm splits the coefficient equally between the copies.
    np.testing.assert_allclose(fit.B_full[0], fit.B_full[1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(2 * fit.B_full[0], single[:, 0], rtol=1e-9, atol=1e-12)
    assert fit.intercept_full is not None
    np.testing.assert_allclose(fit.intercept_full, single[:, 1], rtol=1e-9, atol=1e-12)


def test_neuron_with_fewer_trials_than_regressors_gets_lstsq() -> None:
    # Singular A_i: neuron 2 is observed on 3 trials, with 3 regressors plus
    # the intercept. Its null space contains an intercept direction, so the
    # minimum norm is over the regressor coefficients with the intercept
    # profiled out: lstsq on the neuron's centred design, then the intercept
    # from the means.
    Y, X, mask = _problem(8, N=40, n=5, T=4, P=3)
    mask[:, 2] = False
    mask[[5, 9, 17], 2] = True
    with pytest.warns(DesignWarning, match=r"neurons \[2\] have a rank-deficient"):
        fit = _fit(Y, X, mask, [1, 1, 1])
    np.testing.assert_array_equal(fit.rank_deficient_neurons, [2])
    rows = mask[:, 2]
    Xi, Yi = X[rows], Y[rows, 2, :]
    centred = Xi - Xi.mean(axis=0)
    beta = np.linalg.lstsq(centred, Yi - Yi.mean(axis=0), rcond=None)[0]
    got = np.stack([b[2] for b in fit.B_full])
    np.testing.assert_allclose(got, beta, rtol=1e-9, atol=1e-11)
    assert fit.intercept_full is not None
    np.testing.assert_allclose(
        fit.intercept_full[2], Yi.mean(axis=0) - Xi.mean(axis=0) @ beta, atol=1e-10
    )
    # The other neurons are solved exactly as without neuron 2.
    others = [0, 1, 3, 4]
    ref = _fit(Y[:, others], X, mask[:, others], [1, 1, 1])
    np.testing.assert_allclose(fit.B_full[0][others], ref.B_full[0], rtol=1e-10)


@pytest.mark.parametrize("value", [2.5, 0.1, 1e7 + 0.3, 0.0])
def test_constant_regressor_within_a_neuron_is_rank_deficient(value: float) -> None:
    # A regressor constant over one neuron's trials (a per-session constant)
    # is collinear with the intercept for that neuron at any trial count. The
    # centred moments make its column exactly zero whatever the constant, so
    # the minimum-norm solution gives it a zero coefficient.
    Y, X, mask = _problem(9, N=60, n=4, T=3, P=2)
    X[:30, 1] = value
    mask[:, 1] = False
    mask[:30, 1] = True
    with pytest.warns(DesignWarning, match=r"\[1\]"):
        fit = _fit(Y, X, mask, [1, 1])
    np.testing.assert_array_equal(fit.rank_deficient_neurons, [1])
    np.testing.assert_array_equal(fit.B_full[1][1], 0.0)
    rows = mask[:, 1]
    single = np.linalg.lstsq(
        np.column_stack([X[rows, 0], np.ones(30)]), Y[rows, 1], rcond=None
    )[0]
    np.testing.assert_allclose(fit.B_full[0][1], single[0], rtol=1e-9)


@pytest.mark.parametrize("scale", [1e-12, 1e-15])
def test_singular_and_scaled_design_gets_the_minimum_norm_solution(
    scale: float,
) -> None:
    # Columns [v, a v, 1] with a tiny a. The null direction (a, -1, 0) has no
    # intercept part, so the answer is lstsq on the raw design. Subtracting
    # the null-space component of a solution with an O(1/a) coefficient would
    # lose 6.1e-5 (a = 1e-12) and 0.022 (a = 1e-15) in an order-one
    # coefficient; the retained-range solve does not.
    rng = np.random.default_rng(701)
    Y = rng.normal(size=(40, 7, 5))
    X = rng.normal(size=(40, 2))
    mask = rng.random((40, 7)) > 0.2
    X[:, 1] = X[:, 0] * scale
    with pytest.warns(DesignWarning):
        fit = _fit(Y, X, mask, [1, 1])
    np.testing.assert_array_equal(fit.rank_deficient_neurons, np.arange(7))
    beta = _ols(Y, X, mask)
    assert fit.intercept_full is not None
    got = np.stack([fit.B_full[0], fit.B_full[1], fit.intercept_full], axis=1)
    assert np.abs(got - beta).max() < 1e-12 * np.abs(beta).max()


def test_singular_neuron_in_a_full_rank_pool_gets_the_minimum_norm_solution() -> None:
    # Only neuron 0 sees the first 12 trials, on which its two columns are
    # proportional; the pooled design is full rank. The exact answer is a
    # two-column regression. Projecting out the null-space component instead
    # gives the dominant slope a relative error of 0.056 (max coefficient
    # error 0.0167).
    rng = np.random.default_rng(701)
    Y = rng.normal(size=(40, 7, 5))
    X = rng.normal(size=(40, 2))
    mask = rng.random((40, 7)) > 0.2
    X[:12, 1] = X[:12, 0]
    a = 1e-15
    X[:, 1] *= a
    mask[:, 0] = False
    mask[:12, 0] = True
    with pytest.warns(DesignWarning, match=r"neurons \[0\] have"):
        fit = _fit(Y, X, mask, [1, 1])
    np.testing.assert_array_equal(fit.rank_deficient_neurons, [0])
    q = np.linalg.lstsq(np.c_[X[:12, 0], np.ones(12)], Y[:12, 0], rcond=None)[0]
    expected = np.stack([q[0] / (1 + a * a), a * q[0] / (1 + a * a), q[1]])
    assert fit.intercept_full is not None
    got = np.stack([fit.B_full[0][0], fit.B_full[1][0], fit.intercept_full[0]])
    assert np.abs(got - expected).max() < 1e-12
    assert np.abs(got[0] - expected[0]).max() < 1e-12 * np.abs(expected[0]).max()


def test_ridge_removes_the_deficiency_and_matches_the_closed_form() -> None:
    Y, X1, mask = _problem(10, n=4, T=3, P=1)
    X = np.column_stack([X1, X1])
    gamma = 0.7
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = _fit(Y, X, mask, [2, 2], ridge=gamma)
    assert fit.rank_deficient_neurons.size == 0
    for i in range(4):
        rows = mask[:, i]
        D = np.column_stack([X[rows], np.ones(rows.sum())])
        beta = np.linalg.solve(D.T @ D + gamma * np.eye(3), D.T @ Y[rows, i, :])
        assert fit.intercept_full is not None
        got = np.stack([fit.B_full[0][i], fit.B_full[1][i], fit.intercept_full[i]])
        np.testing.assert_allclose(got, beta, rtol=1e-10)
    # The precision is the plain (unpenalised) residual's, with the intercept
    # refitted and shrunk by n_i / (n_i + ridge).
    np.testing.assert_allclose(
        _rss_direct(Y, X, mask, fit), mask.sum(0) * 3 / fit.noise_precision, rtol=1e-10
    )
    # A ridge negligible against the Gram leaves the design singular.
    with pytest.warns(DesignWarning):
        tiny = _fit(Y, X, mask, [2, 2], ridge=1e-300)
    assert tiny.rank_deficient_neurons.size == 4
    # 0-d arrays and NumPy scalars are scalars.
    same = _fit(Y, X, mask, [2, 2], ridge=np.array(gamma))
    np.testing.assert_array_equal(same.B_full[0], fit.B_full[0])


def test_rank_tolerance_separates_exact_from_near_collinearity() -> None:
    # Exactly collinear designs land within ~60 eps of zero on the unit-
    # diagonal Gram (measured up to 1e5 trials); 1e-10 is far above that and
    # far below a merely ill-conditioned design.
    assert 1e3 * np.finfo(float).eps < RANK_TOLERANCE < 1e-6
    Y, X1, mask = _problem(11, N=200, n=3, T=3, P=1)
    near = np.column_stack(
        [X1, X1 + 1e-3 * np.random.default_rng(0).normal(size=X1.shape)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = _fit(Y, near, mask, [1, 1])
    assert fit.rank_deficient_neurons.size == 0


def test_never_observed_neuron_is_rejected() -> None:
    Y, X, mask = _problem(12)
    mask[:, 3] = False
    with pytest.raises(ValidationError, match=r"neurons \[3\] are never observed"):
        _fit(Y, X, mask, [1, 1])


def test_one_trial_neuron_is_degenerate_at_every_rank() -> None:
    # With the intercept, a neuron observed on one trial is fitted exactly by
    # its refitted intercept whatever its truncated blocks are, so its
    # precision is unbounded at every rank (a joint-OLS intercept would leave
    # a residual at truncated ranks and accept the fit). Without the
    # intercept it is only rank-deficient.
    Y, X, mask = _problem(18, N=40, n=5, T=4, P=2)
    mask[:, 1] = False
    mask[7, 1] = True
    stats = sufficient_statistics(Y, X, mask)
    for ranks in ([1, 1], [4, 4]):
        with (
            pytest.warns(DesignWarning, match=r"\[1\]"),
            pytest.raises(ValidationError, match=r"neurons \[1\] are fitted with zero"),
        ):
            fit_svd(stats, ranks)
    with pytest.warns(DesignWarning, match=r"\[1\]"):
        fit = fit_svd(stats, [1, 1], condition_independent=False)
    assert np.isfinite(fit.noise_precision[1])


def test_zero_energy_is_degenerate_whatever_the_rounding() -> None:
    # A rank-1 truncation can leave a rounding-level row for an all-zero
    # neuron (seen on macOS), so its RSS is a tiny positive number above the
    # floor 1e3 eps T E = 0; zero energy must be degenerate all the same.
    from mtdr import svd_fit

    with pytest.raises(ValidationError, match=r"neurons \[1\]"):
        svd_fit._check_degenerate(np.array([1.0, 1e-40]), np.array([2.0, 0.0]), 3)
    svd_fit._check_degenerate(np.array([1.0, 1e-3]), np.array([2.0, 1e-3]), 3)


def test_zero_residual_is_rejected() -> None:
    # An exact fit has an unbounded noise precision.
    rng = np.random.default_rng(13)
    X = rng.normal(size=(20, 2))
    B = rng.normal(size=(2, 4, 3))
    Y = 1.0 + np.einsum("kp,pit->kit", X, B)
    mask = np.ones((20, 4), dtype=bool)
    with pytest.raises(
        ValidationError, match=r"neurons \[0, 1, 2, 3\] are fitted with zero"
    ):
        _fit(Y, X, mask, [3, 3])
    # A silent neuron (all zeros) is degenerate too.
    Y2 = Y + rng.normal(size=Y.shape)
    Y2[:, 1] = 0.0
    with pytest.raises(ValidationError, match=r"neurons \[1\]"):
        _fit(Y2, X, mask, [1, 1])
    # So is a neuron constant over its trials at any level (its centred
    # energy is exactly zero).
    Y2[:, 1] = 0.1 + np.arange(3) / 7.0
    with pytest.raises(ValidationError, match=r"neurons \[1\]"):
        _fit(Y2, X, mask, [1, 1])
    # Small but genuine noise is not, down to an amplitude ratio of 1e-5 (a
    # residual 1e-10 of the energy), which a rule relative to sqrt(eps) would
    # reject although it is far above rounding.
    for level in (1e-3, 1e-5):
        Y3 = Y + level * rng.normal(size=Y.shape)
        assert np.isfinite(_fit(Y3, X, mask, [3, 3]).noise_precision).all()


@pytest.mark.parametrize("precision", [1e7, 1e8])
def test_high_snr_simulations_fit(precision: float) -> None:
    # A degeneracy rule relative to sqrt(eps) rejects 27-30 of 30 neurons of
    # simulate(noise_precision=1e7 or 1e8); the rounding-relative rule
    # (RSS <= 1e3 eps T E) accepts them.
    sim = mtdr.simulate(
        n_neurons=30,
        n_bins=10,
        n_trials=200,
        ranks=[2, 1, 3],
        noise_precision=precision,
        seed=0,
    )
    fit = _fit(sim.Y, sim.X, sim.mask, [2, 1, 3])
    ratio = fit.noise_precision / sim.noise_precision
    assert 0.9 < float(np.median(ratio)) < 1.2


def test_large_baseline_with_unit_noise_fits() -> None:
    # Y = 1e7 + N(0, 1). A rounding floor on the raw energy (64 eps T upsilon
    # = 1421) exceeds the RSS (about 200) and rejects every neuron; the energy
    # about the mean is what the residual is computed from. Kills a floor
    # taken on the raw energy.
    rng = np.random.default_rng(31)
    Y = 1e7 + rng.normal(size=(40, 3, 5))
    X = rng.normal(size=(40, 1))
    mask = np.ones((40, 3), dtype=bool)
    fit = _fit(Y, X, mask, [1])
    base = _fit(Y - 1e7, X, mask, [1])
    np.testing.assert_allclose(fit.noise_precision, base.noise_precision, rtol=1e-6)


def test_rounding_floor_separates_exact_from_high_snr_fits() -> None:
    # Degenerate iff RSS <= 1e3 eps T E, E the energy about the mean.
    # Responses whose full-rank residual is a chosen multiple of eps T E: 10
    # raises (an exact fit up to rounding), 1e5 fits, at the precision
    # n_i T / RSS. Kills a missing floor, or a floor without its margin.
    rng = np.random.default_rng(32)
    N, n, T = 30, 3, 4
    X = rng.normal(size=(N, 2))
    D = np.column_stack([X, np.ones(N)])
    signal = 5.0 + np.einsum("kp,pit->kit", X, rng.normal(size=(2, n, T)))
    noise = rng.normal(size=(N, n, T))
    noise -= np.einsum("kj,jit->kit", D @ np.linalg.pinv(D), noise)  # all residual
    mask = np.ones((N, n), dtype=bool)
    eps = np.finfo(float).eps
    energy = ((signal - signal.mean(axis=0)) ** 2).sum(axis=(0, 2))
    unit = (noise**2).sum(axis=(0, 2))

    def responses(ratio: float) -> FloatArray:
        scale = np.sqrt(ratio * eps * T * energy / unit)
        out: FloatArray = signal + scale[None, :, None] * noise
        return out

    with pytest.raises(ValidationError, match=r"neurons \[0, 1, 2\] are fitted"):
        _fit(responses(10.0), X, mask, [3, 3])
    fit = _fit(responses(1e5), X, mask, [3, 3])
    np.testing.assert_allclose(
        fit.noise_precision, N * T / (1e5 * eps * T * energy), rtol=1e-3
    )


# ------------------------------------------------------------------ arguments


def test_ndarray_and_numpy_integer_ranks_are_accepted() -> None:
    Y, X, mask = _problem(14)
    a = _fit(Y, X, mask, np.array([2, 1]))
    b = _fit(Y, X, mask, (np.int64(2), np.int8(1)))
    assert a.ranks == b.ranks == (2, 1)
    assert all(isinstance(r, int) for r in a.ranks)


BAD_ARGS: list[tuple[dict[str, Any], str]] = [
    ({"ranks": [1]}, "ranks has 1 entries; expected one per regressor \\(2\\)"),
    ({"ranks": [1, 1, 1]}, "expected one per regressor"),
    ({"ranks": [6, 1]}, r"ranks\[0\] is 6, above min\(n_neurons, n_bins\) = 5"),
    ({"ranks": [-1, 1]}, r"ranks\[0\] must be a non-negative integer"),
    ({"ranks": [True, 1]}, r"ranks\[0\] must be a non-negative integer"),
    ({"ranks": [1.0, 1]}, r"ranks\[0\] must be a non-negative integer"),
    ({"ranks": np.array([1.0, 1.0])}, "non-negative integer"),
    ({"ranks": np.array(2)}, "ranks must be a sequence"),
    ({"ranks": 2}, "ranks must be a sequence"),
    ({"ranks": "12"}, "ranks must be a sequence"),
    ({"ranks": {1, 2}}, "ranks must be a sequence"),
    ({"ranks": []}, "at least one entry"),
    ({"ranks": [1, 1], "ridge": -1.0}, "ridge must be a finite real number >= 0"),
    ({"ranks": [1, 1], "ridge": np.nan}, "ridge"),
    ({"ranks": [1, 1], "ridge": np.inf}, "ridge"),
    ({"ranks": [1, 1], "ridge": True}, "ridge"),
    ({"ranks": [1, 1], "ridge": "0"}, "ridge"),
    (
        {"ranks": [1, 1], "condition_independent": "False"},
        "condition_independent must be a bool",
    ),
    ({"ranks": [1, 1], "condition_independent": 1}, "must be a bool"),
]


@pytest.mark.parametrize(("kwargs", "match"), BAD_ARGS)
def test_invalid_arguments_raise_parameter_error(
    kwargs: dict[str, Any], match: str
) -> None:
    Y, X, mask = _problem(15)
    stats = sufficient_statistics(Y, X, mask)
    with pytest.raises(ParameterError, match=match):
        fit_svd(stats, **kwargs)


def test_stats_must_be_sufficient_stats() -> None:
    with pytest.raises(ParameterError, match="stats must be a SufficientStats"):
        fit_svd({"XtX": 1}, [1])  # type: ignore[arg-type]


def test_numpy_bool_condition_independent() -> None:
    Y, X, mask = _problem(16)
    assert (
        _fit(Y, X, mask, [1, 1], condition_independent=np.bool_(False)).intercept
        is None
    )


def test_result_is_read_only_pickles_and_summarises() -> None:
    Y, X, mask = _problem(17)
    fit = _fit(Y, X, mask, [2, 1])
    arrays = [
        *fit.W,
        *fit.S,
        *fit.B,
        *fit.B_full,
        fit.noise_precision,
        fit.rank_deficient_neurons,
        fit.intercept,
        fit.intercept_full,
    ]
    for a in arrays:
        assert a is not None
        assert not a.flags.writeable
    with pytest.raises(AttributeError):
        fit.aic = 0.0  # type: ignore[misc]
    # Tuples, not lists: an element cannot be rebound.
    for container in (fit.W, fit.S, fit.B, fit.B_full):
        assert isinstance(container, tuple)
    with pytest.raises(TypeError):
        fit.B[0] = np.zeros((6, 5))  # type: ignore[index]
    again = pickle.loads(pickle.dumps(fit))
    np.testing.assert_array_equal(again.B[0], fit.B[0])
    # Unpickling restores read-only arrays.
    for a in (*again.W, *again.B, again.noise_precision, again.intercept):
        assert a is not None
        assert not a.flags.writeable
    assert repr(fit).startswith(
        "SVDFit(ranks=(2, 1), n_neurons=6, n_bins=5, intercept=True"
    )
    assert fit.rank_deficient_neurons.dtype == np.int64
    assert isinstance(fit.n_parameters, int)
    assert isinstance(fit.aic, float)


# ----------------------------------------------- precision-weighted truncation


def _weighted_reference(
    Y: FloatArray,
    X: FloatArray,
    mask: NDArray[np.bool_],
    ranks: list[int],
    ridge: float = 0.0,
) -> list[FloatArray]:
    r"""Independent (M18w): per-neuron ridge least squares on the observed design,
    weights $d_i=\sqrt{\hat\lambda^{\rm OLS}_in_i}$ from its residual, each block
    truncated as $D^{-1}\,\mathrm{trunc}_{r_p}(D\hat B_{{\rm full},p})$ by a dense SVD.
    """
    n, T, P = Y.shape[1], Y.shape[2], X.shape[1]
    coef = np.zeros((n, P, T))
    d = np.zeros(n)
    for i in range(n):
        rows = mask[:, i]
        D = np.column_stack([X[rows], np.ones(rows.sum())])
        beta = np.linalg.solve(D.T @ D + ridge * np.eye(P + 1), D.T @ Y[rows, i, :])
        coef[i] = beta[:P]
        # The residual of the full fit with its refitted intercept.
        level = Y[rows, i, :].mean(axis=0) - X[rows].mean(axis=0) @ beta[:P]
        b = rows.sum() / (rows.sum() + ridge) * level
        rss = float(np.sum((Y[rows, i, :] - X[rows] @ beta[:P] - b) ** 2))
        d[i] = np.sqrt(rows.sum() * T / rss * rows.sum())
    out = []
    for p, r in enumerate(ranks):
        U, s, Vt = np.linalg.svd(d[:, None] * coef[:, p, :], full_matrices=False)
        out.append((U[:, :r] * s[:r]) @ Vt[:r] / d[:, None])
    return out


def test_precision_weighting_is_off_by_default() -> None:
    Y, X, mask = _problem(60)
    stats = sufficient_statistics(Y, X, mask)
    a, b = fit_svd(stats, [2, 1]), fit_svd(stats, [2, 1], precision_weighted=False)
    assert not a.precision_weighted
    for x, y in zip((*a.W, *a.S, *a.B), (*b.W, *b.S, *b.B), strict=True):
        np.testing.assert_array_equal(x, y)
    assert a.aic == b.aic


@pytest.mark.parametrize(
    ("n", "T", "ranks", "ridge"),
    [
        (8, 5, [2, 1], 0.0),
        (6, 6, [3, 2], 0.0),  # n == T: weighting the time axis would broadcast
        (8, 5, [2, 1], 0.7),
    ],
)
def test_weighted_truncation_is_the_weighted_best_approximation(
    n: int, T: int, ranks: list[int], ridge: float
) -> None:
    # (M18w): B_p = D^-1 trunc_{r_p}(D B_full,p), the best rank-r_p
    # approximation of B_full,p in the norm sum_i d_i^2 ||row_i||^2 with
    # d_i = sqrt(lambda_ols_i n_i), against an independent dense computation
    # on a partial mask (so n_i differ and the n_i factor matters).
    Y, X, mask = _problem(61, N=50, n=n, T=T, p_obs=0.6)
    fit = _fit(Y, X, mask, ranks, ridge=ridge, precision_weighted=True)
    assert fit.precision_weighted
    expected = _weighted_reference(Y, X, mask, ranks, ridge)
    for p in range(len(ranks)):
        np.testing.assert_allclose(fit.B[p], expected[p], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(fit.W[p] @ fit.S[p].T, fit.B[p], rtol=1e-13)
        assert np.linalg.matrix_rank(fit.B[p]) == ranks[p]
    # It beats the unweighted truncation in the weighted norm, and loses to it
    # in the unweighted one.
    plain = _fit(Y, X, mask, ranks, ridge=ridge)
    stats = sufficient_statistics(Y, X, mask)
    rss_full = _rss_direct(Y, X, mask, _fit(Y, X, mask, [T, T], ridge=ridge))
    d2 = stats.n_obs**2 * T / rss_full
    for p in range(len(ranks)):
        res_w, res_u = fit.B[p] - plain.B_full[p], plain.B[p] - plain.B_full[p]
        assert np.sum(d2[:, None] * res_w**2) < np.sum(d2[:, None] * res_u**2)
        assert np.sum(res_u**2) < np.sum(res_w**2)


def test_weighted_scores_are_computed_from_its_coefficients() -> None:
    # The intercept refit, RSS, lambda, log-likelihood and AIC follow from
    # the weighted B exactly as from the unweighted one.
    Y, X, mask = _problem(62, N=50, n=8)
    fit = _fit(Y, X, mask, [2, 1], precision_weighted=True)
    stats = sufficient_statistics(Y, X, mask)
    level = stats.Y_mean - np.einsum("ip,pit->it", stats.X_mean, np.stack(fit.B))
    np.testing.assert_allclose(_b0(fit), level, rtol=1e-12, atol=1e-12)
    rss = _rss_direct(Y, X, mask, fit)
    nT = stats.n_obs * stats.n_bins
    np.testing.assert_allclose(fit.noise_precision, nT / rss, rtol=1e-10)
    ll = -0.5 * float(np.sum(nT * (np.log(rss / nT) + 1 + np.log(2 * np.pi))))
    assert fit.log_likelihood == pytest.approx(ll, rel=1e-12)
    assert fit.aic == aic_mod.aic(fit.log_likelihood, fit.n_parameters)
    assert fit.n_parameters == fit_svd(stats, [2, 1]).n_parameters
    full = _fit(Y, X, mask, [5, 5], precision_weighted=True)
    np.testing.assert_allclose(full.B[0], full.B_full[0], rtol=1e-10, atol=1e-12)


def test_equal_weights_give_the_unweighted_fit() -> None:
    # With a full mask and every neuron's full-rank residual of the same size,
    # all d_i are equal and the weighted fit is the unweighted one.
    rng = np.random.default_rng(63)
    N, n, T = 40, 7, 5
    X = rng.normal(size=(N, 2))
    D = np.column_stack([X, np.ones(N)])
    noise = rng.normal(size=(N, n, T))
    for i in range(n):
        e = noise[:, i, :] - D @ np.linalg.lstsq(D, noise[:, i, :], rcond=None)[0]
        noise[:, i, :] = e / np.linalg.norm(e)
    Y = np.einsum("kq,iqt->kit", D, rng.normal(size=(n, 3, T))) + noise
    stats = sufficient_statistics(Y, X, np.ones((N, n), dtype=bool))
    a = fit_svd(stats, [2, 1], precision_weighted=True)
    b = fit_svd(stats, [2, 1])
    for x, y in zip((*a.W, *a.S, *a.B), (*b.W, *b.S, *b.B), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(a.noise_precision, b.noise_precision, rtol=1e-10)
    assert a.log_likelihood == pytest.approx(b.log_likelihood, rel=1e-12)


def test_weighted_fit_is_permutation_and_scale_equivariant() -> None:
    # Neuron permutation permutes the fit; Y -> cY scales B by c and lambda by
    # 1/c^2 and shifts the log-likelihood by -sum n_i T log|c| (the weights are
    # normalised, so they do not change); X[:, p] -> a X[:, p] divides B_p by a.
    Y, X, mask = _problem(64, N=60, n=9, T=6, p_obs=0.7)
    ranks = [2, 1]
    base = _fit(Y, X, mask, ranks, precision_weighted=True)
    perm = np.random.default_rng(0).permutation(9)
    moved = _fit(Y[:, perm], X, mask[:, perm], ranks, precision_weighted=True)
    for x, y in zip(base.B, moved.B, strict=True):
        np.testing.assert_allclose(y, x[perm], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(moved.noise_precision, base.noise_precision[perm])
    for c in (-3.0, 1e4):
        scaled = _fit(c * Y, X, mask, ranks, precision_weighted=True)
        for x, y in zip(base.B, scaled.B, strict=True):
            np.testing.assert_allclose(y, c * x, rtol=1e-9, atol=1e-12 * abs(c))
        np.testing.assert_allclose(scaled.noise_precision * c**2, base.noise_precision)
        m = float(mask.sum()) * 6
        assert scaled.log_likelihood == pytest.approx(
            base.log_likelihood - m * np.log(abs(c)), rel=1e-11
        )
    for a in (1e-3, 50.0):
        X_a = X.copy()
        X_a[:, 1] *= a
        rescaled = _fit(Y, X_a, mask, ranks, precision_weighted=True)
        np.testing.assert_allclose(rescaled.B[1] * a, base.B[1], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(rescaled.B[0], base.B[0], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(rescaled.noise_precision, base.noise_precision)


def test_weighted_fit_needs_a_positive_full_rank_residual() -> None:
    # A neuron the unconstrained fit reproduces exactly has an infinite weight;
    # the unweighted fit at a lower rank accepts it.
    Y, X, mask = _problem(65, N=40, n=6)
    mask[:, 2] = False
    mask[[0, 5, 9], 2] = True  # 3 trials, 2 regressors and the intercept
    stats = sufficient_statistics(Y, X, mask)
    fit_svd(stats, [1, 1])
    with pytest.raises(ValidationError, match=r"neurons \[2\].*precision weights"):
        fit_svd(stats, [1, 1], precision_weighted=True)


def test_precision_weighted_must_be_a_bool() -> None:
    Y, X, mask = _problem(66)
    with pytest.raises(ParameterError, match="precision_weighted must be a bool"):
        _fit(Y, X, mask, [1, 1], precision_weighted=1)
