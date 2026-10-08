"""Tests for `mtdr.mmle`: marginal likelihood, gradients, ECME, refinement, fits.

The reference quantities come from `tests/mmle_dense.py`, dense neuron-by-neuron
implementations of `docs/model.md` with the explicit design matrix and
covariance, which share no code path with the batched kernels under test.
"""

from __future__ import annotations

import functools
import pickle
import re
import types
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import scipy.optimize
from numpy.typing import NDArray

import mmle_dense as dense
import mtdr
import parity_fixtures as pf
from mtdr import mmle
from mtdr.aic import n_parameters_mmle
from mtdr.errors import (
    ConvergenceWarning,
    DesignWarning,
    ParameterError,
    ValidationError,
)
from mtdr.mmle import (
    MMLEFit,
    ecme,
    fit_mmle,
    marginal_log_likelihood,
    marginal_nll_grad_noise,
    marginal_nll_grad_S,
    posterior_weights,
    refine,
    update_intercept,
)
from mtdr.rank_search import greedy_aic
from mtdr.stats import SufficientStats, sufficient_statistics
from mtdr.svd_fit import fit_svd

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


# ------------------------------------------------------------------ problems


class Problem:
    """Random data and a random parameter point (not a fit) of a given shape."""

    def __init__(
        self,
        seed: int,
        N: int,
        n: int,
        T: int,
        ranks: list[int],
        intercept: bool = True,
        p_obs: float = 0.7,
        baseline: float = 1.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        P = len(ranks)
        self.X = rng.normal(size=(N, P))
        self.Y = baseline + rng.normal(size=(N, n, T)) * rng.uniform(0.5, 2, (1, n, 1))
        mask = rng.random((N, n)) < p_obs
        mask[: P + 3] = True
        self.mask: BoolArray = mask
        self.Y_masked = np.where(mask[:, :, None], self.Y, np.nan)
        self.stats = sufficient_statistics(self.Y_masked, self.X, mask)
        self.S = [rng.normal(size=(T, r)) * 0.7 for r in ranks]
        self.lam = rng.uniform(0.3, 3.0, n)
        self.b: FloatArray | None = rng.normal(size=(n, T)) if intercept else None
        self.ranks = ranks


CASES = {
    "base": {"N": 30, "n": 5, "T": 4, "ranks": [2, 0, 1]},
    "one regressor": {"N": 25, "n": 4, "T": 3, "ranks": [2]},
    "one bin": {"N": 20, "n": 6, "T": 1, "ranks": [1, 1]},
    "one neuron": {"N": 15, "n": 1, "T": 5, "ranks": [1, 1]},
    "no intercept": {"N": 30, "n": 5, "T": 4, "ranks": [2, 1], "intercept": False},
    "full mask": {"N": 20, "n": 4, "T": 3, "ranks": [1, 2], "p_obs": 1.0},
    "rank zero": {"N": 20, "n": 4, "T": 3, "ranks": [0, 0]},
    "full-rank bases": {"N": 30, "n": 6, "T": 3, "ranks": [3, 3]},
}


def _problem(name: str, seed: int = 0, **override: Any) -> Problem:
    kwargs: dict[str, Any] = {**CASES[name], **override}
    return Problem(seed, **kwargs)


def _rel(a: Any, b: Any) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-300))


#: The reasons a test that is not about convergence may meet: none, since
#: rounding-level inner ends count as successes (`docs/model.md` § E.3).
ROUNDING: tuple[str, ...] = ()


def _sim_stats(
    seed: int, n: int = 20, T: int = 6, N: int = 60, ranks: Any = (1, 2), **kw: Any
) -> tuple[Any, SufficientStats]:
    sim = mtdr.simulate(
        n_neurons=n, n_bins=T, n_trials=N, ranks=list(ranks), seed=seed, **kw
    )
    return sim, sufficient_statistics(sim.Y, sim.X, sim.mask)


def _start(stats: SufficientStats, ranks: Any, intercept: bool = True) -> MMLEFit:
    """The fit_mmle starting point (M29): SVD bases and precisions, b = y_bar."""
    svd = fit_svd(stats, list(ranks), condition_independent=intercept)
    return MMLEFit.from_parameters(
        stats, svd.S, svd.noise_precision, stats.Y_mean if intercept else None
    )


def _central(f: Any, x: FloatArray, h: float = 1e-6) -> FloatArray:
    out = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        step = h * max(1.0, abs(float(x[idx])))
        xp, xm = x.copy(), x.copy()
        xp[idx] += step
        xm[idx] -= step
        out[idx] = (f(xp) - f(xm)) / (2 * step)
    return out


# ------------------------------------------------------------- the marginal likelihood


@pytest.mark.parametrize("name", CASES)
def test_marginal_log_likelihood_is_the_dense_gaussian(name: str) -> None:
    # (M22)-(M24) with the 2 pi constant: the Gaussian log-density of every neuron's
    # observed responses with the weights integrated out, computed directly from
    # Y with Sigma_i = I / lambda_i + Phi_i Phi_i'.
    pr = _problem(name)
    ll = marginal_log_likelihood(pr.S, pr.lam, pr.b, pr.stats)
    direct = dense.nll(pr.Y, pr.X, pr.mask, pr.S, pr.lam, pr.b)
    assert -ll == pytest.approx(direct.sum(), rel=1e-12)


@pytest.mark.parametrize("name", CASES)
def test_nll_terms_are_the_per_neuron_contributions(name: str) -> None:
    # The N-version hook: per-neuron (M24) without the 2 pi constant.
    pr = _problem(name)
    T = pr.stats.n_bins
    xi, ups = pr.stats.centered(pr.b)
    terms = mmle._nll_terms(pr.S, pr.lam, xi, pr.stats.XtX, ups, pr.stats.n_obs, T)
    direct = dense.nll(pr.Y, pr.X, pr.mask, pr.S, pr.lam, pr.b)
    constant = 0.5 * pr.stats.n_obs * T * np.log(2 * np.pi)
    np.testing.assert_allclose(terms + constant, direct, rtol=1e-12)


def test_never_observed_neuron_contributes_nothing() -> None:
    pr = _problem("base")
    mask = pr.mask.copy()
    mask[:, 2] = False
    stats = sufficient_statistics(pr.Y, pr.X, mask)
    keep = [0, 1, 3, 4]
    sub = sufficient_statistics(pr.Y[:, keep], pr.X, mask[:, keep])
    assert pr.b is not None
    ll = marginal_log_likelihood(pr.S, pr.lam, pr.b, stats)
    assert ll == pytest.approx(
        marginal_log_likelihood(pr.S, pr.lam[keep], pr.b[keep], sub), rel=1e-13
    )
    _, grad = marginal_nll_grad_noise(pr.S, pr.lam, pr.b, stats)
    assert grad[2] == 0.0
    W, cov = posterior_weights(pr.S, pr.lam, pr.b, stats)
    assert all(np.all(w[2] == 0) for w in W)
    np.testing.assert_allclose(cov[2], np.eye(cov.shape[1]), atol=1e-15)


def test_ridge_is_subtracted_once() -> None:
    pr = _problem("base")
    g = 0.37
    ll = marginal_log_likelihood(pr.S, pr.lam, pr.b, pr.stats)
    penalised = marginal_log_likelihood(pr.S, pr.lam, pr.b, pr.stats, basis_ridge=g)
    assert penalised == pytest.approx(
        ll - 0.5 * g * sum(float((s**2).sum()) for s in pr.S), rel=1e-14
    )


# ------------------------------------------------------------------ gradients


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("ridge", [0.0, 0.7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradient_in_the_bases_matches_central_differences(
    name: str, ridge: float, seed: int
) -> None:
    pr = _problem(name, seed)
    value, grad = marginal_nll_grad_S(pr.S, pr.lam, pr.b, pr.stats, basis_ridge=ridge)
    assert value == pytest.approx(
        -marginal_log_likelihood(pr.S, pr.lam, pr.b, pr.stats, basis_ridge=ridge),
        rel=1e-14,
    )
    assert [g.shape for g in grad] == [s.shape for s in pr.S]
    if sum(pr.ranks) == 0:
        return
    flat = np.concatenate([s.ravel() for s in pr.S])
    bounds = np.cumsum([0] + [s.size for s in pr.S])

    def f(x: FloatArray) -> float:
        S = [
            x[a:b].reshape(s.shape)
            for a, b, s in zip(bounds, bounds[1:], pr.S, strict=False)
        ]
        return -marginal_log_likelihood(S, pr.lam, pr.b, pr.stats, basis_ridge=ridge)

    fd = _central(f, flat)
    analytic = np.concatenate([g.ravel() for g in grad])
    assert _rel(analytic, fd) < 1e-6


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradient_in_the_noise_matches_central_differences(
    name: str, seed: int
) -> None:
    pr = _problem(name, seed)
    value, grad = marginal_nll_grad_noise(pr.S, pr.lam, pr.b, pr.stats)
    assert value == pytest.approx(
        -marginal_log_likelihood(pr.S, pr.lam, pr.b, pr.stats), rel=1e-14
    )

    def f(lam: FloatArray) -> float:
        return -marginal_log_likelihood(pr.S, lam, pr.b, pr.stats)

    assert _rel(grad, _central(f, pr.lam)) < 1e-6


@pytest.mark.parametrize("name", CASES)
def test_gradient_equals_the_reference_form_m26a(name: str) -> None:
    # (M26b), the expected complete-data form the port implements, equals the
    # reference's (M26a) built with explicit Kronecker matrices and R_i.
    pr = _problem(name)
    _, grad = marginal_nll_grad_S(pr.S, pr.lam, pr.b, pr.stats, basis_ridge=0.2)
    reference = dense.grad_bases_m26a(pr.stats, pr.S, pr.lam, pr.b, g=0.2)
    for ours, theirs in zip(grad, reference, strict=True):
        np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("name", CASES)
def test_noise_gradient_equals_the_reference_form_m27a(name: str) -> None:
    pr = _problem(name)
    _, grad = marginal_nll_grad_noise(pr.S, pr.lam, pr.b, pr.stats)
    reference = dense.grad_noise_m27a(pr.stats, pr.S, pr.lam, pr.b)
    np.testing.assert_allclose(grad, reference, rtol=1e-9, atol=1e-11)


# ------------------------------------------------------------- posterior and intercept


@pytest.mark.parametrize("name", CASES)
def test_posterior_is_the_dense_gaussian_posterior(name: str) -> None:
    # (M25), (M36): direct conditioning of w_i ~ N(0, I) on the observed responses.
    pr = _problem(name)
    W, cov = posterior_weights(pr.S, pr.lam, pr.b, pr.stats)
    means, covs = dense.posterior(pr.Y, pr.X, pr.mask, pr.S, pr.lam, pr.b)
    stacked = np.concatenate(W, axis=1)
    np.testing.assert_allclose(stacked, means, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cov, covs, rtol=1e-10, atol=1e-12)
    assert [w.shape for w in W] == [(pr.Y.shape[1], r) for r in pr.ranks]


@pytest.mark.parametrize("name", CASES)
def test_posterior_mean_solves_its_normal_equations(name: str) -> None:
    # (M36): C_i mu_i = lambda_i u_i, with C_i = I + lambda_i Phi_i' Phi_i built
    # independently from the explicit design; W_cov is symmetric positive
    # definite and is the inverse of C_i.
    pr = _problem(name)
    W, cov = posterior_weights(pr.S, pr.lam, pr.b, pr.stats)
    mu = np.concatenate(W, axis=1)
    for i in range(pr.Y.shape[1]):
        rows = np.flatnonzero(pr.mask[:, i])
        Phi = dense.design(pr.X[rows], pr.S)
        z = pr.Y[rows, i, :].ravel()
        if pr.b is not None:
            z = z - np.tile(pr.b[i], rows.size)
        C = np.eye(mu.shape[1]) + pr.lam[i] * Phi.T @ Phi
        rhs = pr.lam[i] * Phi.T @ z
        assert np.linalg.norm(C @ mu[i] - rhs) <= 1e-10 * max(np.linalg.norm(rhs), 1.0)
        np.testing.assert_allclose(cov[i] @ C, np.eye(mu.shape[1]), atol=1e-10)
        np.testing.assert_array_equal(cov[i], cov[i].T)
        if mu.shape[1]:
            assert np.linalg.eigvalsh(cov[i]).min() > 0


@pytest.mark.parametrize("name", [c for c in CASES if c != "no intercept"])
def test_intercept_is_the_gls_mean(name: str) -> None:
    # (M33) is the GLS mean of zeta_i ~ N(1 (x) b_i, Sigma_i); the port's centred
    # form equals both the dense GLS solution and the literal (M33).
    pr = _problem(name)
    b = update_intercept(pr.S, pr.lam, pr.stats)
    np.testing.assert_allclose(
        b,
        dense.gls_intercept(pr.Y, pr.X, pr.mask, pr.S, pr.lam),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        b, dense.intercept_m33(pr.stats, pr.S, pr.lam), rtol=1e-10, atol=1e-12
    )


def test_intercept_maximises_the_marginal_likelihood() -> None:
    pr = _problem("base")
    b = update_intercept(pr.S, pr.lam, pr.stats)

    def f(x: FloatArray) -> float:
        return -marginal_log_likelihood(pr.S, pr.lam, x, pr.stats)

    grad = _central(f, b)
    assert np.abs(grad).max() < 1e-6
    rng = np.random.default_rng(1)
    for _ in range(5):
        assert f(b + 1e-3 * rng.normal(size=b.shape)) > f(b)


@pytest.mark.parametrize("baseline", [1e5, 1e7])
def test_intercept_costs_no_accuracy_with_a_large_baseline(baseline: float) -> None:
    # The centred form has no cancellation, so b(Y + c) - c equals b(Y)
    # to the rounding of Y + c itself; the literal (M33) on the raw moments
    # loses digits in proportion to the baseline.
    pr = _problem("base", baseline=0.0)
    shifted = sufficient_statistics(pr.Y_masked + baseline, pr.X, pr.mask)
    b0 = update_intercept(pr.S, pr.lam, pr.stats)
    b1 = update_intercept(pr.S, pr.lam, shifted)
    rounding = baseline * np.finfo(float).eps
    assert np.abs(b1 - baseline - b0).max() < 100 * rounding
    literal = dense.intercept_m33(shifted, pr.S, pr.lam)
    print(
        f"\nbaseline {baseline:g}: centred error "
        f"{np.abs(b1 - baseline - b0).max():.2e}, literal (M33) error "
        f"{np.abs(literal - baseline - b0).max():.2e}"
    )


# ------------------------------------------------------------------ ECME steps


def _state_at(
    stats: SufficientStats, fit: MMLEFit
) -> tuple[Any, FloatArray, FloatArray, FloatArray]:
    blocks = mmle._blocks(fit.S)
    lam = np.array(fit.noise_precision)
    xi, ups = stats.centered(fit.intercept)
    st = mmle._state(blocks, lam, xi, ups, stats.XtX, stats.n_obs, stats.n_bins)
    return st, lam, xi, ups


def _nll(stats: SufficientStats, S: Any, lam: FloatArray, b: Any) -> float:
    return -marginal_log_likelihood(S, lam, b, stats)


@pytest.mark.parametrize("seed", range(8))
def test_each_ecme_step_lowers_the_marginal_nll(seed: int) -> None:
    # The precision step (M30), the multicycle basis step (M31) and the
    # intercept step (M33) are each conditional maximisations, so each one
    # alone does not raise (M24). Checked step by step from the SVD start.
    _, stats = _sim_stats(seed, drop_prob=0.3, noise_precision=None)
    fit = _start(stats, (1, 2))
    for _ in range(3):
        st, lam, xi, ups = _state_at(stats, fit)
        before = _nll(stats, fit.S, lam, fit.intercept)
        lam1 = mmle._precision_step(st, ups, stats.n_obs, stats.n_bins)
        after_lambda = _nll(stats, fit.S, lam1, fit.intercept)
        blocks = mmle._basis_step(st, lam1, xi)
        S1 = mmle._split(blocks.S, blocks.bounds)
        after_bases = _nll(stats, S1, lam1, fit.intercept)
        b1 = mmle._intercept_step(blocks, lam1, stats)
        after_intercept = _nll(stats, S1, lam1, b1)
        scale = 1e-12 * abs(before)
        assert after_lambda <= before + scale
        assert after_bases <= after_lambda + scale
        assert after_intercept <= after_bases + scale
        assert after_intercept < before
        fit = MMLEFit.from_parameters(stats, S1, lam1, b1)


def test_precision_step_is_the_count_over_the_expected_residual() -> None:
    # (M30): lambda'_i = n_i T / E||zeta_i - Phi_i w_i||^2 under the posterior,
    # the expectation computed densely.
    pr = _problem("base")
    fit = MMLEFit.from_parameters(pr.stats, pr.S, pr.lam, pr.b)
    st, _, _, ups = _state_at(pr.stats, fit)
    lam1 = mmle._precision_step(st, ups, pr.stats.n_obs, pr.stats.n_bins)
    E = dense.expected_residual(pr.Y, pr.X, pr.mask, pr.S, pr.lam, pr.b)
    np.testing.assert_allclose(lam1, pr.stats.n_obs * pr.stats.n_bins / E, rtol=1e-10)


def test_basis_step_solves_its_normal_equations() -> None:
    # (M31): at S' the expected complete-data gradient, with the posterior
    # redone at (lambda', S), vanishes.
    pr = _problem("base", N=60, n=8)
    fit = MMLEFit.from_parameters(pr.stats, pr.S, pr.lam, pr.b)
    st, _, xi, ups = _state_at(pr.stats, fit)
    lam1 = mmle._precision_step(st, ups, pr.stats.n_obs, pr.stats.n_bins)
    new = mmle._basis_step(st, lam1, xi)
    Cinv, mu, _, _ = mmle._posterior(lam1, st.K, st.u)
    xig = xi[:, new.group, :]
    grad = mmle._grad_bases(new.S, st.Aexp, mu, Cinv, lam1, xig)
    scale = np.abs(mmle._grad_bases(st.blocks.S, st.Aexp, mu, Cinv, lam1, xig)).max()
    assert np.abs(grad).max() < 1e-10 * scale


def test_precision_step_rejects_a_vanishing_expected_residual() -> None:
    pr = _problem("base")
    fit = MMLEFit.from_parameters(pr.stats, pr.S, pr.lam, pr.b)
    st, _, _, ups = _state_at(pr.stats, fit)
    zero = st._replace(K=np.zeros_like(st.K), u=np.zeros_like(st.u))
    ups0 = ups.copy()
    ups0[3] = 0.0
    with pytest.raises(ValidationError, match=r"neurons \[3\] have zero expected"):
        mmle._precision_step(zero, ups0, pr.stats.n_obs, pr.stats.n_bins)


@pytest.mark.parametrize("r", [0, 1, 2, 7])
def test_lower_inverse_is_the_inverse(r: int) -> None:
    rng = np.random.default_rng(r)
    G = rng.normal(size=(5, r, 2 * r + 1))
    C = G @ G.transpose(0, 2, 1) + np.eye(r)
    Linv, logdet = mmle._cholesky_inverse(C)
    L = np.linalg.cholesky(C)
    np.testing.assert_allclose(Linv, np.linalg.inv(L), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(logdet, np.linalg.slogdet(C)[1], rtol=1e-12, atol=1e-14)


def test_cholesky_failure_names_the_neurons() -> None:
    # A non-positive-definite or non-finite precision raises LinAlgError with
    # the neuron indices, never a silent NaN.
    C = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    C[1, 2, 2] = -1.0
    with pytest.raises(np.linalg.LinAlgError, match=r"neurons \[1\]"):
        mmle._cholesky_inverse(C)
    C = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    C[2, 0, 0] = np.nan
    with pytest.raises(np.linalg.LinAlgError, match=r"neurons \[2\]"):
        mmle._cholesky_inverse(C)


def test_basis_system_without_signal_raises() -> None:
    # A regressor that is zero on every trial makes the (M31) system singular;
    # that is a data condition, so a ValidationError.
    pr = _problem("base")
    X = pr.X.copy()
    X[:, 2] = 0.0
    stats = sufficient_statistics(pr.Y_masked, X, pr.mask)
    fit = MMLEFit.from_parameters(stats, pr.S, pr.lam, pr.b)
    st, _, xi, ups = _state_at(stats, fit)
    lam1 = mmle._precision_step(st, ups, stats.n_obs, stats.n_bins)
    with pytest.raises(ValidationError, match="not positive definite"):
        mmle._basis_step(st, lam1, xi)


# ------------------------------------------------------------------ ecme


def test_ecme_trace_and_fit_agree() -> None:
    _, stats = _sim_stats(0, drop_prob=0.2)
    init = _start(stats, (1, 2))
    with pytest.warns(
        ConvergenceWarning,
        match=r"hit its iteration cap \(max_iter = 15; `MTDR\(ecme_max_iter=",
    ):
        fit, trace = ecme(stats, init, max_iter=15, tol=1e-300)
    assert trace.shape == (16,)
    assert trace[0] == pytest.approx(-init.log_likelihood, rel=1e-14)
    assert trace[-1] == pytest.approx(-fit.log_likelihood, rel=1e-14)
    assert np.all(np.diff(trace) < 0)
    assert dict(fit.n_iter) == {"ecme": 15}
    assert not fit.converged
    W, cov = posterior_weights(fit.S, fit.noise_precision, fit.intercept, stats)
    for a, b in zip(W, fit.W, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(cov, fit.W_cov)
    assert fit.objective == fit.log_likelihood


def test_ecme_default_tolerance_converges_in_a_few_iterations() -> None:
    # The reference's stopcrit = 1 (`docs/model.md` § C.6): a warm start, not an
    # estimator.
    _, stats = _sim_stats(1, n=40, T=10, N=150, ranks=(2, 1, 2), drop_prob=0.3)
    fit, trace = ecme(stats, _start(stats, (2, 1, 2)))
    assert fit.converged
    assert 1 <= fit.n_iter["ecme"] <= 10
    assert trace.size == fit.n_iter["ecme"] + 1


def test_ecme_tolerance_is_a_strict_threshold() -> None:
    # (M34): stop when the change is below tol; a change exactly at tol continues.
    _, stats = _sim_stats(2)
    init = _start(stats, (1, 2))
    with pytest.warns(ConvergenceWarning):
        one, _ = ecme(stats, init, max_iter=1, tol=1e-300)
    change = mmle._relative_change(
        (init.noise_precision, np.concatenate(init.S, 1), init.intercept),
        (one.noise_precision, np.concatenate(one.S, 1), one.intercept),
        1e-12,
    )
    at, _ = ecme(stats, init, max_iter=2, tol=float(np.nextafter(change, np.inf)))
    assert at.n_iter["ecme"] == 1
    assert at.converged
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        equal, _ = ecme(stats, init, max_iter=2, tol=change)
    assert equal.n_iter["ecme"] == 2


def test_ecme_with_zero_iterations_returns_the_start() -> None:
    _, stats = _sim_stats(3)
    init = _start(stats, (1, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit, trace = ecme(stats, init, max_iter=0)
    assert trace.shape == (1,)
    assert fit.converged
    assert fit.n_iter["ecme"] == 0
    for a, b in zip(fit.S, init.S, strict=True):
        np.testing.assert_array_equal(a, b)
    assert fit.log_likelihood == init.log_likelihood


def test_ecme_reports_an_increase(monkeypatch: pytest.MonkeyPatch) -> None:
    # The monotonicity check is wired: a corrupted intercept step that raises the
    # NLL gives a ConvergenceWarning and converged=False.
    _, stats = _sim_stats(4)
    init = _start(stats, (1, 2))
    good = mmle._intercept_step

    def bad(blocks: Any, lam: FloatArray, st: SufficientStats) -> FloatArray:
        return good(blocks, lam, st) + 0.5

    monkeypatch.setattr(mmle, "_intercept_step", bad)
    with pytest.warns(
        ConvergenceWarning, match=r"raised the marginal NLL at iterations \[1"
    ):
        fit, trace = ecme(stats, init, max_iter=3)
    assert not fit.converged
    assert trace[1] > trace[0]


def test_ecme_tolerates_rounding_level_rises(monkeypatch: pytest.MonkeyPatch) -> None:
    # MONOTONE_RTOL separates rounding from a real increase.
    _, stats = _sim_stats(4)
    init = _start(stats, (1, 2))
    good = mmle._intercept_step
    tiny = 1e-13

    def nudged(blocks: Any, lam: FloatArray, st: SufficientStats) -> FloatArray:
        return good(blocks, lam, st) + tiny

    monkeypatch.setattr(mmle, "_intercept_step", nudged)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit, _ = ecme(stats, init)
    assert fit.converged


def test_ecme_without_bases_is_the_intercept_only_model() -> None:
    # With every rank 0 the model is y = b + noise and its MLE is b = y_bar,
    # lambda = n_i T / (energy about the mean): the SVD start (M29) already is
    # it, so one ECME iteration changes nothing and stops even at tol = 1e-300.
    _, stats = _sim_stats(5)
    init = _start(stats, (0, 0))
    fit, trace = ecme(stats, init, tol=1e-300)
    assert fit.converged
    assert fit.n_iter["ecme"] == 1
    assert fit.intercept is not None
    np.testing.assert_allclose(fit.intercept, stats.Y_mean, rtol=1e-14)
    np.testing.assert_allclose(
        fit.noise_precision, stats.n_obs * stats.n_bins / stats.YtY_c, rtol=1e-12
    )
    assert fit.W_cov.shape == (stats.n_neurons, 0, 0)
    assert trace[1] <= trace[0]


def test_ecme_propagates_the_start_convergence() -> None:
    _, stats = _sim_stats(6)
    init = _start(stats, (1, 2))
    fields = {name: getattr(init, name) for name in init.__dataclass_fields__}
    fields["converged"] = False
    fit, _ = ecme(stats, MMLEFit(**fields))
    assert not fit.converged


def test_matlab_compat_is_wired_and_departs_from_the_default() -> None:
    # The stale-variable path runs and gives a different iterate. Parity
    # with the reference is a fixture test (test_mmle_parity.py), not here.
    _, stats = _sim_stats(7, ranks=(2, 2))
    init = _start(stats, (2, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        default, t0 = ecme(stats, init, max_iter=3, tol=1e-300)
        compat, t1 = ecme(stats, init, max_iter=3, tol=1e-300, matlab_compat=True)
    assert np.all(np.isfinite(t1))
    assert t1[0] == t0[0]
    assert max(_rel(a, b) for a, b in zip(compat.B, default.B, strict=True)) > 1e-6
    assert compat.intercept is not None
    assert default.intercept is not None
    assert _rel(compat.intercept, default.intercept) > 1e-8


def test_matlab_compat_steps_reduce_to_the_consistent_ones() -> None:
    # With lambda' = lambda the reference's stale S-step's G_i is the
    # symmetric posterior second moment, so its system equals (M31); with
    # S_old = S_new the mixed C_i^x is C_i and the b-step is (M33).
    pr = _problem("base", N=60, n=8)
    fit = MMLEFit.from_parameters(pr.stats, pr.S, pr.lam, pr.b)
    st, lam, xi, _ = _state_at(pr.stats, fit)
    np.testing.assert_allclose(
        mmle._basis_step_reference(st, lam, xi).S,
        mmle._basis_step(st, lam, xi).S,
        rtol=1e-9,
        atol=1e-12,
    )
    blocks = mmle._blocks(pr.S)
    np.testing.assert_allclose(
        mmle._intercept_step_reference(blocks, blocks, lam, pr.stats),
        mmle._intercept_step(blocks, lam, pr.stats),
        rtol=1e-10,
        atol=1e-12,
    )


def test_matlab_compat_system_is_not_symmetric() -> None:
    # With lambda' != lambda the reference's GG (`ECMEtdr.m`, filled from its
    # upper block triangle and that triangle's transpose) is a general matrix
    # for r_p > 1; the compat step must not symmetrise it.
    pr = _problem("base", N=60, n=8, ranks=[2, 2, 1])
    fit = MMLEFit.from_parameters(pr.stats, pr.S, pr.lam, pr.b)
    st, _, xi, ups = _state_at(pr.stats, fit)
    lam1 = mmle._precision_step(st, ups, pr.stats.n_obs, pr.stats.n_bins)
    compat = mmle._basis_step_reference(st, lam1, xi).S
    default = mmle._basis_step(st, lam1, xi).S
    assert _rel(compat, default) > 1e-6


# ------------------------------------------------------------------ refine


@pytest.mark.parametrize("ridge", [0.0, 0.5])
@pytest.mark.parametrize("seed", range(6))
def test_refine_never_lowers_the_objective(ridge: float, seed: int) -> None:
    _, stats = _sim_stats(seed, drop_prob=0.3, noise_precision=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warm, _ = ecme(stats, _start(stats, (1, 2)))
        fit = refine(stats, warm, basis_ridge=ridge)
    start = marginal_log_likelihood(
        warm.S, warm.noise_precision, warm.intercept, stats, basis_ridge=ridge
    )
    assert fit.objective >= start
    assert fit.objective == pytest.approx(
        marginal_log_likelihood(
            fit.S, fit.noise_precision, fit.intercept, stats, basis_ridge=ridge
        ),
        rel=1e-14,
    )


def test_refine_reaches_a_stationary_point() -> None:
    _, stats = _sim_stats(8, n=30, T=8, N=120, ranks=(2, 1))
    # The basis step runs to minFunc's absolute progTol, so the next precision
    # step can start at its rounding floor and end on an abnormal line search;
    # such an end is polished and counts as converged below a scale-free
    # residual of 1e-8. Stationarity is what this test checks; a warning with
    # any reason fails it.
    fit, caught = pf.run(lambda: fit_mmle(stats, [2, 1]), ROUNDING)
    assert fit.converged == (not caught.reasons)
    _, gS = marginal_nll_grad_S(fit.S, fit.noise_precision, fit.intercept, stats)
    _, gl = marginal_nll_grad_noise(fit.S, fit.noise_precision, fit.intercept, stats)
    # Relative to the size of the gradient at the SVD start; the log-precision
    # gradient is scale-free. Offline, over seeds 0-99 of this configuration,
    # at most 2.4e-4 and 6.6e-5 (refine stops on the relative parameter change,
    # not on these).
    start = _start(stats, (2, 1))
    _, gS0 = marginal_nll_grad_S(start.S, start.noise_precision, start.intercept, stats)
    assert max(np.abs(g).max() for g in gS) < 2e-3 * max(np.abs(g).max() for g in gS0)
    assert np.abs(gl * fit.noise_precision).max() < 1e-3


def test_refine_with_zero_iterations_returns_the_start() -> None:
    _, stats = _sim_stats(9)
    warm, _ = ecme(stats, _start(stats, (1, 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = refine(stats, warm, max_iter=0)
    assert fit.converged
    assert fit.n_iter["refine"] == 0
    assert fit.log_likelihood == warm.log_likelihood


def test_refine_reports_inner_failures() -> None:
    _, stats = _sim_stats(10)
    warm, _ = ecme(stats, _start(stats, (1, 2)))
    with pytest.warns(ConvergenceWarning, match="iteration 1, bases: STOP"):
        fit = refine(stats, warm, optimizer_max_iter=1)
    assert not fit.converged


def test_refine_reports_the_precision_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    # A start ten times too precise with a box of half-width 1e-3 in log
    # precision: every neuron ends on the bound, which counts as a failure.
    _, stats = _sim_stats(11)
    warm, _ = ecme(stats, _start(stats, (1, 2)))
    off = MMLEFit.from_parameters(
        stats, warm.S, 10 * warm.noise_precision, warm.intercept
    )
    monkeypatch.setattr(mmle, "LOG_PRECISION_BOUND", 1e-3)
    with pytest.warns(ConvergenceWarning) as record:
        fit = refine(stats, off, max_iter=1)
    # The bound was hit in the final iteration, so the message may say the
    # likelihood is unbounded.
    text = str(record[0].message)
    assert "ended on the step bound in iteration 1" in text
    assert "were on the step bound in the final iteration" in text
    assert not fit.converged
    np.testing.assert_allclose(
        np.log(fit.noise_precision / off.noise_precision), -1e-3, rtol=1e-9
    )


def test_refine_reports_its_cap() -> None:
    _, stats = _sim_stats(12)
    warm, _ = ecme(stats, _start(stats, (1, 2)))
    with pytest.warns(
        ConvergenceWarning,
        match=r"cap \(max_iter = 1; `MTDR\(refine_max_iter=\.\.\.\)`\) before",
    ):
        fit = refine(stats, warm, max_iter=1, tol=1e-300)
    assert not fit.converged
    assert fit.n_iter == {
        "ecme": warm.n_iter["ecme"],
        "refine": 1,
    }


# ------------------------------------------------------------------ fit_mmle


@pytest.mark.parametrize("seed", range(12))
def test_recovers_the_simulated_coefficients(seed: int) -> None:
    # Equal precisions, n = 60, T = 10, N = 200, ranks (2, 1, 2), 30 % dropped.
    # Offline, over seeds 0-99, the stacked relative Frobenius error
    # ||B_hat - B||_F / ||B||_F had median 0.025 and maximum 0.042, every fit
    # converged; the bound below is about twice that maximum.
    sim, stats = _sim_stats(
        seed, n=60, T=10, N=200, ranks=(2, 1, 2), drop_prob=0.3, noise_precision=1.0
    )
    fit = fit_mmle(stats, [2, 1, 2])
    names = sim.regressor_names
    err = np.sqrt(
        sum(np.linalg.norm(fit.B[p] - sim.B[n]) ** 2 for p, n in enumerate(names))
    ) / np.sqrt(sum(np.linalg.norm(sim.B[n]) ** 2 for n in names))
    assert err < 0.08
    assert fit.converged
    assert np.median(np.abs(np.log(fit.noise_precision))) < 0.1


def test_beats_the_svd_estimator_under_heterogeneous_noise() -> None:
    # The demo's Exponential precisions: offline, the MMLE's largest per-regressor
    # relative error was below the SVD's in 94 of seeds 0-99 (n = 60, T = 10,
    # N = 200). The criterion is the ratio of the summed relative errors over
    # 20 seeds; over the ten blocks of seeds 0-199 it ranged from 0.51 to 0.92
    # (0.51 for the block tested here).
    mmle_err, svd_err = 0.0, 0.0
    for seed in range(20):
        sim, stats = _sim_stats(seed, n=60, T=10, N=200, ranks=(2, 1, 2), drop_prob=0.3)
        fit = fit_mmle(stats, [2, 1, 2])
        svd = fit_svd(stats, [2, 1, 2])
        for p, n in enumerate(sim.regressor_names):
            mmle_err += _rel(fit.B[p], sim.B[n])
            svd_err += _rel(svd.B[p], sim.B[n])
    assert mmle_err < 0.95 * svd_err


def test_fit_is_consistent_with_its_parts() -> None:
    _, stats = _sim_stats(13, drop_prob=0.2)
    fit = fit_mmle(stats, [1, 2])
    W, cov = posterior_weights(fit.S, fit.noise_precision, fit.intercept, stats)
    for p in range(2):
        np.testing.assert_array_equal(W[p], fit.W[p])
        np.testing.assert_allclose(fit.B[p], fit.W[p] @ fit.S[p].T, rtol=1e-15)
    np.testing.assert_array_equal(cov, fit.W_cov)
    ll = marginal_log_likelihood(fit.S, fit.noise_precision, fit.intercept, stats)
    assert fit.log_likelihood == ll == fit.objective
    assert fit.n_parameters == n_parameters_mmle([1, 2], 20, 6)
    assert fit.aic == mtdr.aic.aic(ll, fit.n_parameters)
    assert set(fit.n_iter) == {"ecme", "refine"}
    assert fit.intercept is not None
    np.testing.assert_allclose(
        fit.intercept,
        update_intercept(fit.S, fit.noise_precision, stats),
        rtol=1e-12,
        atol=1e-12,
    )


def test_one_more_ecme_iteration_leaves_the_optimum_in_place() -> None:
    # At a stationary point of (M24) the ECME map is the identity: the
    # gradient (M26b) and the M-step (M31) share their fixed points.
    _, stats = _sim_stats(14, n=30, T=8, N=150, ranks=(2, 1))
    with warnings.catch_warnings():
        # At these tolerances the loops stop on rounding, not on their tests.
        warnings.simplefilter("ignore", ConvergenceWarning)
        tight = fit_mmle(
            stats, [2, 1], refine_max_iter=50, refine_tol=1e-14, optimizer_tol=1e-10
        )
        again, trace = ecme(stats, tight, max_iter=1, tol=1e-300)
    # Offline, over seeds 0-99 of this configuration: at most 1.0e-6 and 8.0e-13.
    assert max(_rel(a, b) for a, b in zip(again.B, tight.B, strict=True)) < 1e-5
    assert abs(trace[1] - trace[0]) < 1e-11 * abs(trace[0])


@pytest.mark.parametrize("kind", ["none", "svd", "mmle"])
def test_initialiser_paths(kind: str) -> None:
    _, stats = _sim_stats(15)
    # Rounding-level inner ends are recorded, not ignored (Windows, CI).
    reference, _ = pf.run(lambda: fit_mmle(stats, [1, 2]), ROUNDING)
    init: Any = {
        "none": None,
        "svd": fit_svd(stats, [1, 2]),
        "mmle": _start(stats, (1, 2)),
    }[kind]
    fit, _ = pf.run(lambda: fit_mmle(stats, [1, 2], init=init), ROUNDING)
    assert fit.log_likelihood == pytest.approx(reference.log_likelihood, rel=1e-13)


def test_starting_from_a_fit_continues_it() -> None:
    _, stats = _sim_stats(16)
    first = fit_mmle(stats, [1, 2])
    again = fit_mmle(stats, [1, 2], init=first)
    assert again.log_likelihood >= first.log_likelihood - 1e-9 * abs(
        first.log_likelihood
    )


def test_rank_zero_everywhere_is_the_intercept_only_mle() -> None:
    _, stats = _sim_stats(17)
    fit = fit_mmle(stats, [0, 0])
    assert fit.intercept is not None
    np.testing.assert_allclose(fit.intercept, stats.Y_mean, rtol=1e-12)
    np.testing.assert_allclose(
        fit.noise_precision, stats.n_obs * stats.n_bins / stats.YtY_c, rtol=1e-6
    )
    assert fit.total_rank == 0
    assert all(w.shape == (20, 0) for w in fit.W)
    assert fit.converged


def test_full_rank_bases_for_every_regressor() -> None:
    # r_p = T for every p (n >= T): the bases span every time course.
    _, stats = _sim_stats(18, n=12, T=4, N=80, ranks=(4, 4), drop_prob=0.2)
    fit = fit_mmle(stats, [4, 4])
    assert all(s.shape == (4, 4) for s in fit.S)
    assert np.isfinite(fit.log_likelihood)
    smaller = fit_mmle(stats, [2, 2])
    assert fit.log_likelihood >= smaller.log_likelihood


def test_neuron_observed_on_few_trials() -> None:
    # On 3 trials, with 3 regressors and the intercept, neuron 4 has no
    # residual degrees of freedom and is rejected. On 6 trials on which x2 is
    # constant its design is rank-deficient (the SVD start warns and uses the
    # minimum-norm solution) but 3 residual degrees of freedom remain, and it
    # is fitted.
    sim, _ = _sim_stats(19, n=10, T=5, N=60, ranks=(1, 1, 1))
    mask = sim.mask.copy()
    mask[:, 4] = False
    mask[[0, 1, 3], 4] = True  # three distinct design rows
    with pytest.raises(ValidationError, match=r"neurons \[4\] have no residual"):
        fit_mmle(sufficient_statistics(sim.Y, sim.X, mask), [1, 1, 1])
    mask[:, 4] = False
    mask[np.flatnonzero(sim.X[:, 2] == -1.0)[:6], 4] = True
    stats = sufficient_statistics(sim.Y, sim.X, mask)
    with pytest.warns(DesignWarning, match="rank-deficient"):
        fit = fit_mmle(stats, [1, 1, 1])
    np.testing.assert_array_equal(fit.rank_deficient_neurons, [4])
    assert np.isfinite(fit.log_likelihood)
    assert 0 < fit.noise_precision[4] < 1e3


def test_without_the_intercept() -> None:
    # Seed 22 is one whose fit raises no ConvergenceWarning (warnings fail the
    # tests): a few seeds of this configuration, seed 20 among them, end a
    # precision step on a rounding-level line-search failure, which is not
    # this test's subject.
    sim = mtdr.simulate(
        n_neurons=20,
        n_bins=6,
        n_trials=80,
        ranks=[1, 2],
        condition_independent=False,
        seed=22,
    )
    stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    fit = fit_mmle(stats, [1, 2], condition_independent=False)
    assert fit.intercept is None
    assert fit.n_parameters == n_parameters_mmle(
        [1, 2], 20, 6, condition_independent=False
    )
    err = max(_rel(fit.B[p], sim.B[n]) for p, n in enumerate(sim.regressor_names))
    assert err < 0.2


@pytest.mark.parametrize(("n", "T"), [(1, 5), (6, 1)])
def test_singleton_shapes(n: int, T: int) -> None:
    # (n, T) = (1, 5) and (6, 1) run (the reference fails: `squeeze` returns the
    # wrong orientation at T = 1, its `slow*` fallbacks misread n = 1).
    _, stats = _sim_stats(21, n=n, T=T, N=40, ranks=(1, 1))
    fit = fit_mmle(stats, [1, 1])
    assert fit.W_cov.shape == (n, 2, 2)
    assert fit.S[0].shape == (T, 1)
    assert np.isfinite(fit.log_likelihood)


def test_deterministic() -> None:
    _, stats = _sim_stats(22)
    # A rounding-level precision-step end may warn on some platforms
    # (SciPy 1.13 here); determinism is the subject, so it is recorded.
    a, _ = pf.run(lambda: fit_mmle(stats, [1, 2]), ROUNDING)
    b, _ = pf.run(lambda: fit_mmle(stats, [1, 2]), ROUNDING)
    for x, y in zip(a.B, b.B, strict=True):
        np.testing.assert_array_equal(x, y)
    assert a.log_likelihood == b.log_likelihood


def test_greedy_search_records_the_mmle_estimator() -> None:
    _, stats = _sim_stats(23, n=40, T=8, N=200, ranks=(2, 1), noise_precision=1.0)
    # Rounding-level inner ends are recorded, not ignored (SciPy 1.13).
    (best, history), _ = pf.run(
        lambda: greedy_aic(
            functools.partial(fit_mmle, stats), lambda f, r: f.aic, [1, 1], max_rank=8
        ),
        ROUNDING,
    )
    assert history.estimator == "mmle"
    assert history.final_ranks() == {"x0": 2, "x1": 1}
    assert isinstance(best, MMLEFit)
    assert best.ranks == (2, 1)


@pytest.mark.parametrize(
    ("ranks", "ci"),
    [([2, 1], True), ([1, 1, 1], True), ([3, 0], True), ([2, 2], False), ([0], True)],
)
def test_identifiable_count_is_the_jacobian_rank(ranks: list[int], ci: bool) -> None:
    # (M38a): the number of parameters the marginal model identifies is the
    # rank of the Jacobian of (lambda, S, b) -> every neuron's marginal mean and
    # covariance, which depends on S only through S_p S_p'.
    n, T, N = 3, 3, 9
    rng = np.random.default_rng(sum(ranks) + 10 * ci)
    X = rng.normal(size=(N, len(ranks)))
    sizes = [T * r for r in ranks]
    theta0 = np.concatenate(
        [rng.normal(size=n), rng.normal(size=sum(sizes))]
        + ([rng.normal(size=n * T)] if ci else [])
    )

    def moments(theta: FloatArray) -> FloatArray:
        lam = np.exp(theta[:n])
        offsets = np.cumsum([n, *sizes])
        S = [
            theta[a:b].reshape(T, r)
            for a, b, r in zip(offsets, offsets[1:], ranks, strict=False)
        ]
        out = []
        for i in range(n):
            Phi = dense.design(X, S)
            Sigma = np.eye(N * T) / lam[i] + Phi @ Phi.T
            out.append(Sigma[np.triu_indices(N * T)])
            if ci:
                out.append(
                    np.tile(theta[offsets[-1] + i * T : offsets[-1] + (i + 1) * T], N)
                )
        return np.concatenate(out)

    J = np.stack(
        [
            (moments(theta0 + 1e-6 * e) - moments(theta0 - 1e-6 * e)) / 2e-6
            for e in np.eye(theta0.size)
        ],
        axis=1,
    )
    s = np.linalg.svd(J, compute_uv=False)
    rank = int((s > 1e-7 * s[0]).sum())
    assert rank == n_parameters_mmle(ranks, n, T, condition_independent=ci)


# ------------------------------------------------------------------ invariances


def test_unobserved_entries_do_not_change_the_fit() -> None:
    sim, stats = _sim_stats(24, drop_prob=0.3)
    Y = np.array(sim.Y)
    Y[~sim.mask] = 1e300
    other = sufficient_statistics(Y, sim.X, sim.mask)
    a, b = fit_mmle(stats, [1, 2]), fit_mmle(other, [1, 2])
    assert a.log_likelihood == b.log_likelihood


def test_neuron_permutation_equivariance() -> None:
    pr = _problem("base", N=40, n=7)
    perm = np.random.default_rng(3).permutation(7)
    stats_p = sufficient_statistics(pr.Y_masked[:, perm], pr.X, pr.mask[:, perm])
    assert pr.b is not None
    terms = mmle._nll_terms(
        pr.S,
        pr.lam,
        pr.stats.centered(pr.b)[0],
        pr.stats.XtX,
        pr.stats.centered(pr.b)[1],
        pr.stats.n_obs,
        pr.stats.n_bins,
    )
    xi_p, ups_p = stats_p.centered(pr.b[perm])
    terms_p = mmle._nll_terms(
        pr.S, pr.lam[perm], xi_p, stats_p.XtX, ups_p, stats_p.n_obs, stats_p.n_bins
    )
    np.testing.assert_allclose(terms_p, terms[perm], rtol=1e-13)
    _, g = marginal_nll_grad_S(pr.S, pr.lam, pr.b, pr.stats)
    _, g_p = marginal_nll_grad_S(pr.S, pr.lam[perm], pr.b[perm], stats_p)
    for a, b in zip(g, g_p, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)
    # The whole fit: summation order is the only difference; offline, over
    # seeds 0-99 of this configuration, B and lambda agreed to at most 9.3e-7.
    sim, stats = _sim_stats(25, n=25, ranks=(2, 1), drop_prob=0.3)
    perm = np.random.default_rng(4).permutation(25)
    other = sufficient_statistics(sim.Y[:, perm], sim.X, sim.mask[:, perm])
    fit0, fit1 = fit_mmle(stats, [2, 1]), fit_mmle(other, [2, 1])
    for x, y in zip(fit0.B, fit1.B, strict=True):
        assert _rel(y, x[perm]) < 1e-5
    np.testing.assert_allclose(
        fit1.noise_precision, fit0.noise_precision[perm], rtol=1e-5
    )


@pytest.mark.parametrize("c", [-3.0, 1e3])
def test_scaling_y(c: float) -> None:
    # The model is equivariant under Y -> cY with S -> cS, b -> cb,
    # lambda -> lambda / c^2: C_i is unchanged, so the posterior of the weights
    # (W, W_cov) is unchanged, B = W S' scales by c, and the log-likelihood
    # shifts by -sum_i n_i T log|c| (the N(0, I) prior on w is unaffected).
    pr = _problem("base")
    assert pr.b is not None
    scaled = sufficient_statistics(c * pr.Y_masked, pr.X, pr.mask)
    S_c = [c * s for s in pr.S]
    lam_c, b_c = pr.lam / c**2, c * pr.b
    m = float(pr.stats.n_obs.sum() * pr.stats.n_bins)
    ll = marginal_log_likelihood(pr.S, pr.lam, pr.b, pr.stats)
    ll_c = marginal_log_likelihood(S_c, lam_c, b_c, scaled)
    assert ll_c == pytest.approx(ll - m * np.log(abs(c)), rel=1e-12)
    W, cov = posterior_weights(pr.S, pr.lam, pr.b, pr.stats)
    W_c, cov_c = posterior_weights(S_c, lam_c, b_c, scaled)
    for a, b in zip(W, W_c, strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cov_c, cov, rtol=1e-9, atol=1e-12)
    # ECME from transformed starts is exactly equivariant.
    init = MMLEFit.from_parameters(pr.stats, pr.S, pr.lam, pr.b)
    init_c = MMLEFit.from_parameters(scaled, S_c, lam_c, b_c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        f, _ = ecme(pr.stats, init, max_iter=5, tol=1e-300)
        f_c, _ = ecme(scaled, init_c, max_iter=5, tol=1e-300)
    for a, b in zip(f.B, f_c.B, strict=True):
        np.testing.assert_allclose(b, c * a, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(f_c.noise_precision * c**2, f.noise_precision, rtol=1e-8)
    # The full fit starts from the SVD's balanced split, which scales by
    # sqrt|c|, and L-BFGS-B's gradient test is absolute, so it agrees only to
    # the optimiser's tolerance: offline, over seeds 0-99 of this
    # configuration, at most 8.0e-5 for c = -3 and 2.0e-4 for c = 1e3.
    sim, stats = _sim_stats(26, ranks=(1, 2), drop_prob=0.2)
    fit0 = fit_mmle(stats, [1, 2])
    fit1 = fit_mmle(sufficient_statistics(c * sim.Y, sim.X, sim.mask), [1, 2])
    for x, y in zip(fit0.B, fit1.B, strict=True):
        assert _rel(y, c * x) < 1e-3
    np.testing.assert_allclose(
        fit1.noise_precision * c**2, fit0.noise_precision, rtol=1e-3
    )


@pytest.mark.parametrize("a", [1e-2, 50.0])
def test_scaling_a_column_of_x(a: float) -> None:
    # X[:, p] -> a X[:, p] with S_p -> S_p / a leaves Phi_i, hence the
    # marginal likelihood, the posterior (W, W_cov), lambda, b and X[:, p] B_p
    # unchanged at basis_ridge = 0, while B_p = W_p S_p' becomes B_p / a.
    pr = _problem("base")
    X = pr.X.copy()
    X[:, 2] *= a
    stats = sufficient_statistics(pr.Y_masked, X, pr.mask)
    S = [pr.S[0], pr.S[1], pr.S[2] / a]
    ll = marginal_log_likelihood(pr.S, pr.lam, pr.b, pr.stats)
    assert marginal_log_likelihood(S, pr.lam, pr.b, stats) == pytest.approx(
        ll, rel=1e-12
    )
    W, cov = posterior_weights(pr.S, pr.lam, pr.b, pr.stats)
    W_a, cov_a = posterior_weights(S, pr.lam, pr.b, stats)
    for x, y in zip(W, W_a, strict=True):
        np.testing.assert_allclose(y, x, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cov_a, cov, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        update_intercept(S, pr.lam, stats),
        update_intercept(pr.S, pr.lam, pr.stats),
        rtol=1e-10,
        atol=1e-12,
    )
    init = MMLEFit.from_parameters(pr.stats, pr.S, pr.lam, pr.b)
    init_a = MMLEFit.from_parameters(stats, S, pr.lam, pr.b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        f, _ = ecme(pr.stats, init, max_iter=5, tol=1e-300)
        f_a, _ = ecme(stats, init_a, max_iter=5, tol=1e-300)
    np.testing.assert_allclose(f_a.S[2] * a, f.S[2], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(f_a.noise_precision, f.noise_precision, rtol=1e-8)
    assert f.intercept is not None
    assert f_a.intercept is not None
    np.testing.assert_allclose(f_a.intercept, f.intercept, rtol=1e-8, atol=1e-10)
    # The full fit: the SVD start scales S_p by 1/sqrt|a|, so agreement is at
    # the optimiser's tolerance: offline, over seeds 0-99 of this configuration,
    # at most 1.3e-3 for a = 1e-2 (7.3e-4 at this seed) and 9.0e-5 for a = 50.
    sim, stats0 = _sim_stats(27, ranks=(1, 2), drop_prob=0.2)
    X = np.array(sim.X)
    X[:, 1] *= a
    g0 = fit_mmle(stats0, [1, 2])
    g1 = fit_mmle(sufficient_statistics(sim.Y, X, sim.mask), [1, 2])
    assert _rel(g1.B[1] * a, g0.B[1]) < 3e-3
    np.testing.assert_allclose(g1.noise_precision, g0.noise_precision, rtol=3e-3)


def test_shifting_a_regressor_changes_the_marginal_likelihood() -> None:
    # Under the weight prior the model is not shift-invariant
    # (`docs/model.md` § 10.2): with X[:, p] -> X[:, p] + c the term c S_p w_p
    # becomes a random per-neuron offset with prior covariance c^2 S_p S_p', so
    # the marginal likelihood, and with it the AIC, depends on where a
    # regressor's zero is, while B and lambda barely move (offline, demo scale:
    # recoding a +-1 regressor to 0/2 lowered the log-likelihood by about 100
    # nats with B unchanged to 1e-4; a shift of 7 changed B by 2-5 % and the
    # log-likelihood by 700-1800 nats).
    sim, stats = _sim_stats(28, n=40, T=8, N=200, ranks=(2, 1), drop_prob=0.3)
    a = fit_mmle(stats, [2, 1])
    shifted = sufficient_statistics(sim.Y, sim.X + np.array([0.0, 1.0]), sim.mask)
    b = fit_mmle(shifted, [2, 1])
    # Over seeds 0-99 of this configuration: the drop was at least 3.5 nats and
    # B moved by at most 4.8e-4.
    assert b.log_likelihood < a.log_likelihood - 1
    assert max(_rel(x, y) for x, y in zip(a.B, b.B, strict=True)) < 5e-3


# ------------------------------------------------------------------ MMLEFit


def test_fit_is_read_only_pickles_and_summarises() -> None:
    _, stats = _sim_stats(29)
    fit = fit_mmle(stats, [1, 2])
    assert isinstance(fit.S, tuple)
    assert isinstance(fit.W, tuple)
    assert isinstance(fit.B, tuple)
    for arr in (
        *fit.S,
        *fit.W,
        *fit.B,
        fit.W_cov,
        fit.noise_precision,
        fit.intercept,
        fit.rank_deficient_neurons,
    ):
        assert arr is not None
        assert not arr.flags.writeable
    with pytest.raises(TypeError):
        fit.n_iter["ecme"] = 99  # type: ignore[index]
    with pytest.raises(AttributeError):
        fit.converged = False  # type: ignore[misc]
    clone = pickle.loads(pickle.dumps(fit))
    assert all(not a.flags.writeable for a in (*clone.S, *clone.B, clone.W_cov))
    np.testing.assert_array_equal(clone.B[1], fit.B[1])
    assert dict(clone.n_iter) == dict(fit.n_iter)
    text = repr(fit)
    assert text.startswith(
        "MMLEFit(ranks=(1, 2), n_neurons=20, n_bins=6, intercept=True"
    )
    assert (fit.n_neurons, fit.n_bins, fit.total_rank) == (20, 6, 3)


def test_fit_copies_its_inputs() -> None:
    _, stats = _sim_stats(30)
    S = [np.ones((6, 1)), np.ones((6, 2))]
    lam = np.ones(20)
    fit = MMLEFit.from_parameters(stats, S, lam, stats.Y_mean)
    S[0][:] = 5.0
    lam[:] = 2.0
    assert fit.S[0][0, 0] == 1.0
    assert fit.noise_precision[0] == 1.0


def _fields(fit: MMLEFit, **change: Any) -> dict[str, Any]:
    out = {name: getattr(fit, name) for name in fit.__dataclass_fields__}
    out.update(change)
    return out


BAD_FIELDS: list[tuple[dict[str, Any], str]] = [
    ({"ranks": (1,)}, "S must be a sequence of 1"),
    ({"ranks": (1, 3)}, r"S\[1\] must be \(n_bins, 3\)"),
    ({"ranks": (1, -2)}, "non-negative integer"),
    ({"noise_precision": np.zeros(20)}, "positive"),
    ({"noise_precision": np.full(20, np.nan)}, "finite"),
    ({"intercept": np.zeros((20, 5))}, "intercept must be"),
    ({"W_cov": np.zeros((20, 2, 2))}, "W_cov must be"),
    ({"W_cov": np.arange(20 * 9.0).reshape(20, 3, 3)}, "symmetric"),
    ({"log_likelihood": np.inf}, "log_likelihood must be a finite"),
    ({"objective": "x"}, "objective must be a finite"),
    ({"n_parameters": -1}, "n_parameters must be"),
    ({"n_parameters": True}, "n_parameters must be"),
    ({"aic": 0.0}, "aic must equal"),
    ({"n_iter": {"ecme": -1}}, "n_iter must map"),
    ({"n_iter": [1]}, "n_iter must be a mapping"),
    ({"converged": 1}, "converged must be a bool"),
    ({"rank_deficient_neurons": np.array([25])}, "rank_deficient_neurons"),
    ({"rank_deficient_neurons": np.array([0.5])}, "rank_deficient_neurons"),
    ({"S": np.zeros((2, 6, 1))}, "S must be a sequence"),
    ({"W_cov": [[1.0], [1.0, 2.0]]}, "W_cov is not an array"),
    ({"W_cov": np.ones((20, 3, 3), dtype=bool)}, "W_cov must be a real 3-D array"),
    ({"W": (np.zeros((20, 1)), np.zeros((19, 2)))}, r"W\[1\] \(20, 2\)"),
]


@pytest.mark.parametrize(("change", "match"), BAD_FIELDS)
def test_fit_checks_its_fields(change: dict[str, Any], match: str) -> None:
    _, stats = _sim_stats(31)
    fit = _start(stats, (1, 2))
    with pytest.raises(ParameterError, match=match):
        MMLEFit(**_fields(fit, **change))


# ------------------------------------------------------------------ argument validation


def _evaluators() -> list[Any]:
    return [
        marginal_log_likelihood,
        marginal_nll_grad_S,
        marginal_nll_grad_noise,
        posterior_weights,
    ]


BAD_EVALUATOR_ARGS: list[tuple[str, Any, str]] = [
    ("S", "not a list", "S must be a sequence"),
    ("S", np.zeros((4, 2)), "S must be a sequence"),
    ("S", [np.zeros((4, 1))], "S has 1 blocks; expected one per regressor"),
    (
        "S",
        [np.zeros((4, 1)), np.zeros((4, 1)), np.zeros((3, 1))],
        r"S\[2\] must be a real",
    ),
    ("S", [np.zeros((4, 1)), np.zeros(4), np.zeros((4, 1))], r"S\[1\] must be a real"),
    ("S", [np.zeros((4, 1), bool)] * 3, r"S\[0\] must be a real"),
    ("S", [np.full((4, 1), np.nan)] * 3, r"S\[0\] must be finite"),
    ("lam", np.zeros(5), "noise_precision must be finite and > 0"),
    (
        "lam",
        -np.ones(5),
        r"noise_precision must be finite and > 0; it is not for neurons \[0, 1",
    ),
    ("lam", np.array([1, 1, np.nan, 1, 1.0]), r"neurons \[2\]"),
    ("lam", np.ones(4), r"shape \(n_neurons,\) = \(5,\)"),
    ("lam", np.ones(5, bool), "noise_precision must be a real array"),
    ("b", np.zeros((5, 3)), "intercept must be a real array"),
    ("b", np.full((5, 4), np.inf), "intercept must be finite"),
    ("stats", "stats", "stats must be a SufficientStats"),
]


@pytest.mark.parametrize("func", _evaluators())
@pytest.mark.parametrize(("which", "value", "match"), BAD_EVALUATOR_ARGS)
def test_evaluator_arguments_are_validated(
    func: Any, which: str, value: Any, match: str
) -> None:
    pr = _problem("base")
    args = {"S": pr.S, "lam": pr.lam, "b": pr.b, "stats": pr.stats}
    args[which] = value
    with pytest.raises(ParameterError, match=match):
        func(args["S"], args["lam"], args["b"], args["stats"])


@pytest.mark.parametrize("ridge", [-1.0, np.nan, True])
def test_basis_ridge_is_validated(ridge: Any) -> None:
    pr = _problem("base")
    for func in (marginal_log_likelihood, marginal_nll_grad_S):
        with pytest.raises(ParameterError, match="basis_ridge"):
            func(pr.S, pr.lam, pr.b, pr.stats, basis_ridge=ridge)


def test_update_intercept_rejects_unobserved_neurons() -> None:
    pr = _problem("base")
    mask = pr.mask.copy()
    mask[:, 1] = False
    stats = sufficient_statistics(pr.Y, pr.X, mask)
    with pytest.raises(ValidationError, match=r"neurons \[1\] are never observed"):
        update_intercept(pr.S, pr.lam, stats)


BAD_FIT_ARGS: list[tuple[dict[str, Any], str]] = [
    ({"ranks": [1]}, "ranks has 1 entries"),
    ({"ranks": [1, 7]}, r"ranks\[1\] is 7, above min"),
    ({"ranks": [1, -1]}, "non-negative integer"),
    ({"ranks": [1.0, 2]}, "non-negative integer"),
    ({"init": "svd"}, "init must be an MMLEFit, an SVDFit or None"),
    ({"ecme_max_iter": -1}, "max_iter must be an integer >= 0"),
    ({"ecme_tol": 0.0}, "tol must be a finite real number > 0"),
    ({"refine_tol": np.inf}, "tol must be a finite real number > 0"),
    ({"refine_max_iter": 1.5}, "max_iter must be an integer >= 0"),
    ({"convergence_eps": 0.0}, "convergence_eps"),
    ({"optimizer_max_iter": 0}, "optimizer_max_iter must be an integer >= 1"),
    ({"optimizer_tol": -1e-6}, "optimizer_tol"),
    ({"basis_ridge": -1.0}, "basis_ridge"),
    ({"basis_span_scale": 0.0}, "basis_span_scale must be a finite real number > 0"),
    ({"basis_span_scale": np.inf}, "basis_span_scale"),
    ({"basis_span_scale": np.nan}, "basis_span_scale"),
    ({"ridge": -1.0}, "ridge"),
    ({"verbose": 3}, "verbose must be 0, 1 or 2"),
    ({"condition_independent": "yes"}, "condition_independent must be a bool"),
]


@pytest.mark.parametrize(("kwargs", "match"), BAD_FIT_ARGS)
def test_fit_arguments_are_validated(kwargs: dict[str, Any], match: str) -> None:
    _, stats = _sim_stats(32)
    args: dict[str, Any] = {"ranks": [1, 2], **kwargs}
    with pytest.raises(ParameterError, match=match):
        fit_mmle(stats, **args)


def test_fit_requires_sufficient_stats() -> None:
    with pytest.raises(ParameterError, match="stats must be a SufficientStats"):
        fit_mmle("stats", [1])  # type: ignore[arg-type]


@pytest.mark.parametrize("func", [ecme, refine])
def test_init_is_checked(func: Any) -> None:
    _, stats = _sim_stats(33)
    _, other = _sim_stats(33, n=19)
    with pytest.raises(ParameterError, match="init must be an MMLEFit"):
        func(stats, fit_svd(stats, [1, 2]))
    with pytest.raises(ParameterError, match="init has 19 neurons"):
        func(stats, _start(other, (1, 2)))
    with pytest.raises(
        ParameterError, match="condition_independent=False, but init has an"
    ):
        func(stats, _start(stats, (1, 2)), condition_independent=False)
    no_b = _start(stats, (1, 2), intercept=False)
    with pytest.raises(
        ParameterError, match="condition_independent=True, but init has no"
    ):
        func(stats, no_b)
    with pytest.raises(ParameterError, match="max_iter must be an integer >= 0"):
        func(stats, _start(stats, (1, 2)), max_iter=-1)
    with pytest.raises(ParameterError, match="tol must be"):
        func(stats, _start(stats, (1, 2)), tol=0)


def test_ecme_flags_are_validated() -> None:
    _, stats = _sim_stats(34)
    with pytest.raises(ParameterError, match="matlab_compat must be a bool"):
        ecme(stats, _start(stats, (1, 2)), matlab_compat=1)  # type: ignore[arg-type]
    with pytest.raises(ParameterError, match="verbose"):
        ecme(stats, _start(stats, (1, 2)), verbose=True)


def test_init_ranks_and_intercept_must_match_fit_mmle() -> None:
    _, stats = _sim_stats(35)
    with pytest.raises(ParameterError, match=r"init has ranks \(1, 1\)"):
        fit_mmle(stats, [1, 2], init=fit_svd(stats, [1, 1]))
    with pytest.raises(ParameterError, match=r"init has ranks \(1, 1\)"):
        fit_mmle(stats, [1, 2], init=_start(stats, (1, 1)))
    with pytest.raises(ParameterError, match="does not match the SVDFit"):
        fit_mmle(
            stats, [1, 2], init=fit_svd(stats, [1, 2], condition_independent=False)
        )
    _, other = _sim_stats(35, n=18)
    with pytest.raises(ParameterError, match="does not match the sizes"):
        fit_mmle(stats, [1, 2], init=fit_svd(other, [1, 2]))


def test_degenerate_neurons_are_rejected() -> None:
    sim, _ = _sim_stats(36)
    init_stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    init = _start(init_stats, (1, 2))
    # never observed
    mask = sim.mask.copy()
    mask[:, 3] = False
    stats = sufficient_statistics(sim.Y, sim.X, mask)
    with pytest.raises(ValidationError, match=r"neurons \[3\] are never observed"):
        ecme(stats, init)
    # observed once, with the intercept (`docs/model.md` § 10.3)
    mask = sim.mask.copy()
    mask[:, 3] = False
    mask[0, 3] = True
    stats = sufficient_statistics(sim.Y, sim.X, mask)
    with pytest.raises(
        ValidationError, match=r"neurons \[3\] are observed on one trial"
    ):
        refine(stats, init)
    # constant over its trials
    Y = np.array(sim.Y)
    Y[:, 5, :] = 2.0
    stats = sufficient_statistics(Y, sim.X, sim.mask)
    with pytest.raises(ValidationError, match=r"neurons \[5\] have zero energy about"):
        ecme(stats, init)
    Y[:, 5, :] = 0.0
    stats = sufficient_statistics(Y, sim.X, sim.mask)
    no_b = MMLEFit.from_parameters(stats, init.S, init.noise_precision, None)
    with pytest.raises(ValidationError, match=r"neurons \[5\] have zero energy on"):
        ecme(stats, no_b, condition_independent=False)


def test_verbose_output(capsys: pytest.CaptureFixture[str]) -> None:
    _, stats = _sim_stats(37)
    fit_mmle(stats, [1, 2], verbose=2)
    out = capsys.readouterr().out
    assert "ecme: iteration 1: nll" in out
    assert "refine: iteration 1: change" in out
    assert "ecme: " in out
    assert "converged True" in out
    fit_mmle(stats, [1, 2], verbose=0)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------- gaps found by mutation


def test_relative_change_has_an_eps_floor() -> None:
    # (M34) with a floor: max_j (theta'_j - theta_j)^2 / (theta_j^2 + eps), relative
    # to the OLD value; an exactly-zero entry is measured against eps.
    empty = np.zeros((1, 0))
    change = mmle._relative_change
    assert change(
        (np.array([2.0]), empty, None), (np.array([1.0]), empty, None), 1e-12
    ) == pytest.approx(0.25, rel=1e-11)
    assert change(
        (np.array([1.0]), empty, None), (np.array([2.0]), empty, None), 1e-12
    ) == pytest.approx(1.0, rel=1e-11)
    old = (np.array([1.0, 2.0]), np.array([[0.0, 1.0]]), np.array([[3.0], [0.0]]))
    new = (np.array([1.5, 2.0]), np.array([[1e-3, 1.0]]), np.array([[3.0], [0.0]]))
    assert change(old, new, 1e-12) == pytest.approx(1e-6 / 1e-12)
    new = (np.array([1.5, 2.0]), np.array([[0.0, 1.0]]), np.array([[3.0], [0.0]]))
    assert change(old, new, 1e-12) == pytest.approx(0.25)


def test_refine_reports_a_failed_inner_call_even_when_its_loop_converges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An inner call that reports failure makes the fit non-converged although
    # the refinement loop met its tolerance.
    _, stats = _sim_stats(38)
    fit = fit_mmle(stats, [1, 2])
    assert fit.converged
    real = scipy.optimize.minimize

    def flaky(*args: Any, **kwargs: Any) -> Any:
        res = real(*args, **kwargs)
        if "bounds" in kwargs:
            res.success = False
            res.message = "pretend line-search failure"
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", flaky)
    with pytest.warns(ConvergenceWarning, match="noise precision: pretend"):
        again = refine(stats, fit)
    assert again.n_iter["refine"] == fit.n_iter["refine"] + 1  # cumulative
    assert not again.converged


def test_matlab_compat_steps_match_the_dense_block_versions() -> None:
    # The stale-variable steps against a second implementation, with block
    # matrices, of the same reading of `ECMEtdr.m` (tests/mmle_dense.py). Both
    # follow that reading, so this catches coding errors, not a misreading of
    # the reference; parity with the reference itself is the fixture's
    # (test_mmle_parity.py).
    pr = _problem("base", N=60, n=8, ranks=[2, 2, 1])
    fit = MMLEFit.from_parameters(pr.stats, pr.S, pr.lam, pr.b)
    st, lam, xi, ups = _state_at(pr.stats, fit)
    lam1 = mmle._precision_step(st, ups, pr.stats.n_obs, pr.stats.n_bins)
    blocks = mmle._basis_step_reference(st, lam1, xi)
    ours = mmle._split(blocks.S, blocks.bounds)
    theirs = dense.compat_basis_step(pr.stats, pr.S, lam, lam1, pr.b)
    for a, b in zip(ours, theirs, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
    b_ours = mmle._intercept_step_reference(mmle._blocks(pr.S), blocks, lam1, pr.stats)
    b_theirs = dense.compat_intercept_step(pr.stats, pr.S, ours, lam1)
    np.testing.assert_allclose(b_ours, b_theirs, rtol=1e-9, atol=1e-12)


def test_fit_mmle_starts_from_the_svd_bases_and_the_mean_response() -> None:
    # (M29): S and lambda from fit_svd at the same ranks, b = y_bar (the SVD's
    # own intercept is discarded, as the reference's ECMEregress_wrapper does).
    _, stats = _sim_stats(39)
    start = fit_mmle(stats, [1, 2], ecme_max_iter=0, refine_max_iter=0)
    svd = fit_svd(stats, [1, 2])
    np.testing.assert_array_equal(start.intercept, stats.Y_mean)
    for a, b in zip(start.S, svd.S, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(start.noise_precision, svd.noise_precision)
    assert start.converged
    assert dict(start.n_iter) == {"ecme": 0, "refine": 0}


# ------------------------------------------------------ residual degrees of freedom


def _few_trial_stats(
    seed: int, n_obs0: int, ranks: list[int], N: int = 40, n: int = 6, T: int = 4
) -> SufficientStats:
    """Random data on which neuron 0 is observed on `n_obs0` trials only."""
    rng = np.random.default_rng(seed)
    P = len(ranks)
    X = rng.normal(size=(N, P))
    Y = 1.0 + rng.normal(size=(N, n, T)) * rng.uniform(0.5, 2, (1, n, 1))
    mask = rng.random((N, n)) < 0.7
    mask[: P + 3] = True
    mask[:, 0] = False
    mask[rng.choice(N, n_obs0, replace=False), 0] = True
    return sufficient_statistics(np.where(mask[..., None], Y, np.nan), X, mask)


def _random_start(stats: SufficientStats, ranks: list[int], intercept: bool) -> MMLEFit:
    rng = np.random.default_rng(1)
    S = [rng.normal(size=(stats.n_bins, r)) for r in ranks]
    b = stats.Y_mean if intercept else None
    return MMLEFit.from_parameters(stats, S, np.ones(stats.n_neurons), b)


def test_a_neuron_with_an_unbounded_likelihood_is_rejected() -> None:
    # Neuron 0 is observed on 2 trials with 2 regressors and the intercept, so
    # least squares fits it exactly and its marginal likelihood has no
    # maximum; without the check this seed returns converged=True at
    # lambda_0 = 2.8e14. The check runs before the SVD start, so no
    # DesignWarning precedes the error.
    stats = _few_trial_stats(2, 2, [2, 1])
    with pytest.raises(
        ValidationError, match=r"neurons \[0\] have no residual degrees of freedom"
    ):
        fit_mmle(stats, [2, 1])


@pytest.mark.parametrize("n_obs0", [2, 3])
@pytest.mark.parametrize("stage", ["ecme", "refine", "fit_mmle"])
def test_neurons_without_residual_degrees_of_freedom_are_rejected(
    n_obs0: int, stage: str
) -> None:
    # With the intercept, n_i <= rank [X_i 1] = P + 1 = 3 trials leave no
    # residual degrees of freedom; every MMLE entry point rejects them.
    stats = _few_trial_stats(0, n_obs0, [2, 1])
    start = _random_start(stats, [2, 1], True)
    calls: dict[str, Callable[[], object]] = {
        "ecme": lambda: ecme(stats, start),
        "refine": lambda: refine(stats, start),
        "fit_mmle": lambda: fit_mmle(stats, [2, 1], init=start),
    }
    with pytest.raises(
        ValidationError, match=r"neurons \[0\] have no residual degrees of freedom"
    ):
        calls[stage]()


def test_residual_degrees_of_freedom_without_the_intercept() -> None:
    # Without the intercept the condition is n_i <= rank X_i: two trials and two
    # regressors are rejected, three are fitted.
    stats = _few_trial_stats(0, 2, [2, 1])
    with pytest.raises(ValidationError, match=r"neurons \[0\] have no residual"):
        ecme(stats, _random_start(stats, [2, 1], False), condition_independent=False)
    stats = _few_trial_stats(0, 3, [2, 1])
    fit, _ = ecme(
        stats, _random_start(stats, [2, 1], False), condition_independent=False
    )
    assert np.isfinite(fit.log_likelihood)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("n_obs0", [4, 5])
def test_neurons_with_residual_degrees_of_freedom_are_fitted(
    seed: int, n_obs0: int
) -> None:
    # The same data with one or two residual degrees of freedom: bounded and
    # converged (lambda_0 measured in [0.4, 2.2] on these seeds).
    stats = _few_trial_stats(seed, n_obs0, [2, 1])
    fit = fit_mmle(stats, [2, 1])
    assert fit.converged
    assert 0.05 < fit.noise_precision[0] < 20


def _exactly_fitted_problem() -> tuple[SufficientStats, MMLEFit]:
    """Neuron 0 on two trials with x = +1 and -1, so x_bar_0 = 0 and A_0 is centred.

    With S = I (one regressor at rank T) its centred responses lie in the column
    space of Phi_0, and at lambda_0 = 1e16 its expected residual (M30) is about
    T / lambda_0, zero to rounding against its energy.
    """
    rng = np.random.default_rng(40)
    N, n, T = 30, 4, 3
    X = np.where(rng.random((N, 1)) < 0.5, -1.0, 1.0)
    X[:2, 0] = [1.0, -1.0]
    Y = 1.0 + rng.normal(size=(N, n, T))
    mask = np.ones((N, n), dtype=bool)
    mask[2:, 0] = False
    stats = sufficient_statistics(Y, X, mask)
    lam = np.ones(n)
    lam[0] = 1e16
    return stats, MMLEFit.from_parameters(stats, [np.eye(T)], lam, stats.Y_mean)


def test_a_diverging_precision_is_an_error_at_every_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The entry check rejects neuron 0; with it bypassed, the ECME precision
    # step and refine's final state still apply the rounding test
    # E_i > 1e3 eps T upsilon_i(b) to the expected residual, so no stage
    # returns a fit at a diverging precision.
    stats, init = _exactly_fitted_problem()
    with pytest.raises(ValidationError, match=r"neurons \[0\] have no residual"):
        refine(stats, init)
    monkeypatch.setattr(mmle, "_require_fittable", lambda stats, ci: None)
    with pytest.raises(
        ValidationError, match=r"neurons \[0\] have zero expected residual energy"
    ):
        refine(stats, init, max_iter=0)
    with pytest.raises(
        ValidationError, match=r"neurons \[0\] have zero expected residual energy"
    ):
        ecme(stats, init, max_iter=1)


def test_precision_step_rejects_an_expected_residual_zero_to_rounding() -> None:
    # (M30) needs E_i > 1e3 eps T upsilon_i(b), a rounding test, not only
    # E_i > 0. With r_tot = 1, K = u = mu = 1 and C^-1 = 0 the expected
    # residual is upsilon - 1.
    pr = _problem("base")
    fit = MMLEFit.from_parameters(pr.stats, pr.S, pr.lam, pr.b)
    st, _, _, _ = _state_at(pr.stats, fit)
    n, T = pr.stats.n_neurons, pr.stats.n_bins
    unit = st._replace(
        K=np.ones((n, 1, 1)),
        u=np.ones((n, 1)),
        mu=np.ones((n, 1)),
        Cinv=np.zeros((n, 1, 1)),
    )
    ups = np.full(n, 3.0)
    ups[3] = 1.0 + 1e-13  # E = 1e-13 against a floor of 1e3 eps T = 8.9e-13
    with pytest.raises(ValidationError, match=r"neurons \[3\] have zero expected"):
        mmle._precision_step(unit, ups, pr.stats.n_obs, T)
    ups[3] = 1.0 + 1e-9
    lam1 = mmle._precision_step(unit, ups, pr.stats.n_obs, T)
    assert lam1[3] == pytest.approx(pr.stats.n_obs[3] * T / 1e-9, rel=1e-6)


# ------------------------------------------ finite extremes and the scaled solve

EXTREMES = [  # (lambda, S, scale of Y, rank, log-likelihood)
    (1e160, 1e-80, 1e-80, 1, 915.2565608324501),
    (1e-200, 1e100, 1e100, 1, -1157.070022862191),
    (1e160, 1.0, 1e-80, 0, 916.189344531595),
]


@pytest.mark.parametrize(("lam", "s", "scale", "rank", "expected"), EXTREMES)
def test_finite_well_conditioned_extremes_are_evaluated(
    lam: float, s: float, scale: float, rank: int, expected: float
) -> None:
    # One neuron, one bin, five trials, no intercept; C = 6.46 (I at rank 0),
    # condition number 1. The expected values were computed in whitened
    # coordinates, x* = sqrt(lambda) X S and y* = sqrt(lambda) Y, without the
    # package; without the scaled solve z = L^-1 (lambda u) all three are NaN.
    x = np.array([-1.0, -0.3, 0.2, 1.2, 1.7])
    y = scale * np.array([0.1, 0.2, -0.4, 0.5, -0.2])
    full = np.ones((5, 1), dtype=bool)
    stats = sufficient_statistics(y[:, None, None], x[:, None], full)
    S, noise = [np.full((1, rank), s)], np.array([lam])
    assert marginal_log_likelihood(S, noise, None, stats) == pytest.approx(
        expected, rel=1e-13
    )
    # Everything else equals its value in the whitened coordinates
    # (`docs/model.md` § 10.2 with c = sqrt(lambda)): d/dS scales by c,
    # d/dlambda by 1/c^2.
    c = np.sqrt(lam)
    white = sufficient_statistics(c * y[:, None, None], x[:, None], full)
    S_w, one = [c * S[0]], np.ones(1)
    value, (gS,) = marginal_nll_grad_S(S, noise, None, stats)
    value_w, (gS_w,) = marginal_nll_grad_S(S_w, one, None, white)
    assert value == pytest.approx(value_w - 5 * np.log(c), rel=1e-13)
    np.testing.assert_allclose(gS, c * gS_w, rtol=1e-12)
    _, gl = marginal_nll_grad_noise(S, noise, None, stats)
    _, gl_w = marginal_nll_grad_noise(S_w, one, None, white)
    np.testing.assert_allclose(gl, gl_w / c**2, rtol=1e-12)
    (W,), cov = posterior_weights(S, noise, None, stats)
    (W_w,), cov_w = posterior_weights(S_w, one, None, white)
    np.testing.assert_allclose(W, W_w, rtol=1e-12)
    np.testing.assert_allclose(cov, cov_w, rtol=1e-12)


@pytest.mark.parametrize("c", [1e-120, 1e120])
def test_scaling_y_at_extreme_scales(c: float) -> None:
    # `docs/model.md` § 10.2 far from unit scale: with S -> cS, b -> cb and
    # lambda -> lambda / c^2 the products lambda u and lambda K are unchanged,
    # so the log-likelihood shifts by exactly -sum_i n_i T log|c| and the
    # posterior is unchanged, although u' C^-1 u and lambda^2 are not
    # representable separately (an unscaled solve returns NaN here).
    pr = _problem("base")
    assert pr.b is not None
    scaled = sufficient_statistics(c * pr.Y_masked, pr.X, pr.mask)
    S_c, lam_c, b_c = [c * s for s in pr.S], pr.lam / c**2, c * pr.b
    m = float(pr.stats.n_obs.sum() * pr.stats.n_bins)
    ll = marginal_log_likelihood(pr.S, pr.lam, pr.b, pr.stats)
    ll_c = marginal_log_likelihood(S_c, lam_c, b_c, scaled)
    assert ll_c == pytest.approx(ll - m * np.log(c), rel=1e-12)
    _, gS = marginal_nll_grad_S(pr.S, pr.lam, pr.b, pr.stats)
    _, gS_c = marginal_nll_grad_S(S_c, lam_c, b_c, scaled)
    for a, b in zip(gS, gS_c, strict=True):
        np.testing.assert_allclose(b * c, a, rtol=1e-10, atol=1e-12)
    _, gl = marginal_nll_grad_noise(pr.S, pr.lam, pr.b, pr.stats)
    _, gl_c = marginal_nll_grad_noise(S_c, lam_c, b_c, scaled)
    np.testing.assert_allclose(gl_c / c**2, gl, rtol=1e-10)
    W, cov = posterior_weights(pr.S, pr.lam, pr.b, pr.stats)
    W_c, cov_c = posterior_weights(S_c, lam_c, b_c, scaled)
    for a, b in zip(W, W_c, strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cov_c, cov, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        update_intercept(S_c, lam_c, scaled) / c,
        update_intercept(pr.S, pr.lam, pr.stats),
        rtol=1e-10,
        atol=1e-12,
    )


def test_a_non_representable_likelihood_is_an_error() -> None:
    # When the value itself overflows float64 the evaluators raise a
    # ValidationError naming the neurons, never return NaN or inf. Here C_i is
    # finite and well conditioned (lambda_2 K_2 is of order 1), but
    # lambda_2 upsilon_2(b) is about 1e312.
    pr = _problem("base", baseline=1e5)
    S = [1e-150 * s for s in pr.S]
    lam = pr.lam.copy()
    lam[2] = 1e300
    for func in _evaluators():
        with pytest.raises(ValidationError, match=r"not representable.*neurons \[2\]"):
            func(S, lam, pr.b, pr.stats)
    with pytest.raises(ValidationError, match=r"not representable.*neurons \[2\]"):
        MMLEFit.from_parameters(pr.stats, S, lam, pr.b)


def test_the_log_precision_box_stays_in_the_float_range() -> None:
    # The refinement's box is intersected with the finite positive float64
    # range, so exp of any point in it is a positive finite number, and it
    # always contains the current value.
    theta0 = np.array([0.0, 700.0, -700.0, -744.0, 709.0])
    lower, upper = mmle._log_precision_box(theta0)
    B = mmle.LOG_PRECISION_BOUND
    np.testing.assert_array_equal(lower[:2], theta0[:2] - B)
    np.testing.assert_array_equal(upper[[0, 2, 3]], theta0[[0, 2, 3]] + B)
    assert np.all(lower <= theta0)
    assert np.all(theta0 <= upper)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.all(np.isfinite(np.exp(upper)))
        assert np.all(np.exp(lower[[0, 1, 2, 4]]) >= np.finfo(np.float64).tiny)
    assert lower[3] == theta0[3]  # a subnormal start is not widened downwards


# ----------------------------------------------- refine accepts only improvements


@pytest.mark.parametrize(
    ("offset", "why"),
    [
        (1.0, "raised the objective"),
        (np.nan, "is not finite"),
        (1e300, "is not finite"),  # its evaluation raises LinAlgError
    ],
)
def test_refine_rejects_a_non_finite_inner_iterate(
    monkeypatch: pytest.MonkeyPatch, offset: float, why: str
) -> None:
    _, stats = _sim_stats(41)
    warm, _ = ecme(stats, _start(stats, (1, 2)))

    def forged(fun: Any, x0: FloatArray, **kwargs: Any) -> Any:
        return types.SimpleNamespace(x=x0 + offset, success=True)

    monkeypatch.setattr(scipy.optimize, "minimize", forged)
    with pytest.warns(ConvergenceWarning) as record:
        fit = refine(stats, warm, max_iter=1)
    text = str(record[0].message)
    assert "bases: no message (L-BFGS-B status unknown, nit unknown" in text
    assert f"its final point {why}" in text
    assert "so the step was not taken" in text
    np.testing.assert_array_equal(fit.noise_precision, warm.noise_precision)


def test_inner_objectives_and_gradients_are_never_nan() -> None:
    # An inner objective hands L-BFGS-B (inf, 0) where its value or gradient is
    # not finite; a public gradient that overflows is a ValidationError.
    value, grad = mmle._finite_or_inf(np.inf, np.array([[1.0, 2.0]]))
    assert value == np.inf
    np.testing.assert_array_equal(grad, [0.0, 0.0])
    value, grad = mmle._finite_or_inf(1.0, np.array([np.nan]))
    assert value == np.inf
    assert mmle._finite_or_inf(2.0, np.array([[3.0]]))[0] == 2.0
    with pytest.raises(ValidationError, match=r"gradient .* not representable"):
        mmle._require_finite_gradient(np.array([1.0, np.inf]))


@pytest.mark.parametrize("ridge", [0.0, 0.5])
@pytest.mark.parametrize("success", [False, True])
def test_refine_rejects_a_deteriorating_inner_iterate(
    monkeypatch: pytest.MonkeyPatch, success: bool, ridge: float
) -> None:
    # An optimiser result is installed only if its penalised objective is finite
    # and no worse than the current point's; an injected result x0 + 1 (with or
    # without a success flag, and without fun, status or nit) is rejected, the
    # point kept and the failure reported. The start is compared at the
    # requested ridge, not at the one it was built with.
    _, stats = _sim_stats(41)
    warm, _ = ecme(stats, _start(stats, (1, 2)))

    def forged(fun: Any, x0: FloatArray, **kwargs: Any) -> Any:
        return types.SimpleNamespace(
            x=x0 + 1.0, success=success, message="injected trial"
        )

    monkeypatch.setattr(scipy.optimize, "minimize", forged)
    with pytest.warns(ConvergenceWarning, match="so the step was not taken"):
        fit = refine(stats, warm, max_iter=1, basis_ridge=ridge)
    assert not fit.converged
    for a, b in zip(fit.S, warm.S, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(fit.noise_precision, warm.noise_precision)
    start = marginal_log_likelihood(
        warm.S, warm.noise_precision, warm.intercept, stats, basis_ridge=ridge
    )
    assert fit.objective >= start - 1e-12 * abs(start)


# --------------------------------------------------- inner-failure messages


def test_inner_failure_messages_name_status_iterations_and_gradient() -> None:
    # With SciPy >= 1.15 a line-search failure's message is the bare
    # "ABNORMAL: "; the status, the iteration count and the largest
    # projected-gradient entry are reported with it, and a failed precision
    # step names the neurons furthest from stationarity,
    # |lambda_i E_i / (n_i T) - 1|.
    _, stats = _sim_stats(10)
    warm, _ = ecme(stats, _start(stats, (1, 2)))
    with pytest.warns(ConvergenceWarning) as record:
        refine(stats, warm, optimizer_max_iter=1)
    text = str(record[0].message)
    assert re.search(
        # SciPy's own wording varies with the version ("NO. of" in 1.13).
        r"iteration 1, bases: STOP: TOTAL NO\. (?i:of) ITERATIONS REACHED LIMIT "
        r"\(L-BFGS-B status 1, nit 1, largest projected-gradient entry "
        r"[0-9.e+-]+\)",
        text,
    )
    assert re.search(
        r"noise precision: .*neurons with the largest "
        r"\|lambda_i E_i / \(n_i T\) - 1\|: \d+ \([0-9.e+-]+\)",
        text,
    )


def test_a_box_contact_before_the_last_iteration_is_not_called_unbounded() -> None:
    # From precisions 1e-12 times too small the first precision step ends on the
    # box, later ones do not, and the fit reaches fit_mmle's optimum. The
    # failure stays sticky, but the message says where it happened and does not
    # claim the likelihood is unbounded.
    _, stats = _sim_stats(0, n=25, T=6, N=60, ranks=(2, 1))
    base = fit_mmle(stats, [2, 1])
    s0 = _start(stats, (2, 1))
    init = MMLEFit.from_parameters(
        stats, s0.S, s0.noise_precision * 1e-12, s0.intercept
    )
    with pytest.warns(ConvergenceWarning) as record:
        fit = refine(stats, init, max_iter=50)
    text = " ".join(str(w.message) for w in record)
    assert "ended on the step bound in iteration 1" in text
    assert "unbounded" not in text
    assert not fit.converged
    assert fit.log_likelihood == pytest.approx(base.log_likelihood, abs=1e-5)


# ------------------------------------------------------- MMLEFit semantics


def test_fit_rejects_an_objective_above_its_log_likelihood() -> None:
    # The objective is log_likelihood - g/2 ||s||^2 with g >= 0.
    _, stats = _sim_stats(31)
    fit = _start(stats, (1, 2))
    with pytest.raises(ParameterError, match="objective must not exceed"):
        MMLEFit(**_fields(fit, objective=fit.log_likelihood + 100.0))
    MMLEFit(**_fields(fit, objective=fit.log_likelihood - 100.0))


def test_fit_rejects_a_covariance_that_is_not_positive_definite() -> None:
    # A negated covariance, W_cov = -cov, is rejected; so is a covariance with
    # one negative eigenvalue. A rounding-level negative eigenvalue (the
    # computed C^-1 of an extremely ill-conditioned C) is accepted.
    _, stats = _sim_stats(31)
    fit = _start(stats, (1, 2))
    with pytest.raises(ParameterError, match=r"positive definite.*neurons \[0, 1"):
        MMLEFit(**_fields(fit, W_cov=-fit.W_cov))
    cov = np.array(fit.W_cov)
    w, V = np.linalg.eigh(cov[4])
    cov[4] = (V * np.array([-1e-3, *w[1:]])) @ V.T
    with pytest.raises(ParameterError, match=r"positive definite.*neurons \[4\]"):
        MMLEFit(**_fields(fit, W_cov=cov))
    cov[4] = (V * np.array([-1e-17 * w[-1], *w[1:]])) @ V.T
    cov[4] = 0.5 * (cov[4] + cov[4].T)
    MMLEFit(**_fields(fit, W_cov=cov))


def test_fit_does_not_check_the_value_of_its_parameter_count() -> None:
    # A caller may carry another count; the validated entry point is
    # MMLEFit.from_parameters.
    _, stats = _sim_stats(31)
    fit = _start(stats, (1, 2))
    other = MMLEFit(**_fields(fit, n_parameters=0, aic=-2.0 * fit.log_likelihood))
    assert other.n_parameters == 0


# --------------------------------------------------- mutation survivors


def test_refine_with_a_basis_ridge_reaches_a_stationary_point() -> None:
    # Guards the sign of the ridge gradient in refine's basis objective: with
    # the sign flipped the penalised refinement stops at its first line search
    # and reports non-convergence.
    _, stats = _sim_stats(23, n=25, T=6, N=60, ranks=(2, 1))
    g = 3.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit = fit_mmle(stats, [2, 1], basis_ridge=g)
    # No basis step fails. On some CI platforms the precision step ends on a
    # rounding-level line-search failure (projected gradient about 1e-5) and
    # warns; that is not this test's subject.
    messages = [str(w.message) for w in caught]
    assert not [m for m in messages if "bases" in m], messages
    assert all("noise precision: ABNORMAL" in m for m in messages), messages
    start = MMLEFit.from_parameters(
        stats, fit.S, fit.noise_precision, fit.intercept, basis_ridge=g
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        tight = refine(
            stats, start, basis_ridge=g, max_iter=200, tol=1e-14, optimizer_tol=1e-10
        )
    assert tight.objective - fit.objective < 1e-4
    # Rounding-level inner ends are recorded, not ignored (CI Windows).
    unpenalised, _ = pf.run(lambda: fit_mmle(stats, [2, 1]), ROUNDING)
    assert sum(_ridge_norm(fit.S)) < sum(_ridge_norm(unpenalised.S))


def _ridge_norm(S: Any) -> list[float]:
    return [float(np.sum(s * s)) for s in S]


def test_refine_propagates_a_non_converged_start() -> None:
    # Guards against refine ignoring init.converged: a fit refined from a
    # non-converged start is not converged.
    _, stats = _sim_stats(24)
    with pytest.warns(ConvergenceWarning, match=r"iteration cap \(max_iter = 1;"):
        capped, _ = ecme(stats, _start(stats, (1, 2)), max_iter=1, tol=1e-300)
    assert not capped.converged
    # refine's own warnings are recorded, not ignored: no rounding-level inner
    # end warns, so `not out.converged` can only come from the start's flag.
    out, caught = pf.run(lambda: refine(stats, capped), ROUNDING)
    assert caught.messages == []
    assert not out.converged
    assert out.n_iter["ecme"] == 1
    assert "refine" in out.n_iter


# ------------------------------------------------- rounding-level inner ends

#: Nine fits of the demo-scale sweep whose inner steps end on a line search at
#: an optimum to rounding (Windows): five basis-step and four precision-step
#: ends. They report `converged=False` unless such ends count as converged
#: (`docs/model.md` § E.3).
ROUNDING_LEVEL_END_FITS = [
    *((s, (2, 1, 3)) for s in (11, 73, 75)),
    *((s, (2, 2, 3)) for s in (55, 62, 72, 84, 89, 96)),
]


def _demo_scale_stats(seed: int) -> SufficientStats:
    sim = mtdr.simulate(
        n_neurons=100,
        n_bins=15,
        n_trials=400,
        ranks=[2, 1, 3],
        drop_prob=0.3,
        seed=seed,
    )
    return sufficient_statistics(sim.Y_masked, sim.X, sim.mask)


@pytest.mark.parametrize(("seed", "ranks"), ROUNDING_LEVEL_END_FITS)
def test_rounding_level_ends_converge(
    seed: int, ranks: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    # No ConvergenceWarning and converged, with the log-likelihood within
    # 1e-9 nats of the fit without the polish (the basis rule changes no
    # point).
    stats = _demo_scale_stats(seed)
    fit, caught = pf.run(lambda: fit_mmle(stats, list(ranks)))
    assert caught.messages == []
    assert fit.converged
    monkeypatch.setattr(mmle, "_newton_polish", lambda *a: a[4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        unpolished = fit_mmle(stats, list(ranks))
    assert abs(fit.log_likelihood - unpolished.log_likelihood) <= 1e-9


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"optimizer_max_iter": 3}, r"(?i)bases: STOP: TOTAL NO\. OF ITERATIONS"),
        ({"refine_max_iter": 1, "refine_tol": 1e-300}, r"refine_max_iter"),
        ({"ecme_max_iter": 1, "ecme_tol": 1e-300}, r"ecme_max_iter"),
    ],
)
def test_genuine_non_convergence_stays_false(kw: dict[str, Any], match: str) -> None:
    # Only rounding-level ends are forgiven: caps are still failures.
    stats = _demo_scale_stats(11)
    with pytest.warns(ConvergenceWarning, match=match):
        fit = fit_mmle(stats, [2, 1, 3], **kw)
    assert not fit.converged


@pytest.mark.slow
def test_the_demo_scale_sweep_converges(monkeypatch: pytest.MonkeyPatch) -> None:
    # The 200 demo-scale fits (seeds 0-99 at the true and an over-ranked
    # vector) all converge; prints the rounding-level basis ends judged by the
    # basis rule and the precision ends the polish fixed.
    judged: list[tuple[bool, float]] = []
    polished: list[tuple[float, float]] = []
    rule, polish = mmle._basis_end_is_rounding, mmle._newton_polish

    def recording_rule(inner: Any, progtol: float, magnitude: float) -> bool:
        ok = rule(inner, progtol, magnitude)
        judged.append((ok, float(inner.last_decrease)))
        return ok

    def recording_polish(*args: Any) -> Any:
        out = polish(*args)
        K, u, ups, counts, theta0 = args[:5]
        before, _ = mmle._precision_derivatives(theta0, K, u, ups, counts)
        after, _ = mmle._precision_derivatives(out, K, u, ups, counts)
        polished.append(
            (
                float(np.max(np.abs(2 * before / counts))),
                float(np.max(np.abs(2 * after / counts))),
            )
        )
        return out

    monkeypatch.setattr(mmle, "_basis_end_is_rounding", recording_rule)
    monkeypatch.setattr(mmle, "_newton_polish", recording_polish)
    failed = []
    for seed in range(100):
        stats = _demo_scale_stats(seed)
        for ranks in ([2, 1, 3], [2, 2, 3]):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = fit_mmle(stats, ranks)
            if not fit.converged:
                failed.append((seed, ranks, [str(w.message) for w in caught]))
    print(f"\n[demo-scale sweep] not converged: {len(failed)} / 200")
    print(f"  basis status-2 ends judged (forgiven, decrease): {judged}")
    print(f"  precision ends polished (residual before, after): {polished}")
    assert failed == []


# --------------------------------------------- cumulative iteration counts


def test_iteration_counts_accumulate_over_repeated_stages() -> None:
    # A fit records every stage that produced it, so repeating a stage adds to
    # its count.
    _, stats = _sim_stats(42)
    start = _start(stats, (1, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        first, _ = ecme(stats, start, max_iter=3, tol=1e-300)
        second, _ = ecme(stats, first, max_iter=1, tol=1e-300)
    assert dict(second.n_iter) == {"ecme": 4}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        r1 = refine(stats, second, max_iter=1, tol=1e-300)
        r2 = refine(stats, r1, max_iter=2, tol=1e-300)
    assert dict(r2.n_iter) == {"ecme": 4, "refine": 3}
    again = fit_mmle(stats, [1, 2], init=r2, ecme_max_iter=0, refine_max_iter=0)
    assert dict(again.n_iter) == {"ecme": 4, "refine": 3}


# --------------------------------------------- the preconditioned basis step


def test_span_maps_scale_the_in_span_directions_only() -> None:
    # D(S0 M) = alpha S0 M for the start's own columns, D leaves the
    # orthogonal complement of each block's span alone, D is symmetric and
    # D^{-1} undoes it; an empty block and a rank-deficient one are fine.
    rng = np.random.default_rng(0)
    T = 7
    S = [rng.standard_normal((T, 3)), np.zeros((T, 0)), rng.standard_normal((T, 2))]
    S[2][:, 1] = 2.0 * S[2][:, 0]  # span of dimension 1
    blocks = mmle._blocks(S)
    alpha = 30.0
    stretch, shrink = mmle._span_maps(blocks, alpha)
    r = blocks.S.shape[1]
    M = [rng.standard_normal((3, 3)), rng.standard_normal((2, 2))]
    inside = np.hstack([S[0] @ M[0], S[2] @ M[1]])
    np.testing.assert_allclose(stretch(inside.ravel()), alpha * inside.ravel())
    np.testing.assert_allclose(shrink(inside.ravel()), inside.ravel() / alpha)
    Q0, _ = np.linalg.qr(S[0])
    Q2, _ = np.linalg.qr(S[2][:, :1])
    outside = rng.standard_normal((T, r))
    outside[:, :3] -= Q0 @ (Q0.T @ outside[:, :3])
    outside[:, 3:] -= Q2 @ (Q2.T @ outside[:, 3:])
    np.testing.assert_allclose(stretch(outside.ravel()), outside.ravel(), atol=1e-12)
    a, b = rng.standard_normal(T * r), rng.standard_normal(T * r)
    assert stretch(a) @ b == pytest.approx(a @ stretch(b), rel=1e-12)
    np.testing.assert_allclose(shrink(stretch(a)), a, rtol=1e-12, atol=1e-12)


def test_unit_span_scale_is_the_plain_fit() -> None:
    _, stats = _sim_stats(43, n=30, T=8, N=120, ranks=(2, 1), drop_prob=0.3)
    plain = fit_mmle(stats, [2, 1])
    same = fit_mmle(stats, [2, 1], basis_span_scale=1)
    for a, b in zip(plain.S, same.S, strict=True):
        np.testing.assert_array_equal(a, b)
    assert same.log_likelihood == plain.log_likelihood


@pytest.mark.parametrize(("seed", "ranks"), [(0, (2, 1)), (1, (2, 1)), (2, (3, 2, 1))])
def test_preconditioned_fit_reaches_the_same_optimum(
    seed: int, ranks: tuple[int, ...]
) -> None:
    # The scaled run changes the path, not the objective. Offline, over
    # seeds 0-19 of both configurations at BASIS_SPAN_SCALE: every fit
    # converged without warnings, its log-likelihood at least the plain fit's
    # less 7.4e-10 (relative 2.4e-14; higher in 39 of 40), B within 1.8e-6,
    # with 1.2-3.5x fewer basis evaluations.
    n, T, N = (30, 8, 120) if len(ranks) == 2 else (60, 10, 200)
    _, stats = _sim_stats(seed, n=n, T=T, N=N, ranks=ranks, drop_prob=0.3)
    evals = {"plain": 0, "scaled": 0}
    original = mmle._lbfgs

    def counted(objective: Any, x0: Any, bounds: Any, *args: Any) -> Any:
        def wrapped(x: Any) -> Any:
            if bounds is None:
                evals[key] += 1
            return objective(x)

        return original(wrapped, x0, bounds, *args)

    fits = {}
    with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings():
        warnings.simplefilter("error")
        mp.setattr(mmle, "_lbfgs", counted)
        for key, scale in (("plain", 1.0), ("scaled", mmle.BASIS_SPAN_SCALE)):
            fits[key] = fit_mmle(stats, list(ranks), basis_span_scale=scale)
    plain, scaled = fits["plain"], fits["scaled"]
    assert scaled.converged
    assert scaled.log_likelihood >= plain.log_likelihood - 1e-11 * abs(
        plain.log_likelihood
    )
    for a, b in zip(scaled.B, plain.B, strict=True):
        assert _rel(a, b) < 1e-5
    assert evals["scaled"] < evals["plain"]


@pytest.mark.parametrize("success", [False, True])
def test_preconditioned_refine_rejects_a_deteriorating_iterate(
    monkeypatch: pytest.MonkeyPatch, success: bool
) -> None:
    # A rejected scaled run returns the start's bases exactly, not D(D^{-1} s),
    # and reports the step as not taken.
    _, stats = _sim_stats(41)
    warm, _ = ecme(stats, _start(stats, (1, 2)))

    def forged(fun: Any, x0: FloatArray, **kwargs: Any) -> Any:
        return types.SimpleNamespace(x=x0 + 1.0, success=success, message="injected")

    monkeypatch.setattr(scipy.optimize, "minimize", forged)
    with pytest.warns(ConvergenceWarning, match="so the step was not taken"):
        fit = refine(stats, warm, max_iter=1, basis_span_scale=30.0)
    for a, b in zip(fit.S, warm.S, strict=True):
        np.testing.assert_array_equal(a, b)
