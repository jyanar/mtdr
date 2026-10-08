"""Property tests of `mtdr.mmle` over random shapes, masks, ranks and seeds.

Hypothesis draws small problems (including rank-0 blocks, one regressor, one
bin, one neuron, sparse masks, with and without the intercept); the seed loops
check the ECME and refinement guarantees on 100+ fixed simulations each, with
margins measured offline and stated next to each assertion (no seed luck).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

import mmle_dense as dense
import mtdr
from mtdr import mmle
from mtdr.errors import ConvergenceWarning
from mtdr.mmle import MMLEFit, ecme, marginal_log_likelihood, refine
from mtdr.stats import SufficientStats, sufficient_statistics
from mtdr.svd_fit import fit_svd

FloatArray = NDArray[np.float64]


@st.composite
def problems(draw: st.DrawFn) -> dict[str, Any]:
    n = draw(st.integers(1, 6))
    T = draw(st.integers(1, 5))
    P = draw(st.integers(1, 3))
    ranks = [draw(st.integers(0, min(n, T))) for _ in range(P)]
    N = draw(st.integers(P + 3, 25))
    intercept = draw(st.booleans())
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, P)) * draw(st.sampled_from([0.1, 1.0, 10.0]))
    baseline = draw(st.sampled_from([0.0, 3.0, -50.0]))
    Y = baseline + rng.normal(size=(N, n, T)) * rng.uniform(0.3, 3.0, (1, n, 1))
    mask = rng.random((N, n)) < draw(st.floats(0.3, 1.0))
    mask[: P + 3] = True
    return {
        "stats": sufficient_statistics(np.where(mask[:, :, None], Y, np.nan), X, mask),
        "Y": Y,
        "X": X,
        "mask": mask,
        "S": [rng.normal(size=(T, r)) for r in ranks],
        "lam": rng.uniform(0.2, 5.0, n),
        "b": rng.normal(size=(n, T)) + baseline if intercept else None,
        "direction": rng.normal(size=T * sum(ranks)),
        "lam_direction": rng.normal(size=n),
    }


@settings(deadline=None)
@given(problems())
def test_marginal_likelihood_is_the_dense_gaussian(pr: dict[str, Any]) -> None:
    ll = marginal_log_likelihood(pr["S"], pr["lam"], pr["b"], pr["stats"])
    direct = dense.nll(pr["Y"], pr["X"], pr["mask"], pr["S"], pr["lam"], pr["b"]).sum()
    assert abs(-ll - direct) <= 1e-10 * max(abs(direct), 1.0)


def _check_directional(f: Any, x: FloatArray, d: FloatArray, g: FloatArray) -> None:
    """Central difference of `f` along the unit vector `d` against `g @ d`.

    Tolerance: 1e-6 of the Cauchy-Schwarz scale `||g||`, plus the rounding floor
    of the difference quotient, `1e3 eps |f| / h`.
    """
    d = d / np.linalg.norm(d)
    h = 1e-6 * max(1.0, float(np.abs(x).max()))
    numeric = (f(x + h * d) - f(x - h * d)) / (2 * h)
    floor = 1e3 * np.finfo(float).eps * max(abs(f(x)), 1.0) / h
    assert abs(float(g @ d) - numeric) <= 1e-6 * float(np.linalg.norm(g)) + floor


@settings(deadline=None)
@given(problems(), st.sampled_from([0.0, 0.3]))
def test_directional_derivatives_match_the_gradients(
    pr: dict[str, Any], ridge: float
) -> None:
    S, lam, b, stats = pr["S"], pr["lam"], pr["b"], pr["stats"]
    shapes = [s.shape for s in S]
    bounds = np.cumsum([0] + [s.size for s in S])

    def unpack(x: FloatArray) -> list[FloatArray]:
        return [
            x[a:c].reshape(sh)
            for a, c, sh in zip(bounds, bounds[1:], shapes, strict=False)
        ]

    def f_bases(x: FloatArray) -> float:
        return -marginal_log_likelihood(unpack(x), lam, b, stats, basis_ridge=ridge)

    _, grad = mmle.marginal_nll_grad_S(S, lam, b, stats, basis_ridge=ridge)
    flat = np.concatenate([s.ravel() for s in S])
    if flat.size:
        g = np.concatenate([x.ravel() for x in grad])
        _check_directional(f_bases, flat, pr["direction"], g)

    def f_lam(x: FloatArray) -> float:
        return -marginal_log_likelihood(S, np.exp(x), b, stats)

    # Along log(lambda), so every step stays positive: the gradient is lam * dL/dlam.
    _, g_lam = mmle.marginal_nll_grad_noise(S, lam, b, stats)
    _check_directional(f_lam, np.log(lam), pr["lam_direction"], lam * g_lam)


@settings(deadline=None, max_examples=30)
@given(problems())
def test_each_ecme_step_is_monotone(pr: dict[str, Any]) -> None:
    stats: SufficientStats = pr["stats"]
    if (stats.n_obs < 2).any() or not (stats.YtY_c > 0).all():
        return
    b = None if pr["b"] is None else stats.Y_mean
    fit = MMLEFit.from_parameters(stats, pr["S"], pr["lam"], b)
    for _ in range(3):
        blocks = mmle._blocks(fit.S)
        lam = np.array(fit.noise_precision)
        xi, ups = stats.centered(fit.intercept)
        st_ = mmle._state(blocks, lam, xi, ups, stats.XtX, stats.n_obs, stats.n_bins)
        before = -fit.log_likelihood
        scale = 1e-12 * max(abs(before), float(stats.n_obs.sum() * stats.n_bins))
        lam1 = mmle._precision_step(st_, ups, stats.n_obs, stats.n_bins)
        nll1 = -marginal_log_likelihood(fit.S, lam1, fit.intercept, stats)
        assert nll1 <= before + scale
        new = mmle._basis_step(st_, lam1, xi) if blocks.S.shape[1] else blocks
        S1 = mmle._split(new.S, new.bounds)
        nll2 = -marginal_log_likelihood(S1, lam1, fit.intercept, stats)
        assert nll2 <= nll1 + scale
        b1 = (
            mmle._intercept_step(new, lam1, stats)
            if fit.intercept is not None
            else None
        )
        nll3 = -marginal_log_likelihood(S1, lam1, b1, stats)
        assert nll3 <= nll2 + scale
        fit = MMLEFit.from_parameters(stats, S1, lam1, b1)


def _random_simulation(seed: int) -> tuple[SufficientStats, list[int], bool]:
    """A random small simulation; shapes, ranks, noise and mask vary with the seed."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 30))
    T = int(rng.integers(1, 8))
    P = int(rng.integers(1, 4))
    ranks = [int(rng.integers(0, min(n, T, 3) + 1)) for _ in range(P)]
    intercept = bool(rng.random() < 0.8)
    sim = mtdr.simulate(
        n_neurons=n,
        n_bins=T,
        n_trials=int(rng.integers(P + 12, 70)),
        ranks=ranks,
        levels=[np.linspace(-2, 2, 5)] * P,
        noise_precision=float(10 ** rng.uniform(-1, 3)),
        drop_prob=float(rng.uniform(0.0, 0.3)),
        condition_independent=intercept,
        seed=seed,
    )
    return sufficient_statistics(sim.Y, sim.X, sim.mask), ranks, intercept


def _svd_start(stats: SufficientStats, ranks: list[int], ci: bool) -> MMLEFit:
    svd = fit_svd(stats, ranks, condition_independent=ci)
    return MMLEFit.from_parameters(
        stats, svd.S, svd.noise_precision, stats.Y_mean if ci else None
    )


def test_ecme_is_monotone_on_120_simulations() -> None:
    # The marginal NLL never rises across ECME iterations. Offline, over
    # these 120 simulations x 30 iterations (and 60 x 1500 at longer runs), the
    # largest rise was 6.4e-16 of max(|NLL|, number of observed entries), i.e.
    # rounding; the bound below is 1e-13, and MONOTONE_RTOL = 1e-10.
    worst = -np.inf
    for seed in range(120):
        stats, ranks, ci = _random_simulation(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fit, trace = ecme(
                stats,
                _svd_start(stats, ranks, ci),
                max_iter=30,
                tol=1e-300,
                condition_independent=ci,
            )
        scale = np.maximum(np.abs(trace[:-1]), stats.n_obs.sum() * stats.n_bins)
        worst = max(worst, float((np.diff(trace) / scale).max()))
        assert fit.converged or fit.n_iter["ecme"] == 30
    assert worst <= 1e-13, worst


def test_refine_never_lowers_the_log_likelihood_on_120_simulations() -> None:
    # Every refinement step lowers the objective (`docs/model.md` § C.7): over
    # these 120 simulations the refined log-likelihood is at least the ECME
    # start's.
    for seed in range(120):
        stats, ranks, ci = _random_simulation(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warm, _ = ecme(
                stats, _svd_start(stats, ranks, ci), condition_independent=ci
            )
            fit = refine(stats, warm, condition_independent=ci)
        assert fit.log_likelihood >= warm.log_likelihood - 1e-12 * abs(
            warm.log_likelihood
        )
