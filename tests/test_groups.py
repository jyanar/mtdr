"""Regressor groups in the positional layers: `docs/model.md` § 11, (M51)-(M60).

The grouped likelihood, posterior, intercept and expected residual are checked
against the dense oracle of `tests/mmle_dense.py` (the design of (M52) built
neuron by neuron, with no tie matrix), the gradients against central
differences, the ECME steps for monotonicity, the forced tie with $M=I$ against
the ungrouped fast path, the counts (M59)-(M60) against a Jacobian rank, and the
SVD start (M58) against a direct truncation. Tolerances are declared next to
each assertion; the margins measured on the machine of record are recorded with
the decision that introduced the feature.
"""

from __future__ import annotations

import pickle
import warnings
from typing import Any, Literal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

import mmle_dense as dense
from mtdr import _groups, mmle, simulate
from mtdr.aic import n_parameters_mmle, n_parameters_svd
from mtdr.data import categorical_design
from mtdr.errors import ConvergenceWarning, ParameterError, ValidationError
from mtdr.mmle import (
    MONOTONE_RTOL,
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
from mtdr.stats import SufficientStats, sufficient_statistics
from mtdr.svd_fit import SVDFit, fit_svd

FloatArray = NDArray[np.float64]

#: The oracle's tolerance (relative), the plan's 1e-10.
ORACLE_RTOL = 1e-10
#: Central-difference agreement of the gradients (relative), the plan's 1e-6.
GRADIENT_RTOL = 1e-6
#: The forced tie with M = I against the ungrouped fast path (relative).
TIE_RTOL = 1e-12

#: Layouts of four columns: (groups, rank per group).
LAYOUTS: dict[str, tuple[list[list[int]], list[int]]] = {
    "one group of four": ([[0, 1, 2, 3]], [2]),
    "a group and singletons": ([[0, 2], [1], [3]], [2, 1, 1]),
    "two multi-column groups": ([[0, 3], [1, 2]], [2, 1]),
    "a group rank above T": ([[0, 1, 2], [3]], [4, 1]),
    "a rank-0 group": ([[0, 1], [2, 3]], [0, 2]),
}


def _rel(a: Any, b: Any) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-300))


class Problem:
    """Four columns: a categorical coding with unequal level counts and a mask.

    Columns 0-3 are the polynomial coding of five levels drawn with unequal
    probabilities, rescaled column by column so the coding is not orthonormal
    on these trials; the bases, precisions and intercept are a random point.
    """

    def __init__(
        self,
        layout: str,
        seed: int = 0,
        N: int = 40,
        n: int = 5,
        T: int = 3,
        intercept: bool = True,
        p_obs: float = 0.75,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.groups, self.ranks = LAYOUTS[layout]
        labels = rng.choice(5, N, p=[0.4, 0.25, 0.15, 0.12, 0.08])
        labels[:5] = np.arange(5)
        X = categorical_design(labels, levels=range(5), name="c").X
        self.X = X * np.array([1.0, 0.5, 2.0, 1.5]) + 0.1 * rng.normal(size=(N, 4))
        self.Y = 1.0 + rng.normal(size=(N, n, T)) * rng.uniform(0.5, 2, (1, n, 1))
        mask = rng.random((N, n)) < p_obs
        mask[:8] = True
        self.mask = mask
        self.stats = sufficient_statistics(
            np.where(mask[:, :, None], self.Y, np.nan), self.X, mask
        )
        widths = _groups.check_grouping(self.groups, 4).column_ranks(self.ranks)
        self.S = [0.6 * rng.normal(size=(T, w)) for w in widths]
        self.lam = rng.uniform(0.3, 3.0, n)
        self.b = rng.normal(size=(n, T)) if intercept else None


# ------------------------------------------------------------------ Grouping


def test_grouping_normalises_members_and_exposes_sizes() -> None:
    g = _groups.check_grouping([[2, 0], [1], [3, 4]], 5)
    assert g.groups == ((0, 2), (1,), (3, 4))
    assert (g.n_groups, g.n_columns, g.sizes) == (3, 5, (2, 1, 2))
    assert g.group_of == (0, 1, 0, 2, 2)
    assert g.column_ranks([4, 1, 2]) == [4, 1, 4, 2, 2]
    assert not g.trivial
    assert _groups.Grouping.singletons(3).trivial
    assert _groups.as_grouping([[0], [1]], 2) is None
    assert _groups.as_grouping(None, 2) is None
    assert _groups.check_grouping([[1, 0]], None).groups == ((0, 1),)
    with pytest.raises(ParameterError, match="2 ranks for 3 groups"):
        g.column_ranks([1, 2])


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        ("01", "non-empty sequence of sequences"),
        ([], "non-empty sequence of sequences"),
        ([[0], []], r"groups\[1\] must be a non-empty"),
        ([[0], "1"], r"groups\[1\] must be a non-empty"),
        ([[0, 1.0]], "not a column index"),
        ([[0, True]], "not a column index"),
        ([[0, 5]], r"not a column index in \[0, 3\)"),
        ([[-1]], "not a column index"),
        ([[0, 0], [1, 2]], "column 0 appears twice"),
        ([[0, 1], [1, 2]], "column 1 appears in groups 0 and 1"),
        ([[0, 1]], r"columns \[2\] are in no group"),
        ([[1, 2], [0]], "ordered by their smallest column"),
    ],
)
def test_grouping_rejects(groups: Any, match: str) -> None:
    with pytest.raises(ParameterError, match=match):
        _groups.check_grouping(groups, 3)


def test_require_ranks_names_the_group_and_its_cap() -> None:
    g = _groups.check_grouping([[0, 1, 2], [3]], 4)
    assert _groups.require_ranks([9, 3], g, 4, 20, 3) == [9, 3]
    with pytest.raises(ParameterError, match="one per group"):
        _groups.require_ranks([1], g, 4, 20, 3)
    with pytest.raises(ParameterError, match=r"columns \[0, 1, 2\]\) is 10, above"):
        _groups.require_ranks([10, 1], g, 4, 20, 3)
    with pytest.raises(ParameterError, match=r"ranks\[1\] .* is 4, above"):
        _groups.require_ranks([1, 4], g, 4, 20, 3)
    # The cap is min(n_neurons, |g| T): here n_neurons.
    with pytest.raises(ParameterError, match="= 5"):
        _groups.require_ranks([6, 1], g, 4, 5, 3)


# ------------------------------------------------------------------ the dense oracle


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("intercept", [True, False])
def test_grouped_likelihood_is_the_dense_gaussian(layout: str, intercept: bool) -> None:
    # (M51)-(M53), with unequal level counts and a mask: the marginal Gaussian
    # built densely from the grouped design.
    pr = Problem(layout, intercept=intercept)
    ll = marginal_log_likelihood(pr.S, pr.lam, pr.b, pr.stats, groups=pr.groups)
    direct = dense.nll(pr.Y, pr.X, pr.mask, pr.S, pr.lam, pr.b, pr.groups)
    assert abs(-ll - direct.sum()) <= ORACLE_RTOL * abs(direct.sum())
    terms = mmle._state(
        mmle._blocks(pr.S, _groups.check_grouping(pr.groups, 4)),
        pr.lam,
        *pr.stats.centered(pr.b),
        pr.stats.XtX,
        pr.stats.n_obs,
        pr.stats.n_bins,
    ).terms
    const = 0.5 * np.log(2 * np.pi) * pr.stats.n_bins * pr.stats.n_obs
    np.testing.assert_allclose(terms + const, direct, rtol=ORACLE_RTOL)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_grouped_posterior_is_the_dense_posterior(layout: str) -> None:
    pr = Problem(layout, seed=1)
    W, cov = posterior_weights(pr.S, pr.lam, pr.b, pr.stats, groups=pr.groups)
    means, covs = dense.posterior(pr.Y, pr.X, pr.mask, pr.S, pr.lam, pr.b, pr.groups)
    assert [w.shape[1] for w in W] == pr.ranks
    joined = np.concatenate(W, axis=1)
    assert _rel(joined, means) <= ORACLE_RTOL or not means.size
    assert _rel(cov, covs) <= ORACLE_RTOL


@pytest.mark.parametrize("layout", LAYOUTS)
def test_grouped_intercept_is_the_gls_mean(layout: str) -> None:
    # (M57): the centred form, tied and pushed forward, against the dense GLS.
    pr = Problem(layout, seed=2)
    b = update_intercept(pr.S, pr.lam, pr.stats, groups=pr.groups)
    want = dense.gls_intercept(pr.Y, pr.X, pr.mask, pr.S, pr.lam, pr.groups)
    assert _rel(b, want) <= ORACLE_RTOL


@pytest.mark.parametrize("layout", LAYOUTS)
def test_grouped_expected_residual_is_dense(layout: str) -> None:
    pr = Problem(layout, seed=3)
    blocks = mmle._blocks(pr.S, _groups.check_grouping(pr.groups, 4))
    xi, ups = pr.stats.centered(pr.b)
    st_ = mmle._state(
        blocks, pr.lam, xi, ups, pr.stats.XtX, pr.stats.n_obs, pr.stats.n_bins
    )
    E = mmle._expected_residual(st_.K, st_.u, st_.mu, st_.Cinv, ups)
    want = dense.expected_residual(pr.Y, pr.X, pr.mask, pr.S, pr.lam, pr.b, pr.groups)
    assert _rel(E, want) <= ORACLE_RTOL


# ------------------------------------------------------------------ gradients


def _central(f: Any, x: FloatArray, h: float = 1e-6) -> FloatArray:
    out = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        step = h * max(1.0, abs(float(x[idx])))
        xp, xm = x.copy(), x.copy()
        xp[idx] += step
        xm[idx] -= step
        out[idx] = (f(xp) - f(xm)) / (2 * step)
    return out


@pytest.mark.parametrize(
    "layout",
    ["one group of four", "a group and singletons", "two multi-column groups"],
)
@pytest.mark.parametrize("ridge", [0.0, 0.4])
def test_grouped_basis_gradient_matches_central_differences(
    layout: str, ridge: float
) -> None:
    # (M55): every entry of every column's basis, the ridge included.
    pr = Problem(layout, seed=5)
    shapes = [s.shape for s in pr.S]
    bounds = np.cumsum([0] + [s.size for s in pr.S])

    def unpack(x: FloatArray) -> list[FloatArray]:
        return [
            x[a:c].reshape(sh)
            for a, c, sh in zip(bounds, bounds[1:], shapes, strict=False)
        ]

    def f(x: FloatArray) -> float:
        return -marginal_log_likelihood(
            unpack(x), pr.lam, pr.b, pr.stats, ridge, groups=pr.groups
        )

    value, grad = marginal_nll_grad_S(
        pr.S, pr.lam, pr.b, pr.stats, ridge, groups=pr.groups
    )
    flat = np.concatenate([s.ravel() for s in pr.S])
    assert value == pytest.approx(f(flat), rel=1e-14)
    numeric = _central(f, flat)
    g = np.concatenate([x.ravel() for x in grad])
    assert _rel(g, numeric) <= GRADIENT_RTOL


@pytest.mark.parametrize(
    "layout",
    ["one group of four", "a group and singletons", "two multi-column groups"],
)
def test_grouped_precision_gradient_matches_central_differences(layout: str) -> None:
    # (M27b) in the latent coordinates, along log(lambda).
    pr = Problem(layout, seed=6)

    def f(theta: FloatArray) -> float:
        return -marginal_log_likelihood(
            pr.S, np.exp(theta), pr.b, pr.stats, groups=pr.groups
        )

    _, grad = marginal_nll_grad_noise(pr.S, pr.lam, pr.b, pr.stats, groups=pr.groups)
    numeric = _central(f, np.log(pr.lam))
    assert _rel(pr.lam * grad, numeric) <= GRADIENT_RTOL


# ------------------------------------------------------------------ forced tie M = I


def _forced(S: list[FloatArray]) -> tuple[Any, Any]:
    """The same bases on the fast path and through the tie with M = I."""
    return mmle._blocks(S), mmle._blocks(S, _groups.Grouping.singletons(len(S)))


def test_the_tie_with_the_identity_is_the_fast_path() -> None:
    # With every column alone and the tie forced, M = I: every kernel matches
    # the fast path (to TIE_RTOL; the products with I are exact in practice).
    pr = Problem("one group of four", seed=7)
    S = [np.ones((3, 2)) * 0.3 + 0.1 * np.arange(6).reshape(3, 2) for _ in range(4)]
    S[2] = S[2][:, :1]
    plain, tied = _forced(S)
    assert plain.tie is None
    assert tied.tie is not None
    np.testing.assert_array_equal(tied.tie, np.eye(7))
    xi, ups = pr.stats.centered(pr.b)
    args = (pr.lam, xi, ups, pr.stats.XtX, pr.stats.n_obs, pr.stats.n_bins)
    a, b = mmle._state(plain, *args), mmle._state(tied, *args)
    for name in ("K", "u", "Cinv", "mu", "terms"):
        assert _rel(getattr(b, name), getattr(a, name)) <= TIE_RTOL, name
    lam1 = mmle._precision_step(a, ups, pr.stats.n_obs, pr.stats.n_bins)
    assert _rel(mmle._basis_step(b, lam1, xi).S, mmle._basis_step(a, lam1, xi).S) <= (
        TIE_RTOL
    )
    assert (
        _rel(
            mmle._intercept_step(tied, pr.lam, pr.stats),
            mmle._intercept_step(plain, pr.lam, pr.stats),
        )
        <= TIE_RTOL
    )
    for blocks in (plain, tied):
        mu, Cinv = mmle._push(a.mu, a.Cinv, blocks)
        np.testing.assert_array_equal(mu, a.mu)
        np.testing.assert_array_equal(Cinv, a.Cinv)
    common = (pr.lam, xi, ups, pr.stats.XtX, pr.stats.n_obs, pr.stats.n_bins)
    fast, _ = mmle._optimize_bases(plain, *common, 0.1, 1e-6, 200, 1e-9)
    slow, _ = mmle._optimize_bases(tied, *common, 0.1, 1e-6, 200, 1e-9)
    assert _rel(slow.S, fast.S) <= TIE_RTOL
    lam_a = mmle._optimize_precision(plain, pr.lam, xi, ups, *common[3:], 1e-6, 200)
    lam_b = mmle._optimize_precision(tied, pr.lam, xi, ups, *common[3:], 1e-6, 200)
    assert _rel(lam_b[0], lam_a[0]) <= TIE_RTOL


def test_a_trivial_grouping_is_the_ungrouped_call_bit_for_bit() -> None:
    # The public functions treat every-column-alone groups as None.
    pr = Problem("one group of four", seed=8)
    S = [np.ones((3, 1)) * k for k in range(1, 5)]
    singles = [[0], [1], [2], [3]]
    assert marginal_log_likelihood(S, pr.lam, pr.b, pr.stats) == (
        marginal_log_likelihood(S, pr.lam, pr.b, pr.stats, groups=singles)
    )
    a = fit_mmle(pr.stats, [1, 1, 1, 1])
    b = fit_mmle(pr.stats, [1, 1, 1, 1], groups=singles)
    assert b.groups is None
    for x, y in zip((*a.S, *a.W, a.W_cov), (*b.S, *b.W, b.W_cov), strict=True):
        np.testing.assert_array_equal(x, y)
    assert a.log_likelihood == b.log_likelihood
    sa, sb = fit_svd(pr.stats, [1] * 4), fit_svd(pr.stats, [1] * 4, groups=singles)
    assert sb.groups is None
    for x, y in zip((*sa.W, *sa.S, *sa.B), (*sb.W, *sb.S, *sb.B), strict=True):
        np.testing.assert_array_equal(x, y)


# ------------------------------------------------------------------ ECME


@st.composite
def grouped_problems(draw: st.DrawFn) -> dict[str, Any]:
    P = draw(st.integers(2, 4))
    # A random partition of range(P), ordered by smallest member.
    owners = [draw(st.integers(0, p)) for p in range(P)]
    groups: list[list[int]] = []
    first: dict[int, int] = {}
    for p, o in enumerate(owners):
        key = min(o, p)
        if key not in first:
            first[key] = len(groups)
            groups.append([])
        groups[first[key]].append(p)
    n = draw(st.integers(1, 5))
    T = draw(st.integers(1, 4))
    ranks = [draw(st.integers(0, min(n, len(g) * T, 3))) for g in groups]
    N = draw(st.integers(P + 4, 25))
    intercept = draw(st.booleans())
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, P)) * draw(st.sampled_from([0.1, 1.0, 10.0]))
    baseline = draw(st.sampled_from([0.0, 3.0, -50.0]))
    Y = baseline + rng.normal(size=(N, n, T)) * rng.uniform(0.3, 3.0, (1, n, 1))
    mask = rng.random((N, n)) < draw(st.floats(0.4, 1.0))
    mask[: P + 4] = True
    grouping = _groups.check_grouping(groups, P)
    return {
        "stats": sufficient_statistics(np.where(mask[:, :, None], Y, np.nan), X, mask),
        "groups": groups,
        "S": [rng.normal(size=(T, w)) for w in grouping.column_ranks(ranks)],
        "lam": rng.uniform(0.2, 5.0, n),
        "intercept": intercept,
    }


@settings(deadline=None, max_examples=30)
@given(grouped_problems())
def test_each_grouped_ecme_step_is_monotone(pr: dict[str, Any]) -> None:
    # (M30), (M56), (M57): each conditional maximisation alone never raises the
    # marginal NLL beyond MONOTONE_RTOL of max(|NLL|, number of observed entries).
    stats: SufficientStats = pr["stats"]
    if (stats.n_obs < 2).any() or not (stats.YtY_c > 0).all():
        return
    grouping = _groups.as_grouping(pr["groups"], stats.n_regressors)
    groups = None if grouping is None else grouping.groups
    b = stats.Y_mean if pr["intercept"] else None
    fit = MMLEFit.from_parameters(stats, pr["S"], pr["lam"], b, groups=groups)

    def nll(S: Any, lam: FloatArray, b: Any) -> float:
        return -marginal_log_likelihood(S, lam, b, stats, groups=groups)

    for _ in range(3):
        blocks = mmle._blocks(fit.S, grouping)
        lam = np.array(fit.noise_precision)
        xi, ups = stats.centered(fit.intercept)
        st_ = mmle._state(blocks, lam, xi, ups, stats.XtX, stats.n_obs, stats.n_bins)
        before = -fit.log_likelihood
        allowance = MONOTONE_RTOL * max(
            abs(before), float(stats.n_obs.sum() * stats.n_bins)
        )
        lam1 = mmle._precision_step(st_, ups, stats.n_obs, stats.n_bins)
        nll1 = nll(fit.S, lam1, fit.intercept)
        assert nll1 <= before + allowance
        new = mmle._basis_step(st_, lam1, xi) if blocks.S.shape[1] else blocks
        S1 = mmle._split(new.S, new.bounds)
        nll2 = nll(S1, lam1, fit.intercept)
        assert nll2 <= nll1 + allowance
        b1 = None
        if fit.intercept is not None:
            b1 = mmle._intercept_step(new, lam1, stats)
        nll3 = nll(S1, lam1, b1)
        assert nll3 <= nll2 + allowance
        fit = MMLEFit.from_parameters(stats, S1, lam1, b1, groups=groups)


def test_the_grouped_basis_step_solves_its_normal_equations() -> None:
    # (M56): at S' the expected complete-data gradient, with the posterior
    # redone at (lambda', S) and pushed forward, vanishes.
    pr = Problem("a group and singletons", seed=9, N=60, n=8)
    grouping = _groups.check_grouping(pr.groups, 4)
    blocks = mmle._blocks(pr.S, grouping)
    xi, ups = pr.stats.centered(pr.b)
    st_ = mmle._state(
        blocks, pr.lam, xi, ups, pr.stats.XtX, pr.stats.n_obs, pr.stats.n_bins
    )
    lam1 = mmle._precision_step(st_, ups, pr.stats.n_obs, pr.stats.n_bins)
    new = mmle._basis_step(st_, lam1, xi)
    Cinv, mu, _, _ = mmle._posterior(lam1, st_.K, st_.u)
    mu, Cinv = mmle._push(mu, Cinv, blocks)
    xig = xi[:, new.column, :]
    grad = mmle._grad_bases(new.S, st_.Aexp, mu, Cinv, lam1, xig)
    scale = np.abs(mmle._grad_bases(blocks.S, st_.Aexp, mu, Cinv, lam1, xig)).max()
    assert np.abs(grad).max() < 1e-10 * scale


def test_a_group_direction_without_support_makes_the_basis_step_singular() -> None:
    # Two columns of a group that are equal on every trial: the direction
    # x_0 - x_1 of the group has no support, so (M56) is singular.
    pr = Problem("two multi-column groups", seed=10)
    X = pr.X.copy()
    X[:, 3] = X[:, 0]
    stats = sufficient_statistics(
        np.where(pr.mask[:, :, None], pr.Y, np.nan), X, pr.mask
    )
    grouping = _groups.check_grouping(pr.groups, 4)
    blocks = mmle._blocks(pr.S, grouping)
    xi, ups = stats.centered(pr.b)
    st_ = mmle._state(blocks, pr.lam, xi, ups, stats.XtX, stats.n_obs, stats.n_bins)
    lam1 = mmle._precision_step(st_, ups, stats.n_obs, stats.n_bins)
    with pytest.raises(ValidationError, match="some direction of a group's design"):
        mmle._basis_step(st_, lam1, xi)


def test_grouped_ecme_and_refine_converge_and_never_lose() -> None:
    pr = Problem("a group and singletons", seed=11, N=80, n=10, T=4)
    init = MMLEFit.from_parameters(
        pr.stats, pr.S, pr.lam, pr.stats.Y_mean, groups=pr.groups
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warm, trace = ecme(pr.stats, init, max_iter=40, tol=1e-300, groups=pr.groups)
    scale = np.maximum(np.abs(trace[:-1]), pr.stats.n_obs.sum() * pr.stats.n_bins)
    assert (np.diff(trace) / scale).max() <= 1e-13
    assert warm.groups == ((0, 2), (1,), (3,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # refine itself meets its tolerance
        fit = refine(pr.stats, warm, groups=pr.groups)
    assert fit.log_likelihood >= warm.log_likelihood - 1e-12 * abs(warm.log_likelihood)
    assert fit.groups == warm.groups
    assert [w.shape for w in fit.W] == [(10, 2), (10, 1), (10, 1)]
    assert [s.shape for s in fit.S] == [(4, 2), (4, 1), (4, 2), (4, 1)]
    for p, g in enumerate((0, 1, 0, 2)):
        np.testing.assert_allclose(fit.B[p], fit.W[g] @ fit.S[p].T)
    # `converged` is provenance: ECME stopped on its cap of 40 on purpose.
    assert not warm.converged
    assert fit.n_iter["ecme"] == 40
    assert fit.n_iter["refine"] < 10


def test_group_span_maps_scale_the_joint_in_span_directions_only() -> None:
    # With groups the soft directions are S_p -> S_p M_g, one M_g per group:
    # D scales those by alpha, leaves the complement of the stacked span alone
    # (directions S_p M_p with different M_p are not all inside it), is
    # symmetric, and D^{-1} undoes it; a rank-0 group is skipped.
    rng = np.random.default_rng(0)
    T = 5
    groups = [[0, 2], [1], [3]]
    S = [rng.normal(size=(T, 2)), rng.normal(size=(T, 3)), rng.normal(size=(T, 2))]
    S.append(np.zeros((T, 0)))
    blocks = mmle._blocks(S, _groups.check_grouping(groups, 4))
    stretch, shrink = mmle._span_maps(blocks, 30.0)
    Mg, M1 = rng.normal(size=(2, 2)), rng.normal(size=(3, 3))
    inside = np.hstack([S[0] @ Mg, S[1] @ M1, S[2] @ Mg])
    np.testing.assert_allclose(stretch(inside.ravel()), 30.0 * inside.ravel())
    np.testing.assert_allclose(shrink(inside.ravel()), inside.ravel() / 30.0)
    stacked = np.vstack([S[0], S[2]])
    Q = np.linalg.qr(stacked)[0]
    out = rng.normal(size=(2 * T, 2))
    out -= Q @ (Q.T @ out)
    outside = np.hstack([out[:T], np.zeros((T, 3)), out[T:]])
    np.testing.assert_allclose(stretch(outside.ravel()), outside.ravel(), atol=1e-12)
    # A per-column in-span direction with different M_p is not scaled by alpha.
    mixed = np.hstack([S[0] @ Mg, np.zeros((T, 3)), S[2] @ rng.normal(size=(2, 2))])
    assert not np.allclose(stretch(mixed.ravel()), 30.0 * mixed.ravel())
    a, b = rng.normal(size=blocks.S.size), rng.normal(size=blocks.S.size)
    assert stretch(a) @ b == pytest.approx(a @ stretch(b), rel=1e-12)
    np.testing.assert_allclose(shrink(stretch(a)), a, rtol=1e-12, atol=1e-12)
    none = mmle._blocks([np.zeros((T, 0))] * 2, _groups.check_grouping([[0, 1]], 2))
    up, _ = mmle._span_maps(none, 30.0)
    assert up(np.zeros(0)).shape == (0,)


@pytest.mark.parametrize(
    ("kind", "equivariant"),
    [("orthogonal", True), ("permutation", True), ("not orthogonal", False)],
)
def test_group_span_maps_are_equivariant_under_orthogonal_recodings(
    kind: str, equivariant: bool
) -> None:
    # Recoding a group's columns by R maps its stacked bases by A = R' (x) I_T;
    # the projector onto their span becomes A Pi A', so D'(A x) = A D(x) for an
    # orthogonal R. A non-orthogonal R changes the metric: no equivariance.
    rng = np.random.default_rng(1)
    T, groups = 5, [[0, 2], [1], [3]]
    S = [rng.normal(size=(T, 2)), rng.normal(size=(T, 3)), rng.normal(size=(T, 2))]
    S.append(np.zeros((T, 0)))
    R = {
        "orthogonal": np.linalg.qr(rng.normal(size=(2, 2)))[0],
        "permutation": np.eye(2)[[1, 0]],
        "not orthogonal": np.array([[1.0, 0.8], [0.0, 1.5]]),
    }[kind]
    grouping = _groups.check_grouping(groups, 4)
    blocks = mmle._blocks(S, grouping)
    cols = [slice(blocks.bounds[p], blocks.bounds[p + 1]) for p in (0, 2)]

    def recode(x: FloatArray) -> FloatArray:
        Z = x.reshape(T, -1).copy()
        a, c = Z[:, cols[0]].copy(), Z[:, cols[1]].copy()
        Z[:, cols[0]], Z[:, cols[1]] = (
            R[0, 0] * a + R[1, 0] * c,
            R[0, 1] * a + R[1, 1] * c,
        )
        return Z.ravel()

    recoded = blocks._replace(S=recode(blocks.S.ravel()).reshape(T, -1))
    stretch = mmle._span_maps(blocks, 30.0)[0]
    stretch_r = mmle._span_maps(recoded, 30.0)[0]
    x = rng.normal(size=blocks.S.size)
    error = _rel(stretch_r(recode(x)), recode(stretch(x)))
    assert (error <= 1e-12) == equivariant, error


@pytest.mark.parametrize("seed", range(3))
def test_grouped_preconditioning_reaches_the_same_optimum_faster(seed: int) -> None:
    # A change of variables: the path changes, the optimum does not. Measured
    # at 300 neurons, 30 bins, 2000 trials (one group of four at rank 4): 28-47
    # basis evaluations against 65-75 plain, log-likelihoods equal to 1e-6.
    labels = np.random.default_rng(seed).choice(5, 300)
    coding = categorical_design(labels, name="c")
    sim = simulate(
        n_neurons=40,
        n_bins=8,
        n_trials=300,
        ranks={"c": 2},
        regressor_names=list(coding.columns),
        groups=coding.groups,
        X=coding.X,
        drop_prob=0.2,
        seed=seed,
    )
    stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    groups = [[0, 1, 2, 3]]
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
            fits[key] = fit_mmle(stats, [2], groups=groups, basis_span_scale=scale)
    plain, scaled = fits["plain"], fits["scaled"]
    assert scaled.converged
    assert scaled.log_likelihood >= plain.log_likelihood - 1e-11 * abs(
        plain.log_likelihood
    )
    assert max(_rel(a, b) for a, b in zip(scaled.B, plain.B, strict=True)) < 1e-4
    assert evals["scaled"] < evals["plain"]


# ------------------------------------------------------------------ fit objects


def test_fit_mmle_with_groups_starts_from_the_grouped_svd() -> None:
    pr = Problem("a group and singletons", seed=13, N=80, n=10, T=4)
    svd = fit_svd(pr.stats, pr.ranks, groups=pr.groups)
    a = fit_mmle(pr.stats, pr.ranks, groups=pr.groups)
    b = fit_mmle(pr.stats, pr.ranks, svd, groups=pr.groups)
    np.testing.assert_array_equal(a.W_cov, b.W_cov)
    assert a.n_parameters == n_parameters_mmle(pr.ranks, 10, 4, groups=pr.groups)
    start = MMLEFit.from_parameters(
        pr.stats, svd.S, svd.noise_precision, pr.stats.Y_mean, groups=pr.groups
    )
    c = fit_mmle(pr.stats, pr.ranks, start, groups=pr.groups)
    assert c.groups == a.groups
    text = repr(a)
    assert text.startswith(
        "MMLEFit(ranks=(2, 1, 1), groups=((0, 2), (1,), (3,)), n_neurons=10"
    )


@pytest.mark.parametrize("kind", ["mmle", "svd"])
def test_an_init_with_other_groups_is_refused(kind: str) -> None:
    pr = Problem("a group and singletons", seed=14)
    if kind == "svd":
        init: Any = fit_svd(pr.stats, pr.ranks, groups=pr.groups)
    else:
        init = MMLEFit.from_parameters(
            pr.stats, pr.S, pr.lam, pr.stats.Y_mean, groups=pr.groups
        )
    with pytest.raises(ParameterError, match="init was fitted with groups"):
        fit_mmle(pr.stats, [2, 1], init, groups=[[0, 1, 2], [3]])
    with pytest.raises(ParameterError, match="init was fitted with groups"):
        fit_mmle(pr.stats, [1, 1, 1, 1], init)
    if kind == "mmle":
        with pytest.raises(ParameterError, match=r"groups=init.groups"):
            ecme(pr.stats, init)
        with pytest.raises(ParameterError, match=r"groups=init.groups"):
            refine(pr.stats, init)
        ungrouped = MMLEFit.from_parameters(
            pr.stats, [np.ones((3, 1))] * 4, pr.lam, pr.stats.Y_mean
        )
        with pytest.raises(ParameterError, match="init was fitted with groups None"):
            ecme(pr.stats, ungrouped, groups=pr.groups)


def test_matlab_compat_is_refused_with_groups() -> None:
    pr = Problem("one group of four", seed=15)
    init = MMLEFit.from_parameters(
        pr.stats, pr.S, pr.lam, pr.stats.Y_mean, groups=pr.groups
    )
    with pytest.raises(ParameterError, match=r"matlab_compat=True .* no regressor"):
        ecme(pr.stats, init, matlab_compat=True, groups=pr.groups)


def test_group_widths_must_agree() -> None:
    pr = Problem("one group of four", seed=16)
    S = list(pr.S)
    S[1] = np.ones((3, 1))
    with pytest.raises(ParameterError, match=r"S\[0\] 2, S\[1\] 1, S\[2\] 2"):
        marginal_log_likelihood(S, pr.lam, pr.b, pr.stats, groups=pr.groups)
    with pytest.raises(ParameterError, match="ordered by their smallest column"):
        marginal_log_likelihood(pr.S, pr.lam, pr.b, pr.stats, groups=[[1, 2, 3], [0]])
    with pytest.raises(ParameterError, match="in no group"):
        posterior_weights(pr.S, pr.lam, pr.b, pr.stats, groups=[[0, 1, 2]])


def test_fit_mmle_checks_grouped_ranks() -> None:
    pr = Problem("a group and singletons", seed=17)
    with pytest.raises(ParameterError, match="expected one per group"):
        fit_mmle(pr.stats, [1, 1, 1, 1], groups=pr.groups)
    with pytest.raises(ParameterError, match=r"min\(n_neurons, 2 \* n_bins\) = 5"):
        fit_mmle(pr.stats, [6, 1, 1], groups=pr.groups)
    with pytest.raises(ParameterError, match="above min"):
        MMLEFit.from_parameters(
            pr.stats,
            [np.ones((3, 6)), np.ones((3, 1))] * 2,
            pr.lam,
            None,
            groups=[[0, 2], [1, 3]],
        )


GROUPED_BAD_FIELDS: list[tuple[dict[str, Any], str]] = [
    ({"groups": ((0, 1), (2, 3))}, "ranks has 3 entries for 2 groups"),
    ({"groups": ((0,), (1,), (2, 3, 4))}, r"S must be a sequence of 5"),
    ({"ranks": (2, 2, 1)}, r"S\[1\] must be \(n_bins, 2\) and W\[1\]"),
    ({"W": (np.zeros((5, 2)), np.zeros((5, 1)))}, "one per group"),
    ({"groups": ((1, 0, 2), (3,))}, "ranks has 3 entries for 2 groups"),
]


@pytest.mark.parametrize(("change", "match"), GROUPED_BAD_FIELDS)
def test_a_grouped_fit_checks_its_fields(change: dict[str, Any], match: str) -> None:
    pr = Problem("a group and singletons", seed=18)
    fit = MMLEFit.from_parameters(
        pr.stats, pr.S, pr.lam, pr.stats.Y_mean, groups=pr.groups
    )
    with pytest.raises(ParameterError, match=match):
        MMLEFit(**{**fields_of(fit), **change})


def test_a_fit_with_trivial_groups_stores_none_and_old_pickles_load() -> None:
    pr = Problem("a group and singletons", seed=19)
    fit = MMLEFit.from_parameters(
        pr.stats, pr.S, pr.lam, pr.stats.Y_mean, groups=pr.groups
    )
    plain = MMLEFit.from_parameters(pr.stats, [np.ones((3, 1))] * 4, pr.lam, None)
    singles = ((0,), (1,), (2,), (3,))
    assert MMLEFit(**{**fields_of(plain), "groups": singles}).groups is None
    clone = pickle.loads(pickle.dumps(fit))
    assert clone.groups == fit.groups
    assert not clone.W[0].flags.writeable
    # A fit pickled before groups existed has no `groups` key in its state.
    state = dict(plain.__dict__)
    del state["groups"]
    old = MMLEFit.__new__(MMLEFit)
    old.__setstate__(state)
    assert old.groups is None
    assert old.B[0].shape == (5, 3)


def fields_of(fit: Any) -> dict[str, Any]:
    return {name: getattr(fit, name) for name in fit.__dataclass_fields__}


# ------------------------------------------------------------------ counts


def test_grouped_counts() -> None:
    # (M59), (M60) and the selection count n + T sum |g| r_g [+ nT].
    groups = [[0, 1, 2], [3]]
    n, T = 40, 6
    assert n_parameters_svd([2, 1], n, T, groups=groups) == (
        2 * (n + 3 * T - 2) + 1 * (n + T - 1) + n + n * T
    )
    assert n_parameters_mmle([2, 1], n, T, groups=groups) == (
        n + (3 * T * 2 - 1) + (T * 1 - 0) + n * T
    )
    assert n_parameters_mmle([2, 1], n, T, groups=groups, formula="reference") == (
        n + T * (3 * 2 + 1) + n * T
    )
    assert n_parameters_mmle(
        [2, 1], n, T, groups=groups, condition_independent=False
    ) == (n + (3 * T * 2 - 1) + T)
    # Singletons reduce to the ungrouped counts.
    singles = [[0], [1], [2]]
    svd_formulas: tuple[Literal["textbook", "reference"], ...] = (
        "textbook",
        "reference",
    )
    for formula in svd_formulas:
        assert n_parameters_svd([2, 1, 3], n, T, formula=formula) == (
            n_parameters_svd([2, 1, 3], n, T, formula=formula, groups=singles)
        )
    mmle_formulas: tuple[Literal["identifiable", "reference"], ...] = (
        "identifiable",
        "reference",
    )
    for m_formula in mmle_formulas:
        assert n_parameters_mmle([2, 1, 3], n, T, formula=m_formula) == (
            n_parameters_mmle([2, 1, 3], n, T, formula=m_formula, groups=singles)
        )
    with pytest.raises(ParameterError, match="MATLAB code has no regressor groups"):
        n_parameters_svd([2, 1], n, T, groups=groups, formula="reference")
    with pytest.raises(ParameterError, match=r"3 \* n_bins\) = 18"):
        n_parameters_mmle([19, 1], n, T, groups=groups)
    # A group rank above T is within the group's cap.
    assert n_parameters_mmle([10, 1], n, T, groups=groups) > 0


@pytest.mark.parametrize(
    ("groups", "ranks", "ci"),
    [
        ([[0, 1]], [2], True),
        ([[0, 1], [2]], [1, 1], True),
        ([[0, 1]], [3], False),  # above T = 2
        ([[0, 2], [1]], [2, 1], False),
    ],
)
def test_the_grouped_identifiable_count_is_the_jacobian_rank(
    groups: list[list[int]], ranks: list[int], ci: bool
) -> None:
    # (M60): the rank of the Jacobian of (lambda, S, b) -> every neuron's
    # marginal mean and covariance, which depends on each group's bases only
    # through [S_p]_p [S_p]_p' (one gauge O_g per group).
    n, T, N = 3, 2, 8
    grouping = _groups.check_grouping(groups, sum(len(g) for g in groups))
    widths = grouping.column_ranks(ranks)
    rng = np.random.default_rng(sum(ranks) + 10 * ci + len(groups))
    X = rng.normal(size=(N, grouping.n_columns))
    sizes = [T * w for w in widths]
    theta0 = np.concatenate(
        [rng.normal(size=n), rng.normal(size=sum(sizes))]
        + ([rng.normal(size=n * T)] if ci else [])
    )

    def moments(theta: FloatArray) -> FloatArray:
        lam = np.exp(theta[:n])
        offsets = np.cumsum([n, *sizes])
        S = [
            theta[a:b].reshape(T, w)
            for a, b, w in zip(offsets, offsets[1:], widths, strict=False)
        ]
        out = []
        Phi = dense.design(X, S, groups)
        for i in range(n):
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
    assert rank == n_parameters_mmle(
        ranks, n, T, condition_independent=ci, groups=groups
    )


# ------------------------------------------------------------------ the SVD start


def test_grouped_svd_truncates_the_concatenated_blocks() -> None:
    # (M58): W_g = U_r Sigma_r^(1/2), S_p the block of V_r Sigma_r^(1/2), the
    # best rank-r approximation of [B_hat_p]_p, largest |U| entry positive.
    pr = Problem("a group and singletons", seed=20, N=80, n=10, T=4)
    full = fit_svd(pr.stats, [4, 1, 1, 1])
    fit = fit_svd(pr.stats, pr.ranks, groups=pr.groups)
    joined = np.concatenate([full.B_full[0], full.B_full[2]], axis=1)
    U, s, Vt = np.linalg.svd(joined, full_matrices=False)
    best = (U[:, :2] * s[:2]) @ Vt[:2]
    np.testing.assert_allclose(
        np.concatenate([fit.B[0], fit.B[2]], axis=1), best, atol=1e-12 * s[0]
    )
    lead = fit.W[0][np.argmax(np.abs(fit.W[0]), axis=0), [0, 1]]
    assert (lead > 0).all()
    np.testing.assert_array_equal(fit.B[1], fit.W[1] @ fit.S[1].T)
    assert fit.n_parameters == n_parameters_svd(pr.ranks, 10, 4, groups=pr.groups)
    assert repr(fit).startswith("SVDFit(ranks=(2, 1, 1), groups=((0, 2), (1,), (3,))")
    # The unconstrained fit does not depend on the grouping.
    for a, b in zip(fit.B_full, full.B_full, strict=True):
        np.testing.assert_array_equal(a, b)


def test_grouped_weighted_svd_is_the_row_weighted_truncation() -> None:
    # (M18w) with (M58), from scratch: each neuron's least squares on its
    # observed trials, d_i ~ sqrt(n_i lambda_i) with lambda_i = n_i T / RSS_i
    # (unit root mean square), the SVD of D [B_hat_p]_p mapped back by D^-1.
    # Heteroskedastic precisions and a mask, truncated below the true rank, so
    # weighting changes the fit.
    groups, ranks, T = [[0, 2, 3], [1]], [1, 1], 4
    names = ["a", "b", "c", "d"]
    sim = simulate(
        n_neurons=12,
        n_bins=T,
        n_trials=150,
        ranks={"g": 3, "b": 2},
        regressor_names=names,
        groups={"g": ["a", "c", "d"]},
        noise_precision=np.geomspace(0.01, 100.0, 12),
        drop_prob=0.3,
        seed=4,
    )
    stats = sufficient_statistics(sim.Y_masked, sim.X, sim.mask)
    beta, count, rss = np.zeros((12, 4, T)), np.zeros(12), np.zeros(12)
    for i in range(12):
        rows = sim.mask[:, i]
        A = np.column_stack([sim.X[rows], np.ones(rows.sum())])
        coef = np.linalg.lstsq(A, sim.Y[rows, i], rcond=None)[0]
        beta[i], count[i] = coef[:4], rows.sum()
        rss[i] = np.sum((sim.Y[rows, i] - A @ coef) ** 2)
    d = np.sqrt(count * count * T / rss)
    d /= np.sqrt(np.mean(d**2))
    fit = fit_svd(stats, ranks, groups=groups, precision_weighted=True)
    plain = fit_svd(stats, ranks, groups=groups)
    errors = []
    for g, (members, r) in enumerate(zip(groups, ranks, strict=True)):
        joined = np.concatenate([beta[:, p] for p in members], axis=1)
        U, s, Vt = np.linalg.svd(d[:, None] * joined, full_matrices=False)
        sign = np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])])[:r]
        root = np.sqrt(s[:r])
        assert _rel(fit.W[g], U[:, :r] * sign * root / d[:, None]) <= ORACLE_RTOL
        V = Vt[:r].T * sign * root
        for j, p in enumerate(members):
            assert _rel(fit.S[p], V[j * T : (j + 1) * T]) <= ORACLE_RTOL
            assert _rel(fit.B_full[p], beta[:, p]) <= ORACLE_RTOL
        errors.append(
            [
                np.linalg.norm(d[:, None] * (_joined(f.B, members) - joined))
                for f in (fit, plain)
            ]
        )
    # The group's D-weighted error is materially below the unweighted one's.
    weighted, unweighted = errors[0]
    assert weighted < (1 - WEIGHTED_GAIN) * unweighted


#: The weighted truncation's relative improvement in its own norm, at least
#: (measured 0.21).
WEIGHTED_GAIN = 0.1


def _joined(B: Any, members: list[int]) -> FloatArray:
    return np.concatenate([B[p] for p in members], axis=1)


def test_the_grouped_svd_start_is_invariant_to_orthogonal_recodings() -> None:
    # [B_hat_p] is right-multiplied by R (x) I_T, so the truncation's implied
    # coefficients map exactly (to rounding).
    pr = Problem("one group of four", seed=22, N=80, n=10, T=4)
    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    stats_r = sufficient_statistics(
        np.where(pr.mask[:, :, None], pr.Y, np.nan), pr.X @ R, pr.mask
    )
    a = fit_svd(pr.stats, [2], groups=pr.groups)
    b = fit_svd(stats_r, [2], groups=pr.groups)
    Ba = np.stack(a.B)  # (4, n, T), per column
    Bb = np.stack(b.B)
    # X B = (X R)(R' B): the recoded coefficients are R' B.
    np.testing.assert_allclose(np.einsum("qp,qit->pit", R, Ba), Bb, atol=1e-10)
    assert a.log_likelihood == pytest.approx(b.log_likelihood, rel=1e-12)


def test_svdfit_checks_its_widths_and_loads_old_pickles() -> None:
    pr = Problem("a group and singletons", seed=23)
    fit = fit_svd(pr.stats, pr.ranks, groups=pr.groups)
    fields = fields_of(fit)
    with pytest.raises(ParameterError, match=r"S\[1\] must have 2 columns"):
        SVDFit(**{**fields, "ranks": (2, 2, 1)})
    with pytest.raises(ParameterError, match="3 ranks and 3 weight factors for 2"):
        SVDFit(**{**fields, "groups": ((0, 1, 2), (3,))})
    assert SVDFit(
        **{**fields_of(fit_svd(pr.stats, [1] * 4)), "groups": None}
    ).groups is (None)
    clone = pickle.loads(pickle.dumps(fit))
    assert clone.groups == fit.groups
    state = dict(fit_svd(pr.stats, [1] * 4).__dict__)
    del state["groups"]
    old = SVDFit.__new__(SVDFit)
    old.__setstate__(state)
    assert old.groups is None
