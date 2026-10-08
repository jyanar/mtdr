"""N-version agreement of `mtdr.mmle` with the MATLAB-derived oracle.

Two implementations written independently, `mtdr.mmle` from `docs/model.md`
only and `tests/nversion/reference_mmle.py` line by line from the MATLAB only,
must agree to 1e-8 before either is merged. They are compared on the datasets
and the ten random parameter points of `tests/fixtures/mmle.mat` (five per
dataset, the last with ridge `g = 0.7`), on the ECME iterates the fixture
recorded, and on small random problems with partial masks and zero ranks;
where the fixture holds MATLAB's own output it is compared too, so a
disagreement can be placed on one side (the oracle matches MATLAB to 1e-12,
`tests/nversion/test_reference_mmle_vs_matlab.py`).

The measure is the max absolute difference relative to the largest entry of
the reference quantity (`_rel`); `pytest -s` prints every measured value. The
oracle implements only the reference's stale-variable ECME, so ECME is
compared on `ecme(..., matlab_compat=True)`; the default path's (M31)/(M33)
steps are tested against dense references in `tests/test_mmle.py`. The tests
skip when the fixture is missing.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import matlab_compat as mc
from mtdr import mmle
from mtdr.errors import ConvergenceWarning
from mtdr.stats import SufficientStats, sufficient_statistics
from nversion import mmle_fixture as mf
from nversion import reference_mmle as rm

TOL = 1e-8
N_POINTS = 5
N_FITS = 3
POINTS = [(name, k) for name in mf.DATASETS for k in range(N_POINTS)]
FITS = [(name, k) for name in mf.DATASETS for k in range(N_FITS)]


def _rel(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    if b.size == 0:
        return 0.0
    scale = max(float(np.abs(b).max()), np.finfo(np.float64).tiny)
    return float(np.abs(a - b).max() / scale)


def _rel_blocks(a: Sequence[NDArray[Any]], b: Sequence[NDArray[Any]]) -> float:
    """Relative to the largest entry over all blocks (a gradient is one vector)."""
    assert len(a) == len(b)
    flat_a = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in a])
    flat_b = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in b])
    for x, y in zip(a, b, strict=True):
        assert np.shape(x) == np.shape(y)
    return _rel(flat_a, flat_b)


def _rel_fit(fit: mmle.MMLEFit, ref: mf.Params) -> float:
    assert fit.intercept is not None
    return max(
        max(_rel(x, y) for x, y in zip(fit.S, ref.S, strict=True)),
        _rel(fit.noise_precision, ref.lam),
        _rel(fit.intercept, ref.b),
    )


def _report(label: str, errors: dict[str, float], tol: float = TOL) -> None:
    shown = ", ".join(f"{k} {v:.1e}" for k, v in errors.items())
    print(f"\n[{label}] max relative error: {shown}")
    bad = {k: v for k, v in errors.items() if not v < tol}
    assert not bad, bad


def _compat_sweeps(
    stats: SufficientStats, start: mmle.MMLEFit, sweeps: int
) -> tuple[mmle.MMLEFit, NDArray[np.float64]]:
    """`sweeps` compat ECME iterations; the stop rule is not what is under test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return mmle.ecme(stats, start, max_iter=sweeps, matlab_compat=True)


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    if not mf.MMLE_FIXTURE.is_file():
        pytest.skip(
            "tests/fixtures/mmle.mat not available; regenerate it with "
            "tests/fixtures/matlab/make_mmle_fixture.m"
        )
    return mf.load()


# ------------------------------------------------------------------- the two loaders


@pytest.mark.parametrize("name", mf.DATASETS)
def test_the_two_loaders_unpack_alike(fixture: dict[str, Any], name: str) -> None:
    # The two independently written unpackers, `fixture_pars_to_mmle` (from
    # `docs/model.md` § 1.2) and the reference port's `unpack_pars` (from the
    # MATLAB), on every parameter vector in the file.
    ds = fixture[name]
    vectors: list[tuple[NDArray[np.float64], mf.Params]] = []
    for fit in ds.fits:
        vectors.append((fit["pars_final"], fit["final"]))
        vectors.append((mf.pack_pars(fit["start"]), fit["start"]))
    for params in ds.search["calls_params"]:
        vectors.append((mf.pack_pars(params), params))
    for v, params in vectors:
        lam, S, b = mc.fixture_pars_to_mmle(v, ds.n, ds.T, params.ranks)
        assert b is not None
        np.testing.assert_array_equal(lam, params.lam)
        np.testing.assert_array_equal(b, params.b)
        for x, y in zip(S, params.S, strict=True):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(mc.mmle_to_pars(lam, S, b), v)


# ------------------------------------------------------------- likelihood, gradients


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_marginal_likelihood(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    pt = ds.points[k]
    st, S, lam, b, g = ds.stats, pt["S"], pt["lam"], pt["b"], pt["g"]
    ll = mmle.marginal_log_likelihood(S, lam, b, st)
    objective = mmle.marginal_log_likelihood(S, lam, b, st, basis_ridge=g)
    const = mmle._constant(st.n_obs, st.n_bins)
    errors = {
        "fully normalised NLL": _rel(-ll, rm.nll(S, lam, b, st)),
        "MATLAB constants only": _rel(
            -ll - const, rm.nll(S, lam, b, st, matlab_constants_only=True)
        ),
        "constant": _rel(const, rm.normalising_constant(st)),
        "penalised objective": _rel(-objective, rm.nll(S, lam, b, st, g=g)),
        "vs MATLAB nllonly": _rel(-objective - const, pt["nll_nllonly"]),
        "vs MATLAB nllonly, uncentred": _rel(
            -mmle.marginal_log_likelihood(S, lam, None, st, basis_ridge=g) - const,
            pt["nll_nllonly_raw"],
        ),
    }
    _report(f"{name} point {k} r={pt['ranks']} g={g}", errors)


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_per_neuron_terms(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    pt = ds.points[k]
    st, S, lam, b = ds.stats, pt["S"], pt["lam"], pt["b"]
    xi, ups = st.centered(b)
    ours = mmle._nll_terms(S, lam, xi, st.XtX, ups, st.n_obs, st.n_bins)
    theirs = rm.nll_terms(S, lam, b, st)
    errors = {
        "per-neuron terms": _rel(ours, theirs),
        "oracle sum vs oracle nll": _rel(
            theirs.sum(), rm.nll(S, lam, b, st, matlab_constants_only=True)
        ),
    }
    _report(f"{name} point {k} r={pt['ranks']}", errors)


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_gradient_in_the_bases(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    pt = ds.points[k]
    st, S, lam, b, g = ds.stats, pt["S"], pt["lam"], pt["b"], pt["g"]
    value, grad = mmle.marginal_nll_grad_S(S, lam, b, st, basis_ridge=g)
    o_value, o_grad = rm.nll_grad_S(S, lam, b, st, g=g)
    const = mmle._constant(st.n_obs, st.n_bins)
    errors = {
        "gradient vs oracle": _rel_blocks(grad, o_grad),
        "gradient vs MATLAB Sonly": _rel_blocks(grad, pt["grad_S"]),
        # The port's value is the function its gradient is of, (M24) plus
        # g/2 ||s||^2; the reference's Sonly value penalises s[n:] only (it
        # slices `pars(n+1:end)` of a vector that does not start with lambda),
        # so with g > 0 it is compared with nll.
        "value vs oracle nll(g)": _rel(value, rm.nll(S, lam, b, st, g=g)),
    }
    if g == 0:
        errors["value vs MATLAB Sonly"] = _rel(value - const, pt["nll_Sonly"])
        errors["value vs oracle Sonly"] = _rel(value, o_value)
    _report(f"{name} point {k} r={pt['ranks']} g={g}", errors)
    if g and ds.n < ds.T * sum(pt["ranks"]):
        # The ridge-slicing defect: reproduced by the oracle and MATLAB, not by
        # the package.
        assert abs(value - const - pt["nll_Sonly"]) > 1e-6 * abs(value)


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_gradient_in_the_precisions(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    pt = ds.points[k]
    st, S, lam, b = ds.stats, pt["S"], pt["lam"], pt["b"]
    value, grad = mmle.marginal_nll_grad_noise(S, lam, b, st)
    o_value, o_grad = rm.nll_grad_lam(S, lam, b, st)
    errors = {
        "gradient vs oracle": _rel(grad, o_grad),
        "gradient vs MATLAB lambonly": _rel(grad, pt["grad_lam"]),
        "value vs oracle": _rel(value, o_value),
        "value vs MATLAB lambonly": _rel(
            value - mmle._constant(st.n_obs, st.n_bins), pt["nll_lambonly"]
        ),
    }
    _report(f"{name} point {k} r={pt['ranks']}", errors)


# -------------------------------------------------------------- posterior, intercept


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_posterior_weights(fixture: dict[str, Any], name: str, k: int) -> None:
    # The port returns the posterior covariance C_i^{-1}; EBpost_W_uneqvar
    # returns the precision C_i (the oracle offers both): both are compared.
    ds = fixture[name]
    pt = ds.points[k]
    st, S, lam, b = ds.stats, pt["S"], pt["lam"], pt["b"]
    W, W_cov = mmle.posterior_weights(S, lam, b, st)
    o_W, o_cov = rm.posterior_W(S, lam, b, st)
    precision = rm.feature_precision(S, lam, st)
    errors = {
        "means vs oracle": max(_rel(x, y) for x, y in zip(W, o_W, strict=True)),
        "covariance vs oracle": _rel(W_cov, o_cov),
        "precision vs oracle": _rel(np.linalg.inv(W_cov), precision),
        "means vs MATLAB Wt": max(_rel(x, y) for x, y in zip(W, pt["W"], strict=True)),
        "precision vs MATLAB Ci": _rel(np.linalg.inv(W_cov), pt["Ci_post"]),
        "covariance vs inv(MATLAB Ci)": _rel(W_cov, np.linalg.inv(pt["Ci_post"])),
    }
    _report(f"{name} point {k} r={pt['ranks']}", errors)


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_intercept_update(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    pt = ds.points[k]
    b = mmle.update_intercept(pt["S"], pt["lam"], ds.stats)
    errors = {
        "vs oracle update_b": _rel(b, rm.update_b(pt["S"], pt["lam"], ds.stats)),
        "vs MATLAB MMLE_b": _rel(b, pt["b_mmle"]),
    }
    _report(f"{name} point {k} r={pt['ranks']}", errors)


# ----------------------------------------------------------------- ECME, compat path


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_one_compat_sweep(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    pt = ds.points[k]
    st = ds.stats
    start = mmle.MMLEFit.from_parameters(st, pt["S"], pt["lam"], pt["b"])
    fit, trace = _compat_sweeps(st, start, 1)
    assert fit.n_iter["ecme"] == 1
    o_S, o_lam, o_b = rm.ecme_step(pt["S"], pt["lam"], pt["b"], st)
    oracle = mf.Params(S=o_S, lam=o_lam, b=o_b)
    const = mmle._constant(st.n_obs, st.n_bins)
    *_, o_trace = rm.ecme("steps", 1, pt["S"], pt["lam"], pt["b"], st)
    errors = {
        "vs oracle ecme_step": _rel_fit(fit, oracle),
        "vs MATLAB ECMEtdr('steps',1)": _rel_fit(fit, pt["ecme_next"]),
        "NLL trace vs oracle": _rel(trace, o_trace),
        "NLL trace vs MATLAB": _rel(trace - const, pt["ecme_nll"]),
    }
    _report(f"{name} point {k} r={pt['ranks']}", errors)


@pytest.mark.parametrize(("name", "k"), FITS)
def test_compat_iterates_step_by_step(
    fixture: dict[str, Any], name: str, k: int
) -> None:
    # One compat sweep from each MATLAB iterate against the next.
    ds = fixture[name]
    ft = ds.fits[k]
    st = ds.stats
    errs = []
    for a, c in itertools.pairwise(ft["ecme_iterates"]):
        start = mmle.MMLEFit.from_parameters(st, a.S, a.lam, a.b)
        errs.append(_rel_fit(_compat_sweeps(st, start, 1)[0], c))
    print(f"\n[{name} r={ft['ranks']}] per-sweep errors: {[f'{e:.1e}' for e in errs]}")
    assert max(errs) < TOL


@pytest.mark.parametrize(("name", "k"), FITS)
def test_compat_loop_matches_the_reference_run(
    fixture: dict[str, Any], name: str, k: int
) -> None:
    # The package's own chained iterates from the recorded SVD start against
    # every recorded ECMEtdr iterate, and the full loop under its own stop rule
    # (relative change with a floor, tol = 1) against ECMEtdr('converge', 1e-0):
    # the same number of sweeps, the same end point and NLL trace.
    ds = fixture[name]
    ft = ds.fits[k]
    st = ds.stats
    s0 = ft["start"]
    start = mmle.MMLEFit.from_parameters(st, s0.S, s0.lam, s0.b)
    chained = []
    cur = start
    for ref in ft["ecme_iterates"][1:]:
        cur, _ = _compat_sweeps(st, cur, 1)
        chained.append(_rel_fit(cur, ref))
    fit, trace = mmle.ecme(st, start, matlab_compat=True)  # no warning: it stops
    assert fit.converged
    assert fit.n_iter["ecme"] == ft["ecme_n_sweeps"]
    o_S, o_lam, o_b, o_parerr, _ = rm.ecme(
        "converge", 1.0, s0.S, s0.lam, s0.b, st, matlab_constants_only=True
    )
    assert o_parerr.size == ft["ecme_n_sweeps"]
    errors = {
        "chained iterates vs MATLAB": max(chained),
        "final vs MATLAB": _rel_fit(fit, ft["ecme"]),
        "final vs oracle": _rel_fit(fit, mf.Params(S=o_S, lam=o_lam, b=o_b)),
        "NLL trace vs MATLAB": _rel(
            trace - mmle._constant(st.n_obs, st.n_bins), ft["ecme_nll"]
        ),
    }
    _report(
        f"{name} r={ft['ranks']} ({fit.n_iter['ecme']} sweeps, MATLAB "
        f"{ft['ecme_n_sweeps']})",
        errors,
    )


# ------------------------------------------------------------- random small problems


def _random_problem(
    seed: int, ranks: Sequence[int]
) -> tuple[SufficientStats, list[NDArray[np.float64]], NDArray[np.float64], Any]:
    rng = np.random.default_rng(seed)
    N, n, T = 40, 6, 5
    X = rng.normal(size=(N, 3))
    Y = rng.normal(size=(N, n, T)) + 1.0
    mask = rng.random((N, n)) < 0.7
    mask[:6] = True
    stats = sufficient_statistics(np.where(mask[:, :, None], Y, np.nan), X, mask)
    S = [rng.normal(size=(T, r)) for r in ranks]
    lam = rng.uniform(0.5, 2.0, n)
    b = rng.normal(size=(n, T)) + 1.0
    return stats, S, lam, b


@pytest.mark.parametrize(
    ("seed", "ranks"),
    [(0, [2, 1, 3]), (1, [0, 2, 1]), (2, [1, 0, 0]), (3, [3, 3, 3]), (4, [5, 1, 2])],
)
def test_random_problems_term_by_term(seed: int, ranks: list[int]) -> None:
    stats, S, lam, b = _random_problem(seed, ranks)
    xi, ups = stats.centered(b)
    T = stats.n_bins
    terms = mmle._nll_terms(S, lam, xi, stats.XtX, ups, stats.n_obs, T)
    _, grad_S = mmle.marginal_nll_grad_S(S, lam, b, stats)
    _, grad_lam = mmle.marginal_nll_grad_noise(S, lam, b, stats)
    W, W_cov = mmle.posterior_weights(S, lam, b, stats)
    o_W, o_cov = rm.posterior_W(S, lam, b, stats)
    start = mmle.MMLEFit.from_parameters(stats, S, lam, b)
    fit, _ = _compat_sweeps(stats, start, 1)
    o_S, o_lam, o_b = rm.ecme_step(S, lam, b, stats)
    errors = {
        "terms": _rel(terms, rm.nll_terms(S, lam, b, stats)),
        "grad S": _rel_blocks(grad_S, rm.nll_grad_S(S, lam, b, stats)[1]),
        "grad lambda": _rel(grad_lam, rm.nll_grad_lam(S, lam, b, stats)[1]),
        "W": _rel_blocks(W, o_W),
        "W_cov": _rel(W_cov, o_cov),
        "intercept": _rel(
            mmle.update_intercept(S, lam, stats), rm.update_b(S, lam, stats)
        ),
        "compat sweep": _rel_fit(fit, mf.Params(S=o_S, lam=o_lam, b=o_b)),
    }
    _report(f"random seed {seed} r={ranks}", errors)
