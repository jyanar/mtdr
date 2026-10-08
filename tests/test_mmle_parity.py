"""MATLAB parity of the marginal-likelihood estimator.

`tests/fixtures/mmle.mat` (generator `tests/fixtures/matlab/make_mmle_fixture.m`,
loader `tests/nversion/mmle_fixture.py`) holds the reference run on the two
datasets of `svd.mat` (A: n=20, B: n=100; T=15, N=40, true ranks [2 1 3]) at
the rank vectors [2 1 3], [2 1 2], [3 1 3]: the SVD start, every ECME iterate,
the coordinate-ascent trace and final estimate, the posterior,
`MakeBhat_data`, the AIC, and the MMLE `EstRankGreedily` search with every
objective call logged.

What is compared, and at what tolerance:

* the package's SVD start against the reference's (`ECMEregress_wrapper`),
  1e-8 after the two documented translations (the column signs of `S`, the
  (M19a) precision replica of `tests/matlab_compat.py`);
* `refine` from the reference's post-ECME estimate, and the default- and
  compat-path pipelines from the reference's start, against the reference's
  final estimate: the gate is the marginal NLL, within 1e-8 relative either
  way from the same start (refine, and the compat path, which reproduces the
  reference's ECME iterates) and at least as good as the reference's to 1e-7
  on the other paths (the reference runs minFunc + fminunc, the port
  L-BFGS-B, so the as-run estimates agree only to optimizer tolerance; their
  gaps are printed and bounded);
* the estimates themselves at the parity target of 1e-4, between the tightly
  refined endpoints: the reference's final estimate and `fit_mmle`'s fit, each
  refined far past the default tolerances, reach the same point;
* the posterior, `MMLEFit.B`, the AIC and the rank search at the reference's
  own estimates, 1e-8.

The ECME iterates themselves are compared in `tests/test_mmle_nversion.py`.
Every fixture test skips when the file is missing. A's loader tests come
first: they pin the `[lambda; s; vec(b)]` packing of `docs/model.md` § 1.2 on
a block with `r_p != T`, where a transposition would show.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
import pytest
from numpy.typing import NDArray

import matlab_compat as mc
from mtdr import aic, mmle
from mtdr.errors import ConvergenceWarning
from mtdr.rank_search import greedy_aic
from mtdr.stats import SufficientStats
from mtdr.svd_fit import fit_svd
from nversion import mmle_fixture as mf

TOL = 1e-8
# The NLL gates of a final estimate against the reference's, relative to
# |NLL|. From the same start (refine; the compat path), two-sided: measured at
# most 1.4e-9. On the other paths (the corrected ECME, the package's own
# start), one-sided, at least as good: measured at most +7.6e-9 (and as low as
# -8.3e-7, the port better). A hundred times looser L-BFGS-B ftol (1e-9)
# raises the same-start ratio to 1.1e-8-4.0e-8.
SAME_START_NLL_RTOL = 1e-8
OTHER_PATH_NLL_RTOL = 1e-7
# Estimate parity between the two tightly refined endpoints:
# measured at most 5.5e-6 (B), 5.8e-7 (lambda), 5.5e-7 (b) with these
# settings; the whole test takes about 4 s.
TIGHT_GAP = 1e-4
TIGHT_MAX_ITER, TIGHT_TOL, TIGHT_OPTIMIZER_TOL = 100, 1e-14, 1e-11
TIGHT_OPTIMIZER_MAX_ITER = 5000
#: Bounds on the estimate gaps (B, lambda, b; max entry difference over the
#: largest reference entry). Same start and algorithm up to the optimizer:
#: 1e-3; the corrected ECME path or the package's own start: 1e-2. The
#: reference's own estimates lie up to 1.7e-3 (B) and 3.2e-3 (lambda) from a
#: tightly converged optimum on the under-ranked [2 1 2] fits, so the 1e-4
#: parity target on estimates is not attainable against them.
SAME_START_GAP = 1e-3
OTHER_PATH_GAP = 1e-2
FITS = [(name, k) for name in mf.DATASETS for k in range(3)]

R = TypeVar("R")


def _rel(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    scale = max(float(np.abs(b).max()), np.finfo(np.float64).tiny)
    return float(np.abs(a - b).max() / scale)


def _rel_blocks(a: Sequence[NDArray[Any]], b: Sequence[NDArray[Any]]) -> float:
    assert len(a) == len(b)
    return max(_rel(x, y) for x, y in zip(a, b, strict=True))


def _report(label: str, errors: dict[str, float], tol: float = TOL) -> None:
    shown = ", ".join(f"{k} {v:.1e}" for k, v in errors.items())
    print(f"\n[{label}] max relative error: {shown}")
    bad = {k: v for k, v in errors.items() if not v < tol}
    assert not bad, bad


def _quiet(func: Callable[[], R]) -> tuple[R, list[str]]:
    """Call `func`, collecting its `ConvergenceWarning`s instead of raising."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        out = func()
    rest = [w for w in caught if not issubclass(w.category, ConvergenceWarning)]
    assert not rest, [str(w.message) for w in rest]
    return out, [str(w.message) for w in caught]


def _start(stats: SufficientStats, params: mf.Params) -> mmle.MMLEFit:
    return mmle.MMLEFit.from_parameters(stats, params.S, params.lam, params.b)


def _final_parity(
    label: str,
    fit: mmle.MMLEFit,
    ft: dict[str, Any],
    const: float,
    gap: float,
    same_start: bool,
) -> None:
    """Check the NLL gate and the estimate gaps of `fit` against the reference's.

    Two-sided at `SAME_START_NLL_RTOL` from the same start, one-sided at
    `OTHER_PATH_NLL_RTOL` otherwise.
    """
    fin = ft["final"]
    nll_ref = ft["nll_final"] + const
    nll = -fit.log_likelihood
    rise = (nll - nll_ref) / abs(nll_ref)
    B_ref = [w @ s.T for w, s in zip(ft["W"], fin.S, strict=True)]
    assert fit.intercept is not None
    gaps = {
        "B": _rel_blocks(fit.B, B_ref),
        "lambda": _rel(fit.noise_precision, fin.lam),
        "b": _rel(fit.intercept, fin.b),
    }
    shown = ", ".join(f"{k} {v:.1e}" for k, v in gaps.items())
    print(
        f"\n[{label}] NLL {nll:.10f} vs reference {nll_ref:.10f}, "
        f"(port - reference)/|NLL| = {rise:.2e}; estimate gaps: {shown}"
    )
    if same_start:
        assert abs(rise) <= SAME_START_NLL_RTOL
    else:
        assert rise <= OTHER_PATH_NLL_RTOL
    bad = {k: v for k, v in gaps.items() if not v < gap}
    assert not bad, bad


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    if not mf.MMLE_FIXTURE.is_file():
        pytest.skip(
            "tests/fixtures/mmle.mat not available; regenerate it with "
            "tests/fixtures/matlab/make_mmle_fixture.m"
        )
    return mf.load()


# ------------------------------------------------------------------ loader


def test_pars_vector_layout_on_blocks_with_r_not_t() -> None:
    n, T, ranks = 3, 5, [2, 1, 3]
    rng = np.random.default_rng(0)
    lam = rng.uniform(1, 2, n)
    S = [rng.normal(size=(T, r)) for r in ranks]
    b = rng.normal(size=(n, T))
    pars = mc.mmle_to_pars(lam, S, b)
    # MATLAB layout, written out: lambda, then every component's time course in
    # regressor order (time fastest), then b column by column of its T x n form.
    courses = [S[p][:, j] for p in range(3) for j in range(ranks[p])]
    expected = np.concatenate([lam, *courses, b.T.ravel(order="F")])
    np.testing.assert_array_equal(pars, expected)
    lam2, S2, b2 = mc.fixture_pars_to_mmle(pars, n, T, ranks)
    np.testing.assert_array_equal(lam2, lam)
    assert b2 is not None
    np.testing.assert_array_equal(b2, b)
    for a, c in zip(S, S2, strict=True):
        np.testing.assert_array_equal(a, c)
    _, S3, b3 = mc.fixture_pars_to_mmle(
        pars[: n + T * 6], n, T, ranks, condition_independent=False
    )
    assert b3 is None
    assert all(np.array_equal(a, c) for a, c in zip(S, S3, strict=True))


def test_pars_vector_length_is_checked() -> None:
    with pytest.raises(ValueError, match="pars has 10 entries"):
        mc.fixture_pars_to_mmle(np.zeros(10), 3, 5, [2, 1])


# ------------------------------------------------------------------ the start


@pytest.mark.parametrize(("name", "k"), FITS)
def test_svd_start_is_the_reference_start(
    fixture: dict[str, Any], name: str, k: int
) -> None:
    # ECMEregress_wrapper: S and lambda from SVDRegress_S_Vdata, b0 = Ybar.
    # fit_svd's S equals the reference's up to the column signs (fit_svd makes
    # each column's largest entry of W positive); its noise precision is the
    # aligned (M19b), not the reference's misaligned (M19a), which
    # tests/matlab_compat.py replicates. fit_svd's intercept refit does not
    # enter: the start discards the SVD's intercept for Ybar.
    ds = fixture[name]
    ft = ds.fits[k]
    s0 = ft["start"]
    svd = fit_svd(ds.stats, ft["ranks"])
    assert svd.intercept_full is not None
    signs = [np.sign(np.sum(a * c, axis=0)) for a, c in zip(svd.S, s0.S, strict=True)]
    replica = mc.svd_lambda_reference(ds.Y, ds.X, ds.mask, [*svd.B, svd.intercept_full])
    errors = {
        "S (signs aligned)": _rel_blocks(
            [a * s for a, s in zip(svd.S, signs, strict=True)], s0.S
        ),
        "lambda, (M19a) replica": _rel(replica, s0.lam),
        "b0 = Ybar": _rel(ds.stats.Y_mean, s0.b),
    }
    flipped = int(sum(int(np.sum(s < 0)) for s in signs))
    ratio = float(np.median(svd.noise_precision / s0.lam))
    print(
        f"\n[{name} r={ft['ranks']}] {flipped} of {sum(ft['ranks'])} columns of S "
        f"differ in sign; median lambda (M19b) / lambda (M19a): {ratio:.3g}"
    )
    _report(f"{name} r={ft['ranks']} start", errors)
    # So fit_mmle's own start is not the reference's: its precision differs.
    assert _rel(svd.noise_precision, s0.lam) > 0.1


@pytest.mark.parametrize(("name", "k"), FITS)
def test_default_ecme_from_the_reference_start(
    fixture: dict[str, Any], name: str, k: int
) -> None:
    # The corrected (M31)/(M33) path from the reference's start: monotone,
    # converged under the reference's loose tol = 1. Its iterates are not the
    # reference's (`docs/model.md` § C.6); the compat path's are
    # (test_mmle_nversion).
    ds = fixture[name]
    ft = ds.fits[k]
    st = ds.stats
    fit, trace = mmle.ecme(st, _start(st, ft["start"]))
    assert fit.converged
    scale = max(abs(trace[0]), float(st.n_obs.sum()) * st.n_bins)
    assert np.all(np.diff(trace) <= mmle.MONOTONE_RTOL * scale)
    const = mmle._constant(st.n_obs, st.n_bins)
    print(
        f"\n[{name} r={ft['ranks']}] default ECME: {fit.n_iter['ecme']} sweeps, NLL "
        f"{trace[-1]:.6f}; reference ECMEtdr: {ft['ecme_n_sweeps']} sweeps, NLL "
        f"{ft['ecme_nll'][-1] + const:.6f}"
    )


# ------------------------------------------------------------------ final estimates


@pytest.mark.parametrize(("name", "k"), FITS)
def test_refine_from_the_reference_ecme(
    fixture: dict[str, Any], name: str, k: int
) -> None:
    # Estpars_CoordAscent_lambi_S_b and refine from the same point.
    ds = fixture[name]
    ft = ds.fits[k]
    st = ds.stats
    fit, caught = _quiet(lambda: mmle.refine(st, _start(st, ft["ecme"])))
    const = mmle._constant(st.n_obs, st.n_bins)
    _final_parity(
        f"{name} r={ft['ranks']} refine", fit, ft, const, SAME_START_GAP, True
    )
    # Both stop on the (M34) change at 1e-4 or at the 10-iteration cap: the
    # port hits the cap exactly where the reference did (B [2 1 2]).
    ref_converged = bool(ft["ca"]["parerr"][-1] < 1e-4)
    print(
        f"  iterations: port {fit.n_iter['refine']}, reference {len(ft['ca']['nll'])}"
        f"; converged: port {fit.converged}, reference {ref_converged}"
    )
    assert fit.converged == ref_converged
    assert bool(caught) == (not ref_converged)


@pytest.mark.parametrize(("name", "k"), FITS)
@pytest.mark.parametrize("matlab_compat", [False, True])
def test_pipeline_from_the_reference_start(
    fixture: dict[str, Any], name: str, k: int, matlab_compat: bool
) -> None:
    # ECME (corrected or compat) then refine, from the reference's SVD start.
    ds = fixture[name]
    ft = ds.fits[k]
    st = ds.stats
    path = "compat" if matlab_compat else "default"

    def run() -> mmle.MMLEFit:
        warm, _ = mmle.ecme(st, _start(st, ft["start"]), matlab_compat=matlab_compat)
        return mmle.refine(st, warm)

    fit, _ = _quiet(run)
    const = mmle._constant(st.n_obs, st.n_bins)
    # The compat path reproduces the reference's ECME iterates to 1e-12
    # (test_mmle_nversion), so its refine starts where the reference's did.
    gap = SAME_START_GAP if matlab_compat else OTHER_PATH_GAP
    label = f"{name} r={ft['ranks']} {path} ECME + refine"
    _final_parity(label, fit, ft, const, gap, matlab_compat)


@pytest.mark.parametrize(("name", "k"), FITS)
def test_fit_mmle_from_its_own_start(
    fixture: dict[str, Any], name: str, k: int
) -> None:
    # The package's estimator end to end: (M19b) start, corrected ECME, refine.
    ds = fixture[name]
    ft = ds.fits[k]
    st = ds.stats
    fit, _ = _quiet(lambda: mmle.fit_mmle(st, ft["ranks"]))
    const = mmle._constant(st.n_obs, st.n_bins)
    _final_parity(
        f"{name} r={ft['ranks']} fit_mmle", fit, ft, const, OTHER_PATH_GAP, False
    )


@pytest.mark.parametrize(("name", "k"), FITS)
def test_tightly_refined_estimates_agree(
    fixture: dict[str, Any], name: str, k: int
) -> None:
    # The 1e-4 parity target on the estimates, checked between optima rather
    # than between as-run estimates: the reference's final estimate and
    # fit_mmle's fit, each refined far past the default tolerances, reach the
    # same point. Those runs end on caps or on rounding-level line-search
    # failures, so they report non-convergence; the point is the best found,
    # not a certified optimum.
    ds = fixture[name]
    ft = ds.fits[k]
    st = ds.stats
    fin = ft["final"]

    def tight(start: mmle.MMLEFit) -> mmle.MMLEFit:
        return mmle.refine(
            st,
            start,
            max_iter=TIGHT_MAX_ITER,
            tol=TIGHT_TOL,
            optimizer_tol=TIGHT_OPTIMIZER_TOL,
            optimizer_max_iter=TIGHT_OPTIMIZER_MAX_ITER,
        )

    from_ref, _ = _quiet(lambda: tight(_start(st, fin)))
    port, _ = _quiet(lambda: mmle.fit_mmle(st, ft["ranks"]))
    from_port, _ = _quiet(lambda: tight(port))
    assert from_ref.intercept is not None
    assert from_port.intercept is not None
    gaps = {
        "B": _rel_blocks(from_port.B, from_ref.B),
        "lambda": _rel(from_port.noise_precision, from_ref.noise_precision),
        "b": _rel(from_port.intercept, from_ref.intercept),
    }
    nll = -from_ref.log_likelihood
    rise = (-from_port.log_likelihood - nll) / abs(nll)
    print(f"\n[{name} r={ft['ranks']} tight] (port - reference)/|NLL| = {rise:.1e}")
    _report(f"{name} r={ft['ranks']} tight", gaps, TIGHT_GAP)
    # Both are at least as good as the reference's own final estimate.
    nll_ref = ft["nll_final"] + mmle._constant(st.n_obs, st.n_bins)
    assert max(nll, -from_port.log_likelihood) <= nll_ref * (1 + 1e-12)


# ------------------------------------------------------- at the reference's estimate


@pytest.mark.parametrize(("name", "k"), FITS)
def test_posterior_and_b_hat_at_the_reference_estimate(
    fixture: dict[str, Any], name: str, k: int
) -> None:
    ds = fixture[name]
    ft = ds.fits[k]
    st = ds.stats
    fin = ft["final"]
    W, W_cov = mmle.posterior_weights(fin.S, fin.lam, fin.b, st)
    fit = _start(st, fin)
    # mTDRdemo.m:155 passes the uncentred statistic: the same call with
    # intercept=None reproduces its Bhat.
    W_raw, _ = mmle.posterior_weights(fin.S, fin.lam, None, st)
    errors = {
        "Wt": _rel_blocks(W, ft["W"]),
        "W_cov vs inv(Ci)": _rel(W_cov, np.linalg.inv(ft["Ci_post"])),
        "inv(W_cov) vs Ci": _rel(np.linalg.inv(W_cov), ft["Ci_post"]),
        "MMLEFit.W vs What": _rel_blocks(fit.W, ft["What"]),
        "MMLEFit.B vs Bhat": _rel_blocks(fit.B, ft["Bhat"]),
        "MMLEFit.S vs Shat'": _rel_blocks(
            fit.S, [s.T for s in ft["S_matlab_makebhat"]]
        ),
        "lambda vs lambhat": _rel(fit.noise_precision, ft["lam_makebhat"]),
        "Bhat, uncentred (mTDRdemo.m:155)": _rel_blocks(
            [w @ s.T for w, s in zip(W_raw, fin.S, strict=True)], ft["Bhat_raw"]
        ),
    }
    _report(f"{name} r={ft['ranks']}", errors)


@pytest.mark.parametrize(("name", "k"), FITS)
def test_aic_at_the_reference_estimate(
    fixture: dict[str, Any], name: str, k: int
) -> None:
    # BTDR_AIC_S_lamb_b_wrapper: 2 * (NLL without the constant) + 2 numel(pars).
    ds = fixture[name]
    ft = ds.fits[k]
    st = ds.stats
    fit = _start(st, ft["final"])
    const = mmle._constant(st.n_obs, st.n_bins)
    count = aic.n_parameters_mmle(ft["ranks"], ds.n, ds.T, formula="reference")
    assert count == ft["pars_final"].size
    reference_aic = aic.aic(fit.log_likelihood, count) - 2.0 * const
    errors = {
        "NLL": _rel(-fit.log_likelihood - const, ft["nll_final"]),
        "AIC (reference count)": _rel(reference_aic, ft["aic"]),
    }
    _report(f"{name} r={ft['ranks']}", errors)
    # The default identifiable count (M38a) runs on the same estimate.
    rotations = sum(r * (r - 1) // 2 for r in ft["ranks"])
    assert fit.n_parameters == count - rotations
    assert fit.aic == aic.aic(fit.log_likelihood, fit.n_parameters)


@pytest.mark.parametrize("name", mf.DATASETS)
def test_rank_search_replay(fixture: dict[str, Any], name: str) -> None:
    # EstRankGreedily with the MMLE objective, replayed through greedy_aic: the
    # fits are the logged parameter vectors, call by call, and the objective is
    # the package's likelihood with the reference count and without the
    # constant, compared with the logged AIC at every call.
    ds = fixture[name]
    sr = ds.search
    st = ds.stats
    const = mmle._constant(st.n_obs, st.n_bins)
    calls = iter(zip(sr["calls_ranks"], sr["calls_params"], strict=True))
    logged = iter(sr["calls_values"])
    by_ranks: dict[tuple[int, ...], mf.Params] = {}
    errs: list[float] = []

    def fit(r: NDArray[np.int64]) -> mmle.MMLEFit:
        ranks, params = next(calls)
        np.testing.assert_array_equal(r, ranks)  # MATLAB's call order
        by_ranks[tuple(int(x) for x in r)] = params
        return _start(st, params)

    def objective(f: mmle.MMLEFit, r: NDArray[np.int64]) -> float:
        count = aic.n_parameters_mmle(r, ds.n, ds.T, formula="reference")
        value = aic.aic(f.log_likelihood, count) - 2.0 * const
        errs.append(_rel(value, next(logged)))
        return value

    best, history = greedy_aic(fit, objective, sr["rest0"], ds.maxrank)
    assert next(calls, None) is None  # every logged call replayed, none added
    np.testing.assert_array_equal(history.ranks, sr["rhist"])
    np.testing.assert_array_equal(history.ranks[-1], sr["rest"])
    assert history.estimator == "mmle"
    assert history.stop_reason == "no_improvement"
    # parhist{k} is the fit at rhist(k+1, :) (the movable set is every
    # regressor here, so the reference's `parhist{iters} = parhat{indmin}`
    # indexing defect is inert), and the returned fit is the last of them.
    assert len(sr["parhist"]) == len(history.accepted)
    for row, params in zip(history.ranks[1:], sr["parhist"], strict=True):
        got = by_ranks[tuple(int(x) for x in row)]
        np.testing.assert_array_equal(mf.pack_pars(got), mf.pack_pars(params))
    last = sr["parhist"][-1]
    assert best.ranks == tuple(last.ranks)
    np.testing.assert_array_equal(best.noise_precision, last.lam)
    errors = {"AIC per call": max(errs), "FunHist": _rel(history.aic, sr["funhist"])}
    _report(f"{name} search {sr['rest0']} -> {sr['rest']} ({len(errs)} calls)", errors)
