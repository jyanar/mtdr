r"""MATLAB parity at the paper's scale.

`tests/fixtures/paper.mat` (generator `tests/fixtures/matlab/make_paper_fixture.m`)
holds the reference run on a dataset at the scale of Aoi, Mante & Pillow (2020):
$n=800$ neurons recorded in 100 sessions of 8, $T=15$, $N=1000$ trial slots,
six correlated task regressors, true ranks $(5,4,3,4,2,2)$ ($r_{tot}=20$),
heterogeneous precisions, and every neuron observed on 91-288 trials. The
responses are not stored: `tests/paper_data.py` regenerates them bit for bit
from the generator's specification, which the two exact checksums confirm.

What is compared, and at what tolerance:

* the regenerated data: its checksums recomputed here from the returned `Y`
  against the generator's (exactly, with an index-swap negative control) and
  its statistics (1e-12 relative);
* the SVD search with the reference objective (replicated in
  `tests/matlab_compat.py`): the rank path exactly, every value to 1e-6;
  and, from the raw data rather than the statistics, the refitted intercept
  and the corrected score at its final ranks (1e-10);
* at MATLAB's post-ECME and final points of each fit: the marginal NLL
  (1e-12), the AIC under the reference count (1e-12) and, at the true ranks,
  MATLAB's posterior mean $\hat W$ and $\hat W_p\hat S_p^\top$ (1e-10); and
  both parameter counts at the true ranks, exactly;
* at each of three rank vectors (the truth, one under- and one over-ranked):
  the start (1e-8); at the reference caps (`parity_fixtures.REFERENCE_CAPS`),
  `refine` from the reference's post-ECME estimate and the compat ECME then
  `refine` from the reference's start (NLL within 1e-11 either way; the first
  basis step ends on the cap of 500 at the reference's iteration count, and
  `converged` stays False); with the package's defaults, `fit_mmle` from its
  own start (NLL at least as good as the reference's to 1e-7: the shipped
  defaults' check, which also asserts that the basis step stays below its cap
  of 2000); estimate gaps printed and bounded (1e-3 same start, 1e-2
  otherwise);
* at the true ranks, the estimates between tightly refined endpoints to 1e-4,
  and both endpoints at least as good as the reference's to `MONOTONE_RTOL`.
  Both endpoints are port refinements (seeded by MATLAB's and by the port's
  fit), so this test checks where the port's optimiser ends, not the
  objective, which the fixed-point tests above pin to MATLAB;
* the reference's MMLE search from the true ranks, at the reference caps: the path
  exactly, every candidate's NLL within 1e-11.

Every `ConvergenceWarning` is recorded and its reasons are checked against the
classes the test expects (`parity_fixtures.run`). The fits take 5-60 s each at
this scale, so every test that fits is marked `slow` (`pytest -m slow`); CI
runs them in their own job.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import matlab_compat as mc
import paper_data as pdm
import parity_fixtures as pf
from mtdr import aic, mmle
from mtdr.rank_search import greedy_aic
from mtdr.stats import SufficientStats, sufficient_statistics
from mtdr.svd_fit import SVDFit, fit_svd

TOL = 1e-8
FIXED_POINT_TOL = 1e-12
POSTERIOR_TOL = 1e-10
SVD_AIC_TOL = 1e-6
COMPAT_TOL = 1e-11
SAME_START_NLL_RTOL = 1e-11  # at the reference caps
OTHER_PATH_NLL_RTOL = 1e-7
SAME_START_GAP = 1e-3
OTHER_PATH_GAP = 1e-2
TIGHT_GAP = 1e-4
TIGHT: dict[str, Any] = {
    "max_iter": 100,
    "tol": 1e-14,
    "optimizer_tol": 1e-11,
    "optimizer_max_iter": 5000,
}
FITS = range(3)
# The reference's minFunc settings (paper.mat's minfunc_options), against which
# test_reference_inner_options checks the package's defaults.
REFERENCE_INNER = {"maxIter": 500, "maxFunEvals": 1000, "optTol": 1e-5, "progTol": 1e-9}
# Reasons a refinement may warn with here: at this scale the first basis step
# ends on the reference's cap, the under-ranked fit ends on refine's own cap,
# and a precision step may end on a rounding-level line search.
EXPECTED = ("basis_cap", "refine_cap", "precision_rounding")
# A rounding-level basis-step end counts as converged, so it has no class.
PARITY = EXPECTED
# The package defaults: the basis step stays below its cap of 2000, but the
# refine cap remains possible.
DEFAULT_REASONS = ("refine_cap", "precision_rounding")


class _Paper:
    def __init__(self) -> None:
        self.ref = pf.load_paper()
        self.data = pdm.paper_data()
        self.stats: SufficientStats = sufficient_statistics(
            self.data.Y, self.data.X, self.data.mask
        )


@pytest.fixture(scope="module")
def paper() -> _Paper:
    # A required parity fixture: its absence is a failure, not a skip.
    assert pf.PAPER_FIXTURE.is_file(), "tests/fixtures/paper.mat is missing"
    return _Paper()


def _checksums(Y: NDArray[np.float64], mask: NDArray[np.bool_]) -> tuple[int, int]:
    # Independent of paper_data.array_checksums: the observed entries of the
    # completed array, neuron-major, ascending observed trial, bins fastest.
    z = (np.transpose(Y, (1, 0, 2))[mask.T] * pdm.GRID).ravel()
    assert np.array_equal(z, np.round(z))
    zi = z.astype(np.int64)
    w = np.arange(1, zi.size + 1, dtype=np.int64) % 997 + 1
    return int(zi.sum()), int((zi * w).sum())


def test_regenerated_data_match_the_generator(paper: _Paper) -> None:
    ref, data, st = paper.ref, paper.data, paper.stats
    assert _checksums(data.Y, data.mask) == ref.checksums
    assert data.checksums == ref.checksums
    assert (ref.n, ref.T, ref.N) == data.Y.shape[1:] + data.Y.shape[:1]
    assert ref.true_ranks == list(pdm.RANKS)
    np.testing.assert_array_equal(st.n_obs, ref.matlab_stats["ni"])
    np.testing.assert_array_equal(data.precision, ref.precision)
    errors = {
        "YtY_raw": pf.rel(st.YtY_raw, ref.matlab_stats["zzi"]),
        "X_mean": pf.rel(st.X_mean, ref.matlab_stats["xbari"]),
        "sum of Y_mean": pf.rel(st.Y_mean.sum(axis=0), ref.matlab_stats["Ybar_sum"]),
        "XtX[0]": pf.rel(st.XtX[0], ref.matlab_stats["A0"]),
        "XtY_raw[0]": pf.rel(st.XtY_raw[0], ref.matlab_stats["xi0"]),
    }
    pf.report("paper data", errors, 1e-12)
    print(
        f"  n_obs {st.n_obs.min()}-{st.n_obs.max()} (median "
        f"{int(np.median(st.n_obs))}), observed fraction {data.mask.mean():.3f}"
    )
    assert st.n_obs.min() >= 90
    assert st.n_obs.max() <= 300


def test_checksums_catch_an_index_swap(paper: _Paper) -> None:
    # Negative control: swapping the first two observed trials of the
    # last neuron at bin 0 leaves every statistic the data test compares
    # unchanged (the raw energies, the means, neuron 0's cross-products), but
    # not the checksums of the completed array.
    data = paper.data
    Y = data.Y.copy()
    i = Y.shape[1] - 1
    k0, k1 = np.flatnonzero(data.mask[:, i])[:2]
    assert Y[k0, i, 0] != Y[k1, i, 0]
    Y[[k0, k1], i, 0] = Y[[k1, k0], i, 0]
    st = sufficient_statistics(Y, data.X, data.mask)
    assert pf.rel(st.YtY_raw, paper.stats.YtY_raw) < 1e-14
    assert pf.rel(st.XtY_raw[0], paper.stats.XtY_raw[0]) == 0.0
    assert _checksums(Y, data.mask) != paper.ref.checksums
    assert pdm.array_checksums(Y, data.mask) != paper.ref.checksums


def test_svd_search_trajectory(paper: _Paper) -> None:
    ref, data = paper.ref, paper.data
    sv = ref.svd
    maxrank = min(ref.n, ref.T)
    calls: list[tuple[list[int], float]] = []

    def objective(fit: SVDFit, ranks: NDArray[np.int64]) -> float:
        assert fit.intercept_full is not None
        value = mc.svd_aic_reference(
            data.Y, data.X, data.mask, [*fit.B, fit.intercept_full], [*ranks, maxrank]
        )
        calls.append(([*map(int, ranks), maxrank], value))
        return value

    best, history = greedy_aic(
        functools.partial(fit_svd, paper.stats), objective, [1] * 6, maxrank
    )
    np.testing.assert_array_equal(history.ranks, sv["rhist"][:, :-1])
    assert [c[0] for c in calls] == [list(r) for r in sv["calls_ranks"]]
    errors = {
        "AIC per call": pf.rel(np.array([c[1] for c in calls]), sv["calls_values"]),
        "FunHist": pf.rel(history.aic, sv["funhist"]),
    }
    print(f"\n[paper SVD search] {history.ranks[-1].tolist()} in {len(calls)} calls")
    pf.report("paper SVD search", errors, SVD_AIC_TOL)
    assert list(best.ranks) == sv["rest"][:-1]


def test_svd_intercept_and_corrected_score_from_the_raw_data(paper: _Paper) -> None:
    # The port's corrected SVD path ((M19b), (M21a)-(M21b)) at the paper's
    # scale and session masks, against the raw responses rather than the
    # sufficient statistics: the refitted intercept zeroes each neuron's mean
    # residual over its observed trials, and the precisions, log-likelihood and
    # AIC follow from those residuals. The reference objective uses the joint
    # least-squares intercept instead, so the search test cannot see this one.
    data = paper.data
    ranks = paper.ref.svd["rest"][:-1]
    fit = fit_svd(paper.stats, ranks)
    assert fit.intercept is not None
    n, T = paper.ref.n, paper.ref.T
    B = np.stack(fit.B, axis=1)  # (n, P, T)
    rss = np.empty(n)
    mean_residual = np.empty((n, T))
    for i in range(n):
        kk = np.flatnonzero(data.mask[:, i])
        resid = data.Y[kk, i, :] - fit.intercept[i] - data.X[kk] @ B[i]
        rss[i] = float(np.sum(resid**2))
        mean_residual[i] = resid.mean(axis=0)
    counts = data.mask.sum(axis=0) * T
    loglik = float(
        -0.5 * np.sum(counts * (np.log(rss / counts) + 1 + np.log(2 * np.pi)))
    )
    # (M21b), textbook: the rank-r_p blocks, the precisions, the full intercept.
    n_pars = sum(r * (n + T - r) for r in ranks) + n + n * T
    errors = {
        "mean residual / intercept": float(
            np.abs(mean_residual).max() / np.abs(fit.intercept).max()
        ),
        "noise precision": pf.rel(fit.noise_precision, counts / rss),
        "log-likelihood": pf.rel(fit.log_likelihood, loglik),
        "AIC": pf.rel(fit.aic, 2 * n_pars - 2 * loglik),
    }
    pf.report(f"paper SVD at {ranks}, from the raw data", errors, 1e-10)
    assert fit.n_parameters == n_pars
    # At ranks below full the conditional intercept is not the joint one.
    assert fit.intercept_full is not None
    assert pf.rel(fit.intercept, fit.intercept_full) > 1e-3


@pytest.mark.parametrize("k", FITS)
def test_likelihood_and_aic_at_the_reference_points(paper: _Paper, k: int) -> None:
    # The marginal likelihood at P = 6, r_tot up to 21 and the session masks,
    # in the default suite (about 0.1 s per point): at MATLAB's post-ECME and
    # final estimates against its own NLL, and the AIC under the reference
    # count against BTDR_AIC_S_lamb_b_wrapper's.
    ft = paper.ref.fits[k]
    st = paper.stats
    e, fin = ft["ecme"], ft["final"]
    at_ecme = mmle.MMLEFit.from_parameters(st, e.S, e.lam, e.b)
    at_final = mmle.MMLEFit.from_parameters(st, fin.S, fin.lam, fin.b)
    errors = {
        "NLL at the post-ECME point": pf.rel(
            pf.nll_without_constant(at_ecme, st), ft["ecme_nll"][-1]
        ),
        "NLL at the final point": pf.rel(
            pf.nll_without_constant(at_final, st), ft["nll_final"]
        ),
        "AIC (reference count)": pf.rel(
            pf.reference_aic(at_final, st, ft["ranks"]), ft["aic"]
        ),
    }
    pf.report(f"paper r={ft['ranks']} at the reference points", errors, FIXED_POINT_TOL)
    # MATLAB's AIC decodes to its NLL with the literal count n + T sum(r) + nT.
    count = 0.5 * (ft["aic"] - 2 * ft["nll_final"])
    assert abs(count - ft["n_pars"]) < 1e-6 * ft["n_pars"]


def test_parameter_counts_at_the_true_ranks(paper: _Paper) -> None:
    # (M38) n + T sum(r) + nT and (M38a) n + sum(T r - r(r-1)/2) + nT, written
    # out for the paper's ranks (5, 4, 3, 4, 2, 2), n = 800, T = 15.
    ref = paper.ref
    assert ref.true_ranks == [5, 4, 3, 4, 2, 2]
    assert aic.n_parameters_mmle(ref.true_ranks, ref.n, ref.T, formula="reference") == (
        13_100
    )
    assert aic.n_parameters_mmle(ref.true_ranks, ref.n, ref.T) == 13_073
    assert ref.fits[0]["n_pars"] == 13_100


def test_posterior_at_the_reference_estimate(paper: _Paper) -> None:
    # MATLAB's EBpost_W_uneqvar at the true-rank final estimate: the
    # posterior mean and B_p = W_p S_p' from it, against the port's at the
    # same parameters.
    ref, st = paper.ref, paper.stats
    ft = ref.fits[0]
    assert ref.posterior["ranks"] == ft["ranks"]
    fin = ft["final"]
    fit = mmle.MMLEFit.from_parameters(st, fin.S, fin.lam, fin.b)
    W_ref = ref.posterior["W"]
    B_ref = [w @ s.T for w, s in zip(W_ref, fin.S, strict=True)]
    errors = {
        "W vs Wt": pf.rel_blocks(fit.W, W_ref),
        "B vs Wt' S'": pf.rel_blocks(fit.B, B_ref),
    }
    pf.report("paper posterior at the reference estimate", errors, POSTERIOR_TOL)


def test_reference_inner_options(paper: _Paper) -> None:
    # The reference's minFunc options against the package's: the parity tests'
    # basis-step cap is the reference's maxIter (500) while the package default
    # is 2000; optimizer_tol's default sits below its optTol (the gradient test
    # is the stricter); optimizer_progtol is its progTol, the absolute function
    # change of the basis step; the precision step keeps a relative ftol.
    opts = {k: float(v) for k, v in paper.ref.minfunc_options.items()}
    assert opts == REFERENCE_INNER
    assert paper.ref.matlab_threads >= 1
    assert {"optimizer_max_iter": opts["maxIter"]} == pf.REFERENCE_CAPS
    for func in (mmle.refine, mmle.fit_mmle):
        defaults = {
            name: p.default for name, p in inspect.signature(func).parameters.items()
        }
        assert defaults["optimizer_max_iter"] == 2000
        assert defaults["optimizer_tol"] <= opts["optTol"]
        assert defaults["optimizer_progtol"] == opts["progTol"]
    assert mmle._LBFGS_MEMORY == 100  # minFunc's Corr
    assert mmle._LBFGS_FTOL == 1e-11  # the precision step's relative ftol


def _final(
    paper: _Paper, k: int, label: str, fit: mmle.MMLEFit, same_start: bool
) -> dict[str, float]:
    ft = paper.ref.fits[k]
    nll = pf.nll_without_constant(fit, paper.stats)
    rise = (nll - ft["nll_final"]) / abs(ft["nll_final"])
    gaps = pf.final_gaps(fit, ft["final"], paper.stats)
    print(
        f"\n[paper r={ft['ranks']} {label}] NLL {nll:.8f} vs reference "
        f"{ft['nll_final']:.8f}, (port - reference)/|NLL| = {rise:.2e}; gaps "
        + ", ".join(f"{key} {v:.1e}" for key, v in gaps.items())
        + f"; n_iter {dict(fit.n_iter)}, converged {fit.converged}"
    )
    if same_start:
        assert abs(rise) <= SAME_START_NLL_RTOL
        assert all(v < SAME_START_GAP for v in gaps.values())
    else:
        assert rise <= OTHER_PATH_NLL_RTOL
        assert all(v < OTHER_PATH_GAP for v in gaps.values())
    return gaps


def _capped(caught: pf.Caught, fit: mmle.MMLEFit, ft: dict[str, Any]) -> None:
    # At this scale the first basis step ends on the reference's cap of 500 in
    # every fit (minFunc's first iteration count in the fixture), at the same
    # iteration count; refine reports it and converged stays False.
    print(f"  warnings: {caught.reasons}")
    assert "basis_cap" in caught.reasons
    cap = int(ft["ca"]["minfunc_iterations"][0])
    assert cap == REFERENCE_INNER["maxIter"]
    first = (
        "iteration 1, bases: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT "
        f"(L-BFGS-B status 1, nit {cap},"
    )
    assert any(first.lower() in m.lower() for m in caught.messages), caught.messages
    assert not fit.converged


@pytest.mark.slow
@pytest.mark.parametrize("k", FITS)
def test_start_is_the_reference_start(paper: _Paper, k: int) -> None:
    ft = paper.ref.fits[k]
    data = paper.data
    start = pf.reference_start(paper.stats, data.Y, data.X, data.mask, ft["ranks"])
    signs = [
        np.sign(np.sum(a * c, axis=0))
        for a, c in zip(start.S, ft["start_S"], strict=True)
    ]
    errors = {
        "S (signs aligned)": pf.rel_blocks(
            [a * s for a, s in zip(start.S, signs, strict=True)], ft["start_S"]
        ),
        "lambda, (M19a) replica": pf.rel(start.noise_precision, ft["start_lam"]),
    }
    pf.report(f"paper r={ft['ranks']} start", errors, TOL)


@pytest.mark.slow
@pytest.mark.parametrize("k", FITS)
def test_refine_from_the_reference_ecme(paper: _Paper, k: int) -> None:
    ft = paper.ref.fits[k]
    st = paper.stats
    e = ft["ecme"]
    start = mmle.MMLEFit.from_parameters(st, e.S, e.lam, e.b)
    fit, caught = pf.run(lambda: mmle.refine(st, start, **pf.REFERENCE_CAPS), PARITY)
    _final(paper, k, "refine from the reference's ECME, reference caps", fit, True)
    _capped(caught, fit, ft)
    minfunc = ft["ca"]["minfunc_iterations"].astype(int).tolist()
    print(
        f"  iterations: port {fit.n_iter['refine']}, reference {ft['ca']['nll'].size}; "
        f"reference minFunc iterations {minfunc}"
    )
    # Measured 3/10/5 against the reference's 3/10/4 (Windows): the over-ranked
    # fit takes one more outer iteration, from refine's change test (relative,
    # with a floor) against the reference's pure ratio.
    assert abs(fit.n_iter["refine"] - ft["ca"]["nll"].size) <= 1


@pytest.mark.slow
@pytest.mark.parametrize("k", FITS)
def test_compat_pipeline_from_the_reference_start(paper: _Paper, k: int) -> None:
    ft = paper.ref.fits[k]
    data = paper.data
    st = paper.stats
    start = pf.reference_start(st, data.Y, data.X, data.mask, ft["ranks"])
    warm, _ = mmle.ecme(st, start, matlab_compat=True)
    assert warm.n_iter["ecme"] == ft["ecme_n_sweeps"]
    assert warm.intercept is not None
    e = ft["ecme"]
    signs = [np.sign(np.sum(a * c, axis=0)) for a, c in zip(warm.S, e.S, strict=True)]
    errors = {
        "S (signs aligned)": pf.rel_blocks(
            [a * s for a, s in zip(warm.S, signs, strict=True)], e.S
        ),
        "lambda": pf.rel(warm.noise_precision, e.lam),
        "b": pf.rel(warm.intercept, e.b),
    }
    # Measured at most 2.6e-13.
    pf.report(f"paper r={ft['ranks']} compat ECME vs ECMEtdr", errors, COMPAT_TOL)
    fit, caught = pf.run(lambda: mmle.refine(st, warm, **pf.REFERENCE_CAPS), PARITY)
    _final(paper, k, "compat ECME + refine, reference caps", fit, True)
    _capped(caught, fit, ft)


@pytest.mark.slow
@pytest.mark.parametrize("k", FITS)
def test_fit_mmle_from_its_own_start(paper: _Paper, k: int) -> None:
    # The package's defaults (optimizer_progtol 1e-9, cap 2000), one-sided at
    # 1e-7: the shipped path stays covered. At the cap of 2000 the first basis
    # step does not end on its cap (it needs 582-909 iterations), so only the
    # under-ranked fit's own refine cap or a rounding-level end may remain.
    ft = paper.ref.fits[k]
    fit, caught = pf.run(
        lambda: mmle.fit_mmle(paper.stats, ft["ranks"]), DEFAULT_REASONS
    )
    _final(paper, k, "fit_mmle, defaults", fit, False)
    print(f"  warnings: {caught.reasons}")
    assert "basis_cap" not in caught.reasons
    # The true and over-ranked fits converge (the over-ranked one can end on a
    # rounding-level basis step, which counts as converged); the under-ranked
    # fit stops on refine's cap, as the reference's does.
    if [int(r) for r in ft["ranks"]] == [4, 4, 3, 4, 2, 2]:
        assert caught.reasons == ["refine_cap"]
    else:
        assert caught.reasons == []
        assert fit.converged
    for message in caught.messages:
        print(f"  ConvergenceWarning: {message[:160]}")


@pytest.mark.slow
def test_tightly_refined_estimates_agree(paper: _Paper) -> None:
    # The 1e-4 target on estimates, between two port refinements seeded by
    # MATLAB's final estimate and by fit_mmle's (default) fit. This needs the
    # basis step's absolute progtol: under a relative ftol of 1e-11 instead, the
    # refinement from fit_mmle's fit is still 3.4e-3 nats short after 100
    # iterations.
    ft = paper.ref.fits[0]
    st = paper.stats
    fin = ft["final"]
    port, _ = pf.run(lambda: mmle.fit_mmle(st, ft["ranks"]), DEFAULT_REASONS)
    from_ref, caught_ref = pf.run(
        lambda: mmle.refine(
            st, mmle.MMLEFit.from_parameters(st, fin.S, fin.lam, fin.b), **TIGHT
        ),
        PARITY,
    )
    from_port, caught_port = pf.run(lambda: mmle.refine(st, port, **TIGHT), PARITY)
    assert from_ref.intercept is not None
    assert from_port.intercept is not None
    gaps = {
        "B": pf.rel_blocks(from_port.B, from_ref.B),
        "lambda": pf.rel(from_port.noise_precision, from_ref.noise_precision),
        "b": pf.rel(from_port.intercept, from_ref.intercept),
    }
    ref_nll = ft["nll_final"] + mmle._constant(st.n_obs, st.n_bins)
    nll = -from_ref.log_likelihood
    rise_ref = (nll - ref_nll) / abs(ref_nll)
    rise_port = (-from_port.log_likelihood - ref_nll) / abs(ref_nll)
    print(
        f"\n[paper tight, reference caps] (endpoint - reference final)/|NLL|: from the "
        f"reference {rise_ref:.1e}, from fit_mmle {rise_port:.1e}; n_iter "
        f"{dict(from_ref.n_iter)} / {dict(from_port.n_iter)}; warnings "
        f"{caught_ref.reasons} / {caught_port.reasons}"
    )
    pf.report("paper tight", gaps, TIGHT_GAP)
    # Both at least as good as the reference's own final estimate, to the
    # package's rounding allowance, as at the demo scale.
    assert max(rise_ref, rise_port) <= mmle.MONOTONE_RTOL


@pytest.mark.slow
def test_mmle_search_from_the_true_ranks(paper: _Paper) -> None:
    # EstRankGreedily with the MMLE objective from the true ranks, as the
    # reference runs it (reference count (M38)); replayed by greedy_aic with the
    # reference-compatible fit (parity_fixtures.reference_fit) at the reference caps.
    # The rank path exactly, every candidate's NLL within the same-start gate.
    sr = paper.ref.mmle
    assert sr, "paper.mat was generated without the MMLE search"
    data, st = paper.data, paper.stats
    n, T = paper.ref.n, paper.ref.T
    calls: list[tuple[list[int], float]] = []

    def fit(r: NDArray[np.int64]) -> mmle.MMLEFit:
        out, _ = pf.run(
            lambda: pf.reference_fit(st, data.Y, data.X, data.mask, r), PARITY
        )
        return out

    def objective(f: mmle.MMLEFit, r: NDArray[np.int64]) -> float:
        calls.append(([int(x) for x in r], pf.nll_without_constant(f, st)))
        return pf.reference_aic(f, st, r)

    _, history = greedy_aic(fit, objective, sr["rest0"], min(n, T))
    np.testing.assert_array_equal(history.ranks, sr["rhist"])
    assert [c[0] for c in calls] == [list(r) for r in sr["calls_ranks"]]
    ref = np.array(
        [
            0.5 * (v - 2 * (n + T * int(sum(r)) + n * T))
            for r, v in zip(sr["calls_ranks"], sr["calls_values"], strict=True)
        ]
    )
    rise = (np.array([c[1] for c in calls]) - ref) / np.abs(ref)
    print(
        f"\n[paper MMLE search, reference caps] {history.ranks.tolist()} ({len(calls)} "
        f"calls; reference {sr['seconds']:.0f} s); (port - reference)/|NLL| per call "
        f"in [{rise.min():.1e}, {rise.max():.1e}]"
    )
    assert np.all(np.abs(rise) <= SAME_START_NLL_RTOL)
    # The reference's own AIC trace, decoded with the literal count above,
    # against the port's AIC under the package's reference count.
    assert pf.rel(history.aic, sr["funhist"]) < SAME_START_NLL_RTOL
