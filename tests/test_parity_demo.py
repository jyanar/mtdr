"""MATLAB parity on the two shipped demos, end to end.

`tests/fixtures/demo.mat` (generator `tests/fixtures/matlab/make_demo_fixture.m`)
runs `mTDRdemo.m` and `demoLearning.m` unmodified at `rng(0)` on a scratch copy
of the reference. Both draw the same data ($n=100$, $T=15$, $N=100$, true ranks
$r_P=(5,6,1)$, the demo's precisions and 30 % trial loss); `mTDRdemo.m`'s SVD
search (on the reference's defective `SVDRegB_AIC`, see
`docs/differences-from-matlab.md`) moves six steps to $(3,5,1)$ and its MMLE
search three more to $(5,6,1)$; `demoLearning.m` fits $(5,6,1)$, saved with its
ranks (the shipped `EstimatedPars/LearnDemoMMLE.mat` stores none, and is not
reproduced at `rng(0)`). The fixture also holds an instrumented replay of the
estimation (asserted identical to the scripts' own histories) with every
rank-search objective evaluation and the fit's intermediates.

What is compared, and at what tolerance:

* the data and the reference statistics, 1e-10 relative;
* the SVD search, run by `greedy_aic` on `fit_svd` with the reference objective
  replicated in `tests/matlab_compat.py`: the rank path exactly, every
  objective value to 1e-6 relative and every accepted fit to 1e-8;
* the MMLE search under the reference-compatible options (the reference's
  start, `matlab_compat` ECME, the reference count (M38)) at the reference caps
  (`parity_fixtures.REFERENCE_CAPS`): the rank path exactly, and every
  candidate's marginal NLL within 1e-8 relative either way, looser than the
  single fits' 1e-11 because refine's change test (relative, with a floor,
  where the reference's is a pure ratio) ends two candidates one iteration
  before the reference's does;
* the MMLE search with the package's own estimator at its defaults
  (`fit_mmle`, corrected ECME, its own start) and the reference count: the
  rank path exactly, every candidate's NLL at least as good as the
  reference's to 1e-7;
* the demoLearning fit: the start (1e-8), at the reference caps `refine` and the
  compat path from the reference's points (two-sided 1e-11), at the defaults
  `fit_mmle` (one-sided 1e-7), the estimates between tightly refined
  endpoints (1e-4; both endpoints are port refinements, seeded by MATLAB's and
  the port's fits, so this checks where the port's optimiser ends, not the
  objective), the posterior, `B` and the AIC at the reference's estimate
  (1e-8);
* the shipped `EstimatedPars/*.mat`, where `MTDR_REFERENCE_DIR` points at the
  reference: the internal consistency of `LearnDemoMMLE.mat` and
  `RankEstDemoMMLE.mat`.

Every `ConvergenceWarning` is recorded and its reasons are checked against the
classes the test expects (`parity_fixtures.run`). `pytest -s` prints every
measured value.
"""

from __future__ import annotations

import functools
import itertools
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.io import loadmat

import matlab_compat as mc
import parity_fixtures as pf
from mtdr import mmle
from mtdr.rank_search import RankSearchHistory, greedy_aic
from mtdr.svd_fit import SVDFit, fit_svd
from nversion import mmle_fixture as mf

REFERENCE_DIR = os.environ.get("MTDR_REFERENCE_DIR")
STATS_TOL = 1e-10
SVD_AIC_TOL = 1e-6
TOL = 1e-8
COMPAT_TOL = 1e-11
SAME_START_NLL_RTOL = 1e-11  # at the reference caps
# The reference-compatible search is gated at 1e-8: with tight inner stopping
# (L-BFGS-B `ftol = 1e-15`) 10 of its 13 calls agree to 1.6e-11, but on
# (3,6,2) and (4,6,2) refine's change test (relative, with a floor) stops one
# iteration before the reference's pure ratio does.
SEARCH_NLL_RTOL = 1e-8
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
# Reasons a refinement may warn with here: refine's own cap on under-ranked
# candidates and a rounding-level precision line search.
EXPECTED = ("refine_cap", "precision_rounding")
# The parity paths run the package defaults (minFunc's progTol) with the
# reference's basis cap; a basis step there may also end on a rounding-level
# line search, which counts as converged, so they expect the same reasons.
PARITY = EXPECTED


@pytest.fixture(scope="module")
def demo() -> pf.DemoFixture:
    # A required parity fixture: its absence is a failure, not a skip.
    assert pf.DEMO_FIXTURE.is_file(), "tests/fixtures/demo.mat is missing"
    return pf.load_demo()


def _nll_from_aic(value: float, ranks: Any, n: int, T: int) -> float:
    # BTDR_AIC_S_lamb_b_wrapper's count written out, n + T sum(r) + n T, not
    # the package's n_parameters_mmle(formula="reference"): a wrong package
    # count would otherwise shift the decoded reference NLL with it and pass
    # the one-sided check.
    return 0.5 * (value - 2 * (n + T * int(sum(ranks)) + n * T))


# ------------------------------------------------------------------ data


def test_data_statistics_and_truth(demo: pf.DemoFixture) -> None:
    st, ms = demo.stats, demo.matlab_stats
    errors = {
        "XtX": pf.rel(st.XtX, ms["A"]),
        "XtY_raw": pf.rel(st.XtY_raw, ms["xi"]),
        "YtY_raw": pf.rel(st.YtY_raw, ms["zzi"]),
        "X_mean": pf.rel(st.X_mean, ms["xbari"]),
        "Y_mean": pf.rel(st.Y_mean, ms["Ybar"]),
    }
    pf.report("demo statistics", errors, STATS_TOL)
    np.testing.assert_array_equal(st.n_obs, ms["ni"])
    assert (demo.n, demo.T, demo.N, demo.seed) == (100, 15, 100, 0)
    assert demo.true_ranks == demo.learn["ranks"] == [5, 6, 1]
    # SimWeights: BB stacks W_p S_p' for p = 1..P, the constant term last.
    assert [w.shape[1] for w in demo.W_true] == [*demo.true_ranks, demo.T]
    stacked = np.vstack(
        [w @ s.T for w, s in zip(demo.W_true, demo.S_true, strict=True)]
    )
    assert pf.rel(stacked, demo.BB) < 1e-14
    assert np.all(demo.precision > 0)


# ------------------------------------------------------------------ SVD search


def _replay(
    history: RankSearchHistory,
    calls: list[tuple[list[int], float]],
    ref_rhist: NDArray[np.int64],
    ref_funhist: NDArray[np.float64],
    ref_calls_ranks: NDArray[np.int64],
) -> None:
    np.testing.assert_array_equal(history.ranks, ref_rhist)
    assert [c[0] for c in calls] == [list(r) for r in ref_calls_ranks]
    assert history.aic.shape == ref_funhist.shape


def test_svd_search_trajectory(demo: pf.DemoFixture) -> None:
    # mTDRdemo.m's first search: EstRankGreedily(SVDRegB_AIC, SVDRegressB, ones,
    # maxrank), the constant term at maxrank. greedy_aic on fit_svd with the
    # replicated objective (the port's corrected objective is tested in
    # test_svd_parity; on these data it is not the reference's).
    sv = demo.svd
    maxrank = demo.maxrank
    calls: list[tuple[list[int], float]] = []

    def objective(fit: SVDFit, ranks: NDArray[np.int64]) -> float:
        assert fit.intercept_full is not None
        value = mc.svd_aic_reference(
            demo.Y, demo.X, demo.mask, [*fit.B, fit.intercept_full], [*ranks, maxrank]
        )
        calls.append(([*map(int, ranks), maxrank], value))
        return value

    best, history = greedy_aic(
        functools.partial(fit_svd, demo.stats), objective, [1, 1, 1], maxrank
    )
    rhist = sv["rhist"]
    assert np.all(rhist[:, -1] == maxrank)
    _replay(history, calls, rhist[:, :-1], sv["funhist"], sv["calls_ranks"])
    values = np.array([c[1] for c in calls])
    errors = {
        "AIC per call": pf.rel(values, sv["calls_values"]),
        "FunHist": pf.rel(history.aic, sv["funhist"]),
    }
    # Every accepted fit against parhist (the constant block is the joint
    # least-squares intercept, SVDFit.intercept_full).
    worst = 0.0
    for ranks, ref in zip(history.ranks[1:], sv["parhist"], strict=True):
        fit = fit_svd(demo.stats, ranks)
        assert fit.intercept_full is not None
        worst = max(
            worst, pf.rel(np.stack([*fit.B, fit.intercept_full]), np.stack(ref))
        )
    errors["parhist (B)"] = worst
    print(
        f"\n[demo SVD search] {history.ranks.tolist()} ({len(calls)} calls), "
        f"stop: {history.stop_reason}"
    )
    pf.report("demo SVD search", errors, SVD_AIC_TOL)
    assert worst < TOL
    assert list(best.ranks) == sv["rest"][:-1] == [3, 5, 1]
    assert history.stop_reason == "no_improvement"
    # The decisions are far from ties: no round's best candidate lies within
    # 1e3 times the measured error of the acceptance boundary.
    for k, round_ in enumerate(history.candidates):
        gap = abs(min(round_.values()) - history.aic[k]) / abs(history.aic[k])
        assert gap > 1e3 * max(errors["AIC per call"], 1e-15)


# ------------------------------------------------------------------ MMLE search


class _Search:
    """One MMLE search's record: the history, every call and every fit."""

    def __init__(
        self,
        demo: pf.DemoFixture,
        fit_fn: Callable[[NDArray[np.int64]], mmle.MMLEFit],
        allowed: tuple[str, ...] = EXPECTED,
    ) -> None:
        self.calls: list[tuple[list[int], float]] = []
        self.fits: dict[tuple[int, ...], mmle.MMLEFit] = {}
        self.warned: list[str] = []
        st = demo.stats

        def fit(r: NDArray[np.int64]) -> mmle.MMLEFit:
            out, caught = pf.run(lambda: fit_fn(r), allowed)
            self.warned.extend(caught.messages)
            self.fits[tuple(int(x) for x in r)] = out
            return out

        def objective(f: mmle.MMLEFit, r: NDArray[np.int64]) -> float:
            self.calls.append(([int(x) for x in r], pf.nll_without_constant(f, st)))
            return pf.reference_aic(f, st, r)

        self.best, self.history = greedy_aic(
            fit, objective, demo.mmle["rest0"], demo.maxrank
        )


def _compare_calls(
    demo: pf.DemoFixture, calls: list[tuple[list[int], float]]
) -> NDArray[np.float64]:
    mm = demo.mmle
    ref = np.array(
        [
            _nll_from_aic(v, r, demo.n, demo.T)
            for r, v in zip(mm["calls_ranks"], mm["calls_values"], strict=True)
        ]
    )
    port = np.array([c[1] for c in calls])
    return np.asarray((port - ref) / np.abs(ref))


def test_mmle_search_trajectory_reference_compatible(demo: pf.DemoFixture) -> None:
    # mTDRdemo.m's second search, from the SVD search's ranks: each candidate is
    # fitted from the reference's start by the compat ECME and refine (at the
    # reference caps), and scored with the reference's count, as
    # BTDR_AIC_S_lamb_b_wrapper does.
    mm = demo.mmle
    run = _Search(
        demo,
        lambda r: pf.reference_fit(demo.stats, demo.Y, demo.X, demo.mask, r),
        PARITY,
    )
    history = run.history
    _replay(history, run.calls, mm["rhist"], mm["funhist"], mm["calls_ranks"])
    rise = _compare_calls(demo, run.calls)
    gaps = [
        pf.final_gaps(run.fits[tuple(int(x) for x in ranks)], ref, demo.stats)
        for ranks, ref in zip(history.ranks[1:], mm["parhist"], strict=True)
    ]
    worst_gap = {key: max(g[key] for g in gaps) for key in ("B", "lambda", "b")}
    print(
        f"\n[demo MMLE search, reference-compatible, reference caps] "
        f"{history.ranks.tolist()} "
        f"({len(run.calls)} calls, {len(run.warned)} ConvergenceWarnings); "
        f"(port - reference)/|NLL| per call in [{rise.min():.1e}, {rise.max():.1e}]; "
        "accepted-fit gaps " + ", ".join(f"{k} {v:.1e}" for k, v in worst_gap.items())
    )
    assert np.all(np.abs(rise) <= SEARCH_NLL_RTOL)
    assert all(v < SAME_START_GAP for v in worst_gap.values())
    assert list(run.best.ranks) == mm["rest"] == demo.true_ranks
    assert history.stop_reason == "no_improvement"
    assert pf.rel(history.aic, mm["funhist"]) < SEARCH_NLL_RTOL


def test_mmle_search_trajectory_package_estimator(demo: pf.DemoFixture) -> None:
    # The same search with the package's own estimator at its defaults
    # (fit_mmle: the (M19b) start, the corrected ECME, refine) and the
    # reference count.
    mm = demo.mmle
    run = _Search(demo, functools.partial(mmle.fit_mmle, demo.stats), PARITY)
    _replay(run.history, run.calls, mm["rhist"], mm["funhist"], mm["calls_ranks"])
    rise = _compare_calls(demo, run.calls)
    print(
        f"\n[demo MMLE search, fit_mmle, defaults] {run.history.ranks.tolist()} "
        f"({len(run.calls)} calls, {len(run.warned)} ConvergenceWarnings); "
        f"(port - reference)/|NLL| per call in [{rise.min():.1e}, {rise.max():.1e}]"
    )
    assert np.all(rise <= OTHER_PATH_NLL_RTOL)
    assert list(run.best.ranks) == demo.true_ranks


# ------------------------------------------------------------------ demoLearning.m


def _final_parity(
    demo: pf.DemoFixture, label: str, fit: mmle.MMLEFit, same_start: bool
) -> None:
    le = demo.learn
    nll = pf.nll_without_constant(fit, demo.stats)
    rise = (nll - le["nll_final"]) / abs(le["nll_final"])
    gaps = pf.final_gaps(fit, le["final"], demo.stats)
    print(
        f"\n[demoLearning {label}] NLL {nll:.10f} vs reference {le['nll_final']:.10f}, "
        f"(port - reference)/|NLL| = {rise:.2e}; gaps "
        + ", ".join(f"{k} {v:.1e}" for k, v in gaps.items())
    )
    if same_start:
        assert abs(rise) <= SAME_START_NLL_RTOL
        assert all(v < SAME_START_GAP for v in gaps.values())
    else:
        assert rise <= OTHER_PATH_NLL_RTOL
        assert all(v < OTHER_PATH_GAP for v in gaps.values())


def test_learn_start_is_the_reference_start(demo: pf.DemoFixture) -> None:
    le = demo.learn
    start = pf.reference_start(demo.stats, demo.Y, demo.X, demo.mask, le["ranks"])
    ref = le["start"]
    signs = [
        np.sign(np.sum(a * c, axis=0)) for a, c in zip(start.S, ref.S, strict=True)
    ]
    assert start.intercept is not None
    errors = {
        "S (signs aligned)": pf.rel_blocks(
            [a * s for a, s in zip(start.S, signs, strict=True)], ref.S
        ),
        "lambda, (M19a) replica": pf.rel(start.noise_precision, ref.lam),
        "b0 = Ybar": pf.rel(start.intercept, ref.b),
    }
    pf.report("demoLearning start", errors, TOL)


def test_learn_refine_from_the_reference_ecme(demo: pf.DemoFixture) -> None:
    le = demo.learn
    st = demo.stats
    start = mmle.MMLEFit.from_parameters(st, le["ecme"].S, le["ecme"].lam, le["ecme"].b)
    fit, caught = pf.run(lambda: mmle.refine(st, start, **pf.REFERENCE_CAPS), PARITY)
    _final_parity(demo, "refine from the reference's ECME, reference caps", fit, True)
    # The reference stopped on its (M34) change; so does the port here, unless
    # a platform ends an inner line search at rounding level.
    print(f"  converged {fit.converged}, n_iter {dict(fit.n_iter)}, {caught.reasons}")
    assert "refine_cap" not in caught.reasons


def test_learn_compat_pipeline_from_the_reference_start(demo: pf.DemoFixture) -> None:
    le = demo.learn
    st = demo.stats
    start = mmle.MMLEFit.from_parameters(
        st, le["start"].S, le["start"].lam, le["start"].b
    )
    warm, _ = mmle.ecme(st, start, matlab_compat=True)
    assert warm.n_iter["ecme"] == le["ecme_n_sweeps"]
    assert warm.intercept is not None
    errors = {
        "S": pf.rel_blocks(warm.S, le["ecme"].S),
        "lambda": pf.rel(warm.noise_precision, le["ecme"].lam),
        "b": pf.rel(warm.intercept, le["ecme"].b),
    }
    # Measured 1.2e-13.
    pf.report("demoLearning compat ECME vs ECMEtdr", errors, COMPAT_TOL)
    fit, _ = pf.run(lambda: mmle.refine(st, warm, **pf.REFERENCE_CAPS), PARITY)
    # And the helper the search tests use, from fit_svd's start (signs differ).
    fit2, _ = pf.run(
        lambda: pf.reference_fit(st, demo.Y, demo.X, demo.mask, le["ranks"]),
        PARITY,
    )
    _final_parity(demo, "compat ECME + refine, reference caps", fit, True)
    _final_parity(demo, "reference_fit, reference caps", fit2, True)


def test_learn_fit_mmle_from_its_own_start(demo: pf.DemoFixture) -> None:
    # The package's defaults, one-sided at 1e-7, so the shipped defaults stay
    # covered.
    fit, _ = pf.run(lambda: mmle.fit_mmle(demo.stats, demo.learn["ranks"]), PARITY)
    _final_parity(demo, "fit_mmle, defaults", fit, False)


def test_learn_tightly_refined_estimates_agree(demo: pf.DemoFixture) -> None:
    # The 1e-4 target on estimates, between tightly refined endpoints, as at
    # the paper scale. Both endpoints are port refinements: this checks where
    # the optimiser ends; the objective is pinned at the reference estimate by
    # the test below.
    le = demo.learn
    st = demo.stats
    fin = le["final"]
    port, _ = pf.run(lambda: mmle.fit_mmle(st, le["ranks"]), PARITY)
    from_ref, _ = pf.run(
        lambda: mmle.refine(
            st, mmle.MMLEFit.from_parameters(st, fin.S, fin.lam, fin.b), **TIGHT
        ),
        PARITY,
    )
    from_port, _ = pf.run(lambda: mmle.refine(st, port, **TIGHT), PARITY)
    assert from_ref.intercept is not None
    assert from_port.intercept is not None
    gaps = {
        "B": pf.rel_blocks(from_port.B, from_ref.B),
        "lambda": pf.rel(from_port.noise_precision, from_ref.noise_precision),
        "b": pf.rel(from_port.intercept, from_ref.intercept),
    }
    nll = -from_ref.log_likelihood
    rise = (-from_port.log_likelihood - nll) / abs(nll)
    print(f"\n[demoLearning tight] (port - reference)/|NLL| = {rise:.1e}")
    pf.report("demoLearning tight", gaps, TIGHT_GAP)
    # Both at least as good as the reference's own final estimate, to the
    # package's rounding allowance MONOTONE_RTOL (a tight refinement that ends
    # on rounding was measured 2.5e-12 above it on one Ubuntu runner; the same
    # allowance at both scales).
    ref_nll = le["nll_final"] + mmle._constant(st.n_obs, st.n_bins)
    assert max(nll, -from_port.log_likelihood) <= ref_nll * (1 + mmle.MONOTONE_RTOL)


def test_learn_posterior_b_hat_and_aic_at_the_reference_estimate(
    demo: pf.DemoFixture,
) -> None:
    # demoLearning.m's Bhat: MakeBhat_data with the centred statistic.
    le = demo.learn
    st = demo.stats
    fin = le["final"]
    fit = mmle.MMLEFit.from_parameters(st, fin.S, fin.lam, fin.b)
    errors = {
        "W vs Wt": pf.rel_blocks(fit.W, le["W"]),
        "B vs Bhat": pf.rel_blocks(fit.B, le["Bhat"]),
        "S vs Shat'": pf.rel_blocks(fit.S, [s.T for s in le["S_matlab"]]),
        "NLL": pf.rel(pf.nll_without_constant(fit, st), le["nll_final"]),
        "AIC (reference count)": pf.rel(
            pf.reference_aic(fit, st, le["ranks"]), le["aic"]
        ),
    }
    pf.report("demoLearning at the reference estimate", errors, TOL)


def test_learn_fit_records_its_ranks(demo: pf.DemoFixture) -> None:
    # The shipped LearnDemoMMLE.mat stores parhist without the ranks. The
    # fixture regenerates demoLearning.m's fit with rP saved. Its ranks sum to
    # 12, as the shipped vector's length implies, but the shipped file is not
    # this run's output: the generator recorded differences in the precisions
    # and the intercept far above rounding.
    le = demo.learn
    assert le["ranks"] == demo.true_ranks
    assert le["pars_final"].size == demo.n + sum(le["ranks"]) * demo.T + demo.n * demo.T
    diff = le["shipped_diff"]
    print(f"\n[demoLearning vs shipped LearnDemoMMLE.mat] {diff}")
    assert diff["diff_lambda"] > 1.0
    assert diff["diff_b"] > 1.0


# ------------------------------------------------------- the shipped EstimatedPars


def _shipped(name: str) -> dict[str, Any]:
    if not REFERENCE_DIR:
        pytest.skip(
            "set MTDR_REFERENCE_DIR to the mTDRdemo directory to check the shipped "
            f"{name}"
        )
    path = Path(REFERENCE_DIR) / "EstimatedPars" / name
    if not path.is_file():
        pytest.skip(f"{path} not found (MTDR_REFERENCE_DIR={REFERENCE_DIR})")
    return loadmat(path, simplify_cells=True)


def test_shipped_learn_demo_is_internally_consistent(demo: pf.DemoFixture) -> None:
    # LearnDemoMMLE.mat: its length n + 15 rtot + 1500 gives rtot = 12, the
    # precisions are positive, and the generator's recorded differences from
    # the fixture's fit are reproduced. Its split of 12 into three ranks cannot
    # be recovered, and its data are not the fixture's: its precisions
    # correlate 0.12 with the rng(0) draw's true precisions, the fixture fit's
    # 0.998.
    shipped = _shipped("LearnDemoMMLE.mat")
    parhist = np.asarray(shipped["parhist"], dtype=np.float64).reshape(-1)
    n, T = demo.n, demo.T
    rtot, rem = divmod(parhist.size - n - n * T, T)
    assert (rtot, rem) == (12, 0)
    lam = parhist[:n]
    assert np.all(lam > 0)
    ours = demo.learn["pars_final"]
    L = n + rtot * T
    assert (
        float(np.abs(lam - ours[:n]).max()) == demo.learn["shipped_diff"]["diff_lambda"]
    )
    assert (
        float(np.abs(parhist[L:] - ours[L:]).max())
        == (demo.learn["shipped_diff"]["diff_b"])
    )
    corr = float(np.corrcoef(lam, demo.precision)[0, 1])
    corr_fit = float(np.corrcoef(ours[:n], demo.precision)[0, 1])
    print(
        f"\n[LearnDemoMMLE.mat] corr(lambda, true precision) {corr:.3f}; "
        f"fixture {corr_fit:.3f}"
    )
    assert corr < 0.5 < 0.99 < corr_fit


def test_shipped_mmle_history_is_internally_consistent() -> None:
    # RankEstDemoMMLE.mat's inputs are unknown (mTDRdemo.m draws without a
    # seed, and no seed tried reproduces the shipped results), so only its own
    # structure is testable: parhist{k} has the length of rhist(k+1, :); each
    # accepted step raises one rank by one and lowers FunHist; the precisions
    # are positive; the search starts where the shipped SVD search ended; and
    # the fits along the path are fits to one dataset: consecutive precision
    # vectors correlate (measured 0.944-0.950; each added component raises the
    # median precision by 3-6 %, the largest entry change is 89 %) and the
    # intercepts agree (largest change 4.1 % of the largest entry).
    shipped = _shipped("RankEstDemoMMLE.mat")
    svd = _shipped("RankEstDemo_SVD.mat")
    n, T = 100, 15
    rhist = np.asarray(shipped["rhist"]).astype(int)
    funhist = np.asarray(shipped["FunHist"], dtype=np.float64).reshape(-1)
    parhist = [np.asarray(v, dtype=np.float64).reshape(-1) for v in shipped["parhist"]]
    assert len(parhist) == rhist.shape[0] - 1 == funhist.size - 1
    np.testing.assert_array_equal(rhist[0], np.asarray(svd["rhist"])[-1, :3])
    steps = np.diff(rhist, axis=0)
    assert np.all(steps.sum(axis=1) == 1)
    assert np.all(steps >= 0)
    assert np.all(np.diff(funhist) < 0)
    params = [
        mf.unpack_pars(v, n, T, list(r))
        for v, r in zip(parhist, rhist[1:], strict=True)
    ]
    assert all(np.all(p.lam > 0) for p in params)
    corr = min(
        float(np.corrcoef(a.lam, c.lam)[0, 1]) for a, c in itertools.pairwise(params)
    )
    b_spread = max(pf.rel(p.b, params[-1].b) for p in params)
    print(
        f"\n[RankEstDemoMMLE.mat] {rhist.tolist()}; smallest correlation of "
        f"consecutive precisions {corr:.3f}; largest intercept change {b_spread:.2e}"
    )
    assert corr > 0.9
    assert b_spread < 0.1
