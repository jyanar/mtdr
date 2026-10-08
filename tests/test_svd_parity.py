"""MATLAB parity of the SVD path: statistics, fits, AIC and the greedy search.

`tests/fixtures/matlab/make_svd_fixture.m` runs the reference on two datasets:
A is the simulation fixture (`simulation.mat`, n=20, T=15, N=40), so the
fixtures chain; B is a demo-scale draw (n=100, T=15, N=40, rng(1)) on which the
reference rank search moves. Every comparison translates the MATLAB arrays to
the port's layouts at the boundary (`tests/matlab_compat.py`). Each test prints
the measured maximum errors; `pytest -s` shows them. Regenerating the fixture
reproduces every array and the serialised payload; only the timestamp in the
128-byte MAT header differs.

The internal-consistency test of the reference's own `RankEstDemo_SVD.mat`
needs the reference directory, which is not redistributed: set
`MTDR_REFERENCE_DIR` to the `mTDRdemo` directory (the one holding
`EstimatedPars/`); the test skips when it is unset.

Tolerances: statistics 1e-10 relative; coefficients 1e-8 relative to the
largest coefficient; reference AIC replica 1e-6 relative. Measured errors are
many orders below (see the printed values).

The reference keeps the joint least-squares intercept next to the truncated
blocks; the port refits it given them. Every comparison with a reference
routine therefore uses `SVDFit.intercept_full`, the joint least-squares
intercept, and the independent MATLAB implementation of the corrected score in
the fixture uses the conditional intercept, as the port does.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.io import loadmat

import matlab_compat as mc
from mtdr import aic as aic_mod
from mtdr import svd_fit
from mtdr.rank_search import greedy_aic
from mtdr.stats import SufficientStats, sufficient_statistics
from mtdr.svd_fit import fit_svd

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SVD_FIXTURE = FIXTURES / "svd.mat"
SIM_FIXTURE = FIXTURES / "simulation.mat"
REFERENCE_DIR = os.environ.get("MTDR_REFERENCE_DIR")
DATASETS = ("A", "B")


def _rel(a: NDArray[Any], b: NDArray[Any]) -> float:
    """Max abs difference relative to the largest reference magnitude."""
    scale = max(float(np.abs(b).max()), np.finfo(np.float64).tiny)
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / scale)


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    if not SVD_FIXTURE.is_file():  # pragma: no cover - the fixture is committed
        pytest.skip("tests/fixtures/svd.mat not available")
    raw = loadmat(SVD_FIXTURE, simplify_cells=True)
    sim = loadmat(SIM_FIXTURE)
    data: dict[str, Any] = {}
    for name in DATASETS:
        d = dict(raw[name])
        if name == "A":
            Z, X_full, hk = sim["Z"], sim["X"], sim["hk"]
        else:
            Z, X_full, hk = d["Z"], d["X"], d["hk"]
        X_full = np.asarray(X_full, dtype=np.float64)
        assert np.all(X_full[:, -1] == 1)  # the reference's constant term is last
        mask = np.asarray(hk).astype(bool)
        Y = mc.fixture_responses(Z)
        # Unobserved entries are zero in Z; the port must ignore them, so make
        # them NaN here.
        d["Y"] = np.where(mask[:, :, None], Y, np.nan)
        d["X"] = X_full[:, :-1]
        d["mask"] = mask
        d["stats"] = sufficient_statistics(d["Y"], d["X"], mask)
        for key in ("n", "T", "N", "maxrank", "k_factors"):
            d[key] = int(d[key])
        # MATLAB saves small integer arrays as uint8 or int16; widen them so
        # arithmetic such as ni * T cannot wrap.
        for key in ("ni", "Ai", "nparam_textbook", "nparam_reference"):
            d[key] = np.asarray(d[key]).astype(np.int64)
        d["rank_list"] = np.atleast_2d(d["rank_list"]).astype(int)
        searches = d["searches"]
        d["searches"] = searches if isinstance(searches, list) else [searches]
        data[name] = d
    return data


def _sizes(d: dict[str, Any]) -> tuple[int, int, int]:
    return d["n"], d["T"], d["X"].shape[1]


# ------------------------------------------------------------ sufficient stats


@pytest.mark.parametrize("name", DATASETS)
def test_sufficient_statistics_match_reference(
    fixture: dict[str, Any], name: str
) -> None:
    d = fixture[name]
    _n, T, P = _sizes(d)
    stats: SufficientStats = d["stats"]
    errors = {
        "XtX": _rel(stats.XtX, np.moveaxis(d["Ai"], 2, 0)),
        "XtY_raw": _rel(stats.XtY_raw, mc.fixture_xzetai_to_xi(d["Xzetai0"], P, T)),
        "YtY_raw": _rel(stats.YtY_raw, d["zzi"]),
        "X_mean": _rel(stats.X_mean, d["xbari"]),
        "Y_mean": _rel(stats.Y_mean, d["Ybar"].T),
    }
    print(f"\n[{name}] sufficient statistics, max relative error: {errors}")
    np.testing.assert_array_equal(stats.n_obs, d["ni"])
    for key, err in errors.items():
        assert err < 1e-10, key


@pytest.mark.parametrize("name", DATASETS)
def test_centered_statistics_match_ecmesuffstat(
    fixture: dict[str, Any], name: str
) -> None:
    d = fixture[name]
    _n, T, P = _sizes(d)
    XtY, YtY = d["stats"].centered(d["b_test"].T)  # MATLAB b is T x n
    errors = (
        _rel(XtY, mc.fixture_xzetai_to_xi(d["Xzetai_c"], P, T)),
        _rel(YtY, d["zzi_c"]),
    )
    print(f"\n[{name}] centered(b) vs ECMEsuffstat, max relative error: {errors}")
    assert max(errors) < 1e-10


def _augmented(stats: SufficientStats) -> tuple[NDArray[Any], NDArray[Any]]:
    # A^f_i = [[A_i, n_i xbar_i], [n_i xbar_i', n_i]], xi^f_i = [xi_i; n_i ybar_i'],
    # the constant column last (`docs/model.md` § A.6), from the raw views of
    # the port.
    n, P, T = stats.n_neurons, stats.n_regressors, stats.n_bins
    count = stats.n_obs.astype(np.float64)
    A = np.empty((n, P + 1, P + 1))
    A[:, :P, :P] = stats.XtX
    A[:, :P, P] = A[:, P, :P] = count[:, None] * stats.X_mean
    A[:, P, P] = count
    xi = np.empty((n, P + 1, T))
    xi[:, :P] = stats.XtY_raw
    xi[:, P] = count[:, None] * stats.Y_mean
    return A, xi


def test_full_design_statistics_match_bilinreg(fixture: dict[str, Any]) -> None:
    # MkSuffStats_BilinReg_Sims: XX = blkdiag_i(A^f_i kron I_T) and XY, both
    # permuted to the index t + T*i + T*n*p (`docs/model.md` § A.6).
    d = fixture["A"]
    n, T, P = _sizes(d)
    A, xi = _augmented(d["stats"])
    Q = P + 1
    # Row and column index t + T*i + T*n*p is C order over (p, i, t).
    XX = np.einsum("ipq,ij,tu->pitqju", A, np.eye(n), np.eye(T)).reshape(Q * n * T, -1)
    XY = xi.transpose(1, 0, 2).reshape(-1)
    errors = (_rel(XX, d["XX"].toarray()), _rel(XY, d["XY"]))
    print(f"\n[A] full-design XX, XY vs MkSuffStats_BilinReg_Sims: {errors}")
    assert max(errors) < 1e-10


# ------------------------------------------------------------------ SVD fits


@pytest.mark.parametrize("name", DATASETS)
def test_unconstrained_solution_matches_backslash(
    fixture: dict[str, Any], name: str
) -> None:
    d = fixture[name]
    n, T, P = _sizes(d)
    fit = fit_svd(d["stats"], [1] * P)
    ref = mc.fixture_w0_to_b(d["w0"], n, T)
    assert fit.intercept_full is not None
    err = _rel(np.stack([*fit.B_full, fit.intercept_full]), np.stack(ref))
    print(f"\n[{name}] B_full and intercept vs XX\\XY: {err:.3g}")
    assert err < 1e-8


@pytest.mark.parametrize("name", DATASETS)
def test_truncations_match_svdregressb(fixture: dict[str, Any], name: str) -> None:
    d = fixture[name]
    n, T, _P = _sizes(d)
    worst = 0.0
    for k, r_full in enumerate(d["rank_list"]):
        assert r_full[-1] == d["maxrank"]
        ranks = r_full[:-1]
        # A block with r_p != T exists in every rank vector, so a transposed
        # layout cannot pass silently (`docs/model.md` § 1.2).
        assert (ranks != T).any()
        fit = fit_svd(d["stats"], ranks)
        assert fit.intercept_full is not None
        ref = mc.fixture_wsvd_to_b(d["wsvd_all"][..., k])
        err = _rel(np.stack([*fit.B, fit.intercept_full]), np.stack(ref))
        for p, r in enumerate(ranks):
            assert np.linalg.matrix_rank(fit.B[p]) == min(r, n, T)
        worst = max(worst, err)
    print(f"\n[{name}] rank truncations vs SVDRegressB wsvd, worst: {worst:.3g}")
    assert worst < 1e-8


@pytest.mark.parametrize("name", DATASETS)
def test_factors_match_up_to_column_sign(fixture: dict[str, Any], name: str) -> None:
    d = fixture[name]
    n, T, P = _sizes(d)
    ranks = d["rank_list"][d["k_factors"] - 1, :-1]
    fit = fit_svd(d["stats"], ranks)
    worst = 0.0
    for p in range(P):
        wt, wx = np.atleast_2d(d["wt"][p]), np.atleast_2d(d["wx"][p])
        wt = wt.reshape(T, -1)
        wx = wx.reshape(n, -1)
        assert fit.S[p].shape == wt.shape == (T, ranks[p])
        sign = np.sign(np.sum(fit.S[p] * wt, axis=0))
        assert np.all(sign != 0)
        worst = max(worst, _rel(fit.S[p] * sign, wt), _rel(fit.W[p] * sign, wx))
    print(f"\n[{name}] W, S vs wx, wt up to column sign: {worst:.3g}")
    assert worst < 1e-8


# ----------------------------------------------------------- reference AIC


@pytest.mark.parametrize("name", DATASETS)
def test_reference_aic_replica_on_matlab_coefficients(
    fixture: dict[str, Any], name: str
) -> None:
    # Isolates the replica: it is fed MATLAB's own wsvd.
    d = fixture[name]
    errs = []
    for k, r_full in enumerate(d["rank_list"]):
        B = mc.fixture_wsvd_to_b(d["wsvd_all"][..., k])
        value = mc.svd_aic_reference(d["Y"], d["X"], d["mask"], B, r_full)
        errs.append(abs(value - d["aic_ref"][k]) / abs(d["aic_ref"][k]))
    print(f"\n[{name}] SVDRegB_AIC replica on MATLAB wsvd, relative: {max(errs):.3g}")
    assert max(errs) < 1e-6


@pytest.mark.parametrize("name", DATASETS)
def test_reference_aic_replica_on_port_fits(fixture: dict[str, Any], name: str) -> None:
    d = fixture[name]
    errs = []
    for k, r_full in enumerate(d["rank_list"]):
        fit = fit_svd(d["stats"], r_full[:-1])
        assert fit.intercept_full is not None
        value = mc.svd_aic_reference(
            d["Y"], d["X"], d["mask"], [*fit.B, fit.intercept_full], r_full
        )
        errs.append(abs(value - d["aic_ref"][k]) / abs(d["aic_ref"][k]))
    print(f"\n[{name}] SVDRegB_AIC replica on port fits, relative: {max(errs):.3g}")
    assert max(errs) < 1e-6


@pytest.mark.parametrize("name", DATASETS)
def test_reference_count_matches(fixture: dict[str, Any], name: str) -> None:
    d = fixture[name]
    n, T, _ = _sizes(d)
    for k, r_full in enumerate(d["rank_list"]):
        assert aic_mod.n_parameters_svd(r_full[:-1], n, T, formula="reference") == int(
            d["nparam_reference"][k]
        )


@pytest.mark.parametrize("name", DATASETS)
def test_reference_precision_and_packed_bases(
    fixture: dict[str, Any], name: str
) -> None:
    # SVDRegress_S_Vdata returns [lambda_code; s], s = [vec(wt{1}); ...].
    d = fixture[name]
    n, T, P = _sizes(d)
    r_full = d["rank_list"][d["k_factors"] - 1]
    fit = fit_svd(d["stats"], r_full[:-1])
    assert fit.intercept_full is not None
    lam = mc.svd_lambda_reference(
        d["Y"], d["X"], d["mask"], [*fit.B, fit.intercept_full]
    )
    err_lam = _rel(lam, d["pars_vdata"][:n])
    # The task blocks of s, unpacked as in `docs/model.md` § 1.2:
    # S[p] = s[...].reshape(T, r_p, "F").
    s = d["pars_vdata"][n:]
    offsets = np.concatenate([[0], np.cumsum(r_full)]) * T
    err_s = 0.0
    for p in range(P):
        S_ref = s[offsets[p] : offsets[p + 1]].reshape(T, r_full[p], order="F")
        sign = np.sign(np.sum(fit.S[p] * S_ref, axis=0))
        err_s = max(err_s, _rel(fit.S[p] * sign, S_ref))
    # The last block is the constant term's T x min(n, T) factor.
    S0 = s[offsets[P] :].reshape(T, r_full[P], order="F")
    W0 = np.atleast_2d(d["wx"][P]).reshape(n, -1)
    err_s = max(err_s, _rel(W0 @ S0.T, fit.intercept_full))
    print(f"\n[{name}] lambda (M19a) replica: {err_lam:.3g}; packed s: {err_s:.3g}")
    assert err_lam < 1e-8
    assert err_s < 1e-8
    # The code's precision is far from the aligned one (`docs/model.md` § 3.2).
    ratio = np.median(d["pars_vdata"][:n] / fit.noise_precision)
    print(f"[{name}] median lambda_code / lambda_aligned: {ratio:.3g}")
    assert ratio < 0.5


# --------------------------------------------------- corrected (port) quantities


@pytest.mark.parametrize("name", DATASETS)
def test_corrected_score_matches_independent_matlab(
    fixture: dict[str, Any], name: str
) -> None:
    # make_svd_fixture.m's corrected_score: (M19b), (M21a), (M21b) from the
    # per-neuron residuals, with the intercept refitted given the truncated
    # blocks and the task blocks of an independent centred solve and
    # truncation, written without reference to the port.
    d = fixture[name]
    _n, T, _ = _sizes(d)
    errs: dict[str, float] = {"lambda": 0.0, "loglik": 0.0, "aic": 0.0}
    for k, r_full in enumerate(d["rank_list"]):
        fit = fit_svd(d["stats"], r_full[:-1])
        lam_ref = d["ni"] * T / d["rss_aligned"][:, k]
        errs["lambda"] = max(errs["lambda"], _rel(fit.noise_precision, lam_ref))
        ll = float(d["loglik_corrected"][k])
        errs["loglik"] = max(errs["loglik"], abs(fit.log_likelihood - ll) / abs(ll))
        a = float(d["aic_corrected"][k])
        errs["aic"] = max(errs["aic"], abs(fit.aic - a) / abs(a))
        assert fit.n_parameters == int(d["nparam_textbook"][k])
    print(f"\n[{name}] corrected score vs independent MATLAB: {errs}")
    assert max(errs.values()) < 1e-10


# ------------------------------------------------------------- greedy search


def _check_search(
    d: dict[str, Any],
    history: Any,
    rhist: NDArray[Any],
    funhist: NDArray[Any],
    calls_ranks: NDArray[Any],
    calls_values: NDArray[Any],
) -> float:
    """Compare a port history with an EstRankGreedily run; return the max error."""
    rhist = np.atleast_2d(rhist).astype(int)
    funhist = np.atleast_1d(funhist).astype(float)
    calls_ranks = np.atleast_2d(calls_ranks).astype(int)
    calls_values = np.atleast_1d(calls_values).astype(float)
    assert (rhist[:, -1] == d["maxrank"]).all()  # the constant term never moves
    np.testing.assert_array_equal(history.ranks, rhist[:, :-1])
    err = float(np.max(np.abs(history.aic - funhist) / np.abs(funhist)))
    # Every objective evaluation, in order: the start, then each round's
    # candidates in regressor order (MATLAB evaluates the movable ones only).
    port_calls = [(history.ranks[0].tolist(), float(history.aic[0]))]
    for k, round_ in enumerate(history.candidates):
        for p, name in enumerate(history.regressor_names):
            if name in round_:
                trial = history.ranks[k].copy()
                trial[p] += 1
                port_calls.append((trial.tolist(), round_[name]))
    assert [c[0] for c in port_calls] == calls_ranks[:, :-1].tolist()
    values = np.array([c[1] for c in port_calls])
    err = max(err, float(np.max(np.abs(values - calls_values) / np.abs(calls_values))))
    # Decisions are robust to the measured error: no round's best candidate is
    # within 1e3 times the error of the acceptance boundary.
    for k, round_ in enumerate(history.candidates):
        if round_:
            gap = abs(min(round_.values()) - history.aic[k]) / abs(history.aic[k])
            assert gap > 1e3 * max(err, 1e-15)
    return err


@pytest.mark.parametrize("name", DATASETS)
def test_reference_greedy_path_with_replica_objective(
    fixture: dict[str, Any], name: str
) -> None:
    d = fixture[name]
    _n, _T, _P = _sizes(d)
    stats, maxrank = d["stats"], d["maxrank"]

    def objective(fit: svd_fit.SVDFit, ranks: NDArray[np.int64]) -> float:
        assert fit.intercept_full is not None
        return mc.svd_aic_reference(
            d["Y"], d["X"], d["mask"], [*fit.B, fit.intercept_full], [*ranks, maxrank]
        )

    worst = 0.0
    for search in d["searches"]:
        start = np.atleast_1d(search["start"]).astype(int)
        best, history = greedy_aic(
            functools.partial(fit_svd, stats), objective, start, maxrank
        )
        worst = max(
            worst,
            _check_search(
                d,
                history,
                search["rhist"],
                search["funhist"],
                search["calls_ranks"],
                search["calls_values"],
            ),
        )
        assert best.ranks == tuple(history.ranks[-1])
        # parhist: one wsvd per accepted step (the start when none was), each
        # equal to the port's fit at that step's ranks.
        parhist = np.asarray(search["parhist"])
        parhist = parhist[..., None] if parhist.ndim == 3 else parhist
        steps = history.ranks[1:] if len(history.accepted) else history.ranks
        assert parhist.shape[-1] == len(steps)
        for j, ranks in enumerate(steps):
            fit = fit_svd(stats, ranks)
            assert fit.intercept_full is not None
            ref = mc.fixture_wsvd_to_b(parhist[..., j])
            assert _rel(np.stack([*fit.B, fit.intercept_full]), np.stack(ref)) < 1e-8
        print(
            f"\n[{name}] reference search from {start.tolist()}: "
            f"{history.ranks.tolist()}, max relative error {worst:.3g}"
        )
    assert worst < 1e-6
    if name == "B":  # the dataset on which the reference search moves
        assert len(history.accepted) >= 2


@pytest.mark.parametrize("name", DATASETS)
def test_corrected_greedy_path_matches_matlab(
    fixture: dict[str, Any], name: str
) -> None:
    d = fixture[name]
    _, _, P = _sizes(d)
    search = d["corrected_search"]
    best, history = greedy_aic(
        functools.partial(fit_svd, d["stats"]),
        lambda f, r: f.aic,
        [1] * P,
        d["maxrank"],
    )
    err = _check_search(
        d,
        history,
        search["rhist"],
        search["funhist"],
        search["calls_ranks"],
        search["calls_values"],
    )
    print(
        f"\n[{name}] corrected search: {history.ranks[-1].tolist()} in "
        f"{len(history.accepted)} steps, max relative error {err:.3g}"
    )
    assert err < 1e-10
    assert best.ranks == tuple(history.ranks[-1])
    assert history.stop_reason == "no_improvement"


# ------------------------------------------- the shipped RankEstDemo_SVD.mat


@pytest.fixture(scope="module")
def shipped() -> dict[str, Any]:
    # The reference zip's own history, outside the repository (not
    # redistributed). Its inputs (Z, X, hk) are not in the file and could not
    # be regenerated: mTDRdemo.m draws without seeding, and rng(s) for
    # twister seeds 0-9999 and six other generators at seeds 0-100 do not
    # reproduce parhist{1}. Only internal consistency is testable.
    if not REFERENCE_DIR:
        pytest.skip(
            "set MTDR_REFERENCE_DIR to the mTDRdemo directory to check the shipped "
            "RankEstDemo_SVD.mat"
        )
    path = Path(REFERENCE_DIR) / "EstimatedPars" / "RankEstDemo_SVD.mat"
    if not path.is_file():
        pytest.skip(f"{path} not found (MTDR_REFERENCE_DIR={REFERENCE_DIR})")
    return loadmat(path, simplify_cells=True)


def test_shipped_history_is_nested_truncations_of_one_solution(
    shipped: dict[str, Any],
) -> None:
    # Every parhist entry is a rank truncation of the same B_full, so (i) each
    # block has exactly the rank rhist records, (ii) the constant block is the
    # same in every entry, and (iii) truncating a later (higher-rank) entry with
    # the port's truncation gives every earlier one. This checks the layout
    # translation B[p] = wsvd[:, :, p].T and the truncation on the reference's
    # own output.
    rhist = np.asarray(shipped["rhist"]).astype(int)
    parhist = [mc.fixture_wsvd_to_b(np.asarray(w)) for w in shipped["parhist"]]
    assert rhist.shape == (7, 4)
    assert len(parhist) == 6
    last = parhist[-1]
    worst = 0.0
    for k, B in enumerate(parhist):
        ranks = rhist[k + 1]
        for p, b in enumerate(B):
            sv = np.linalg.svd(b, compute_uv=False)
            assert (sv[ranks[p] :] < 1e-10 * sv[0]).all()
            assert (sv[: ranks[p]] > 1e-6 * sv[0]).all()
        worst = max(worst, _rel(B[3], last[3]))
        trunc = svd_fit._truncate(np.stack(last[:3]), ranks[:3])[2]
        for p in range(3):
            worst = max(worst, _rel(trunc[p], B[p]))
    print(f"\n[RankEstDemo_SVD.mat] nested truncation consistency: {worst:.3g}")
    assert worst < 1e-10
