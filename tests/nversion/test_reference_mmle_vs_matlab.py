"""The N-version oracle `reference_mmle` against the MATLAB reference itself.

`tests/fixtures/matlab/make_mmle_fixture.m` runs the reference on two datasets
(A: the draw of `simulation.mat`, n=20; B: the demo-scale draw of `svd.mat`,
n=100; T=15, N=40, true ranks [2 1 3] for both) and saves

* five random parameter points per dataset (the last with ridge `g = 0.7`),
  with the three NLL files, both gradients, `EBpost_W_uneqvar`, `MMLE_b` and
  one `ECMEtdr` sweep evaluated at each;
* the `mTDRdemo.m` MMLE pipeline at ranks [2 1 3], [2 1 2], [3 1 3]: SVD
  start, every ECME iterate, the coordinate-ascent trace, the posterior,
  `MakeBhat_data` and the AIC;
* `EstRankGreedily` with the MMLE objective from the recorded SVD ranks.

Every comparison is in the port's layouts (`mmle_fixture`). The measure is the
max absolute difference relative to the largest reference magnitude; the gate
is 1e-10 (the oracle must be good to 1e-8 for the package's checks).
`ECMEtdr`'s `parerr` is a ratio `(new-old)^2/old^2` whose denominator can be
tiny, so it is compared to 1e-8; its decisive use, the number of sweeps, is
compared exactly. `pytest -s` prints the measured errors. The tests skip when
the fixture is missing.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import matlab_compat as mc
from nversion import mmle_fixture as mf
from nversion import reference_mmle as rm

TOL = 1e-10
PARERR_TOL = 1e-8
N_POINTS = 5
N_FITS = 3
POINTS = [(name, k) for name in mf.DATASETS for k in range(N_POINTS)]
FITS = [(name, k) for name in mf.DATASETS for k in range(N_FITS)]


def _rel(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    scale = max(float(np.abs(b).max()), np.finfo(np.float64).tiny)
    return float(np.abs(a - b).max() / scale)


def _rel_blocks(a: Sequence[NDArray[Any]], b: Sequence[NDArray[Any]]) -> float:
    assert len(a) == len(b)
    return max(_rel(x, y) for x, y in zip(a, b, strict=True))


def _rel_params(a: tuple[Sequence[NDArray[Any]], Any, Any], b: mf.Params) -> float:
    return max(_rel_blocks(a[0], b.S), _rel(a[1], b.lam), _rel(a[2], b.b))


def _report(label: str, errors: dict[str, float], tol: float = TOL) -> None:
    shown = ", ".join(f"{k} {v:.1e}" for k, v in errors.items())
    print(f"\n[{label}] max relative error: {shown}")
    bad = {k: v for k, v in errors.items() if not v < tol}
    assert not bad, bad


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    if not mf.MMLE_FIXTURE.is_file():
        pytest.skip(
            "tests/fixtures/mmle.mat not available; regenerate it with "
            "tests/fixtures/matlab/make_mmle_fixture.m"
        )
    return mf.load()


# --------------------------------------------------------------------------- setup


def test_fixture_contents(fixture: dict[str, Any]) -> None:
    assert fixture["mmx_available"] == 0  # every mmx call ran its slow* fallback
    mfo = fixture["minfunc_options"]
    assert (mfo["maxIter"], mfo["maxFunEvals"], mfo["corrections"]) == (500, 1000, 100)
    assert (mfo["optTol"], mfo["progTol"]) == (1e-5, 1e-9)
    assert mfo["useMex"] == 1
    assert fixture["fminunc_options"]["algorithm_reported"] == "trust-region"
    for name in mf.DATASETS:
        ds = fixture[name]
        assert len(ds.points) == N_POINTS
        assert len(ds.fits) == N_FITS
        assert ds.true_ranks == [2, 1, 3]
        print(
            f"\n[{name}] n={ds.n} T={ds.T} N={ds.N}; SVD ranks {ds.svd_ranks}; "
            f"ECME sweeps {[f['ecme_n_sweeps'] for f in ds.fits]}; CA minFunc exit "
            f"{[f['ca']['minfunc_exitflag'].astype(int).tolist() for f in ds.fits]}; "
            "fminunc exit "
            f"{[f['ca']['fminunc_exitflag'].astype(int).tolist() for f in ds.fits]}"
        )


@pytest.mark.parametrize("name", mf.DATASETS)
def test_statistics_are_the_reference_ones(fixture: dict[str, Any], name: str) -> None:
    ds = fixture[name]
    st, ms = ds.stats, ds.matlab_stats
    errors = {
        "Ai": _rel(st.XtX, ms["A"]),
        "Xzetai0": _rel(st.XtY_raw, ms["xi"]),
        "zzi": _rel(st.YtY_raw, ms["zzi"]),
        "xbari": _rel(st.X_mean, ms["xbari"]),
        "Ybar": _rel(st.Y_mean, ms["Ybar"]),
    }
    # ECMEsuffstat(zetai, Xi, b) is stats.centered(b) at every point.
    errors["ECMEsuffstat Xzetai"] = max(
        _rel(st.centered(pt["b"])[0], pt["xi_c"]) for pt in ds.points
    )
    errors["ECMEsuffstat zzi"] = max(
        _rel(st.centered(pt["b"])[1], pt["zzi_c"]) for pt in ds.points
    )
    np.testing.assert_array_equal(st.n_obs, ms["ni"])
    _report(f"{name} statistics", errors)


# --------------------------------------------------------------------------- points


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_nll_matches_nllonly(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    pt = ds.points[k]
    S, lam, b, g = pt["S"], pt["lam"], pt["b"], pt["g"]
    ref = rm.nll(S, lam, b, ds.stats, g=g, matlab_constants_only=True)
    full = rm.nll(S, lam, b, ds.stats, g=g)
    errors = {
        "nllonly": _rel(ref, pt["nll_nllonly"]),
        "nllonly, uncentred": _rel(
            rm.nll(S, lam, None, ds.stats, g=g, matlab_constants_only=True),
            pt["nll_nllonly_raw"],
        ),
        "full - constant": _rel(
            full - rm.normalising_constant(ds.stats), pt["nll_nllonly"]
        ),
    }
    _report(f"{name} point {k} r={pt['ranks']} g={g}", errors)


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_grad_S_matches_Sonly(fixture: dict[str, Any], name: str, k: int) -> None:  # noqa: N802
    ds = fixture[name]
    pt = ds.points[k]
    value, grad = rm.nll_grad_S(
        pt["S"], pt["lam"], pt["b"], ds.stats, g=pt["g"], matlab_constants_only=True
    )
    errors = {
        "Sonly value": _rel(value, pt["nll_Sonly"]),
        "Sonly grad": _rel_blocks(grad, pt["grad_S"]),
    }
    _report(f"{name} point {k} r={pt['ranks']} g={pt['g']}", errors)


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_grad_lam_matches_lambonly(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    pt = ds.points[k]
    # lambonly's regterm slices an empty vector: g never enters (lambonly.m:88).
    value, grad = rm.nll_grad_lam(
        pt["S"], pt["lam"], pt["b"], ds.stats, matlab_constants_only=True
    )
    errors = {
        "lambonly value": _rel(value, pt["nll_lambonly"]),
        "lambonly grad": _rel(grad, pt["grad_lam"]),
    }
    _report(f"{name} point {k} r={pt['ranks']} g={pt['g']}", errors)


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_posterior_matches_ebpost(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    pt = ds.points[k]
    W, cov = rm.posterior_W(pt["S"], pt["lam"], pt["b"], ds.stats)
    errors = {
        "Wt": _rel_blocks(W, pt["W"]),
        "Ci (precision)": _rel(
            rm.feature_precision(pt["S"], pt["lam"], ds.stats), pt["Ci_post"]
        ),
        "inv(Ci)": _rel(cov, np.linalg.inv(pt["Ci_post"])),
    }
    _report(f"{name} point {k} r={pt['ranks']}", errors)


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_update_b_matches_mmle_b(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    pt = ds.points[k]
    errors = {
        "Ci (CoordAscent)": _rel(
            rm.feature_precision(pt["S"], pt["lam"], ds.stats), pt["Ci"]
        ),
        "MMLE_b": _rel(rm.update_b(pt["S"], pt["lam"], ds.stats), pt["b_mmle"]),
    }
    _report(f"{name} point {k} r={pt['ranks']}", errors)


@pytest.mark.parametrize(("name", "k"), POINTS)
def test_ecme_step_matches_one_sweep(
    fixture: dict[str, Any], name: str, k: int
) -> None:
    ds = fixture[name]
    pt = ds.points[k]
    step = rm.ecme_step(pt["S"], pt["lam"], pt["b"], ds.stats)
    S1, lam1, b1, parerr, trace = rm.ecme(
        "steps", 1, pt["S"], pt["lam"], pt["b"], ds.stats, matlab_constants_only=True
    )
    nx = pt["ecme_next"]
    errors = {
        "S": _rel_blocks(step[0], nx.S),
        "lambda": _rel(step[1], nx.lam),
        "b": _rel(step[2], nx.b),
        "ecme('steps',1)": _rel_params((S1, lam1, b1), nx),
        "nll trace": _rel(trace, pt["ecme_nll"]),
    }
    _report(f"{name} point {k} r={pt['ranks']}", errors)
    assert _rel(parerr, pt["ecme_parerr"]) < PARERR_TOL


# --------------------------------------------------------------------------- pipeline


@pytest.mark.parametrize(("name", "k"), FITS)
def test_ecme_iterates_step_by_step(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    fit = ds.fits[k]
    its = fit["ecme_iterates"]
    assert len(its) == fit["ecme_n_sweeps"] + 1
    errs = [
        _rel_params(rm.ecme_step(a.S, a.lam, a.b, ds.stats), c)
        for a, c in itertools.pairwise(its)
    ]
    print(f"\n[{name} r={fit['ranks']}] per-sweep errors: {[f'{e:.1e}' for e in errs]}")
    assert max(errs) < TOL


@pytest.mark.parametrize(("name", "k"), FITS)
def test_ecme_converge_run(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    fit = ds.fits[k]
    start = fit["start"]
    S, lam, b, parerr, trace = rm.ecme(
        "converge",
        1.0,
        start.S,
        start.lam,
        start.b,
        ds.stats,
        matlab_constants_only=True,
    )
    assert parerr.size == fit["ecme_n_sweeps"]
    errors = {
        "final": _rel_params((S, lam, b), fit["ecme"]),
        "nll trace": _rel(trace, fit["ecme_nll"]),
    }
    _report(f"{name} r={fit['ranks']} ({parerr.size} sweeps)", errors)
    perr = _rel(parerr, fit["ecme_parerr"])
    print(f"  parerr {perr:.1e} (tolerance {PARERR_TOL:g})")
    assert perr < PARERR_TOL
    # The start is the SVD fit with the intercept replaced by Ybar
    # (ECMEregress_wrapper.m:5).
    np.testing.assert_array_equal(start.b, ds.matlab_stats["Ybar"])


@pytest.mark.parametrize(("name", "k"), FITS)
def test_coordinate_ascent_trace(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    fit = ds.fits[k]
    ca = fit["ca"]
    st = ds.stats
    prev = fit["ecme"]
    errs: dict[str, list[float]] = {
        "minFunc f": [],
        "fminunc f": [],
        "nll": [],
        "MMLE_b": [],
    }
    for j, cur in enumerate(ca["params"]):
        # minFunc minimised Sonly over S at (lambda, b) of the previous iterate,
        # fminunc minimised lambonly over lambda at the new S and the old b.
        errs["minFunc f"].append(
            _rel(
                rm.nll_grad_S(cur.S, prev.lam, prev.b, st, matlab_constants_only=True)[
                    0
                ],
                ca["minfunc_f"][j],
            )
        )
        errs["fminunc f"].append(
            _rel(
                rm.nll_grad_lam(cur.S, cur.lam, prev.b, st, matlab_constants_only=True)[
                    0
                ],
                ca["fminunc_f"][j],
            )
        )
        errs["nll"].append(
            _rel(
                rm.nll(cur.S, cur.lam, cur.b, st, matlab_constants_only=True),
                ca["nll"][j],
            )
        )
        errs["MMLE_b"].append(_rel(rm.update_b(cur.S, cur.lam, st), cur.b))
        prev = cur
    final = fit["final"]
    assert mf.pack_pars(ca["params"][-1]).tolist() == mf.pack_pars(final).tolist()
    nll_ref = rm.nll(final.S, final.lam, final.b, st, matlab_constants_only=True)
    count, aic = rm.aic_mmle(nll_ref, final.ranks, ds.n, ds.T)
    assert count == fit["pars_final"].size
    errors = {key: max(v) for key, v in errs.items()}
    errors["nll_final"] = _rel(nll_ref, fit["nll_final"])
    errors["AIC"] = _rel(aic, fit["aic"])
    _report(f"{name} r={fit['ranks']} ({len(ca['params'])} CA iterations)", errors)


@pytest.mark.parametrize(("name", "k"), FITS)
def test_posterior_and_bhat_at_fit(fixture: dict[str, Any], name: str, k: int) -> None:
    ds = fixture[name]
    fit = ds.fits[k]
    final = fit["final"]
    W, _cov = rm.posterior_W(final.S, final.lam, final.b, ds.stats)
    W_raw, _ = rm.posterior_W(final.S, final.lam, None, ds.stats)
    errors = {
        "Wt": _rel_blocks(W, fit["W"]),
        "Ci": _rel(rm.feature_precision(final.S, final.lam, ds.stats), fit["Ci_post"]),
        "Bhat": _rel_blocks(
            [w @ s.T for w, s in zip(W, final.S, strict=True)], fit["Bhat"]
        ),
        "Bhat (uncentred, mTDRdemo.m:155)": _rel_blocks(
            [w @ s.T for w, s in zip(W_raw, final.S, strict=True)], fit["Bhat_raw"]
        ),
    }
    _report(f"{name} r={fit['ranks']}", errors)


@pytest.mark.parametrize("name", mf.DATASETS)
def test_mmle_rank_search_replay(fixture: dict[str, Any], name: str) -> None:
    ds = fixture[name]
    sr = ds.search
    assert sr["rest0"] == ds.svd_ranks[:-1]  # seeded from the recorded SVD ranks
    calls = iter(zip(sr["calls_ranks"], sr["calls_params"], strict=True))
    errs: list[float] = []

    def estfun(r: NDArray[np.int64]) -> mf.Params:
        ranks, params = next(calls)
        np.testing.assert_array_equal(r, ranks)  # same call order as MATLAB
        assert isinstance(params, mf.Params)
        return params

    def objective(params: mf.Params, r: NDArray[np.int64]) -> float:
        value = rm.nll(
            params.S, params.lam, params.b, ds.stats, matlab_constants_only=True
        )
        return rm.aic_mmle(value, [int(x) for x in r], ds.n, ds.T)[1]

    rest, rhist, funhist, parhist = mc.est_rank_greedily_reference(
        objective, estfun, sr["rest0"], ds.maxrank
    )
    for params, value in zip(sr["calls_params"], sr["calls_values"], strict=True):
        errs.append(_rel(objective(params, np.asarray(params.ranks)), value))
    assert next(calls, None) is None  # every logged call was replayed
    np.testing.assert_array_equal(rest, sr["rest"])
    np.testing.assert_array_equal(rhist, sr["rhist"])
    assert len(parhist) == len(sr["parhist"])
    for got, want in zip(parhist, sr["parhist"], strict=True):
        assert mf.pack_pars(got).tolist() == mf.pack_pars(want).tolist()
    errors = {"AIC per call": max(errs), "FunHist": _rel(funhist, sr["funhist"])}
    _report(f"{name} search {sr['rest0']} -> {sr['rest']} ({len(errs)} calls)", errors)
