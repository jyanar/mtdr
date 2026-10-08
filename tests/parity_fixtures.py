"""Loaders for the parity fixtures and the reference-compatible MMLE fit.

`tests/fixtures/demo.mat` (generator `tests/fixtures/matlab/make_demo_fixture.m`)
holds the two shipped demos run end to end at `rng(0)`; `paper.mat`
(`make_paper_fixture.m`) holds reference fits on the paper-scale dataset that
`tests/paper_data.py` regenerates. Everything is translated to the port's layouts
here, at the boundary (`tests/nversion/mmle_fixture.py` for the parameter
vectors).

`reference_fit` is the reference's `MMLE_CoordAscentWrapper` as closely as the
package can run it: the reference's start (`ECMEregress_wrapper`: the SVD bases,
the misaligned (M19a) precision of `tests/matlab_compat.py`, `b = Ybar`), the
`matlab_compat` ECME at the reference's loose tolerance, then `refine` with the
reference's caps. The SVD bases differ from the reference's in the signs of some
columns (the package fixes each column's sign), which the marginal likelihood,
ECME and L-BFGS-B are all equivariant to, so the start is the reference's up to
rounding.

`REFERENCE_CAPS` holds the one inner setting the same-start parity paths change:
the package's basis step stops on minFunc's absolute `progTol` by default
(`optimizer_progtol = 1e-9`), so those paths run the package's defaults with
only the reference's basis-step cap of 500 iterations restored (the package
default is 2000).

`run` records the `ConvergenceWarning`s a call raises and fails on any reason
outside the classes the test declares, so a parity test cannot discard an
unexpected inner failure.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

import matlab_compat as mc
from mtdr import aic, mmle
from mtdr.errors import ConvergenceWarning
from mtdr.stats import SufficientStats, sufficient_statistics
from mtdr.svd_fit import fit_svd
from nversion import mmle_fixture as mf

FloatArray = NDArray[np.float64]
R = TypeVar("R")
Ranks = Sequence[int] | NDArray[np.integer[Any]]

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEMO_FIXTURE = FIXTURES / "demo.mat"
PAPER_FIXTURE = FIXTURES / "paper.mat"


def _ints(x: Any) -> list[int]:
    return [int(v) for v in np.atleast_1d(np.asarray(x)).reshape(-1)]


def _vec(x: Any) -> FloatArray:
    return np.atleast_1d(np.asarray(x, dtype=np.float64)).reshape(-1)


def _rows(x: Any, width: int) -> NDArray[np.int64]:
    return np.asarray(x).astype(np.int64).reshape(-1, width)


def _cells(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray) and x.dtype == object:
        return list(x.reshape(-1))
    return [x]


@dataclass
class DemoFixture:
    """`demo.mat`: data, truth, both searches and the demoLearning fit."""

    seed: int
    n: int
    T: int
    N: int
    maxrank: int
    Y: FloatArray
    X: FloatArray
    mask: NDArray[np.bool_]
    stats: SufficientStats
    true_ranks: list[int]
    precision: FloatArray
    W_true: list[FloatArray]
    S_true: list[FloatArray]
    BB: FloatArray
    matlab_stats: dict[str, Any] = field(default_factory=dict)
    svd: dict[str, Any] = field(default_factory=dict)
    mmle: dict[str, Any] = field(default_factory=dict)
    learn: dict[str, Any] = field(default_factory=dict)


def _demo_data(raw: dict[str, Any]) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    n, T, N = int(raw["n"]), int(raw["T"]), int(raw["N"])
    hk = np.asarray(raw["hk"]).astype(bool).reshape(N, n)
    X_full = np.asarray(raw["X"], dtype=np.float64).reshape(N, -1)
    if not np.all(X_full[:, -1] == 1):
        raise ValueError("the reference's constant term must be the last column of X")
    # Zobs is Z(mask3) for Z (n x T x N) in column-major order, mask3(i, t, k)
    # = hk(k, i).
    mask3 = np.broadcast_to(hk.T[:, None, :], (n, T, N)).ravel(order="F")
    Z = np.zeros(n * T * N)
    Z[mask3] = _vec(raw["Zobs"])
    Y = mc.fixture_responses(Z.reshape(n, T, N, order="F"))
    Y = np.where(hk[:, :, None], Y, np.nan)
    return Y, np.ascontiguousarray(X_full[:, :-1]), hk


def load_demo(path: Path = DEMO_FIXTURE) -> DemoFixture:
    """Read `demo.mat` and translate it to the port's layouts."""
    raw = loadmat(path, simplify_cells=True)
    n, T, N = int(raw["n"]), int(raw["T"]), int(raw["N"])
    Y, X, mask = _demo_data(raw)
    P = X.shape[1]
    ranks = _ints(raw["rP"])
    W_true = [np.asarray(w, dtype=np.float64).reshape(n, -1) for w in raw["Wtrue"]]
    S_true = [np.asarray(s, dtype=np.float64).reshape(T, -1) for s in raw["Strue"]]
    st = raw["stats"]
    fx = DemoFixture(
        seed=int(raw["seed"]),
        n=n,
        T=T,
        N=N,
        maxrank=int(raw["maxrank"]),
        Y=Y,
        X=X,
        mask=mask,
        stats=sufficient_statistics(Y, X, mask),
        true_ranks=ranks,
        precision=_vec(raw["d"]),
        W_true=W_true,
        S_true=S_true,
        BB=np.asarray(raw["BB"], dtype=np.float64),
        matlab_stats={
            "A": np.moveaxis(np.asarray(st["Ai"], dtype=np.float64), 2, 0),
            "zzi": _vec(st["zzi"]),
            "ni": _ints(st["ni"]),
            "xi": mc.fixture_xzetai_to_xi(np.asarray(st["Xzetai0"]), P, T),
            "xbari": np.asarray(st["xbari"], dtype=np.float64),
            "Ybar": np.asarray(st["Ybar"], dtype=np.float64).T,
        },
    )
    sv = raw["svd"]
    rhist = _rows(sv["rhist"], P + 1)
    parhist = np.asarray(sv["parhist"], dtype=np.float64)
    parhist = parhist[..., None] if parhist.ndim == 3 else parhist
    fx.svd = {
        "rest": _ints(sv["rest"]),
        "rhist": rhist,
        "funhist": _vec(sv["funhist"]),
        "parhist": [
            mc.fixture_wsvd_to_b(parhist[..., j]) for j in range(parhist.shape[-1])
        ],
        "calls_ranks": _rows(sv["calls_ranks"], P + 1),
        "calls_values": _vec(sv["calls_values"]),
    }
    mm = raw["mmle"]
    rhist = _rows(mm["rhist"], P)
    fx.mmle = {
        "rest0": _ints(mm["rest0"]),
        "rest": _ints(mm["rest"]),
        "rhist": rhist,
        "funhist": _vec(mm["funhist"]),
        # parhist{k} holds the estimate at rhist(k + 1, :).
        "parhist": [
            mf.unpack_pars(_vec(v), n, T, list(rhist[k + 1]))
            for k, v in enumerate(_cells(mm["parhist"]))
        ],
        "calls_ranks": _rows(mm["calls_ranks"], P),
        "calls_values": _vec(mm["calls_values"]),
    }
    le = raw["learn"]
    r = _ints(le["r"])
    fx.learn = {
        "ranks": r,
        "start": mf.unpack_pars(_vec(le["pars0"]), n, T, r),
        "ecme": mf.unpack_pars(_vec(le["ecme_pars"]), n, T, r),
        "ecme_n_sweeps": int(le["ecme_n_sweeps"]),
        "ecme_nll": _vec(le["ecme_nll"]),
        "final": mf.unpack_pars(_vec(le["pars_final"]), n, T, r),
        "pars_final": _vec(le["pars_final"]),
        "S_matlab": [
            np.asarray(s, dtype=np.float64).reshape(rp, T)
            for s, rp in zip(_cells(le["Shat"]), r, strict=True)
        ],
        "W": mf.unpack_wt(le["Wt"], n, r),
        "Bhat": [
            np.asarray(b, dtype=np.float64).reshape(n, T) for b in _cells(le["Bhat"])
        ],
        "nll_final": float(le["nll_final"]),
        "aic": float(le["aic"]),
        "shipped_diff": {
            key: float(le[f"shipped_{key}"])
            for key in ("max_abs_diff", "diff_lambda", "diff_b", "diff_S_signed")
        },
    }
    return fx


@dataclass
class PaperFixture:
    """`paper.mat`: the reference's fits on the regenerated paper-scale data."""

    n: int
    T: int
    N: int
    true_ranks: list[int]
    precision: FloatArray
    checksums: tuple[int, int]
    matlab_stats: dict[str, Any]
    svd: dict[str, Any]
    fits: list[dict[str, Any]]
    mmle: dict[str, Any]
    minfunc_options: dict[str, Any]
    posterior: dict[str, Any]
    matlab_threads: int


def load_paper(path: Path = PAPER_FIXTURE) -> PaperFixture:
    """Read `paper.mat` (the data come from `tests/paper_data.py`)."""
    raw = loadmat(path, simplify_cells=True)
    n, T, N = int(raw["n"]), int(raw["T"]), int(raw["N"])
    ranks = _ints(raw["rP"])
    P = len(ranks)
    st = raw["stats"]
    sv = raw["svd"]
    fits = []
    for ft in _cells(raw["fits"]):
        r = _ints(ft["r"])
        rtot = sum(r)
        ca = ft["ca"]
        fits.append(
            {
                "ranks": r,
                "start_lam": _vec(ft["start_lam"]),
                "start_S": mf.unpack_s(_vec(ft["start_s"]), T, r),
                "ecme": mf.unpack_pars(_vec(ft["ecme_pars"]), n, T, r),
                "ecme_n_sweeps": int(ft["ecme_n_sweeps"]),
                "ecme_nll": _vec(ft["ecme_nll"]),
                "ca": {k: _vec(v) for k, v in ca.items()},
                "final": mf.unpack_pars(_vec(ft["pars_final"]), n, T, r),
                "nll_final": float(ft["nll_final"]),
                "aic": float(ft["aic"]),
                "seconds": float(ft["seconds"]),
                "n_pars": n + T * rtot + n * T,
            }
        )
    mm = raw["mmle"]
    mmle_search: dict[str, Any] = {}
    if isinstance(mm, dict) and "rhist" in mm:
        mmle_search = {
            "rest0": _ints(mm["rest0"]),
            "rest": _ints(mm["rest"]),
            "rhist": _rows(mm["rhist"], P),
            "funhist": _vec(mm["funhist"]),
            "calls_ranks": _rows(mm["calls_ranks"], P),
            "calls_values": _vec(mm["calls_values"]),
            "seconds": float(mm["seconds"]),
        }
    checksums = _vec(raw["checksums"])
    po = raw["posterior"]
    post_ranks = _ints(po["r"])
    return PaperFixture(
        n=n,
        T=T,
        N=N,
        true_ranks=ranks,
        precision=_vec(raw["d"]),
        checksums=(int(checksums[0]), int(checksums[1])),
        matlab_stats={
            "ni": _ints(st["ni"]),
            "zzi": _vec(st["zzi"]),
            "xbari": np.asarray(st["xbari"], dtype=np.float64),
            "Ybar_sum": _vec(st["Ybar_sum"]),
            "A0": np.asarray(st["Ai_first"], dtype=np.float64),
            "xi0": np.asarray(st["Xzetai0_first"], dtype=np.float64)
            .reshape(T, P, order="F")
            .T,
        },
        svd={
            "rest": _ints(sv["rest"]),
            "rhist": _rows(sv["rhist"], P + 1),
            "funhist": _vec(sv["funhist"]),
            "calls_ranks": _rows(sv["calls_ranks"], P + 1),
            "calls_values": _vec(sv["calls_values"]),
            "seconds": float(sv["seconds"]),
        },
        fits=fits,
        mmle=mmle_search,
        minfunc_options=dict(raw["minfunc_options"]),
        # EBpost_W_uneqvar at the true-rank fit's final estimate.
        posterior={"ranks": post_ranks, "W": mf.unpack_wt(po["Wt"], n, post_ranks)},
        matlab_threads=int(raw["matlab_threads"]),
    )


# ------------------------------------------------------------------ reference fit


def reference_start(
    stats: SufficientStats,
    Y: FloatArray,
    X: FloatArray,
    mask: NDArray[np.bool_],
    ranks: Ranks,
) -> mmle.MMLEFit:
    """`ECMEregress_wrapper`'s start: SVD bases, (M19a) precision, `b = Ybar`."""
    svd = fit_svd(stats, list(ranks))
    assert svd.intercept_full is not None
    lam = mc.svd_lambda_reference(Y, X, mask, [*svd.B, svd.intercept_full])
    return mmle.MMLEFit.from_parameters(stats, svd.S, lam, stats.Y_mean)


def reference_fit(
    stats: SufficientStats,
    Y: FloatArray,
    X: FloatArray,
    mask: NDArray[np.bool_],
    ranks: Ranks,
) -> mmle.MMLEFit:
    """`MMLE_CoordAscentWrapper` as the package runs it (module docstring)."""
    start = reference_start(stats, Y, X, mask, ranks)
    warm, _ = mmle.ecme(stats, start, matlab_compat=True)
    return mmle.refine(stats, warm, **REFERENCE_CAPS)


def nll_without_constant(fit: mmle.MMLEFit, stats: SufficientStats) -> float:
    """The marginal NLL as the reference reports it, without the $2\\pi$ term."""
    return -fit.log_likelihood - mmle._constant(stats.n_obs, stats.n_bins)


def reference_aic(
    fit: mmle.MMLEFit,
    stats: SufficientStats,
    ranks: Ranks,
) -> float:
    """`BTDR_AIC_S_lamb_b_wrapper`: twice that NLL plus twice the count (M38)."""
    count = aic.n_parameters_mmle(
        ranks, stats.n_neurons, stats.n_bins, formula="reference"
    )
    return 2.0 * nll_without_constant(fit, stats) + 2.0 * count


# ------------------------------------------------------------------ comparisons


def rel(a: Any, b: Any) -> float:
    """Max absolute difference over the largest reference entry."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    scale = max(float(np.abs(b).max()), np.finfo(np.float64).tiny)
    return float(np.abs(a - b).max() / scale)


def rel_blocks(a: Sequence[NDArray[Any]], b: Sequence[NDArray[Any]]) -> float:
    """`rel` per block, the largest."""
    assert len(a) == len(b)
    return max(rel(x, y) for x, y in zip(a, b, strict=True))


def report(label: str, errors: dict[str, float], tol: float) -> None:
    """Print the measured errors and assert each is below `tol`."""
    shown = ", ".join(f"{k} {v:.1e}" for k, v in errors.items())
    print(f"\n[{label}] {shown} (tolerance {tol:.0e})")
    bad = {k: v for k, v in errors.items() if not v < tol}
    assert not bad, bad


#: The reference's minFunc `maxIter` for the basis step; every other inner
#: setting is the package default, minFunc's `progTol` included.
REFERENCE_CAPS: dict[str, Any] = {"optimizer_max_iter": 500}


# ------------------------------------------------------------------ warnings

_NUM = r"[-+0-9.e]+"
_LBFGS = (
    r"\(L-BFGS-B status {status}, nit \d+, largest projected-gradient entry {num}\)"
)
#: The reasons of a `refinement:` warning (`mmle.refine`) that a parity test
#: may declare as expected. Anything else fails the test.
REASONS: dict[str, re.Pattern[str]] = {
    # The 500-iteration cap of the first basis step at the paper's scale.
    "basis_cap": re.compile(
        r"iteration \d+, bases: STOP: TOTAL NO\. OF ITERATIONS REACHED LIMIT "
        + _LBFGS.format(status=1, num=_NUM)
        + r"; hit its iteration cap \(max_iter = \d+; "
        r"`MTDR\(optimizer_max_iter=\.\.\.\)`\)",
        re.IGNORECASE,
    ),
    # A rounding-level line-search end of the precision step; the listed
    # scale-free stationarity gaps must all be below ROUNDING_GAP.
    "precision_rounding": re.compile(
        r"iteration \d+, noise precision: ABNORMAL(?::|_TERMINATION_IN_LNSRCH) "
        + _LBFGS.format(status=2, num=_NUM)
        + r"; neurons with the largest \|lambda_i E_i / \(n_i T\) - 1\|: "
        r"(?P<gaps>\d+ \(" + _NUM + r"\)(?:, \d+ \(" + _NUM + r"\))*)"
    ),
    # The refinement's own iteration cap.
    "refine_cap": re.compile(
        r"hit its iteration cap \(max_iter = \d+; `MTDR\(refine_max_iter=\.\.\.\)`\) "
        r"before the change fell below " + _NUM
    ),
}
ROUNDING_GAP = 1e-6
# There is no basis-step rounding class: the package counts a rounding-level
# basis-step end as a success, so a basis-step status-2 warning is a genuine
# failure. A precision-step end can still warn when the package's polish of it
# (at most three diagonal Newton steps) is not installed.


#: The aggregated warning of an `MTDR` rank search whose selected fit
#: converged: only rejected candidates failed.
SEARCH_REJECTED = "search_rejected"
_SEARCH = "rank search: "
_SELECTED = "the selected fit did: "


def reasons(message: str) -> list[str]:
    """Split a `refinement:` warning into its reason classes; fail on others.

    An `MTDR` search's aggregated warning gives `SEARCH_REJECTED` when the
    selected fit converged, else the reasons of the selected fit's warnings.
    """
    if message.startswith(_SEARCH):
        if _SELECTED not in message:
            assert "the selected fit converged" in message, message
            return [SEARCH_REJECTED]
        parts = message.split(_SELECTED, 1)[1].split(" | ")
        return [name for part in parts for name in reasons(part)]
    prefix = "refinement: "
    assert message.startswith(prefix), f"not a refinement warning: {message}"
    body, pos, out = message[len(prefix) :], 0, []
    while True:
        for name, pattern in REASONS.items():
            m = pattern.match(body, pos)
            if m is None:
                continue
            if name == "precision_rounding":
                gaps = [float(g) for g in re.findall(r"\(([^)]*)\)", m["gaps"])]
                assert max(abs(g) for g in gaps) < ROUNDING_GAP, message
            out.append(name)
            pos = m.end()
            break
        else:
            raise AssertionError(f"unexpected reason at {body[pos:]!r} in: {message}")
        if pos == len(body):
            return out
        assert body.startswith("; ", pos), message
        pos += 2


@dataclass
class Caught:
    """The `ConvergenceWarning`s of one call: messages and their reason classes."""

    messages: list[str]
    reasons: list[str]


def run(func: Callable[[], R], allowed: Collection[str] = ()) -> tuple[R, Caught]:
    """Call `func`, recording its `ConvergenceWarning`s.

    Every reason of every warning must be one of `allowed` (keys of
    `REASONS`, or `SEARCH_REJECTED`); warnings of other categories are
    re-emitted, so the suite's policy (or the caller's `pytest.warns`) applies.
    """
    unknown = set(allowed) - {*REASONS, SEARCH_REJECTED}
    assert not unknown, unknown
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        out = func()
    caught = []
    for w in recorded:
        if issubclass(w.category, ConvergenceWarning):
            caught.append(w)
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    messages = [str(w.message) for w in caught]
    found = [name for text in messages for name in reasons(text)]
    unexpected = sorted(set(found) - set(allowed))
    assert not unexpected, (unexpected, messages)
    return out, Caught(messages, found)


def final_gaps(
    fit: mmle.MMLEFit, ref: mf.Params, stats: SufficientStats
) -> dict[str, float]:
    """Estimate gaps of `fit` against a reference point: B, lambda, b."""
    assert fit.intercept is not None
    W_ref, _ = mmle.posterior_weights(ref.S, ref.lam, ref.b, stats)
    B_ref = [w @ s.T for w, s in zip(W_ref, ref.S, strict=True)]
    return {
        "B": rel_blocks(fit.B, B_ref),
        "lambda": rel(fit.noise_precision, ref.lam),
        "b": rel(fit.intercept, ref.b),
    }
