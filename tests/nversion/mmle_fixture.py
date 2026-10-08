"""Loader for `tests/fixtures/mmle.mat`: MATLAB arrays to the port's layouts.

`tests/fixtures/matlab/make_mmle_fixture.m` writes the file. Everything is
translated here, at the boundary, so the tests and the oracle only see the
port's layouts (`docs/model.md` § 1.2):

* a reference parameter vector `[lambda; s; vec(b)]` becomes
  `Params(S, lam, b)` with `S[p] = s[T*c[p]:T*c[p+1]].reshape(T, r_p, order="F")`
  (`(T, r_p)`) and `b = v[n + T*rtot:].reshape(n, T)` (`(n, T)`, MATLAB's
  `T x n` transposed);
* a gradient in `s` packing (`Sonly.m:102-112`) unpacks like `s`;
* `Wt` (`rtot x n`) becomes `W[p] = Wt[c[p]:c[p+1], :].T` (`(n, r_p)`);
* `Ci` (`rtot x rtot x n`) becomes `(n, rtot, rtot)`;
* `Xzetai` (`TP x n`) becomes `xi[i, p, t]`;
* MATLAB's `r_p x T` blocks (`Shat{p}`, `mat2cell` output) are kept as they
  are under `*_matlab` names so a test can check `S[p] == Shat{p}.T`, which
  is how the loader is tested on blocks with `r_p != T`.

`scipy.io.loadmat(..., simplify_cells=True)` squeezes singleton dimensions;
every array is reshaped back to its MATLAB shape before translation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

import matlab_compat as mc
from mtdr.stats import SufficientStats, sufficient_statistics

FloatArray = NDArray[np.float64]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MMLE_FIXTURE = FIXTURES / "mmle.mat"
SIM_FIXTURE = FIXTURES / "simulation.mat"
SVD_FIXTURE = FIXTURES / "svd.mat"
DATASETS = ("A", "B")


@dataclass
class Params:
    """One MMLE parameter point in the port's layouts."""

    S: list[FloatArray]
    lam: FloatArray
    b: FloatArray

    @property
    def ranks(self) -> list[int]:
        return [int(Sp.shape[1]) for Sp in self.S]


@dataclass
class Dataset:
    """One dataset of the fixture: data, statistics and the MATLAB outputs."""

    name: str
    n: int
    T: int
    N: int
    maxrank: int
    Y: FloatArray
    X: FloatArray
    mask: NDArray[np.bool_]
    stats: SufficientStats
    matlab_stats: dict[str, Any]
    points: list[dict[str, Any]] = field(default_factory=list)
    fits: list[dict[str, Any]] = field(default_factory=list)
    search: dict[str, Any] = field(default_factory=dict)
    true_ranks: list[int] = field(default_factory=list)
    svd_ranks: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------- layouts


def offsets(ranks: Sequence[int]) -> list[int]:
    """`c[p] = sum(ranks[:p])`, `c[P] = rtot`."""
    c = [0]
    for r in ranks:
        c.append(c[-1] + int(r))
    return c


def unpack_s(s: FloatArray, T: int, ranks: Sequence[int]) -> list[FloatArray]:
    """The packed `s` (regressor by regressor, time fastest) to `(T, r_p)` blocks."""
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    c = offsets(ranks)
    if s.size != T * c[-1]:
        raise ValueError(
            f"s has {s.size} entries; ranks {list(ranks)} need {T * c[-1]}"
        )
    return [
        np.ascontiguousarray(s[T * c[p] : T * c[p + 1]].reshape(T, r, order="F"))
        for p, r in enumerate(ranks)
    ]


def pack_s(S: Sequence[FloatArray]) -> FloatArray:
    """Inverse of `unpack_s`."""
    return np.concatenate([np.asarray(Sp).ravel(order="F") for Sp in S])


def unpack_pars(v: FloatArray, n: int, T: int, ranks: Sequence[int]) -> Params:
    """`[lambda; s; vec(b)]` to `Params`."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    rtot = int(sum(ranks))
    L = n + T * rtot
    if v.size != L + n * T:
        raise ValueError(f"parameter vector has {v.size} entries; expected {L + n * T}")
    return Params(
        S=unpack_s(v[n:L], T, ranks),
        lam=v[:n].copy(),
        b=np.ascontiguousarray(v[L:].reshape(n, T)),
    )


def pack_pars(params: Params) -> FloatArray:
    """Inverse of `unpack_pars`."""
    return np.concatenate([params.lam, pack_s(params.S), params.b.ravel()])


def unpack_wt(Wt: Any, n: int, ranks: Sequence[int]) -> list[FloatArray]:
    """`Wt` (`rtot x n`, `EBpost_W_uneqvar`) to `W[p]`, `(n, r_p)`."""
    c = offsets(ranks)
    Wt = np.asarray(Wt, dtype=np.float64).reshape(c[-1], n, order="F")
    return [np.ascontiguousarray(Wt[c[p] : c[p + 1], :].T) for p in range(len(ranks))]


def ci_to_port(Ci: Any, n: int, rtot: int) -> FloatArray:
    """`Ci` (`rtot x rtot x n`) to `(n, rtot, rtot)`."""
    Ci = np.asarray(Ci, dtype=np.float64).reshape(rtot, rtot, n, order="F")
    return np.ascontiguousarray(np.moveaxis(Ci, 2, 0))


def _blocks(cells: Any, shapes: Sequence[tuple[int, int]]) -> list[FloatArray]:
    """A MATLAB cell of matrices, each reshaped to its MATLAB shape."""
    items = _list(cells)
    if len(items) != len(shapes):
        raise ValueError(f"expected {len(shapes)} cells; got {len(items)}")
    return [
        np.asarray(x, dtype=np.float64).reshape(shape, order="F")
        for x, shape in zip(items, shapes, strict=True)
    ]


def _vec(x: Any) -> FloatArray:
    return np.atleast_1d(np.asarray(x, dtype=np.float64)).reshape(-1)


def _ranks(x: Any) -> list[int]:
    return [int(r) for r in np.atleast_1d(np.asarray(x)).reshape(-1)]


def _columns(x: Any, rows: int) -> FloatArray:
    """A `rows x K` MATLAB matrix that loadmat may have squeezed to `(rows,)`."""
    return np.asarray(x, dtype=np.float64).reshape(rows, -1, order="F")


def _list(x: Any) -> list[Any]:
    """A MATLAB cell as a list (loadmat gives a list, an object array, or the item)."""
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray) and x.dtype == object:
        return list(x.reshape(-1))
    return [x]


# --------------------------------------------------------------------------- loading


def _data(name: str, raw_svd: dict[str, Any], sim: dict[str, Any]) -> tuple[Any, ...]:
    if name == "A":
        Z, X_full, hk = sim["Z"], sim["X"], sim["hk"]
    else:
        d = raw_svd[name]
        Z, X_full, hk = d["Z"], d["X"], d["hk"]
    X_full = np.asarray(X_full, dtype=np.float64)
    if not np.all(X_full[:, -1] == 1):
        raise ValueError("the reference's constant term must be the last column of X")
    mask = np.asarray(hk).astype(bool)
    Y = mc.fixture_responses(Z)
    # Unobserved entries are zero in Z; make them NaN so nothing reads them.
    Y = np.where(mask[:, :, None], Y, np.nan)
    X = np.ascontiguousarray(X_full[:, :-1])
    return Y, X, mask


def _point(pt: dict[str, Any], n: int, T: int, P: int) -> dict[str, Any]:
    ranks = _ranks(pt["r"])
    rtot = sum(ranks)
    s = _vec(pt["s"])
    S = unpack_s(s, T, ranks)
    return {
        "ranks": ranks,
        "g": float(pt["g"]),
        "S": S,
        "lam": _vec(pt["lam"]),
        "b": np.ascontiguousarray(np.asarray(pt["b"], dtype=np.float64).T),
        "S_matlab": _blocks(pt["Shat"], [(r, T) for r in ranks]),
        "xi_c": mc.fixture_xzetai_to_xi(np.asarray(pt["Xzetai_c"]), P, T),
        "zzi_c": _vec(pt["zzi_c"]),
        "nll_nllonly": float(pt["nll_nllonly"]),
        "nll_nllonly_raw": float(pt["nll_nllonly_raw"]),
        "nll_Sonly": float(pt["nll_Sonly"]),
        "grad_S": unpack_s(_vec(pt["grad_Sonly"]), T, ranks),
        "nll_lambonly": float(pt["nll_lambonly"]),
        "grad_lam": _vec(pt["grad_lambonly"]),
        "W": unpack_wt(pt["Wt"], n, ranks),
        "Ci_post": ci_to_port(pt["Ci_post"], n, rtot),
        "Ci": ci_to_port(pt["Ci"], n, rtot),
        "b_mmle": np.ascontiguousarray(np.asarray(pt["b_mmle"], dtype=np.float64).T),
        "ecme_next": unpack_pars(_vec(pt["ecme_parhat"]), n, T, ranks),
        "ecme_parerr": _vec(pt["ecme_parerr"]),
        "ecme_nll": _vec(pt["ecme_nll"]),
        "ecme_Q": _vec(pt["ecme_Q"]),
    }


def _fit(ft: dict[str, Any], n: int, T: int, P: int) -> dict[str, Any]:
    ranks = _ranks(ft["r"])
    rtot = sum(ranks)
    L = n + T * rtot + n * T
    iterates = _columns(ft["ecme_iterates"], L)
    ca = dict(ft["ca"])
    ca_pars = _columns(ca.pop("pars"), L)
    trace: dict[str, Any] = {
        k: _vec(v) for k, v in ca.items() if k != "fminunc_algorithm"
    }
    trace["fminunc_algorithm"] = str(ca["fminunc_algorithm"])
    trace["params"] = [
        unpack_pars(ca_pars[:, j], n, T, ranks) for j in range(ca_pars.shape[1])
    ]
    return {
        "ranks": ranks,
        "pars_svd": _vec(ft["pars_svd"]),
        "start": unpack_pars(_vec(ft["pars0"]), n, T, ranks),
        "ecme": unpack_pars(_vec(ft["ecme_pars"]), n, T, ranks),
        "ecme_iterates": [
            unpack_pars(iterates[:, j], n, T, ranks) for j in range(iterates.shape[1])
        ],
        "ecme_n_sweeps": int(ft["ecme_n_sweeps"]),
        "ecme_parerr": _vec(ft["ecme_parerr"]),
        "ecme_nll": _vec(ft["ecme_nll"]),
        "ecme_Q": _vec(ft["ecme_Q"]),
        "ca": trace,
        "final": unpack_pars(_vec(ft["pars_final"]), n, T, ranks),
        "pars_final": _vec(ft["pars_final"]),
        "S_matlab": _blocks(ft["Shat"], [(r, T) for r in ranks]),
        "S_matlab_makebhat": _blocks(ft["Shat_mb"], [(r, T) for r in ranks]),
        "lam_makebhat": _vec(ft["lambhat_mb"]),
        "xi_f": mc.fixture_xzetai_to_xi(np.asarray(ft["Xzetai_f"]), P, T),
        "zzi_f": _vec(ft["zzi_f"]),
        "W": unpack_wt(ft["Wt"], n, ranks),
        "Ci_post": ci_to_port(ft["Ci_post"], n, rtot),
        "Bhat": _blocks(ft["Bhat"], [(n, T)] * P),
        "What": _blocks(ft["What"], [(n, r) for r in ranks]),
        "Bhat_raw": _blocks(ft["Bhat_raw"], [(n, T)] * P),
        "What_raw": _blocks(ft["What_raw"], [(n, r) for r in ranks]),
        "nll_final": float(ft["nll_final"]),
        "aic": float(ft["aic"]),
    }


def _search(sr: dict[str, Any], n: int, T: int, P: int) -> dict[str, Any]:
    rhist = np.asarray(sr["rhist"]).astype(np.int64).reshape(-1, P)
    calls_ranks = np.asarray(sr["calls_ranks"]).astype(np.int64).reshape(-1, P)
    calls_pars = _list(sr["calls_pars"])
    parhist = _list(sr["parhist"])
    return {
        "rest0": _ranks(sr["rest0"]),
        "rest": _ranks(sr["rest"]),
        "rhist": rhist,
        "funhist": _vec(sr["funhist"]),
        # parhist{k} holds the estimate at rhist(k+1, :)
        "parhist": [
            unpack_pars(_vec(v), n, T, [int(x) for x in rhist[k + 1]])
            for k, v in enumerate(parhist)
        ],
        "calls_ranks": calls_ranks,
        "calls_values": _vec(sr["calls_values"]),
        "calls_params": [
            unpack_pars(_vec(v), n, T, [int(x) for x in calls_ranks[k]])
            for k, v in enumerate(calls_pars)
        ],
    }


def load(path: Path = MMLE_FIXTURE) -> dict[str, Any]:
    """Read the fixture and translate every array to the port's layouts.

    Returns a dict with one `Dataset` per name in `DATASETS` and the scalar
    metadata (`mmx_available`, `minfunc_options`, `fminunc_options`,
    `matlab_version`).
    """
    raw = loadmat(path, simplify_cells=True)
    raw_svd = loadmat(SVD_FIXTURE, simplify_cells=True)
    sim = loadmat(SIM_FIXTURE)
    out: dict[str, Any] = {
        "mmx_available": int(raw["mmx_available"]),
        "minfunc_options": dict(raw["minfunc_options"]),
        "fminunc_options": dict(raw["fminunc_options"]),
        "matlab_version": str(raw["matlab_version"]),
    }
    for name in DATASETS:
        d = raw[name]
        Y, X, mask = _data(name, raw_svd, sim)
        n, T, N = int(d["n"]), int(d["T"]), int(d["N"])
        P = X.shape[1]
        if Y.shape != (N, n, T):
            raise ValueError(
                f"dataset {name}: Y has shape {Y.shape}, expected {(N, n, T)}"
            )
        ds = Dataset(
            name=name,
            n=n,
            T=T,
            N=N,
            maxrank=int(d["maxrank"]),
            Y=Y,
            X=X,
            mask=mask,
            stats=sufficient_statistics(Y, X, mask),
            matlab_stats={
                "A": np.moveaxis(
                    np.asarray(d["Ai"], dtype=np.float64).reshape(P, P, n, order="F"),
                    2,
                    0,
                ),
                "zzi": _vec(d["zzi"]),
                "ni": np.asarray(d["ni"]).astype(np.int64).reshape(-1),
                "xi": mc.fixture_xzetai_to_xi(np.asarray(d["Xzetai0"]), P, T),
                "xbari": np.asarray(d["xbari"], dtype=np.float64).reshape(
                    n, P, order="F"
                ),
                "Ybar": np.asarray(d["Ybar"], dtype=np.float64)
                .reshape(T, n, order="F")
                .T,
            },
            true_ranks=_ranks(d["rP"]),
            svd_ranks=_ranks(d["svd_ranks"]),
        )
        ds.points = [_point(pt, n, T, P) for pt in _list(d["points"])]
        ds.fits = [_fit(ft, n, T, P) for ft in _list(d["fits"])]
        ds.search = _search(d["search"], n, T, P)
        out[name] = ds
    return out
