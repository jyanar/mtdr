"""Exact replicas of reference behaviour that the package corrects.

Test helpers, not package code. Each function takes the port's layouts
(`Y[k, i, t]`, `X[k, p]` without the constant column, `mask[k, i]`,
per-regressor `(n, T)` coefficient lists) and reproduces, inside the function,
the reference's own index arithmetic, defects included, so that a parity test
can compare it with the MATLAB output. The MATLAB-ordered arrays built here
(`wsvd`, `Yn{i}`, `allstim{i}`, `kronmult` products) never leave these
functions. The `fixture_*` helpers translate fixture arrays to the port's
layouts at the boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _full_design(X: FloatArray, condition_independent: bool) -> FloatArray:
    """The reference's `X` with the constant term as the **last** column."""
    if not condition_independent:
        return np.asarray(X, dtype=np.float64)
    return np.column_stack([X, np.ones(X.shape[0])])


def _kronmult_eye_stim(Bi: FloatArray, stim: FloatArray, T: int) -> FloatArray:
    """`kronmult({speye(T), allstim{i}}, Bi)`: `vec(reshape(Bi, T, P) * stim')`.

    Time-fastest ordering: entry `t + T * j` is the prediction for bin `t` of
    the neuron's `j`-th observed trial.
    """
    P = stim.shape[1]
    M = Bi.reshape(T, P, order="F")
    return np.asarray((M @ stim.T).ravel(order="F"))


def _vec_yn_transpose(Y_i: FloatArray) -> FloatArray:
    """`vec(Yn{i}')` for `Yn{i}` the `T x n_i` responses: trial-fastest ordering.

    `Y_i` is the port's `(n_i, T)` block `Y[mask[:, i], i, :]`, which is
    `Yn{i}'`; its column-major `vec` puts the trial index fastest.
    """
    return np.asarray(Y_i.ravel(order="F"))


def svd_aic_reference(
    Y: FloatArray,
    X: FloatArray,
    mask: NDArray[np.bool_],
    B: Sequence[FloatArray],
    ranks: Sequence[int],
    *,
    condition_independent: bool = True,
) -> float:
    """Replica of `SVDRegB_AIC(Yn, allstim, wsvd, r)`, (M20a)-(M20c).

    All four defects of `SVDRegB_AIC.m` (lines 6-15) are reproduced: the
    `reshape(pars, n, [])` row extraction of the `T x n x P` array (which mixes
    neurons), the trial-fastest minus time-fastest residual, the precision
    `P * T / RSS` (regressor count), the `+ n_i T log(lambda)` sign, and the
    count `(nP + TP - sum(r)) * sum(r)`.

    Parameters
    ----------
    Y, X, mask
        Port layouts; `X` without the constant column.
    B
        Per-regressor `(n, T)` coefficients, the intercept **last** when
        `condition_independent` (the reference's `wsvd(:, :, p)'`).
    ranks
        The reference's `r`, including the constant term's rank.
    """
    X_full = _full_design(X, condition_independent)
    n = Y.shape[1]
    T = Y.shape[2]
    P = X_full.shape[1]
    if len(B) != P or len(ranks) != P:
        raise ValueError("B and ranks need one entry per column of the full design")
    wsvd = np.stack([np.asarray(b).T for b in B], axis=2)  # T x n x P
    Bhat = wsvd.reshape(n, -1, order="F")  # reshape(pars, n, []), line 6: (M20a)
    negloglik = 0.0
    for i in range(n):
        rows = np.asarray(mask[:, i], dtype=bool)
        stim = X_full[rows]  # allstim{i}
        Bi = Bhat[i, :]
        ri = _vec_yn_transpose(Y[rows, i, :]) - _kronmult_eye_stim(Bi, stim, T)
        rr = float(ri @ ri)
        lam = stim.shape[1] * T / rr  # size(allstim{ii}, 2) * T, line 11
        negloglik += rr * lam + Y[rows, i, :].size * np.log(lam)  # sign, line 12
    total = int(sum(ranks))
    K = (n * P + T * P - total) * total  # line 15
    return float(negloglik + 2 * K)


def svd_lambda_reference(
    Y: FloatArray,
    X: FloatArray,
    mask: NDArray[np.bool_],
    B: Sequence[FloatArray],
    *,
    condition_independent: bool = True,
) -> FloatArray:
    """Replica of the precision in `SVDRegress_S_Vdata` (lines 9-17), (M19a).

    Unlike `SVDRegB_AIC`, the row extraction is right (`B = [B1' ... BP']`) and
    the count is `n_i * T`; the residual is the misaligned one (trial-fastest
    minus time-fastest, line 15).
    """
    X_full = _full_design(X, condition_independent)
    n = Y.shape[1]
    T = Y.shape[2]
    Bmat = np.concatenate([np.asarray(b) for b in B], axis=1)  # n x TP
    lam = np.empty(n)
    for i in range(n):
        rows = np.asarray(mask[:, i], dtype=bool)
        stim = X_full[rows]
        ri = _vec_yn_transpose(Y[rows, i, :]) - _kronmult_eye_stim(Bmat[i], stim, T)
        lam[i] = stim.shape[0] * T / float(ri @ ri)
    return lam


def est_rank_greedily_reference(
    objective: Callable[[Any, NDArray[np.int64]], float],
    estfun: Callable[[NDArray[np.int64]], Any],
    r0: Sequence[int],
    maxrank: int,
    stepthresh: float = 0.0,
) -> tuple[NDArray[np.int64], NDArray[np.int64], FloatArray, list[Any]]:
    """Replica of `EstRankGreedily`'s bookkeeping, (M39) as the code runs it.

    Reproduces the acceptance rule `min dObj <= stepthresh` (an equal score is
    accepted), ties to the first movable index, the line-67
    `parhist{iters} = parhat{indmin}` indexing defect (`indmin` indexes the
    movable set, while `parhat` is indexed by regressor; `parhat` keeps entries
    from earlier rounds), and the `iters == 0` special case.
    Returns `(rest, rhist, FunHist, parhist)`; no file is written.
    """
    rest = np.array(r0, dtype=np.int64)
    P = rest.size
    rhist = [rest.copy()]
    parhat0 = estfun(rest.copy())
    obj0 = float(objective(parhat0, rest.copy()))
    funhist = [obj0]
    parhat: dict[int, Any] = {}
    parhist: list[Any] = []
    iters = 0
    while np.all(rest <= maxrank):
        obj = np.zeros(P)
        indmove = [p for p in range(P) if rest[p] < maxrank]
        for p in indmove:
            rtest = rest.copy()
            rtest[p] += 1
            parhat[p] = estfun(rtest.copy())
            obj[p] = float(objective(parhat[p], rtest.copy()))
        dobj = obj - obj0
        if all(dobj[p] > stepthresh for p in indmove):  # all([]) is true
            break
        indmin = int(np.argmin(dobj[indmove]))
        rest[indmove[indmin]] += 1
        obj0 = float(obj[indmove[indmin]])
        funhist.append(obj0)
        rhist.append(rest.copy())
        iters += 1
        parhist.append(parhat.get(indmin))  # the defect: indmin, not indmove[indmin]
    if iters == 0:
        parhist = [parhat0]
        funhist = [obj0]
        rhist = [np.array(r0, dtype=np.int64)]
    return rest, np.stack(rhist), np.array(funhist), parhist


# --------------------------------------------------------------------------- fixtures


def fixture_wsvd_to_b(wsvd: FloatArray) -> list[FloatArray]:
    """`wsvd` (`T x n x P`) to per-regressor `(n, T)`: `B[p] = wsvd[:, :, p].T`."""
    return [np.ascontiguousarray(wsvd[:, :, p].T) for p in range(wsvd.shape[2])]


def fixture_w0_to_b(w0: FloatArray, n: int, T: int) -> list[FloatArray]:
    """`XX \\ XY` (index `t + T*i + T*n*p`, 0-based) to per-regressor `(n, T)`."""
    blocks = np.asarray(w0, dtype=np.float64).reshape(-1, n, T)
    return [np.ascontiguousarray(b) for b in blocks]


def fixture_xzetai_to_xi(Xzetai: FloatArray, P: int, T: int) -> FloatArray:
    """`Xzetai` (`TP x n`, time fastest) to the port's `xi[i, p, t]`, (M10)."""
    n = Xzetai.shape[1]
    return np.ascontiguousarray(
        np.transpose(Xzetai.reshape(T, P, n, order="F"), (2, 1, 0))
    )


def fixture_responses(Z: FloatArray) -> FloatArray:
    """`Z` (`n x T x N`) to `Y[k, i, t]`."""
    return np.ascontiguousarray(
        np.transpose(np.asarray(Z, dtype=np.float64), (2, 0, 1))
    )


# ---------------------------------------------------------------------- MMLE vectors


def fixture_pars_to_mmle(
    pars: FloatArray,
    n: int,
    T: int,
    ranks: Sequence[int],
    *,
    condition_independent: bool = True,
) -> tuple[FloatArray, list[FloatArray], FloatArray | None]:
    """Unpack the reference's MMLE vector `[lambda; s; vec(b)]`, (M28).

    `docs/model.md` § 1.2: `lam = pars[:n]`; the bases segment
    `s = pars[n : n + T * rtot]` holds, for each regressor in turn, its `r_p`
    time courses one after the other (`S[p] = s[T c_p : T c_{p+1}]` reshaped
    `(T, r_p)` in column-major order, MATLAB's `reshape(., T, r_p)`); the
    intercept `b = pars[n + T * rtot :]` reshaped `(n, T)` in C order, which is
    MATLAB's `reshape(., T, n)'`. The length is checked: `n + T rtot + n T`,
    or `n + T rtot` without the intercept. A `parhist{k}` entry pairs with
    `rhist(k + 1, :)` (`docs/model.md` § 6.2); pass those ranks.
    """
    v = np.asarray(pars, dtype=np.float64).ravel()
    rtot = int(sum(ranks))
    expected = n + T * rtot + (n * T if condition_independent else 0)
    if v.size != expected:
        raise ValueError(
            f"pars has {v.size} entries; n + T * rtot (+ n T) = {expected} for "
            f"ranks {list(ranks)}"
        )
    lam = v[:n].copy()
    s = v[n : n + T * rtot]
    offsets = np.concatenate([[0], np.cumsum(ranks)]).astype(int)
    S = [
        s[T * offsets[p] : T * offsets[p + 1]].reshape(T, int(r), order="F").copy()
        for p, r in enumerate(ranks)
    ]
    b = v[n + T * rtot :].reshape(n, T).copy() if condition_independent else None
    return lam, S, b


def mmle_to_pars(
    lam: FloatArray, S: Sequence[FloatArray], b: FloatArray | None
) -> FloatArray:
    """Pack `(lambda, S, b)` as the reference's `[lambda; s; vec(b)]`, (M28)."""
    parts = [np.asarray(lam, dtype=np.float64).ravel()]
    parts += [np.asarray(s, dtype=np.float64).ravel(order="F") for s in S]
    if b is not None:
        parts.append(np.asarray(b, dtype=np.float64).ravel(order="C"))
    return np.concatenate(parts)
