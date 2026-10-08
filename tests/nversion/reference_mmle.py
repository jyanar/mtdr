r"""Line-by-line port of the reference's MMLE files: the N-version oracle.

This module is the second, independent implementation of the marginal
likelihood that `mtdr.mmle` is checked against. It is ported from the MATLAB
reference (`mTDRdemo/functionFiles/*.m`) only, not from `docs/model.md`:
every function names the file and lines it ports, keeps MATLAB's variable
names in comments, and keeps the reference's order of operations, including
its stale-variable updates. Per-neuron Python loops are deliberate: this is a
correctness oracle, not a fast implementation. It imports nothing from
`mtdr.mmle`; the input statistics are a `mtdr.stats.SufficientStats`.

Layouts and the translation to the reference's arrays
-----------------------------------------------------
* `S` is a list of `P` arrays `S[p]` of shape `(T, r_p)`. The reference packs
  them in `s` (component-major, time fastest) and builds the block-diagonal
  `rtot x TP` matrix `S = blkdiag(S_1', ..., S_P')` (`S_nllonly.m:41-47`,
  `Sonly.m:38-39`); block `p` sits in rows `c[p]:c[p+1]` and columns
  `p*T:(p+1)*T`, so `Sblk[c[p] + l, p*T + t] = S[p][t, l]` (`_block_s`).
* The "TP" axis is flattened `p*T + t` here, which is MATLAB's 1-based
  `t + T*(p-1)`: `Xzetai(:, i)` is `xi[i].reshape(P*T)`, and
  `kron(A_i, I_T)` (what the files' comments call `kron(I, Ai)`) is
  `np.kron(A[i], np.eye(T))`.
* `lam` is `(n,)`, `b` is `(n, T)` (MATLAB `b` is `T x n`, its transpose).

Statistics
----------
`ECMEsuffstat(zetai, Xi, b)` returns `Xzetai(:, i) = vec((Z_i - b_i) X_i)` and
`zzi(i) = ||Z_i - b_i||^2` over neuron `i`'s observed trials, and
`Ri = Xzetai(:, i) Xzetai(:, i)'`. These are exactly
`stats.centered(b)` (`xi(b)` and `upsilon(b)`; the outer product `Ri` is
formed where a file uses it). `MkSuffStatsBTDR_IncompObs_uneqvar_S_fast`'s
uncentred `Xzetai0` and `zzi` are `stats.XtY_raw` and `stats.YtY_raw` (what
`centered(None)` returns), `Ai` is `stats.XtX`, `ni` is `stats.n_obs`, and
the `mTDRdemo.m:126-132` loop's `xbari`, `Ybar` are `stats.X_mean`,
`stats.Y_mean.T`. `test_svd_parity.py` and `test_reference_mmle_vs_matlab.py`
check these correspondences on the fixtures.

Constants
---------
All three `neglogLik*` files return
`.5*(-T*ni*log(lambi) + logdetterm + zzi'*lambi - Qterm + regterm)`, the
negative log of the Gaussian marginal density of the observed responses
(weights `w_i ~ N(0, I)` integrated out) **without** its normalising term.
Writing `Sigma_i = lam_i^{-1} I + (X_i (x) I) S' S (X_i (x) I)'` for neuron
`i`'s `T n_i`-dimensional marginal covariance, the matrix determinant lemma
gives `log det Sigma_i = -T n_i log lam_i + log det C_i` and Woodbury gives
`z' Sigma_i^{-1} z = lam_i zz_i - lam_i^2 (S Xzeta_i)' C_i^{-1} (S Xzeta_i)`,
so the full negative log-density is the reference's value plus
`0.5 * T * sum_i n_i * log(2*pi)`. That is the one term this module adds
(`matlab_constants_only=False`, the default); with
`matlab_constants_only=True` the reference's own value is returned (the same
terms in the same order; agreement with MATLAB is measured in
`test_reference_mmle_vs_matlab.py`). Nothing else is dropped: the
`-T n_i log lam_i` and `log det C_i` terms are already present. The ridge
`regterm` is reproduced as each file computes it (see `nll_grad_S` for the
slicing defect) and is not part of the normalisation.

Reference paths
---------------
The `mmx_mkl_single` MEX is not shipped; every `try mmx_mkl_single(...)` in
the reference falls through to its `catch` branch (`slowMult`,
`slowBackslash`, `slowChol`), which this module follows. The unreachable
Hessian branch of `Sonly.m` (lines 118-151, calling helpers absent from the
reference) is not ported.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mtdr.stats import SufficientStats

__all__ = [
    "aic_mmle",
    "ecme",
    "ecme_step",
    "feature_precision",
    "nll",
    "nll_grad_S",
    "nll_grad_lam",
    "nll_terms",
    "normalising_constant",
    "posterior_W",
    "update_b",
]

FloatArray = NDArray[np.float64]

LOG_2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- helpers


def _ranks(S: Sequence[FloatArray]) -> list[int]:
    return [int(np.asarray(Sp).shape[1]) for Sp in S]


def _offsets(ranks: Sequence[int]) -> list[int]:
    """`c[p] = sum(ranks[:p])`, `c[P] = rtot`."""
    c = [0]
    for r in ranks:
        c.append(c[-1] + int(r))
    return c


def _block_s(S: Sequence[FloatArray], T: int) -> FloatArray:
    """The reference's block-diagonal `S` (`rtot x TP`).

    `S = blkdiag(mat2cell(reshape(s,T,rtot)',r,T){:})` (`Sonly.m:38-39`,
    `ECMEtdr.m:49-50`): block `p` is `S_p'` (`r_p x T`) in rows `c[p]:c[p+1]`
    and columns `p*T:(p+1)*T`.
    """
    ranks = _ranks(S)
    P = len(S)
    c = _offsets(ranks)
    Sblk = np.zeros((c[-1], P * T))
    for p, Sp in enumerate(S):
        Sp = np.asarray(Sp, dtype=np.float64)
        if Sp.shape[0] != T:
            raise ValueError(f"S[{p}] must have {T} rows (bins); got {Sp.shape}")
        Sblk[c[p] : c[p + 1], p * T : (p + 1) * T] = Sp.T
    return Sblk


def _unblock_s(M: FloatArray, ranks: Sequence[int], T: int) -> list[FloatArray]:
    """`keepActive_S` (`keepActive_S.m:8-17`) returned in the `(T, r_p)` layout.

    Block `p` is `M(rowind, colind)` with rows `c[p]:c[p+1]` and columns
    `p*T:(p+1)*T`; the reference stacks `vec(Sp')`, i.e. `S[p] = Sp'`.
    """
    c = _offsets(ranks)
    return [
        np.ascontiguousarray(M[c[p] : c[p + 1], p * T : (p + 1) * T].T)
        for p in range(len(ranks))
    ]


def _s_vector(S: Sequence[FloatArray]) -> FloatArray:
    """The packed `s` (`pars(n+1:n+rtot*T)`)."""
    parts = [np.asarray(Sp, dtype=np.float64).ravel(order="F") for Sp in S]
    return np.concatenate(parts) if parts else np.zeros(0)


def _kron_ai(A_i: FloatArray, T: int) -> FloatArray:
    """`kron(Ai, I_T)` on the flattened `p*T + t` axis (`kronmult({I_T, Ai}, .)`)."""
    return np.asarray(np.kron(A_i, np.eye(T)), dtype=np.float64)


def _check(
    S: Sequence[FloatArray], lam: FloatArray, stats: SufficientStats
) -> tuple[int, int, int, FloatArray]:
    n, P, T = stats.n_neurons, stats.n_regressors, stats.n_bins
    if len(S) != P:
        raise ValueError(f"S needs one block per regressor ({P}); got {len(S)}")
    lam_arr = np.asarray(lam, dtype=np.float64).reshape(-1)
    if lam_arr.shape != (n,):
        raise ValueError(f"lam must have shape ({n},); got {lam_arr.shape}")
    return n, P, T, lam_arr


def _suffstat(
    stats: SufficientStats, b: FloatArray | None
) -> tuple[FloatArray, FloatArray]:
    """`[Ri, zzi, Xzetai] = ECMEsuffstat(zetai, Xi, b)` (`ECMEsuffstat.m:9-19`).

    Returns `Xzetai` as rows `(n, TP)` (row `i` is MATLAB column `i`) and `zzi`
    `(n,)`; `Ri(:, :, i)` is `np.outer(Xzetai[i], Xzetai[i])`. `b = None`
    gives the uncentred `Xzetai0`, `zzi` of
    `MkSuffStatsBTDR_IncompObs_uneqvar_S_fast.m:35-43`.
    """
    XtY, YtY = stats.centered(None if b is None else np.asarray(b, dtype=np.float64))
    n = stats.n_neurons
    Xzetai = np.ascontiguousarray(XtY.reshape(n, -1))  # [i, p*T + t]
    return Xzetai, np.asarray(YtY, dtype=np.float64)


def _ci(Sblk: FloatArray, lam_i: float, A_i: FloatArray, T: int) -> FloatArray:
    """`Ci := lambi*S*kron(I,Ai)*S' + eye(rtot)` for one neuron.

    `S_nllonly.m:59-74`: `lambAiIS = lambi*kron(Ai, I_T)*S'` (`TP x rtot`),
    `lambiSAiSold = S*lambAiIS`, `Ci = lambiSAiSold + I`.
    """
    lambAiIS = lam_i * (_kron_ai(A_i, T) @ Sblk.T)
    return np.asarray(Sblk @ lambAiIS + np.eye(Sblk.shape[0]))


def _logdet_chol(Ci: FloatArray) -> float:
    """`logdetterm` contribution `2*sum(log(diag(chol(Ci))))` (`S_nllonly.m:88-95`)."""
    cholCi = np.linalg.cholesky(Ci)  # slowChol: chol(Ci) (upper); same diagonal
    return float(2.0 * np.sum(np.log(np.diag(cholCi))))


def normalising_constant(stats: SufficientStats) -> float:
    r"""The term the reference drops: $\tfrac12\,T\sum_i n_i\log 2\pi$."""
    return 0.5 * stats.n_bins * float(np.sum(stats.n_obs)) * LOG_2PI


# --------------------------------------------------------------------------- likelihood


def nll(
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray | None,
    stats: SufficientStats,
    *,
    g: float = 0.0,
    matlab_constants_only: bool = False,
) -> float:
    """Marginal NLL, `neglogLikBTDR_IncompObs_uneqvar_S_nllonly.m` (lines 32-101).

    The weights are integrated out under `w_i ~ N(0, I_rtot)` with `S`, `lam`
    and `b` fixed. Per neuron, with `Xzetai`, `zzi` from `ECMEsuffstat` at `b`:

    * `Ci = lambi*S*kron(Ai,I)*S' + I` (lines 59-74),
    * `SXzetai = S*Xzetai`, `Qtermi = SXzetai'*(Ci\\SXzetai)` (lines 77-85),
    * `logdetterm = 2*sum(log(diag(chol(Ci))))` (lines 88-95),
    * `negloglik = .5*(-T*ni*log(lambi) + logdetterm + zzi'*lambi - Qterm +
      regterm)` with `Qterm = (lambi.^2)'*Qtermi` and
      `regterm = g*pars(n+1:end)'*pars(n+1:end)`, where `pars = [lambi; s]`,
      so `regterm = g*||s||^2` (lines 98-101).

    Parameters
    ----------
    S, lam, b, stats
        Bases `(T, r_p)` per regressor, precisions `(n,)`, intercept `(n, T)`
        (`None`: the uncentred statistics, as `mTDRdemo.m:155` uses them),
        and the sufficient statistics.
    g
        The reference's ridge argument (`0` everywhere in the demo).
    matlab_constants_only
        `True` returns the reference's value; `False` (default) adds
        `normalising_constant(stats)`, the only term it drops.
    """
    n, _P, T, lam = _check(S, lam, stats)
    Sblk = _block_s(S, T)
    Xzetai, zzi = _suffstat(stats, b)
    A = stats.XtX
    ni = stats.n_obs.astype(np.float64)
    logdetterm = 0.0
    Qtermi = np.zeros(n)
    for i in range(n):
        Ci = _ci(Sblk, float(lam[i]), A[i], T)
        SXzetai = Sblk @ Xzetai[i]  # S*Xzetai(:, i)
        invCiSXzetai = np.linalg.solve(Ci, SXzetai)  # slowBackslash
        Qtermi[i] = SXzetai @ invCiSXzetai  # slowMult(SXzetai', invCiSXzetai)
        logdetterm += _logdet_chol(Ci)
    Qterm = float((lam**2) @ Qtermi)
    s = _s_vector(S)
    regterm = g * float(s @ s)  # pars(n+1:end) is s
    value = 0.5 * (
        -T * float(ni @ np.log(lam)) + logdetterm + float(zzi @ lam) - Qterm + regterm
    )
    if not matlab_constants_only:
        value += normalising_constant(stats)
    return value


def nll_terms(
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray | None,
    stats: SufficientStats,
) -> FloatArray:
    """Per-neuron terms of `nll` at `g = 0`, without the constant, `(n,)`.

    Added at integration (not part of the line-by-line port): the reference
    sums its per-neuron quantities with inner products (`-T*ni*log(lambi)`,
    `zzi'*lambi`, `(lambi.^2)'*Qtermi`, `logdetterm` accumulated over the
    `slowChol` loop, `S_nllonly.m:88-101`), so its value is
    `sum(nll_terms(...))`; this returns the summands before the sums, neuron
    `i`'s `.5*(-T*ni(i)*log(lambi(i)) + 2*sum(log(diag(chol(Ci)))) +
    zzi(i)*lambi(i) - lambi(i)^2*Qtermi(i))`, each built exactly as in `nll`.
    """
    n, _P, T, lam = _check(S, lam, stats)
    Sblk = _block_s(S, T)
    Xzetai, zzi = _suffstat(stats, b)
    A = stats.XtX
    ni = stats.n_obs.astype(np.float64)
    terms = np.zeros(n)
    for i in range(n):
        Ci = _ci(Sblk, float(lam[i]), A[i], T)
        SXzetai = Sblk @ Xzetai[i]
        Qtermi = float(SXzetai @ np.linalg.solve(Ci, SXzetai))
        terms[i] = 0.5 * (
            -T * ni[i] * np.log(lam[i])
            + _logdet_chol(Ci)
            + zzi[i] * lam[i]
            - lam[i] ** 2 * Qtermi
        )
    return terms


def nll_grad_S(  # noqa: N802 - the brief's name
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray | None,
    stats: SufficientStats,
    *,
    g: float = 0.0,
    matlab_constants_only: bool = False,
) -> tuple[float, list[FloatArray]]:
    """NLL and its gradient in `S`, `neglogLikBTDR_IncompObs_uneqvar_Sonly.m`.

    Value (lines 41-80): as `nll`, built with `S2 = reshape(S,[],P)`,
    `lambAiIS = permute(reshape(S2*lambAi,rtot,TP,n),[2 1 3])`
    (`= lambi*kron(Ai,I)*S'`), `Ci = S*lambAiIS + I`,
    `invCiSRi = Ci\\(S*Ri)`, `Qtermi = <invCiSRi, S>`.

    Gradient (lines 83-115), per neuron:

    * `lambinvCiSAiI = Ci\\(lambi*S*kron(Ai,I))` (`rtot x TP`, line 85),
    * `lambSinvCiSAiI = S'*lambinvCiSAiI` (`TP x TP`, line 90),
    * `M = lambi^2*(I_TP - lambSinvCiSAiI)` (lines 91-92),
    * `dLdS = sum_i (lambinvCiSAiI - invCiSRi*M) + g*S` (lines 95-99),

    and only the diagonal blocks are kept (lines 102-112, `keepActive_S`), so
    `grad[p]` is block `p` of `dLdS` transposed to `(T, r_p)`.

    Defect reproduced: the value's ridge is
    `regterm = g*pars(n+1:end)'*pars(n+1:end)` (line 78) where `pars` is `s`
    alone, so it penalises `s[n:]` (nothing when `n >= rtot*T`), while the
    gradient adds `g*S` (line 99), the gradient of `0.5*g*||s||^2`. With
    `g != 0` the returned gradient is therefore that of `nll(..., g=g)`, not of
    the value returned here. Every caller in the reference passes `g = 0`.

    Returns
    -------
    value : float
        The reference's value, plus `normalising_constant(stats)` unless
        `matlab_constants_only`.
    grad : list of numpy.ndarray
        `dL/dS_p`, shape `(T, r_p)` per regressor.
    """
    n, P, T, lam = _check(S, lam, stats)
    ranks = _ranks(S)
    rtot = sum(ranks)
    TP = T * P
    Sblk = _block_s(S, T)
    Xzetai, zzi = _suffstat(stats, b)
    A = stats.XtX
    ni = stats.n_obs.astype(np.float64)
    logdetterm = 0.0
    Qtermi = np.zeros(n)
    dLdS = np.zeros((rtot, TP))
    for i in range(n):
        Ri = np.outer(Xzetai[i], Xzetai[i])
        lambAiIS = lam[i] * (_kron_ai(A[i], T) @ Sblk.T)  # TP x rtot
        Ci = Sblk @ lambAiIS + np.eye(rtot)
        SRi = Sblk @ Ri
        invCiSRi = np.linalg.solve(Ci, SRi)  # rtot x TP
        Qtermi[i] = float(np.sum(invCiSRi * Sblk))  # ...*sparse(vec(S))
        logdetterm += _logdet_chol(Ci)
        lambinvCiSAiI = np.linalg.solve(Ci, lambAiIS.T)  # Ci\permute(lambAiIS,[2 1 3])
        lambSinvCiSAiI = Sblk.T @ lambinvCiSAiI  # TP x TP
        M = (np.eye(TP) - lambSinvCiSAiI) * lam[i] ** 2
        dLdS += lambinvCiSAiI - invCiSRi @ M
    dLdS = dLdS + g * Sblk
    Qterm = float((lam**2) @ Qtermi)
    s = _s_vector(S)
    s_tail = s[n:]  # pars(n+1:end) of the s-only vector: the slicing defect
    regterm = g * float(s_tail @ s_tail)
    value = 0.5 * (
        -T * float(ni @ np.log(lam)) + logdetterm + float(zzi @ lam) - Qterm + regterm
    )
    if not matlab_constants_only:
        value += normalising_constant(stats)
    return value, _unblock_s(dLdS, ranks, T)


def nll_grad_lam(
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray | None,
    stats: SufficientStats,
    *,
    matlab_constants_only: bool = False,
) -> tuple[float, FloatArray]:
    """NLL and its gradient in `lam`, `neglogLikBTDR_IncompObs_uneqvar_lambonly.m`.

    Value (lines 41-90): `Ci = S*lambAiIS + I` assembled over components
    (lines 42-52, 71-74), `invCiSRi = (Ci\\S)*Ri` (lines 77-78),
    `Qtermi = <invCiSRi, S>` (line 81), `logdetterm` from `chol` (lines
    82-83). Here `pars` is `lambi` alone, so
    `regterm = g*pars(n+1:end)'*pars(n+1:end)` (line 88) is always `0`; the
    argument `g` is therefore not offered.

    Gradient (lines 92-116), per neuron:

    * `invCiSAiI = (Ci\\(lambi*S*kron(Ai,I)))/lambi` (lines 94, 99),
    * `SinvCiSRi = S'*invCiSRi` (line 101), `F = I + lambi^2*SinvCiSRi`
      (lines 102-104), `F = invCiSAiI*F` (line 106),
    * `dQdlambi = <F, S>` (line 111),
    * `dLdlambi = .5*(-T*ni/lambi + zzi - 2*lambi*Qtermi + dQdlambi)`
      (lines 112-113).

    The objective separates over neurons: entry `i` of the gradient depends on
    `lam[i]` only.
    """
    n, P, T, lam = _check(S, lam, stats)
    rtot = sum(_ranks(S))
    TP = T * P
    Sblk = _block_s(S, T)
    Xzetai, zzi = _suffstat(stats, b)
    A = stats.XtX
    ni = stats.n_obs.astype(np.float64)
    logdetterm = 0.0
    Qtermi = np.zeros(n)
    dQdlambi = np.zeros(n)
    for i in range(n):
        Ri = np.outer(Xzetai[i], Xzetai[i])
        lambAiIS = lam[i] * (_kron_ai(A[i], T) @ Sblk.T)  # TP x rtot
        Ci = Sblk @ lambAiIS + np.eye(rtot)  # slowMult(repmat(S,1,1,n), lambAiIS) + I
        invCiSRi = np.linalg.solve(Ci, Sblk) @ Ri  # slowBackslash then slowMult
        Qtermi[i] = float(np.sum(invCiSRi * Sblk))
        logdetterm += _logdet_chol(Ci)
        lambinvCiSAiI = np.linalg.solve(Ci, lambAiIS.T)
        invCiSAiI = lambinvCiSAiI * (1.0 / lam[i])
        SinvCiSRi = Sblk.T @ invCiSRi  # TP x TP
        F = SinvCiSRi * lam[i] ** 2 + np.eye(TP)
        F = invCiSAiI @ F  # rtot x TP
        dQdlambi[i] = float(np.sum(F * Sblk))  # ...*vec(S')
    Qterm = float((lam**2) @ Qtermi)
    value = 0.5 * (-T * float(ni @ np.log(lam)) + logdetterm + float(zzi @ lam) - Qterm)
    if not matlab_constants_only:
        value += normalising_constant(stats)
    dLdlambi = 0.5 * (-T * ni / lam + zzi - 2.0 * lam * Qtermi + dQdlambi)
    return value, dLdlambi


# --------------------------------------------------------------------------- posterior


def feature_precision(
    S: Sequence[FloatArray], lam: FloatArray, stats: SufficientStats
) -> FloatArray:
    """`Ci` of every likelihood file, `I + lam_i S kron(A_i, I_T) S'`, `(n, r, r)`.

    The second output of `EBpost_W_uneqvar.m` (lines 11-12), i.e. the
    posterior **precision** of `w_i`; also the `Ci` that
    `Estpars_CoordAscent_lambi_S_b.m:31-35` passes to `MMLE_b`.
    """
    n, _P, T, lam = _check(S, lam, stats)
    Sblk = _block_s(S, T)
    A = stats.XtX
    return np.stack([_ci(Sblk, float(lam[i]), A[i], T) for i in range(n)])


def posterior_W(  # noqa: N802 - the brief's name
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray | None,
    stats: SufficientStats,
) -> tuple[list[FloatArray], FloatArray]:
    """Posterior of the weights, `EBpost_W_uneqvar.m` (lines 3-14).

    Per neuron: `AiIS = kronmult({I_T, Ai}, S')` (`kron(Ai,I)*S'`),
    `Ci = lambi*S*AiIS + I_r` (the posterior precision), and
    `Wt(:, i) = (Ci\\SXIZi(:, i))*lambi` with `SXIZi = S*Xzetai` (the
    posterior mean of `w_i`).

    `b` selects the `Xzetai` passed in: the centred `ECMEsuffstat(..., b)`
    one (as `demoLearning.m` does), or, with `b = None`, the uncentred
    `Xzetai0` that `mTDRdemo.m:155` passes.

    Returns
    -------
    W : list of numpy.ndarray
        Posterior means per regressor, `(n, r_p)`: `W[p] = Wt(c[p]:c[p+1], :)'`
        (`MakeBhat_data.m:36-37`).
    cov : numpy.ndarray
        Posterior covariance of `w_i`, `(n, rtot, rtot)`, `inv(Ci)`; the
        reference returns `Ci` itself (see `feature_precision`).
    """
    n, _P, T, lam = _check(S, lam, stats)
    ranks = _ranks(S)
    c = _offsets(ranks)
    rtot = c[-1]
    Sblk = _block_s(S, T)
    Xzetai, _zzi = _suffstat(stats, b)
    A = stats.XtX
    SXIZi = Xzetai @ Sblk.T  # rows: (S*Xzetai)(:, i)'
    Wt = np.zeros((rtot, n))
    cov = np.zeros((n, rtot, rtot))
    for i in range(n):
        AiIS = _kron_ai(A[i], T) @ Sblk.T
        Ci = lam[i] * (Sblk @ AiIS) + np.eye(rtot)
        Wt[:, i] = np.linalg.solve(Ci, SXIZi[i]) * lam[i]
        cov[i] = np.linalg.inv(Ci)
    W = [np.ascontiguousarray(Wt[c[p] : c[p + 1], :].T) for p in range(len(ranks))]
    return W, cov


# --------------------------------------------------------------------------- intercept


def _mmle_b(
    Ci: FloatArray,
    S: Sequence[FloatArray],
    lam: FloatArray,
    stats: SufficientStats,
) -> FloatArray:
    """`bhati = MMLE_b(Ci,S,lambi,ni,xbari,Ybar,Xzetai,r)` (`MMLE_b.m:1-26`).

    `Ci` is used as given (ECME passes an asymmetric one). `Xzetai` is the
    **uncentred** `Xzetai0` (every caller passes it), `xbari` and `Ybar` the
    means. Per neuron:

    * `XiS(:, rowind_p) = xbari(ii,p)*Sp'` (`T x rtot`, lines 11-21),
    * `CIXS = Ci(:,:,ii)\\XiS'` (line 22),
    * `A = eye(T) - lambi(ii)*ni(ii)*XiS*CIXS` (line 23),
    * `ybarhat = lambi(ii)*CIXS'*S*Xzetai(:,ii)` (line 24),
    * `bhati(:,ii) = A\\(Ybar(:,ii) - ybarhat)` (line 25).

    Returns `(n, T)` (MATLAB's `T x n` transposed).
    """
    n, _P, T, lam = _check(S, lam, stats)
    ranks = _ranks(S)
    c = _offsets(ranks)
    rtot = c[-1]
    Sblk = _block_s(S, T)
    Xzetai0, _zzi0 = _suffstat(stats, None)
    xbari = stats.X_mean
    Ybar = stats.Y_mean  # rows are MATLAB's Ybar(:, ii)
    ni = stats.n_obs.astype(np.float64)
    bhati = np.zeros((n, T))
    for ii in range(n):
        XiS = np.zeros((T, rtot))
        for p, Sp in enumerate(S):
            XiS[:, c[p] : c[p + 1]] = xbari[ii, p] * np.asarray(Sp)  # xbari*Sp'
        CIXS = np.linalg.solve(Ci[ii], XiS.T)  # rtot x T
        Amat = np.eye(T) - lam[ii] * ni[ii] * (XiS @ CIXS)
        ybarhat = lam[ii] * (CIXS.T @ (Sblk @ Xzetai0[ii]))
        bhati[ii] = np.linalg.solve(Amat, Ybar[ii] - ybarhat)
    return bhati


def update_b(
    S: Sequence[FloatArray], lam: FloatArray, stats: SufficientStats
) -> FloatArray:
    """The intercept M-step with a consistent `Ci`, `(n, T)`.

    `Estpars_CoordAscent_lambi_S_b.m:31-36`: `Ci = I + lambhat*S*kron(Ai,I)*S'`
    from the current `S` and `lam`, then `MMLE_b(Ci, S, lam, ...)` (see
    `_mmle_b`). The ECME b-step uses a different, mixed `Ci`
    (`ecme_step`).
    """
    return _mmle_b(feature_precision(S, lam, stats), S, lam, stats)


# --------------------------------------------------------------------------- ECME


def ecme_step(
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray,
    stats: SufficientStats,
) -> tuple[list[FloatArray], FloatArray, FloatArray]:
    """One sweep of `ECMEtdr.m`'s loop body (lines 41-155), stale variables kept.

    Input is the current iterate (`lambinew`, `news`, `bnew` of the previous
    sweep, or `pars0`); output the next `(S, lam, b)`. In order:

    1. Statistics at the current `b` (line 46): `[Ri,zzi,Xzetai] =
       ECMEsuffstat(zetai,Xi,bold)`.
    2. lambda-step (lines 49-89), with `Sold = Snew` the current `S` and
       `Ciold = lambiold*Sold*kron(Ai,I)*Sold' + I`:
       `BiSoldRi = Ciold\\(Sold*Ri)`; `g1i = 2*lambiold*<BiSoldRi, Snew>`;
       `g2i = trace(Ciold\\(Snew*kron(Ai,I)*Snew'))`;
       `g3i = lambiold^2*<RSBSASB', Sold>` with
       `RSBSASB = BiSoldRi'*BiSAS'`; `lambinew = T*ni'./(zzi - g1i + g2i + g3i)`.
    3. S-step (lines 92-143) and 4. b-step (lines 148-155), see the hazards.

    Hazards reproduced, quoted from `ECMEtdr.m`:

    * line 89, `lambiold = lambinew;` -- every later "old" lambda is the new
      one. Hence line 100, `bsxfun(@times,BiSoldRi,permute(lambinew.*lambiold,
      [3 2 1]))`, weights `M0` by `lambinew.^2`, and line 105,
      `lamb2oldSRSB_old = bsxfun(@times,SRSB_old,permute(lambiold.^2,[3,2,1]))`,
      by `lambinew.^2`.
    * lines 97-98 rebuild `Ciold` with the new lambda, but `BiSoldRi` (line
      66) is not recomputed: `M0 = sum_i lambinew_i^2 * Ciold(lambiold)\\(S*Ri)`
      and `Gi = Ciold(lambinew)\\(lambinew^2*Sold*BiSoldRi' + I)` (line 107)
      mix the inverses of `C` at the old and the new lambda.
    * lines 129-141, the assembly of `GG` from `Gammap{p}` (`Gammap{p}(:,
      block q) = sum_i lambinew_i*Ai(p,q)*Gi(:, block q)`):
      `GG(startind:endind,startind:rtot) = G(startind:endind,startind:rtot);`
      followed by
      `GG(startind:rtot,startind:endind) = G(startind:endind,startind:rtot)';`
      copies each upper block row to the lower triangle and so also replaces
      every **diagonal** block by its transpose; `Gi` is not symmetric, so
      neither is that block. `news = GG\\reshape(m0,T,rtot)'` (line 142) is a
      general solve.
    * lines 150-154: `S2 = reshape(S,[],P)` is from the **new** `S`, but
      `SAiISold = reshape(Sold*reshape(AiISold,TP,rtot*n),rtot,rtot,n)` uses
      `Sold`, so the `Ci` handed to `MMLE_b` is
      `I + lambinew*Sold*kron(Ai,I)*Snew'`, which is not symmetric (and
      `MMLE_b` uses `Ci\\` and `CIXS'`, so its transpose enters `ybarhat`).
    * line 111, `SSnew`, is computed and never used (not ported). Line 128,
      `G = cat(1,Gammap{:})`, is overwritten at line 131 (not ported).
    """
    n, P, T, lambiold = _check(S, lam, stats)
    ranks = _ranks(S)
    c = _offsets(ranks)
    rtot = c[-1]
    TP = T * P
    bold = np.asarray(b, dtype=np.float64)
    if bold.shape != (n, T):
        raise ValueError(f"b must have shape ({n}, {T}); got {bold.shape}")
    A = stats.XtX
    ni = stats.n_obs.astype(np.float64)
    Xzetai, zzi = _suffstat(stats, bold)  # line 46
    Snew = _block_s(S, T)  # lines 49-50
    Sold = Snew  # line 51

    # ---- M-step for lambda (lines 53-89)
    SAiISold = np.zeros((n, rtot, rtot))
    BiSoldRi = np.zeros((n, rtot, TP))
    g1i = np.zeros(n)
    g2i = np.zeros(n)
    g3i = np.zeros(n)
    for i in range(n):
        K = _kron_ai(A[i], T)
        AiISold = K @ Sold.T  # line 57, TP x rtot
        SAiISold[i] = Sold @ AiISold  # line 58
        Ciold = lambiold[i] * SAiISold[i] + np.eye(rtot)  # lines 59-60
        Ri = np.outer(Xzetai[i], Xzetai[i])
        SoldRi = Sold @ Ri  # line 64
        BiSoldRi[i] = np.linalg.solve(Ciold, SoldRi)  # line 68
        g1i[i] = 2.0 * lambiold[i] * float(np.sum(BiSoldRi[i] * Snew))  # line 70
        AiISnew = K @ Sold.T  # line 72: S2 is still reshape(Sold,[],P)
        SnewAiSnew = Snew @ AiISnew  # line 73
        BiSAS = np.linalg.solve(Ciold, SnewAiSnew)  # line 77
        g2i[i] = float(np.trace(BiSAS))  # line 80
        RSBSASB = BiSoldRi[i].T @ BiSAS.T  # line 84, TP x rtot
        g3i[i] = float(np.sum(RSBSASB.T * Sold))  # line 86: ...*sparse(vec(Sold))
    g3i = g3i * lambiold**2  # line 87
    lambinew = T * ni / (zzi - g1i + g2i + g3i)  # line 88
    lambiold = lambinew  # line 89 (hazard: the "old" lambda is now the new one)

    # ---- M-step for S (lines 92-143)
    M0 = np.zeros((rtot, TP))
    Gi = np.zeros((n, rtot, rtot))
    for i in range(n):
        Ciold = lambiold[i] * SAiISold[i] + np.eye(rtot)  # lines 97-98, new lambda
        M0 += (lambinew[i] * lambiold[i]) * BiSoldRi[i]  # lines 100-101, stale BiSoldRi
        SRSB_old = Sold @ BiSoldRi[i].T  # line 104
        lamb2oldSRSB_old = SRSB_old * lambiold[i] ** 2  # line 105
        Gi[i] = np.linalg.solve(Ciold, lamb2oldSRSB_old + np.eye(rtot))  # line 109
    m0 = _unblock_s(M0, ranks, T)  # line 102, keepActive_S (as (T, r_p) blocks)
    rhs = np.concatenate([m.T for m in m0], axis=0)  # reshape(m0,T,rtot)', rtot x T
    Gammap = []
    for p in range(P):  # lines 114-127
        gammapq = []
        for q in range(P):
            lambda_ipq = A[:, p, q] * lambinew  # Ai(p,q,:).*lambinew
            gammapq.append(
                np.einsum("ijk,i->jk", Gi[:, :, c[q] : c[q + 1]], lambda_ipq)
            )
        Gammap.append(np.concatenate(gammapq, axis=1))  # rtot x rtot
    GG = np.zeros((rtot, rtot))
    for p in range(P):  # lines 130-141 (p == 1 is the same with startind = 1)
        G = Gammap[p]
        startind, endind = c[p], c[p + 1]
        GG[startind:endind, startind:rtot] = G[startind:endind, startind:rtot]
        GG[startind:rtot, startind:endind] = G[startind:endind, startind:rtot].T
    news = np.linalg.solve(GG, rhs)  # line 142, rtot x T
    S_new = [np.ascontiguousarray(news[c[p] : c[p + 1], :].T) for p in range(P)]

    # ---- MMLE of the condition-independent term (lines 148-155)
    Sb = _block_s(S_new, T)  # line 149, S
    Ci_b = np.zeros((n, rtot, rtot))
    for i in range(n):
        AiISold = _kron_ai(A[i], T) @ Sb.T  # line 151: from the new S
        SAiISold_b = Sold @ AiISold  # line 152: Sold * kron(Ai,I) * Snew'
        Ci_b[i] = lambiold[i] * SAiISold_b + np.eye(rtot)  # lines 153-154
    bnew = _mmle_b(Ci_b, S_new, lambiold, stats)  # line 155
    return S_new, np.asarray(lambinew), bnew


def _parerr(
    new: tuple[Sequence[FloatArray], FloatArray, FloatArray],
    old: tuple[Sequence[FloatArray], FloatArray, FloatArray],
) -> float:
    """`max((newpars-oldpars).^2./oldpars.^2)` over `[lambda; s; vec(b)]` (line 161)."""
    newpars = np.concatenate([new[1], _s_vector(new[0]), new[2].ravel()])
    oldpars = np.concatenate([old[1], _s_vector(old[0]), old[2].ravel()])
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.max((newpars - oldpars) ** 2 / oldpars**2))


def ecme(
    stopmode: Literal["steps", "converge"],
    stopcrit: float,
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray,
    stats: SufficientStats,
    *,
    maxsteps: int = 100,
    matlab_constants_only: bool = False,
) -> tuple[list[FloatArray], FloatArray, FloatArray, FloatArray, FloatArray]:
    """`[parhat,Q,parerr,nll] = ECMEtdr(stopmode,stopcrit,pars0,...)` (lines 11-179).

    Runs `ecme_step` until the reference's stop rule fires: `'steps'` runs
    `stopcrit` sweeps (`stopcrit = stopcrit+1` at line 31, stop when
    `k >= stopcrit`); `'converge'` stops when `parerr(k-1) < stopcrit` or
    `k >= maxsteps` (`maxsteps = 100`, line 11), so at most 99 sweeps. The
    demo uses `'converge'` with `stopcrit = 1e-0`.

    `nll(1)` is the NLL at `pars0` with the statistics centred at `b0`
    (line 37); `nll(k)` is the NLL at the new `[lambda; s]` with the
    statistics centred at the new `b` (lines 167-168). `Q` is not ported.

    Returns
    -------
    S, lam, b
        The final iterate (`parhat`).
    parerr : numpy.ndarray
        One entry per sweep.
    nll_trace : numpy.ndarray
        `nll`, one entry more than `parerr`.
    """
    if stopmode not in ("steps", "converge"):
        raise ValueError(f"unknown stopmode {stopmode!r}")
    cur: tuple[list[FloatArray], FloatArray, FloatArray] = (
        [np.asarray(Sp, dtype=np.float64) for Sp in S],
        np.asarray(lam, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
    )
    if stopmode == "steps":
        stopcrit = stopcrit + 1
    trace = [
        nll(cur[0], cur[1], cur[2], stats, matlab_constants_only=matlab_constants_only)
    ]
    parerr: list[float] = []
    k = 2
    while True:
        new = ecme_step(cur[0], cur[1], cur[2], stats)
        parerr.append(_parerr(new, cur))
        trace.append(
            nll(
                new[0],
                new[1],
                new[2],
                stats,
                matlab_constants_only=matlab_constants_only,
            )
        )
        cur = new
        if stopmode == "steps":
            if k >= stopcrit:
                break
        elif parerr[-1] < stopcrit or k >= maxsteps:
            break
        k += 1
    return cur[0], cur[1], cur[2], np.asarray(parerr), np.asarray(trace)


# --------------------------------------------------------------------------- AIC


def aic_mmle(
    nll_value: float, ranks: Sequence[int], n: int, T: int
) -> tuple[int, float]:
    """`AIC = 2*negloglik + 2*numel(pars)`, `BTDR_AIC_S_lamb_b_wrapper.m:10`.

    `pars = [lambhat; shat; vec(bhat)]` (`MMLE_CoordAscentWrapper.m:8`), so
    the count is `n + T*sum(r) + n*T`: every `lambda`, every entry of the
    packed bases and every intercept entry (not the identifiable count, which
    is smaller by `sum(r .* (r - 1)) / 2`). The reference's `negloglik` is
    `nll(..., matlab_constants_only=True)` at `b` (lines 6-9); pass that value
    to reproduce its AIC, or the default `nll` for the fully normalised one.

    Returns
    -------
    count : int
        `numel(pars)`.
    aic : float
    """
    count = int(n) + int(T) * int(sum(int(r) for r in ranks)) + int(n) * int(T)
    return count, 2.0 * float(nll_value) + 2.0 * count
