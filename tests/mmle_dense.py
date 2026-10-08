"""Dense reference implementations of the marginal-likelihood quantities (tests).

Each function is written directly from `docs/model.md`, neuron by neuron, with the
explicit design Phi_i (M8), the covariance Sigma_i = I / lambda_i + Phi_i Phi_i',
or the Kronecker forms of (M26a), (M27a) and (M33) that the reference uses, so
that they share no code path with the batched kernels of `mtdr.mmle`. They are
slow and meant for small problems.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from mtdr.stats import SufficientStats

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def design(X_i: FloatArray, S: Sequence[FloatArray]) -> FloatArray:
    """Phi_i of (M8), `(n_i T, r_tot)`: row block j is [x_j1 S_1, ..., x_jP S_P]."""
    T = S[0].shape[0]
    cols = [np.kron(X_i[:, [p]], S[p]) for p in range(len(S)) if S[p].shape[1]]
    if not cols:
        return np.zeros((X_i.shape[0] * T, 0))
    return np.concatenate(cols, axis=1)


def _neuron(
    Y: FloatArray, X: FloatArray, mask: BoolArray, i: int
) -> tuple[FloatArray, FloatArray]:
    rows = np.flatnonzero(mask[:, i])
    return X[rows], Y[rows, i, :]


def nll(
    Y: FloatArray,
    X: FloatArray,
    mask: BoolArray,
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray | None,
) -> FloatArray:
    """Per-neuron fully normalised NLL of zeta_i ~ N(1 (x) b_i, Sigma_i)."""
    out = np.zeros(Y.shape[1])
    for i in range(Y.shape[1]):
        X_i, Y_i = _neuron(Y, X, mask, i)
        if X_i.shape[0] == 0:
            continue
        Phi = design(X_i, S)
        z = Y_i.ravel()  # trial-major, time fastest, as in (M5)
        if b is not None:
            z = z - np.tile(b[i], X_i.shape[0])
        Sigma = np.eye(z.size) / lam[i] + Phi @ Phi.T
        logdet = np.linalg.slogdet(Sigma)[1]
        out[i] = 0.5 * (
            logdet + z @ np.linalg.solve(Sigma, z) + z.size * np.log(2 * np.pi)
        )
    return out


def posterior(
    Y: FloatArray,
    X: FloatArray,
    mask: BoolArray,
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray | None,
) -> tuple[FloatArray, FloatArray]:
    """Posterior mean and covariance of w_i, (M25): direct Gaussian conditioning."""
    rtot = sum(s.shape[1] for s in S)
    n = Y.shape[1]
    means = np.zeros((n, rtot))
    covs = np.broadcast_to(np.eye(rtot), (n, rtot, rtot)).copy()
    for i in range(n):
        X_i, Y_i = _neuron(Y, X, mask, i)
        if X_i.shape[0] == 0:
            continue
        Phi = design(X_i, S)
        z = Y_i.ravel()
        if b is not None:
            z = z - np.tile(b[i], X_i.shape[0])
        # Joint Gaussian: w ~ N(0, I), z | w ~ N(Phi w, I / lam); condition on z.
        Sigma = np.eye(z.size) / lam[i] + Phi @ Phi.T
        gain = np.linalg.solve(Sigma, Phi).T  # Phi' Sigma^{-1}
        means[i] = gain @ z
        covs[i] = np.eye(rtot) - gain @ Phi
    return means, covs


def gls_intercept(
    Y: FloatArray,
    X: FloatArray,
    mask: BoolArray,
    S: Sequence[FloatArray],
    lam: FloatArray,
) -> FloatArray:
    """GLS mean of zeta_i ~ N(1 (x) b_i, Sigma_i): what (M33) computes."""
    n, T = Y.shape[1], Y.shape[2]
    out = np.zeros((n, T))
    for i in range(n):
        X_i, Y_i = _neuron(Y, X, mask, i)
        Phi = design(X_i, S)
        Sigma_inv = np.linalg.inv(np.eye(Y_i.size) / lam[i] + Phi @ Phi.T)
        M = np.kron(np.ones((X_i.shape[0], 1)), np.eye(T))
        out[i] = np.linalg.solve(M.T @ Sigma_inv @ M, M.T @ Sigma_inv @ Y_i.ravel())
    return out


def block_bases(S: Sequence[FloatArray]) -> FloatArray:
    """The block-diagonal S of (M7), `(r_tot, P T)`, with S_p' in block p."""
    T = S[0].shape[0]
    P = len(S)
    rtot = sum(s.shape[1] for s in S)
    out = np.zeros((rtot, P * T))
    row = 0
    for p, s in enumerate(S):
        r = s.shape[1]
        out[row : row + r, p * T : (p + 1) * T] = s.T
        row += r
    return out


def grad_bases_m26a(
    stats: SufficientStats,
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray | None,
    g: float = 0.0,
) -> list[FloatArray]:
    """(M26a) as the reference forms it; the active blocks, transposed to `(T, r_p)`."""
    T = stats.n_bins
    xi, _ = stats.centered(b)
    Sb = block_bases(S)
    rtot = Sb.shape[0]
    grad = g * Sb
    for i in range(stats.n_neurons):
        K = np.kron(stats.XtX[i], np.eye(T))  # A_i (x) I_T, time fastest
        x = xi[i].ravel()  # [xi_i]_{t + T p}
        R = np.outer(x, x)
        C = np.eye(rtot) + lam[i] * Sb @ K @ Sb.T
        Ci = np.linalg.inv(C)
        inner = np.eye(K.shape[0]) - lam[i] * Sb.T @ Ci @ Sb @ K
        grad = grad + lam[i] * Ci @ Sb @ K - lam[i] ** 2 * Ci @ Sb @ R @ inner
    out = []
    row = 0
    for p, s in enumerate(S):
        r = s.shape[1]
        out.append(grad[row : row + r, p * T : (p + 1) * T].T)
        row += r
    return out


def grad_noise_m27a(
    stats: SufficientStats,
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray | None,
) -> FloatArray:
    """(M27a) with the reference's `dQdlambi` trace form."""
    T = stats.n_bins
    xi, ups = stats.centered(b)
    Sb = block_bases(S)
    rtot = Sb.shape[0]
    out = np.zeros(stats.n_neurons)
    for i in range(stats.n_neurons):
        K = np.kron(stats.XtX[i], np.eye(T))
        x = xi[i].ravel()
        R = np.outer(x, x)
        SKS = Sb @ K @ Sb.T
        Ci = np.linalg.inv(np.eye(rtot) + lam[i] * SKS)
        q = x @ Sb.T @ Ci @ Sb @ x
        out[i] = 0.5 * (
            -stats.n_obs[i] * T / lam[i]
            + np.trace(Ci @ SKS)
            + ups[i]
            - 2 * lam[i] * q
            + lam[i] ** 2 * np.trace(Ci @ SKS @ Ci @ Sb @ R @ Sb.T)
        )
    return out


def intercept_m33(
    stats: SufficientStats, S: Sequence[FloatArray], lam: FloatArray
) -> FloatArray:
    """(M33) literally: the T x T bracket and the raw xi_i, as `MMLE_b` does."""
    T = stats.n_bins
    Sb = block_bases(S)
    rtot = Sb.shape[0]
    out = np.zeros((stats.n_neurons, T))
    for i in range(stats.n_neurons):
        K = np.kron(stats.XtX[i], np.eye(T))
        C = np.eye(rtot) + lam[i] * Sb @ K @ Sb.T
        Psi = np.kron(stats.X_mean[i][None, :], np.eye(T)) @ Sb.T  # (x_bar' (x) I_T) S'
        CIXS = np.linalg.solve(C, Psi.T)
        bracket = np.eye(T) - lam[i] * stats.n_obs[i] * Psi @ CIXS
        rhs = stats.Y_mean[i] - lam[i] * Psi @ np.linalg.solve(
            C, Sb @ stats.XtY_raw[i].ravel()
        )
        out[i] = np.linalg.solve(bracket, rhs)
    return out


def expected_residual(
    Y: FloatArray,
    X: FloatArray,
    mask: BoolArray,
    S: Sequence[FloatArray],
    lam: FloatArray,
    b: FloatArray | None,
) -> FloatArray:
    """E ||zeta_i - Phi_i w_i||^2 under the posterior of w_i, (M27b), (M30)."""
    means, covs = posterior(Y, X, mask, S, lam, b)
    out = np.zeros(Y.shape[1])
    for i in range(Y.shape[1]):
        X_i, Y_i = _neuron(Y, X, mask, i)
        Phi = design(X_i, S)
        z = Y_i.ravel()
        if b is not None:
            z = z - np.tile(b[i], X_i.shape[0])
        resid = z - Phi @ means[i]
        out[i] = resid @ resid + np.trace(Phi @ covs[i] @ Phi.T)
    return out


def compat_basis_step(
    stats: SufficientStats,
    S: Sequence[FloatArray],
    lam: FloatArray,
    lam_new: FloatArray,
    b: FloatArray | None,
) -> list[FloatArray]:
    """The reference's stale-variable S-step (`ECMEtdr.m` lines 66-110).

    Block matrices throughout: `BiSoldRi = C_i(lam)^{-1} S R_i` from the old
    precisions, right-hand side `sum_i lam'_i^2 BiSoldRi` restricted to the
    active blocks; `G_i = C_i(lam')^{-1} (lam'_i^2 u_i u_i' C_i(lam)^{-1} + I)`;
    `GG` from the upper block triangle of `sum_i lam'_i A_i[pq] G_i[pq]`, with
    the transpose of that triangle written on and below the block diagonal;
    solved as a general matrix.
    """
    T = stats.n_bins
    xi, _ = stats.centered(b)
    Sb = block_bases(S)
    rtot = Sb.shape[0]
    ranks = [s.shape[1] for s in S]
    offsets = np.concatenate([[0], np.cumsum(ranks)]).astype(int)
    M0 = np.zeros_like(Sb)
    Gt = np.zeros((rtot, rtot))
    for i in range(stats.n_neurons):
        K = np.kron(stats.XtX[i], np.eye(T))
        x = xi[i].ravel()
        SKS = Sb @ K @ Sb.T
        C_old = np.eye(rtot) + lam[i] * SKS
        C_new = np.eye(rtot) + lam_new[i] * SKS
        u = Sb @ x
        M0 += lam_new[i] ** 2 * np.linalg.solve(C_old, Sb @ np.outer(x, x))
        G = np.linalg.inv(C_new) @ (
            lam_new[i] ** 2 * np.outer(u, u) @ np.linalg.inv(C_old) + np.eye(rtot)
        )
        for p in range(len(S)):
            for q in range(len(S)):
                rp = slice(offsets[p], offsets[p + 1])
                rq = slice(offsets[q], offsets[q + 1])
                Gt[rp, rq] += lam_new[i] * stats.XtX[i, p, q] * G[rp, rq]
    GG = np.zeros_like(Gt)
    for p in range(len(S)):
        rp = slice(offsets[p], offsets[p + 1])
        upper = Gt[rp, offsets[p] :]
        GG[rp, offsets[p] :] = upper
        GG[offsets[p] :, rp] = upper.T  # also overwrites the diagonal block
    rhs = np.zeros((rtot, T))
    for p in range(len(S)):
        rp = slice(offsets[p], offsets[p + 1])
        rhs[rp] = M0[rp, p * T : (p + 1) * T]
    new = np.linalg.solve(GG, rhs)  # rows: components; columns: time
    return [new[offsets[p] : offsets[p + 1]].T for p in range(len(S))]


def compat_intercept_step(
    stats: SufficientStats,
    S_old: Sequence[FloatArray],
    S_new: Sequence[FloatArray],
    lam_new: FloatArray,
) -> FloatArray:
    """The reference's stale-variable b-step (`ECMEtdr.m` lines 148-154).

    `C_i^x = I + lam'_i S_old (A_i (x) I_T) S_new'`, `CIXS = solve(C^x, Psi')`, the
    bracket `I - lam'_i n_i Psi CIXS` and `ybarhat = lam'_i CIXS' S_new xi_i`
    with the raw `xi_i`, `Psi = (x_bar' (x) I_T) S_new'`.
    """
    T = stats.n_bins
    So, Sn = block_bases(S_old), block_bases(S_new)
    rtot = Sn.shape[0]
    out = np.zeros((stats.n_neurons, T))
    for i in range(stats.n_neurons):
        K = np.kron(stats.XtX[i], np.eye(T))
        Cx = np.eye(rtot) + lam_new[i] * So @ K @ Sn.T
        Psi = np.kron(stats.X_mean[i][None, :], np.eye(T)) @ Sn.T
        CIXS = np.linalg.solve(Cx, Psi.T)
        bracket = np.eye(T) - lam_new[i] * stats.n_obs[i] * Psi @ CIXS
        ybarhat = lam_new[i] * CIXS.T @ (Sn @ stats.XtY_raw[i].ravel())
        out[i] = np.linalg.solve(bracket, stats.Y_mean[i] - ybarhat)
    return out
