r"""Maximum marginal likelihood: the mTDR estimator (`docs/model.md` § 4-6).

The neuron weights are integrated out under the prior $w_i\sim\mathcal N(0,I)$
of (M4), so the temporal bases $S_p$, the noise precisions $\lambda_i$ and the
intercept $b$ are the parameters. By the determinant lemma and Woodbury, each
neuron's marginal likelihood reduces to one $r_{tot}\times r_{tot}$ system,
(M22)-(M25):

$$
C_i=I+\lambda_i\Phi_i^\top\Phi_i,\qquad
(\Phi_i^\top\Phi_i)_{[pq]}=[A_i]_{pq}\,S_p^\top S_q,\qquad
(u_i)_{[p]}=S_p^\top\xi_i(b)_{\langle p\rangle},
$$

$$
\mathcal L(\lambda,S;b)=\frac12\sum_{i=1}^n\Big[-n_iT\log\lambda_i
+\log\lvert C_i\rvert+\lambda_i\upsilon_i(b)-\lambda_i^2\,u_i^\top C_i^{-1}u_i\Big]
+\frac g2\lVert s\rVert^2 ,
$$

with $A_i$ the neuron's Gram matrix (M6) and $\xi_i(b)$, $\upsilon_i(b)$ its
intercept-centred cross-moment and energy (M13)-(M14), all read from the
[`SufficientStats`][mtdr.stats.SufficientStats]; the data are never touched.
The public log-likelihoods are fully normalised, $-\mathcal L$ at $g=0$ minus
$\tfrac12\sum_in_iT\log2\pi$.

The estimator (`docs/model.md` § 5) is the reference's
`MMLE_CoordAscentWrapper`: [`fit_svd`][mtdr.svd_fit.fit_svd] initialises
$(S,\lambda)$ and $b=\bar y$ (M29); [`ecme`][mtdr.mmle.ecme] takes closed-form
conditional-maximisation steps (M30)-(M33) in the consistent multicycle form
(each step built from the current iterate); [`refine`][mtdr.mmle.refine]
maximises the marginal likelihood directly, alternating L-BFGS over $S$, a
bounded L-BFGS step over $\log\lambda$ and the closed-form intercept;
[`posterior_weights`][mtdr.mmle.posterior_weights] gives the posterior of the
weights (M36). [`fit_mmle`][mtdr.mmle.fit_mmle] chains them and returns an
[`MMLEFit`][mtdr.mmle.MMLEFit].

Everything works in the estimator's raw frame: the likelihood is invariant
only under orthogonal rotations $S_p\to S_pQ_p$ (`docs/model.md` § 10.1), so
a canonicalised `S_` of a fitted `MTDR` is not a valid `S` argument.
Every kernel is batched over neurons (stacks of `(n, r_tot, r_tot)`
matrices); no loop over neurons runs outside an error path.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import scipy.linalg
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

from mtdr import aic as _aic
from mtdr._args import (
    as_bool,
    as_int,
    as_list,
    as_real,
    require_int_vector,
    require_non_negative_real,
)
from mtdr._frozen import ReadOnlyMapping, freeze, restore_frozen
from mtdr.errors import ConvergenceWarning, ParameterError, ValidationError
from mtdr.stats import SufficientStats
from mtdr.svd_fit import (
    SVDFit,
    _unconstrained_residual,
    _zero_to_rounding,
    fit_svd,
)

__all__ = [
    "LOG_PRECISION_BOUND",
    "MONOTONE_RTOL",
    "MMLEFit",
    "ecme",
    "fit_mmle",
    "marginal_log_likelihood",
    "marginal_nll_grad_S",
    "marginal_nll_grad_noise",
    "posterior_weights",
    "refine",
    "update_intercept",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Float1D = np.ndarray[tuple[int], np.dtype[np.float64]]

_LOG_2PI = float(np.log(2.0 * np.pi))

MONOTONE_RTOL = 1e-10
"""Relative rounding allowance for "no worse" in `ecme` and `refine`.

An ECME iteration counts as raising the marginal NLL when it rises by more than
this times `max(|NLL|, number of observed entries)`; a refinement step's
L-BFGS-B result is installed only if it raises the objective by no more than
this times `max(|objective|, number of observed entries)`.
"""

LOG_PRECISION_BOUND = 23.0
r"""Half-width, in log units, of each refinement precision step's box.

The box is centred on the current $\log\lambda_i$ (a factor of about $10^{10}$
either way) and intersected with the finite positive float64 range; ending on
it counts as an inner failure.
"""

# The log of the finite positive float64 range (smallest normal, largest
# finite): the precision step's box is intersected with it, so every trial
# precision exp(theta) is a finite positive number.
_LOG_FLOAT_MIN = float(np.log(np.finfo(np.float64).tiny))
_LOG_FLOAT_MAX = float(np.nextafter(np.log(np.finfo(np.float64).max), -np.inf))

PRECISION_RESIDUAL_TOL = 1e-8
r"""Scale-free stationarity a rounding-level precision-step end must reach.

A precision step whose L-BFGS-B run ends with status 2 (an abnormal line-search
end at rounding level) is first polished by at most `POLISH_MAX_ITER` diagonal
Newton steps, and counts as a success iff the installed point satisfies
$\max_i\lvert\lambda_i\mathcal E_i/(n_iT)-1\rvert\le$ this, the projected
$\theta$-gradient relative to $n_iT/2$; otherwise it is an inner failure.
"""

ROUNDING_PROGRESS_FACTOR = 100.0
r"""$c$ of the basis rule: a rounding-level end is a success.

A basis-step L-BFGS-B run that ends with status 2 (its line search found no
further decrease) at an accepted point with a finite gradient counts as a
success iff its last accepted decrease $\Delta_{\rm last}$ ($f_{k-1}-f_k$, or
$f_0-f_{\rm end}$ after no accepted iterate) is at most
$c\max(\texttt{optimizer\_progtol},\epsilon M)$, with $M=\sum_i\lvert$terms$_i\rvert$
the objective's rounding scale at the returned point (`_basis_magnitude`):
minFunc ends such a run normally ("function value changing by less than
progTol").
"""

BASIS_SPAN_SCALE = 30.0
r"""The `basis_span_scale` of `MTDR(basis_preconditioning=True)`'s cold fits.

The basis objective is flat along the gauge $S_p\to S_pQ_p$ and soft along the
rest of the in-span directions $\delta S_p=S_pM_p$, which only the weight prior
pins down: at the paper's scale the stiffest of them is 8 to 190 times below
the softest other direction, so L-BFGS-B spends the second half of a run
crawling along them. Scaling them by $\alpha$ in the optimiser's variables cuts
a cold fit's basis evaluations 7-10x; every $\alpha$ from 30 to 100 does about
as well. The MATLAB reference has no counterpart: its `minFunc` runs on the raw
bases, which is $\alpha=1$. `MTDR` reads this value when `mtdr.model` is
imported, so changing it at run time does not reach the class; pass
`basis_span_scale` to `fit_mmle` instead.
"""

POLISH_MAX_ITER = 3
r"""Most diagonal Newton steps that polish a status-2 precision-step end.

The precision problem is separable per neuron given the bases, so its Hessian
in $\theta=\log\lambda$ is diagonal; see `_precision_derivatives`.
"""

#: L-BFGS-B's relative function-decrease tolerance of the precision step (the
#: basis step stops on the absolute `optimizer_progtol` instead); `maxcor` is
#: minFunc's `Corr`.
_LBFGS_FTOL = 1e-11
_LBFGS_MEMORY = 100
#: float64's machine epsilon: a basis end reports its rounding as eps * M.
_EPS = float(np.finfo(np.float64).eps)
#: The polish stops once every scale-free residual is below this.
_POLISH_TARGET = 1e-10
#: Largest Newton step in log precision the polish takes.
_POLISH_STEP = 0.5
#: The polished point is installed if its objective is at most f_end plus this
#: many eps * M, M the objective's rounding scale sum_i |terms_i| at the
#: unpolished end.
_POLISH_ALLOWANCE = 10.0

_MAX_LISTED = 20


# =========================================================================== kernels


class _Blocks(NamedTuple):
    """The bases side by side, `[S_1, ..., S_P]`, with each component's regressor."""

    S: FloatArray  # (T, r_tot)
    group: IntArray  # (r_tot,) regressor of each component
    bounds: tuple[int, ...]  # (P + 1,) cumulative offsets c[p]


def _blocks(S: Sequence[FloatArray]) -> _Blocks:
    ranks = [int(s.shape[1]) for s in S]
    group = np.repeat(np.arange(len(S), dtype=np.int64), ranks)
    bounds = tuple(int(c) for c in np.concatenate([[0], np.cumsum(ranks)]))
    stacked = np.concatenate([np.asarray(s, dtype=np.float64) for s in S], axis=1)
    return _Blocks(stacked, group, bounds)


def _split(M: FloatArray, bounds: tuple[int, ...]) -> list[FloatArray]:
    """Split the last axis of `M` into the per-regressor blocks."""
    return [
        np.ascontiguousarray(M[..., bounds[p] : bounds[p + 1]])
        for p in range(len(bounds) - 1)
    ]


def _expand(A: FloatArray, group: IntArray) -> FloatArray:
    r"""$A_i[g_j,g_k]$, the Gram entry of the regressors of components $j$ and $k$."""
    out: FloatArray = A[:, group[:, None], group[None, :]]
    return out


def _project(xi: FloatArray, blocks: _Blocks) -> FloatArray:
    r"""Return $u_i[k]=\sum_tS[t,k]\,\xi_i[g_k,t]$, i.e. $(u_i)_{[p]}$ of (M23)."""
    out: FloatArray = np.einsum("ikt,tk->ik", xi[:, blocks.group, :], blocks.S)
    return out


def _identity_plus(lam: FloatArray, K: FloatArray) -> FloatArray:
    r"""$I+\lambda_iK_i$ for every neuron, (M22)."""
    C = lam[:, None, None] * K
    r = C.shape[1]
    C[:, np.arange(r), np.arange(r)] += 1.0
    return C


def _cholesky_inverse(C: FloatArray) -> tuple[FloatArray, FloatArray]:
    r"""Return $L_i^{-1}$ and $\log\lvert C_i\rvert$ for a stack of SPD matrices.

    $C_i=L_iL_i^\top$ (Cholesky), $\log\lvert C_i\rvert=2\sum_j\log(L_i)_{jj}$. A
    matrix that is not positive definite, or not finite, raises
    `numpy.linalg.LinAlgError` naming the neurons: numpy's batched Cholesky
    raises without saying which slice failed, and returns `NaN` for non-finite
    input, so the culprits are found slice by slice on that path only.
    """
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError as err:
        raise np.linalg.LinAlgError(_not_pd_message(C)) from err
    diag = np.diagonal(L, axis1=1, axis2=2)
    if not (np.isfinite(L).all() and (diag > 0).all()):
        raise np.linalg.LinAlgError(_not_pd_message(C))
    logdet: FloatArray = 2.0 * np.log(diag).sum(axis=1)
    return _lower_inverse(L, diag), logdet


def _lower_inverse(L: FloatArray, diag: FloatArray) -> FloatArray:
    r"""Invert a stack of lower-triangular matrices by forward substitution.

    Row $j$ of $X=L^{-1}$ is $(e_j-\sum_{k<j}L_{jk}X_{k\cdot})/L_{jj}$. The loop
    runs over the $r_{tot}$ rows, each step batched over neurons; it is about
    three times faster than `numpy.linalg.inv` on the stacks of small matrices
    this module factors (800 matrices of order 20: 2.2 ms against 5.8 ms), whose
    cost is per-matrix LAPACK overhead.
    """
    r = L.shape[1]
    X = np.zeros_like(L)
    inv_diag = 1.0 / diag
    for j in range(r):
        row = -np.einsum("ik,ikm->im", L[:, j, :j], X[:, :j, :])
        row[:, j] += 1.0
        X[:, j, :] = row * inv_diag[:, j, None]
    return X


def _not_pd_message(C: FloatArray) -> str:
    bad = []
    for i in range(C.shape[0]):  # error path only
        try:
            ok = bool(
                np.isfinite(C[i]).all() and np.isfinite(np.linalg.cholesky(C[i])).all()
            )
        except np.linalg.LinAlgError:
            ok = False
        if not ok:
            bad.append(i)
    return (
        "the posterior precision C_i = I + lambda_i Phi_i' Phi_i is not positive "
        f"definite, or not finite, for neurons {_listed(np.array(bad))}"
    )


def _posterior(
    lam: FloatArray, K: FloatArray, u: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    r"""Return $C_i^{-1}$, $\mu_i$, $\lambda_i^2u_i^\top C_i^{-1}u_i$ and $\log|C_i|$.

    The posterior of (M25), $\mu_i=\lambda_iC_i^{-1}u_i$, and the two terms of
    (M24) that need $C_i$. Both go through the scaled solve
    $z_i=L_i^{-1}(\lambda_iu_i)$, with $C_i=L_iL_i^\top$: $\mu_i=L_i^{-\top}z_i$
    and the explained quadratic $\lambda_i^2u_i^\top C_i^{-1}u_i=\lVert
    z_i\rVert^2$. $\lambda_iu_i$ and $\lambda_iK_i$ are the whitened quantities,
    of ordinary size whatever the units of $Y$ and $\lambda$, whereas
    $u_i^\top C_i^{-1}u_i$ and $\lambda_i^2$ can each overflow or underflow on
    finite, well-conditioned inputs (`docs/model.md` § 10.2).
    """
    Linv, logdet = _cholesky_inverse(_identity_plus(lam, K))
    z = np.einsum("ijk,ik->ij", Linv, lam[:, None] * u)  # L^{-1} (lambda u)
    explained = np.einsum("ij,ij->i", z, z)  # lambda^2 u' C^{-1} u
    Cinv = Linv.transpose(0, 2, 1) @ Linv
    Cinv = 0.5 * (Cinv + Cinv.transpose(0, 2, 1))
    mu = np.einsum("ikj,ik->ij", Linv, z)  # L^{-T} z = lambda C^{-1} u
    return Cinv, mu, explained, logdet


class _State(NamedTuple):
    """Everything the likelihood, its gradients and the posterior need at one point."""

    blocks: _Blocks
    Aexp: FloatArray  # (n, r, r)  A_i[g_j, g_k]
    K: FloatArray  # (n, r, r)  Phi_i' Phi_i, (M23)
    u: FloatArray  # (n, r)     S xi_i(b), (M23)
    Cinv: FloatArray  # (n, r, r)  C_i^{-1}, (M22)
    mu: FloatArray  # (n, r)     posterior mean, (M25)
    terms: FloatArray  # (n,)       per-neuron (M24), no constant, no ridge


def _state(
    blocks: _Blocks,
    lam: FloatArray,
    xi: FloatArray,
    ups: FloatArray,
    A: FloatArray,
    n_obs: NDArray[Any],
    T: int,
) -> _State:
    r"""Evaluate (M22)-(M25) at $(S,\lambda)$, centred $\xi_i(b)$, $\upsilon_i(b)$.

    Raises `ValidationError` naming the neurons whose term or posterior mean is
    not representable in float64.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        Aexp = _expand(A, blocks.group)
        K = Aexp * (blocks.S.T @ blocks.S)
        u = _project(xi, blocks)
        Cinv, mu, explained, logdet = _posterior(lam, K, u)
        counts = n_obs.astype(np.float64) * T
        terms = 0.5 * (-counts * np.log(lam) + logdet + lam * ups - explained)
    bad = ~(np.isfinite(terms) & np.isfinite(mu).all(axis=1))
    if bad.any():
        raise ValidationError(
            "the marginal likelihood is not representable in float64 at these "
            f"parameters: it overflows for neurons {_listed(np.flatnonzero(bad))}; "
            "rescale Y, or the noise precisions and bases with it (docs/model.md "
            "section 10.2)"
        )
    return _State(blocks, Aexp, K, u, Cinv, mu, terms)


def _nll_terms(
    S_list: Sequence[FloatArray],
    lam: FloatArray,
    xi_c: FloatArray,
    A: FloatArray,
    upsilon_c: FloatArray,
    n_obs: NDArray[Any],
    T: int,
) -> FloatArray:
    r"""Per-neuron marginal NLL, (M24), as the reference computes it.

    A pure function of the parameters and the statistics, for term-by-term
    comparison with an independent implementation (`tests/test_mmle_nversion.py`
    compares it with the MATLAB-derived oracle):

    $$
    \mathcal L_i=\tfrac12\big[-n_iT\log\lambda_i+\log\lvert C_i\rvert
    +\lambda_i\upsilon_i(b)-\lambda_i^2u_i^\top C_i^{-1}u_i\big],
    $$

    without the $\tfrac12n_iT\log2\pi$ constant and without the ridge.

    Parameters
    ----------
    S_list : sequence of numpy.ndarray
        Per regressor, `(T, r_p)` bases.
    lam : numpy.ndarray
        `(n,)` precisions, positive.
    xi_c : numpy.ndarray
        `(n, P, T)` intercept-centred cross-moment $\xi_i(b)$, (M13)
        (`stats.centered(b)[0]`).
    A : numpy.ndarray
        `(n, P, P)` raw Gram matrices $A_i$, (M6) (`stats.XtX`).
    upsilon_c : numpy.ndarray
        `(n,)` intercept-centred energy $\upsilon_i(b)$, (M14).
    n_obs : numpy.ndarray
        `(n,)` observed trials per neuron.
    T : int
        Number of bins.

    Returns
    -------
    numpy.ndarray
        `(n,)`, $\mathcal L_i$; a never-observed neuron contributes 0.
    """
    return _state(_blocks(S_list), lam, xi_c, upsilon_c, A, n_obs, T).terms


def _second_moment(mu: FloatArray, Cinv: FloatArray) -> FloatArray:
    r"""$\Omega_i=\mu_i\mu_i^\top+C_i^{-1}$, the posterior second moment, (M25)."""
    out: FloatArray = mu[:, :, None] * mu[:, None, :] + Cinv
    return out


def _expected_residual(
    K: FloatArray, u: FloatArray, mu: FloatArray, Cinv: FloatArray, ups: FloatArray
) -> FloatArray:
    r"""Return the expected residual energy $\mathcal E_i$ of (M27b).

    $\mathcal E_i=\upsilon_i(b)-2u_i^\top\mu_i+\mathrm{tr}(K_i\Omega_i)$ with
    $K_i=\Phi_i^\top\Phi_i$.
    """
    out: FloatArray = (
        ups
        - 2.0 * np.einsum("ij,ij->i", u, mu)
        + np.einsum("ijk,ijk->i", K, _second_moment(mu, Cinv))
    )
    return out


def _grad_bases(
    S: FloatArray,
    Aexp: FloatArray,
    mu: FloatArray,
    Cinv: FloatArray,
    lam: FloatArray,
    xig: FloatArray,
) -> FloatArray:
    r"""Return $\partial\mathcal L/\partial[S_1,\dots,S_P]$ at $g=0$, (M26b).

    One `(T, r_tot)` array.

    Column $k\in\mathcal R_p$ is
    $\sum_i\lambda_i\big[\sum_jS[:,j]\,A_i[g_j,g_k](\Omega_i)_{jk}
    -\xi_{i\langle p\rangle}(\mu_i)_k\big]$;
    `xig` is $\xi_i(b)$ gathered by component, `xi[:, group, :]`.
    """
    H = np.einsum("i,ijk->jk", lam, Aexp * _second_moment(mu, Cinv))
    E = np.einsum("i,ikt,ik->tk", lam, xig, mu)
    out: FloatArray = S @ H - E
    return out


def _constant(n_obs: NDArray[Any], T: int) -> float:
    r"""Return $\tfrac12\sum_in_iT\log2\pi$, the constant the reference drops."""
    return 0.5 * _LOG_2PI * T * float(n_obs.sum())


def _ridge(S: FloatArray) -> float:
    return float(np.einsum("tk,tk->", S, S))


# =========================================================================== evaluators


def marginal_log_likelihood(
    S: Sequence[ArrayLike],
    noise_precision: ArrayLike,
    intercept: ArrayLike | None,
    stats: SufficientStats,
    basis_ridge: float = 0.0,
) -> float:
    r"""Marginal log-likelihood with the weights integrated out, (M22)-(M24).

    $$
    \ell(S,\lambda,b)=-\mathcal L(\lambda,S;b)\big|_{g=0}-\tfrac12\sum_in_iT\log2\pi
    -\tfrac g2\sum_p\lVert S_p\rVert_F^2 ,
    $$

    the fully normalised Gaussian log-density of every neuron's observed
    responses under
    $\mathcal N(\mathbf 1\otimes b_i,\ \lambda_i^{-1}I+\Phi_i\Phi_i^\top)$,
    minus the ridge when `basis_ridge` $=g>0$ (the value is then the penalised
    objective, not a log-likelihood). A neuron never observed contributes 0.

    Parameters
    ----------
    S : sequence of array_like
        Per regressor, the `(n_bins, r_p)` bases in `X` column order, in the
        estimator's raw frame; `r_p` may be 0.
    noise_precision : array_like of shape (n_neurons,)
        $\lambda_i>0$.
    intercept : array_like of shape (n_neurons, n_bins), or None
        $b$; `None` for a model without the intercept.
    stats : SufficientStats
        From [`sufficient_statistics`][mtdr.stats.sufficient_statistics].
    basis_ridge : float
        $g\ge0$, the ridge on the bases.

    Returns
    -------
    float
        The log-likelihood (or the penalised objective).

    Raises
    ------
    ParameterError
        For a `stats` that is not a `SufficientStats`; an `S` with the wrong
        number of blocks or a block that is not a finite real `(n_bins, r_p)`
        array; a `noise_precision` that is not a finite positive
        `(n_neurons,)` array; a bad `intercept`; a negative `basis_ridge`.
    ValidationError
        If the value is not representable in float64 at these parameters (it
        overflows), naming the neurons; never a `NaN`.
    numpy.linalg.LinAlgError
        If a posterior precision $C_i$ is not finite or not numerically
        positive definite, naming the neurons.

    Notes
    -----
    Cost: $O(n\,r_{tot}^3+n\,P\,T\,r_{tot}+T\,r_{tot}^2)$ time (one batched
    Cholesky factorisation and inverse) and $O(n\,r_{tot}^2)$ memory. The
    explained quadratic is evaluated as $\lambda_i^2u_i^\top C_i^{-1}u_i=\lVert
    z_i\rVert^2$ with the scaled solve $z_i=L_i^{-1}(\lambda_iu_i)$,
    $C_i=L_iL_i^\top$, so that only the whitened products $\lambda_iu_i$ and
    $\lambda_iK_i$ are formed, which are of ordinary size at any scaling of
    $Y$ and $\lambda$ (`docs/model.md` § 10.2); $u_i^\top C_i^{-1}u_i$ and
    $\lambda_i^2$ separately can overflow on finite, well-conditioned inputs.
    The quadratic term $\lambda_i\upsilon_i(b)-\lVert z_i\rVert^2$ is a
    difference of terms of the size of the neuron's energy, so it loses about
    $\log_{10}$ of the signal-to-noise power ratio in digits, as the
    reference's does.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import simulate
    >>> from mtdr.mmle import marginal_log_likelihood
    >>> from mtdr.stats import sufficient_statistics
    >>> sim = simulate(n_neurons=8, n_bins=5, n_trials=40, ranks=[1, 2], seed=0)
    >>> stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    >>> S = [sim.S["x0"], sim.S["x1"]]
    >>> ll = marginal_log_likelihood(S, sim.noise_precision, sim.intercept, stats)
    >>> bool(np.isfinite(ll))
    True
    """
    stats, blocks, lam, xi, ups = _evaluation_args(S, noise_precision, intercept, stats)
    g = require_non_negative_real("basis_ridge", basis_ridge)
    st = _state(blocks, lam, xi, ups, stats.XtX, stats.n_obs, stats.n_bins)
    nll = float(st.terms.sum()) + _constant(stats.n_obs, stats.n_bins)
    return -nll - 0.5 * g * _ridge(blocks.S)


def marginal_nll_grad_S(  # noqa: N802 - the public API name
    S: Sequence[ArrayLike],
    noise_precision: ArrayLike,
    intercept: ArrayLike | None,
    stats: SufficientStats,
    basis_ridge: float = 0.0,
) -> tuple[float, list[FloatArray]]:
    r"""Marginal NLL and its gradient with respect to the bases, (M24), (M26a)-(M26b).

    $$
    \frac{\partial\mathcal L}{\partial S_p}=\sum_{i=1}^n\lambda_i\Big[\sum_q[A_i]_{pq}\,
    S_q\,(\Omega_i)_{[qp]}-\xi_i(b)_{\langle p\rangle}(\mu_i)_{[p]}^\top\Big]+g\,S_p ,
    $$

    with $\mu_i$, $\Omega_i$ the posterior mean and second moment of the
    weights (M25): (M26b), the expected complete-data gradient, equal to the
    reference's (M26a) identically.

    Parameters
    ----------
    S : sequence of array_like
        Per regressor, the `(n_bins, r_p)` bases, as
        [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood].
    noise_precision : array_like of shape (n_neurons,)
        $\lambda_i>0$, as `marginal_log_likelihood`.
    intercept : array_like of shape (n_neurons, n_bins), or None
        $b$; `None` without the intercept.
    stats : SufficientStats
        The statistics.
    basis_ridge : float
        $g\ge0$, the ridge on the bases.

    Returns
    -------
    value : float
        `-marginal_log_likelihood(S, noise_precision, intercept, stats,
        basis_ridge)`: the fully normalised NLL plus $\tfrac g2\lVert s\rVert^2$,
        the function the gradient is of.
    gradient : list of numpy.ndarray
        Per regressor, `(n_bins, r_p)`, $\partial(\text{value})/\partial S_p$.

    Raises
    ------
    ParameterError, ValidationError, numpy.linalg.LinAlgError
        As [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood]; the
        `ValidationError` also if the gradient overflows.

    Examples
    --------
    >>> from mtdr import simulate
    >>> from mtdr.mmle import marginal_nll_grad_S
    >>> from mtdr.stats import sufficient_statistics
    >>> sim = simulate(n_neurons=8, n_bins=5, n_trials=40, ranks=[1, 2], seed=0)
    >>> stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    >>> S = [sim.S["x0"], sim.S["x1"]]
    >>> value, grad = marginal_nll_grad_S(S, sim.noise_precision, sim.intercept, stats)
    >>> [g.shape for g in grad]
    [(5, 1), (5, 2)]
    """
    stats, blocks, lam, xi, ups = _evaluation_args(S, noise_precision, intercept, stats)
    g = require_non_negative_real("basis_ridge", basis_ridge)
    st = _state(blocks, lam, xi, ups, stats.XtX, stats.n_obs, stats.n_bins)
    value = (
        float(st.terms.sum())
        + _constant(stats.n_obs, stats.n_bins)
        + 0.5 * g * _ridge(blocks.S)
    )
    xig = xi[:, blocks.group, :]
    with np.errstate(over="ignore", invalid="ignore"):
        grad = _grad_bases(blocks.S, st.Aexp, st.mu, st.Cinv, lam, xig) + g * blocks.S
    _require_finite_gradient(grad)
    return value, _split(grad, blocks.bounds)


def marginal_nll_grad_noise(
    S: Sequence[ArrayLike],
    noise_precision: ArrayLike,
    intercept: ArrayLike | None,
    stats: SufficientStats,
) -> tuple[float, FloatArray]:
    r"""Marginal NLL and its gradient in the noise precisions, (M27a)-(M27b).

    $$
    \frac{\partial\mathcal L}{\partial\lambda_i}
    =\frac12\Big[-\frac{n_iT}{\lambda_i}+\mathcal E_i\Big],\qquad
    \mathcal E_i=\upsilon_i(b)-2u_i^\top\mu_i
    +\mathrm{tr}\big(\Phi_i^\top\Phi_i\,\Omega_i\big),
    $$

    $\mathcal E_i$ being the posterior expected residual energy
    $\mathbb E\lVert\tilde\zeta_i-\Phi_iw_i\rVert^2$. Each $\lambda_i$ enters only
    neuron $i$'s term.

    Parameters
    ----------
    S : sequence of array_like
        Per regressor, the `(n_bins, r_p)` bases, as
        [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood].
    noise_precision : array_like of shape (n_neurons,)
        $\lambda_i>0$, as `marginal_log_likelihood`.
    intercept : array_like of shape (n_neurons, n_bins), or None
        $b$; `None` without the intercept.
    stats : SufficientStats
        The statistics.

    Returns
    -------
    value : float
        The fully normalised marginal NLL, `-marginal_log_likelihood(S,
        noise_precision, intercept, stats)` (no ridge: it does not depend on
        $\lambda$).
    gradient : numpy.ndarray
        `(n_neurons,)`, $\partial(\text{value})/\partial\lambda_i$; 0 for a
        neuron never observed.

    Raises
    ------
    ParameterError, ValidationError, numpy.linalg.LinAlgError
        As [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood]; the
        `ValidationError` also if the gradient overflows.

    Examples
    --------
    >>> from mtdr import simulate
    >>> from mtdr.mmle import marginal_nll_grad_noise
    >>> from mtdr.stats import sufficient_statistics
    >>> sim = simulate(n_neurons=8, n_bins=5, n_trials=40, ranks=[1, 2], seed=0)
    >>> stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    >>> S = [sim.S["x0"], sim.S["x1"]]
    >>> value, grad = marginal_nll_grad_noise(S, sim.noise_precision,
    ...                                       sim.intercept, stats)
    >>> grad.shape
    (8,)
    """
    stats, blocks, lam, xi, ups = _evaluation_args(S, noise_precision, intercept, stats)
    st = _state(blocks, lam, xi, ups, stats.XtX, stats.n_obs, stats.n_bins)
    value = float(st.terms.sum()) + _constant(stats.n_obs, stats.n_bins)
    counts = stats.n_obs.astype(np.float64) * stats.n_bins
    with np.errstate(over="ignore", invalid="ignore"):
        expected = _expected_residual(st.K, st.u, st.mu, st.Cinv, ups)
        grad: FloatArray = 0.5 * (-counts / lam + expected)
    _require_finite_gradient(grad)
    return value, grad


def posterior_weights(
    S: Sequence[ArrayLike],
    noise_precision: ArrayLike,
    intercept: ArrayLike | None,
    stats: SufficientStats,
) -> tuple[list[FloatArray], FloatArray]:
    r"""Posterior of each neuron's stacked weight vector, (M25), (M36).

    $$
    w_i\mid\zeta_i,S,\lambda,b\sim\mathcal N\big(\mu_i,\ C_i^{-1}\big),\qquad
    \mu_i=\lambda_iC_i^{-1}u_i,\qquad C_i=I+\lambda_i\Phi_i^\top\Phi_i ,
    $$

    the reference's `EBpost_W_uneqvar` with the intercept-centred statistic
    $\xi_i(b)$, as `demoLearning.m` calls it (`mTDRdemo.m:155` passes the raw
    statistic, a bug; see `docs/differences-from-matlab.md`).
    $\hat W_p[i,:]=(\mu_i)_{[p]}$, and $\hat B_p=\hat W_pS_p^\top$ is (M37).

    Parameters
    ----------
    S : sequence of array_like
        Per regressor, the `(n_bins, r_p)` bases, as
        [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood].
    noise_precision : array_like of shape (n_neurons,)
        $\lambda_i>0$, as `marginal_log_likelihood`.
    intercept : array_like of shape (n_neurons, n_bins), or None
        $b$; `None` without the intercept.
    stats : SufficientStats
        The statistics.

    Returns
    -------
    W_mean : list of numpy.ndarray
        Per regressor, `(n_neurons, r_p)` posterior means.
    W_cov : numpy.ndarray
        `(n_neurons, total_rank, total_rank)` posterior covariances
        $C_i^{-1}$, blocks in regressor order; the identity for a neuron never
        observed (its posterior is the prior).

    Raises
    ------
    ParameterError, ValidationError, numpy.linalg.LinAlgError
        As [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood].

    Examples
    --------
    >>> from mtdr import simulate
    >>> from mtdr.mmle import posterior_weights
    >>> from mtdr.stats import sufficient_statistics
    >>> sim = simulate(n_neurons=8, n_bins=5, n_trials=40, ranks=[1, 2], seed=0)
    >>> stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    >>> S = [sim.S["x0"], sim.S["x1"]]
    >>> W, W_cov = posterior_weights(S, sim.noise_precision, sim.intercept, stats)
    >>> [w.shape for w in W], W_cov.shape
    ([(8, 1), (8, 2)], (8, 3, 3))
    """
    stats, blocks, lam, xi, ups = _evaluation_args(S, noise_precision, intercept, stats)
    st = _state(blocks, lam, xi, ups, stats.XtX, stats.n_obs, stats.n_bins)
    return _split(st.mu, blocks.bounds), st.Cinv


def update_intercept(
    S: Sequence[ArrayLike], noise_precision: ArrayLike, stats: SufficientStats
) -> FloatArray:
    r"""Return the intercept maximising the marginal likelihood given $(S,\lambda)$.

    `docs/model.md` (M32)-(M33).

    (M33) is the generalised-least-squares mean of
    $\zeta_i\sim\mathcal N(\mathbf 1\otimes b_i,\Sigma_i)$,

    $$
    b_i=\big[I_T-\lambda_in_i\Psi_iC_i^{-1}\Psi_i^\top\big]^{-1}
    \big(\bar y_i-\lambda_i\Psi_iC_i^{-1}\mathbf S\xi_i\big),\qquad
    \Psi_i=\big[\bar x_{i1}S_1,\dots,\bar x_{iP}S_P\big],
    $$

    with the raw $\xi_i$ of (M10), so no current intercept is needed. It is
    evaluated in the equivalent form

    $$
    b_i=\bar y_i-\Psi_i\tilde\mu_i,\qquad
    \tilde\mu_i=\lambda_i\tilde C_i^{-1}\mathbf S\tilde\xi_i,\qquad
    \tilde C_i=I+\lambda_i\mathbf S(\tilde A_i\otimes I_T)\mathbf S^\top ,
    $$

    from the moments about the neuron's means ($\tilde A_i$, $\tilde\xi_i$):
    the mean response minus the fit at the mean regressor, with the weights at
    their posterior mean under the model with the intercept profiled out. The
    forms are equal by Woodbury
    ($[I_T-\lambda_in_i\Psi_iC_i^{-1}\Psi_i^\top]^{-1}=I_T+\lambda_in_i\Psi_i
    \tilde C_i^{-1}\Psi_i^\top$ and $C_i=\tilde C_i+\lambda_in_i\Psi_i^\top\Psi_i$);
    this one needs no $T\times T$ solve and has no cancellation with a large
    baseline.

    Parameters
    ----------
    S : sequence of array_like
        Per regressor, the `(n_bins, r_p)` bases, as
        [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood].
    noise_precision : array_like of shape (n_neurons,)
        $\lambda_i>0$, as `marginal_log_likelihood`.
    stats : SufficientStats
        The statistics.

    Returns
    -------
    numpy.ndarray
        `(n_neurons, n_bins)`, the intercept $b$.

    Raises
    ------
    ParameterError
        As [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood].
    ValidationError
        If a neuron was never observed (its intercept is undefined).

    Examples
    --------
    >>> from mtdr import simulate
    >>> from mtdr.mmle import update_intercept
    >>> from mtdr.stats import sufficient_statistics
    >>> sim = simulate(n_neurons=8, n_bins=5, n_trials=40, ranks=[1, 2], seed=0)
    >>> stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    >>> b = update_intercept([sim.S["x0"], sim.S["x1"]], sim.noise_precision, stats)
    >>> b.shape
    (8, 5)
    """
    stats, blocks, lam, _, _ = _evaluation_args(S, noise_precision, None, stats)
    _require_observed(stats)
    return _intercept_step(blocks, lam, stats)


# =========================================================================== ECME steps


def _precision_step(
    st: _State, ups: FloatArray, n_obs: NDArray[Any], T: int
) -> FloatArray:
    r"""CM-step for $\lambda$, (M30): $\lambda'_i=n_iT/\mathcal E_i$.

    $\mathcal E_i$ must exceed the rounding floor $10^3\epsilon T\upsilon_i(b)$.
    """
    E = _expected_residual(st.K, st.u, st.mu, st.Cinv, ups)
    _require_expected_residual(E, ups, T)
    out: FloatArray = n_obs.astype(np.float64) * T / E
    return out


def _require_expected_residual(E: FloatArray, ups: FloatArray, T: int) -> None:
    r"""Raise if $\mathcal E_i$ of (M30) is zero to rounding against $\upsilon_i(b)$.

    That is, $\mathcal E_i\le10^3\epsilon T\upsilon_i(b)$, or not finite: the
    precision $n_iT/\mathcal E_i$ would then be unbounded.
    """
    bad = np.flatnonzero(~np.isfinite(E) | _zero_to_rounding(E, ups, T))
    if bad.size:
        raise ValidationError(
            f"neurons {_listed(bad)} have zero expected residual energy under the "
            "posterior (to rounding), so their noise precision is unbounded: drop "
            "them or require more observations per neuron"
        )


def _basis_step(st: _State, lam_new: FloatArray, xi: FloatArray) -> _Blocks:
    r"""Multicycle CM-step for $S$, (M31).

    The E-step is redone at $(\lambda',S)$: $C'_i=I+\lambda'_i\Phi_i^\top\Phi_i$,
    $\mu'_i=\lambda'_iC_i'^{-1}u_i$, $\Omega'_i=\mu'_i\mu_i'^\top+C_i'^{-1}$. Then
    $\mathcal G\,\mathcal S'^\top=\mathcal M$ with
    $\mathcal G_{jk}=\sum_i\lambda'_iA_i[g_j,g_k](\Omega'_i)_{jk}$ (symmetric
    positive definite) and $\mathcal M[k,:]=\sum_i\lambda'_i(\mu'_i)_k\,
    \xi_i(b)_{\langle g_k\rangle}^\top$.
    """
    blocks = st.blocks
    Cinv, mu, _, _ = _posterior(lam_new, st.K, st.u)
    G = np.einsum("i,ijk->jk", lam_new, st.Aexp * _second_moment(mu, Cinv))
    G = 0.5 * (G + G.T)
    M = np.einsum("i,ik,ikt->kt", lam_new, mu, xi[:, blocks.group, :])
    try:
        new = scipy.linalg.solve(G, M, assume_a="pos")
    except np.linalg.LinAlgError as err:
        # A data condition, so a ValidationError: by the Schur product theorem
        # G is positive definite whenever every regressor has some [A_i]_pp > 0.
        raise ValidationError(
            "the ECME basis system (M31) is not positive definite: some regressor "
            "has no observed trial with a nonzero value"
        ) from err
    return _Blocks(np.ascontiguousarray(new.T), blocks.group, blocks.bounds)


def _intercept_step(
    blocks: _Blocks, lam: FloatArray, stats: SufficientStats
) -> FloatArray:
    r"""(M33) in the centred form $b_i=\bar y_i-\Psi_i\tilde\mu_i$."""
    Kc = _expand(stats.XtX_c, blocks.group) * (blocks.S.T @ blocks.S)
    _, mu_c, _, _ = _posterior(lam, Kc, _project(stats.XtY_c, blocks))
    x_mean = stats.X_mean[:, blocks.group]
    out: FloatArray = stats.Y_mean - np.einsum("tk,ik,ik->it", blocks.S, x_mean, mu_c)
    return out


def _basis_step_reference(st: _State, lam_new: FloatArray, xi: FloatArray) -> _Blocks:
    r"""Take the reference's stale-variable S-step, as `ECMEtdr.m` does.

    With $C_i(\lambda)$ from the E-step at the old precisions and
    $C_i(\lambda')$ rebuilt with the new ones (both with the current $S$):
    $\hat\mu_i=\lambda'_iC_i(\lambda)^{-1}u_i$,
    $\mu'_i=\lambda'_iC_i(\lambda')^{-1}u_i$,
    the non-symmetric $G_i=\mu'_i\hat\mu_i^\top+C_i(\lambda')^{-1}$ and
    $\tilde{\mathcal G}_{jk}=\sum_i\lambda'_iA_i[g_j,g_k](G_i)_{jk}$. `GG` holds
    $\tilde{\mathcal G}$ on the strictly upper block triangle and its transpose
    on and below the block diagonal; the right-hand side is
    $\sum_i\lambda'_i(\hat\mu_i)_k\,\xi_i(b)_{\langle g_k\rangle}^\top$; `GG` is
    solved as a general matrix.
    """
    blocks = st.blocks
    Cinv_new, mu_new, _, _ = _posterior(lam_new, st.K, st.u)
    mu_hat = lam_new[:, None] * np.einsum("ijk,ik->ij", st.Cinv, st.u)
    Gi = mu_new[:, :, None] * mu_hat[:, None, :] + Cinv_new
    Gt = np.einsum("i,ijk->jk", lam_new, st.Aexp * Gi)
    upper = blocks.group[:, None] < blocks.group[None, :]
    GG = np.where(upper, Gt, Gt.T)
    M = np.einsum("i,ik,ikt->kt", lam_new, mu_hat, xi[:, blocks.group, :])
    new = np.linalg.solve(GG, M)
    return _Blocks(np.ascontiguousarray(new.T), blocks.group, blocks.bounds)


def _intercept_step_reference(
    old: _Blocks, new: _Blocks, lam_new: FloatArray, stats: SufficientStats
) -> FloatArray:
    r"""Take the reference's stale-variable b-step, as `ECMEtdr.m` does.

    $C^\times_i=I+\lambda'_i\mathbf S_{\rm old}K_i\mathbf S_{\rm new}^\top$ (not
    symmetric) and, with $\Psi_i$ from the new bases,
    $b_i=[I-\lambda'_in_i\Psi_i(C^\times_i)^{-1}\Psi_i^\top]^{-1}
    [\bar y_i-\lambda'_i\Psi_i(C^\times_i)^{-\top}\mathbf S_{\rm new}\xi_i]$ with the
    raw $\xi_i$, evaluated literally from the raw moments.
    """
    T = stats.n_bins
    Cx = _identity_plus(lam_new, _expand(stats.XtX, new.group) * (old.S.T @ new.S))
    Psi = new.S[None, :, :] * stats.X_mean[:, None, new.group]  # (n, T, r)
    CIXS = np.linalg.solve(Cx, Psi.transpose(0, 2, 1))  # (n, r, T) = C^-1 Psi'
    counts = stats.n_obs.astype(np.float64)
    bracket = np.eye(T) - (lam_new * counts)[:, None, None] * (Psi @ CIXS)
    u_raw = _project(stats.XtY_raw, new)
    ybarhat = lam_new[:, None] * np.einsum("ikt,ik->it", CIXS, u_raw)
    rhs = (stats.Y_mean - ybarhat)[..., None]
    out: FloatArray = np.linalg.solve(bracket, rhs)[..., 0]
    return out


def _relative_change(
    old: tuple[FloatArray, FloatArray, FloatArray | None],
    new: tuple[FloatArray, FloatArray, FloatArray | None],
    eps: float,
) -> float:
    r"""Return $\max_j(\theta'_j-\theta_j)^2/(\theta_j^2+\epsilon)$, (M34) floored.

    Over $\theta=(\lambda;s;\mathrm{vec}\,b)$, entries in any order.
    """
    worst = 0.0
    for a, b in zip(old, new, strict=True):
        if a is None or b is None or a.size == 0:
            continue
        worst = max(worst, float(np.max((b - a) ** 2 / (a**2 + eps))))
    return worst


# =========================================================================== MMLEFit


@dataclass(frozen=True, eq=False)
class MMLEFit:
    r"""Result of the marginal-likelihood estimator.

    Returned by [`fit_mmle`][mtdr.mmle.fit_mmle], [`ecme`][mtdr.mmle.ecme] and
    [`refine`][mtdr.mmle.refine], and accepted by the last two as a starting
    point; [`from_parameters`][mtdr.mmle.MMLEFit.from_parameters] builds one
    at given $(S,\lambda,b)$. Everything is in the estimator's **raw** frame.
    Per-regressor containers are tuples in `X` column order and every array is
    a read-only `float64` copy (`rank_deficient_neurons` is `int64`), also
    after unpickling. The constructor checks that the fields are consistent
    with `ranks` and with each other (`ParameterError`): the shapes, finite
    numbers, positive precisions, a symmetric `W_cov` that is positive
    definite to rounding, `objective <= log_likelihood` and the AIC
    arithmetic. It does not evaluate anything against data, nor check the
    value of `n_parameters` (a caller may carry another count); the validated
    entry point is [`from_parameters`][mtdr.mmle.MMLEFit.from_parameters].

    Attributes
    ----------
    S : tuple of numpy.ndarray
        `(n_bins, r_p)` temporal bases, the fitted parameters.
    noise_precision : numpy.ndarray
        `(n_neurons,)` noise precisions $\lambda_i>0$.
    intercept : numpy.ndarray or None
        `(n_neurons, n_bins)` intercept $b$, or `None` without one.
    ranks : tuple of int
        $r_p$, the widths of `S[p]` and `W[p]`.
    W : tuple of numpy.ndarray
        `(n_neurons, r_p)` posterior means of the weights at the fitted
        parameters, (M36).
    W_cov : numpy.ndarray
        `(n_neurons, total_rank, total_rank)` posterior covariances
        $C_i^{-1}$ of the stacked weight vectors, (M36); blocks in regressor
        order.
    log_likelihood : float
        Marginal log-likelihood at the fitted parameters, fully normalised,
        without the ridge.
    objective : float
        `log_likelihood - basis_ridge / 2 * sum(||S_p||^2)`, what the
        refinement maximised (equal to `log_likelihood` at `basis_ridge=0`).
    n_parameters : int
        The identifiable parameter count (M38a),
        [`n_parameters_mmle`][mtdr.aic.n_parameters_mmle].
    aic : float
        `2 * n_parameters - 2 * log_likelihood`.
    n_iter : Mapping of str to int
        Iterations used by the stages that produced the fit, `"ecme"` and
        `"refine"`, summed over repeated calls of a stage (read-only; empty
        for a fit built from parameters).
    converged : bool
        Provenance, not a certificate of stationarity: `True` iff no stage
        that produced the fit failed, that is, no ECME or refinement loop hit
        its cap, no ECME iteration raised the marginal NLL (not checked under
        `matlab_compat`), and every inner optimiser call succeeded with an
        accepted result and off its precision bound. A fit from `from_parameters`,
        or from zero-iteration stages, is `True`; one failed stage keeps every
        later fit built on it `False`.
    rank_deficient_neurons : numpy.ndarray
        `(n_flagged,)` int, the neurons the SVD initializer solved by minimum
        norm; empty when the fit did not start from `fit_svd`.

    Examples
    --------
    >>> from mtdr import simulate
    >>> from mtdr.mmle import MMLEFit
    >>> from mtdr.stats import sufficient_statistics
    >>> sim = simulate(n_neurons=8, n_bins=5, n_trials=40, ranks=[1, 2], seed=0)
    >>> stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    >>> fit = MMLEFit.from_parameters(stats, [sim.S["x0"], sim.S["x1"]],
    ...                               sim.noise_precision, sim.intercept)
    >>> fit
    MMLEFit(ranks=(1, 2), n_neurons=8, n_bins=5, intercept=True, ...)
    >>> fit.B[1].shape, fit.n_parameters == 8 + (5 + 9) + 40
    ((8, 5), True)
    """

    S: tuple[FloatArray, ...]
    noise_precision: FloatArray
    intercept: FloatArray | None
    ranks: tuple[int, ...]
    W: tuple[FloatArray, ...]
    W_cov: FloatArray
    log_likelihood: float
    objective: float
    n_parameters: int
    aic: float
    n_iter: Mapping[str, int]
    converged: bool
    rank_deficient_neurons: IntArray

    def __post_init__(self) -> None:
        """Check consistency with `ranks`; copy into read-only containers."""
        ranks = tuple(require_int_vector("ranks", self.ranks, minimum=0))
        P = len(ranks)
        lam = _float_array("noise_precision", self.noise_precision, 1)
        if lam.size == 0 or not (lam > 0).all():
            raise ParameterError(
                "noise_precision must be a non-empty array of positive numbers"
            )
        n = lam.size
        S = _array_tuple("S", self.S, P)
        T = S[0].shape[0]
        W = _array_tuple("W", self.W, P)
        for p, r in enumerate(ranks):
            if S[p].shape != (T, r) or W[p].shape != (n, r):
                raise ParameterError(
                    f"S[{p}] must be (n_bins, {r}) and W[{p}] ({n}, {r}) for "
                    f"ranks[{p}] = {r}; got {S[p].shape} and {W[p].shape}"
                )
        intercept = None
        if self.intercept is not None:
            intercept = _float_array("intercept", self.intercept, 2)
            if intercept.shape != (n, T):
                raise ParameterError(
                    f"intercept must be (n_neurons, n_bins) = {(n, T)}; got "
                    f"{intercept.shape}"
                )
        rtot = sum(ranks)
        cov = _float_array("W_cov", self.W_cov, 3)
        if cov.shape != (n, rtot, rtot):
            raise ParameterError(
                "W_cov must be (n_neurons, total_rank, total_rank) = "
                f"{(n, rtot, rtot)}; got {cov.shape}"
            )
        if rtot and np.abs(cov - cov.transpose(0, 2, 1)).max() > 1e-12 * max(
            float(np.abs(cov).max()), 1.0
        ):
            raise ParameterError("W_cov must be symmetric")
        if rtot:
            _require_positive_definite(cov)
        numbers: dict[str, float] = {}
        for name in ("log_likelihood", "objective", "aic"):
            value = as_real(getattr(self, name))
            if value is None or not np.isfinite(value):
                raise ParameterError(f"{name} must be a finite real number")
            numbers[name] = value
        ll = numbers["log_likelihood"]
        if numbers["objective"] > ll + 1e-12 * max(abs(ll), 1.0):
            raise ParameterError(
                "objective must not exceed log_likelihood: it is the log-likelihood "
                "minus a non-negative ridge"
            )
        k = as_int(self.n_parameters)
        if k is None or k < 0:
            raise ParameterError("n_parameters must be a non-negative integer")
        expected = 2.0 * k - 2.0 * numbers["log_likelihood"]
        if abs(numbers["aic"] - expected) > 1e-12 * max(abs(expected), 1.0):
            raise ParameterError("aic must equal 2 * n_parameters - 2 * log_likelihood")
        n_iter = _iteration_counts(self.n_iter)
        if not isinstance(self.converged, bool | np.bool_):
            raise ParameterError(f"converged must be a bool; got {self.converged!r}")
        deficient = np.array(self.rank_deficient_neurons, copy=True)
        if deficient.ndim != 1 or (
            deficient.size
            and (
                deficient.dtype.kind not in "iu"
                or not ((deficient >= 0) & (deficient < n)).all()
            )
        ):
            raise ParameterError(
                "rank_deficient_neurons must be a 1-D array of neuron indices"
            )
        values: dict[str, Any] = {
            "S": S,
            "noise_precision": lam,
            "intercept": intercept,
            "ranks": ranks,
            "W": W,
            "W_cov": cov,
            **numbers,
            "n_parameters": k,
            "n_iter": n_iter,
            "converged": bool(self.converged),
            "rank_deficient_neurons": deficient.astype(np.int64),
            "_B": tuple(W[p] @ S[p].T for p in range(P)),
        }
        freeze(values)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled object with its arrays read-only again."""
        restore_frozen(self, state)

    @property
    def B(self) -> tuple[FloatArray, ...]:  # noqa: N802 - the model's name, (M37)
        r"""Per regressor, `(n_neurons, n_bins)` coefficients $\hat W_pS_p^\top$, (M37).

        Derived once on construction, read-only; frame-independent.
        """
        out: tuple[FloatArray, ...] = self.__dict__["_B"]
        return out

    @property
    def n_neurons(self) -> int:
        """Number of neurons."""
        return int(self.noise_precision.size)

    @property
    def n_bins(self) -> int:
        """Number of time bins."""
        return int(self.S[0].shape[0])

    @property
    def total_rank(self) -> int:
        r"""$r_{tot}=\sum_pr_p$."""
        return int(sum(self.ranks))

    @classmethod
    def from_parameters(
        cls,
        stats: SufficientStats,
        S: Sequence[ArrayLike],
        noise_precision: ArrayLike,
        intercept: ArrayLike | None,
        *,
        basis_ridge: float = 0.0,
    ) -> MMLEFit:
        r"""Build a fit at given parameters: posterior, likelihood and counts there.

        Evaluates (M24) and (M36) at $(S,\lambda,b)$, so the result can start
        [`ecme`][mtdr.mmle.ecme] or [`refine`][mtdr.mmle.refine] from any point
        or score a parameter set. `n_iter` is empty, `converged` is `True` (no
        stage ran, so none failed; it does not say the point is stationary) and
        `rank_deficient_neurons` is empty. This is the validated way to build an
        `MMLEFit`: every field is computed from the statistics.

        Parameters
        ----------
        stats : SufficientStats
            The statistics the posterior and the likelihood are evaluated on.
        S : sequence of array_like
            Per regressor, the `(n_bins, r_p)` bases, as
            [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood]; each `r_p`
            at most `min(n_neurons, n_bins)`, as the count (M38a) requires.
        noise_precision : array_like of shape (n_neurons,)
            $\lambda_i>0$, as `marginal_log_likelihood`.
        intercept : array_like of shape (n_neurons, n_bins), or None
            $b$; `None` without the intercept.
        basis_ridge : float
            $g\ge0$, for `objective`.

        Returns
        -------
        MMLEFit
            The fit at the given parameters.

        Raises
        ------
        ParameterError
            As [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood],
            or for a rank above `min(n_neurons, n_bins)`.
        ValidationError, numpy.linalg.LinAlgError
            As [`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood].
        """
        stats, blocks, lam, xi, ups = _evaluation_args(
            S, noise_precision, intercept, stats
        )
        g = require_non_negative_real("basis_ridge", basis_ridge)
        b = None if intercept is None else np.asarray(intercept, dtype=np.float64)
        st = _state(blocks, lam, xi, ups, stats.XtX, stats.n_obs, stats.n_bins)
        return _assemble(stats, st, lam, b, g, {}, True, np.zeros(0, dtype=np.int64))

    def __repr__(self) -> str:
        """Summarise ranks and scores instead of printing the arrays."""
        return (
            f"MMLEFit(ranks={self.ranks}, n_neurons={self.n_neurons}, "
            f"n_bins={self.n_bins}, intercept={self.intercept is not None}, "
            f"log_likelihood={self.log_likelihood:.6g}, aic={self.aic:.6g}, "
            f"n_iter={dict(self.n_iter)}, converged={self.converged})"
        )


def _add_count(n_iter: Mapping[str, int], stage: str, count: int) -> dict[str, int]:
    """`n_iter` with `count` added to `stage`'s entry."""
    return {**n_iter, stage: n_iter.get(stage, 0) + count}


def _require_positive_definite(cov: FloatArray) -> None:
    r"""Raise unless every `W_cov[i]` is positive definite to rounding.

    A batched Cholesky factorisation decides; a slice it rejects passes if its
    smallest eigenvalue is at least $-10^{-12}$ times its largest, which
    accepts the computed $C_i^{-1}$ of an extremely ill-conditioned $C_i$.
    """
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w = np.linalg.eigvalsh(cov)  # error path only
        bad = np.flatnonzero(~(w[:, 0] >= -1e-12 * np.abs(w).max(axis=1)))
        if bad.size:
            raise ParameterError(
                f"W_cov must be positive definite; it is not for neurons {_listed(bad)}"
            ) from None


def _iteration_counts(value: object) -> ReadOnlyMapping[int]:
    if not isinstance(value, Mapping):
        raise ParameterError("n_iter must be a mapping of str to non-negative int")
    counts: dict[str, int] = {}
    for key, count in value.items():
        as_integer = as_int(count)
        if not isinstance(key, str) or as_integer is None or as_integer < 0:
            raise ParameterError("n_iter must map str to non-negative int")
        counts[key] = as_integer
    return ReadOnlyMapping(counts)


def _assemble(
    stats: SufficientStats,
    st: _State,
    lam: FloatArray,
    intercept: FloatArray | None,
    basis_ridge: float,
    n_iter: Mapping[str, int],
    converged: bool,
    deficient: IntArray,
) -> MMLEFit:
    """Build the `MMLEFit` at the point `st` was evaluated at."""
    n, T = stats.n_neurons, stats.n_bins
    bounds = st.blocks.bounds
    ranks = tuple(bounds[p + 1] - bounds[p] for p in range(len(bounds) - 1))
    log_likelihood = -(float(st.terms.sum()) + _constant(stats.n_obs, T))
    k = _aic.n_parameters_mmle(ranks, n, T, condition_independent=intercept is not None)
    return MMLEFit(
        S=tuple(_split(st.blocks.S, bounds)),
        noise_precision=lam,
        intercept=intercept,
        ranks=ranks,
        W=tuple(_split(st.mu, bounds)),
        W_cov=st.Cinv,
        log_likelihood=log_likelihood,
        objective=log_likelihood - 0.5 * basis_ridge * _ridge(st.blocks.S),
        n_parameters=k,
        aic=_aic.aic(log_likelihood, k),
        n_iter=n_iter,
        converged=converged,
        rank_deficient_neurons=deficient,
    )


# =========================================================================== ECME


def ecme(
    stats: SufficientStats,
    init: MMLEFit,
    *,
    max_iter: int = 100,
    tol: float = 1.0,
    convergence_eps: float = 1e-12,
    condition_independent: bool = True,
    matlab_compat: bool = False,
    verbose: int = 0,
) -> tuple[MMLEFit, FloatArray]:
    r"""ECME: closed-form conditional maximisation, `docs/model.md` § C.6, (M30)-(M34).

    One iteration maps $(\lambda,S,b)\mapsto(\lambda',S',b')$:

    1. **E-step** at $(\lambda,S,b)$: $C_i$, $u_i$, $\mu_i$, $\Omega_i$, (M22)-(M25).
    2. **Precisions**, (M30): $\lambda'_i=n_iT/\mathcal E_i$ with
       $\mathcal E_i=\upsilon_i(b)-2u_i^\top\mu_i
       +\mathrm{tr}(\Phi_i^\top\Phi_i\Omega_i)$.
    3. **Bases**, (M31), the multicycle form: the E-step is redone at
       $(\lambda',S)$ and the expected complete-data likelihood maximised over
       $S$, one symmetric positive-definite $r_{tot}\times r_{tot}$ system.
    4. **Intercept**, (M33): the exact maximiser of the marginal likelihood
       given $(\lambda',S')$, in the centred form of `update_intercept`
       (skipped without the intercept).
    5. **Stop** when $\max_j(\theta'_j-\theta_j)^2/(\theta_j^2+\epsilon)<$ `tol`
       over $\theta=(\lambda;s;\mathrm{vec}\,b)$, (M34) with the floor
       $\epsilon=$ `convergence_eps` for entries near zero, or after `max_iter`
       iterations.

    Each step conditionally maximises the expected complete-data or the
    marginal likelihood, so the marginal NLL (M24) does not increase; the
    returned trace shows it.

    Parameters
    ----------
    stats : SufficientStats
        The statistics.
    init : MMLEFit
        The starting point, from
        [`MMLEFit.from_parameters`][mtdr.mmle.MMLEFit.from_parameters] or a
        previous fit; its sizes and intercept must match `stats` and
        `condition_independent`.
    max_iter : int
        Iteration cap, `>= 0`; 0 returns the starting point re-evaluated
        (`docs/model.md` § C.6: zero steps return the initial state), which is
        not reported as non-convergence.
    tol : float
        Stopping threshold, finite and `> 0` (the reference's `stopcrit`,
        loose by default on purpose: ECME is a warm start).
    convergence_eps : float
        $\epsilon>0$, the floor of the stopping test's denominator.
    condition_independent : bool
        Whether the model has the intercept; must match `init.intercept`.
    matlab_compat : bool
        Reproduce the reference's stale-variable S- and b-steps (`ECMEtdr.m`)
        instead of (M31) and (M33): the S-step's right-hand side and second
        moment mix $C_i(\lambda)$ and $C_i(\lambda')$, its system is assembled
        with transposed diagonal blocks and solved as a general matrix, and the
        b-step uses $C_i^\times=I+\lambda'_i\mathbf S_{\rm old}K_i\mathbf
        S_{\rm new}^\top$ with $(C^\times_i)^{-\top}$ on its right-hand side.
        One sweep and the whole loop agree with the reference's `ECMEtdr`
        iterate by iterate to $10^{-12}$, with the same number of sweeps, on
        the reference run recorded in `tests/fixtures/mmle.mat`. For parity
        fixtures only: the trace is not guaranteed monotone, and an increase is
        not reported.
    verbose : int
        `0` silent; `1` one summary line; `2` also one line per iteration.

    Returns
    -------
    fit : MMLEFit
        The fit after the last iteration, with its posterior and likelihood;
        `n_iter["ecme"]` adds this call's iterations to `init`'s count, and
        `converged` is `False` if the cap was hit, an iteration raised the
        marginal NLL by more than `MONOTONE_RTOL` (relative to
        `max(|NLL|, number of observed entries)`), or `init.converged` was
        `False`. Under `matlab_compat=True` rises are not checked, so
        `converged=True` there does not certify a monotone trace.
    trace : numpy.ndarray
        `(n_iter + 1,)` fully normalised marginal NLL at the start and after
        each iteration (no ridge: ECME has none).

    Raises
    ------
    ParameterError
        For bad arguments, or an `init` that is not an `MMLEFit` or does not
        match `stats` or `condition_independent`.
    ValidationError
        For a neuron never observed; observed once while the model has an
        intercept (`docs/model.md` § 10.3); with zero energy about its mean
        (zero raw energy without the intercept); without residual degrees of
        freedom, its unconstrained least-squares residual on $[X_i\ \mathbf 1]$
        ($X_i$ without the intercept) zero to rounding, as whenever
        $n_i\le\operatorname{rank}[X_i\ \mathbf 1]$ (its marginal likelihood is
        unbounded); or whose expected residual $\mathcal E_i$ of (M30) is zero
        to rounding, $\mathcal E_i\le10^3\epsilon T\upsilon_i(b)$. Also when
        the basis system (M31) is singular because a regressor is zero on every
        observed trial, and when the likelihood is not representable in
        float64.
    numpy.linalg.LinAlgError
        If a posterior precision is not positive definite.

    Warns
    -----
    ConvergenceWarning
        When this call hit its cap or raised the NLL.

    Notes
    -----
    Cost per iteration: three batched factorisations of `(n, r_tot, r_tot)`
    precisions ((M22) at the current point, again at $\lambda'$, and the
    centred one of the intercept step) and one $r_{tot}\times r_{tot}$ solve,
    $O(n\,r_{tot}^3+n\,P\,T\,r_{tot}+r_{tot}^2T)$.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import simulate
    >>> from mtdr.mmle import MMLEFit, ecme
    >>> from mtdr.stats import sufficient_statistics
    >>> from mtdr.svd_fit import fit_svd
    >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=60, ranks=[1, 2], seed=0)
    >>> stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    >>> svd = fit_svd(stats, [1, 2])
    >>> init = MMLEFit.from_parameters(stats, svd.S, svd.noise_precision, stats.Y_mean)
    >>> fit, trace = ecme(stats, init)          # the reference's loose tol = 1.0
    >>> bool(np.all(np.diff(trace) <= 0)), fit.n_iter["ecme"] == trace.size - 1
    (True, True)
    >>> fit.converged
    True
    """
    stats = _require_stats(stats)
    ci = as_bool("condition_independent", condition_independent)
    compat = as_bool("matlab_compat", matlab_compat)
    cap = _require_count("max_iter", max_iter)
    threshold = _require_positive("tol", tol)
    eps = _require_positive("convergence_eps", convergence_eps)
    level = _require_verbose(verbose)
    _require_init(init, stats, ci)
    _require_fittable(stats, ci)

    A, n_obs, T = stats.XtX, stats.n_obs, stats.n_bins
    const = _constant(n_obs, T)
    scale = float(n_obs.sum()) * T
    blocks = _blocks(init.S)
    lam = np.array(init.noise_precision)
    b = None if init.intercept is None else np.array(init.intercept)
    xi, ups = stats.centered(b)
    st = _state(blocks, lam, xi, ups, A, n_obs, T)
    trace = [float(st.terms.sum()) + const]
    met = cap == 0
    increased: list[int] = []
    iteration = 0
    for iteration in range(1, cap + 1):
        lam_new = _precision_step(st, ups, n_obs, T)
        if not blocks.S.shape[1]:
            blocks_new = blocks
        elif compat:
            blocks_new = _basis_step_reference(st, lam_new, xi)
        else:
            blocks_new = _basis_step(st, lam_new, xi)
        b_new = None
        if ci:
            b_new = (
                _intercept_step_reference(blocks, blocks_new, lam_new, stats)
                if compat
                else _intercept_step(blocks_new, lam_new, stats)
            )
        change = _relative_change(
            (lam, blocks.S, b), (lam_new, blocks_new.S, b_new), eps
        )
        lam, blocks, b = lam_new, blocks_new, b_new
        xi, ups = stats.centered(b)
        st = _state(blocks, lam, xi, ups, A, n_obs, T)
        trace.append(float(st.terms.sum()) + const)
        rise = trace[-1] - trace[-2]
        if not compat and rise > MONOTONE_RTOL * max(abs(trace[-2]), scale):
            increased.append(iteration)
        if level >= 2:
            print(
                f"ecme: iteration {iteration}: nll {trace[-1]:.12g}, "
                f"change {change:.3g}"
            )
        if change < threshold:
            met = True
            break
    ok = met and not increased
    if not ok:
        reasons = []
        if not met:
            reasons.append(
                f"hit its iteration cap (max_iter = {cap}; `MTDR(ecme_max_iter=...)`) "
                f"before the change fell below {tol}"
            )
        if increased:
            reasons.append(f"raised the marginal NLL at iterations {increased}")
        warnings.warn("ECME " + " and ".join(reasons), ConvergenceWarning, stacklevel=2)
    if level:
        print(
            f"ecme: {iteration} iterations, nll {trace[0]:.12g} -> {trace[-1]:.12g}, "
            f"converged {ok}"
        )
    fit = _assemble(
        stats,
        st,
        lam,
        b,
        0.0,
        _add_count(init.n_iter, "ecme", iteration),
        ok and init.converged,
        init.rank_deficient_neurons,
    )
    return fit, np.array(trace)


# =========================================================================== refine


def refine(
    stats: SufficientStats,
    init: MMLEFit,
    *,
    max_iter: int = 10,
    tol: float = 1e-4,
    convergence_eps: float = 1e-12,
    optimizer_max_iter: int = 2000,
    optimizer_tol: float = 1e-6,
    optimizer_progtol: float = 1e-9,
    basis_ridge: float = 0.0,
    basis_span_scale: float = 1.0,
    condition_independent: bool = True,
    verbose: int = 0,
) -> MMLEFit:
    r"""Coordinate ascent on the marginal likelihood, `docs/model.md` § C.7.

    Repeats, from `init`:

    1. the intercept-centred statistics at the current $b$, (M13)-(M14);
    2. $S\leftarrow\arg\min_S\,\mathcal L(\lambda,S;b)+\tfrac g2\lVert s\rVert^2$
       by L-BFGS-B with the analytic gradient (M26b) (the reference's
       `minFunc`);
    3. $\lambda\leftarrow\arg\min_\lambda\mathcal L(\lambda,S;b)$ by L-BFGS-B over
       $\log\lambda$, with the gradient $\lambda_i\,\partial\mathcal
       L/\partial\lambda_i$ of (M27b), within `LOG_PRECISION_BOUND` of the
       current $\log\lambda_i$ (the reference's unconstrained `fminunc` can
       make $\lambda_i\le0$; see `docs/differences-from-matlab.md`);
    4. $b\leftarrow$ (M33) at the new $(S,\lambda)$ (skipped without the intercept);
    5. stop when the change (M34) of $(\lambda;s;\mathrm{vec}\,b)$, floored as
       in `ecme`, falls below `tol`, or after `max_iter` iterations.

    An L-BFGS-B result is installed only if its objective (the ridge included)
    is finite and no higher than at the current point, within `MONOTONE_RTOL`
    times $\max(\lvert f\rvert,\sum_in_iT)$; otherwise the current point is
    kept and the run counts as a failure. The intercept step is the exact
    conditional maximiser. So the result is never worse than `init`
    re-evaluated at `basis_ridge`, to that rounding allowance. The inner
    problems minimise (M24) as the reference computes it, without the
    $\tfrac12\sum_in_iT\log2\pi$ constant, so L-BFGS-B's relative
    function-decrease test sees the reference's function.

    Parameters
    ----------
    stats : SufficientStats
        The statistics.
    init : MMLEFit
        The starting point, usually the output of [`ecme`][mtdr.mmle.ecme].
    max_iter : int
        Iteration cap, `>= 0` (0 returns `init` re-evaluated).
    tol : float
        Stopping threshold on the change (M34), as in `ecme`, finite and `> 0`.
    convergence_eps : float
        $\epsilon>0$, the floor of the stopping test's denominator.
    optimizer_max_iter : int
        `maxiter` of each `scipy.optimize.minimize` call, positive (default
        2000: the first basis step at the paper's scale needs 582-909
        iterations; the reference's minFunc stops at 500, which the parity
        tests pass explicitly).
    optimizer_tol : float
        Finite and `> 0`: L-BFGS-B's projected-gradient test `gtol` (minFunc's
        `optTol` in the reference).
    optimizer_progtol : float
        Finite and `> 0`: the absolute function change at which an inner run
        stops, minFunc's `progTol`, applied as L-BFGS-B's
        `ftol = optimizer_progtol / max(|f0|, 1)` with `f0` the objective at
        the run's start, in the basis step (the precision step keeps a fixed
        relative `ftol = 1e-11`).
    basis_ridge : float
        $g\ge0$, the ridge on the bases; `objective` includes it,
        `log_likelihood` does not.
    basis_span_scale : float
        Finite and `> 0`: $\alpha$ of the basis step's preconditioning.
        Each basis run's L-BFGS-B works in variables in which every block's
        component in the span of the run's start $S_p$ is scaled by $\alpha$,
        stiffening the soft in-span directions $S_pM_p$; `1` (the default) is
        the plain run, the MATLAB reference's (`minFunc` on the raw bases).
        From a cold start (the SVD fit) `BASIS_SPAN_SCALE` reaches the same
        optimum in about 7-10x fewer evaluations; from a warm start whose new
        columns are not yet in their final span it can take more. It changes
        the optimiser's path, not its objective: `optimizer_tol` then applies
        to the scaled gradient, and a failure's projected-gradient entry is the
        scaled one.
    condition_independent : bool
        Whether the model has the intercept; must match `init.intercept`.
    verbose : int
        `0` silent; `1` one summary line; `2` also one line per iteration.

    Returns
    -------
    MMLEFit
        The refined fit. `n_iter["refine"]` adds this call's iterations to
        `init`'s count; `converged` is `False` if the cap was hit, an inner
        call reported failure, had its result rejected or ended on its
        precision bound in any iteration, or `init.converged` was `False`.

    Raises
    ------
    ParameterError, ValidationError, numpy.linalg.LinAlgError
        As [`ecme`][mtdr.mmle.ecme]; the `ValidationError` for an expected
        residual zero to rounding is also raised at the refined point, so no
        stage returns a fit at a diverging precision.

    Warns
    -----
    ConvergenceWarning
        When this call hit its cap or an inner call failed (a rounding-level
        line-search end is not a failure). Each failure names the iteration and
        the step, with SciPy's message, status, iteration count (`nit`) and the
        largest projected-gradient entry; a failed basis step at an accepted
        point also gives its last accepted decrease, and a failed precision
        step names the neurons with the largest
        $\lvert\lambda_i\mathcal E_i/(n_iT)-1\rvert$. Contact with the
        precision box is reported with its iteration; only contact in the
        final iteration is reported as a possibly unbounded likelihood.

    Notes
    -----
    Cost: an inner evaluation is one evaluation of (M24) with its gradient,
    $O(n\,r_{tot}^3+n\,P\,T\,r_{tot}+T\,r_{tot}^2)$; an iteration is two
    L-BFGS-B runs (tens of evaluations each) and one intercept step.

    Examples
    --------
    >>> from mtdr import simulate
    >>> from mtdr.mmle import MMLEFit, ecme, refine
    >>> from mtdr.stats import sufficient_statistics
    >>> from mtdr.svd_fit import fit_svd
    >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=60, ranks=[1, 2], seed=0)
    >>> stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    >>> svd = fit_svd(stats, [1, 2])
    >>> init = MMLEFit.from_parameters(stats, svd.S, svd.noise_precision, stats.Y_mean)
    >>> warm, _ = ecme(stats, init)
    >>> fit = refine(stats, warm)
    >>> bool(fit.log_likelihood >= warm.log_likelihood), fit.converged
    (True, True)
    """
    stats = _require_stats(stats)
    ci = as_bool("condition_independent", condition_independent)
    cap = _require_count("max_iter", max_iter)
    threshold = _require_positive("tol", tol)
    eps = _require_positive("convergence_eps", convergence_eps)
    inner_cap = _require_count("optimizer_max_iter", optimizer_max_iter, minimum=1)
    inner_tol = _require_positive("optimizer_tol", optimizer_tol)
    progtol = _require_positive("optimizer_progtol", optimizer_progtol)
    g = require_non_negative_real("basis_ridge", basis_ridge)
    span_scale = _require_positive("basis_span_scale", basis_span_scale)
    level = _require_verbose(verbose)
    _require_init(init, stats, ci)
    _require_fittable(stats, ci)

    A, n_obs, T = stats.XtX, stats.n_obs, stats.n_bins
    blocks = _blocks(init.S)
    lam = np.array(init.noise_precision)
    b = None if init.intercept is None else np.array(init.intercept)
    failures: list[str] = []
    on_bound = np.zeros(0, dtype=np.int64)
    met = cap == 0
    iteration = 0
    for iteration in range(1, cap + 1):
        xi, ups = stats.centered(b)
        blocks_new = blocks
        if blocks.S.shape[1]:
            blocks_new, message = _optimize_bases(
                blocks,
                lam,
                xi,
                ups,
                A,
                n_obs,
                T,
                g,
                inner_tol,
                inner_cap,
                progtol,
                span_scale,
            )
            if message:
                failures.append(f"iteration {iteration}, bases: {message}")
        lam_new, message, on_bound = _optimize_precision(
            blocks_new, lam, xi, ups, A, n_obs, T, inner_tol, inner_cap
        )
        if message:
            failures.append(f"iteration {iteration}, noise precision: {message}")
        if on_bound.size:
            failures.append(
                f"neurons {_listed(on_bound)} ended on the step bound in iteration "
                f"{iteration} (log precision moves at most {LOG_PRECISION_BOUND} "
                "per iteration)"
            )
        b_new = _intercept_step(blocks_new, lam_new, stats) if ci else None
        change = _relative_change(
            (lam, blocks.S, b), (lam_new, blocks_new.S, b_new), eps
        )
        lam, blocks, b = lam_new, blocks_new, b_new
        if level >= 2:
            print(f"refine: iteration {iteration}: change {change:.3g}")
        if change < threshold:
            met = True
            break
    xi, ups = stats.centered(b)
    st = _state(blocks, lam, xi, ups, A, n_obs, T)
    # No stage may return a fit at a diverging precision.
    _require_expected_residual(
        _expected_residual(st.K, st.u, st.mu, st.Cinv, ups), ups, T
    )
    ok = met and not failures
    if not ok:
        reasons = list(failures)
        if on_bound.size:
            reasons.append(
                f"neurons {_listed(on_bound)} were on the step bound in the final "
                "iteration, so their likelihood may be unbounded"
            )
        if not met:
            reasons.insert(
                0,
                f"hit its iteration cap (max_iter = {cap}; "
                f"`MTDR(refine_max_iter=...)`) before the change fell below {tol}",
            )
        warnings.warn(
            "refinement: " + "; ".join(reasons), ConvergenceWarning, stacklevel=2
        )
    fit = _assemble(
        stats,
        st,
        lam,
        b,
        g,
        _add_count(init.n_iter, "refine", iteration),
        ok and init.converged,
        init.rank_deficient_neurons,
    )
    if level:
        print(
            f"refine: {iteration} iterations, log-likelihood "
            f"{init.log_likelihood:.12g} -> {fit.log_likelihood:.12g}, converged {ok}"
        )
    return fit


class _Inner(NamedTuple):
    """The outcome of one inner L-BFGS-B run."""

    x: Float1D  # the accepted point: the optimiser's, or the start if rejected
    grad: Float1D  # the objective's gradient there
    failure: str  # empty on success
    status: object  # SciPy's status code (an int; "unknown" if absent)
    accepted: bool  # whether the optimiser's point was installed
    value: float  # the objective at `x`
    last_decrease: float  # f_{k-1} - f_k over the accepted iterates


def _ftol(f0: float, progtol: float | None) -> float:
    r"""L-BFGS-B's `ftol` for a run starting at objective value `f0`.

    `progtol` is an absolute function change, minFunc's `progTol`: L-BFGS-B
    stops when $(f_k-f_{k+1})/\max(\lvert f_k\rvert,\lvert f_{k+1}\rvert,1)\le$
    `ftol`, so `ftol = progtol / max(|f0|, 1)` stops it when the decrease per
    iteration falls to approximately `progtol`: exactly, when
    $f_k-f_{k+1}\le\texttt{progtol}\,\max(\lvert f_k\rvert,\lvert f_{k+1}\rvert,1)
    /\max(\lvert f_0\rvert,1)$, a ratio near 1 for a warm start. minFunc also
    stops on a small step or directional derivative, which this rule does not
    reproduce. `None` keeps the precision step's fixed relative `ftol`.
    """
    if progtol is None:
        return _LBFGS_FTOL
    scale = abs(f0) if np.isfinite(f0) else 1.0
    return progtol / max(scale, 1.0)


def _lbfgs(
    objective: Callable[[Float1D], tuple[float, Float1D]],
    x0: Float1D,
    bounds: scipy.optimize.Bounds | None,
    tol: float,
    max_iter: int,
    scale: float,
    progtol: float | None = None,
) -> _Inner:
    r"""Run L-BFGS-B from `x0`; install its result only if it is no worse.

    The function tolerance is `progtol / max(|f(x0)|, 1)` (`_ftol`). The
    optimiser's final point is accepted when its objective is finite and at
    most the start's plus `MONOTONE_RTOL` times `max(|f(x0)|, scale)` (the
    rounding allowance of ECME's monotonicity check); otherwise the start is
    kept. Both values are read from the evaluations L-BFGS-B made, or computed
    when it did not make them. A run that reports failure, or whose point is
    rejected, returns a message with SciPy's message, status, iteration count
    and the largest projected-gradient entry at the returned point.
    """
    seen: dict[bytes, tuple[float, Float1D]] = {}

    def recorded(x: Float1D) -> tuple[float, Float1D]:
        key = x.tobytes()
        if key not in seen:
            seen[key] = objective(x)
        value, grad = seen[key]
        return value, grad.copy()

    f0, g0 = recorded(x0)
    # The objective at every accepted iterate (from the evaluations L-BFGS-B
    # made), for the last accepted decrease.
    iterates: list[float] = []

    def accepted_iterate(xk: Float1D) -> None:
        iterates.append(recorded(np.asarray(xk, dtype=np.float64))[0])

    res = scipy.optimize.minimize(
        recorded,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        callback=accepted_iterate,
        options={
            "maxiter": max_iter,
            "gtol": tol,
            "ftol": _ftol(f0, progtol),
            "maxcor": _LBFGS_MEMORY,
        },
    )
    x1 = np.asarray(getattr(res, "x", x0), dtype=np.float64).reshape(x0.shape)
    candidate = seen.get(x1.tobytes())
    if candidate is None and np.isfinite(x1).all():
        try:
            candidate = objective(x1)
        except np.linalg.LinAlgError:
            candidate = None
    allowance = MONOTONE_RTOL * max(abs(f0), scale)
    if candidate is not None and bool(candidate[0] <= f0 + allowance):
        accepted, x, grad, value = True, x1, candidate[1], float(candidate[0])
    else:
        accepted, x, grad, value = False, x0, g0, float(f0)
    # f_{k-1} - f_k after k >= 1 accepted iterates; f_0 - f_end after none.
    history = [f0, *iterates]
    last = history[-2] - history[-1] if iterates else f0 - value
    status = getattr(res, "status", "unknown")
    if accepted and bool(getattr(res, "success", False)):
        return _Inner(x, grad, "", status, True, value, last)
    # The projected gradient, L-BFGS-B's stationarity measure (never empty:
    # the basis step is skipped at total rank 0).
    step = -grad if bounds is None else np.clip(x - grad, bounds.lb, bounds.ub) - x
    largest = float(np.max(np.abs(step)))
    message = str(getattr(res, "message", "")).strip() or "no message"
    nit = getattr(res, "nit", "unknown")
    text = (
        f"{message} (L-BFGS-B status {status}, nit {nit}, largest "
        f"projected-gradient entry {largest:.2e})"
    )
    if status == 1:  # name the knob
        text += (
            f"; hit its iteration cap (max_iter = {max_iter}; "
            "`MTDR(optimizer_max_iter=...)`)"
        )
    if not accepted:
        if candidate is None or not np.isfinite(candidate[0]):
            what = "is not finite"
        else:
            what = f"raised the objective from {f0:.12g} to {candidate[0]:.12g}"
        text += f"; its final point {what}, so the step was not taken"
    return _Inner(x, grad, text, status, accepted, value, last)


def _finite_or_inf(value: float, grad: FloatArray) -> tuple[float, Float1D]:
    """Return an inner objective's value and gradient, `(inf, 0)` if not finite."""
    flat: Float1D = np.ravel(grad)
    if np.isfinite(value) and np.isfinite(flat).all():
        return value, flat
    return np.inf, np.zeros_like(flat)


def _span_maps(
    blocks: _Blocks, alpha: float
) -> tuple[Callable[[Float1D], Float1D], Callable[[Float1D], Float1D]]:
    r"""Return $D$ and $D^{-1}$, which scale each block's span by $\alpha$.

    $D(Y)_p=(I+(\alpha-1)\Pi_p)Y_p$ with $\Pi_p$ the orthogonal projector onto
    the span of the start's $S_p$, on flattened `(T, r_tot)` arrays; $D$ is
    symmetric and $D^{-1}$ is the same with $1/\alpha$. In $y=D^{-1}s$ the
    curvature along the in-span directions $S_pM_p$ is $\alpha^2$ times larger,
    and every other direction's is unchanged.
    """
    T, r = blocks.S.shape
    eye = np.eye(T)
    columns: list[slice] = []
    up: list[FloatArray] = []  # I + (alpha - 1) Pi_p
    down: list[FloatArray] = []  # I + (1 / alpha - 1) Pi_p
    for p in range(len(blocks.bounds) - 1):
        cols = slice(blocks.bounds[p], blocks.bounds[p + 1])
        if cols.start == cols.stop:
            continue
        Q = scipy.linalg.orth(blocks.S[:, cols])
        proj = Q @ Q.T
        columns.append(cols)
        up.append(eye + (alpha - 1.0) * proj)
        down.append(eye + (1.0 / alpha - 1.0) * proj)

    def apply(x: Float1D, factors: list[FloatArray]) -> Float1D:
        X = np.array(x, dtype=np.float64).reshape(T, r)
        for cols, M in zip(columns, factors, strict=True):
            X[:, cols] = M @ X[:, cols]
        out: Float1D = X.ravel()
        return out

    return (lambda y: apply(y, up)), (lambda s: apply(s, down))


def _optimize_bases(
    blocks: _Blocks,
    lam: FloatArray,
    xi: FloatArray,
    ups: FloatArray,
    A: FloatArray,
    n_obs: NDArray[Any],
    T: int,
    g: float,
    tol: float,
    max_iter: int,
    progtol: float,
    span_scale: float = 1.0,
) -> tuple[_Blocks, str]:
    r"""L-BFGS-B over the stacked bases: (M24) plus the ridge, gradient (M26b).

    Stops on the gradient test `tol`, the absolute function change `progtol` or
    `max_iter`. An accepted line-search end (status 2) whose last accepted
    decrease is at rounding level is a success (`ROUNDING_PROGRESS_FACTOR`);
    any other failure counts, and an accepted status-2 failure reports the
    last accepted decrease $f_{k-1}-f_k$ ($f_0-f_{\rm end}$ after no accepted
    iterate), the objective's rounding $\epsilon M$ (`_basis_magnitude`) and
    `progtol`.

    With `span_scale` $\alpha\ne1$, L-BFGS-B runs on $y$ with $S=D(y)$
    (`_span_maps`): the objective and the acceptance are unchanged, and `tol`
    applies to the gradient in $y$, $D(\nabla_S)$. A rejected run returns the
    start exactly.
    """
    group, r = blocks.group, blocks.S.shape[1]
    Aexp = _expand(A, group)
    xig = xi[:, group, :]
    counts = n_obs.astype(np.float64) * T
    fixed = 0.5 * float(np.sum(-counts * np.log(lam) + lam * ups))

    def objective(x: Float1D) -> tuple[float, Float1D]:
        S = x.reshape(T, r)
        with np.errstate(over="ignore", invalid="ignore"):
            K = Aexp * (S.T @ S)
            u = np.einsum("ikt,tk->ik", xig, S)
            Cinv, mu, explained, logdet = _posterior(lam, K, u)
            value = fixed + 0.5 * float(np.sum(logdet - explained))
            grad = _grad_bases(S, Aexp, mu, Cinv, lam, xig)
            if g:
                value += 0.5 * g * _ridge(S)
                grad = grad + g * S
        return _finite_or_inf(value, grad)

    x0 = blocks.S.ravel()
    if span_scale == 1.0:
        inner = _lbfgs(objective, x0, None, tol, max_iter, counts.sum(), progtol)
    else:
        stretch, shrink = _span_maps(blocks, span_scale)

        def scaled(y: Float1D) -> tuple[float, Float1D]:
            value, grad = objective(stretch(y))
            return value, stretch(grad)  # D is symmetric: grad_y = D grad_S

        inner = _lbfgs(scaled, shrink(x0), None, tol, max_iter, counts.sum(), progtol)
        x = stretch(inner.x) if inner.accepted else x0
        inner = inner._replace(x=x, grad=shrink(inner.grad))
    S = np.ascontiguousarray(inner.x.reshape(T, r))
    message = inner.failure
    if message and inner.accepted and inner.status == 2:
        magnitude = _basis_magnitude(S, Aexp, xig, lam, ups, counts, g)
        if _basis_end_is_rounding(inner, progtol, magnitude):
            message = ""  # minFunc's normal "changing by less than progTol" end
        else:
            message += (
                f"; last accepted decrease {inner.last_decrease:.3g} (rounding "
                f"{_EPS * magnitude:.3g}, progtol {progtol:.3g})"
            )
    return _Blocks(S, group, blocks.bounds), message


def _basis_magnitude(
    S: FloatArray,
    Aexp: FloatArray,
    xig: FloatArray,
    lam: FloatArray,
    ups: FloatArray,
    counts: FloatArray,
    g: float,
) -> float:
    r"""Return $M=\sum_i\lvert$terms$_i\rvert$ of the basis objective at `S`.

    The objective $\tfrac12\sum_i[-n_iT\log\lambda_i+\lambda_i\upsilon_i+\log\lvert
    C_i\rvert-\lambda_i^2u_i^\top C_i^{-1}u_i]$ (plus the ridge) sums terms of both
    signs much larger than itself, so its evaluated value carries rounding of
    about $\epsilon M$, not $\epsilon\lvert f\rvert$.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        K = Aexp * (S.T @ S)
        u = np.einsum("ikt,tk->ik", xig, S)
        _, _, explained, logdet = _posterior(lam, K, u)
        terms = counts * np.abs(np.log(lam)) + lam * ups + np.abs(logdet) + explained
        out = 0.5 * float(np.sum(terms))
        if g:
            out += 0.5 * g * _ridge(S)
    return out


def _basis_end_is_rounding(inner: _Inner, progtol: float, magnitude: float) -> bool:
    r"""Whether an accepted status-2 basis-step end is a success.

    True iff the value and gradient at the accepted point are finite and the
    last accepted decrease is at most `ROUNDING_PROGRESS_FACTOR` times
    $\max(\texttt{progtol},\epsilon M)$, $M$ the objective's rounding scale
    `magnitude` there (`_basis_magnitude`; $M\ge\lvert f\rvert$).
    """
    if not (
        inner.accepted
        and inner.status == 2
        and np.isfinite(inner.value)
        and np.isfinite(inner.grad).all()
        and np.isfinite(magnitude)
    ):
        return False
    floor = max(progtol, _EPS * magnitude)
    return bool(inner.last_decrease <= ROUNDING_PROGRESS_FACTOR * floor)


def _log_precision_box(theta0: FloatArray) -> tuple[FloatArray, FloatArray]:
    r"""Return the precision step's box, $\theta^{(0)}\pm$ `LOG_PRECISION_BOUND`.

    Intersected with the log of the finite positive float64 range, without ever
    excluding the current value (a subnormal start is kept).
    """
    half = LOG_PRECISION_BOUND
    lower = np.maximum(theta0 - half, np.minimum(theta0, _LOG_FLOAT_MIN))
    upper = np.minimum(theta0 + half, np.maximum(theta0, _LOG_FLOAT_MAX))
    return lower, upper


def _optimize_precision(
    blocks: _Blocks,
    lam: FloatArray,
    xi: FloatArray,
    ups: FloatArray,
    A: FloatArray,
    n_obs: NDArray[Any],
    T: int,
    tol: float,
    max_iter: int,
) -> tuple[FloatArray, str, IntArray]:
    r"""L-BFGS-B over $\log\lambda$ in a box around the current value.

    Returns the precisions, the failure message (empty on success) and the
    neurons that ended on the box. A status-2 end (a rounding-level line
    search) at an accepted point is polished by `_newton_polish` (the result
    installed if its objective is at most $f_{\rm end}+10\epsilon M$, $M$ the
    objective's rounding scale `_precision_magnitude` at the end), and counts
    as success if every scale-free residual
    $\lvert\lambda_i\mathcal E_i/(n_iT)-1\rvert$ at the installed point is at
    most `PRECISION_RESIDUAL_TOL`. Status-0 and status-1 ends are not
    polished, so a converged step is unchanged.
    """
    K = _expand(A, blocks.group) * (blocks.S.T @ blocks.S)
    u = _project(xi, blocks)
    counts = n_obs.astype(np.float64) * T

    def objective(theta: Float1D) -> tuple[float, Float1D]:
        with np.errstate(over="ignore", invalid="ignore"):
            lam_ = np.exp(theta)
            Cinv, mu, explained, logdet = _posterior(lam_, K, u)
            value = 0.5 * float(
                np.sum(-counts * theta + logdet + lam_ * ups - explained)
            )
            expected = _expected_residual(K, u, mu, Cinv, ups)
            grad = 0.5 * (-counts + lam_ * expected)
        return _finite_or_inf(value, grad)

    theta0 = np.log(lam)
    lower, upper = _log_precision_box(theta0)
    inner = _lbfgs(
        objective,
        theta0,
        scipy.optimize.Bounds(lower, upper),
        tol,
        max_iter,
        counts.sum(),
        None,  # the fixed relative ftol; the absolute progtol is the basis step's
    )
    theta, grad = inner.x, inner.grad
    message = inner.failure
    if message and inner.accepted and inner.status == 2:
        # Polish a rounding-level end by diagonal Newton steps; install the
        # result if its objective is no worse beyond rounding.
        polished = _newton_polish(K, u, ups, counts, theta, lower, upper)
        if not np.array_equal(polished, theta):
            value, polished_grad = objective(polished)
            scale = max(
                abs(inner.value), _precision_magnitude(theta, K, u, ups, counts)
            )  # the objective's rounding scale
            limit = inner.value + _POLISH_ALLOWANCE * _EPS * scale
            if np.isfinite(value) and value <= limit:
                theta, grad = polished, polished_grad
    # The theta-gradient is (n_i T / 2) (lambda_i E_i / (n_i T) - 1).
    gap = 2.0 * grad / counts
    if (
        message
        and inner.accepted
        and inner.status == 2
        and float(np.max(np.abs(gap))) <= PRECISION_RESIDUAL_TOL
    ):
        message = ""  # stationary to rounding, a success
    if message:
        worst = np.argsort(-np.abs(gap), kind="stable")[:3]
        listed = ", ".join(f"{int(i)} ({gap[i]:.2e})" for i in worst)
        message += "; neurons with the largest |lambda_i E_i / (n_i T) - 1|: " + listed
    at_bound = np.flatnonzero((theta <= lower) | (theta >= upper)).astype(np.int64)
    # A precision the step did not move (a rejected result, or an entry
    # L-BFGS-B left in place) is returned exactly, not as exp(log(lambda)),
    # which can differ from it in the last bit.
    out: FloatArray = np.where(theta == theta0, lam, np.exp(theta))
    return out, message, at_bound


def _precision_magnitude(
    theta: FloatArray, K: FloatArray, u: FloatArray, ups: FloatArray, counts: FloatArray
) -> float:
    r"""Return $M=\sum_i\lvert$terms$_i\rvert$ of the precision objective.

    $\tfrac12\sum_i[n_iT\lvert\theta_i\rvert+\lvert\log\lvert C_i\rvert\rvert
    +\lambda_i\upsilon_i+\lambda_i^2u_i^\top C_i^{-1}u_i]$: the evaluated objective
    carries rounding of about $\epsilon M$, the scale of the polish's "no worse"
    allowance.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        lam = np.exp(theta)
        _, _, explained, logdet = _posterior(lam, K, u)
        terms = counts * np.abs(theta) + np.abs(logdet) + lam * ups + explained
    return 0.5 * float(np.sum(terms))


def _precision_derivatives(
    theta: FloatArray, K: FloatArray, u: FloatArray, ups: FloatArray, counts: FloatArray
) -> tuple[FloatArray, FloatArray]:
    r"""Gradient and diagonal Hessian of the precision step's objective.

    Per neuron, with $\lambda=e^\theta$, $C=I+\lambda K$, $\mu=\lambda C^{-1}u$
    and $\mathcal E$ the expected residual energy of (M27b),
    $g=\tfrac12(\lambda\mathcal E-n_iT)$ and, since
    $d\mu/d\theta=C^{-1}\mu$ and $dC^{-1}/d\theta=-\lambda C^{-1}KC^{-1}$,

    $$
    \frac{d\mathcal E}{d\theta}=2(C^{-1}\mu)^\top(K\mu-u)
    -\lambda\,\mathrm{tr}\big[(KC^{-1})^2\big],\qquad
    h=\frac{dg}{d\theta}=\tfrac12\lambda\Big(\mathcal E
    +\frac{d\mathcal E}{d\theta}\Big),
    $$

    `docs/model.md` § E.3. The problem is separable across neurons given $S$, so
    the Hessian is diagonal.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        lam = np.exp(theta)
        Cinv, mu, _, _ = _posterior(lam, K, u)
        E = _expected_residual(K, u, mu, Cinv, ups)
        Cmu = np.einsum("ijk,ik->ij", Cinv, mu)
        Kmu = np.einsum("ijk,ik->ij", K, mu)
        KC = K @ Cinv
        dE = 2.0 * np.einsum("ij,ij->i", Cmu, Kmu - u) - lam * np.einsum(
            "ijk,ikj->i", KC, KC
        )
        grad = 0.5 * (lam * E - counts)
        hess = 0.5 * lam * (E + dE)
    return grad, hess


def _newton_polish(
    K: FloatArray,
    u: FloatArray,
    ups: FloatArray,
    counts: FloatArray,
    theta0: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> FloatArray:
    r"""Polish a status-2 precision end by diagonal Newton steps.

    At most `POLISH_MAX_ITER` steps
    $\theta_i\leftarrow\theta_i-g_i/h_i$, each clipped to
    $\lvert\Delta\theta_i\rvert\le0.5$ and to the box; a neuron with
    $h_i\le0$ or on the box takes no step. Stops once every scale-free
    residual $2\lvert g_i\rvert/(n_iT)$ is at most $10^{-10}$. The caller
    installs the result only if its objective is no worse beyond rounding.
    """
    theta = np.array(theta0, dtype=np.float64)
    for _ in range(POLISH_MAX_ITER):
        grad, hess = _precision_derivatives(theta, K, u, ups, counts)
        if float(np.max(np.abs(2.0 * grad / counts))) <= _POLISH_TARGET:
            break
        free = (theta > lower) & (theta < upper) & np.isfinite(hess) & (hess > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.clip(-grad / hess, -_POLISH_STEP, _POLISH_STEP)
        theta = np.clip(theta + np.where(free, step, 0.0), lower, upper)
    return theta


# =========================================================================== fit_mmle


def fit_mmle(
    stats: SufficientStats,
    ranks: Sequence[int] | NDArray[np.integer[Any]],
    init: MMLEFit | SVDFit | None = None,
    *,
    condition_independent: bool = True,
    ecme_max_iter: int = 100,
    ecme_tol: float = 1.0,
    refine_max_iter: int = 10,
    refine_tol: float = 1e-4,
    convergence_eps: float = 1e-12,
    optimizer_max_iter: int = 2000,
    optimizer_tol: float = 1e-6,
    optimizer_progtol: float = 1e-9,
    ridge: float = 0.0,
    basis_ridge: float = 0.0,
    basis_span_scale: float = 1.0,
    verbose: int = 0,
) -> MMLEFit:
    r"""Fit the marginal-likelihood estimator at fixed ranks, `docs/model.md` § 5.

    The reference's `MMLE_CoordAscentWrapper`:

    1. **Initialise**, (M29): [`fit_svd`][mtdr.svd_fit.fit_svd] at the same
       ranks (the intercept handled inside it as a full-rank regressor), then
       $S^{(0)}_p=\hat S^{svd}_p$, $\lambda^{(0)}=\hat\lambda^{svd}$ and
       $b^{(0)}_i=\bar y_i$ (the SVD's own intercept is discarded, as in the
       reference).
    2. [`ecme`][mtdr.mmle.ecme] to the loose tolerance `ecme_tol`.
    3. [`refine`][mtdr.mmle.refine] to `refine_tol`.
    4. The posterior of the weights (M36) and the scores at the result, with
       the identifiable count (M38a).

    Parameters
    ----------
    stats : SufficientStats
        The statistics.
    ranks : sequence of int or 1-D integer array
        $r_p$ per regressor, `0 <= r_p <= min(n_neurons, n_bins)`.
    init : MMLEFit, SVDFit or None
        `None` runs `fit_svd` (with `ridge`); an `SVDFit` is used as step 1's
        initializer; an `MMLEFit` skips step 1 and starts ECME from it. Its
        ranks must equal `ranks` and its intercept match
        `condition_independent`.
    condition_independent : bool
        Fit the intercept.
    ecme_max_iter : int
        ECME's cap, `>= 0`: `max_iter` of `ecme`.
    ecme_tol : float
        ECME's stopping threshold on the change (M34): `tol` of `ecme`.
    refine_max_iter : int
        The refinement's cap: `max_iter` of `refine`.
    refine_tol : float
        The refinement's stopping threshold on the change (M34): `tol` of
        `refine`.
    convergence_eps : float
        The stopping test's $\epsilon$, shared by both loops.
    optimizer_max_iter : int
        `maxiter` of every L-BFGS-B call of `refine` (default 2000).
    optimizer_tol : float
        Their projected-gradient test `gtol`.
    optimizer_progtol : float
        The absolute function change at which the basis step stops.
    ridge : float
        The SVD initializer's ridge, `ridge` of `fit_svd`.
    basis_ridge : float
        $g$, the ridge on the bases in the refinement.
    basis_span_scale : float
        $\alpha$ of the refinement's basis-step preconditioning: `1` (the
        default) is off, the MATLAB reference's basis step; `BASIS_SPAN_SCALE`
        suits a fit from its SVD start (`init` `None` or an `SVDFit`), not a
        warm `MMLEFit` start whose new columns are far from their final span.
        See [`refine`][mtdr.mmle.refine].
    verbose : int
        `0`, `1` or `2`, passed to `ecme` and `refine`.

    Returns
    -------
    MMLEFit
        The fit; `n_iter` holds `"ecme"` and `"refine"` (added to an
        `MMLEFit` start's counts), and `converged` covers both loops and every
        inner optimiser call; it reports that no stage failed, not that
        the result is stationary (zero caps return the start with `True`).

    Raises
    ------
    ParameterError
        For bad arguments or ranks, or an `init` that does not match.
    ValidationError, numpy.linalg.LinAlgError
        As [`fit_svd`][mtdr.svd_fit.fit_svd] and [`ecme`][mtdr.mmle.ecme].

    Warns
    -----
    DesignWarning
        From `fit_svd`, for rank-deficient neurons.
    ConvergenceWarning
        From `ecme` or `refine`.

    Notes
    -----
    The `05_benchmarks` example times fits at the demo's and the paper's scale.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import simulate
    >>> from mtdr.mmle import fit_mmle
    >>> from mtdr.stats import sufficient_statistics
    >>> sim = simulate(n_neurons=30, n_bins=10, n_trials=200, ranks=[2, 1],
    ...                drop_prob=0.2, seed=1)
    >>> stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    >>> fit = fit_mmle(stats, [2, 1])
    >>> fit
    MMLEFit(ranks=(2, 1), n_neurons=30, n_bins=10, intercept=True, ...)
    >>> err = np.linalg.norm(fit.B[0] - sim.B["x0"]) / np.linalg.norm(sim.B["x0"])
    >>> bool(err < 0.1), fit.converged
    (True, True)
    """
    stats = _require_stats(stats)
    ci = as_bool("condition_independent", condition_independent)
    n, T = stats.n_neurons, stats.n_bins
    rank_list = require_int_vector("ranks", ranks, minimum=0)
    if len(rank_list) != stats.n_regressors:
        raise ParameterError(
            f"ranks has {len(rank_list)} entries; expected one per regressor "
            f"({stats.n_regressors}), without the intercept"
        )
    for p, r in enumerate(rank_list):
        if r > min(n, T):
            raise ParameterError(
                f"ranks[{p}] is {r}, above min(n_neurons, n_bins) = {min(n, T)}"
            )
    common: dict[str, Any] = {
        "convergence_eps": convergence_eps,
        "condition_independent": ci,
        "verbose": verbose,
    }
    # Before the SVD start, so that an unfittable neuron is not first reported
    # as rank-deficient.
    _require_fittable(stats, ci)
    if init is None or isinstance(init, SVDFit):
        if init is None:
            svd = fit_svd(stats, rank_list, ridge=ridge, condition_independent=ci)
        else:
            _require_svd_init(init, stats, rank_list, ci)
            svd = init
        start = MMLEFit.from_parameters(
            stats, svd.S, svd.noise_precision, stats.Y_mean if ci else None
        )
        start = _with_deficient(start, svd.rank_deficient_neurons)
    elif isinstance(init, MMLEFit):
        if list(init.ranks) != rank_list:
            raise ParameterError(
                f"init has ranks {init.ranks}; expected {tuple(rank_list)}"
            )
        start = init
    else:
        raise ParameterError(
            f"init must be an MMLEFit, an SVDFit or None; got {type(init).__name__}"
        )
    warm, _ = ecme(stats, start, max_iter=ecme_max_iter, tol=ecme_tol, **common)
    return refine(
        stats,
        warm,
        max_iter=refine_max_iter,
        tol=refine_tol,
        optimizer_max_iter=optimizer_max_iter,
        optimizer_tol=optimizer_tol,
        optimizer_progtol=optimizer_progtol,
        basis_ridge=basis_ridge,
        basis_span_scale=basis_span_scale,
        **common,
    )


def _with_deficient(fit: MMLEFit, deficient: IntArray) -> MMLEFit:
    """`fit` recording the SVD initializer's flagged neurons."""
    fields = {name: getattr(fit, name) for name in fit.__dataclass_fields__}
    fields["rank_deficient_neurons"] = deficient
    return MMLEFit(**fields)


# =========================================================================== validation


def _require_stats(stats: object) -> SufficientStats:
    if not isinstance(stats, SufficientStats):
        raise ParameterError(
            "stats must be a SufficientStats (from mtdr.stats.sufficient_statistics); "
            f"got {type(stats).__name__}"
        )
    return stats


def _evaluation_args(
    S: object, noise_precision: object, intercept: object, stats: object
) -> tuple[SufficientStats, _Blocks, FloatArray, FloatArray, FloatArray]:
    """Validate the evaluators' arguments: stats, bases, precisions, centred stats."""
    stats = _require_stats(stats)
    bases = _require_bases(S, stats.n_regressors, stats.n_bins)
    lam = _require_precision(noise_precision, stats.n_neurons)
    xi, ups = stats.centered(intercept)  # type: ignore[arg-type]
    return stats, _blocks(bases), lam, xi, ups


def _require_bases(value: object, n_regressors: int, n_bins: int) -> list[FloatArray]:
    entries = as_list(value)
    if entries is None:
        raise ParameterError(
            "S must be a sequence of (n_bins, r_p) arrays, one per regressor; "
            f"got {type(value).__name__}"
        )
    if len(entries) != n_regressors:
        raise ParameterError(
            f"S has {len(entries)} blocks; expected one per regressor ({n_regressors})"
        )
    out = []
    for p, block in enumerate(entries):
        arr = np.asarray(block)
        if arr.dtype.kind not in "iuf" or arr.ndim != 2 or arr.shape[0] != n_bins:
            raise ParameterError(
                f"S[{p}] must be a real (n_bins, r_p) = ({n_bins}, r_p) array; got "
                f"dtype {arr.dtype}, shape {arr.shape}"
            )
        arr = arr.astype(np.float64, copy=False)
        if not np.isfinite(arr).all():
            raise ParameterError(f"S[{p}] must be finite")
        out.append(arr)
    return out


def _require_precision(value: object, n_neurons: int) -> FloatArray:
    arr = np.asarray(value)
    if arr.dtype.kind not in "iuf" or arr.shape != (n_neurons,):
        raise ParameterError(
            "noise_precision must be a real array of shape (n_neurons,) = "
            f"({n_neurons},); got dtype {arr.dtype}, shape {arr.shape}"
        )
    out = arr.astype(np.float64, copy=False)
    bad = np.flatnonzero(~(np.isfinite(out) & (out > 0)))
    if bad.size:
        raise ParameterError(
            f"noise_precision must be finite and > 0; it is not for neurons "
            f"{_listed(bad)}"
        )
    return out


def _require_count(name: str, value: object, minimum: int = 0) -> int:
    count = as_int(value)
    if count is None or count < minimum:
        raise ParameterError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return count


def _require_positive(name: str, value: object) -> float:
    number = as_real(value)
    if number is None or not np.isfinite(number) or number <= 0:
        raise ParameterError(f"{name} must be a finite real number > 0; got {value!r}")
    return number


def _require_verbose(value: object) -> int:
    level = as_int(value)
    if level is None or level not in (0, 1, 2):
        raise ParameterError(f"verbose must be 0, 1 or 2; got {value!r}")
    return level


def _require_init(init: object, stats: SufficientStats, ci: bool) -> None:
    if not isinstance(init, MMLEFit):
        raise ParameterError(
            "init must be an MMLEFit (see MMLEFit.from_parameters); got "
            f"{type(init).__name__}"
        )
    if (init.n_neurons, init.n_bins, len(init.ranks)) != (
        stats.n_neurons,
        stats.n_bins,
        stats.n_regressors,
    ):
        raise ParameterError(
            f"init has {init.n_neurons} neurons, {init.n_bins} bins and "
            f"{len(init.ranks)} regressors; stats has {stats.n_neurons}, "
            f"{stats.n_bins} and {stats.n_regressors}"
        )
    if (init.intercept is not None) != ci:
        has = "has an" if init.intercept is not None else "has no"
        raise ParameterError(f"condition_independent={ci}, but init {has} intercept")


def _require_svd_init(
    svd: SVDFit, stats: SufficientStats, ranks: list[int], ci: bool
) -> None:
    sizes = (*svd.B_full[0].shape, len(svd.ranks))
    if sizes != (stats.n_neurons, stats.n_bins, stats.n_regressors):
        raise ParameterError("init (an SVDFit) does not match the sizes of stats")
    if list(svd.ranks) != ranks:
        raise ParameterError(f"init has ranks {svd.ranks}; expected {tuple(ranks)}")
    if (svd.intercept is not None) != ci:
        raise ParameterError(
            f"condition_independent={ci} does not match the SVDFit initializer"
        )


def _require_observed(stats: SufficientStats) -> None:
    never = np.flatnonzero(stats.n_obs == 0)
    if never.size:
        raise ValidationError(
            f"neurons {_listed(never)} are never observed; drop them before fitting"
        )


def _require_fittable(stats: SufficientStats, ci: bool) -> None:
    r"""Reject neurons the marginal likelihood has no finite maximiser for.

    In order: never observed; observed once with the intercept
    (`docs/model.md` § 10.3); zero energy about the mean, or zero raw energy
    without the intercept; and no residual degrees of freedom: the
    unconstrained least-squares residual on $[X_i\ \mathbf 1]$ ($X_i$ without
    the intercept) is zero to rounding, at most $10^3\epsilon T$ times the
    energy it is computed from, as it is whenever
    $n_i\le\operatorname{rank}[X_i\ \mathbf 1]$. Such a neuron's centred
    responses lie in the column space of its design, so a basis column can be
    turned towards its exact fit and $\lambda_i\to\infty$ raises its marginal
    likelihood without bound.
    """
    _require_observed(stats)
    if ci:
        once = np.flatnonzero(stats.n_obs < 2)
        if once.size:
            raise ValidationError(
                f"neurons {_listed(once)} are observed on one trial; with the "
                "intercept their marginal likelihood has no finite maximiser "
                "(docs/model.md section 10.3): drop them"
            )
    energy = stats.YtY_c if ci else stats.YtY_raw
    flat = np.flatnonzero(~(energy > 0))
    if flat.size:
        about = " about their means" if ci else ""
        raise ValidationError(
            f"neurons {_listed(flat)} have zero energy{about} on their observed "
            "trials, so their noise precision is unbounded: drop them"
        )
    rss, energy = _unconstrained_residual(stats, ci)
    exact = np.flatnonzero(_zero_to_rounding(rss, energy, stats.n_bins))
    if exact.size:
        design = "[X_i 1]" if ci else "X_i"
        raise ValidationError(
            f"neurons {_listed(exact)} have no residual degrees of freedom: least "
            f"squares on their observed design {design} fits their responses "
            f"exactly (as it does whenever n_i <= rank {design}, at most "
            f"{stats.n_regressors + int(ci)} here), so their marginal likelihood is "
            "unbounded (docs/model.md section 10.3): drop them or require more "
            "observations per neuron"
        )


def _require_finite_gradient(grad: FloatArray) -> None:
    if not np.isfinite(grad).all():
        raise ValidationError(
            "the gradient of the marginal likelihood is not representable in "
            "float64 at these parameters (it overflows): rescale Y, or the noise "
            "precisions and bases with it (docs/model.md section 10.2)"
        )


def _float_array(name: str, value: object, ndim: int) -> FloatArray:
    try:
        arr = np.array(value, copy=True)
    except (TypeError, ValueError) as err:
        raise ParameterError(f"{name} is not an array: {err}") from err
    if arr.dtype.kind not in "iuf" or arr.ndim != ndim:
        raise ParameterError(
            f"{name} must be a real {ndim}-D array; got dtype {arr.dtype}, "
            f"shape {arr.shape}"
        )
    out = arr.astype(np.float64, copy=False)
    if not np.isfinite(out).all():
        raise ParameterError(f"{name} must be finite")
    return out


def _array_tuple(name: str, value: object, length: int) -> tuple[FloatArray, ...]:
    entries = as_list(value)
    if entries is None or len(entries) != length:
        raise ParameterError(
            f"{name} must be a sequence of {length} 2-D arrays, one per regressor"
        )
    return tuple(_float_array(f"{name}[{p}]", e, 2) for p, e in enumerate(entries))


def _listed(indices: NDArray[np.integer[Any]]) -> str:
    items = [int(i) for i in indices]
    head = ", ".join(str(i) for i in items[:_MAX_LISTED])
    more = f" and {len(items) - _MAX_LISTED} more" if len(items) > _MAX_LISTED else ""
    return f"[{head}{more}]"
