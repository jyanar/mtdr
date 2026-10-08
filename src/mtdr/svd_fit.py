r"""Reduced-rank regression: the SVD estimator (`docs/model.md` § 3).

Per neuron, ordinary (or ridge) least squares of the observed responses on the
design gives the unconstrained coefficients $\hat B_p$, (M15)-(M16); each
$\hat B_p$ is truncated to rank $r_p$ by its singular value decomposition,
(M17)-(M18), optionally after weighting each neuron's row by its precision,
(M18w); the intercept is refitted given the truncated blocks; the noise
precision of each neuron is the inverse residual variance of that fit,
computed with the **aligned** residual (M19b); and the fit is scored by the
plug-in Gaussian log-likelihood and the textbook parameter count, (M21a)-(M21b).
This is the reference's `SVDRegressB`, `SVDRegress_S_Vdata` and `SVDRegB_AIC`
with the four `SVDRegB_AIC` defects corrected (it misaligns the residual,
scrambles neurons when reshaping, counts regressors in place of observations
and flips the sign of the log-precision term; see
`docs/differences-from-matlab.md`) and the intercept refitted as above; the
replica of the defective objective lives in the test suite
(`tests/matlab_compat.py`).

Everything is computed from the [`SufficientStats`][mtdr.stats.SufficientStats]
(the reference's `XX`/`XY` block matrices, $nPT\times nPT$ for `XX`, are never
formed), batched over neurons; with the intercept, the solve and the residual
use the moments about the neurons' means, so a shifted regressor or a large
baseline costs no accuracy.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mtdr import aic as _aic
from mtdr._args import as_bool, require_int_vector, require_non_negative_real
from mtdr._frozen import restore_frozen
from mtdr.errors import DesignWarning, ParameterError, ValidationError
from mtdr.stats import SufficientStats

__all__ = ["RANK_TOLERANCE", "SVDFit", "fit_svd"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

RANK_TOLERANCE = 1e-10
"""Threshold of the scale-invariant rank test.

A neuron's design is rank-deficient when the smallest eigenvalue of its
unit-diagonal (equilibrated) Gram matrix is at or below this.
"""

#: A neuron is degenerate when its residual is at most this many times
#: `eps * n_bins` times the energy its residual is computed from.
_ROUNDING_MARGIN = 1e3

_EPS = float(np.finfo(np.float64).eps)
_LOG_2PI = float(np.log(2.0 * np.pi))
_MAX_LISTED = 20


@dataclass(frozen=True, eq=False)
class SVDFit:
    r"""Result of [`fit_svd`][mtdr.svd_fit.fit_svd].

    Per-regressor tuples are in `X` column order and exclude the intercept.
    Arrays are read-only, also after unpickling.

    Attributes
    ----------
    W : tuple of numpy.ndarray
        `(n_neurons, r_p)` left factors $U_p\Sigma_p^{1/2}$ of the truncated
        SVD of `B_full[p]`, (M17); `(n_neurons, 0)` at rank 0. Signs are
        deterministic: the entry of largest absolute value of each column of
        $U_p$ is positive. With `precision_weighted`, $D^{-1}U_p\Sigma_p^{1/2}$
        with $U_p$, $\Sigma_p$ from the SVD of $D\,$`B_full[p]`.
    S : tuple of numpy.ndarray
        `(n_bins, r_p)` temporal factors $V_p\Sigma_p^{1/2}$, (M17), the
        reference's `wt{p}`; columns ordered by decreasing singular value.
    B : tuple of numpy.ndarray
        `(n_neurons, n_bins)` rank-`r_p` coefficients `W[p] @ S[p].T`, (M18):
        the best rank-`r_p` approximation of `B_full[p]` in the Frobenius norm,
        or, with `precision_weighted`, in the row-weighted norm of (M18w).
    B_full : tuple of numpy.ndarray
        `(n_neurons, n_bins)` unconstrained per-neuron least-squares
        coefficients, (M15)-(M16).
    intercept : numpy.ndarray or None
        `(n_neurons, n_bins)` condition-independent term refitted given the
        truncated `B`, (M19b):
        $\hat b_i=\frac{n_i}{n_i+\gamma}\big(\bar y_i-\sum_p\bar x_{ip}B_p[i,:]\big)$,
        the exact maximiser over $b$ of the plug-in likelihood at `B`; `None`
        when fitted with `condition_independent=False`.
    intercept_full : numpy.ndarray or None
        `(n_neurons, n_bins)` the intercept of the unconstrained fit, the
        coefficient of the appended column of ones (the reference's `b0`,
        $\frac{n_i}{n_i+\gamma}(\bar y_i-\sum_p\bar x_{ip}\hat B_p[i,:])$);
        equal to `intercept` at full rank. `None` without an intercept.
    noise_precision : numpy.ndarray
        `(n_neurons,)`, $\hat\lambda_i=n_iT/\mathrm{RSS}_i$ with the aligned
        residual of the rank-truncated fit and `intercept`, (M19b).
    log_likelihood : float
        Plug-in Gaussian log-likelihood of the observed entries at
        $(B, \hat b, \hat\lambda)$, fully normalised:
        $-\tfrac12\sum_in_iT\,[\log(\mathrm{RSS}_i/n_iT)+1+\log2\pi]$, (M21a).
    n_parameters : int
        `aic.n_parameters_svd(ranks, ...)`, the textbook count (M21b).
    aic : float
        `2 * n_parameters - 2 * log_likelihood`, (M21b).
    rank_deficient_neurons : numpy.ndarray
        `(n_flagged,)` int, neurons whose observed design was rank-deficient
        and which got the minimum-norm least-squares solution.
    ranks : tuple of int
        The ranks fitted, one per regressor.
    precision_weighted : bool
        Whether `B` is the precision-weighted truncation.
    """

    W: tuple[FloatArray, ...]
    S: tuple[FloatArray, ...]
    B: tuple[FloatArray, ...]
    B_full: tuple[FloatArray, ...]
    intercept: FloatArray | None
    intercept_full: FloatArray | None
    noise_precision: FloatArray
    log_likelihood: float
    n_parameters: int
    aic: float
    rank_deficient_neurons: IntArray
    ranks: tuple[int, ...]
    precision_weighted: bool = False

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled object with its arrays read-only again."""
        restore_frozen(self, state)

    def __repr__(self) -> str:
        """Summarise ranks and scores instead of printing the arrays."""
        n_neurons, n_bins = self.B_full[0].shape
        return (
            f"SVDFit(ranks={self.ranks}, n_neurons={n_neurons}, n_bins={n_bins}, "
            f"intercept={self.intercept is not None}, "
            f"log_likelihood={self.log_likelihood:.6g}, aic={self.aic:.6g}, "
            f"rank_deficient_neurons={self.rank_deficient_neurons.size})"
        )


def fit_svd(
    stats: SufficientStats,
    ranks: Sequence[int] | NDArray[np.integer[Any]],
    *,
    ridge: float = 0.0,
    condition_independent: bool = True,
    precision_weighted: bool = False,
) -> SVDFit:
    r"""Fit the SVD / reduced-rank estimator, (M15)-(M21).

    1. **Least squares, (M15)-(M16).** For each neuron, with the design
       $X^f_i=[X_i\ \mathbf 1]$ of its observed trials (the column of ones only
       when `condition_independent=True`), $\hat\beta_i=(A^f_i+\gamma I)^{-1}
       \xi^f_i$ gives the rows `B_full[p][i, :]` and the intercept
       `intercept_full[i, :]`. The ridge $\gamma$ is added to every
       coefficient, the intercept's included, as `SVDRegressB` does. With the
       intercept the system is solved in its exact centred form: the
       intercept is eliminated, leaving
       $(\tilde A_i+\gamma I+c_i\bar x_i\bar x_i^\top)\,\hat B_{\cdot,i}=
       \tilde\xi_i+c_i\bar x_i\bar y_i^\top$ with $c_i=n_i\gamma/(n_i+\gamma)$
       ($c_i=0$ without a ridge), on the moments about the means.
    2. **Rank-deficient neurons.** When that system is numerically singular
       (the smallest eigenvalue of its unit-diagonal rescaling is at most
       `RANK_TOLERANCE`, a scale-invariant test that, on the centred moments,
       is also shift-invariant), the neuron gets the minimum-norm
       least-squares solution for its regressor coefficients, computed in the
       retained range, it is listed in `rank_deficient_neurons`, and one
       [`DesignWarning`][mtdr.errors.DesignWarning] names such neurons. A
       `ridge` that is not negligible against the Gram removes the case.
    3. **Truncation, (M17)-(M18).** $\hat B_p=U_p\Sigma_pV_p^\top$;
       `W[p]` $=U_p[:, :r_p]\Sigma_p^{1/2}$, `S[p]` $=V_p[:, :r_p]\Sigma_p^{1/2}$,
       `B[p] = W[p] @ S[p].T`. With `precision_weighted=True`, (M18w), the
       rows are weighted first: with
       $d_i\propto\sqrt{\hat\lambda^{\rm OLS}_in_i}$ (scaled to unit root mean
       square), $\hat\lambda^{\rm OLS}_i=n_iT/\mathrm{RSS}^{\rm full}_i$ from the
       residual of the unconstrained fit of step 1 and $D=\mathrm{diag}(d)$,
       $D\hat B_p=U_p\Sigma_pV_p^\top$, `W[p]` $=D^{-1}U_p[:, :r_p]\Sigma_p^{1/2}$
       and `S[p]` as above: `B[p]` is the best rank-$r_p$ approximation of
       $\hat B_p$ in the norm $\sum_id_i^2\lVert B[i,:]\rVert^2$.
    4. **Intercept.** Refitted given the truncated blocks,
       $\hat b_i=\frac{n_i}{n_i+\gamma}\big(\bar y_i-\sum_p\bar x_{ip}B_p[i,:]\big)$,
       the maximiser over $b$ of the plug-in likelihood at `B`, so the score
       does not depend on where a regressor's zero is.
    5. **Noise precision, (M19b).** $\mathrm{RSS}_i=\sum_{k\in\mathcal K_i}
       \lVert Y_{ki\cdot}-\hat b_i-\sum_px_{kp}B_p[i,:]\rVert^2$ from the
       sufficient statistics, and $\hat\lambda_i=n_iT/\mathrm{RSS}_i$.
    6. **Score, (M21a)-(M21b).** The fully normalised plug-in
       log-likelihood, the textbook count
       [`n_parameters_svd`][mtdr.aic.n_parameters_svd] and the AIC.

    Parameters
    ----------
    stats : SufficientStats
        From [`sufficient_statistics`][mtdr.stats.sufficient_statistics].
    ranks : sequence of int or 1-D integer array
        Rank of each regressor's coefficient matrix, in `X` column order,
        `0 <= r_p <= min(n_neurons, n_bins)`. Rank 0 removes the regressor's
        term from the fitted model (its `B_full` is still estimated). The
        intercept is not an entry.
    ridge : float
        $\gamma\ge0$, added to every neuron's $A^f_i$ (the reference's
        `ridgeparam`).
    condition_independent : bool
        Fit the full-rank intercept. With `False`, `Y` should already be
        centred per neuron and bin.
    precision_weighted : bool
        Truncate in the row-weighted norm of step 3 instead of the Frobenius
        norm. Steps 4-6 are unchanged: the intercept, the residual, the
        precisions, the log-likelihood and the AIC are computed from the
        weighted `B` exactly as from the unweighted one. Off by default, which
        is the reference's estimator; the weighted truncation seeds the
        marginal-likelihood rank search.

    Returns
    -------
    SVDFit
        Factors, coefficients, intercepts, noise precisions and scores.

    Raises
    ------
    ParameterError
        If `stats` is not a `SufficientStats`, `ranks` has the wrong length or
        an entry outside `[0, min(n_neurons, n_bins)]` (bools and floats are
        rejected), `ridge` is not a finite real `>= 0`, or
        `condition_independent` or `precision_weighted` is not a bool.
    ValidationError
        If a neuron was never observed, or if a neuron's residual is zero to
        rounding: $\mathrm{RSS}_i\le10^3\,\epsilon\,T\,E_i$, $E_i$ its energy
        about its mean (its raw energy without the intercept), as for an exact
        fit, a neuron constant over its trials, or one observed on no more
        trials than it has coefficients; its noise precision would be
        unbounded. A neuron with zero energy, $E_i=0$, is rejected whatever
        its truncated residual, which can be a rounding-level positive number
        on some platforms. With `precision_weighted=True`, also if a neuron's
        unconstrained residual $\mathrm{RSS}^{\rm full}_i$ is zero to rounding
        by the same test (its weight would be infinite). The message lists the
        neurons and the remedies.

    Warns
    -----
    DesignWarning
        Once, listing the neurons given the minimum-norm solution.

    Notes
    -----
    Cost: $O(n\,P^3 + n\,P^2\,T)$ for the per-neuron solves, batched, plus one
    batched SVD of the $P$ matrices $n\times T$, $O(P\,n\,T\min(n,T))$. No
    loop over neurons.

    $\mathrm{RSS}_i$ is computed from the statistics. With the intercept it
    uses the moments about the means, so its rounding error is about
    $\epsilon$ times the energy about the mean, whatever the baseline;
    without it, about $\epsilon$ times the raw energy.

    The SVD of the least-squares solution is not the rank-constrained
    maximum-likelihood estimate under unequal precisions or a mask, so
    `log_likelihood` is a plug-in score at the truncated estimate, and it is
    not on the scale of the marginal likelihood (`docs/model.md` § 3.3). The
    weighted truncation is that estimate at the precisions
    $\hat\lambda^{\rm OLS}$ when each neuron's regressors are uncorrelated over
    its trials, with variances that do not depend on the neuron, and close to
    it otherwise; in the demo's heterogeneous-noise regime its AIC search
    recovered the true ranks far more often than the unweighted one.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import simulate
    >>> from mtdr.stats import sufficient_statistics
    >>> from mtdr.svd_fit import fit_svd
    >>> sim = simulate(n_neurons=30, n_bins=10, n_trials=200, ranks=[2, 1],
    ...                drop_prob=0.2, seed=1)
    >>> fit = fit_svd(sufficient_statistics(sim.Y, sim.X, sim.mask), [2, 1])
    >>> fit
    SVDFit(ranks=(2, 1), n_neurons=30, n_bins=10, intercept=True, ...)
    >>> [s.shape for s in fit.S], fit.W[0].shape
    ([(10, 2), (10, 1)], (30, 2))
    >>> err = np.linalg.norm(fit.B[0] - sim.B["x0"]) / np.linalg.norm(sim.B["x0"])
    >>> bool(err < 0.1)
    True
    """
    if not isinstance(stats, SufficientStats):
        raise ParameterError(
            "stats must be a SufficientStats (from mtdr.stats.sufficient_statistics); "
            f"got {type(stats).__name__}"
        )
    ci = as_bool("condition_independent", condition_independent)
    weighted = as_bool("precision_weighted", precision_weighted)
    gamma = require_non_negative_real("ridge", ridge)
    n, P, T = stats.n_neurons, stats.n_regressors, stats.n_bins
    rank_list = _ranks(ranks, P, min(n, T))

    never = np.flatnonzero(stats.n_obs == 0)
    if never.size:
        raise ValidationError(
            f"neurons {_listed(never)} are never observed; drop them before fitting"
        )

    G, rhs = _normal_equations(stats, ci, gamma)
    beta, deficient = _least_squares(G, rhs)
    if deficient.size:
        warnings.warn(
            f"neurons {_listed(deficient)} have a rank-deficient observed design "
            "(fewer observed trials than coefficients, collinear regressors, or a "
            "regressor constant over their trials); they get the minimum-norm "
            "least-squares solution. A ridge that is not negligible against "
            "their X_i'X_i removes this.",
            DesignWarning,
            stacklevel=2,
        )

    B_full = [np.ascontiguousarray(beta[:, p, :]) for p in range(P)]
    energy = stats.YtY_c if ci else stats.YtY_raw
    blocks = beta.transpose(1, 0, 2)  # (P, n, T)
    if weighted:
        # (M18w): truncate D B_full,p and map back with D^-1.
        d = _precision_weights(stats, ci, gamma, beta, energy)
        W, S, _ = _truncate(blocks * d[None, :, None], rank_list)
        W = [np.ascontiguousarray(w / d[:, None]) for w in W]
        B = [w @ s.T for w, s in zip(W, S, strict=True)]
    else:
        W, S, B = _truncate(blocks, rank_list)
    B_arr = np.stack(B, axis=1)  # (n, P, T)
    count = stats.n_obs.astype(np.float64)
    intercept: FloatArray | None = None
    intercept_full: FloatArray | None = None
    if ci:
        shrink = (count / (count + gamma))[:, None]
        intercept_full = shrink * (stats.Y_mean - _at_mean(stats.X_mean, beta))
        intercept = shrink * (stats.Y_mean - _at_mean(stats.X_mean, B_arr))
    rss = _residual(stats, ci, gamma, B_arr)
    _check_degenerate(rss, energy, T)
    n_entries = count * T
    precision = n_entries / rss
    log_likelihood = float(
        -0.5 * np.sum(n_entries * (np.log(rss / n_entries) + 1.0 + _LOG_2PI))
    )
    n_parameters = _aic.n_parameters_svd(
        rank_list, n, T, condition_independent=ci, formula="textbook"
    )

    arrays = [*W, *S, *B, *B_full, precision, deficient]
    for extra in (intercept, intercept_full):
        if extra is not None:
            arrays.append(extra)
    for a in arrays:
        a.setflags(write=False)
    return SVDFit(
        W=tuple(W),
        S=tuple(S),
        B=tuple(B),
        B_full=tuple(B_full),
        intercept=intercept,
        intercept_full=intercept_full,
        noise_precision=precision,
        log_likelihood=log_likelihood,
        n_parameters=n_parameters,
        aic=_aic.aic(log_likelihood, n_parameters),
        rank_deficient_neurons=deficient,
        ranks=tuple(rank_list),
        precision_weighted=weighted,
    )


# --------------------------------------------------------------------------- steps


def _ranks(ranks: object, n_regressors: int, max_rank: int) -> list[int]:
    rank_list = require_int_vector("ranks", ranks, minimum=0)
    if len(rank_list) != n_regressors:
        raise ParameterError(
            f"ranks has {len(rank_list)} entries; expected one per regressor "
            f"({n_regressors}), without the intercept"
        )
    for p, r in enumerate(rank_list):
        if r > max_rank:
            raise ParameterError(
                f"ranks[{p}] is {r}, above min(n_neurons, n_bins) = {max_rank}"
            )
    return rank_list


def _normal_equations(
    stats: SufficientStats, condition_independent: bool, ridge: float
) -> tuple[FloatArray, FloatArray]:
    r"""Return the per-neuron system for the regressor coefficients, (M15)-(M16).

    Without the intercept, $(A_i+\gamma I)\beta_i=\xi_i$ on the raw moments.
    With it, the joint system on $A^f_i=\begin{pmatrix}A_i & n_i\bar x_i\\
    n_i\bar x_i^\top & n_i\end{pmatrix}$ (the per-neuron blocks of the
    reference's `MkSuffStats_BilinReg_Sims`) with the intercept eliminated, in
    the moments about the means:
    $G_i=\tilde A_i+\gamma I+c_i\bar x_i\bar x_i^\top$ and
    $\tilde\xi_i+c_i\bar x_i\bar y_i^\top$, $c_i=n_i\gamma/(n_i+\gamma)$, the
    exact Schur complement. Returns `(G, rhs)` of shapes `(n, P, P)` and
    `(n, P, T)`.
    """
    P = stats.n_regressors
    eye = ridge * np.eye(P)
    if not condition_independent:
        return stats.XtX + eye, np.array(stats.XtY_raw)
    count = stats.n_obs.astype(np.float64)
    c = (count * ridge / (count + ridge))[:, None, None]
    x = stats.X_mean
    G = stats.XtX_c + eye + c * (x[:, :, None] * x[:, None, :])
    rhs = stats.XtY_c + c * (x[:, :, None] * stats.Y_mean[:, None, :])
    return G, rhs


def _least_squares(G: FloatArray, rhs: FloatArray) -> tuple[FloatArray, IntArray]:
    r"""Per-neuron solutions of $G_i\beta_i=\mathrm{rhs}_i$, minimum norm if singular.

    The Gram is rescaled to unit diagonal, $\tilde G_i=D_iG_iD_i$ with
    $D_i=\mathrm{diag}(G_i)^{-1/2}$ (1 for an all-zero column), and
    eigendecomposed, $\tilde G_i=V_i\Lambda_iV_i^\top$. Eigenvalues at or below
    `RANK_TOLERANCE` are null directions (the test does not depend on column
    scale). A full-rank neuron gets $D_iV_i\Lambda_i^{-1}V_i^\top
    D_i\,\mathrm{rhs}_i$. A rank-deficient neuron gets the minimum-norm
    solution in its retained range, without subtracting a large null-space
    component: with $F_i=D_i^{-1}V_{R}\Lambda_{R}^{1/2}$ over the $k_i$
    retained eigenpairs, $F_i=U_i\,\mathrm{diag}(s_i)\,Q_i^\top$ its SVD,
    $\beta_i=U_i\,\mathrm{diag}(s_i^{-2})\,U_i^\top\mathrm{rhs}_i$, using the
    top $k_i$ singular vectors (no second tolerance). That is
    `numpy.linalg.lstsq` on the neuron's (centred) design. Returns
    `(beta (n, P, T), deficient neuron indices)`.
    """
    Q = G.shape[1]
    diag = np.diagonal(G, axis1=1, axis2=2)
    positive = diag > 0
    d = np.where(positive, 1.0 / np.sqrt(np.where(positive, diag, 1.0)), 1.0)
    w, V = np.linalg.eigh(G * d[:, :, None] * d[:, None, :])
    null = w <= RANK_TOLERANCE
    w_inv = np.divide(1.0, w, out=np.zeros_like(w), where=~null)
    y = V @ (w_inv[:, :, None] * (V.transpose(0, 2, 1) @ (d[:, :, None] * rhs)))
    beta: FloatArray = d[:, :, None] * y
    deficient = np.flatnonzero(null.any(axis=1)).astype(np.int64)
    if deficient.size:
        kept = ~null[deficient]
        root = np.sqrt(np.where(kept, w[deficient], 0.0))
        F = V[deficient] * root[:, None, :] / d[deficient, :, None]
        U, s = np.linalg.svd(F)[:2]
        k = kept.sum(axis=1)
        top = np.arange(Q)[None, :] < k[:, None]
        s_inv2 = np.divide(1.0, s * s, out=np.zeros_like(s), where=top)
        Ut_rhs = U.transpose(0, 2, 1) @ rhs[deficient]
        beta[deficient] = U @ (s_inv2[:, :, None] * Ut_rhs)
    return beta, deficient


def _truncate(
    B_full: FloatArray, ranks: Sequence[int]
) -> tuple[list[FloatArray], list[FloatArray], list[FloatArray]]:
    """Rank-`r_p` truncated SVD factors of each `(n, T)` block, (M17)-(M18).

    Signs: the entry of largest absolute value of each left singular
    vector is made positive (ties to the first such entry), and the right
    singular vector flipped with it.
    """
    U, s, Vt = np.linalg.svd(B_full, full_matrices=False)
    lead = np.take_along_axis(U, np.argmax(np.abs(U), axis=1)[:, None, :], axis=1)[
        :, 0, :
    ]
    sign = np.where(lead < 0, -1.0, 1.0)
    U = U * sign[:, None, :]
    Vt = Vt * sign[:, :, None]
    root = np.sqrt(s)
    W: list[FloatArray] = []
    S: list[FloatArray] = []
    B: list[FloatArray] = []
    for p, r in enumerate(ranks):
        Wp = np.ascontiguousarray(U[p, :, :r] * root[p, :r])
        Sp = np.ascontiguousarray(Vt[p, :r, :].T * root[p, :r])
        W.append(Wp)
        S.append(Sp)
        B.append(Wp @ Sp.T)
    return W, S, B


def _residual(
    stats: SufficientStats, condition_independent: bool, ridge: float, coef: FloatArray
) -> FloatArray:
    r"""$\mathrm{RSS}_i$ of the fit with regressor coefficients `coef`, (M19b).

    With the intercept refitted given `coef`: the residual about the
    means, plus the part of the level the ridge keeps out of the intercept,
    $y-b-Bx=(\tilde y-B\tilde x)+(1-\frac{n_i}{n_i+\gamma})\,\mathrm{level}_i$.
    Without the intercept, from the raw moments.
    """
    if not condition_independent:
        return _rss(stats.XtX, stats.XtY_raw, stats.YtY_raw, coef)
    count = stats.n_obs.astype(np.float64)
    level = stats.Y_mean - _at_mean(stats.X_mean, coef)
    kept = (ridge / (count + ridge))[:, None] * level
    rss = _rss(stats.XtX_c, stats.XtY_c, stats.YtY_c, coef)
    out: FloatArray = rss + count * np.einsum("it,it->i", kept, kept)
    return out


def _precision_weights(
    stats: SufficientStats,
    condition_independent: bool,
    ridge: float,
    beta: FloatArray,
    energy: FloatArray,
) -> FloatArray:
    r"""Return the row weights $d_i\propto\sqrt{\hat\lambda^{\rm OLS}_in_i}$ of (M18w).

    $\hat\lambda^{\rm OLS}_i=n_iT/\mathrm{RSS}^{\rm full}_i$ from the residual of
    the unconstrained fit `beta` (with the ridge, when any) and its refitted
    intercept. The weights are scaled to unit root mean square, which does not
    change the truncated `B` and makes equal weights the identity, so that the
    fit then equals the unweighted one, factors included. A neuron whose
    unconstrained residual is zero to rounding (`_zero_to_rounding`) has an
    infinite weight and raises.
    """
    T = stats.n_bins
    rss = _residual(stats, condition_independent, ridge, beta)
    exact = np.flatnonzero(_zero_to_rounding(rss, energy, T))
    if exact.size:
        raise ValidationError(
            f"neurons {_listed(exact)} are fitted with zero residual by the "
            "unconstrained least-squares fit, so their precision weights are "
            "infinite: drop them, require more observations per neuron, "
            "set ridge > 0, or use precision_weighted=False"
        )
    count = stats.n_obs.astype(np.float64)
    d = np.sqrt(count * T / rss * count)
    out: FloatArray = d / np.sqrt(np.mean(d * d))
    return out


def _at_mean(x_mean: FloatArray, beta: FloatArray) -> FloatArray:
    r"""$\sum_p\bar x_{ip}\beta_i[p,:]$, the regressor part of the fit at the mean."""
    out: FloatArray = np.einsum("ip,ipt->it", x_mean, beta)
    return out


def _rss(
    A: FloatArray, xi: FloatArray, upsilon: FloatArray, beta: FloatArray
) -> FloatArray:
    r"""Aligned residual sums of squares from the statistics, (M19b).

    $\mathrm{RSS}_i=\upsilon_i-2\sum_{p,t}\beta_i[p,t]\,\xi_i[p,t]
    +\sum_{p,q,t}\beta_i[p,t]\,A_i[p,q]\,\beta_i[q,t]$ with the un-ridged
    moments: the squared residual of every observed entry against its own
    prediction (the reference's `SVDRegB_AIC` pairs mismatched entries,
    trial-fastest data against time-fastest predictions). With the moments
    about the means this is the residual of the fit whose intercept is
    refitted given `beta`.
    """
    cross = np.einsum("ipt,ipt->i", beta, xi)
    quad = np.einsum("ipt,ipq,iqt->i", beta, A, beta)
    rss: FloatArray = upsilon - 2.0 * cross + quad
    return rss


def _zero_to_rounding(
    value: FloatArray, energy: FloatArray, n_bins: int
) -> NDArray[np.bool_]:
    r"""Where a residual is zero to rounding, $v_i\le10^3\,\epsilon\,T\,E_i$.

    $E_i$ is the energy the residual is computed from. `True` also where `value`
    is `NaN`. Shared with `mtdr.mmle`, which applies it to the unconstrained
    least-squares residual and to the expected residual of (M30).
    """
    floor = _ROUNDING_MARGIN * _EPS * n_bins * energy
    out: NDArray[np.bool_] = ~(value > floor)
    return out


def _unconstrained_residual(
    stats: SufficientStats, condition_independent: bool
) -> tuple[FloatArray, FloatArray]:
    r"""Return $\mathrm{RSS}^{\rm full}_i$ of the unconstrained fit, and its energy.

    The minimum-norm least-squares fit of each neuron's observed responses on
    $[X_i\ \mathbf 1]$ (on $X_i$ without the intercept), without a ridge, from
    the statistics; $E_i$ is $\tilde\upsilon_i$ with the intercept and
    $\upsilon_i$ without, as in `_zero_to_rounding`. `mtdr.mmle` uses it to
    find the neurons without residual degrees of freedom.
    """
    G, rhs = _normal_equations(stats, condition_independent, 0.0)
    beta, _ = _least_squares(G, rhs)
    if condition_independent:
        return _rss(stats.XtX_c, stats.XtY_c, stats.YtY_c, beta), stats.YtY_c
    return _rss(stats.XtX, stats.XtY_raw, stats.YtY_raw, beta), stats.YtY_raw


def _check_degenerate(rss: FloatArray, energy: FloatArray, n_bins: int) -> None:
    r"""Raise if a neuron's residual is zero to rounding.

    Degenerate iff $\mathrm{RSS}_i\le10^3\,\epsilon\,T\,E_i$, with $E_i$ the
    energy the residual is computed from (about the mean with the intercept,
    raw without): the rounding error of $\mathrm{RSS}_i$ is a few $\epsilon E_i$,
    so the margin is three orders of magnitude above it, while a residual
    above $10^3\epsilon T\approx3\times10^{-12}$ of the energy at $T=15$ (an
    amplitude signal-to-noise ratio below about $5\times10^5$) still passes.
    A neuron with $E_i=0$ (silent, or constant over its trials) is degenerate
    whatever $\mathrm{RSS}_i$.
    """
    # A neuron with no energy is degenerate whatever its truncated fit's
    # residual: the truncation can leave a rounding-level row for it on some
    # LAPACK builds, so that RSS > 0 = the floor.
    degenerate = np.flatnonzero(
        _zero_to_rounding(rss, energy, n_bins) | ~(energy > 0.0)
    )
    if degenerate.size:
        raise ValidationError(
            f"neurons {_listed(degenerate)} are fitted with zero residual, so their "
            "noise precision is unbounded: drop them, require more "
            "observations per neuron, or set ridge > 0"
        )


def _listed(indices: NDArray[np.integer[Any]]) -> str:
    items = [int(i) for i in indices]
    head = ", ".join(str(i) for i in items[:_MAX_LISTED])
    more = f" and {len(items) - _MAX_LISTED} more" if len(items) > _MAX_LISTED else ""
    return f"[{head}{more}]"
