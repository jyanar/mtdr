r"""Parameter counts and the Akaike information criterion.

The two estimators count parameters differently because they estimate
different things (`docs/model.md` § 3.3, § 7.1):

- the SVD / reduced-rank estimator fits low-rank coefficient matrices
  $B_p=W_pS_p^\top$, a precision per neuron and a full-rank intercept, and its
  likelihood is a plug-in profile likelihood at the truncated least-squares
  estimate, (M21);
- the marginal-likelihood estimator integrates the weights $W$ out, so it
  counts the temporal bases $S_p$ (modulo the orthogonal rotation the marginal
  likelihood is invariant to), the precisions and the intercept, (M38a).

The two likelihoods are different objects, so the AIC values of the two
estimators are never comparable. `formula="reference"` gives the counts the
MATLAB demo uses, for parity fixtures only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mtdr._args import as_bool, as_int, as_real, positive_int, require_int_vector
from mtdr.errors import ParameterError

__all__ = ["aic", "n_parameters_mmle", "n_parameters_svd"]


def n_parameters_svd(
    ranks: Sequence[int] | NDArray[np.integer[Any]],
    n_neurons: int,
    n_bins: int,
    *,
    condition_independent: bool = True,
    formula: Literal["textbook", "reference"] = "textbook",
) -> int:
    r"""Count the parameters of the SVD / reduced-rank fit, (M21b) or (M20b).

    `"textbook"` (the default, (M21b)) counts the dimension of the
    set of rank-$r_p$ matrices $n\times T$ for every regressor, one noise
    precision per neuron, and, with the intercept, a full-rank $n\times T$
    term (rank $m=\min(n,T)$, which counts $m(n+T-m)=nT$):

    $$
    K_{svd}(r)=\sum_p r_p\,(n+T-r_p)+n\;\big[+\,nT\big].
    $$

    `"reference"` is the count in the MATLAB `SVDRegB_AIC`, (M20b),
    $K=(nP'+TP'-\sum r')\sum r'$, where, as in the demo, $P'$ and $r'$
    include the intercept as a last regressor of rank $\min(n,T)$ when
    `condition_independent=True`. It is not a parameter count (it
    over-penalises by roughly a factor $P$); it exists for parity fixtures.
    Only the count changes; the reference's other three defects live in the
    test helpers.

    Parameters
    ----------
    ranks : sequence of int
        Rank $r_p$ of each regressor, `0 <= r_p <= min(n_neurons, n_bins)`,
        in `X` column order; the intercept is not an entry.
    n_neurons : int
        $n$, positive.
    n_bins : int
        $T$, positive.
    condition_independent : bool
        Count the full-rank intercept.
    formula : {"textbook", "reference"}
        Which count.

    Returns
    -------
    int
        The count.

    Raises
    ------
    ParameterError
        For a rank outside `[0, min(n_neurons, n_bins)]`, non-positive sizes,
        an empty `ranks`, or an unknown `formula`.

    Examples
    --------
    >>> from mtdr.aic import n_parameters_svd
    >>> n_parameters_svd([2, 1], n_neurons=10, n_bins=5)   # 2*13 + 1*14 + 10 + 50
    100
    >>> n_parameters_svd([2, 1], 10, 5, formula="reference")   # (30 + 15 - 8) * 8
    296
    """
    r, n, T = _counts_args(ranks, n_neurons, n_bins)
    ci = as_bool("condition_independent", condition_independent)
    m = min(n, T)
    if formula == "textbook":
        k = sum(rp * (n + T - rp) for rp in r) + n
        return k + (m * (n + T - m) if ci else 0)
    if formula == "reference":
        r_full = [*r, m] if ci else r
        P = len(r_full)
        total = sum(r_full)
        return (n * P + T * P - total) * total
    raise ParameterError(f"formula must be 'textbook' or 'reference'; got {formula!r}")


def n_parameters_mmle(
    ranks: Sequence[int] | NDArray[np.integer[Any]],
    n_neurons: int,
    n_bins: int,
    *,
    condition_independent: bool = True,
    formula: Literal["identifiable", "reference"] = "identifiable",
) -> int:
    r"""Count the parameters of the marginal-likelihood fit, (M38a) or (M38).

    `"identifiable"` (the default, (M38a)): one precision per neuron, the
    bases $S_p$ modulo the orthogonal rotation $S_p\to S_pQ_p$ that leaves the
    marginal likelihood unchanged (`docs/model.md` § 10.1), and the
    $n\times T$ intercept when present:

    $$
    K^{\rm id}_{mml}(r)=n+\sum_p\Big(T\,r_p-\tfrac12r_p(r_p-1)\Big)\;\big[+\,nT\big].
    $$

    `"reference"` is the demo's count, the length of its parameter vector
    $(\lambda;s;\mathrm{vec}\,b)$, (M38): $n+T\sum_pr_p\,[+\,nT]$. The
    integrated-out weights are not counted by either. Without the intercept
    the reference count drops the $nT$ entries of $b$ as well.

    Parameters
    ----------
    ranks : sequence of int
        Rank $r_p$ of each regressor, `0 <= r_p <= min(n_neurons, n_bins)`.
    n_neurons : int
        $n$, positive.
    n_bins : int
        $T$, positive.
    condition_independent : bool
        Count the intercept.
    formula : {"identifiable", "reference"}
        Which count.

    Returns
    -------
    int
        The count.

    Raises
    ------
    ParameterError
        As [`n_parameters_svd`][mtdr.aic.n_parameters_svd].

    Examples
    --------
    >>> from mtdr.aic import n_parameters_mmle
    >>> n_parameters_mmle([3, 4, 2], n_neurons=100, n_bins=15)   # 100 + 125 + 1500
    1725
    >>> n_parameters_mmle([3, 4, 2], 100, 15, formula="reference")   # 100 + 135 + 1500
    1735
    """
    r, n, T = _counts_args(ranks, n_neurons, n_bins)
    ci = as_bool("condition_independent", condition_independent)
    intercept = n * T if ci else 0
    if formula == "identifiable":
        return n + sum(T * rp - rp * (rp - 1) // 2 for rp in r) + intercept
    if formula == "reference":
        return n + T * sum(r) + intercept
    raise ParameterError(
        f"formula must be 'identifiable' or 'reference'; got {formula!r}"
    )


def aic(log_likelihood: float, n_parameters: int) -> float:
    r"""Akaike information criterion $2K-2\ell$ (lower is better).

    (M21b), (M38). `log_likelihood` should be a fully normalised
    log-likelihood; the result is comparable only between fits of the same
    estimator to the same `Y` and `mask`.

    Parameters
    ----------
    log_likelihood : float
        $\ell$, finite.
    n_parameters : int
        $K\ge0$.

    Returns
    -------
    float
        $2K-2\ell$.

    Raises
    ------
    ParameterError
        If `log_likelihood` is not a finite real number or `n_parameters` is
        not a non-negative integer (bools are rejected).

    Examples
    --------
    >>> from mtdr.aic import aic
    >>> aic(-120.5, 10)
    261.0
    """
    ll = as_real(log_likelihood)
    if ll is None or not np.isfinite(ll):
        raise ParameterError(
            f"log_likelihood must be a finite real number; got {log_likelihood!r}"
        )
    k = as_int(n_parameters)
    if k is None or k < 0:
        raise ParameterError(
            f"n_parameters must be a non-negative integer; got {n_parameters!r}"
        )
    return 2.0 * k - 2.0 * ll


def _counts_args(
    ranks: object, n_neurons: object, n_bins: object
) -> tuple[list[int], int, int]:
    n = positive_int("n_neurons", n_neurons)
    T = positive_int("n_bins", n_bins)
    r = require_int_vector("ranks", ranks, minimum=0)
    m = min(n, T)
    for p, rp in enumerate(r):
        if rp > m:
            raise ParameterError(
                f"ranks[{p}] is {rp}, above min(n_neurons, n_bins) = {m}"
            )
    return r, n, T
