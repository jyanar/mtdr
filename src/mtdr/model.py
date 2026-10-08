r"""The `MTDR` estimator.

[`MTDR`][mtdr.model.MTDR] composes the package's functional building blocks:
[`validate_inputs`][mtdr.data.validate_inputs], the sufficient statistics of
[`mtdr.stats`][mtdr.stats], the SVD estimator of [`mtdr.svd_fit`][mtdr.svd_fit],
the marginal-likelihood estimator of [`mtdr.mmle`][mtdr.mmle] and the greedy
AIC search of [`mtdr.rank_search`][mtdr.rank_search]. It fits

$$
Y[k] = \text{intercept} + \sum_p X[k,p]\,W_pS_p^\top + E_k,\qquad
E_k[i,:]\sim\mathcal N\big(0,\lambda_i^{-1}I\big),
$$

(M1)-(M4), under the observation mask (M3), and derives from the fit the
projections (M49)-(M49a), the plug-in decoder (M46)-(M48c) and the held-out
plug-in likelihood. Per-regressor results are dicts keyed by regressor name, in
`X` column order.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import io
import itertools
import math
import multiprocessing
import os
import sys
import threading
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import (
    FIRST_EXCEPTION,
    Executor,
    Future,
    ProcessPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, Any, Literal, TextIO, overload

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

from mtdr import aic as _aic
from mtdr._args import (
    as_int,
    as_list,
    as_real,
    require_names,
)
from mtdr.data import (
    _as_array,
    _deficient_message,
    _observation_floor,
    _soft_warnings,
    _uncentred_message,
    _validated,
    validate_inputs,
)
from mtdr.errors import (
    ConvergenceWarning,
    DecodingWarning,
    DesignWarning,
    NotFittedError,
    ParameterError,
    ProjectionWarning,
    SingularDesignError,
    ValidationError,
)
from mtdr.mmle import BASIS_SPAN_SCALE, MMLEFit, fit_mmle
from mtdr.rank_search import RankSearchHistory, greedy_aic
from mtdr.stats import SufficientStats, sufficient_statistics
from mtdr.svd_fit import SVDFit, fit_svd

if TYPE_CHECKING:
    import xarray

__all__ = ["MTDR", "canonical_factors"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

_LOG_2PI = float(np.log(2.0 * np.pi))
_EPS = float(np.finfo(np.float64).eps)
_SQRT_EPS = float(np.sqrt(_EPS))
#: Most discrete level combinations `decode` enumerates.
_MAX_COMBINATIONS = 64
#: Most residual entries `decode` holds at once (trials x neurons x bins).
_DECODE_BLOCK = 2**22
_MAX_LISTED = 20
#: Characters of a failed candidate's first reason in a search's warning.
_REASON_CHARS = 200

#: Fitted attributes; reading one before `fit` raises NotFittedError.
_FITTED = frozenset(
    {
        "regressor_names_",
        "n_trials_",
        "n_neurons_",
        "n_bins_",
        "n_regressors_",
        "n_observations_",
        "ranks_",
        "total_rank_",
        "W_",
        "S_",
        "B_",
        "intercept_",
        "noise_precision_",
        "weight_posterior_cov_",
        "log_likelihood_",
        "objective_",
        "n_parameters_",
        "aic_",
        "rank_search_history_",
        "estimator_",
        "converged_",
        "n_iter_",
        "condition_independent_",
    }
)


class MTDR:
    r"""Model-based targeted dimensionality reduction.

    Fits, for every regressor $p$, a low-rank coefficient matrix
    $B_p=W_pS_p^\top$ (`(n_neurons, n_bins)`, rank $r_p$), a per-neuron noise
    precision $\lambda_i$ and, by default, a full-rank condition-independent
    intercept, (M1)-(M4). `estimator="mmle"` (the default, the estimator of the
    paper) integrates the weights out under $w_i\sim\mathcal N(0,I)$ and
    maximises the marginal likelihood (M24) by ECME and coordinate ascent
    (`docs/model.md` § 5), then sets `W_` to the posterior mean (M36);
    `estimator="svd"` is reduced-rank regression, (M15)-(M18). `ranks="aic"`
    chooses the ranks by the greedy AIC search (M39).

    The constructor only stores and checks hyper-parameters (sklearn style);
    [`fit`][mtdr.model.MTDR.fit] learns, and fitted attributes end in `_`.
    Fitting is deterministic: there is no random state.

    Parameters
    ----------
    ranks : "aic", sequence of int or mapping of str to int
        `"aic"` chooses every rank by AIC (positive ranks only); a sequence
        (one per column of `X`) or a mapping naming every regressor fixes them;
        `0` removes a regressor's term.
    estimator : {"mmle", "svd"}
        The marginal-likelihood estimator (default) or reduced-rank regression.
    condition_independent : bool
        Fit the full-rank intercept. `False` only for `Y` already
        mean-subtracted per neuron and bin.
    max_rank : int, optional
        Largest rank the search may assign (and a bound on fixed ranks);
        `None` is `min(n_neurons, n_bins)`.
    min_observations : int
        A neuron observed on fewer trials is a `ValidationError` in `fit`;
        under `"mmle"` the floor applied is at least `n_regressors + 2`
        (`n_regressors + 1` without the intercept), below which a neuron has no
        residual degree of freedom and its marginal likelihood is unbounded.
    ridge : float
        L2 penalty on every neuron's $X_i^\top X_i$ in the least-squares stage.
    basis_ridge : float
        L2 penalty $\tfrac g2\lVert S\rVert^2$ on the bases (`"mmle"` only).
    rank_search_init : {"svd_weighted", "svd", "ones"} or sequence of int
        Where the search starts: `"svd_weighted"` runs the AIC search of the
        precision-weighted SVD fit (M18w) from ones and starts the MMLE search
        at its ranks; `"svd"` the same with the unweighted fit, as the
        reference does; `"ones"` a single MMLE search from ones; a sequence of
        positive ints starts there. With `estimator="svd"` the strings
        coincide (one unweighted SVD search from ones).
    rank_search_threshold : float
        $\delta\ge0$: a step is accepted only if it lowers the AIC by more.
    rank_search_warm_start : bool
        Start each candidate of the MMLE search from the accepted fit instead of
        from its own SVD fit: the accepted bases, noise precisions and
        intercept, plus, for the raised regressor, the last component of the
        unweighted SVD fit at the candidate's ranks. The search's first fit
        stays cold. About 6x fewer basis-step evaluations per candidate at the
        paper's scale and no change in any selected rank vector measured, so the
        default; `False` is the reference's procedure, an SVD start per
        candidate (M29). No effect with `estimator="svd"` or fixed ranks.
    ecme_max_iter : int
        ECME's iteration cap (`docs/model.md` § C.6).
    ecme_tol : float
        ECME's threshold on the largest squared relative change of a parameter
        in an iteration, (M34) with `convergence_eps` added to each denominator
        (`docs/model.md` § C.6).
    refine_max_iter : int
        The coordinate ascent's iteration cap (`docs/model.md` § C.7).
    refine_tol : float
        The coordinate ascent's threshold on the same change
        (`docs/model.md` § C.7).
    convergence_eps : float
        The term added to each $\theta_j^2$ in that change, in both loops, so
        that a parameter that is exactly zero does not make it infinite.
    optimizer_max_iter : int
        `maxiter` of the inner L-BFGS-B runs (default 2000: the first basis
        step at the paper's scale needs 582-909 iterations).
    optimizer_tol : float
        Their projected-gradient test `gtol`, minFunc's `optTol` (the
        reference's is $10^{-5}$). With `basis_preconditioning=True` the basis
        step tests it on the rescaled gradient.
    optimizer_progtol : float
        The absolute function change at which the inner basis step stops,
        minFunc's `progTol` (default $10^{-9}$).
    canonicalize : bool
        Rotate each $(W_p,S_p)$ into the PC orientation after fitting
        (`docs/model.md` § 10.1): `W_[p]` orthonormal, ordered by singular
        value, largest entry of each column positive; `S_[p]` carries the
        scale. Likelihoods and AIC are computed in the raw frame before the
        rotation.
    verbose : int
        `0` silent; `1` one line per accepted rank-search step; `2` also per
        ECME and refinement iteration.
    n_jobs : int or None
        Worker processes for the candidate fits of each round of the MMLE
        rank search: `None` or `1` fits them in turn, `-1` uses every CPU
        available to the process. A round has one candidate per regressor
        below `max_rank`, so no more workers than that are started, and none
        for one. The workers are started by `spawn` when the MMLE search
        begins, so they start up during its first fit, which they cannot
        share; each runs its linear algebra on one thread (the
        `OPENBLAS_NUM_THREADS`-like variables are 1 while they start), without
        which the workers compete and gain nothing. The search, the fit and
        the warnings are those of `n_jobs=1` up to the rounding of the BLAS's
        thread count (identical bit for bit on the builds measured; a
        near-tie between candidates could in principle resolve differently).
        The workers import `mtdr` afresh: the caller's `numpy.seterr` is
        passed on, changes made to `mtdr`'s modules at run time are not. At
        the paper's scale on 4 CPUs a round runs about 2.3-2.9x faster and a
        search about 1.3-1.5x. A script that uses it needs the
        `if __name__ == "__main__":` guard that `multiprocessing` requires; if
        the workers cannot start (a daemonic process) or stop (no guard, a
        dead worker), the candidates are fitted in turn, with a
        `RuntimeWarning`. With `verbose=2`, a round's iteration lines are
        printed together when it ends, in regressor order. No effect with
        `estimator="svd"`, on the SVD stage of `rank_search_init`, or with
        fixed `ranks`.
    basis_preconditioning : bool
        Speed up the MMLE refinement's basis step in every *cold* fit, one that
        starts from its own SVD fit: the fixed-rank fit, the MMLE search's first
        fit and, with `rank_search_warm_start=False`, every candidate. Warm-
        started candidates are never preconditioned. This has no MATLAB
        counterpart: `False` is the reference's basis step, `minFunc` on the
        raw bases (`Estpars_CoordAscent_lambi_S_b.m`). With `True`, L-BFGS-B
        works in rescaled variables in which each regressor's directions inside
        the span of its starting bases are stretched by a fixed factor of 30
        (`mtdr.mmle.BASIS_SPAN_SCALE`; use `fit_mmle(basis_span_scale=)` for
        another value). The likelihood is nearly flat along those directions,
        where an unscaled run spends about half its evaluations. The objective
        is unchanged, so the fit ends at the same optimum to within the
        optimiser's tolerance; at the paper's scale a cold fit takes about 7x
        fewer basis-step evaluations and a whole search runs 1.3-1.8x faster
        with the same ranks. Because the path differs, results differ from
        `False`'s at the optimiser's tolerance, and `optimizer_tol` is then
        tested on the rescaled gradient. Off by default. No effect with
        `estimator="svd"`.

    Attributes
    ----------
    regressor_names_ : tuple of str
        Names in `X` column order; the keys of every per-regressor dict.
    n_trials_ : int
        Trials seen in `fit`.
    n_neurons_ : int
        Neurons seen in `fit`.
    n_bins_ : int
        Time bins seen in `fit`.
    n_regressors_ : int
        Columns of `X`.
    n_observations_ : numpy.ndarray
        `(n_neurons,)` int, the trials on which each neuron was observed (the
        mask's column sums).
    ranks_ : dict of str to int
        The rank $r_p$ of every regressor, chosen or fixed.
    total_rank_ : int
        $\sum_pr_p$.
    W_ : dict of str to numpy.ndarray
        `(n_neurons, r_p)` neuron weights. With `canonicalize=True` the
        orthonormal encoding-subspace basis $U_p$; otherwise the
        posterior mean (M36) in the estimator's frame (`"mmle"`) or the left
        factor of (M17) (`"svd"`). `(n_neurons, 0)` at rank 0.
    S_ : dict of str to numpy.ndarray
        `(n_bins, r_p)` temporal bases. With `canonicalize=True`,
        $V_p\Sigma_p$: column $j$ has norm $\sigma_j$, the $j$-th singular
        value of $B_p$. A display frame: never pass it to `mtdr.mmle`.
    B_ : dict of str to numpy.ndarray
        `(n_neurons, n_bins)` coefficient matrices $B_p=W_pS_p^\top$ (M1);
        frame-independent.
    intercept_ : numpy.ndarray or None
        `(n_neurons, n_bins)` condition-independent term; `None` with
        `condition_independent=False`.
    noise_precision_ : numpy.ndarray
        `(n_neurons,)` noise precisions $\lambda_i$ (inverse variances).
    weight_posterior_cov_ : numpy.ndarray or None
        `"mmle"`: `(n_neurons, total_rank_, total_rank_)`, the posterior
        covariance $C_i^{-1}$ of neuron $i$'s stacked weights (M25), (M36), in
        the frame of `W_` (`R.T @ Cov @ R` in the canonical frame).
        `None` for `"svd"`.
    log_likelihood_ : float
        The training log-likelihood at the fit, fully normalised, in the
        raw frame. Under `"mmle"` it is the **marginal** likelihood (M24),
        the weights integrated out, and so is `to_xarray()`'s
        `log_likelihood` attribute; the method
        [`log_likelihood`][mtdr.model.MTDR.log_likelihood] is the plug-in
        likelihood (M46), which differs from it on the same data. Under
        `"svd"` the plug-in likelihood of the least-squares fit (M21a). Not
        comparable across estimators; excludes the `basis_ridge` penalty.
    objective_ : float
        What the fit maximised: `log_likelihood_` less
        $\tfrac g2\sum_p\lVert S_p\rVert_F^2$ (`basis_ridge` $=g$).
    n_parameters_ : int
        The count in `aic_`: (M38a) for `"mmle"`, (M21b) for `"svd"`.
    aic_ : float
        `2 * n_parameters_ - 2 * log_likelihood_`. Under `"mmle"` the rank
        search selects with the reference count (M38) instead, so
        `rank_search_history_.aic` holds scores that exceed `aic_` by
        $\sum_pr_p(r_p-1)$ at the same ranks.
    rank_search_history_ : RankSearchHistory or None
        The record of the greedy AIC search (M39); `None` when the ranks were
        fixed.
    estimator_ : str
        The estimator that produced the parameters, `"mmle"` or `"svd"`.
    converged_ : bool
        Whether every stage of the selected MMLE fit passed its convergence
        checks. `False` after an iteration cap, an ECME increase in the NLL,
        or an inner optimiser failure, rejected result or precision bound.
        Parameters are still returned; inspect the warning and check the fit
        as described in the MTDR docs page. `True` does not guarantee a global
        optimum. Always `True` for the SVD estimator.
    n_iter_ : dict of str to int
        Iterations of `"ecme"` and `"refine"` in the final fit, summed over
        its calls of each (empty for `"svd"`).
    condition_independent_ : bool
        Whether the model has the intercept.

    Raises
    ------
    ParameterError
        For an invalid hyper-parameter (checks that need the data happen in
        `fit`).

    Notes
    -----
    `get_params` / `set_params` follow the sklearn contract, so
    `sklearn.base.clone` works; sklearn's cross-validation utilities do not
    (`fit(Y, X)` takes the 3-D activity first). Fitted models pickle.

    Examples
    --------
    >>> from mtdr import MTDR, simulate
    >>> sim = simulate(n_neurons=40, n_bins=8, n_trials=200, ranks=[2, 1],
    ...                drop_prob=0.2, seed=0)
    >>> model = MTDR(ranks=[2, 1]).fit(sim.Y_masked, sim.X,
    ...                                regressor_names=["stim", "choice"])
    >>> model.ranks_, model.W_["stim"].shape, model.B_["choice"].shape
    ({'stim': 2, 'choice': 1}, (40, 2), (40, 8))
    >>> MTDR(ranks="aic", estimator="svd", max_rank=4)
    MTDR(estimator='svd', max_rank=4)
    """

    def __init__(
        self,
        ranks: Sequence[int] | Mapping[str, int] | Literal["aic"] = "aic",
        estimator: Literal["svd", "mmle"] = "mmle",
        condition_independent: bool = True,
        max_rank: int | None = None,
        min_observations: int = 2,
        ridge: float = 0.0,
        basis_ridge: float = 0.0,
        rank_search_init: Literal["svd_weighted", "svd", "ones"]
        | Sequence[int] = "svd_weighted",
        rank_search_threshold: float = 0.0,
        rank_search_warm_start: bool = True,
        ecme_max_iter: int = 100,
        ecme_tol: float = 1.0,
        refine_max_iter: int = 10,
        refine_tol: float = 1e-4,
        convergence_eps: float = 1e-12,
        optimizer_max_iter: int = 2000,
        optimizer_tol: float = 1e-6,
        optimizer_progtol: float = 1e-9,
        canonicalize: bool = True,
        verbose: int = 0,
        n_jobs: int | None = None,
        basis_preconditioning: bool = False,
    ) -> None:
        self.ranks = ranks
        self.estimator = estimator
        self.condition_independent = condition_independent
        self.max_rank = max_rank
        self.min_observations = min_observations
        self.ridge = ridge
        self.basis_ridge = basis_ridge
        self.rank_search_init = rank_search_init
        self.rank_search_threshold = rank_search_threshold
        self.rank_search_warm_start = rank_search_warm_start
        self.ecme_max_iter = ecme_max_iter
        self.ecme_tol = ecme_tol
        self.refine_max_iter = refine_max_iter
        self.refine_tol = refine_tol
        self.convergence_eps = convergence_eps
        self.optimizer_max_iter = optimizer_max_iter
        self.optimizer_tol = optimizer_tol
        self.optimizer_progtol = optimizer_progtol
        self.canonicalize = canonicalize
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.basis_preconditioning = basis_preconditioning
        self._params()

    # ------------------------------------------------------------------ sklearn

    @classmethod
    def _param_names(cls) -> list[str]:
        return [p for p in inspect.signature(cls.__init__).parameters if p != "self"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the hyper-parameters, as passed to the constructor (sklearn).

        Parameters
        ----------
        deep : bool
            Ignored (no nested estimators); accepted for sklearn compatibility.

        Returns
        -------
        dict of str to object
            Every constructor argument by name.

        Examples
        --------
        >>> from mtdr import MTDR
        >>> MTDR(ranks=[1, 2]).get_params()["ranks"]
        [1, 2]
        """
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> MTDR:
        """Set hyper-parameters by name and check them (sklearn).

        Parameters
        ----------
        **params
            Constructor arguments to change.

        Returns
        -------
        MTDR
            `self`.

        Raises
        ------
        ParameterError
            For an unknown name or an invalid value (the previous values are
            kept).

        Examples
        --------
        >>> from mtdr import MTDR
        >>> MTDR().set_params(estimator="svd", max_rank=3)
        MTDR(estimator='svd', max_rank=3)
        """
        names = self._param_names()
        unknown = sorted(set(params) - set(names))
        if unknown:
            raise ParameterError(f"unknown MTDR parameters {unknown}; valid: {names}")
        saved = {name: getattr(self, name) for name in params}
        for name, value in params.items():
            setattr(self, name, value)
        try:
            self._params()
        except ParameterError:
            for name, value in saved.items():
                setattr(self, name, value)
            raise
        return self

    def __repr__(self) -> str:
        """Print the class with its non-default hyper-parameters."""
        defaults = {
            name: p.default
            for name, p in inspect.signature(type(self).__init__).parameters.items()
            if name != "self"
        }
        shown = [
            f"{name}={getattr(self, name)!r}"
            for name, default in defaults.items()
            if not _same(getattr(self, name), default)
        ]
        return f"{type(self).__name__}({', '.join(shown)})"

    def __getattr__(self, name: str) -> Any:
        """Raise `NotFittedError` for a fitted attribute read before `fit`."""
        if name in _FITTED:
            raise NotFittedError(
                f"this MTDR instance is not fitted yet; call fit before reading {name}"
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def _params(self) -> dict[str, Any]:
        """Check every hyper-parameter; return them normalised."""
        out: dict[str, Any] = {}
        ranks = self.ranks
        if isinstance(ranks, str):
            if ranks != "aic":
                raise ParameterError(
                    f"ranks must be 'aic', a sequence of int or a mapping of name to "
                    f"int; got {ranks!r}"
                )
            out["ranks"] = "aic"
        elif isinstance(ranks, Mapping):
            fixed: dict[str, int] = {}
            for key, value in ranks.items():
                r = as_int(value)
                if r is None or r < 0 or not isinstance(key, str):
                    raise ParameterError(
                        f"ranks must map regressor names to non-negative ints; got "
                        f"{dict(ranks)!r}"
                    )
                fixed[key] = r
            out["ranks"] = fixed
        else:
            entries = as_list(ranks)
            if entries is None or not entries:
                raise ParameterError(
                    f"ranks must be 'aic', a non-empty sequence of int or a mapping; "
                    f"got {ranks!r}"
                )
            seq = [as_int(e) for e in entries]
            if any(r is None or r < 0 for r in seq):
                raise ParameterError(
                    f"ranks must be non-negative integers; got {ranks!r}"
                )
            out["ranks"] = [int(r) for r in seq if r is not None]
        if self.estimator not in ("svd", "mmle"):
            raise ParameterError(
                f"estimator must be 'svd' or 'mmle'; got {self.estimator!r}"
            )
        out["estimator"] = self.estimator
        out["condition_independent"] = _bool(
            "condition_independent", self.condition_independent
        )
        if self.max_rank is None:
            out["max_rank"] = None
        else:
            out["max_rank"] = _positive_int("max_rank", self.max_rank)
        out["min_observations"] = _positive_int(
            "min_observations", self.min_observations
        )
        for name in ("ridge", "basis_ridge", "rank_search_threshold"):
            out[name] = _non_negative(name, getattr(self, name))
        init = self.rank_search_init
        if isinstance(init, str):
            if init not in ("svd_weighted", "svd", "ones"):
                raise ParameterError(
                    "rank_search_init must be 'svd_weighted', 'svd', 'ones' or a "
                    f"sequence of positive int; got {init!r}"
                )
            out["rank_search_init"] = init
        else:
            entries = as_list(init)
            seq = [as_int(e) for e in entries] if entries else [None]
            if any(r is None or r < 1 for r in seq):
                raise ParameterError(
                    "rank_search_init must be 'svd_weighted', 'svd', 'ones' or a "
                    f"non-empty sequence of positive int; got {init!r}"
                )
            out["rank_search_init"] = [int(r) for r in seq if r is not None]
        out["rank_search_warm_start"] = _bool(
            "rank_search_warm_start", self.rank_search_warm_start
        )
        for name in ("ecme_max_iter", "refine_max_iter", "optimizer_max_iter"):
            out[name] = _positive_int(name, getattr(self, name))
        for name in (
            "ecme_tol",
            "refine_tol",
            "convergence_eps",
            "optimizer_tol",
            "optimizer_progtol",
        ):
            out[name] = _positive(name, getattr(self, name))
        out["canonicalize"] = _bool("canonicalize", self.canonicalize)
        level = as_int(self.verbose)
        if level not in (0, 1, 2):
            raise ParameterError(f"verbose must be 0, 1 or 2; got {self.verbose!r}")
        out["verbose"] = level
        out["n_jobs"] = _n_jobs(self.n_jobs)
        out["basis_preconditioning"] = _bool(
            "basis_preconditioning", self.basis_preconditioning
        )
        return out

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        Y: ArrayLike,
        X: ArrayLike,
        mask: ArrayLike | None = None,
        regressor_names: Sequence[str] | None = None,
    ) -> MTDR:
        r"""Fit the model to `Y` given `X`.

        In order: [`validate_inputs`][mtdr.data.validate_inputs] with the
        fit-only checks; the sufficient statistics (M6), (M10)-(M14); for
        `ranks="aic"` the greedy AIC search (M39) of the estimator(s) chosen by
        `rank_search_init`, else one fit at the fixed ranks; then the posterior
        weights (M36) (`"mmle"`), the coefficient matrices (M37), the raw-frame
        likelihood and AIC, and canonicalisation.

        The MMLE search selects with the reference count (M38),
        $n+T\sum_pr_p+nT$, and `n_parameters_` / `aic_` report the identifiable
        count (M38a); its history records the selection scores. The
        `DesignWarning`s of the candidate fits (rank-deficient neurons) are
        aggregated into one, and so are the `ConvergenceWarning`s of a search.

        Parameters
        ----------
        Y : array_like
            `(n_trials, n_neurons, n_bins)`; `NaN` allowed (see Notes).
        X : array_like
            `(n_trials, n_regressors)`, no constant column.
        mask : array_like, optional
            `(n_trials, n_neurons)` bool; `None` infers it from `NaN`.
        regressor_names : sequence of str, optional
            Unique names; `"intercept"` and `"total"` are reserved. `None`
            takes the column names of an `X` that has string `columns` (a
            pandas DataFrame), in order, else `("x0", "x1", ...)`.

        Returns
        -------
        MTDR
            `self`, fitted.

        Raises
        ------
        ValidationError
            For invalid or degenerate data, including neurons below the
            observation floor, zero-variance neurons, a rank-deficient pooled
            `X`, a column collinear with the intercept, or a fitted neuron with
            zero residual.
        ParameterError
            For names of the wrong length or repeated; a `ranks` mapping that
            does not name exactly the regressors, a sequence of the wrong
            length, or a rank above `max_rank`; `max_rank` above
            `min(n_neurons, n_bins)`.
        numpy.linalg.LinAlgError
            If a posterior precision $C_i$ fails its Cholesky.

        Warns
        -----
        ConvergenceWarning
            When a loop hit its cap or an inner optimiser failed.
        DesignWarning
            For rank-deficient neurons, trials with no observed neuron,
            near-collinear columns, column-scale disparities when
            `basis_ridge > 0`, and, under `"mmle"`, continuous columns whose
            mean is more than one standard deviation from zero (the marginal
            likelihood depends on a column's origin, `docs/model.md` § 10.2).

        Notes
        -----
        `fit` cannot tell the neuron axis from the bin axis: `Y` laid out
        `(n_trials, n_bins, n_neurons)` fits without an error, with neurons and
        bins swapped, so check `model.n_neurons_` and `model.n_bins_` after
        fitting. The observation unit is the neuron-trial: without a
        `mask`, one `NaN` bin makes the whole neuron-trial unobserved, its
        finite bins included; impute partial `NaN`s first to keep those bins
        (with a `mask`, a `NaN` under `True` is an error).

        Examples
        --------
        >>> from mtdr import MTDR, simulate
        >>> sim = simulate(n_neurons=60, n_bins=10, n_trials=300, ranks=[2, 1],
        ...                drop_prob=0.2, seed=3)
        >>> model = MTDR(ranks="aic", max_rank=5).fit(sim.Y_masked, sim.X)
        >>> model.ranks_, model.rank_search_history_.stop_reason
        ({'x0': 2, 'x1': 1}, 'no_improvement')
        """
        prm = self._params()
        ci, estimator = prm["condition_independent"], prm["estimator"]
        if regressor_names is None:
            regressor_names = _column_names(X)  # a DataFrame's names
        Y_arr = _as_array("Y", Y)
        X_raw = _as_array("X", X)
        n_regressors = X_raw.shape[1] if X_raw.ndim == 2 else 0
        # The floor is raised under "mmle", as in check_design(estimator=...).
        floor, note = _observation_floor(
            prm["min_observations"], estimator, n_regressors, ci
        )
        Y_v, X_v, mask_v = _validated(Y_arr, X_raw, mask, True, ci, floor, note)
        assert X_v is not None
        _, n_neurons, n_bins = Y_v.shape
        P = X_v.shape[1]
        names = require_names(regressor_names, P)
        cap = min(n_neurons, n_bins)
        max_rank = cap if prm["max_rank"] is None else prm["max_rank"]
        if max_rank > cap:
            raise ParameterError(
                f"max_rank is {max_rank}, above min(n_neurons, n_bins) = {cap}"
            )
        fixed = _fixed_ranks(prm["ranks"], names, max_rank)
        for message in _soft_warnings(X_v, mask_v, names, prm["basis_ridge"]):
            warnings.warn(message, DesignWarning, stacklevel=2)
        uncentred = _uncentred_message(X_v, mask_v, names)
        if estimator == "mmle" and uncentred is not None:
            warnings.warn(uncentred, DesignWarning, stacklevel=2)
        stats = sufficient_statistics(Y_v, X_v, mask_v)
        runner = _Runner(stats, prm, names, max_rank)
        if fixed is not None:
            fit_obj = runner.single(fixed)
            history = None
        else:
            fit_obj, history = runner.search()
        runner.report()
        self._store(fit_obj, history, stats, names, mask_v, prm)
        return self

    def _store(
        self,
        fit_obj: MMLEFit | SVDFit,
        history: RankSearchHistory | None,
        stats: SufficientStats,
        names: tuple[str, ...],
        mask: BoolArray,
        prm: dict[str, Any],
    ) -> None:
        """Set the fitted attributes from a fit object."""
        ranks = tuple(int(r) for r in fit_obj.ranks)
        B = tuple(np.array(b) for b in fit_obj.B)
        cov: FloatArray | None
        if isinstance(fit_obj, MMLEFit):
            W_raw, S_raw = fit_obj.W, fit_obj.S
            cov = np.array(fit_obj.W_cov)
            objective = fit_obj.objective
            converged = fit_obj.converged
            n_iter = dict(fit_obj.n_iter)
            estimator = "mmle"
        else:
            W_raw, S_raw = fit_obj.W, fit_obj.S
            cov = None
            objective = fit_obj.log_likelihood
            converged = True
            n_iter = {}
            estimator = "svd"
        if prm["canonicalize"]:
            W, S, R = _canonical(B, W_raw, ranks)
            if cov is not None and any(ranks):
                rot = scipy.linalg.block_diag(*R)
                cov = rot.T @ cov @ rot
                cov = 0.5 * (cov + cov.transpose(0, 2, 1))
        else:
            W = [np.array(w) for w in W_raw]
            S = [np.array(s) for s in S_raw]
        intercept = None if fit_obj.intercept is None else np.array(fit_obj.intercept)
        self._fit_object = fit_obj
        self._canonical_frame = bool(prm["canonicalize"])
        self.regressor_names_ = names
        self.n_trials_ = int(mask.shape[0])
        self.n_neurons_ = int(stats.n_neurons)
        self.n_bins_ = int(stats.n_bins)
        self.n_regressors_ = len(names)
        self.n_observations_ = np.array(stats.n_obs, dtype=np.int64)
        self.ranks_ = dict(zip(names, ranks, strict=True))
        self.total_rank_ = int(sum(ranks))
        self.W_ = dict(zip(names, W, strict=True))
        self.S_ = dict(zip(names, S, strict=True))
        self.B_ = dict(zip(names, B, strict=True))
        self.intercept_ = intercept
        self.noise_precision_ = np.array(fit_obj.noise_precision)
        self.weight_posterior_cov_ = cov
        self.log_likelihood_ = float(fit_obj.log_likelihood)
        self.objective_ = float(objective)
        self.n_parameters_ = int(fit_obj.n_parameters)
        self.aic_ = float(fit_obj.aic)
        self.rank_search_history_ = history
        self.estimator_ = estimator
        self.converged_ = bool(converged)
        self.n_iter_ = n_iter
        self.condition_independent_ = fit_obj.intercept is not None

    def _check_fitted(self) -> None:
        if "_fit_object" not in self.__dict__:
            raise NotFittedError("this MTDR instance is not fitted yet; call fit first")

    # ------------------------------------------------------------------ evaluation

    def predict(self, X: ArrayLike) -> FloatArray:
        r"""Return the expected response `intercept_ + sum_p X[k, p] B_[p]`, (M1).

        Parameters
        ----------
        X : array_like
            `(n_trials, n_regressors)`, finite.

        Returns
        -------
        numpy.ndarray
            `(n_trials, n_neurons, n_bins)`, for every neuron on every trial
            (no mask; slice it yourself).

        Raises
        ------
        ValidationError
            If `X` is not finite or has the wrong number of columns.
        NotFittedError
            Before `fit`.

        Examples
        --------
        >>> import numpy as np
        >>> from mtdr import MTDR, simulate
        >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=120, ranks=[1], seed=0)
        >>> model = MTDR(ranks=[1], estimator="svd").fit(sim.Y, sim.X)
        >>> model.predict(sim.X[:3]).shape
        (3, 20, 6)
        """
        self._check_fitted()
        X_arr = self._design(X)
        out = np.einsum("kp,pit->kit", X_arr, np.stack(list(self.B_.values())))
        if self.intercept_ is not None:
            out = out + self.intercept_[None]
        result: FloatArray = out
        return result

    @overload
    def log_likelihood(
        self,
        Y: ArrayLike,
        X: ArrayLike,
        mask: ArrayLike | None = ...,
        per_trial: Literal[False] = ...,
    ) -> float: ...

    @overload
    def log_likelihood(
        self,
        Y: ArrayLike,
        X: ArrayLike,
        mask: ArrayLike | None = ...,
        *,
        per_trial: Literal[True],
    ) -> FloatArray: ...

    def log_likelihood(
        self,
        Y: ArrayLike,
        X: ArrayLike,
        mask: ArrayLike | None = None,
        per_trial: bool = False,
    ) -> float | FloatArray:
        r"""Plug-in Gaussian log-likelihood of `Y` under the fit, fully normalised.

        $\ell=\sum_{(k,i):\,h_{ki}}\sum_t\big[\tfrac12\log\lambda_i-\tfrac12\log2\pi
        -\tfrac{\lambda_i}2(Y_{kit}-\hat Y_{kit})^2\big]$ with $\hat Y$ from
        [`predict`][mtdr.model.MTDR.predict]: (M46) with $x$ the trial's
        regressors, summed over observed entries. Comparable across
        estimators, unlike `log_likelihood_`. Neurons must be the fitted ones in
        the same order; a neuron never observed in `Y` is allowed.

        Parameters
        ----------
        Y : array_like
            `(n_trials, n_neurons, n_bins)`, `NaN` allowed; the fitted neurons
            in the same order, trials may be held out.
        X : array_like
            `(n_trials, n_regressors)`, in the coding used at `fit`.
        mask : array_like, optional
            `(n_trials, n_neurons)` bool; `None` infers it from `NaN`.
        per_trial : bool
            Return the `(n_trials,)` per-trial sums instead (`NaN` for a trial
            with no observed neuron).

        Returns
        -------
        float or numpy.ndarray
            The log-likelihood.

        Raises
        ------
        ValidationError
            On a shape mismatch with the fit, or invalid data.
        NotFittedError
            Before `fit`.

        Examples
        --------
        >>> from mtdr import MTDR, simulate
        >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=120, ranks=[1], seed=0)
        >>> model = MTDR(ranks=[1], estimator="svd").fit(sim.Y, sim.X)
        >>> bool(abs(model.log_likelihood(sim.Y, sim.X) - model.log_likelihood_) < 1e-6)
        True
        """
        self._check_fitted()
        flag = _bool("per_trial", per_trial)
        Y_v, X_v, mask_v = self._evaluation_inputs(Y, X, mask)
        assert X_v is not None
        resid = np.where(mask_v[:, :, None], Y_v - self.predict(X_v), 0.0)
        lam = self.noise_precision_
        T = self.n_bins_
        per_neuron = 0.5 * T * (np.log(lam) - _LOG_2PI)
        terms = mask_v * per_neuron[None, :] - 0.5 * np.einsum(
            "i,kit->ki", lam, resid * resid
        )
        trials = terms.sum(axis=1)
        if flag:
            out: FloatArray = np.where(mask_v.any(axis=1), trials, np.nan)
            return out
        return float(trials.sum())

    def score(self, Y: ArrayLike, X: ArrayLike, mask: ArrayLike | None = None) -> float:
        """Return the plug-in log-likelihood (M46) summed over trials.

        The same number as [`log_likelihood`][mtdr.model.MTDR.log_likelihood],
        for sklearn's convention (higher is better).

        Parameters
        ----------
        Y : array_like
            `(n_trials, n_neurons, n_bins)`, `NaN` allowed; the fitted neurons
            in the same order, trials may be held out.
        X : array_like
            `(n_trials, n_regressors)`, in the coding used at `fit`.
        mask : array_like, optional
            `(n_trials, n_neurons)` bool; `None` infers it from `NaN`.

        Returns
        -------
        float
            The plug-in log-likelihood.

        Examples
        --------
        >>> from mtdr import MTDR, simulate
        >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=120, ranks=[1], seed=0)
        >>> model = MTDR(ranks=[1], estimator="svd").fit(sim.Y, sim.X)
        >>> model.score(sim.Y, sim.X) == model.log_likelihood(sim.Y, sim.X)
        True
        """
        out = self.log_likelihood(Y, X, mask)
        assert isinstance(out, float)
        return out

    def aic(self) -> float:
        """Return `aic_`, `2 * n_parameters_ - 2 * log_likelihood_`.

        There is no data-taking form: an "AIC" from the plug-in likelihood and
        the `"mmle"` count would not be an AIC.

        Returns
        -------
        float
            The training AIC ((M21b) for `"svd"`, (M38a) for `"mmle"`).

        Examples
        --------
        >>> from mtdr import MTDR, simulate
        >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=120, ranks=[1], seed=0)
        >>> model = MTDR(ranks=[1], estimator="svd").fit(sim.Y, sim.X)
        >>> model.aic() == model.aic_
        True
        """
        self._check_fitted()
        return self.aic_

    # ------------------------------------------------------------------ project

    @overload
    def project(
        self,
        Y: ArrayLike,
        regressor: str,
        mask: ArrayLike | None = ...,
        remove_intercept: bool = ...,
        weighted: bool = ...,
        method: Literal["gls", "paper"] = ...,
        basis: Mapping[str, ArrayLike] | None = ...,
    ) -> FloatArray: ...

    @overload
    def project(
        self,
        Y: ArrayLike,
        regressor: None,
        mask: ArrayLike | None = ...,
        remove_intercept: bool = ...,
        weighted: bool = ...,
        method: Literal["gls", "paper"] = ...,
        basis: Mapping[str, ArrayLike] | None = ...,
    ) -> dict[str, FloatArray]: ...

    def project(
        self,
        Y: ArrayLike,
        regressor: str | None,
        mask: ArrayLike | None = None,
        remove_intercept: bool = True,
        weighted: bool = True,
        method: Literal["gls", "paper"] = "gls",
        basis: Mapping[str, ArrayLike] | None = None,
    ) -> FloatArray | dict[str, FloatArray]:
        r"""Project activity onto regressor subspaces: trajectories.

        `method="gls"` (default, (M49a)): per trial $k$ and bin $t$, the
        weighted least-squares coordinates
        $z=(U^\top D_kU)^{-1}U^\top D_k(y-\hat b)$ over the neurons observed on
        trial $k$, $U$ the basis (`W_[regressor]` in the model's frame, or
        `basis`), $D_k$ their precisions (identity with `weighted=False`).
        `method="paper"`: $U^\top D(y-\hat b)$ (M49), without normalisation,
        over the observed neurons
        ([Nature Neuroscience supplement, §7](https://doi.org/10.1038/s41593-020-0696-5)).
        `regressor=None` solves jointly on the concatenated bases of every
        regressor (`"gls"` only) and returns a dict.

        A trial whose observed (weighted) basis lacks full column rank, at
        tolerance $\sqrt\epsilon$ relative to its largest singular value,
        gets `NaN` coordinates and one
        [`ProjectionWarning`][mtdr.errors.ProjectionWarning] counts them
        (`"gls"` only, where the inverse is taken). A trial with no observed
        neuron gives `NaN` without a warning.

        Parameters
        ----------
        Y : array_like
            `(n_trials, n_neurons, n_bins)`, `NaN` allowed.
        regressor : str or None
            A regressor name, or `None` for the joint projection.
        mask : array_like, optional
            `(n_trials, n_neurons)`.
        remove_intercept : bool
            Subtract `intercept_` first (when the model has one).
        weighted : bool
            Weight neurons by `noise_precision_` (`False`: unit weights).
        method : {"gls", "paper"}
            See above.
        basis : mapping of str to array_like, optional
            `(n_neurons, r)` bases replacing `W_[name]`, e.g. from
            [`orthogonalize`][mtdr.model.MTDR.orthogonalize].

        Returns
        -------
        numpy.ndarray or dict of str to numpy.ndarray
            `(n_trials, n_bins, r)`, or one per regressor for `regressor=None`.

        Raises
        ------
        ParameterError
            For an unknown regressor, a rank-0 regressor without a `basis`, a
            bad `basis`, `method`, or `regressor=None` with `method="paper"`.
        SingularDesignError
            For `regressor=None` when the joint basis is rank-deficient (shared
            directions or `total_rank_ > n_neurons`), naming the regressors.
        ValidationError
            On a shape mismatch or invalid data.

        Warns
        -----
        ProjectionWarning
            Once, with the number of trials whose basis was singular.

        Examples
        --------
        >>> from mtdr import MTDR, simulate
        >>> sim = simulate(n_neurons=30, n_bins=6, n_trials=150, ranks=[2, 1], seed=0)
        >>> model = MTDR(ranks=[2, 1], estimator="svd").fit(sim.Y, sim.X)
        >>> model.project(sim.Y, "x0").shape
        (150, 6, 2)
        >>> {k: v.shape for k, v in model.project(sim.Y, None).items()}
        {'x0': (150, 6, 2), 'x1': (150, 6, 1)}
        """
        self._check_fitted()
        if method not in ("gls", "paper"):
            raise ParameterError(f"method must be 'gls' or 'paper'; got {method!r}")
        if regressor is None and method == "paper":
            raise ParameterError(
                "method='paper' has no joint form (M49); pass a regressor name"
            )
        rm = _bool("remove_intercept", remove_intercept)
        wt = _bool("weighted", weighted)
        bases = self._bases(basis)
        Y_v, _, mask_v = self._evaluation_inputs(Y, None, mask)
        Yc = Y_v - self.intercept_[None] if rm and self.intercept_ is not None else Y_v
        Yc = np.where(mask_v[:, :, None], Yc, 0.0)
        weights: FloatArray = mask_v * (
            self.noise_precision_[None, :] if wt else np.ones((1, self.n_neurons_))
        )
        if regressor is None:
            return self._project_joint(Yc, weights, mask_v, bases)
        if regressor not in self.regressor_names_:
            raise ParameterError(
                f"unknown regressor {regressor!r}; names {self.regressor_names_}"
            )
        U = bases[regressor]
        if U is None:
            raise ParameterError(
                f"regressor {regressor!r} has rank 0: nothing to project onto (pass "
                "a basis for it)"
            )
        if method == "paper":
            z = np.einsum("ir,ki,kit->ktr", U, weights, Yc)
            out: FloatArray = np.where(mask_v.any(axis=1)[:, None, None], z, np.nan)
            return out
        return _gls(U, weights, Yc, mask_v)

    def _project_joint(
        self,
        Yc: FloatArray,
        weights: FloatArray,
        mask: BoolArray,
        bases: dict[str, FloatArray | None],
    ) -> dict[str, FloatArray]:
        live = {n: b for n, b in bases.items() if b is not None and b.shape[1]}
        names = [n for n in self.regressor_names_ if n in live]
        if not names:
            raise ParameterError("every regressor has rank 0: nothing to project onto")
        joint = np.concatenate([live[n] for n in names], axis=1)
        shared = _joint_null(joint, [live[n].shape[1] for n in names], names)
        if shared is not None:
            raise SingularDesignError(
                "the joint basis of the regressors is rank-deficient (they share "
                f"directions, or total width {joint.shape[1]} exceeds n_neurons = "
                f"{joint.shape[0]}); regressors involved: {shared}; project them one "
                "at a time, or orthogonalize first"
            )
        z = _gls(joint, weights, Yc, mask)
        out: dict[str, FloatArray] = {}
        offset = 0
        for name in self.regressor_names_:
            width = live[name].shape[1] if name in live else 0
            out[name] = np.ascontiguousarray(z[:, :, offset : offset + width])
            offset += width
        return out

    def _bases(self, basis: object) -> dict[str, FloatArray | None]:
        """`W_` with the overrides of `basis`; `None` for a rank-0 regressor."""
        out: dict[str, FloatArray | None] = {
            name: (w if w.shape[1] else None) for name, w in self.W_.items()
        }
        if basis is None:
            return out
        if not isinstance(basis, Mapping):
            raise ParameterError(
                f"basis must be a mapping of regressor name to (n_neurons, r) arrays; "
                f"got {type(basis).__name__}"
            )
        for name, value in basis.items():
            if name not in self.regressor_names_:
                raise ParameterError(
                    f"basis names unknown regressor {name!r}; names "
                    f"{self.regressor_names_}"
                )
            arr = np.asarray(value)
            if (
                arr.dtype.kind not in "iuf"
                or arr.ndim != 2
                or arr.shape[0] != self.n_neurons_
                or arr.shape[1] > self.n_neurons_
                or not np.isfinite(arr).all()
            ):
                raise ParameterError(
                    f"basis[{name!r}] must be a finite real (n_neurons, r) array with "
                    f"n_neurons = {self.n_neurons_} and 0 <= r <= n_neurons; got "
                    f"dtype {arr.dtype}, shape {arr.shape}"
                )
            out[name] = arr.astype(np.float64)
        return out

    # ------------------------------------------------------------------ decode

    @overload
    def decode(
        self,
        Y: ArrayLike,
        regressor: str | Sequence[str] | None = ...,
        mask: ArrayLike | None = ...,
        known: Mapping[str, ArrayLike] | None = ...,
        levels: Mapping[str, Sequence[float]] | None = ...,
        bins: int | slice | Sequence[int] | None = ...,
        return_log_likelihood: Literal[False] = ...,
        return_llr: Literal[False] = ...,
    ) -> dict[str, FloatArray]: ...

    @overload
    def decode(
        self,
        Y: ArrayLike,
        regressor: str | Sequence[str] | None = ...,
        mask: ArrayLike | None = ...,
        known: Mapping[str, ArrayLike] | None = ...,
        levels: Mapping[str, Sequence[float]] | None = ...,
        bins: int | slice | Sequence[int] | None = ...,
        *,
        return_log_likelihood: Literal[True],
        return_llr: Literal[False] = ...,
    ) -> tuple[dict[str, FloatArray], FloatArray]: ...

    @overload
    def decode(
        self,
        Y: ArrayLike,
        regressor: str | Sequence[str] | None = ...,
        mask: ArrayLike | None = ...,
        known: Mapping[str, ArrayLike] | None = ...,
        levels: Mapping[str, Sequence[float]] | None = ...,
        bins: int | slice | Sequence[int] | None = ...,
        *,
        return_log_likelihood: Literal[False] = ...,
        return_llr: Literal[True],
    ) -> tuple[dict[str, FloatArray], FloatArray]: ...

    @overload
    def decode(
        self,
        Y: ArrayLike,
        regressor: str | Sequence[str] | None = ...,
        mask: ArrayLike | None = ...,
        known: Mapping[str, ArrayLike] | None = ...,
        levels: Mapping[str, Sequence[float]] | None = ...,
        bins: int | slice | Sequence[int] | None = ...,
        *,
        return_log_likelihood: Literal[True],
        return_llr: Literal[True],
    ) -> tuple[dict[str, FloatArray], FloatArray, FloatArray]: ...

    def decode(
        self,
        Y: ArrayLike,
        regressor: str | Sequence[str] | None = None,
        mask: ArrayLike | None = None,
        known: Mapping[str, ArrayLike] | None = None,
        levels: Mapping[str, Sequence[float]] | None = None,
        bins: int | slice | Sequence[int] | None = None,
        return_log_likelihood: bool = False,
        return_llr: bool = False,
    ) -> dict[str, FloatArray] | tuple[Any, ...]:
        r"""Model-based maximum-likelihood decoding.

        Following the
        [Nature Neuroscience supplement, §6](https://doi.org/10.1038/s41593-020-0696-5),
        for each trial return the regressor values that maximise the plug-in
        log-likelihood (M46) of its observed activity over the selected
        `bins`, given `B_`, `intercept_` and `noise_precision_`. Every
        regressor (of positive rank) is either decoded or `known`. Continuous
        unknowns are solved jointly in closed form, (M47), or (M47a) given the
        known ones. Discrete unknowns (in `levels`) are decoded by the profile
        likelihood (M48a): for every combination of their levels (in
        `itertools.product` order over the decoded regressors, each in the
        given level order; at most 64), the continuous unknowns are re-solved
        and (M46) evaluated; the first maximising combination wins. With
        one two-level discrete regressor `[a, b]`, the log-likelihood ratio
        $\ell(b,\hat x_-\mid b)-\ell(a,\hat x_-\mid a)$ of (M48b) is available;
        `scipy.special.expit(llr)` is the supplement's normalised profile
        likelihood for level `b`. The decoder depends only on `B_`, so it is
        frame-independent.

        A trial whose continuous-unknown normal matrix
        $\Xi_\mathcal U^\top\Lambda\Xi_\mathcal U$ is singular or has a
        condition number above $\epsilon^{-1/2}$ gets `NaN` outputs, and one
        [`DecodingWarning`][mtdr.errors.DecodingWarning] gives the count and
        the bins. A trial with no observed neuron is `NaN` too.

        Parameters
        ----------
        Y : array_like
            `(n_trials, n_neurons, n_bins)`, `NaN` allowed.
        regressor : str, sequence of str or None
            The regressors to decode; `None` is every regressor not in `known`.
            Rank-0 regressors are ignored.
        mask : array_like, optional
            `(n_trials, n_neurons)`.
        known : mapping of str to array_like, optional
            `{name: (n_trials,)}` values held fixed, in the coding used at `fit`.
        levels : mapping of str to sequence of float, optional
            Candidate values of decoded discrete regressors.
        bins : int, slice, sequence of int or None
            The evidence window: one bin (the paper's instantaneous decoder), a
            slice, a list, or `None` for every bin.
        return_log_likelihood : bool
            Also return the maximised log-likelihood `(n_trials,)`, or the
            profile log-likelihoods `(n_trials, n_combinations)` when there are
            discrete unknowns (fully normalised).
        return_llr : bool
            Also return the LLR `(n_trials,)` (second level minus first).

        Returns
        -------
        dict of str to numpy.ndarray, or tuple
            `{name: (n_trials,)}` for every decoded regressor; with either flag,
            the tuple `(values, log_likelihood, llr)` restricted to the
            requested parts, in that order.

        Raises
        ------
        ParameterError
            If a name is both decoded and known, a regressor is in neither,
            `levels` names a regressor not decoded, there are more than 64 level
            combinations, `return_llr` lacks exactly one two-level discrete
            regressor, every decoded regressor has rank 0, or `bins` is invalid.
        ValidationError
            On a shape mismatch or invalid data.

        Warns
        -----
        DecodingWarning
            Once, when some trials' systems are singular.

        Examples
        --------
        >>> import numpy as np
        >>> from mtdr import MTDR, simulate
        >>> sim = simulate(n_neurons=40, n_bins=8, n_trials=300, ranks=[2, 1],
        ...                levels=[[-1, 0, 1], [-1, 1]], seed=0)
        >>> model = MTDR(ranks=[2, 1], estimator="svd").fit(sim.Y, sim.X,
        ...                                                  regressor_names=["s", "c"])
        >>> out, llr = model.decode(sim.Y, ["s", "c"], levels={"c": [-1, 1]},
        ...                         return_llr=True)
        >>> bool(np.mean(out["c"] == sim.X[:, 1]) > 0.9), llr.shape
        (True, (300,))
        """
        self._check_fitted()
        want_ll = _bool("return_log_likelihood", return_log_likelihood)
        want_llr = _bool("return_llr", return_llr)
        Y_v, _, mask_v = self._evaluation_inputs(Y, None, mask)
        names = self.regressor_names_
        live = [n for n in names if self.ranks_[n] > 0]
        # Rank-0 names are checked, then ignored, before any value check.
        known_values = self._known(known, Y_v.shape[0], live)
        decoded = self._decoded(regressor, known_values, live)
        level_sets = self._levels(levels, decoded, live)
        missing = [n for n in live if n not in decoded and n not in known_values]
        if missing:
            raise ParameterError(
                f"regressors {missing} are neither decoded nor known: every regressor "
                "must be one or the other; zero is not a default value"
            )
        if not decoded:
            raise ParameterError(
                "every decoded regressor has rank 0: nothing to decode from"
            )
        discrete = [n for n in decoded if n in level_sets]
        continuous = [n for n in decoded if n not in level_sets]
        n_combos = math.prod(len(level_sets[n]) for n in discrete)
        if n_combos > _MAX_COMBINATIONS:  # before enumerating them
            raise ParameterError(
                f"{n_combos} level combinations; at most {_MAX_COMBINATIONS}"
            )
        combos = list(itertools.product(*(level_sets[n] for n in discrete)))
        if want_llr and not (len(discrete) == 1 and len(level_sets[discrete[0]]) == 2):
            raise ParameterError(
                "return_llr needs exactly one discrete regressor with exactly two "
                "levels"
            )
        t_idx = self._bins(bins)
        result = _decode_trials(
            Y_v,
            mask_v,
            self.intercept_,
            self.noise_precision_,
            {n: self.B_[n] for n in live},
            t_idx,
            continuous,
            discrete,
            combos,
            {n: known_values[n] for n in live if n in known_values},
        )
        if result.n_singular:
            warnings.warn(
                f"{result.n_singular} trials have a singular or ill-conditioned "
                f"decoding system on bins {t_idx.tolist()}; their outputs are NaN",
                DecodingWarning,
                stacklevel=2,
            )
        values = {n: result.values[n] for n in decoded}
        if not (want_ll or want_llr):
            return values
        parts: list[Any] = [values]
        if want_ll:
            parts.append(result.profile if discrete else result.best)
        if want_llr:
            parts.append(result.profile[:, 1] - result.profile[:, 0])
        return tuple(parts)

    def _known(
        self, known: object, n_trials: int, live: Sequence[str]
    ) -> dict[str, FloatArray]:
        if known is None:
            return {}
        if not isinstance(known, Mapping):
            raise ParameterError(
                f"known must be a mapping of name to (n_trials,) values; got "
                f"{type(known).__name__}"
            )
        for name in known:
            if name not in self.regressor_names_:
                raise ParameterError(
                    f"known names unknown regressor {name!r}; names "
                    f"{self.regressor_names_}"
                )
        out: dict[str, FloatArray] = {}
        for name, value in known.items():
            if name not in live:
                continue  # rank 0: ignored, value unchecked
            arr = np.asarray(value)
            if arr.dtype.kind not in "biuf" or arr.shape != (n_trials,):
                raise ValidationError(
                    f"known[{name!r}] must be a real array of shape (n_trials,) = "
                    f"({n_trials},); got dtype {arr.dtype}, shape {arr.shape}"
                )
            arr = arr.astype(np.float64)
            if not np.isfinite(arr).all():
                raise ValidationError(f"known[{name!r}] must be finite")
            out[name] = arr
        return out

    def _decoded(
        self, regressor: object, known: Mapping[str, Any], live: Sequence[str]
    ) -> list[str]:
        """Return the live decoded names; `known` holds live names only."""
        names = self.regressor_names_
        if regressor is None:
            return [n for n in live if n not in known]
        if isinstance(regressor, str):
            requested: list[object] = [regressor]
        else:
            entries = as_list(regressor)
            if entries is None:
                raise ParameterError(
                    f"regressor must be a name, a sequence of names or None; got "
                    f"{regressor!r}"
                )
            requested = entries
        out: list[str] = []
        for j, name in enumerate(requested):
            if not isinstance(name, str) or name not in names:
                raise ParameterError(f"unknown regressor {name!r}; names {names}")
            if name in requested[:j]:
                raise ParameterError(f"regressor {name!r} is listed twice")
            if name not in live:
                continue  # rank 0: ignored
            if name in known:
                raise ParameterError(f"regressor {name!r} is both decoded and known")
            out.append(name)
        return out

    def _levels(
        self, levels: object, decoded: Sequence[str], live: Sequence[str]
    ) -> dict[str, list[float]]:
        if levels is None:
            return {}
        if not isinstance(levels, Mapping):
            raise ParameterError(
                f"levels must be a mapping of name to candidate values; got "
                f"{type(levels).__name__}"
            )
        out: dict[str, list[float]] = {}
        for name, values in levels.items():
            if name in self.regressor_names_ and name not in live:
                continue  # rank 0: ignored
            if name not in decoded:
                raise ParameterError(f"levels names {name!r}, which is not decoded")
            entries = as_list(values)
            reals = [as_real(v) for v in entries] if entries else [None]
            if any(v is None or not np.isfinite(v) for v in reals):
                raise ParameterError(
                    f"levels[{name!r}] must be a non-empty sequence of finite real "
                    f"numbers; got {values!r}"
                )
            out[name] = [float(v) for v in reals if v is not None]
        return out

    def _bins(self, bins: object) -> IntArray:
        T = self.n_bins_
        if bins is None:
            return np.arange(T, dtype=np.int64)
        if isinstance(bins, slice):
            parts = (bins.start, bins.stop, bins.step)
            ints = [None if v is None else as_int(v) for v in parts]
            if any(
                v is not None and i is None for v, i in zip(parts, ints, strict=True)
            ):
                raise ParameterError(
                    f"a bins slice must have int or None start, stop and step; got "
                    f"{bins!r}"
                )
            if ints[2] == 0:
                raise ParameterError(
                    f"a bins slice needs a non-zero step; got {bins!r}"
                )
            idx = np.arange(T)[slice(*ints)]
        else:
            single = as_int(bins)
            entries = [bins] if single is not None else as_list(bins)
            if entries is None:
                raise ParameterError(
                    f"bins must be an int, a slice, a sequence of int or None; got "
                    f"{bins!r}"
                )
            ints = [as_int(b) for b in entries]
            if any(b is None or not -T <= b < T for b in ints):
                raise ParameterError(f"bins must index the {T} time bins; got {bins!r}")
            idx = np.array([b % T for b in ints if b is not None], dtype=np.int64)
        if idx.size == 0:
            raise ParameterError(f"bins selects no time bin: {bins!r}")
        out: IntArray = np.asarray(idx, dtype=np.int64)
        return out

    # ------------------------------------------------------------------ summaries

    @overload
    def explained_variance(
        self,
        Y: ArrayLike,
        X: ArrayLike,
        mask: ArrayLike | None = ...,
        per_bin: Literal[False] = ...,
    ) -> dict[str, float]: ...

    @overload
    def explained_variance(
        self,
        Y: ArrayLike,
        X: ArrayLike,
        mask: ArrayLike | None = ...,
        *,
        per_bin: Literal[True],
    ) -> dict[str, FloatArray]: ...

    def explained_variance(
        self,
        Y: ArrayLike,
        X: ArrayLike,
        mask: ArrayLike | None = None,
        per_bin: bool = False,
    ) -> dict[str, float] | dict[str, FloatArray]:
        r"""Fraction of observed variance each term explains.

        For each term, $1-\lVert Y-\hat Y_p\rVert^2/\lVert Y-\bar Y\rVert^2$ over
        the observed entries, with $\bar Y_i$ neuron $i$'s mean over its
        observed trials **and all bins** and $\hat Y_p$ the prediction that
        keeps only term $p$: $\bar Y+X_{:p}B_p$ for a regressor, `intercept_`
        alone for `"intercept"` (present only when the model has one), and the
        full prediction (M1) for `"total"`. Terms are not orthogonal, so the
        regressors' values need not add up to `"total"`. On held-out trials
        $\bar Y_i$ is the held-out data's own mean, not the training one, so the
        baseline is the best flat prediction of those trials and a held-out
        `"total"` is not a training-mean comparison.

        Parameters
        ----------
        Y : array_like
            `(n_trials, n_neurons, n_bins)`, `NaN` allowed; the fitted neurons
            in the same order, trials may be held out.
        X : array_like
            `(n_trials, n_regressors)`, in the coding used at `fit`.
        mask : array_like, optional
            `(n_trials, n_neurons)` bool; `None` infers it from `NaN`.
        per_bin : bool
            Return `(n_bins,)` arrays, the sums restricted to each bin (the same
            per-neuron $\bar Y_i$).

        Returns
        -------
        dict of str to float or numpy.ndarray
            Keys: the regressor names, `"intercept"` (with an intercept) and
            `"total"`.

        Raises
        ------
        ValidationError
            If an observed total variance it divides by is zero (naming the
            bins), on a shape mismatch, or for invalid data.

        Examples
        --------
        >>> from mtdr import MTDR, simulate
        >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=120, ranks=[1], seed=0)
        >>> model = MTDR(ranks=[1], estimator="svd").fit(sim.Y, sim.X)
        >>> ev = model.explained_variance(sim.Y, sim.X)
        >>> sorted(ev), bool(0 < ev["total"] < 1)
        (['intercept', 'total', 'x0'], True)
        """
        self._check_fitted()
        flag = _bool("per_bin", per_bin)
        Y_v, X_v, mask_v = self._evaluation_inputs(Y, X, mask)
        assert X_v is not None
        m = mask_v[:, :, None]
        counts = mask_v.sum(axis=0) * self.n_bins_
        sums = np.where(m, Y_v, 0.0).sum(axis=(0, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        base = np.broadcast_to(mean[None, :, None], Y_v.shape)
        axes = (0, 1) if flag else (0, 1, 2)

        def energy(pred: FloatArray | np.ndarray[Any, Any]) -> FloatArray:
            d = np.where(m, Y_v - pred, 0.0)
            out: FloatArray = np.sum(d * d, axis=axes)
            return out

        total_energy = energy(base)
        zero = np.atleast_1d(~(total_energy > 0))
        if zero.any():
            where = (
                f"in bins {np.flatnonzero(zero).tolist()}" if flag else "over all bins"
            )
            raise ValidationError(
                f"the observed variance of Y about each neuron's mean is zero {where}; "
                "the explained variance is undefined"
            )
        out: dict[str, Any] = {}
        for p, name in enumerate(self.regressor_names_):
            pred = base + X_v[:, p, None, None] * self.B_[name][None]
            out[name] = 1.0 - energy(pred) / total_energy
        if self.intercept_ is not None:
            out["intercept"] = 1.0 - energy(self.intercept_[None]) / total_energy
        out["total"] = 1.0 - energy(self.predict(X_v)) / total_energy
        if not flag:
            return {k: float(v) for k, v in out.items()}
        return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}

    def orthogonalize(self, order: Sequence[str]) -> dict[str, FloatArray]:
        r"""Ordered Gram-Schmidt of the PC-oriented bases, for display.

        Following the
        [Nature Neuroscience supplement, §4.2](https://doi.org/10.1038/s41593-020-0696-5),
        the PC bases $U_p$ of `B_` (`docs/model.md` § 10.1, computed whatever
        frame the model holds) are concatenated in `order` and orthonormalised
        by rank-revealing modified Gram-Schmidt, column by column with
        re-orthogonalisation: the first regressor's basis is unchanged, and a
        later column whose residual after removing every earlier direction has
        norm at most $\sqrt\epsilon$ (its norm before is 1) is dropped. So the
        width of regressor $p$ is $0\le r^{\rm disp}_p\le r_p$, and a subspace
        inside the earlier ones returns `(n_neurons, 0)`. Not a model frame;
        feed it to `project(..., basis=...)`, never to `subspace_angles`.

        Parameters
        ----------
        order : sequence of str
            Regressor names, the first kept intact; others are omitted and
            rank-0 ones skipped.

        Returns
        -------
        dict of str to numpy.ndarray
            `{name: (n_neurons, r_display)}` with orthonormal columns.

        Raises
        ------
        ParameterError
            For an unknown or repeated name.

        Examples
        --------
        >>> import numpy as np
        >>> from mtdr import MTDR, simulate
        >>> sim = simulate(n_neurons=30, n_bins=6, n_trials=150, ranks=[2, 1], seed=0)
        >>> model = MTDR(ranks=[2, 1], estimator="svd").fit(sim.Y, sim.X)
        >>> Q = model.orthogonalize(["x1", "x0"])
        >>> Q["x1"].shape, Q["x0"].shape
        ((30, 1), (30, 2))
        >>> bool(np.allclose(Q["x1"].T @ Q["x0"], 0))
        True
        """
        self._check_fitted()
        entries = as_list(order)
        if entries is None:
            raise ParameterError(f"order must be a sequence of names; got {order!r}")
        seen: list[str] = []
        for name in entries:
            if not isinstance(name, str) or name not in self.regressor_names_:
                raise ParameterError(
                    f"unknown regressor {name!r}; names {self.regressor_names_}"
                )
            if name in seen:
                raise ParameterError(f"regressor {name!r} is repeated in order")
            seen.append(name)
        Q = np.zeros((self.n_neurons_, 0))
        out: dict[str, FloatArray] = {}
        for name in seen:
            r = self.ranks_[name]
            if r == 0:
                continue
            U = _pc_basis(self.B_[name], r)[0]
            kept: list[FloatArray] = []
            for j in range(r):
                v = U[:, j].copy()
                for _ in range(2):  # re-orthogonalise (twice is enough)
                    if Q.shape[1]:
                        v -= Q @ (Q.T @ v)
                    for q in kept:
                        v -= q * (q @ v)
                norm = float(np.linalg.norm(v))
                if norm > _SQRT_EPS:
                    kept.append(v / norm)
            block = np.column_stack(kept) if kept else np.zeros((self.n_neurons_, 0))
            out[name] = block
            Q = np.concatenate([Q, block], axis=1)
        return out

    def subspace_angles(self, a: str, b: str) -> FloatArray:
        r"""Principal angles between the column spaces of `W_[a]` and `W_[b]`.

        `scipy.linalg.subspace_angles` on orthonormalised copies, so the result
        does not depend on `canonicalize`; the cosines are the canonical
        correlations of the
        [Nature Neuroscience supplement, §8.2](https://doi.org/10.1038/s41593-020-0696-5)
        (`docs/model.md` § 9.3). Do not use the output of `orthogonalize`
        here.

        Parameters
        ----------
        a : str
            A regressor name.
        b : str
            Another (or the same) regressor name.

        Returns
        -------
        numpy.ndarray
            `(min(r_a, r_b),)` angles in radians, ascending; empty at rank 0.

        Raises
        ------
        ParameterError
            For an unknown name.

        Examples
        --------
        >>> from mtdr import MTDR, simulate
        >>> sim = simulate(n_neurons=30, n_bins=6, n_trials=150, ranks=[2, 1], seed=0)
        >>> model = MTDR(ranks=[2, 1], estimator="svd").fit(sim.Y, sim.X)
        >>> model.subspace_angles("x0", "x1").shape
        (1,)
        >>> model.subspace_angles("x0", "x0").round(6)
        array([0., 0.])
        """
        self._check_fitted()
        for name in (a, b):
            if name not in self.regressor_names_:
                raise ParameterError(
                    f"unknown regressor {name!r}; names {self.regressor_names_}"
                )
        Wa, Wb = self.W_[a], self.W_[b]
        if Wa.shape[1] == 0 or Wb.shape[1] == 0:
            return np.zeros(0)
        angles = scipy.linalg.subspace_angles(
            scipy.linalg.orth(Wa), scipy.linalg.orth(Wb)
        )
        out: FloatArray = np.sort(np.clip(angles, 0.0, np.pi / 2))
        return out

    def to_xarray(self) -> xarray.Dataset:
        """Return the fit as an `xarray.Dataset` (optional extra).

        Dims `regressor`, `neuron`, `time`, `component`; variables `B`, `W`,
        `S` (padded with `NaN` beyond each rank), `rank`, `intercept` (absent
        without one) and `noise_precision`; attributes `estimator`, `aic`,
        `log_likelihood`, `n_parameters` and `mtdr_version`. See
        [`mtdr.xarray_io.to_dataset`][mtdr.xarray_io.to_dataset].

        Returns
        -------
        xarray.Dataset
            The dataset.

        Raises
        ------
        ImportError
            If xarray is not installed (`pip install 'mtdr[xarray]'`).
        NotFittedError
            Before `fit`.

        Examples
        --------
        >>> from mtdr import MTDR, simulate
        >>> sim = simulate(n_neurons=20, n_bins=6, n_trials=120, ranks=[1], seed=0)
        >>> ds = MTDR(ranks=[1], estimator="svd").fit(sim.Y, sim.X).to_xarray()
        >>> dict(ds.sizes)
        {'regressor': 1, 'neuron': 20, 'time': 6, 'component': 1}
        """
        self._check_fitted()
        from mtdr.xarray_io import to_dataset

        return to_dataset(self)

    # ------------------------------------------------------------------ helpers

    def _design(self, X: ArrayLike) -> FloatArray:
        arr = _as_array("X", X)
        if arr.dtype.kind not in "biuf" or arr.ndim != 2:
            raise ValidationError(
                f"X must be a real 2-D (n_trials, n_regressors) array; got dtype "
                f"{arr.dtype}, shape {arr.shape}"
            )
        if arr.shape[1] != self.n_regressors_:
            raise ValidationError(
                f"X has {arr.shape[1]} columns; the model has {self.n_regressors_} "
                "regressors"
            )
        out = arr.astype(np.float64)
        if not np.isfinite(out).all():
            raise ValidationError("X must be finite")
        return out

    def _evaluation_inputs(
        self, Y: ArrayLike, X: ArrayLike | None, mask: ArrayLike | None
    ) -> tuple[FloatArray, FloatArray | None, BoolArray]:
        Y_v, X_v, mask_v = validate_inputs(Y, X, mask)
        if Y_v.shape[1:] != (self.n_neurons_, self.n_bins_):
            raise ValidationError(
                f"Y has {Y_v.shape[1]} neurons and {Y_v.shape[2]} bins; the model was "
                f"fitted to {self.n_neurons_} and {self.n_bins_} (the same neurons, in "
                "the same order)"
            )
        if X_v is not None:
            X_v = self._design(X_v)
        return Y_v, X_v, mask_v


# =========================================================================== canonical


def canonical_factors(
    B: ArrayLike, rank: int | None = None
) -> tuple[FloatArray, FloatArray]:
    r"""Factor a coefficient matrix in the paper's PC orientation.

    The frame [`MTDR`][mtdr.model.MTDR] reports `W_` and `S_` in
    (`canonicalize=True`; `docs/model.md` § 10.1;
    [Nature Neuroscience supplement, §4.1](https://doi.org/10.1038/s41593-020-0696-5)),
    for any `(n_neurons, n_bins)` matrix: with the thin SVD
    $B=U\Sigma V^\top$ truncated to `rank` components,

    $$
    W=U_{:,1:r},\qquad S=V_{:,1:r}\,\Sigma_{1:r},\qquad WS^\top=B^{(r)},
    $$

    the columns ordered by decreasing singular value, `W` orthonormal, and the
    column norms of `S` the singular values. Each column's sign is fixed so
    that the entry of `W` of **largest absolute value** is positive;
    of entries exactly equal in absolute value the first (lowest neuron
    index) decides, but entries equal in exact arithmetic usually differ at
    rounding level in the computed SVD, so for such a column the sign is
    arbitrary. Use it to put a known coefficient matrix, such as a
    simulation's true `B[p]`, into the frame of a fit before comparing time
    components column by column.

    Parameters
    ----------
    B : array_like
        `(n_neurons, n_bins)` real, finite matrix.
    rank : int, optional
        Number of components, `0 <= rank <= min(n_neurons, n_bins)`. `None`
        keeps the numerical rank of `B` ([`numpy.linalg.matrix_rank`][]'s
        default tolerance, $\sigma_1\max(n,T)\,\epsilon$), so a true $B_p$ of
        rank $r_p$ gives $r_p$ components rather than rounding noise.

    Returns
    -------
    W : numpy.ndarray
        `(n_neurons, rank)` orthonormal columns.
    S : numpy.ndarray
        `(n_bins, rank)` time components, column `j` of norm $\sigma_j$.

    Raises
    ------
    ParameterError
        If `B` is not a 2-D real finite array, or `rank` is not an integer in
        `[0, min(n_neurons, n_bins)]`.

    Notes
    -----
    Components whose singular values are equal, or nearly so, are determined
    only up to a rotation within their block, not merely up to sign: compare
    such components as a subspace (principal angles), not column by column.
    The frame is a display convention; it is not a symmetry of the marginal
    likelihood, so canonical factors are not inputs for
    [`mtdr.mmle`][mtdr.mmle] (`docs/model.md` § 10.1).

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import canonical_factors
    >>> B = np.outer([1.0, -3.0, 2.0], [1.0, 0.5])      # rank 1
    >>> W, S = canonical_factors(B)
    >>> W.shape, S.shape
    ((3, 1), (2, 1))
    >>> bool(W[1, 0] > 0), bool(np.allclose(W @ S.T, B))
    (True, True)
    """
    try:
        arr = np.asarray(B)
    except (TypeError, ValueError) as err:
        raise ParameterError(f"B must be a 2-D real array; got {B!r}") from err
    if arr.ndim != 2 or arr.size == 0:
        raise ParameterError(
            f"B must be a non-empty 2-D real array; got shape {arr.shape}"
        )
    if arr.dtype.kind not in "iuf":
        raise ParameterError(f"B must be a 2-D real array; got dtype {arr.dtype}")
    mat = arr.astype(np.float64)
    if not np.isfinite(mat).all():
        raise ParameterError("B must be finite")
    cap = min(mat.shape)
    if rank is None:
        r = int(np.linalg.matrix_rank(mat))
    else:
        r_or_none = as_int(rank)
        if r_or_none is None:
            raise ParameterError(f"rank must be an integer or None; got {rank!r}")
        r = r_or_none
        if not 0 <= r <= cap:
            raise ParameterError(f"rank must be between 0 and {cap}; got {r}")
    U, s, V = _pc_basis(mat, r)
    return U, np.ascontiguousarray(V * s)


# =========================================================================== fitting


class _Runner:
    """Runs the fits of one `MTDR.fit` and aggregates their warnings."""

    def __init__(
        self,
        stats: SufficientStats,
        prm: dict[str, Any],
        names: tuple[str, ...],
        max_rank: int,
    ) -> None:
        self.stats = stats
        self.prm = prm
        self.names = names
        self.max_rank = max_rank
        self.ci: bool = prm["condition_independent"]
        self.deficient: set[int] = set()
        self.failed: list[tuple[str, tuple[int, ...], list[str]]] = []
        self.n_fits: dict[str, int] = {}
        self.searched = False
        self.selected: tuple[str, tuple[int, ...]] | None = None
        # The MMLE search's warm starts: the accepted fit, and the current
        # round's candidate fits by rank vector, one of which `_accept` promotes.
        self.accepted: MMLEFit | None = None
        self.round: dict[tuple[int, ...], MMLEFit] = {}
        # The MMLE search's worker processes while `search` runs them;
        # `None` fits the candidates in turn.
        self.pool: Executor | None = None

    def _call(
        self, kind: str, func: Callable[[], MMLEFit | SVDFit]
    ) -> MMLEFit | SVDFit:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = func()
        self._record(kind, out, caught)
        return out

    def _record(
        self, kind: str, out: MMLEFit | SVDFit, caught: list[warnings.WarningMessage]
    ) -> None:
        """Count a fit and keep its warnings for `report`."""
        self.n_fits[kind] = self.n_fits.get(kind, 0) + 1
        self.deficient.update(int(i) for i in out.rank_deficient_neurons)
        messages = []
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                messages.append(str(w.message))
            elif not issubclass(w.category, DesignWarning):
                warnings.warn_explicit(
                    w.message, w.category, w.filename, w.lineno, source=w.source
                )
        if messages:
            self.failed.append((kind, tuple(int(r) for r in out.ranks), messages))

    def _svd(self, ranks: NDArray[np.int64], weighted: bool = False) -> SVDFit:
        out = self._call(
            "svd",
            lambda: fit_svd(
                self.stats,
                ranks,
                ridge=self.prm["ridge"],
                condition_independent=self.ci,
                precision_weighted=weighted,
            ),
        )
        assert isinstance(out, SVDFit)
        return out

    def _mmle(
        self, ranks: NDArray[np.int64], accepted: MMLEFit | None = None
    ) -> MMLEFit:
        """Fit the MMLE at `ranks`, from its SVD fit or warm from `accepted`."""
        options = _mmle_options(self.prm)

        def run() -> MMLEFit:
            # verbose=2: each ECME / refinement line names its candidate.
            active = self.prm["verbose"] == 2
            with _prefixed_stdout(_prefix(ranks), active, sys.stdout):
                return _run_mmle(self.stats, ranks, accepted, options)

        out = self._call("mmle", run)
        assert isinstance(out, MMLEFit)
        return out

    def _mmle_round(self, trials: list[NDArray[np.int64]]) -> list[MMLEFit]:
        """Fit one round of MMLE candidates in the worker processes.

        `greedy_aic`'s `fit_many`. Each worker returns its fit, its warnings and
        what it printed; they are recorded and printed here in regressor order,
        so the warnings, `verbose` output and `n_fits` are those of `_mmle`. If
        the pool breaks (a worker died, or could not start: a script without
        the `__main__` guard), this round and the rest are fitted in turn.
        """
        warm = self.prm["rank_search_warm_start"]
        fit = self._mmle_warm if warm else self._mmle
        if self.pool is None:
            return [fit(ranks) for ranks in trials]
        accepted = self.accepted if warm else None
        options = _mmle_options(self.prm)
        jobs = [(self.stats, r, accepted, options, np.geterr()) for r in trials]
        try:
            results = _gather(self.pool, _pool_fit, jobs)
        except BrokenProcessPool as exc:
            _no_pool_warning(f"the worker processes stopped ({exc})")
            self.pool = None
            return [fit(ranks) for ranks in trials]
        outs = []
        for ranks, (out, caught, printed) in zip(trials, results, strict=True):
            if printed:
                sys.stdout.write(printed)
            self._record("mmle", out, caught)
            if warm:
                self.round[tuple(int(r) for r in ranks)] = out
            outs.append(out)
        return outs

    def _mmle_warm(self, ranks: NDArray[np.int64]) -> MMLEFit:
        """`_mmle` for the search, each candidate warm from the accepted fit.

        The first call is the search's start fit, which stays cold and becomes
        the accepted fit; `_accept`, `greedy_aic`'s callback, moves it on.
        """
        if self.accepted is None:
            self.accepted = self._mmle(ranks)
            return self.accepted
        out = self._mmle(ranks, self.accepted)
        self.round[tuple(int(r) for r in ranks)] = out
        return out

    def _accept(self, step: int, ranks: NDArray[np.int64], score: float) -> None:
        self.accepted = self.round[tuple(int(r) for r in ranks)]
        self.round = {}

    def _selection_score(self, fit: Any, ranks: NDArray[np.int64]) -> float:
        """Return the MMLE search's selection AIC, with the count (M38)."""
        count = _aic.n_parameters_mmle(
            ranks,
            self.stats.n_neurons,
            self.stats.n_bins,
            condition_independent=self.ci,
            formula="reference",
        )
        return _aic.aic(fit.log_likelihood, count)

    def single(self, ranks: list[int]) -> MMLEFit | SVDFit:
        arr = np.array(ranks, dtype=np.int64)
        out: MMLEFit | SVDFit
        out = self._mmle(arr) if self.prm["estimator"] == "mmle" else self._svd(arr)
        self.selected = (self.prm["estimator"], tuple(ranks))
        return out

    def search(self) -> tuple[MMLEFit | SVDFit, RankSearchHistory]:
        self.searched = True
        p = self.prm
        P = len(self.names)
        init = p["rank_search_init"]
        common: dict[str, Any] = {
            "max_rank": self.max_rank,
            "threshold": p["rank_search_threshold"],
            "regressor_names": self.names,
            "verbose": p["verbose"],
        }
        if isinstance(init, list) and len(init) != P:
            raise ParameterError(
                f"rank_search_init has {len(init)} entries; expected {P}"
            )
        if isinstance(init, list) and max(init) > self.max_rank:
            raise ParameterError(
                f"rank_search_init {init} has a rank above max_rank = {self.max_rank}"
            )
        best: MMLEFit | SVDFit
        if p["estimator"] == "svd":
            start = init if isinstance(init, list) else [1] * P
            best, history = greedy_aic(
                self._svd, lambda f, r: f.aic, init_ranks=start, **common
            )
        else:
            svd_stage = None
            if init in ("svd_weighted", "svd"):
                weighted = init == "svd_weighted"
                _, svd_stage = greedy_aic(
                    lambda r: self._svd(r, weighted),
                    lambda f, r: f.aic,
                    init_ranks=[1] * P,
                    **common,
                )
                start = svd_stage.ranks[-1].tolist()
            elif init == "ones":
                start = [1] * P
            else:
                start = init
            warm = p["rank_search_warm_start"]
            # A round has one candidate per regressor below max_rank, and that
            # number only falls as the search goes on.
            movable = sum(r < self.max_rank for r in start)
            n_workers = min(p["n_jobs"], movable)
            with contextlib.ExitStack() as stack:
                # Started before the search's first fit, so that the workers
                # start up while it runs.
                if n_workers > 1:
                    self.pool = _start_pool(stack, n_workers)
                try:
                    best, history = greedy_aic(
                        self._mmle_warm if warm else self._mmle,
                        self._selection_score,
                        init_ranks=start,
                        callback=self._accept if warm else None,
                        fit_many=None if self.pool is None else self._mmle_round,
                        **common,
                    )
                finally:
                    self.pool = None
            if svd_stage is not None:
                history = dataclasses.replace(history, svd_stage=svd_stage)
        self.selected = (p["estimator"], tuple(int(r) for r in best.ranks))
        return best, history

    def report(self) -> None:
        """Emit the aggregated warnings."""
        if self.deficient:
            warnings.warn(
                _deficient_message(np.array(sorted(self.deficient), dtype=np.int64)),
                DesignWarning,
                stacklevel=3,
            )
        if not self.failed:
            return
        if not self.searched:
            for _, _, messages in self.failed:
                for message in messages:
                    warnings.warn(message, ConvergenceWarning, stacklevel=3)
            return
        selected = [
            m for k, r, ms in self.failed for m in ms if (k, r) == self.selected
        ]
        # The denominator counts the fits of the estimators that failed: the
        # SVD stage's fits cannot fail; each failed fit is listed with its first
        # reason, truncated.
        kinds = sorted({k for k, _, _ in self.failed})
        n_fits = sum(self.n_fits[k] for k in kinds)
        listed = ", ".join(
            f"{k} {list(r)} ({_truncated(ms[0])})"
            for k, r, ms in self.failed[:_MAX_LISTED]
        )
        more = (
            f" and {len(self.failed) - _MAX_LISTED} more"
            if len(self.failed) > _MAX_LISTED
            else ""
        )
        head = (
            f"rank search: {len(self.failed)} of {n_fits} {'/'.join(kinds)} fits "
            f"reported non-convergence: {listed}{more}; "
        )
        tail = (
            "the selected fit did: " + " | ".join(selected)
            if selected
            else "the selected fit converged; rejected candidates' failures can "
            "affect which ranks were chosen"
        )
        warnings.warn(head + tail, ConvergenceWarning, stacklevel=3)


def _warm_start(
    stats: SufficientStats,
    accepted: MMLEFit,
    ranks: NDArray[np.int64],
    ridge: float,
    condition_independent: bool,
) -> MMLEFit:
    r"""Return the start for candidate `ranks` from the accepted fit.

    `ranks` is `accepted.ranks` with one regressor $p$ raised by one. The start
    keeps the accepted $S$, $\lambda$ and $b$, and appends to $S_p$ the last
    component of the unweighted SVD fit at `ranks`, the column that fit's own
    cold start (M29) would add, carrying that fit's rank-deficient neurons as
    the cold start does.
    """
    raised = np.flatnonzero(ranks != np.asarray(accepted.ranks))
    assert raised.size == 1
    p = int(raised[0])
    assert ranks[p] == accepted.ranks[p] + 1
    svd = fit_svd(
        stats, ranks, ridge=ridge, condition_independent=condition_independent
    )
    S = list(accepted.S)
    S[p] = np.column_stack([S[p], svd.S[p][:, -1]])
    start = MMLEFit.from_parameters(
        stats, S, accepted.noise_precision, accepted.intercept
    )
    return dataclasses.replace(start, rank_deficient_neurons=svd.rank_deficient_neurons)


def _truncated(message: str, limit: int = _REASON_CHARS) -> str:
    """Return `message` cut to about `limit` characters."""
    return message if len(message) <= limit else message[: limit - 3].rstrip() + "..."


def _mmle_options(prm: dict[str, Any]) -> dict[str, Any]:
    """Return `fit_mmle`'s keyword arguments from the checked hyper-parameters."""
    names = (
        "condition_independent",
        "ecme_max_iter",
        "ecme_tol",
        "refine_max_iter",
        "refine_tol",
        "convergence_eps",
        "optimizer_max_iter",
        "optimizer_tol",
        "optimizer_progtol",
        "ridge",
        "basis_ridge",
    )
    out = {name: prm[name] for name in names}
    out["verbose"] = 2 if prm["verbose"] == 2 else 0
    out["basis_span_scale"] = BASIS_SPAN_SCALE if prm["basis_preconditioning"] else 1.0
    return out


def _run_mmle(
    stats: SufficientStats,
    ranks: NDArray[np.int64],
    accepted: MMLEFit | None,
    options: dict[str, Any],
) -> MMLEFit:
    """Fit the MMLE at `ranks`, from its SVD fit or warm from `accepted`.

    A warm start is never preconditioned: its new column is not yet in
    the span the fit ends with.
    """
    init = None
    if accepted is not None:
        ci = options["condition_independent"]
        init = _warm_start(stats, accepted, ranks, options["ridge"], ci)
        options = {**options, "basis_span_scale": 1.0}
    return fit_mmle(stats, ranks, init, **options)


def _prefix(ranks: NDArray[np.int64]) -> str:
    return f"mmle {[int(r) for r in ranks]}: "


def _prefixed_stdout(
    prefix: str, active: bool, target: TextIO
) -> contextlib.AbstractContextManager[Any]:
    """Send printed lines to `target`, each starting with `prefix`, when `active`."""
    if not active:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(_Prefixed(prefix, target))


# Variables that set the thread count of the BLAS libraries NumPy and SciPy can
# be built with (OpenBLAS, MKL, Accelerate, OpenMP builds); read when a process
# loads its BLAS. `_ENVIRON` serialises the pools that set them.
_BLAS_THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_ENVIRON = threading.Lock()


@contextlib.contextmanager
def _candidate_pool(n_workers: int) -> Iterator[Executor]:
    """Start `n_workers` processes for the MMLE search's candidates.

    They are spawned, so they load NumPy afresh, with the `_BLAS_THREADS`
    variables set to 1: with their default thread counts the workers' BLAS
    threads compete for the cores and a pool gains nothing (a round ran
    0.91-1.19x faster on 4 CPUs, against 2.9x with one thread each). Every
    worker is started here, by a no-op job each (a submit starts a worker
    while none is idle), so the variables are set in `os.environ` only for
    those few milliseconds, under `_ENVIRON`, and the workers start up while
    the caller does other work. A job carries all its inputs, because
    arguments given at a worker's start would make the caller wait for it
    (about 0.35 s per worker). On an error or an interrupt the workers are
    terminated, not waited for.
    """
    pool = ProcessPoolExecutor(
        n_workers, mp_context=multiprocessing.get_context("spawn")
    )
    try:
        with _ENVIRON:
            saved = {name: os.environ.get(name) for name in _BLAS_THREADS}
            os.environ.update(dict.fromkeys(_BLAS_THREADS, "1"))
            try:
                for _ in range(n_workers):
                    pool.submit(_pool_ready)
            finally:
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
        yield pool
    except BaseException:
        _terminate(pool)
        raise
    finally:
        pool.shutdown(wait=True)


def _terminate(pool: ProcessPoolExecutor) -> None:
    """Stop `pool`'s workers now, without waiting for their fits."""
    # The executor has no public way to stop its workers before Python 3.14
    # (`terminate_workers`); its `_processes` mapping is the same from 3.11 on.
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        process.terminate()


def _start_pool(stack: contextlib.ExitStack, n_workers: int) -> Executor | None:
    """Return a started `_candidate_pool` that `stack` closes, or `None`."""
    try:
        return stack.enter_context(_candidate_pool(n_workers))
    except Exception as exc:  # e.g. in a daemonic process, which has no children
        _no_pool_warning(f"could not start {n_workers} worker processes ({exc!r})")
        return None


def _no_pool_warning(reason: str) -> None:
    warnings.warn(
        f"n_jobs: {reason}; the rank search fits its candidates in turn. A script "
        'that sets n_jobs needs the `if __name__ == "__main__":` guard, and a '
        "daemonic process (a multiprocessing.Pool worker) cannot start workers",
        RuntimeWarning,
        stacklevel=4,
    )


def _gather(
    pool: Executor, fn: Callable[..., Any], jobs: list[tuple[Any, ...]]
) -> list[Any]:
    """Run `fn(*job)` for every job in `pool`; return the results in job order.

    A job's error is raised as soon as no earlier job can fail before it, the
    error fitting in turn would have raised; the jobs after it are not waited
    for.
    """
    futures: list[Future[Any]] = [pool.submit(fn, *job) for job in jobs]
    while True:
        failed = next(
            (i for i, f in enumerate(futures) if f.done() and f.exception()), None
        )
        waiting = [f for f in futures[:failed] if not f.done()]
        if not waiting:
            return [f.result() for f in futures]
        wait(waiting, return_when=FIRST_EXCEPTION)


def _pool_ready() -> None:
    """Do nothing: the job that starts a worker."""


def _pool_fit(
    stats: SufficientStats,
    ranks: NDArray[np.int64],
    accepted: MMLEFit | None,
    options: dict[str, Any],
    floating: Mapping[str, Any],
) -> tuple[MMLEFit, list[warnings.WarningMessage], str]:
    """Run one candidate in a worker: its fit, its warnings and its printed lines.

    A worker runs one fit at a time, so recording warnings and redirecting
    `stdout` here is safe, unlike in threads. `floating` is the caller's
    `numpy.geterr()`, which a spawned worker does not inherit.
    """
    printed = io.StringIO()
    with np.errstate(**floating), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _prefixed_stdout(_prefix(ranks), options["verbose"] == 2, printed):
            out = _run_mmle(stats, ranks, accepted, options)
    # The warning's `source` object may not pickle; `_record` does not use it.
    kept = [
        warnings.WarningMessage(w.message, w.category, w.filename, w.lineno)
        for w in caught
    ]
    return out, kept, printed.getvalue()


class _Prefixed(io.TextIOBase):
    """A text stream that prefixes every line it forwards to `target`."""

    def __init__(self, prefix: str, target: TextIO) -> None:
        self.prefix = prefix
        self.target = target
        self.at_start = True

    def write(self, text: str) -> int:
        for piece in text.splitlines(keepends=True):
            if self.at_start:
                self.target.write(self.prefix)
            self.target.write(piece)
            self.at_start = piece.endswith("\n")
        return len(text)


# =========================================================================== kernels


def _canonical(
    B: Sequence[FloatArray], W_raw: Sequence[FloatArray], ranks: Sequence[int]
) -> tuple[list[FloatArray], list[FloatArray], list[FloatArray]]:
    r"""Return the PC orientation and the maps of the covariance.

    $W_p=U_p$, $S_p=V_p\Sigma_p$, and $R_p$ with $U_p=W^{\rm raw}_pR_p$ (least
    squares), so that the covariance becomes $R^\top CR$.
    """
    W: list[FloatArray] = []
    S: list[FloatArray] = []
    R: list[FloatArray] = []
    for Bp, Wp, r in zip(B, W_raw, ranks, strict=True):
        U, s, V = _pc_basis(Bp, r)
        W.append(U)
        S.append(V * s)
        if r:
            R.append(np.linalg.lstsq(Wp, U, rcond=None)[0])
        else:
            R.append(np.zeros((0, 0)))
    return W, S, R


def _pc_basis(B: FloatArray, r: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Top-`r` singular triplets of `B`, largest-entry-positive signs."""
    n, T = B.shape
    if r == 0:
        return np.zeros((n, 0)), np.zeros(0), np.zeros((T, 0))
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    U, s, V = U[:, :r], s[:r], Vt[:r].T
    lead = U[np.argmax(np.abs(U), axis=0), np.arange(r)]
    sign = np.where(lead < 0, -1.0, 1.0)
    return np.ascontiguousarray(U * sign), s, np.ascontiguousarray(V * sign)


def _gls(
    U: FloatArray, weights: FloatArray, Yc: FloatArray, mask: BoolArray
) -> FloatArray:
    r"""Return (M49a) per trial, `NaN` where singular.

    $z_k=(U^\top D_kU)^{-1}U^\top D_k(y_k-\hat b)$, batched over trials.
    """
    G = np.einsum("ki,ir,is->krs", weights, U, U)
    rhs = np.einsum("ki,ir,kit->krt", weights, U, Yc)
    r = U.shape[1]
    out = np.full((Yc.shape[0], Yc.shape[2], r), np.nan)
    if r == 0:
        return np.zeros((Yc.shape[0], Yc.shape[2], 0))
    ok = _well_conditioned(G, _EPS) & mask.any(axis=1)
    if ok.any():
        z = np.linalg.solve(G[ok], rhs[ok])
        out[ok] = z.transpose(0, 2, 1)
    singular = int(np.sum(~ok & mask.any(axis=1)))
    if singular:
        warnings.warn(
            f"{singular} trials have too few observed neurons, or dependent observed "
            "rows, for the basis; their coordinates are NaN",
            ProjectionWarning,
            stacklevel=3,
        )
    return out


def _well_conditioned(G: FloatArray, ratio: float) -> BoolArray:
    r"""Return where symmetric PSD matrices have $\lambda_{min}>r\lambda_{max}$.

    $r$ is `ratio`; a zero matrix is never well conditioned.
    """
    w = np.linalg.eigvalsh(G)
    top = w[:, -1]
    out: BoolArray = (top > 0) & (w[:, 0] > ratio * top)
    return out


def _joint_null(
    joint: FloatArray, widths: Sequence[int], names: Sequence[str]
) -> list[str] | None:
    """Return the regressors in the joint basis's null space, `None` if none.

    `None` means the joint basis has full column rank.
    """
    if joint.shape[1] > joint.shape[0]:
        return list(names)
    _, s, Vt = np.linalg.svd(joint, full_matrices=False)
    tiny = s <= _SQRT_EPS * s[0] if s[0] > 0 else np.ones_like(s, dtype=bool)
    if not tiny.any():
        return None
    null = np.abs(Vt[tiny]).max(axis=0)
    owner = np.repeat(np.arange(len(widths)), widths)
    involved = sorted({int(owner[j]) for j in np.flatnonzero(null > _SQRT_EPS)})
    return [names[j] for j in involved]


@dataclasses.dataclass
class _Decoded:
    values: dict[str, FloatArray]
    profile: FloatArray  # (N, n_combinations)
    best: FloatArray  # (N,)
    n_singular: int


def _decode_trials(
    Y: FloatArray,
    mask: BoolArray,
    intercept: FloatArray | None,
    lam: FloatArray,
    B: Mapping[str, FloatArray],
    bins: IntArray,
    continuous: Sequence[str],
    discrete: Sequence[str],
    combos: Sequence[tuple[float, ...]],
    known: Mapping[str, FloatArray],
) -> _Decoded:
    r"""Run the profile-likelihood decoder (M46)-(M48b), batched over trials.

    The continuous unknowns solve the normal equations (M47a): with
    $H_{pq}[i]=\sum_{t\in\mathcal T}\hat B_p[i,t]\hat B_q[i,t]$, the normal
    matrix of trial $k$ is $G_k=\sum_i h_{ki}\lambda_iH[i]$ and
    $g_k[p]=\sum_i h_{ki}\lambda_i\sum_t\hat B_p[i,t]y_{kit}$ ($y$ less the
    intercept), the known and level terms moved to the right-hand side. Each
    combination's log-likelihood (M46) is then evaluated from the masked
    residuals at the full regressor vector $x_k$ (known, level, solved),
    $\ell_k=c_k-\tfrac12\sum_ih_{ki}\lambda_i\sum_{t\in\mathcal T}
    (y_{kit}-\sum_px_{kp}\hat B_p[i,t])^2$, not from the expanded quadratic
    $\upsilon_k-2x^\top g_k+x^\top G_kx$, whose terms cancel when the
    regressor values are large. The residuals are formed in blocks of trials
    of at most `_DECODE_BLOCK` entries, so the cost is
    $O(N\,n\,\lvert\mathcal T\rvert\,P)$ per combination in bounded memory.
    """
    order = list(B)
    index = {n: j for j, n in enumerate(order)}
    Bsel = np.stack([B[n][:, bins] for n in order])  # (P, n, |T|)
    y = Y[:, :, bins]
    if intercept is not None:
        y = y - intercept[None][:, :, bins]
    y = np.where(mask[:, :, None], y, 0.0)
    w = mask * lam[None, :]
    H = np.einsum("pit,qit->ipq", Bsel, Bsel)
    G = np.einsum("ki,ipq->kpq", w, H)
    g = np.einsum("ki,pit,kit->kp", w, Bsel, y)
    const = 0.5 * bins.size * (mask * (np.log(lam) - _LOG_2PI)[None, :]).sum(axis=1)
    rows_per_block = max(1, _DECODE_BLOCK // max(1, y.shape[1] * y.shape[2]))

    def weighted_rss(x: FloatArray) -> FloatArray:
        r"""Return $\sum_ih_{ki}\lambda_i\sum_t r_{kit}^2$ at regressor values `x`."""
        out = np.empty(x.shape[0])
        for start in range(0, x.shape[0], rows_per_block):
            k = slice(start, start + rows_per_block)
            r = y[k] - np.einsum("kp,pit->kit", x[k], Bsel)
            out[k] = np.einsum("ki,kit->k", w[k], r * r)
        return out

    N = Y.shape[0]
    observed = mask.any(axis=1)
    U = [index[n] for n in continuous]
    D = [index[n] for n in discrete]
    K = [index[n] for n in known]
    x = np.zeros((N, len(order)))
    for n in known:
        x[:, index[n]] = known[n]
    ok = observed.copy()
    if U:
        G_UU = G[:, U][:, :, U]
        ok &= _well_conditioned(G_UU, _SQRT_EPS)
    n_singular = int(np.sum(observed & ~ok))
    profile = np.full((N, len(combos)), np.nan)
    solutions = np.full((N, len(combos), len(U)), np.nan)
    for c, combo in enumerate(combos):
        xc = x.copy()
        for j, value in zip(D, combo, strict=True):
            xc[:, j] = value
        if U:
            fixed = K + D
            rhs = g[:, U] - np.einsum("kuf,kf->ku", G[:, U][:, :, fixed], xc[:, fixed])
            sol = np.full((N, len(U)), np.nan)
            if ok.any():
                sol[ok] = np.linalg.solve(G_UU[ok], rhs[ok][:, :, None])[:, :, 0]
            xc[:, U] = np.where(ok[:, None], sol, 0.0)  # finite residuals
            solutions[:, c] = sol
        ll = const - 0.5 * weighted_rss(xc)
        profile[:, c] = np.where(ok, ll, np.nan)
    best_c = np.zeros(N, dtype=np.int64)
    if ok.any():
        best_c[ok] = np.argmax(profile[ok], axis=1)  # first maximum on ties
    rows = np.arange(N)
    best = np.where(ok, profile[rows, best_c], np.nan)
    values: dict[str, FloatArray] = {}
    for u, name in enumerate(continuous):
        values[name] = np.where(ok, solutions[rows, best_c, u], np.nan)
    combo_arr = np.array(combos, dtype=np.float64).reshape(len(combos), len(D))
    for d, name in enumerate(discrete):
        values[name] = np.where(ok, combo_arr[best_c, d], np.nan)
    return _Decoded(values, profile, best, n_singular)


# =========================================================================== arguments


def _fixed_ranks(
    spec: object, names: tuple[str, ...], max_rank: int
) -> list[int] | None:
    """Resolve a fixed `ranks` specification against the names."""
    if spec == "aic":
        return None
    if isinstance(spec, dict):
        if set(spec) != set(names):
            raise ParameterError(
                f"a ranks mapping must name exactly the regressors {list(names)}; got "
                f"{sorted(spec)}"
            )
        ranks = [spec[n] for n in names]
    else:
        assert isinstance(spec, list)
        if len(spec) != len(names):
            raise ParameterError(
                f"ranks has {len(spec)} entries; X has {len(names)} regressors"
            )
        ranks = list(spec)
    for name, r in zip(names, ranks, strict=True):
        if r > max_rank:
            raise ParameterError(f"ranks[{name!r}] is {r}, above max_rank = {max_rank}")
    return ranks


def _column_names(X: object) -> list[str] | None:
    """Return `X.columns` if they are all strings (a DataFrame's names).

    Duck-typed (no pandas import); anything else gives `None`, the default
    names. The names are validated as `regressor_names` would be.
    """
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    try:
        names = list(columns)
    except TypeError:
        return None
    if names and all(isinstance(name, str) for name in names):
        return names
    return None


def _same(a: object, b: object) -> bool:
    """Whether a hyper-parameter equals its default (for `repr`)."""
    if isinstance(a, bool | np.bool_) or isinstance(b, bool | np.bool_):
        return type(a) is type(b) and a == b
    if a is None or b is None:
        return a is b
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    ra, rb = as_real(a), as_real(b)
    return ra is not None and rb is not None and ra == rb


def _bool(name: str, value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    raise ParameterError(f"{name} must be a bool; got {value!r}")


def _positive_int(name: str, value: object) -> int:
    out = as_int(value)
    if out is None or out < 1:
        raise ParameterError(f"{name} must be a positive integer; got {value!r}")
    return out


def _n_jobs(value: object) -> int:
    """Return the number of worker processes `n_jobs` asks for."""
    if value is None:
        return 1
    if as_int(value) == -1:
        return _cpu_count()
    return _positive_int("n_jobs", value)


def _cpu_count() -> int:
    """Return the CPUs this process may use (all of them before Python 3.13)."""
    count: int | None = getattr(os, "process_cpu_count", os.cpu_count)()
    return max(1, count or 1)


def _positive(name: str, value: object) -> float:
    out = as_real(value)
    if out is None or not np.isfinite(out) or out <= 0:
        raise ParameterError(f"{name} must be a finite real number > 0; got {value!r}")
    return out


def _non_negative(name: str, value: object) -> float:
    out = as_real(value)
    if out is None or not np.isfinite(out) or out < 0:
        raise ParameterError(f"{name} must be a finite real number >= 0; got {value!r}")
    return out
