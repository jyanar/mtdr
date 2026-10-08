r"""Greedy AIC rank search (`docs/model.md` § 7.2, (M39)).

Starting from an initial rank vector, every round refits with each
regressor's rank raised by one, scores each candidate, and accepts the one
that lowers the objective most if it lowers it by more than a threshold;
regressors at `max_rank` leave the candidate set. The reference's
`EstRankGreedily`, with three corrections: acceptance requires a **strict**
decrease (the reference accepts an equal score), the fit stored for an
accepted step is always that step's (the reference indexes its candidate list
by position in the movable set), and the history is kept in memory, including
the terminal rejected round.

[`greedy_aic`][mtdr.rank_search.greedy_aic] is agnostic to the estimator: it
takes a `fit(ranks)` callable and an `objective(fit_object, ranks)` callable,
which is also the only place where the parameter count or the selection
criterion can be changed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mtdr._args import (
    as_int,
    as_list,
    as_real,
    positive_int,
    require_int_vector,
    require_names,
    require_non_negative_real,
)
from mtdr._frozen import ReadOnlyMapping, restore_frozen
from mtdr.errors import ParameterError
from mtdr.mmle import MMLEFit
from mtdr.svd_fit import SVDFit

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["NEAR_TIE_AIC", "RankSearchHistory", "greedy_aic"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_STOP_REASONS = ("no_improvement", "max_rank")

NEAR_TIE_AIC = 2.0
"""Default margin, in AIC units, below which `RankSearchHistory.near_ties` reports
a round."""


@dataclass(frozen=True, eq=False)
class RankSearchHistory:
    r"""Record of a greedy rank search, (M39).

    Row `k` of `ranks` and entry `k` of `aic` and `converged` describe the
    fit after `k` accepted steps (row 0 is the initial vector); `candidates[k]`
    holds the score of every `+1` move tried from row `k`, so the last entry is
    the terminal round, whose best move was rejected (empty when every
    regressor was already at `max_rank`); `accepted[k]` names the regressor
    raised from row `k` to row `k + 1`. Invariant:
    `ranks.shape[0] == aic.shape[0] == len(candidates) == len(converged) ==
    len(accepted) + 1`. The constructor also checks that the record is a
    coherent search: `ranks[0] == init_ranks`; ranks are non-negative
    integers; row `k + 1` is row `k` with `accepted[k]` raised by one, and
    `aic[k + 1] == candidates[k][accepted[k]]`; every name is a regressor name;
    `converged` holds bools; `threshold >= 0`; and the last round is empty
    exactly when `stop_reason == "max_rank"`. The containers are immutable
    copies (tuples, read-only mappings, read-only arrays), also after
    unpickling, so a history cannot be changed after it was checked.

    Attributes
    ----------
    regressor_names : tuple of str
        Names, in rank-vector order.
    estimator : str
        `"svd"` when the fit objects were [`SVDFit`][mtdr.svd_fit.SVDFit]s,
        `"mmle"` when they were [`MMLEFit`][mtdr.mmle.MMLEFit]s, else
        `"custom"`.
    init_ranks : numpy.ndarray
        `(P,)` int, the starting vector.
    ranks : numpy.ndarray
        `(n_accepted + 1, P)` int, the accepted rank vectors.
    aic : numpy.ndarray
        `(n_accepted + 1,)` objective value at each accepted vector.
    candidates : tuple of Mapping of str to float
        Per round, the objective value of each tried move, keyed by the name
        of the regressor raised (read-only mappings).
    accepted : tuple of str
        Per accepted step, the regressor raised.
    converged : tuple of bool
        Per accepted vector, the `converged` attribute of its fit object
        (`True` when the object has none, as for the closed-form SVD fit).
    threshold : float
        The improvement threshold $\delta$ used.
    stop_reason : str
        `"no_improvement"` or `"max_rank"` (every regressor capped).
    svd_stage : RankSearchHistory or None
        The SVD-estimator search that seeded this one, when `MTDR` runs the
        two-stage search; `None` from [`greedy_aic`][mtdr.rank_search.greedy_aic].
    """

    regressor_names: tuple[str, ...]
    estimator: str
    init_ranks: IntArray
    ranks: IntArray
    aic: FloatArray
    candidates: tuple[Mapping[str, float], ...]
    accepted: tuple[str, ...]
    converged: tuple[bool, ...]
    threshold: float
    stop_reason: str
    svd_stage: RankSearchHistory | None = field(default=None)

    def __post_init__(self) -> None:
        """Check the record, copy it into immutable containers."""
        names = require_names(self.regressor_names, _length(self.regressor_names))
        P = len(names)
        init = _rank_array("init_ranks", self.init_ranks)
        ranks = _rank_array("ranks", self.ranks)
        aic = np.array(self.aic, dtype=np.float64)
        accepted = _str_tuple("accepted", self.accepted)
        candidates = _candidate_tuple(self.candidates)
        converged = tuple(self.converged)
        rows = len(accepted) + 1
        if (
            init.shape != (P,)
            or ranks.shape != (rows, P)
            or aic.shape != (rows,)
            or len(candidates) != rows
            or len(converged) != rows
        ):
            raise ParameterError(
                "RankSearchHistory needs ranks.shape[0] == aic.shape[0] == "
                "len(candidates) == len(converged) == len(accepted) + 1 and "
                "P == len(regressor_names)"
            )
        if self.stop_reason not in _STOP_REASONS:
            raise ParameterError(
                f"stop_reason must be one of {_STOP_REASONS}; got {self.stop_reason!r}"
            )
        threshold = require_non_negative_real("threshold", self.threshold)
        if not all(isinstance(flag, bool | np.bool_) for flag in converged):
            raise ParameterError(f"converged must hold bools; got {converged!r}")
        if self.svd_stage is not None and not isinstance(
            self.svd_stage, RankSearchHistory
        ):
            raise ParameterError("svd_stage must be a RankSearchHistory or None")
        if not np.array_equal(ranks[0], init):
            raise ParameterError("ranks[0] must equal init_ranks")
        for k, round_ in enumerate(candidates):
            unknown = set(round_) - set(names)
            if unknown:
                raise ParameterError(
                    f"candidates[{k}] has keys {sorted(unknown)} that are not "
                    "regressor names"
                )
        for k, name in enumerate(accepted):
            if name not in names:
                raise ParameterError(f"accepted[{k}] is {name!r}, not a regressor name")
            step = np.zeros(P, dtype=np.int64)
            step[names.index(name)] = 1
            if not np.array_equal(ranks[k + 1] - ranks[k], step):
                raise ParameterError(
                    f"ranks[{k + 1}] must be ranks[{k}] with {name!r} raised by one"
                )
            if name not in candidates[k] or candidates[k][name] != aic[k + 1]:
                raise ParameterError(
                    f"aic[{k + 1}] must equal candidates[{k}][{name!r}]"
                )
        if (self.stop_reason == "max_rank") != (not candidates[-1]):
            raise ParameterError(
                "the last round of candidates is empty exactly when stop_reason is "
                "'max_rank'"
            )
        for array in (init, ranks, aic):
            array.setflags(write=False)
        for attr, value in (
            ("regressor_names", names),
            ("init_ranks", init),
            ("ranks", ranks),
            ("aic", aic),
            ("candidates", candidates),
            ("accepted", accepted),
            ("converged", tuple(bool(flag) for flag in converged)),
            ("threshold", threshold),
        ):
            object.__setattr__(self, attr, value)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled object with its arrays read-only again."""
        restore_frozen(self, state)

    def final_ranks(self) -> dict[str, int]:
        """Return the chosen ranks, the last row of `ranks`, keyed by name.

        Returns
        -------
        dict of str to int
            `{name: rank}` in regressor order.

        Examples
        --------
        >>> from mtdr.rank_search import greedy_aic
        >>> _, h = greedy_aic(lambda r: None, lambda f, r: (r[0] - 2.0) ** 2, [1], 3)
        >>> h.final_ranks()
        {'x0': 2}
        """
        return {
            name: int(r)
            for name, r in zip(self.regressor_names, self.ranks[-1], strict=True)
        }

    @property
    def n_fits(self) -> int:
        r"""Number of fits this search made: the start plus every candidate.

        One fit at `init_ranks` and one per candidate of every round of (M39),
        the terminal rejected round included: $1+\sum_k$ `len(candidates[k])`.
        It counts this stage's top-level fits only: the preceding SVD stage
        has its own (`svd_stage.n_fits`), and the work inside a fit (an MMLE
        fit's SVD initialiser, its ECME and L-BFGS-B iterations) is not
        counted.

        Examples
        --------
        >>> from mtdr.rank_search import greedy_aic
        >>> _, h = greedy_aic(lambda r: None, lambda f, r: (r[0] - 2.0) ** 2, [1], 3)
        >>> [len(c) for c in h.candidates], h.n_fits
        ([1, 1], 3)
        """
        return 1 + sum(len(round_) for round_ in self.candidates)

    def margins(self) -> dict[str, FloatArray]:
        r"""Return how decisive each round of the greedy search (M39) was.

        For round $k$ (the moves tried from row $k$), with $F_k=$ `aic[k]` and
        the round's candidate scores $F_{k,p}$:

        - `"improvement"`: $F_k-\min_pF_{k,p}$, what the best move gains
          (`NaN` for an empty round);
        - `"decision"`: the improvement minus `threshold`, positive in every
          accepted round and at most 0 in the terminal one: its distance from
          zero is the margin by which the round was decided;
        - `"runner_up"`: the second-smallest candidate score minus the
          smallest (`NaN` with fewer than two candidates), the margin by which
          the move taken beat the next one.

        Returns
        -------
        dict of str to numpy.ndarray
            Three `(n_accepted + 1,)` arrays.

        Examples
        --------
        >>> from mtdr.rank_search import greedy_aic
        >>> _, h = greedy_aic(lambda r: None, lambda f, r: (r[0] - 2.5) ** 2, [1], 4)
        >>> {k: v.round(2).tolist() for k, v in h.margins().items()}
        {'improvement': [2.0, 0.0], 'decision': [2.0, 0.0], 'runner_up': [nan, nan]}
        """
        rounds = len(self.candidates)
        improvement = np.full(rounds, np.nan)
        runner_up = np.full(rounds, np.nan)
        for k, round_ in enumerate(self.candidates):
            scores = np.sort(np.array(list(round_.values()), dtype=np.float64))
            if scores.size:
                improvement[k] = self.aic[k] - scores[0]
            if scores.size > 1:
                runner_up[k] = scores[1] - scores[0]
        return {
            "improvement": improvement,
            "decision": improvement - self.threshold,
            "runner_up": runner_up,
        }

    def near_ties(self, within: float = NEAR_TIE_AIC) -> list[str]:
        r"""List the rounds of the search (M39) decided by less than `within`.

        A round is a near tie when its accepted move gained less than `within`
        beyond the threshold, when its rejected best move came within
        `within` of being accepted (the terminal round), or when the move taken
        beat the runner-up by less than `within`. With the default of 2 AIC
        units, a near tie is an alternative whose relative likelihood
        $\exp(-\Delta\mathrm{AIC}/2)$ is above $1/e$.

        Parameters
        ----------
        within : float
            The margin, `>= 0` (`NEAR_TIE_AIC` = 2 by default).

        Returns
        -------
        list of str
            One readable line per near tie, in round order; empty if none.

        Raises
        ------
        ParameterError
            For a negative or non-finite `within`.

        Examples
        --------
        >>> from mtdr.rank_search import greedy_aic
        >>> _, h = greedy_aic(lambda r: None, lambda f, r: (r[0] - 2.4) ** 2, [1], 4)
        >>> for line in h.near_ties():
        ...     print(line)
        round 0 (accepted x0+1): gained 1.8 beyond the threshold
        round 1 (rejected): best move x0+1 missed acceptance by 0.2
        """
        limit = require_non_negative_real("within", within)
        m = self.margins()
        out: list[str] = []
        for k, round_ in enumerate(self.candidates):
            if not round_:
                continue
            best = min(round_, key=round_.__getitem__)
            decision = float(m["decision"][k])
            if k < len(self.accepted):
                if decision < limit:
                    out.append(
                        f"round {k} (accepted {best}+1): gained {decision:.3g} beyond "
                        "the threshold"
                    )
            elif -decision < limit:
                out.append(
                    f"round {k} (rejected): best move {best}+1 missed acceptance by "
                    f"{0.0 - decision:.3g}"
                )
            runner = float(m["runner_up"][k])
            if np.isfinite(runner) and runner < limit:
                if k < len(self.accepted):
                    out.append(
                        f"round {k}: {best}+1 beat the runner-up by {runner:.3g}"
                    )
                else:  # no move was taken: say so
                    out.append(
                        f"round {k} (no move accepted): best move {best}+1 beat the "
                        f"runner-up by {runner:.3g} and missed acceptance by "
                        f"{0.0 - decision:.3g}"
                    )
        return out

    def to_frame(self) -> pd.DataFrame:
        """Return the history as a `pandas.DataFrame`, one row per accepted vector.

        Columns: `rank[<name>]` (int) per regressor, `aic`, `converged`,
        `candidate[<name>]` (the score of raising that regressor from this row;
        `NaN` when it was capped), `accepted` (the regressor raised from this
        row; missing on the last row), and the round margins `improvement`,
        `decision` and `runner_up` of
        [`margins`][mtdr.rank_search.RankSearchHistory.margins]. The index is
        the step number.

        Returns
        -------
        pandas.DataFrame
            The table.

        Raises
        ------
        ImportError
            If pandas is not installed (it is an optional dependency, part of
            the `xarray` extra).

        Examples
        --------
        >>> from mtdr.rank_search import greedy_aic
        >>> _, h = greedy_aic(lambda r: None, lambda f, r: (r[0] - 2.0) ** 2, [1], 3)
        >>> h.to_frame()[["rank[x0]", "aic", "candidate[x0]"]].to_numpy().tolist()
        [[1.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError(
                "RankSearchHistory.to_frame needs pandas: pip install "
                "'mtdr[xarray]' (from a clone of the repository: pip install "
                "'.[xarray]' in it, or pip install pandas)"
            ) from err
        data: dict[str, Any] = {}
        for p, name in enumerate(self.regressor_names):
            data[f"rank[{name}]"] = self.ranks[:, p].astype(np.int64)
        data["aic"] = np.asarray(self.aic)
        data["converged"] = list(self.converged)
        for name in self.regressor_names:
            data[f"candidate[{name}]"] = [
                round_.get(name, np.nan) for round_ in self.candidates
            ]
        data["accepted"] = [*self.accepted, None]
        data.update(self.margins())
        return pd.DataFrame(data, index=pd.RangeIndex(len(self.aic), name="step"))

    def __repr__(self) -> str:
        """Summarise the search instead of printing every round."""
        return (
            f"RankSearchHistory(estimator={self.estimator!r}, "
            f"init_ranks={self.init_ranks.tolist()}, "
            f"final_ranks={self.final_ranks()}, n_accepted={len(self.accepted)}, "
            f"aic={float(self.aic[-1]):.6g}, stop_reason={self.stop_reason!r})"
        )


def greedy_aic(
    fit: Callable[[IntArray], Any],
    objective: Callable[[Any, IntArray], float],
    init_ranks: Sequence[int] | NDArray[np.integer[Any]],
    max_rank: int,
    threshold: float = 0.0,
    regressor_names: Sequence[str] | None = None,
    verbose: int = 0,
    callback: Callable[[int, IntArray, float], None] | None = None,
    fit_many: Callable[[list[IntArray]], Sequence[Any]] | None = None,
) -> tuple[Any, RankSearchHistory]:
    r"""Choose ranks by greedy coordinate search on an objective, (M39).

    With $F(r)=$ `objective(fit(r), r)` and $r^{(0)}=$ `init_ranks`, each
    round evaluates $F(r+e_p)$ for every regressor $p$ with $r_p<$
    `max_rank` (in regressor order), takes $p^*$, the candidate with the
    smallest score (the lowest index on ties), and accepts it iff

    $$
    F(r)-F(r+e_{p^*})>\delta ,
    $$

    the decrease exceeding the threshold $\delta=$ `threshold` $\ge0$; with
    the default $\delta=0$ an accepted step strictly lowers the score (the
    reference also accepts a tie). The search
    stops at the first round with no accepted move (`"no_improvement"`) or
    with no movable regressor (`"max_rank"`). Ties go to the lowest index;
    there is no random state.

    Parameters
    ----------
    fit : callable
        `fit(ranks)` with `ranks` a fresh `(P,)` `int64` array; returns any fit
        object. Typically `functools.partial(svd_fit.fit_svd, stats)`.
    objective : callable
        `objective(fit_object, ranks)` returns the score to minimise, a real
        number (AIC for `MTDR`, `lambda f, r: f.aic`). `+inf` marks an
        infeasible candidate; `NaN` and `-inf` (an unbounded likelihood, which
        would win every round and freeze the search) are errors.
    init_ranks : sequence of int or 1-D integer array
        Positive starting ranks, each at most `max_rank`.
    max_rank : int
        Largest rank any regressor may reach, positive. For the SVD estimator
        it may not exceed `min(n_neurons, n_bins)`, which `fit_svd` enforces
        when the search reaches it, part-way through (`MTDR` bounds it up
        front).
    threshold : float
        $\delta\ge0$, finite.
    regressor_names : sequence of str, optional
        Names for the history, `("x0", "x1", ...)` by default; unique,
        `"intercept"` and `"total"` reserved.
    verbose : int
        `0` silent; `1` prints the initial vector and every accepted step
        (rank vector and score), so the last line is the chosen vector; `2`
        also prints every round's candidate scores.
    callback : callable, optional
        `callback(step, ranks, score)` after each accepted step, `step`
        counting accepted steps from 1, `ranks` a copy.
    fit_many : callable, optional
        `fit_many(trials)` fits one round's candidates, `trials` a list of
        fresh `(P,)` `int64` arrays in regressor order, and returns their fit
        objects in the same order; for example, fitting them concurrently.
        The default calls `fit` on each in turn. The initial fit always uses
        `fit`. The search depends only on the returned fits, so the result is
        the same either way when `fit_many` returns what `fit` would.

    Returns
    -------
    fit_object
        The fit at the chosen ranks, as returned by `fit`.
    history : RankSearchHistory
        The full trace.

    Raises
    ------
    ParameterError
        For a non-callable `fit`, `objective`, `callback` or `fit_many`; a
        `fit_many` that returns the wrong number of fits; `init_ranks` not a
        sequence of positive integers or above `max_rank`; a non-positive
        `max_rank`; a negative or non-finite `threshold`; bad names; `verbose`
        not 0, 1 or 2; or an objective that returns `NaN`, `-inf` or a
        non-number.
        Errors raised by `fit`, `objective` or `fit_many` propagate.

    Notes
    -----
    Cost: each round costs one fit per movable regressor, so a search that
    accepts $s$ steps over $P$ regressors costs about $(s+1)P$ fits plus the
    initial one. Only the current round's fits are held in memory.

    Examples
    --------
    >>> import functools
    >>> from mtdr import simulate
    >>> from mtdr.rank_search import greedy_aic
    >>> from mtdr.stats import sufficient_statistics
    >>> from mtdr.svd_fit import fit_svd
    >>> sim = simulate(n_neurons=60, n_bins=8, n_trials=300, ranks=[2, 1],
    ...                noise_precision=1.0, seed=0)
    >>> stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    >>> best, history = greedy_aic(functools.partial(fit_svd, stats),
    ...                            lambda f, r: f.aic, init_ranks=[1, 1],
    ...                            max_rank=8, regressor_names=["sa", "choice"])
    >>> history.final_ranks()
    {'sa': 2, 'choice': 1}
    >>> history.ranks.tolist(), history.stop_reason
    ([[1, 1], [2, 1]], 'no_improvement')
    >>> sorted(history.candidates[-1])
    ['choice', 'sa']
    >>> best.ranks
    (2, 1)
    """
    for name, func in (("fit", fit), ("objective", objective)):
        if not callable(func):
            raise ParameterError(f"{name} must be callable; got {func!r}")
    for name, hook in (("callback", callback), ("fit_many", fit_many)):
        if hook is not None and not callable(hook):
            raise ParameterError(f"{name} must be callable or None; got {hook!r}")
    init = require_int_vector("init_ranks", init_ranks, minimum=1)
    cap = positive_int("max_rank", max_rank)
    for p, r in enumerate(init):
        if r > cap:
            raise ParameterError(f"init_ranks[{p}] is {r}, above max_rank = {cap}")
    delta = require_non_negative_real("threshold", threshold)
    names = require_names(regressor_names, len(init))
    level = as_int(verbose)
    if level not in (0, 1, 2):
        raise ParameterError(f"verbose must be 0, 1 or 2; got {verbose!r}")

    current = np.array(init, dtype=np.int64)
    current_fit = fit(current.copy())
    score = _score(objective(current_fit, current.copy()), current)
    estimator = _estimator_name(current_fit)
    ranks_rows = [current.copy()]
    scores = [score]
    converged = [_converged(current_fit)]
    candidates: list[dict[str, float]] = []
    accepted: list[str] = []
    if level:
        print(f"rank search ({estimator}): step 0: {_fmt(names, current)}, {score:.6g}")

    while True:
        movable = [p for p in range(current.size) if current[p] < cap]
        if not movable:
            candidates.append({})
            stop_reason = "max_rank"
            break
        trials = []
        for p in movable:
            trial = current.copy()
            trial[p] += 1
            trials.append(trial)
        round_fits: dict[int, Any] = {}
        round_scores: dict[int, float] = {}
        if fit_many is None:
            for p, trial in zip(movable, trials, strict=True):
                round_fits[p] = fit(trial.copy())
                round_scores[p] = _score(objective(round_fits[p], trial.copy()), trial)
        else:
            fitted = list(fit_many([trial.copy() for trial in trials]))
            if len(fitted) != len(trials):
                raise ParameterError(
                    f"fit_many returned {len(fitted)} fits for {len(trials)} candidates"
                )
            for p, trial, fitted_p in zip(movable, trials, fitted, strict=True):
                round_fits[p] = fitted_p
                round_scores[p] = _score(objective(fitted_p, trial.copy()), trial)
        candidates.append({names[p]: round_scores[p] for p in movable})
        if level >= 2:
            tried = ", ".join(f"{names[p]}+1: {round_scores[p]:.6g}" for p in movable)
            print(f"rank search ({estimator}): candidates {tried}")
        best = min(movable, key=round_scores.__getitem__)  # first index on ties
        if not score - round_scores[best] > delta:
            stop_reason = "no_improvement"
            break
        current[best] += 1
        current_fit = round_fits[best]
        score = round_scores[best]
        ranks_rows.append(current.copy())
        scores.append(score)
        converged.append(_converged(current_fit))
        accepted.append(names[best])
        if level:
            print(
                f"rank search ({estimator}): step {len(accepted)}: "
                f"{_fmt(names, current)}, {score:.6g}"
            )
        if callback is not None:
            callback(len(accepted), current.copy(), score)

    history = RankSearchHistory(
        regressor_names=names,
        estimator=estimator,
        init_ranks=np.array(init, dtype=np.int64),
        ranks=np.stack(ranks_rows),
        aic=np.array(scores, dtype=np.float64),
        candidates=tuple(candidates),
        accepted=tuple(accepted),
        converged=tuple(converged),
        threshold=delta,
        stop_reason=stop_reason,
    )
    return current_fit, history


def _score(value: object, ranks: IntArray) -> float:
    as_float = as_real(value)
    if as_float is None or np.isnan(as_float) or as_float == -np.inf:
        raise ParameterError(
            "objective must return a real number that is not NaN or -inf; got "
            f"{value!r} at ranks {ranks.tolist()}"
        )
    return as_float


def _length(value: object) -> int:
    entries = as_list(value)
    if entries is None:
        raise ParameterError(
            f"regressor_names must be a sequence of str; got {value!r}"
        )
    return len(entries)


def _rank_array(name: str, value: object) -> IntArray:
    """`value` as an `int64` copy: integer dtype (not bool), non-negative."""
    arr = np.asarray(value)
    if arr.dtype.kind not in "iu" or (arr < 0).any():
        raise ParameterError(
            f"{name} must hold non-negative integers; got {arr.tolist()!r}"
        )
    out: IntArray = arr.astype(np.int64, copy=True)
    return out


def _str_tuple(name: str, value: object) -> tuple[str, ...]:
    entries = as_list(value)
    if entries is None or not all(isinstance(e, str) for e in entries):
        raise ParameterError(f"{name} must be a sequence of str; got {value!r}")
    return tuple(str(e) for e in entries)


def _candidate_tuple(value: object) -> tuple[Mapping[str, float], ...]:
    entries = as_list(value)
    message = f"candidates must be a sequence of mappings of str to float; got {value}"
    if entries is None:
        raise ParameterError(message)
    out = []
    for round_ in entries:
        if not isinstance(round_, Mapping):
            raise ParameterError(message)
        scores: dict[str, float] = {}
        for key, score in round_.items():
            as_float = as_real(score)
            if not isinstance(key, str) or as_float is None:
                raise ParameterError(
                    f"candidates must map str to a real number; got {round_!r}"
                )
            scores[key] = as_float
        out.append(ReadOnlyMapping(scores))
    return tuple(out)


def _converged(fit_object: object) -> bool:
    flag = getattr(fit_object, "converged", True)
    if not isinstance(flag, bool | np.bool_):
        raise ParameterError(
            f"the fit object's converged attribute must be a bool; got {flag!r}"
        )
    return bool(flag)


def _estimator_name(fit_object: object) -> str:
    if isinstance(fit_object, SVDFit):
        return "svd"
    return "mmle" if isinstance(fit_object, MMLEFit) else "custom"


def _fmt(names: Sequence[str], ranks: IntArray) -> str:
    return (
        "{"
        + ", ".join(f"{n}: {int(r)}" for n, r in zip(names, ranks, strict=True))
        + "}"
    )
