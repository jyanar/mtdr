"""Ground-truth simulation of the mTDR generative model.

[`simulate`][mtdr.simulation.simulate] draws data the way the reference demo does
(`SimWeights`, `SimConditions`, `SimPopData` and the mask step of `mTDRdemo.m`;
`docs/model.md` § 8, (M40)-(M45)) and returns them with the ground truth in a
[`SimulatedData`][mtdr.simulation.SimulatedData].

Random draws and the transforms applied to them are kept apart: `simulate`
draws standard normals, exponentials, uniform condition indices and uniforms
for the mask from one [`numpy.random.Generator`][] in a fixed order, and the
private functions below map those draws to the outputs deterministically. The
parity tests inject the draws exported from MATLAB
(`tests/fixtures/matlab/make_simulation_fixture.m`) both into these transforms
and, through a replaying generator, into `simulate` itself; the two RNG streams
are never compared.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mtdr._args import RESERVED_NAMES as _RESERVED_NAMES
from mtdr._args import as_bool as _as_bool
from mtdr._args import as_int as _as_int
from mtdr._args import as_list as _as_list
from mtdr._args import as_real as _as_real
from mtdr._args import positive_int as _positive_int
from mtdr._frozen import ReadOnlyMapping as _ReadOnlyMapping
from mtdr._frozen import restore_frozen
from mtdr.errors import ParameterError

__all__ = ["SimulatedData", "simulate"]

FloatArray = NDArray[np.float64]

#: Level sets for the first three regressors when ``levels=None``; any further
#: regressor gets ``[-1, 1]``. The demo's `var_uniq`.
_DEFAULT_LEVELS: tuple[tuple[float, ...], ...] = (
    (-2.0, -1.0, 0.0, 1.0, 2.0),
    (-2.0, -1.0, 0.0, 1.0, 2.0),
    (-1.0, 1.0),
)
_EXTRA_LEVELS: tuple[float, ...] = (-1.0, 1.0)
_DEFAULT_LENGTH_SCALE = 2.0
_DEFAULT_AMPLITUDE = 1.0
#: Mean of the Exponential draw of the noise precisions (`mTDRdemo.m`: `mstdnse`).
_DEFAULT_MEAN_PRECISION = 1.0 / 0.8
#: The validity promise matches `MTDR(min_observations=2)`, the default.
_MIN_OBSERVATIONS = 2
_MAX_MASK_DRAWS = 100
_MAX_CONDITIONS = int(np.iinfo(np.int64).max)


# --------------------------------------------------------------------------- result


@dataclass(frozen=True, eq=False)
class SimulatedData:
    """Simulated responses, design and mask, with the ground truth that made them.

    Returned by [`simulate`][mtdr.simulation.simulate]. Every array is `float64`
    (the mask `bool`, `condition_of_trial` integer) and read-only, and the
    per-regressor mappings are read-only too, so the ground truth cannot be
    changed by accident; copy an array to modify it.

    Attributes
    ----------
    Y : numpy.ndarray
        `(n_trials, n_neurons, n_bins)` noisy responses, **every** entry filled,
        including the unobserved ones, so recovery tests can look under the
        mask. Feed [`Y_masked`][mtdr.simulation.SimulatedData.Y_masked], or `Y`
        together with `mask`, to a fit.
    X : numpy.ndarray
        `(n_trials, n_regressors)` design. It has no column of ones (the
        intercept is separate), but with few trials a sampled column can be
        constant, or the columns collinear, by chance.
    mask : numpy.ndarray
        `(n_trials, n_neurons)` bool; `True` = observed.
    W : Mapping of str to numpy.ndarray
        Per regressor, `(n_neurons, r_p)` neuron weights.
    S : Mapping of str to numpy.ndarray
        Per regressor, `(n_bins, r_p)` temporal bases (GP draws, not
        orthonormalised; `docs/model.md` § 8).
    B : Mapping of str to numpy.ndarray
        Per regressor, `(n_neurons, n_bins)` coefficients `W[p] @ S[p].T`.
    intercept : numpy.ndarray or None
        `(n_neurons, n_bins)` condition-independent term, or `None` when
        simulated with `condition_independent=False`.
    noise_precision : numpy.ndarray
        `(n_neurons,)` noise precision (inverse variance) per neuron.
    ranks : Mapping of str to int
        Per regressor, the generating rank `r_p`: the number of columns of
        `W[p]` and `S[p]`. `B[p]` has exactly this rank unless the amplitude is
        zero or the GP kernel is numerically singular (see
        [`simulate`][mtdr.simulation.simulate]).
    regressor_names : tuple of str
        Names, in `X` column order.
    levels : tuple of numpy.ndarray
        Per regressor, the level set the conditions were drawn from (with a
        supplied `X`, each column's distinct values, sorted).
    condition_of_trial : numpy.ndarray
        `(n_trials,)` index of each trial's condition in the grid of all level
        combinations, ordered with the **first** regressor varying fastest
        (MATLAB `ndgrid`): the condition with level indices `j_0, j_1, ...` has
        index `j_0 + L_0 * (j_1 + L_1 * (j_2 + ...))`, `L_p` the number of
        levels of regressor `p`. With a supplied `X`, the index of the
        trial's row among the distinct rows of `X` in
        [`numpy.unique`][]`(X, axis=0)` order; either way a valid `stratify`
        argument for [`split_trials`][mtdr.data.split_trials].
    seed : int or None
        The integer seed, if one was given; `None` for a
        [`numpy.random.Generator`][], a [`numpy.random.SeedSequence`][] or no
        seed.
    """

    Y: FloatArray
    X: FloatArray
    mask: NDArray[np.bool_]
    W: Mapping[str, FloatArray]
    S: Mapping[str, FloatArray]
    B: Mapping[str, FloatArray]
    intercept: FloatArray | None
    noise_precision: FloatArray
    ranks: Mapping[str, int]
    regressor_names: tuple[str, ...]
    levels: tuple[FloatArray, ...]
    condition_of_trial: NDArray[np.intp]
    seed: int | None

    @property
    def Y_masked(self) -> FloatArray:  # noqa: N802 - the API name
        """`Y` with `NaN` where `mask` is `False`: what a real dataset looks like.

        The port's form of the reference's zeroed `Z_k = diag(h_k) Y_k`, (M45).
        A new, writable array on every access.
        """
        return np.where(self.mask[:, :, None], self.Y, np.nan)

    def fit_kwargs(self) -> dict[str, Any]:
        """Keyword arguments that pass the mask and names to a fit.

        Returns
        -------
        dict
            `{"mask": mask, "regressor_names": regressor_names}`.
        """
        return {"mask": self.mask, "regressor_names": self.regressor_names}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled object with its arrays read-only again."""
        restore_frozen(self, state)

    def __repr__(self) -> str:
        """Summarise shapes and ranks instead of printing the arrays."""
        n_trials, n_neurons, n_bins = self.Y.shape
        return (
            f"SimulatedData(n_trials={n_trials}, n_neurons={n_neurons}, "
            f"n_bins={n_bins}, ranks={self.ranks}, "
            f"intercept={self.intercept is not None}, "
            f"observed={self.mask.mean():.3f}, seed={self.seed})"
        )


# --------------------------------------------------------------------------- public


def simulate(
    n_neurons: int = 100,
    n_bins: int = 15,
    n_trials: int = 100,
    ranks: Sequence[int] | Mapping[str, int] | NDArray[np.integer[Any]] = (2, 1, 3),
    levels: Sequence[ArrayLike] | NDArray[Any] | None = None,
    # Literal defaults, so the docs render the values;
    # tests/test_docs.py checks they equal the module constants.
    length_scale: float | Sequence[float] | NDArray[np.floating[Any]] = 2.0,
    amplitude: float | Sequence[float] | NDArray[np.floating[Any]] = 1.0,
    noise_precision: float | ArrayLike | None = None,
    drop_prob: float = 0.0,
    condition_independent: bool = True,
    regressor_names: Sequence[str] | None = None,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    *,
    X: ArrayLike | None = None,
    S: Sequence[ArrayLike] | Mapping[str, ArrayLike] | None = None,
    W: Sequence[ArrayLike] | Mapping[str, ArrayLike] | None = None,
) -> SimulatedData:
    r"""Simulate population responses with known low-rank structure.

    The generative model (M40)-(M45), with the reference demo's
    choices:

    - each column of `S[p]` is an independent draw from a Gaussian process over
      bins with unit-variance squared-exponential kernel
      $\exp(-(t-t')^2 / 2\ell_p^2)$, time-reversed as in the reference (M41);
    - `W[p]` entries are i.i.d. $\mathcal N(0, a_p^2)$, $a_p$ the amplitude
      (M40), and `B[p] = W[p] @ S[p].T`;
    - with `condition_independent=True` the intercept is $W_0S_0^\top$, with
      $W_0$ of shape `(n_neurons, n_bins)` i.i.d. $\mathcal N(0, a_0^2)$ and
      $S_0$ an `n_bins`-column GP draw: the reference's constant term, with
      `n_bins` generating columns;
    - each trial's regressor values are drawn uniformly, with replacement, from
      the grid of all level combinations (M43);
    - `Y[k] = intercept + sum_p X[k, p] B[p] + E_k`, the rows of `E_k`
      Gaussian with per-neuron precision (M44);
    - each neuron-trial is unobserved independently with probability
      `drop_prob` (M45), conditioned on the validity promise (Notes).

    Any of the drawn pieces can be supplied instead: a design `X`
    replaces the condition draw of (M43), so a task's own incomplete or
    correlated design can be simulated; `S` and `W` replace the GP bases of
    (M41) and the weight draw of (M40), regressor by regressor set, so
    prescribed time courses (epochs that are exactly zero outside a window,
    say) and weight patterns can be simulated. The responses, noise and mask
    are then generated by (M44)-(M45) as usual, and `B[p] = W[p] @ S[p].T`.

    Draw order (`docs/model.md` § D.4): for each regressor in turn, then the
    intercept, the weight normals and then the basis normals; the precisions
    (only when `noise_precision` is `None`); the condition indices; the noise;
    the mask uniforms, once per mask draw. A supplied piece skips its draw
    (the weight normals when `W` is given, the basis normals when `S` is, the
    condition indices when `X` is) and leaves the order of the rest
    unchanged, so with none supplied the draws are exactly those above.

    Parameters
    ----------
    n_neurons : int
        Number of neurons, `>= 1`.
    n_bins : int
        Number of time bins, `>= 1`.
    n_trials : int
        Number of trials: at least 2, so every neuron can be observed twice,
        and at least `n_regressors` plus 1 for the intercept, so the design can
        have full column rank.
    ranks : sequence of int, 1-D integer array, or mapping of str to int
        Generating rank of each regressor's coefficient matrix, `0 <= r_p <=
        min(n_neurons, n_bins)`. A mapping also names the regressors, in
        insertion order. At least one regressor.
    levels : sequence of array_like, or 2-D array, optional
        Level set per regressor (one row per regressor for a 2-D array),
        distinct finite values. `None` gives `[-2, -1, 0, 1, 2]` to the first two
        regressors and `[-1, 1]` to every other. A continuous regressor is a
        level set with many values, e.g. `np.linspace(-1, 1, 101)`; the grid
        of combinations is never materialised, so many continuous regressors
        cost no memory. With `condition_independent=True` every level set
        needs at least two values (a constant column is collinear with the
        intercept).
    length_scale : float or sequence of float
        GP length scale, in bins, of the temporal bases. A sequence has one
        entry per regressor, plus optionally a last entry for the intercept
        (the reference's `len` has one per regressor and the constant term);
        without it the intercept uses the default `2.0`.
    amplitude : float or sequence of float
        Standard deviation of the weight entries (the reference's `rho`), `>=
        0`. Same per-regressor layout as `length_scale`; the intercept default
        is `1.0`.
    noise_precision : float or array_like of shape (n_neurons,), optional
        Noise precision (inverse variance) per neuron, `> 0`. `None` draws each
        from an Exponential distribution with mean `1.25`, as the demo does
        (the reference's `d`; `docs/model.md` § D.3).
    drop_prob : float
        Probability, in `[0, 1)`, that a neuron is unobserved on a trial,
        before the validity conditioning (Notes).
    condition_independent : bool
        Add the intercept. Must be a `bool` (NumPy booleans included).
    regressor_names : sequence of str, optional
        Names in `X` column order; default `("x0", "x1", ...)`. Not allowed
        when `ranks` is a mapping. Unique; `"intercept"` and `"total"` are
        reserved.
    seed : int, numpy.random.Generator, numpy.random.SeedSequence or None
        Seed for [`numpy.random.default_rng`][], or a generator, which is
        advanced.
    X : array_like, optional
        `(n_trials, n_regressors)` design to use instead of drawing trials
        from the level grid (keyword-only). Real (not bool) and finite,
        with exactly `n_trials` rows and one column per entry of `ranks`;
        with `condition_independent=True` no column may be constant (it would
        be collinear with the intercept). `levels` must then be `None`: the
        returned `levels` are each column's distinct values and
        `condition_of_trial` indexes the distinct rows of `X`.
    S : sequence or mapping of array_like, optional
        Temporal bases to use instead of the GP draws (keyword-only): an
        `(n_bins, r_p)` array per regressor (a 1-D array is one column),
        either a sequence in regressor order whose `None` entries are drawn,
        or a mapping from regressor names, the unnamed ones drawn. Each
        width must equal its entry of `ranks`. A supplied regressor's
        `length_scale` entry is unused.
    W : sequence or mapping of array_like, optional
        Neuron weights to use instead of the normal draws (keyword-only): an
        `(n_neurons, r_p)` array per regressor, laid out as `S`. A supplied
        regressor's `amplitude` entry is unused.

    Returns
    -------
    SimulatedData
        Responses, design, mask and ground truth.

    Raises
    ------
    ParameterError
        For an invalid argument: wrong types; mismatched lengths of
        `ranks`, `levels`, `length_scale`, `amplitude` or `regressor_names`; a
        rank outside `[0, min(n_neurons, n_bins)]`; too few trials for the
        design; `drop_prob` outside `[0, 1)`; scales so extreme that the
        simulated values are not finite; a mask that cannot satisfy the
        validity promise in 100 draws; or a supplied `X`, `S` or `W` of the
        wrong type, shape or names, non-finite, or (`X`) with a constant
        column next to the intercept, or given together with `levels`.

    Notes
    -----
    Validity promise: the mask is re-drawn until every trial has an observed
    neuron and every neuron is observed on at least 2 trials (the default
    `min_observations`). The re-draw conditions the mask, so in
    sparse regimes the observed fraction exceeds `1 - drop_prob` (about 0.73
    rather than 0.40 for 3 neurons, 3 trials and `drop_prob=0.6`). Nothing is
    promised about `X`: with few trials a sampled column can be constant, or
    the columns collinear, by chance, which `fit` rejects; check `X` (or use
    more trials) when simulating small designs.

    Ranks: `B[p]` has rank `r_p` when its amplitude is positive and its
    kernel is numerically full rank to `r_p`. At long length scales the
    squared-exponential kernel's trailing eigenvalues fall below machine
    precision (at `n_bins=15`, from a length scale of about 3.5), so the
    trailing basis directions are rounding noise and, once the kernel is
    singular in floating point, `B[p]` and the intercept are rank-deficient.
    Keep `r_p` below the number of kernel eigenvalues above about `1e-12` of
    the largest; the default length scale `2.0` is unaffected.

    Examples
    --------
    >>> import numpy as np
    >>> from mtdr import simulate
    >>> sim = simulate(n_neurons=30, n_bins=10, n_trials=50,
    ...                ranks={"sa": 2, "choice": 1},
    ...                levels=[[-2, -1, 0, 1, 2], [-1, 1]],
    ...                drop_prob=0.2, seed=0)
    >>> sim
    SimulatedData(n_trials=50, n_neurons=30, n_bins=10, ranks={'sa': 2, ...}, ...)
    >>> sim.Y.shape, sim.X.shape, sim.mask.shape
    ((50, 30, 10), (50, 2), (50, 30))
    >>> sim.W["sa"].shape, sim.S["sa"].shape
    ((30, 2), (10, 2))
    >>> int(np.linalg.matrix_rank(sim.B["sa"]))
    2
    >>> bool(np.isnan(sim.Y_masked[~sim.mask]).all())
    True

    A design of one's own, with a prescribed time course for one regressor:

    >>> X = np.column_stack([np.repeat([-1.0, 1.0], 25), np.tile([-1.0, 1.0], 25)])
    >>> profile = np.where(np.arange(10) >= 4, 1.0, 0.0)  # zero before bin 4
    >>> sim = simulate(n_neurons=30, n_bins=10, n_trials=50,
    ...                ranks={"sa": 1, "choice": 1}, X=X,
    ...                S={"sa": profile, "choice": np.ones(10)}, seed=0)
    >>> bool(np.array_equal(sim.X, X)), bool((sim.B["sa"][:, :4] == 0).all())
    (True, True)
    """
    n_neurons = _positive_int("n_neurons", n_neurons)
    n_bins = _positive_int("n_bins", n_bins)
    n_trials = _positive_int("n_trials", n_trials)
    condition_independent = _as_bool("condition_independent", condition_independent)
    names, rank_list = _names_and_ranks(ranks, regressor_names)
    n_regressors = len(names)
    min_trials = max(_MIN_OBSERVATIONS, n_regressors + int(condition_independent))
    if n_trials < min_trials:
        columns = (
            "regressor plus the intercept" if condition_independent else "regressor"
        )
        raise ParameterError(
            f"n_trials must be at least {min_trials} (two observations per neuron, "
            f"and one trial per {columns}, so the design can have full column "
            f"rank); got {n_trials}"
        )
    max_rank = min(n_neurons, n_bins)
    for name, r in zip(names, rank_list, strict=True):
        if r > max_rank:
            raise ParameterError(
                f"rank of {name!r} is {r}, above min(n_neurons, n_bins) = {max_rank}"
            )
    X_given: FloatArray | None = None
    if X is None:
        level_sets = _level_sets(levels, n_regressors, condition_independent)
        sizes = [lv.size for lv in level_sets]
        n_conditions = math.prod(sizes)
        if n_conditions > _MAX_CONDITIONS:
            raise ParameterError(
                f"the levels define {n_conditions} conditions, more than a 64-bit "
                "index can address; use fewer levels or regressors"
            )
    else:
        if levels is not None:
            raise ParameterError(
                "levels must be None when X is given: the design is X itself"
            )
        X_given = _supplied_design(X, n_trials, names, condition_independent)
        level_sets = tuple(np.unique(X_given[:, p]) for p in range(n_regressors))
    W_given = _supplied_factors("W", W, names, rank_list, n_neurons, "n_neurons")
    S_given = _supplied_factors("S", S, names, rank_list, n_bins, "n_bins")
    scales, intercept_scale = _per_regressor(
        "length_scale",
        length_scale,
        n_regressors,
        condition_independent,
        _DEFAULT_LENGTH_SCALE,
        allow_zero=False,
    )
    amps, intercept_amp = _per_regressor(
        "amplitude",
        amplitude,
        n_regressors,
        condition_independent,
        _DEFAULT_AMPLITUDE,
        allow_zero=True,
    )
    precision = _noise_precision(noise_precision, n_neurons)
    drop_prob = _drop_prob(drop_prob)
    rng, int_seed = _generator(seed)

    # Overflow is possible only for amplitudes or precisions near the float
    # limits; it is detected below by the finiteness check, not by a warning.
    with np.errstate(over="ignore", invalid="ignore"):
        # Weights and bases, (M40)-(M41): per regressor, weight normals then
        # basis normals; the intercept last, as the reference's constant term.
        # A supplied W or S skips its draw.
        W_list: list[FloatArray] = []
        S_list: list[FloatArray] = []
        for r, ell, amp, w_p, s_p in zip(
            rank_list, scales, amps, W_given, S_given, strict=True
        ):
            if w_p is None:
                w_p = amp * rng.standard_normal((n_neurons, r))
            if s_p is None:
                basis_normals = rng.standard_normal((r, n_bins))
                s_p = _gp_bases(basis_normals, _gp_factor(_se_kernel(n_bins, ell)))
            W_list.append(w_p)
            S_list.append(s_p)
        B = [w @ s.T for w, s in zip(W_list, S_list, strict=True)]
        intercept: FloatArray | None = None
        if condition_independent:
            weight_normals = rng.standard_normal((n_neurons, n_bins))
            basis_normals = rng.standard_normal((n_bins, n_bins))
            factor = _gp_factor(_se_kernel(n_bins, intercept_scale))
            intercept = (intercept_amp * weight_normals) @ _gp_bases(
                basis_normals, factor
            ).T

        if precision is None:
            precision = rng.exponential(_DEFAULT_MEAN_PRECISION, n_neurons)

        # Conditions, (M43); a supplied design skips the draw.
        if X_given is None:
            condition_of_trial = rng.integers(0, n_conditions, n_trials).astype(np.intp)
            design = _condition_values(level_sets, condition_of_trial)
        else:
            design = X_given
            rows = np.unique(X_given, axis=0, return_inverse=True)[1]
            condition_of_trial = rows.reshape(-1).astype(np.intp)

        # Responses, (M44).
        noise_normals = rng.standard_normal((n_trials, n_neurons, n_bins))
        Y = _responses(design, B, intercept, precision, noise_normals)

    generated = [Y, *B] if intercept is None else [Y, intercept, *B]
    if not all(np.isfinite(a).all() for a in generated):
        raise ParameterError(
            "the simulated values are not finite: amplitude, noise_precision or "
            "the supplied X, W or S are too extreme for float64"
        )

    # Mask, (M45).
    mask = _draw_mask(rng, n_trials, n_neurons, drop_prob)

    arrays: list[NDArray[Any]] = [Y, design, mask, precision, condition_of_trial]
    arrays += W_list + S_list + B + list(level_sets)
    if intercept is not None:
        arrays.append(intercept)
    for a in arrays:
        a.setflags(write=False)

    return SimulatedData(
        Y=Y,
        X=design,
        mask=mask,
        W=_ReadOnlyMapping(dict(zip(names, W_list, strict=True))),
        S=_ReadOnlyMapping(dict(zip(names, S_list, strict=True))),
        B=_ReadOnlyMapping(dict(zip(names, B, strict=True))),
        intercept=intercept,
        noise_precision=precision,
        ranks=_ReadOnlyMapping(dict(zip(names, rank_list, strict=True))),
        regressor_names=names,
        levels=level_sets,
        condition_of_trial=condition_of_trial,
        seed=int_seed,
    )


# --------------------------------------------------------------------------- transforms


def _se_kernel(n_bins: int, length_scale: float) -> FloatArray:
    """Unit-variance squared-exponential kernel over bins, (M40).

    The reference's `toeplitz(exp(-((0:T-1)/len).^2/2))`. For a tiny length
    scale the squared lag overflows to `inf` and its exponential is exactly
    `0`, the correct limit (the identity kernel), so that overflow is silenced.
    """
    lag = np.arange(n_bins, dtype=np.float64)
    with np.errstate(over="ignore"):
        first_row = np.exp(-((lag / length_scale) ** 2) / 2)
    idx = np.abs(np.subtract.outer(np.arange(n_bins), np.arange(n_bins)))
    return first_row[idx]


def _gp_factor(kernel: FloatArray) -> FloatArray:
    """Square factor `F` with `F.T @ F == kernel`, as MATLAB `cholcov`.

    The upper Cholesky factor whenever Cholesky succeeds (the demo's case, so
    injected MATLAB normals reproduce `mvnrnd`). That includes kernels that are
    numerically singular but not detected as such. When Cholesky fails,
    the factor is `sqrt(D) @ U.T` from the eigendecomposition, with eigenvalues
    at or below `cholcov`'s tolerance `eps(max(D)) * n_bins` set to zero.
    Unlike `cholcov`, which then drops those rows, the factor stays square, so
    the number of normals drawn never depends on the kernel.
    """
    try:
        return np.linalg.cholesky(kernel).T
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(kernel)
        tol = np.spacing(eigvals.max()) * kernel.shape[0]
        eigvals = np.where(eigvals > tol, eigvals, 0.0)
        return np.sqrt(eigvals)[:, None] * eigvecs.T


def _gp_bases(normals: FloatArray, factor: FloatArray) -> FloatArray:
    """Map `(r, n_bins)` standard normals to `(n_bins, r)` GP bases, (M41).

    `mvnrnd(0, K, r)` is `normals @ factor`; the reference transposes it and
    flips the time axis (`flip(Sp)`, explicit `axis=0` here; `docs/model.md`
    § D.1).
    """
    return np.ascontiguousarray((normals @ factor).T[::-1])


def _condition_values(
    levels: Sequence[FloatArray], condition_of_trial: NDArray[np.intp]
) -> FloatArray:
    """Rows of the condition grid for the given indices, (M43), without the grid.

    Equal to `_condition_grid(levels)[condition_of_trial]`: the index is
    unravelled with the first regressor varying fastest (`order="F"`).
    """
    sizes = tuple(lv.size for lv in levels)
    level_idx = np.unravel_index(condition_of_trial, sizes, order="F")
    return np.stack(
        [lv[j] for lv, j in zip(levels, level_idx, strict=True)], axis=1
    ).astype(np.float64)


def _condition_grid(levels: Sequence[FloatArray]) -> FloatArray:
    """All level combinations, `(n_conditions, n_regressors)`, first varying fastest.

    The grid of (M43) in MATLAB `ndgrid` row order (`SimConditions`), for any
    number of regressors. Used by the tests only; `simulate` indexes it
    implicitly through `_condition_values`.
    """
    mesh = np.meshgrid(*levels, indexing="ij")
    return np.stack([m.reshape(-1, order="F") for m in mesh], axis=1)


def _responses(
    X: FloatArray,
    B: Sequence[FloatArray],
    intercept: FloatArray | None,
    precision: FloatArray,
    noise_normals: FloatArray,
) -> FloatArray:
    """Responses `(n_trials, n_neurons, n_bins)` from standard-normal noise, (M44).

    The noise is scaled by `1 / sqrt(precision)` per neuron, the reference's
    `diag(1./sqrt(d)) * randn(n, T)`.
    """
    Y: FloatArray = np.einsum("kp,pit->kit", X, np.stack(B))
    if intercept is not None:
        Y += intercept
    Y += noise_normals * (1.0 / np.sqrt(precision))[None, :, None]
    return Y


def _mask_is_valid(mask: NDArray[np.bool_]) -> bool:
    """Every trial has an observed neuron, every neuron `_MIN_OBSERVATIONS` trials."""
    return bool(
        mask.any(axis=1).all() and (mask.sum(axis=0) >= _MIN_OBSERVATIONS).all()
    )


def _draw_mask(
    rng: np.random.Generator, n_trials: int, n_neurons: int, drop_prob: float
) -> NDArray[np.bool_]:
    """Bernoulli(1 - drop_prob) mask, re-drawn until valid, (M45).

    A neuron-trial is observed when its uniform is at least `drop_prob`. MATLAB's
    `binornd(1, 1 - pdrop)` maps the same uniforms the other way (`u < 1 -
    pdrop`); the distributions agree, and draw-level mask parity is not a
    target.
    """
    for _ in range(_MAX_MASK_DRAWS):
        mask = rng.random((n_trials, n_neurons)) >= drop_prob
        if _mask_is_valid(mask):
            return mask
    raise ParameterError(
        f"no mask in {_MAX_MASK_DRAWS} draws had every trial observed and every "
        f"neuron observed on at least {_MIN_OBSERVATIONS} trials (drop_prob="
        f"{drop_prob}, n_trials={n_trials}, n_neurons={n_neurons}); lower "
        "drop_prob or raise n_trials or n_neurons"
    )


# --------------------------------------------------------------------------- validation
#
# Argument normalisation: NumPy scalars and 0-d arrays count as scalars, 1-D
# arrays as sequences; `bool` is never accepted as a number; every failure is a
# `ParameterError`. The shared predicates live in `mtdr._args`.


def _names_and_ranks(
    ranks: object, regressor_names: object
) -> tuple[tuple[str, ...], list[int]]:
    if isinstance(ranks, Mapping):
        if regressor_names is not None:
            raise ParameterError(
                "pass regressor names either as the keys of ranks or as "
                "regressor_names, not both"
            )
        names_seq: list[object] = list(ranks.keys())
        values: list[object] = list(ranks.values())
    else:
        values_or_none = _as_list(ranks)
        if values_or_none is None:
            raise ParameterError(
                "ranks must be a sequence of int, a 1-D integer array or a mapping "
                f"of str to int; got {ranks!r}"
            )
        values = values_or_none
        if regressor_names is None:
            names_seq = [f"x{p}" for p in range(len(values))]
        else:
            names_or_none = _as_list(regressor_names)
            if names_or_none is None:
                raise ParameterError(
                    "regressor_names must be a sequence of str; got "
                    f"{regressor_names!r}"
                )
            names_seq = names_or_none
            if len(names_seq) != len(values):
                raise ParameterError(
                    f"regressor_names has {len(names_seq)} entries but ranks has "
                    f"{len(values)}"
                )
    if not values:
        raise ParameterError("ranks must name at least one regressor")
    for name in names_seq:
        if not isinstance(name, str):
            raise ParameterError(f"regressor names must be str; got {name!r}")
        if name in _RESERVED_NAMES:
            raise ParameterError(f"regressor name {name!r} is reserved")
    names = tuple(str(name) for name in names_seq)
    if len(set(names)) != len(names):
        raise ParameterError(f"regressor names must be unique; got {names}")
    rank_list: list[int] = []
    for name, r in zip(names, values, strict=True):
        r_int = _as_int(r)
        if r_int is None or r_int < 0:
            raise ParameterError(
                f"rank of {name!r} must be a non-negative integer; got {r!r}"
            )
        rank_list.append(r_int)
    return names, rank_list


def _level_sets(
    levels: object,
    n_regressors: int,
    condition_independent: bool,
) -> tuple[FloatArray, ...]:
    if levels is None:
        defaults = _DEFAULT_LEVELS + (_EXTRA_LEVELS,) * n_regressors
        return tuple(np.array(lv, dtype=np.float64) for lv in defaults[:n_regressors])
    if isinstance(levels, np.ndarray):
        if levels.ndim != 2:
            raise ParameterError(
                "levels as an array must be 2-D, one row per regressor; got "
                f"{levels.ndim}-D"
            )
        level_list: list[object] = list(levels)
    elif isinstance(levels, str | bytes) or not isinstance(levels, Sequence):
        raise ParameterError(
            "levels must be a sequence of level sets, one per regressor; "
            f"got {levels!r}"
        )
    else:
        level_list = list(levels)
    if len(level_list) != n_regressors:
        raise ParameterError(
            f"levels has {len(level_list)} entries but there are {n_regressors} "
            "regressors"
        )
    out: list[FloatArray] = []
    for p, lv in enumerate(level_list):
        try:
            arr = np.asarray(lv, dtype=np.float64)
        except (TypeError, ValueError) as err:
            raise ParameterError(f"levels[{p}] is not numeric: {lv!r}") from err
        if arr.ndim != 1 or arr.size == 0:
            raise ParameterError(f"levels[{p}] must be a non-empty 1-D sequence")
        if not np.isfinite(arr).all():
            raise ParameterError(f"levels[{p}] must be finite")
        if np.unique(arr).size != arr.size:
            raise ParameterError(f"levels[{p}] has repeated values: {arr.tolist()}")
        if condition_independent and arr.size < 2:
            raise ParameterError(
                f"levels[{p}] has a single value, which makes a constant column "
                "collinear with the intercept; give two or more levels or pass "
                "condition_independent=False"
            )
        out.append(arr.copy())
    return tuple(out)


def _real_array(label: str, value: object, what: str) -> NDArray[Any]:
    """`value` as an array of a real (not bool) dtype, else a `ParameterError`."""
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as err:
        raise ParameterError(f"{label} must be {what}; got {value!r}") from err
    return arr


def _supplied_design(
    value: object, n_trials: int, names: tuple[str, ...], condition_independent: bool
) -> FloatArray:
    """Check a supplied design `X`; return a finite real `(n_trials, P)` copy."""
    what = "a 2-D real array of shape (n_trials, n_regressors)"
    arr = _real_array("X", value, what)
    if arr.ndim != 2:
        raise ParameterError(f"X must be {what}; got {arr.ndim}-D")
    if arr.dtype.kind not in "iuf":
        raise ParameterError(f"X must be real (not bool); got dtype {arr.dtype}")
    if arr.shape[0] != n_trials:
        raise ParameterError(
            f"X has {arr.shape[0]} rows but n_trials is {n_trials}; pass "
            f"n_trials={arr.shape[0]}"
        )
    if arr.shape[1] != len(names):
        raise ParameterError(
            f"X has {arr.shape[1]} columns but ranks has {len(names)} regressors"
        )
    out = arr.astype(np.float64, copy=True)
    if not np.isfinite(out).all():
        raise ParameterError("X must be finite")
    if condition_independent:
        constant = [names[p] for p in range(len(names)) if np.ptp(out[:, p]) == 0]
        if constant:
            raise ParameterError(
                f"column(s) {constant} of X are constant, which makes them collinear "
                "with the intercept; drop them or pass condition_independent=False"
            )
    return out


def _supplied_factors(
    label: str,
    value: object,
    names: tuple[str, ...],
    ranks: Sequence[int],
    n_rows: int,
    rows_name: str,
) -> list[FloatArray | None]:
    """Supplied `W` or `S`: per regressor a finite `(n_rows, r_p)` copy.

    `None` for a regressor that is drawn: every regressor when `value` is
    `None`, a regressor a mapping does not name, or a `None` entry.
    """
    if value is None:
        return [None] * len(names)
    if isinstance(value, Mapping):
        unknown = [key for key in value if key not in names]
        if unknown:
            raise ParameterError(
                f"{label} names {unknown}, which are not regressors; the regressors "
                f"are {list(names)}"
            )
        entries = [value.get(name) for name in names]
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        entries = list(value)
        if len(entries) != len(names):
            raise ParameterError(
                f"{label} has {len(entries)} entries but ranks has {len(names)} "
                "regressors"
            )
    else:
        raise ParameterError(
            f"{label} must be a sequence or a mapping of arrays, one per regressor; "
            f"got {value!r}"
        )
    out: list[FloatArray | None] = []
    for name, r, entry in zip(names, ranks, entries, strict=True):
        if entry is None:
            out.append(None)
            continue
        tag = f"{label}[{name!r}]"
        arr = _real_array(tag, entry, "a 1-D or 2-D real array")
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ParameterError(f"{tag} must be a 1-D or 2-D array; got {arr.ndim}-D")
        if arr.dtype.kind not in "iuf":
            raise ParameterError(
                f"{tag} must be real (not bool); got dtype {arr.dtype}"
            )
        if arr.shape != (n_rows, r):
            raise ParameterError(
                f"{tag} must have shape ({n_rows}, {r}): {rows_name} rows and the "
                f"regressor's rank as columns; got {arr.shape}"
            )
        copy = arr.astype(np.float64, copy=True)
        if not np.isfinite(copy).all():
            raise ParameterError(f"{tag} must be finite")
        out.append(copy)
    return out


def _per_regressor(
    name: str,
    value: object,
    n_regressors: int,
    condition_independent: bool,
    default: float,
    *,
    allow_zero: bool,
) -> tuple[list[float], float]:
    """Expand a scalar or per-regressor sequence; return (per regressor, intercept)."""
    scalar = _as_real(value)
    if scalar is not None:
        entries = [scalar] * n_regressors
        intercept = scalar
    else:
        entries_raw = _as_list(value)
        if entries_raw is None:
            raise ParameterError(
                f"{name} must be a float or a sequence of float; got {value!r}"
            )
        allowed = (
            {n_regressors, n_regressors + 1}
            if condition_independent
            else {n_regressors}
        )
        if len(entries_raw) not in allowed:
            extra = " (or one more, for the intercept)" if condition_independent else ""
            raise ParameterError(
                f"{name} has {len(entries_raw)} entries; expected {n_regressors}{extra}"
            )
        entries = []
        for v in entries_raw:
            as_real = _as_real(v)
            if as_real is None:
                raise ParameterError(f"{name} entries must be real numbers; got {v!r}")
            entries.append(as_real)
        intercept = entries.pop() if len(entries) > n_regressors else default
    for v in [*entries, intercept]:
        if not np.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
            bound = ">= 0" if allow_zero else "> 0"
            raise ParameterError(f"{name} entries must be finite and {bound}; got {v}")
    return entries, intercept


def _noise_precision(value: object, n_neurons: int) -> FloatArray | None:
    if value is None:
        return None
    if np.asarray(value).dtype == np.bool_:
        raise ParameterError(
            f"noise_precision must be numeric, not bool; got {value!r}"
        )
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise ParameterError(f"noise_precision is not numeric: {value!r}") from err
    if arr.ndim == 0:
        arr = np.full(n_neurons, float(arr))
    elif arr.shape != (n_neurons,):
        raise ParameterError(
            f"noise_precision must be a scalar or have shape ({n_neurons},); "
            f"got shape {arr.shape}"
        )
    else:
        arr = arr.copy()
    if not (np.isfinite(arr).all() and (arr > 0).all()):
        raise ParameterError("noise_precision must be finite and > 0")
    return arr


def _drop_prob(value: object) -> float:
    prob = _as_real(value)
    if prob is None or not 0 <= prob < 1:
        raise ParameterError(f"drop_prob must be a number in [0, 1); got {value!r}")
    return prob


def _generator(seed: object) -> tuple[np.random.Generator, int | None]:
    if isinstance(seed, np.random.Generator):
        return seed, None
    if seed is None:
        return np.random.default_rng(), None
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed), None
    int_seed = _as_int(seed)
    if int_seed is None or int_seed < 0:
        raise ParameterError(
            "seed must be a non-negative int, a numpy.random.Generator, a "
            f"numpy.random.SeedSequence or None; got {seed!r}"
        )
    return np.random.default_rng(int_seed), int_seed
