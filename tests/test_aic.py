"""Tests for `mtdr.aic`: the parameter counts and the criterion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mtdr.aic import aic, n_parameters_mmle, n_parameters_svd
from mtdr.errors import ParameterError


def _jacobian_rank(f: Any, x0: np.ndarray, eps: float = 1e-6) -> int:
    """Numerical rank of the Jacobian of `f` at `x0` (central differences)."""
    cols = []
    for j in range(x0.size):
        e = np.zeros_like(x0)
        e[j] = eps
        cols.append((f(x0 + e) - f(x0 - e)) / (2 * eps))
    J = np.stack(cols, axis=1)
    s = np.linalg.svd(J, compute_uv=False)
    return int((s > 1e-6 * s[0]).sum())


@given(n=st.integers(1, 6), T=st.integers(1, 6), data=st.data())
def test_textbook_count_is_the_dimension_of_rank_r_matrices(
    n: int, T: int, data: st.DataObject
) -> None:
    # (M21b): r (n + T - r) is the dimension of the rank-r n x T matrices,
    # checked as the rank of the Jacobian of (W, S) -> W S^T at a random point.
    r = data.draw(st.integers(1, min(n, T)))
    rng = np.random.default_rng(n * 100 + T * 10 + r)
    x0 = rng.normal(size=n * r + T * r)

    def f(x: np.ndarray) -> Any:
        W, S = x[: n * r].reshape(n, r), x[n * r :].reshape(T, r)
        return (W @ S.T).ravel()

    assert _jacobian_rank(f, x0) == r * (n + T - r)
    # Without the intercept: the one block plus one precision per neuron.
    assert (
        n_parameters_svd([r], n, T, condition_independent=False) == r * (n + T - r) + n
    )


@given(T=st.integers(1, 7), data=st.data())
def test_identifiable_count_is_the_dimension_modulo_rotations(
    T: int, data: st.DataObject
) -> None:
    # (M38a): S_p carries T r - r (r - 1) / 2 parameters modulo S -> S Q, Q
    # orthogonal; S S^T is a complete invariant of that action for full
    # column rank S, so its Jacobian rank is that number.
    r = data.draw(st.integers(1, T))
    x0 = np.random.default_rng(T * 10 + r).normal(size=T * r)

    def f(x: np.ndarray) -> Any:
        S = x.reshape(T, r)
        return (S @ S.T).ravel()

    assert _jacobian_rank(f, x0) == T * r - r * (r - 1) // 2
    n = 7  # n >= T >= r
    assert (
        n_parameters_mmle([r], n, T, condition_independent=False)
        == n + T * r - r * (r - 1) // 2
    )


@given(
    ranks=st.lists(st.integers(0, 5), min_size=1, max_size=4),
    n=st.integers(5, 30),
    T=st.integers(5, 20),
    ci=st.booleans(),
)
def test_counts_match_their_formulas(
    ranks: list[int], n: int, T: int, ci: bool
) -> None:
    m = min(n, T)
    textbook = sum(r * (n + T - r) for r in ranks) + n + (n * T if ci else 0)
    assert n_parameters_svd(ranks, n, T, condition_independent=ci) == textbook
    # The intercept at full rank m counts m (n + T - m) = n T.
    assert m * (n + T - m) == n * T
    # SVDRegB_AIC.m line 15 with the constant term as a last regressor of rank m.
    r_full = [*ranks, m] if ci else ranks
    P, total = len(r_full), sum(r_full)
    assert (
        n_parameters_svd(ranks, n, T, condition_independent=ci, formula="reference")
        == (n * P + T * P - total) * total
    )
    ident = n + sum(T * r - r * (r - 1) // 2 for r in ranks) + (n * T if ci else 0)
    assert n_parameters_mmle(ranks, n, T, condition_independent=ci) == ident
    reference = n + T * sum(ranks) + (n * T if ci else 0)
    assert (
        n_parameters_mmle(ranks, n, T, condition_independent=ci, formula="reference")
        == reference
    )
    # The identifiable count is lower by sum r_p (r_p - 1) / 2 ...
    assert reference - ident == sum(r * (r - 1) // 2 for r in ranks)


@given(ranks=st.lists(st.integers(0, 9), min_size=1, max_size=3), p=st.integers(0, 2))
def test_penalty_per_rank_unit(ranks: list[int], p: int) -> None:
    # Raising r_p by one costs T - r_p identifiable parameters (2(T - r_p) in
    # AIC) against T for the reference count; the textbook SVD count rises by
    # n + T - 2 r_p - 1.
    n, T = 12, 10
    p = p % len(ranks)
    up = list(ranks)
    up[p] += 1
    r = ranks[p]
    assert n_parameters_mmle(up, n, T) - n_parameters_mmle(ranks, n, T) == T - r
    assert (
        n_parameters_mmle(up, n, T, formula="reference")
        - n_parameters_mmle(ranks, n, T, formula="reference")
        == T
    )
    assert (
        n_parameters_svd(up, n, T) - n_parameters_svd(ranks, n, T) == n + T - 2 * r - 1
    )


def test_reference_mmle_count_is_the_demo_parameter_length() -> None:
    # The demo's RankEstDemoMMLE.mat has parhist vectors of lengths 1750 to
    # 1810 for rhist[1:] = [3 4 3], [4 4 3], [4 5 3], [4 5 4], [4 6 4] at
    # n = 100, T = 15: (M38) is numel(pars) of BTDR_AIC_S_lamb_b_wrapper.
    rhist = [[3, 4, 3], [4, 4, 3], [4, 5, 3], [4, 5, 4], [4, 6, 4]]
    lengths = [n_parameters_mmle(r, 100, 15, formula="reference") for r in rhist]
    assert lengths == [1750, 1765, 1780, 1795, 1810]


def test_counts_accept_arrays_and_numpy_integers() -> None:
    # 1-D arrays are sequences, NumPy scalars and 0-d arrays are scalars.
    svd_args: Any = (np.array([2, 1]), np.int64(10), np.int32(5))
    mmle_args: Any = ((np.int8(1),), 4, np.array(3))
    assert n_parameters_svd(*svd_args) == 100
    assert n_parameters_mmle(*mmle_args) == 4 + 3 + 12
    assert isinstance(n_parameters_svd([1], 2, 2), int)


def test_aic_arithmetic() -> None:
    assert aic(-120.5, 10) == 261.0
    numpy_args: Any = (np.float64(3.0), np.int64(0))
    assert aic(*numpy_args) == -6.0
    zero_d: Any = np.array(1.5)
    assert aic(zero_d, 2) == 1.0
    assert isinstance(aic(0, 1), float)


BAD_COUNTS: list[tuple[dict[str, Any], str]] = [
    ({"ranks": [6], "n_neurons": 10, "n_bins": 5}, r"ranks\[0\] is 6, above min"),
    ({"ranks": [-1], "n_neurons": 10, "n_bins": 5}, "non-negative integer"),
    ({"ranks": [True], "n_neurons": 10, "n_bins": 5}, "non-negative integer"),
    ({"ranks": [1.5], "n_neurons": 10, "n_bins": 5}, "non-negative integer"),
    ({"ranks": [], "n_neurons": 10, "n_bins": 5}, "at least one entry"),
    ({"ranks": 2, "n_neurons": 10, "n_bins": 5}, "ranks must be a sequence"),
    (
        {"ranks": [1], "n_neurons": 0, "n_bins": 5},
        "n_neurons must be a positive integer",
    ),
    ({"ranks": [1], "n_neurons": True, "n_bins": 5}, "n_neurons"),
    (
        {"ranks": [1], "n_neurons": 3, "n_bins": 2.0},
        "n_bins must be a positive integer",
    ),
    (
        {"ranks": [1], "n_neurons": 3, "n_bins": 2, "condition_independent": "yes"},
        "condition_independent must be a bool",
    ),
]


@pytest.mark.parametrize("func", [n_parameters_svd, n_parameters_mmle])
@pytest.mark.parametrize(("kwargs", "match"), BAD_COUNTS)
def test_count_arguments_are_validated(
    func: Any, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ParameterError, match=match):
        func(**kwargs)


def test_unknown_formula() -> None:
    with pytest.raises(ParameterError, match="'textbook' or 'reference'"):
        n_parameters_svd([1], 3, 3, formula="identifiable")  # type: ignore[arg-type]
    with pytest.raises(ParameterError, match="'identifiable' or 'reference'"):
        n_parameters_mmle([1], 3, 3, formula="textbook")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("ll", "k", "match"),
    [
        (np.nan, 1, "log_likelihood must be a finite real number"),
        (np.inf, 1, "log_likelihood"),
        (True, 1, "log_likelihood"),
        ("1", 1, "log_likelihood"),
        (1.0, -1, "n_parameters must be a non-negative integer"),
        (1.0, 1.0, "n_parameters"),
        (1.0, False, "n_parameters"),
    ],
)
def test_aic_arguments_are_validated(ll: Any, k: Any, match: str) -> None:
    with pytest.raises(ParameterError, match=match):
        aic(ll, k)
