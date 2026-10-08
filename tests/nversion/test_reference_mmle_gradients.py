"""Finite-difference checks of the oracle's two gradients.

Central differences of `reference_mmle.nll` against `nll_grad_S` and
`nll_grad_lam`, on simulated data (no fixture needed) and, when
`tests/fixtures/mmle.mat` is present, at the first random point of each
fixture dataset. The measure is the max absolute difference relative to the
largest gradient entry; the gate is 1e-6. `pytest -s` prints the values.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from mtdr import simulate
from mtdr.stats import SufficientStats, sufficient_statistics
from nversion import mmle_fixture as mf
from nversion import reference_mmle as rm

FloatArray = NDArray[np.float64]
FD_TOL = 1e-6


def _simulated(
    n: int, T: int, N: int, ranks: Sequence[int], seed: int
) -> tuple[SufficientStats, list[FloatArray], FloatArray, FloatArray]:
    sim = simulate(
        n_neurons=n, n_bins=T, n_trials=N, ranks=list(ranks), drop_prob=0.3, seed=seed
    )
    stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    rng = np.random.default_rng(seed + 100)
    S = [0.5 * rng.standard_normal((T, r)) for r in ranks]
    lam = 0.5 + rng.random(n)
    b = stats.Y_mean + 0.2 * rng.standard_normal((n, T))
    return stats, S, lam, b


def _fd_s(
    f: Callable[[list[FloatArray]], float], S: Sequence[FloatArray]
) -> list[FloatArray]:
    out = []
    for p, Sp in enumerate(S):
        G = np.zeros_like(Sp)
        for idx in np.ndindex(Sp.shape):
            h = 1e-5 * max(1.0, abs(float(Sp[idx])))
            Sa = [x.copy() for x in S]
            Sb = [x.copy() for x in S]
            Sa[p][idx] += h
            Sb[p][idx] -= h
            G[idx] = (f(Sa) - f(Sb)) / (2 * h)
        out.append(G)
    return out


def _fd_lam(f: Callable[[FloatArray], float], lam: FloatArray) -> FloatArray:
    G = np.zeros_like(lam)
    for i in range(lam.size):
        h = 1e-6 * max(1.0, abs(float(lam[i])))
        la = lam.copy()
        lb = lam.copy()
        la[i] += h
        lb[i] -= h
        G[i] = (f(la) - f(lb)) / (2 * h)
    return G


def _rel_blocks(a: Sequence[FloatArray], b: Sequence[FloatArray]) -> float:
    scale = max(float(np.abs(x).max()) for x in b)
    return max(float(np.abs(x - y).max()) for x, y in zip(a, b, strict=True)) / scale


def _check_gradients(
    label: str,
    stats: SufficientStats,
    S: list[FloatArray],
    lam: FloatArray,
    b: FloatArray | None,
) -> None:
    _, grad_S = rm.nll_grad_S(S, lam, b, stats)
    fd_S = _fd_s(lambda Sx: rm.nll(Sx, lam, b, stats), S)
    err_S = _rel_blocks(fd_S, grad_S)
    _, grad_lam = rm.nll_grad_lam(S, lam, b, stats)
    fd_lam = _fd_lam(lambda lx: rm.nll(S, lx, b, stats), lam)
    err_lam = float(np.abs(fd_lam - grad_lam).max() / np.abs(grad_lam).max())
    print(
        f"\n[{label}] central-difference relative error: "
        f"dS {err_S:.1e}, dlam {err_lam:.1e}"
    )
    assert err_S < FD_TOL
    assert err_lam < FD_TOL


@pytest.mark.parametrize(
    ("n", "T", "N", "ranks", "seed"),
    [(6, 7, 30, (2, 1, 3), 0), (9, 5, 25, (1, 2), 1), (4, 6, 20, (3,), 2)],
)
def test_gradients_match_central_differences(
    n: int, T: int, N: int, ranks: tuple[int, ...], seed: int
) -> None:
    stats, S, lam, b = _simulated(n, T, N, ranks, seed)
    _check_gradients(f"sim n={n} T={T} r={ranks}", stats, S, lam, b)
    _check_gradients(f"sim n={n} T={T} r={ranks}, b=None", stats, S, lam, None)


def test_value_functions_agree_and_constant_is_the_only_offset() -> None:
    stats, S, lam, b = _simulated(6, 7, 30, (2, 1, 3), 3)
    full = rm.nll(S, lam, b, stats)
    ref = rm.nll(S, lam, b, stats, matlab_constants_only=True)
    const = 0.5 * stats.n_bins * stats.n_obs.sum() * np.log(2 * np.pi)
    assert full - ref == pytest.approx(const, rel=1e-12)
    for value in (
        rm.nll_grad_S(S, lam, b, stats)[0],
        rm.nll_grad_lam(S, lam, b, stats)[0],
    ):
        assert value == pytest.approx(full, rel=1e-12)


def test_ridge_gradient_is_that_of_the_nllonly_penalty() -> None:
    # Sonly.m:78 penalises s[n:], Sonly.m:99 adds g*S: the returned gradient
    # is that of nll(..., g) = nllonly's 0.5*g*||s||^2, and not that of the
    # Sonly value whenever n < rtot*T.
    stats, S, lam, b = _simulated(6, 7, 30, (2, 1, 3), 4)
    g = 0.7
    value, grad = rm.nll_grad_S(S, lam, b, stats, g=g)
    fd_nllonly = _fd_s(lambda Sx: rm.nll(Sx, lam, b, stats, g=g), S)
    fd_sonly = _fd_s(lambda Sx: rm.nll_grad_S(Sx, lam, b, stats, g=g)[0], S)
    err_nllonly = _rel_blocks(fd_nllonly, grad)
    err_sonly = _rel_blocks(fd_sonly, grad)
    print(
        f"\n[ridge g={g}] vs FD of nll: {err_nllonly:.1e}; "
        f"vs FD of the Sonly value: {err_sonly:.1e}"
    )
    assert err_nllonly < FD_TOL
    assert err_sonly > 1e-3
    s = mf.pack_s(S)
    n = stats.n_neurons
    expected_gap = 0.5 * g * float(s[:n] @ s[:n])
    assert rm.nll(S, lam, b, stats, g=g) - value == pytest.approx(
        expected_gap, rel=1e-10
    )


def test_lambda_gradient_separates_over_neurons() -> None:
    stats, S, lam, b = _simulated(6, 7, 30, (2, 1, 3), 5)
    _, g0 = rm.nll_grad_lam(S, lam, b, stats)
    lam2 = lam.copy()
    lam2[2] *= 1.7
    _, g1 = rm.nll_grad_lam(S, lam2, b, stats)
    changed = np.flatnonzero(np.abs(g1 - g0) > 1e-12 * np.abs(g0).max())
    assert changed.tolist() == [2]


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    if not mf.MMLE_FIXTURE.is_file():
        pytest.skip(
            "tests/fixtures/mmle.mat not available; regenerate it with "
            "tests/fixtures/matlab/make_mmle_fixture.m"
        )
    return mf.load()


@pytest.mark.parametrize("name", mf.DATASETS)
def test_gradients_at_fixture_point(fixture: dict[str, Any], name: str) -> None:
    ds = fixture[name]
    pt = ds.points[0]
    _check_gradients(f"{name} point 0", ds.stats, pt["S"], pt["lam"], pt["b"])
