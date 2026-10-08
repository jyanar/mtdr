"""The fixture loader's layout translation.

Every block in the fixture has $r_p \\ne T$ ($T = 15$, $r_p \\le 3$), so a
transposed or mis-strided unpacking fails here: the loader's `S[p]` is
compared with the reference's own `mat2cell`/`MakeBhat_data` unpacking
(`r_p x T` blocks), and its `W[p]` with `MakeBhat_data`'s `What{p}`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nversion import mmle_fixture as mf


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    if not mf.MMLE_FIXTURE.is_file():
        pytest.skip(
            "tests/fixtures/mmle.mat not available; regenerate it with "
            "tests/fixtures/matlab/make_mmle_fixture.m"
        )
    return mf.load()


def test_unpack_pars_index_formulas() -> None:
    # Distinct entries, so every index is checked: S[p][t, j] is entry
    # n + T*(c[p] + j) + t and b[i, t] is entry n + T*rtot + T*i + t (0-based).
    n, T, ranks = 4, 5, [2, 1, 3]
    c = mf.offsets(ranks)
    L = n + T * c[-1]
    v = np.arange(L + n * T, dtype=np.float64)
    params = mf.unpack_pars(v, n, T, ranks)
    np.testing.assert_array_equal(params.lam, v[:n])
    for p, r in enumerate(ranks):
        assert params.S[p].shape == (T, r)
        for t in range(T):
            for j in range(r):
                assert params.S[p][t, j] == v[n + T * (c[p] + j) + t]
    for i in range(n):
        for t in range(T):
            assert params.b[i, t] == v[L + T * i + t]
    np.testing.assert_array_equal(mf.pack_pars(params), v)


def test_matrix_translations() -> None:
    n, rtot = 3, 4
    M = np.arange(rtot * rtot * n, dtype=np.float64).reshape(rtot, rtot, n, order="F")
    Ci = mf.ci_to_port(M, n, rtot)
    for i in range(n):
        np.testing.assert_array_equal(Ci[i], M[:, :, i])
    Wt = np.arange(rtot * n, dtype=np.float64).reshape(rtot, n, order="F")
    W = mf.unpack_wt(Wt, n, [3, 1])
    np.testing.assert_array_equal(W[0], Wt[:3, :].T)
    np.testing.assert_array_equal(W[1], Wt[3:, :].T)


@pytest.mark.parametrize("name", mf.DATASETS)
def test_bases_match_the_reference_unpacking(
    fixture: dict[str, Any], name: str
) -> None:
    ds = fixture[name]
    checked = 0
    for pt in ds.points:
        for Sp, Sm in zip(pt["S"], pt["S_matlab"], strict=True):
            assert Sp.shape[1] != ds.T
            np.testing.assert_array_equal(Sp, Sm.T)
            checked += 1
    for fit in ds.fits:
        for key in ("S_matlab", "S_matlab_makebhat"):
            for Sp, Sm in zip(fit["final"].S, fit[key], strict=True):
                np.testing.assert_array_equal(Sp, Sm.T)
                checked += 1
        np.testing.assert_array_equal(fit["final"].lam, fit["lam_makebhat"])
    assert checked > 0


@pytest.mark.parametrize("name", mf.DATASETS)
def test_weights_match_makebhat(fixture: dict[str, Any], name: str) -> None:
    ds = fixture[name]
    for fit in ds.fits:
        for p, (W, What) in enumerate(zip(fit["W"], fit["What"], strict=True)):
            np.testing.assert_array_equal(W, What)
            B = W @ fit["final"].S[p].T  # Bhat{p} = What{p} * Shat{p}
            scale = np.abs(fit["Bhat"][p]).max()
            assert np.abs(B - fit["Bhat"][p]).max() <= 1e-14 * scale


@pytest.mark.parametrize("name", mf.DATASETS)
def test_search_history_ranks(fixture: dict[str, Any], name: str) -> None:
    sr = fixture[name].search
    assert list(sr["rhist"][0]) == sr["rest0"]
    assert list(sr["rhist"][-1]) == sr["rest"]
    for k, params in enumerate(sr["parhist"]):
        assert params.ranks == list(sr["rhist"][k + 1])
    for r, params in zip(sr["calls_ranks"], sr["calls_params"], strict=True):
        assert params.ranks == list(r)
