"""Tests for `mtdr.simulation`: MATLAB parity, generative properties, arguments."""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray
from scipy.io import loadmat

import mtdr
from mtdr import simulation as sm
from mtdr.errors import ParameterError

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "simulation.mat"

# ------------------------------------------------------------------ MATLAB parity
#
# `tests/fixtures/matlab/make_simulation_fixture.m` runs the reference
# SimWeights / SimConditions / SimPopData and the demo's mask step at rng(0),
# replays the generator to recover every normal they consumed, asserts the
# replay reproduces the reference outputs bit for bit, and saves both. Here the
# MATLAB draws go through the port: through `simulate` itself, by a replaying
# generator (the binding test), and through each private transform (which
# localises a failure). Tests named `test_reference_*` call no port code: they
# document what the reference computes.

# Regressor order in the fixture is 1, 2, 3 and then the constant term ("0").
FIXTURE_BLOCKS = ("1", "2", "3", "0")


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    if not FIXTURE.is_file():  # pragma: no cover - the fixture is committed
        pytest.skip("tests/fixtures/simulation.mat not available")
    return loadmat(FIXTURE)


def _scalar(ref: dict[str, Any], key: str) -> int:
    return int(np.asarray(ref[key]).item())


def _ref_bases(ref: dict[str, Any]) -> list[tuple[float, float]]:
    """(length scale, weight scale) per fixture block, in FIXTURE_BLOCKS order."""
    lens = np.asarray(ref["len"]).ravel()
    rhos = np.asarray(ref["rho"]).ravel()
    return list(zip(lens.tolist(), rhos.tolist(), strict=True))


class ReplayRng:
    """Serves queued draws to `simulate` in place of a `numpy.random.Generator`.

    Each call must match the next queued draw's kind and shape, so the draw
    order of `docs/model.md` § D.4 is checked along with the values.
    """

    def __init__(self, queue: list[tuple[str, NDArray[Any]]]) -> None:
        self.queue = list(queue)
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def _pop(self, kind: str, shape: Any) -> NDArray[Any]:
        shape = tuple(np.atleast_1d(shape).tolist())
        assert self.queue, f"unexpected extra draw: {kind} {shape}"
        queued_kind, value = self.queue.pop(0)
        assert (queued_kind, value.shape) == (kind, shape), (queued_kind, kind, shape)
        self.calls.append((kind, shape))
        return value.copy()

    def standard_normal(self, size: Any) -> NDArray[Any]:
        return self._pop("standard_normal", size)

    def exponential(self, scale: float, size: Any) -> NDArray[Any]:
        assert scale == 1 / 0.8
        return self._pop("exponential", size)

    def integers(self, low: int, high: int, size: Any) -> NDArray[Any]:
        assert (low, high) == (0, 50)  # 5 x 5 x 2 conditions
        return self._pop("integers", size)

    def random(self, size: Any) -> NDArray[Any]:
        return self._pop("random", size)


def test_simulate_end_to_end_with_matlab_draws(
    ref: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    n = _scalar(ref, "n")
    queue: list[tuple[str, NDArray[Any]]] = []
    for b in FIXTURE_BLOCKS:
        queue += [("standard_normal", ref[f"G{b}"]), ("standard_normal", ref[f"Z{b}"])]
    hk = ref["hk"].astype(bool)
    queue += [
        ("exponential", np.asarray(ref["d"]).ravel()),
        ("integers", np.asarray(ref["cond_idx"]).ravel().astype(np.int64)),
        ("standard_normal", np.moveaxis(ref["noise"], 2, 0)),
        # The fixture exports the mask, not MATLAB's uniforms: serve uniforms
        # that the port maps to the same mask (observed iff u >= 0.3).
        ("random", np.where(hk, 0.99, 0.0)),
    ]
    rng = ReplayRng(queue)
    monkeypatch.setattr(sm, "_generator", lambda seed: (rng, None))

    sim = mtdr.simulate(
        n_neurons=n,
        n_bins=_scalar(ref, "T"),
        n_trials=_scalar(ref, "N"),
        ranks={"a": 2, "b": 1, "c": 3},
        levels=[np.arange(-2, 3), np.arange(-2, 3), [-1, 1]],
        # Public sequences, the last entry the intercept's: unequal per block,
        # so a mis-assigned entry fails.
        length_scale=np.asarray(ref["len"]).ravel().tolist(),
        amplitude=np.asarray(ref["rho"]).ravel().tolist(),
        drop_prob=0.3,
    )

    assert not rng.queue
    for name, b, p in zip("abc", FIXTURE_BLOCKS[:3], range(3), strict=True):
        np.testing.assert_array_equal(sim.W[name], ref[f"W{b}"])
        np.testing.assert_allclose(sim.S[name], ref[f"S{b}"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            sim.B[name], ref["BB"][p * n : (p + 1) * n], rtol=0, atol=1e-7
        )
    assert sim.intercept is not None
    np.testing.assert_allclose(sim.intercept, ref["BB"][3 * n :], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(sim.X, ref["X"][:, :3])
    np.testing.assert_array_equal(
        sim.condition_of_trial, np.asarray(ref["cond_idx"]).ravel()
    )
    np.testing.assert_array_equal(sim.noise_precision, np.asarray(ref["d"]).ravel())
    np.testing.assert_allclose(sim.Y, np.moveaxis(ref["Y"], 2, 0), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(sim.mask, hk)


def test_fixture_shapes_are_as_documented(ref: dict[str, Any]) -> None:
    n, T, N = (_scalar(ref, k) for k in ("n", "T", "N"))
    ranks = np.asarray(ref["ranks"]).ravel().tolist()
    assert (n, T, N) == (20, 15, 40)
    assert ranks == [2, 1, 3]
    # r_p != T in every task block, so a transposed S fails.
    for b, r in zip(FIXTURE_BLOCKS[:3], ranks, strict=True):
        assert ref[f"S{b}"].shape == (T, r)
        assert ref[f"W{b}"].shape == (n, r)
    assert ref["Y"].shape == (n, T, N)


@pytest.mark.parametrize("block", range(4))
def test_reference_weights_are_scaled_normals(ref: dict[str, Any], block: int) -> None:
    # (M40) in the reference: W_p = rho_p * G_p.
    b = FIXTURE_BLOCKS[block]
    _, rho = _ref_bases(ref)[block]
    np.testing.assert_array_equal(rho * ref[f"G{b}"], ref[f"W{b}"])


@pytest.mark.parametrize("block", range(4))
def test_kernel_and_factor_match_cholcov(ref: dict[str, Any], block: int) -> None:
    b = FIXTURE_BLOCKS[block]
    T = _scalar(ref, "T")
    ell, _ = _ref_bases(ref)[block]
    K = sm._se_kernel(T, ell)
    R = ref[f"R{b}"]
    F = sm._gp_factor(K)
    # Both are factors of the same kernel to machine precision ...
    np.testing.assert_allclose(R.T @ R, K, rtol=0, atol=1e-14)
    np.testing.assert_allclose(F.T @ F, K, rtol=0, atol=1e-14)
    # ... and both are the upper Cholesky factor, equal up to LAPACK rounding,
    # which scales with the kernel's condition number (up to 1e11 here). The
    # measured gap is 4 orders below this bound; the binding check is the
    # end-to-end basis comparison below.
    assert np.allclose(np.triu(F), F)
    bound = np.linalg.cond(K) * np.finfo(np.float64).eps
    assert np.abs(F - R).max() <= bound


@pytest.mark.parametrize("block", range(4))
def test_bases_match_matlab_given_its_factor(ref: dict[str, Any], block: int) -> None:
    b = FIXTURE_BLOCKS[block]
    # (M41): S_p = flip((Z_p R_p)^T); the transform alone, factor held fixed.
    S = sm._gp_bases(ref[f"Z{b}"], ref[f"R{b}"])
    np.testing.assert_allclose(S, ref[f"S{b}"], rtol=0, atol=1e-14)


@pytest.mark.parametrize("block", range(4))
def test_bases_match_matlab_end_to_end(ref: dict[str, Any], block: int) -> None:
    b = FIXTURE_BLOCKS[block]
    T = _scalar(ref, "T")
    ell, _ = _ref_bases(ref)[block]
    S = sm._gp_bases(ref[f"Z{b}"], sm._gp_factor(sm._se_kernel(T, ell)))
    # Only the factor's rounding (previous tests) separates the two: measured
    # 6.5e-10 at ell = 3, the worst block.
    np.testing.assert_allclose(S, ref[f"S{b}"], rtol=0, atol=1e-8)


def test_reference_bb_stacks_the_coefficients(ref: dict[str, Any]) -> None:
    # (M42) in the reference: BB = [W_1 S_1'; ...; W_P S_P'], constant term last.
    n = _scalar(ref, "n")
    for p, b in enumerate(FIXTURE_BLOCKS):
        B = ref[f"W{b}"] @ ref[f"S{b}"].T
        np.testing.assert_allclose(
            B, ref["BB"][p * n : (p + 1) * n], rtol=0, atol=1e-13
        )


def test_condition_grid_matches_ndgrid_order(ref: dict[str, Any]) -> None:
    levels = [np.arange(-2.0, 3.0), np.arange(-2.0, 3.0), np.array([-1.0, 1.0])]
    # The reference's constant regressor is a fourth level set {1}.
    grid = sm._condition_grid([*levels, np.array([1.0])])
    np.testing.assert_array_equal(grid, ref["Xcond"])
    # The documented index formula reproduces the reference's sampled rows.
    idx = np.asarray(ref["cond_idx"]).ravel().astype(np.intp)
    np.testing.assert_array_equal(grid[idx], ref["X"])
    np.testing.assert_array_equal(sm._condition_values(levels, idx), ref["X"][:, :3])


def test_responses_match_matlab(ref: dict[str, Any]) -> None:
    X = ref["X"][:, :3]  # the port carries the constant term as `intercept`
    B = [ref[f"W{b}"] @ ref[f"S{b}"].T for b in FIXTURE_BLOCKS[:3]]
    intercept = ref["W0"] @ ref["S0"].T
    precision = np.asarray(ref["d"]).ravel()
    noise = np.moveaxis(ref["noise"], 2, 0)  # (n, T, N) -> (N, n, T)
    Y = sm._responses(X, B, intercept, precision, noise)
    np.testing.assert_allclose(Y, np.moveaxis(ref["Y"], 2, 0), rtol=1e-12, atol=1e-12)


def test_reference_mask_zeroes_unobserved(ref: dict[str, Any]) -> None:
    # (M45) in the reference: Z = diag(h_k) Y_k. The port keeps Y whole and
    # marks unobserved entries with the boolean mask (`Y_masked` uses NaN).
    mask = ref["hk"].astype(bool)
    Y = np.moveaxis(ref["Y"], 2, 0)
    Z = np.moveaxis(ref["Z"], 2, 0)
    np.testing.assert_array_equal(np.where(mask[:, :, None], Y, 0.0), Z)


# --------------------------------------------------------------- generative model


def _sim(**kwargs: Any) -> mtdr.SimulatedData:
    kwargs.setdefault("seed", 0)
    return mtdr.simulate(**kwargs)


def test_defaults_follow_the_api() -> None:
    sim = mtdr.simulate(seed=0)
    assert sim.Y.shape == (100, 100, 15)
    assert sim.X.shape == (100, 3)
    assert dict(sim.ranks) == {"x0": 2, "x1": 1, "x2": 3}
    assert sim.regressor_names == ("x0", "x1", "x2")
    assert [lv.tolist() for lv in sim.levels] == [
        [-2, -1, 0, 1, 2],
        [-2, -1, 0, 1, 2],
        [-1, 1],
    ]
    assert sim.intercept is not None
    assert sim.mask.all()  # drop_prob defaults to 0
    assert sim.seed == 0


def test_default_levels_extend_with_plus_minus_one() -> None:
    sim = _sim(ranks=[1] * 5, n_neurons=10, n_bins=5, n_trials=20)
    assert [lv.tolist() for lv in sim.levels] == [
        [-2, -1, 0, 1, 2],
        [-2, -1, 0, 1, 2],
        [-1, 1],
        [-1, 1],
        [-1, 1],
    ]
    assert [lv.tolist() for lv in _sim(ranks=[1]).levels] == [[-2, -1, 0, 1, 2]]


@given(
    n_neurons=st.integers(1, 8),
    n_bins=st.integers(1, 8),
    n_regressors=st.integers(1, 4),
    data=st.data(),
    condition_independent=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_shapes_dtypes_and_ground_truth_identities(
    n_neurons: int,
    n_bins: int,
    n_regressors: int,
    data: st.DataObject,
    condition_independent: bool,
    seed: int,
) -> None:
    r_max = min(n_neurons, n_bins)
    ranks = data.draw(
        st.lists(st.integers(0, r_max), min_size=n_regressors, max_size=n_regressors)
    )
    min_trials = max(2, n_regressors + int(condition_independent))
    n_trials = data.draw(st.integers(min_trials, 12))
    sim = mtdr.simulate(
        n_neurons=n_neurons,
        n_bins=n_bins,
        n_trials=n_trials,
        ranks=ranks,
        condition_independent=condition_independent,
        seed=seed,
    )
    assert sim.Y.shape == (n_trials, n_neurons, n_bins)
    assert sim.X.shape == (n_trials, n_regressors)
    assert sim.mask.shape == (n_trials, n_neurons)
    assert sim.Y.dtype == sim.X.dtype == sim.noise_precision.dtype == np.float64
    assert sim.mask.dtype == np.bool_
    assert np.issubdtype(sim.condition_of_trial.dtype, np.integer)
    assert np.isfinite(sim.Y).all()
    assert (sim.noise_precision > 0).all()
    for name, r in zip(sim.regressor_names, ranks, strict=True):
        assert sim.W[name].shape == (n_neurons, r)
        assert sim.S[name].shape == (n_bins, r)
        np.testing.assert_allclose(sim.B[name], sim.W[name] @ sim.S[name].T)
    if condition_independent:
        assert sim.intercept is not None
        assert sim.intercept.shape == (n_neurons, n_bins)
    else:
        assert sim.intercept is None
    # Every X row is a level combination, and condition_of_trial indexes it.
    grid = sm._condition_grid(sim.levels)
    np.testing.assert_array_equal(grid[sim.condition_of_trial], sim.X)
    # The mask's promise (every trial observes a neuron, every neuron enough
    # trials) holds whenever simulate returns.
    assert sm._mask_is_valid(sim.mask)


@given(
    sizes=st.lists(st.integers(1, 4), min_size=1, max_size=4),
    data=st.data(),
)
def test_condition_values_equal_grid_rows(
    sizes: list[int], data: st.DataObject
) -> None:
    levels = [np.arange(s, dtype=np.float64) * 10**p for p, s in enumerate(sizes)]
    n_conditions = int(np.prod(sizes))
    idx = np.array(
        data.draw(st.lists(st.integers(0, n_conditions - 1), min_size=1, max_size=20)),
        dtype=np.intp,
    )
    np.testing.assert_array_equal(
        sm._condition_values(levels, idx), sm._condition_grid(levels)[idx]
    )


def test_condition_index_formula_in_docstring() -> None:
    sim = _sim(ranks=[1, 1, 1], n_trials=200)
    sizes = [lv.size for lv in sim.levels]
    level_idx = np.stack(
        [np.searchsorted(lv, sim.X[:, p]) for p, lv in enumerate(sim.levels)], axis=1
    )
    expected = level_idx[:, 0] + sizes[0] * (
        level_idx[:, 1] + sizes[1] * level_idx[:, 2]
    )
    np.testing.assert_array_equal(sim.condition_of_trial, expected)


def test_many_continuous_regressors_do_not_build_the_grid() -> None:
    # 101^4 ~ 1e8 conditions: about 10 GB as a materialised grid.
    grid = np.linspace(-1, 1, 101)
    sim = _sim(ranks=[1] * 4, levels=[grid] * 4, n_neurons=5, n_bins=3)
    assert np.isin(sim.X, grid).all()
    assert sim.condition_of_trial.max() < 101**4


def test_true_ranks_are_attained() -> None:
    sim = _sim(ranks=[2, 1, 3])
    for name, r in sim.ranks.items():
        assert np.linalg.matrix_rank(sim.B[name]) == r


def test_ranks_are_generating_widths_at_long_length_scales() -> None:
    # A numerically singular kernel caps the attained rank.
    sim = _sim(ranks=[3], length_scale=1e6, n_neurons=20, n_bins=15)
    assert sim.ranks["x0"] == 3
    assert sim.S["x0"].shape == (15, 3)
    assert np.linalg.matrix_rank(sim.B["x0"]) < 3
    assert np.isfinite(sim.Y).all()


def test_rank_zero_regressor_contributes_nothing() -> None:
    sim = _sim(ranks={"a": 0, "b": 2}, n_neurons=10, n_bins=6, n_trials=30)
    assert sim.W["a"].shape == (10, 0)
    assert sim.S["a"].shape == (6, 0)
    np.testing.assert_array_equal(sim.B["a"], 0.0)


@pytest.mark.parametrize("condition_independent", [True, False])
def test_residuals_have_the_stated_noise_precision(condition_independent: bool) -> None:
    sim = _sim(
        n_neurons=20,
        n_bins=10,
        n_trials=2000,
        condition_independent=condition_independent,
    )
    mean = np.einsum("kp,pit->kit", sim.X, np.stack(list(sim.B.values())))
    if sim.intercept is not None:
        mean = mean + sim.intercept
    z = (sim.Y - mean) * np.sqrt(sim.noise_precision)[None, :, None]
    # 20,000 standard normals per neuron: mean ~ 0.007 s.e., var ~ 0.01 s.e.
    assert np.abs(z.mean(axis=(0, 2))).max() < 0.05
    assert np.abs(z.var(axis=(0, 2)) - 1).max() < 0.06


def test_default_precision_is_exponential_with_mean_one_point_two_five() -> None:
    sim = _sim(n_neurons=20000, n_bins=1, n_trials=2, ranks=[1])
    assert abs(sim.noise_precision.mean() - 1.25) < 0.03
    # Exponential: the standard deviation equals the mean.
    assert abs(sim.noise_precision.std() - 1.25) < 0.05


def test_gp_bases_have_the_squared_exponential_covariance() -> None:
    rng = np.random.default_rng(1)
    K = sm._se_kernel(12, 2.0)
    S = sm._gp_bases(rng.standard_normal((40000, 12)), sm._gp_factor(K))
    assert S.shape == (12, 40000)
    np.testing.assert_allclose(np.cov(S), K, atol=0.03)


def test_se_kernel_values() -> None:
    K = sm._se_kernel(4, 2.0)
    expected = np.exp(-(np.subtract.outer(np.arange(4), np.arange(4)) ** 2) / 8)
    np.testing.assert_allclose(K, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(sm._se_kernel(1, 3.0), [[1.0]])


def test_tiny_length_scale_gives_the_identity_kernel_silently() -> None:
    # The squared lag overflows; the limit is exact (filterwarnings = error).
    np.testing.assert_array_equal(sm._se_kernel(4, 1e-300), np.eye(4))
    assert np.isfinite(_sim(length_scale=1e-300, n_neurons=5, n_bins=4).Y).all()


def test_gp_factor_falls_back_on_a_singular_kernel() -> None:
    K = np.ones((5, 5))  # rank 1: Cholesky fails
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(K)
    F = sm._gp_factor(K)
    assert F.shape == (5, 5)  # square, so the number of normals drawn is fixed
    np.testing.assert_allclose(F.T @ F, K, atol=1e-14)


def test_long_length_scales_simulate_finite_data() -> None:
    # SE kernels with long length scales are numerically singular.
    sim = _sim(n_bins=40, length_scale=20.0, ranks=[2, 2])
    assert np.isfinite(sim.Y).all()
    for S in sim.S.values():
        assert np.isfinite(S).all()


def test_flip_reverses_time() -> None:
    normals = np.arange(6.0).reshape(2, 3)
    S = sm._gp_bases(normals, np.eye(3))
    np.testing.assert_array_equal(S, normals.T[::-1])


def test_conditions_are_sampled_uniformly() -> None:
    sim = _sim(ranks=[1, 1], n_neurons=4, n_bins=2, n_trials=25000)
    counts = np.bincount(sim.condition_of_trial, minlength=25)
    # 1000 expected per condition, s.d. ~31.
    assert counts.min() > 850
    assert counts.max() < 1150


def test_continuous_levels() -> None:
    grid = np.linspace(-1, 1, 101)
    sim = _sim(ranks=[1], levels=[grid], n_trials=500)
    assert np.isin(sim.X[:, 0], grid).all()
    assert np.unique(sim.X[:, 0]).size > 50


# ----------------------------------------------------------- per-regressor scales


def test_amplitude_scales_weights_and_intercept_entry() -> None:
    a = _sim(amplitude=1.0)
    b = _sim(amplitude=[2.0, 2.0, 2.0, 3.0])
    for name in a.regressor_names:
        np.testing.assert_allclose(b.W[name], 2.0 * a.W[name])
    assert a.intercept is not None
    assert b.intercept is not None
    np.testing.assert_allclose(b.intercept, 3.0 * a.intercept)


def test_amplitude_without_intercept_entry_uses_the_default() -> None:
    a = _sim()
    b = _sim(amplitude=[0.0, 0.0, 0.0])
    for name in b.regressor_names:
        np.testing.assert_array_equal(b.B[name], 0.0)
    assert a.intercept is not None
    assert b.intercept is not None
    np.testing.assert_array_equal(b.intercept, a.intercept)


def test_length_scale_intercept_entry_changes_only_the_intercept() -> None:
    a = _sim(length_scale=[2.0, 2.0, 2.0])
    b = _sim(length_scale=[2.0, 2.0, 2.0, 5.0])
    for name in a.regressor_names:
        np.testing.assert_array_equal(a.B[name], b.B[name])
    assert a.intercept is not None
    assert b.intercept is not None
    assert not np.allclose(a.intercept, b.intercept)


def test_noise_precision_scalar_and_array() -> None:
    sim = _sim(n_neurons=5, noise_precision=4.0, n_bins=3, ranks=[1])
    np.testing.assert_array_equal(sim.noise_precision, np.full(5, 4.0))
    prec = np.arange(1.0, 6.0)
    sim = _sim(n_neurons=5, noise_precision=prec, n_bins=3, ranks=[1])
    prec[0] = 100.0  # the simulation keeps its own copy
    np.testing.assert_array_equal(sim.noise_precision, np.arange(1.0, 6.0))
    sim = _sim(n_neurons=3, noise_precision=[1.0, 2.0, 3.0], n_bins=3, ranks=[1])
    np.testing.assert_array_equal(sim.noise_precision, [1.0, 2.0, 3.0])


# --------------------------------------------------------- array-valued arguments


def test_numpy_scalars_and_zero_d_arrays_are_scalars() -> None:
    a = _sim(n_neurons=6, n_bins=4, ranks=[1], length_scale=2.0, amplitude=1.5)
    b = _sim(
        n_neurons=np.int64(6),
        n_bins=np.array(4),
        ranks=[np.int32(1)],
        length_scale=np.array(2.0),
        amplitude=np.float32(1.5),
        drop_prob=np.array(0.0),
        seed=np.array(0),
    )
    np.testing.assert_array_equal(a.Y, b.Y)


def test_array_ranks_levels_and_names() -> None:
    a = _sim(ranks=[2, 1, 3], levels=[[-1, 1], [-1, 1], [-1, 1]])
    b = _sim(
        ranks=np.array([2, 1, 3]),
        levels=np.array([[-1.0, 1.0]] * 3),
        regressor_names=np.array(["x0", "x1", "x2"]),
    )
    np.testing.assert_array_equal(a.Y, b.Y)
    assert b.regressor_names == ("x0", "x1", "x2")
    assert all(type(name) is str for name in b.regressor_names)


def test_array_length_scale_and_amplitude() -> None:
    a = _sim(length_scale=[2.0, 3.0, 1.5, 2.5], amplitude=[1.0, 0.5, 2.0])
    b = _sim(
        length_scale=np.array([2.0, 3.0, 1.5, 2.5]), amplitude=np.array([1, 0.5, 2])
    )
    np.testing.assert_array_equal(a.Y, b.Y)


def test_numpy_bool_condition_independent() -> None:
    assert _sim(condition_independent=np.bool_(False)).intercept is None
    assert _sim(condition_independent=np.bool_(True)).intercept is not None


# --------------------------------------------------------------------------- mask


def test_drop_prob_zero_observes_everything() -> None:
    assert _sim(drop_prob=0.0).mask.all()


def test_drop_prob_sets_the_observed_fraction() -> None:
    sim = _sim(drop_prob=0.3, n_neurons=200, n_trials=500)
    assert abs(sim.mask.mean() - 0.7) < 0.01


@pytest.mark.parametrize("seed", range(20))
def test_mask_validity_promise(seed: int) -> None:
    # 3 neurons x 4 trials at drop 0.5: the first draw is invalid 74 % of the
    # time, and 100 invalid draws have probability 4.9e-14, so the re-draw is
    # exercised and the seed choice cannot matter.
    sim = mtdr.simulate(
        n_neurons=3, n_bins=2, n_trials=4, ranks=[1], drop_prob=0.5, seed=seed
    )
    assert sim.mask.any(axis=1).all()
    assert (sim.mask.sum(axis=0) >= 2).all()


class _MaskRng:
    """Serves fixed uniform arrays to `_draw_mask`, counting the calls."""

    def __init__(self, draws: list[NDArray[np.float64]]) -> None:
        self.draws = draws
        self.calls = 0

    def random(self, size: tuple[int, int]) -> NDArray[np.float64]:
        self.calls += 1
        draw = self.draws[min(self.calls, len(self.draws)) - 1]
        assert draw.shape == size
        return draw


def test_draw_mask_redraws_until_valid() -> None:
    invalid = np.zeros((3, 4))  # every uniform < drop_prob: nothing observed
    valid = np.ones((3, 4))
    rng = _MaskRng([invalid, invalid, valid])
    mask = sm._draw_mask(rng, 3, 4, 0.5)  # type: ignore[arg-type]
    assert rng.calls == 3
    assert mask.all()


def test_draw_mask_gives_up_after_100_draws() -> None:
    rng = _MaskRng([np.zeros((3, 4))])
    with pytest.raises(ParameterError, match="100 draws"):
        sm._draw_mask(rng, 3, 4, 0.5)  # type: ignore[arg-type]
    assert rng.calls == 100


def test_unsatisfiable_mask_raises() -> None:
    with pytest.raises(ParameterError, match="100 draws"):
        _sim(n_neurons=5, n_trials=2, n_bins=2, ranks=[1], drop_prob=0.99)


# -------------------------------------------------------------------------- seeds


def test_integer_seed_is_reproducible() -> None:
    a, b = _sim(seed=7, drop_prob=0.3), _sim(seed=7, drop_prob=0.3)
    for field in ("Y", "X", "mask", "noise_precision", "condition_of_trial"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert not np.array_equal(a.Y, _sim(seed=8).Y)


def test_numpy_integer_seed_is_recorded_as_int() -> None:
    sim = _sim(seed=np.int64(3))
    assert sim.seed == 3
    assert type(sim.seed) is int


def test_generator_is_advanced_and_seed_is_none() -> None:
    rng = np.random.default_rng(0)
    a = mtdr.simulate(seed=rng)
    b = mtdr.simulate(seed=rng)
    assert a.seed is None
    assert not np.array_equal(a.Y, b.Y)
    np.testing.assert_array_equal(a.Y, mtdr.simulate(seed=np.random.default_rng(0)).Y)


def test_seed_sequence_is_accepted() -> None:
    a = mtdr.simulate(seed=np.random.SeedSequence(5))
    b = mtdr.simulate(seed=np.random.SeedSequence(5))
    assert a.seed is None
    np.testing.assert_array_equal(a.Y, b.Y)


def test_no_seed_draws_fresh_data() -> None:
    assert mtdr.simulate(seed=None).seed is None


def test_fixed_precision_skips_its_draw() -> None:
    # Draw order: the precisions are drawn only when noise_precision is None,
    # so fixing them shifts every later draw.
    a = _sim(n_neurons=5, n_bins=3, ranks=[1])
    b = _sim(n_neurons=5, n_bins=3, ranks=[1], noise_precision=a.noise_precision)
    np.testing.assert_array_equal(a.B["x0"], b.B["x0"])
    assert not np.array_equal(a.condition_of_trial, b.condition_of_trial)


# ------------------------------------------------------------------ result object


def test_names_from_mapping_and_sequence() -> None:
    sim = _sim(ranks={"sa": 2, "sb": 1, "choice": 3})
    assert sim.regressor_names == ("sa", "sb", "choice")
    assert list(sim.W) == list(sim.S) == list(sim.B) == ["sa", "sb", "choice"]
    sim = _sim(ranks=[1, 2], regressor_names=["a", "b"])
    assert dict(sim.ranks) == {"a": 1, "b": 2}


def test_y_masked_and_fit_kwargs() -> None:
    sim = _sim(drop_prob=0.3)
    Ym = sim.Y_masked
    assert np.isnan(Ym[~sim.mask]).all()
    np.testing.assert_array_equal(Ym[sim.mask], sim.Y[sim.mask])
    Ym[0, 0, 0] = 1.0  # a fresh, writable array
    assert sim.fit_kwargs() == {
        "mask": sim.mask,
        "regressor_names": sim.regressor_names,
    }


def test_ground_truth_is_read_only() -> None:
    sim = _sim()
    arrays: list[NDArray[Any]] = [
        sim.Y,
        sim.X,
        sim.mask,
        sim.noise_precision,
        sim.condition_of_trial,
        *sim.W.values(),
        *sim.S.values(),
        *sim.B.values(),
        *sim.levels,
    ]
    assert sim.intercept is not None
    arrays.append(sim.intercept)
    for a in arrays:
        with pytest.raises(ValueError, match="read-only"):
            a[(0,) * a.ndim] = 0.0
    for mapping in (sim.W, sim.S, sim.B, sim.ranks):
        with pytest.raises(TypeError):
            mapping["x0"] = 0  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        sim.Y = sim.Y  # type: ignore[misc]


def test_simulated_data_pickles() -> None:
    sim = _sim(n_neurons=5, n_bins=3, ranks={"a": 1, "b": 2}, drop_prob=0.2)
    restored = pickle.loads(pickle.dumps(sim))
    assert dict(restored.ranks) == {"a": 1, "b": 2}
    np.testing.assert_array_equal(restored.Y, sim.Y)
    np.testing.assert_array_equal(restored.W["b"], sim.W["b"])
    assert repr(restored) == repr(sim)
    # Unpickling restores read-only arrays, in the mappings too.
    for array in (restored.Y, restored.X, restored.mask, restored.W["b"]):
        assert not array.flags.writeable


def test_repr_summarises() -> None:
    text = repr(_sim(ranks={"sa": 2}, n_trials=10, n_neurons=4, n_bins=3))
    assert text == (
        "SimulatedData(n_trials=10, n_neurons=4, n_bins=3, ranks={'sa': 2}, "
        "intercept=True, observed=1.000, seed=0)"
    )


def test_singleton_shapes() -> None:
    sim = _sim(n_neurons=1, n_bins=5, n_trials=2, ranks=[1])
    assert sim.Y.shape == (2, 1, 5)
    sim = _sim(n_neurons=6, n_bins=1, n_trials=3, ranks=[1, 0])
    assert sim.S["x0"].shape == (1, 1)


def test_top_level_exports() -> None:
    assert mtdr.simulate is sm.simulate
    assert mtdr.SimulatedData is sm.SimulatedData
    assert set(sm.__all__) <= set(mtdr.__all__)


# ------------------------------------------------------- a supplied design, W and S
#
# simulate(X=..., S=..., W=...) uses a given design and, optionally, given
# temporal bases and weights; omitted pieces are drawn as by default, in the
# same order, with the supplied pieces' draws skipped.


def _design(n_trials: int = 60, seed: int = 9) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    sa = rng.integers(0, 4, n_trials) - 1.5
    sb = sa + rng.choice([-1.0, 1.0], n_trials)
    return np.column_stack([sa, sb, np.where(sa > sb, 1.0, -1.0)])


def test_supplied_pieces_reproduce_a_default_draw() -> None:
    # Feeding a default simulation's X, W, S and precisions back gives the same
    # coefficients and the same responses up to the noise draw.
    a = _sim(n_neurons=12, n_bins=6, n_trials=50, ranks={"u": 2, "v": 1})
    b = _sim(
        n_neurons=12,
        n_bins=6,
        n_trials=50,
        ranks={"u": 2, "v": 1},
        X=a.X,
        W=a.W,
        S=dict(a.S),
        noise_precision=a.noise_precision,
        seed=1,
    )
    np.testing.assert_array_equal(b.X, a.X)
    for name in ("u", "v"):
        np.testing.assert_array_equal(b.W[name], a.W[name])
        np.testing.assert_array_equal(b.S[name], a.S[name])
        np.testing.assert_array_equal(b.B[name], a.B[name])
    assert b.intercept is not None
    signal = (
        b.Y - b.intercept - np.einsum("kp,pit->kit", b.X, np.stack(list(b.B.values())))
    )
    resid = signal * np.sqrt(b.noise_precision)[None, :, None]
    assert abs(resid.std() - 1.0) < 0.05


def test_supplied_design_skips_the_condition_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `docs/model.md` § D.4's draw order with the supplied pieces' draws removed.
    X = _design()
    rng = np.random.default_rng(0)
    W = {"sa": rng.normal(size=(5, 2)), "sb": rng.normal(size=(5, 1))}
    S = {"choice": rng.normal(size=(4, 1))}
    queue: list[tuple[str, NDArray[Any]]] = [
        ("standard_normal", np.ones((2, 4))),  # sa: basis normals only (W given)
        ("standard_normal", np.ones((1, 4))),  # sb: basis normals only
        ("standard_normal", np.ones((5, 1))),  # choice: weight normals only (S given)
        ("standard_normal", np.ones((5, 4))),  # intercept weights
        ("standard_normal", np.ones((4, 4))),  # intercept bases
        ("exponential", np.ones(5)),  # precisions; no condition draw follows
        ("standard_normal", np.zeros((60, 5, 4))),  # noise
        ("random", np.ones((60, 5))),  # mask uniforms
    ]
    replay = ReplayRng(queue)
    monkeypatch.setattr(sm, "_generator", lambda seed: (replay, None))
    out = mtdr.simulate(
        n_neurons=5,
        n_bins=4,
        n_trials=60,
        ranks={"sa": 2, "sb": 1, "choice": 1},
        X=X,
        W=W,
        S=S,
    )
    assert not replay.queue
    np.testing.assert_array_equal(out.W["sa"], W["sa"])
    np.testing.assert_array_equal(out.S["choice"], S["choice"])
    np.testing.assert_array_equal(out.B["sb"], W["sb"] @ out.S["sb"].T)
    assert out.intercept is not None
    np.testing.assert_allclose(
        out.Y,
        out.intercept + np.einsum("kp,pit->kit", X, np.stack(list(out.B.values()))),
    )


def test_supplied_design_levels_and_condition_index() -> None:
    X = _design()
    sim = _sim(n_neurons=8, n_bins=5, n_trials=60, ranks=[1, 1, 1], X=X)
    np.testing.assert_array_equal(sim.X, X)
    assert sim.X is not X
    assert not sim.X.flags.writeable
    for p in range(3):
        np.testing.assert_array_equal(sim.levels[p], np.unique(X[:, p]))
    rows = np.unique(X, axis=0)
    np.testing.assert_array_equal(rows[sim.condition_of_trial], X)
    assert sim.condition_of_trial.dtype == np.intp


def test_supplied_one_dimensional_factors_are_one_column() -> None:
    rng = np.random.default_rng(1)
    w, s = rng.normal(size=7), rng.normal(size=4)
    sim = _sim(
        n_neurons=7, n_bins=4, n_trials=20, ranks=[1], W=[w], S=[s], amplitude=5.0
    )
    np.testing.assert_array_equal(sim.W["x0"], w[:, None])
    np.testing.assert_array_equal(sim.B["x0"], np.outer(w, s))


def test_supplied_rank_zero_and_partial_supply() -> None:
    sim = _sim(
        n_neurons=6,
        n_bins=5,
        n_trials=30,
        ranks=[1, 0],
        W=[np.ones((6, 1)), np.zeros((6, 0))],
    )
    assert sim.S["x1"].shape == (5, 0)
    np.testing.assert_array_equal(sim.B["x1"], 0.0)
    # S alone: W is drawn with the amplitude.
    s_only = _sim(n_neurons=6, n_bins=5, n_trials=30, ranks=[1], S=[np.ones(5)])
    np.testing.assert_array_equal(s_only.S["x0"], np.ones((5, 1)))
    assert s_only.W["x0"].shape == (6, 1)
    # A None entry, or a regressor a mapping does not name, is drawn: the same
    # draws as with nothing supplied for it.
    a = _sim(n_neurons=6, n_bins=5, n_trials=30, ranks=[1, 2])
    b = _sim(n_neurons=6, n_bins=5, n_trials=30, ranks=[1, 2], S=[None, a.S["x1"]])
    c = _sim(n_neurons=6, n_bins=5, n_trials=30, ranks=[1, 2], S={"x1": a.S["x1"]})
    for other in (b, c):
        np.testing.assert_array_equal(other.W["x0"], a.W["x0"])
        np.testing.assert_array_equal(other.S["x0"], a.S["x0"])


def test_supplied_design_without_intercept_allows_a_constant_column() -> None:
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    sim = _sim(
        n_neurons=4,
        n_bins=3,
        n_trials=10,
        ranks=[1, 1],
        X=X,
        condition_independent=False,
    )
    assert sim.intercept is None
    np.testing.assert_array_equal(sim.X, X)


SUPPLIED_ERRORS: list[tuple[dict[str, Any], str]] = [
    ({"X": np.ones((60, 2))}, "X has 2 columns"),
    ({"X": np.ones((50, 3))}, "X has 50 rows"),
    ({"X": np.ones(60)}, "X must be a 2-D"),
    ({"X": np.full((60, 3), np.nan)}, "X must be finite"),
    ({"X": np.ones((60, 3), dtype=bool)}, "X must be real"),
    ({"X": np.array([["a"] * 3] * 60)}, "X must be real"),
    ({"X": "X"}, "X must be a 2-D"),
    ({"X": [[1.0, 2.0], [3.0]]}, "X must be a 2-D"),
    ({"S": [[[1.0, 2.0], [3.0]], np.ones((4, 1)), np.ones((4, 1))]}, "S\\['sa'\\]"),
    ({"X": np.column_stack([np.ones(60), _design()[:, 1:]])}, "constant"),
    ({"X": _design(), "levels": [[0, 1]] * 3}, "levels must be None"),
    ({"W": [np.ones((5, 2))] * 2}, "W has 2 entries"),
    ({"W": {"sa": np.ones((5, 2)), "sc": np.ones((5, 1))}}, "W names \\['sc'\\]"),
    ({"W": {"sa": 1, "sb": 1, "choice": 1}}, "W\\['sa'\\] must be"),
    ({"W": 3}, "W must be a sequence or a mapping"),
    ({"W": "abc"}, "W must be a sequence or a mapping"),
    ({"W": [np.ones((4, 2)), np.ones((5, 1)), np.ones((5, 1))]}, "shape \\(5, 2\\)"),
    ({"W": [np.ones((5, 1)), np.ones((5, 1)), np.ones((5, 1))]}, "shape \\(5, 2\\)"),
    ({"S": [np.ones((4, 2)), np.ones(3), np.ones((4, 1))]}, "shape \\(4, 1\\)"),
    ({"S": [np.ones((4, 2)), np.ones((4, 1)), np.full((4, 1), np.inf)]}, "finite"),
    ({"S": [np.ones((4, 2, 1)), np.ones((4, 1)), np.ones((4, 1))]}, "1-D or 2-D"),
    ({"S": [np.ones((4, 2)), np.ones((4, 1)), np.ones((4, 1), dtype=bool)]}, "real"),
]


@pytest.mark.parametrize(("kwargs", "match"), SUPPLIED_ERRORS)
def test_supplied_arguments_are_checked(kwargs: dict[str, Any], match: str) -> None:
    base: dict[str, Any] = {
        "n_neurons": 5,
        "n_bins": 4,
        "n_trials": 60,
        "ranks": {"sa": 2, "sb": 1, "choice": 1},
        "seed": 0,
    }
    with pytest.raises(ParameterError, match=match):
        mtdr.simulate(**{**base, **kwargs})


# --------------------------------------------------------------- argument errors

BAD_ARGUMENTS: list[tuple[dict[str, Any], str]] = [
    ({"n_neurons": 0}, "n_neurons must be a positive integer"),
    ({"n_neurons": True}, "n_neurons must be a positive integer"),
    ({"n_bins": 2.5}, "n_bins must be a positive integer"),
    ({"n_trials": 1, "ranks": [1]}, "n_trials must be at least 2"),
    ({"n_trials": 3}, "n_trials must be at least 4"),
    ({"n_trials": 2, "ranks": [1, 1]}, "n_trials must be at least 3"),
    ({"ranks": {"a": 1}, "regressor_names": ["a"]}, "not both"),
    ({"ranks": "123"}, "ranks must be a sequence"),
    ({"ranks": 3}, "ranks must be a sequence"),
    ({"ranks": np.array([[1, 2]])}, "ranks must be a sequence"),
    ({"ranks": []}, "at least one regressor"),
    ({"ranks": [1, -1]}, "non-negative integer"),
    ({"ranks": [1, True]}, "non-negative integer"),
    ({"ranks": [1.0]}, "non-negative integer"),
    ({"ranks": [16]}, r"above min\(n_neurons, n_bins\) = 15"),
    ({"ranks": [1, 2], "regressor_names": ["a"]}, "regressor_names has 1"),
    ({"ranks": [1], "regressor_names": "a"}, "regressor_names must be a sequence"),
    ({"ranks": [1], "regressor_names": 3}, "regressor_names must be a sequence"),
    ({"ranks": [1, 1], "regressor_names": {"a", "b"}}, "must be a sequence"),
    ({"ranks": [1, 2], "regressor_names": ["a", "a"]}, "unique"),
    ({"ranks": [1], "regressor_names": ["intercept"]}, "reserved"),
    ({"ranks": {"total": 1}}, "reserved"),
    ({"ranks": [1], "regressor_names": [3]}, "must be str"),
    ({"ranks": {1: 1}}, "must be str"),
    ({"condition_independent": "False"}, "condition_independent must be a bool"),
    ({"condition_independent": 2}, "condition_independent must be a bool"),
    ({"condition_independent": np.array([True, False])}, "must be a bool"),
    ({"ranks": [1, 1], "levels": [[1, 2]]}, "levels has 1 entries"),
    ({"ranks": [1], "levels": "ab"}, "levels must be a sequence"),
    ({"ranks": [1], "levels": np.array([-1.0, 1.0])}, "must be 2-D"),
    ({"ranks": [1], "levels": [["a", "b"]]}, "not numeric"),
    ({"ranks": [1], "levels": [[[1, 2]]]}, "non-empty 1-D"),
    ({"ranks": [1], "levels": [[]]}, "non-empty 1-D"),
    ({"ranks": [1], "levels": [[1, np.inf]]}, "finite"),
    ({"ranks": [1], "levels": [[1, 1, 2]]}, "repeated values"),
    ({"ranks": [1], "levels": [[1]]}, "single value"),
    ({"ranks": [0] * 64, "levels": [[-1, 1]] * 64}, "64-bit"),
    ({"length_scale": 0.0}, "length_scale entries must be finite and > 0"),
    ({"length_scale": [1.0, 1.0]}, "expected 3 \\(or one more"),
    (
        {"length_scale": [1.0] * 4, "condition_independent": False},
        "has 4 entries; expected 3$",
    ),
    ({"length_scale": [1.0, "a", 1.0]}, "entries must be real numbers"),
    ({"length_scale": [1.0, True, 1.0]}, "entries must be real numbers"),
    ({"length_scale": np.ones((3, 1))}, "float or a sequence"),
    ({"length_scale": "2"}, "float or a sequence"),
    ({"length_scale": True}, "float or a sequence"),
    ({"length_scale": np.nan}, "finite"),
    ({"amplitude": -1.0}, "amplitude entries must be finite and >= 0"),
    ({"amplitude": [1.0, 1.0, 1.0, -1.0]}, ">= 0"),
    ({"amplitude": 1e308}, "not finite"),
    ({"noise_precision": [1.0, 2.0]}, "shape \\(100,\\)"),
    ({"noise_precision": -1.0}, "> 0"),
    ({"noise_precision": "x"}, "not numeric"),
    ({"noise_precision": True}, "not bool"),
    ({"drop_prob": 1.0}, r"drop_prob must be a number in \[0, 1\)"),
    ({"drop_prob": -0.1}, "drop_prob"),
    ({"drop_prob": True}, "drop_prob"),
    ({"seed": -1}, "seed must be a non-negative int"),
    ({"seed": 1.5}, "seed must be"),
    ({"seed": True}, "seed must be"),
    ({"seed": np.random.RandomState(0)}, "seed must be"),
]


@pytest.mark.parametrize(("kwargs", "match"), BAD_ARGUMENTS)
def test_invalid_arguments_raise_parameter_error(
    kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ParameterError, match=match):
        mtdr.simulate(**{"seed": 0, **kwargs})


def test_parameter_errors_are_value_errors() -> None:
    with pytest.raises(ValueError, match="drop_prob"):
        mtdr.simulate(drop_prob=2.0)


def test_minimum_trials_without_intercept() -> None:
    sim = _sim(n_trials=2, ranks=[1, 1], condition_independent=False)
    assert sim.X.shape == (2, 2)


def test_single_level_allowed_without_intercept() -> None:
    sim = _sim(ranks=[1, 1], levels=[[1.0], [-1.0, 1.0]], condition_independent=False)
    np.testing.assert_array_equal(sim.X[:, 0], 1.0)


def test_amplitude_zero_is_allowed() -> None:
    sim = _sim(amplitude=0.0, n_neurons=5, n_bins=4, ranks=[1])
    np.testing.assert_array_equal(sim.B["x0"], 0.0)
    assert sim.intercept is not None
    np.testing.assert_array_equal(sim.intercept, 0.0)
