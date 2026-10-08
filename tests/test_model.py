"""The `MTDR` estimator: construction, fit, fitted attributes, the canonical
frame, and every method on a fitted model.

Per method: a simulation-recovery test (the method recovers a quantity known
from the simulation, or equals an independent computation from the fitted
objects), property tests with hypothesis (invariances of the model: neuron
permutation, unobserved entries, `Y` scaling, affinity of the LLR (M48c)), and
the documented errors. MATLAB parity through the class is in
`tests/test_model_parity.py`.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import itertools
import os
import pickle
import re
import sys
import time
import types
import warnings
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import numpy as np
import pytest
import scipy.linalg
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import expit

import mtdr
import parity_fixtures as pf
from mtdr import MTDR, mmle, simulate
from mtdr.aic import aic as aic_value
from mtdr.aic import n_parameters_mmle
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
from mtdr.mmle import fit_mmle
from mtdr.stats import sufficient_statistics
from mtdr.svd_fit import fit_svd

NAMES = ("sa", "sb", "choice")


def _sim(seed: int = 0, **kw: Any) -> mtdr.SimulatedData:
    args: dict[str, Any] = {
        "n_neurons": 40,
        "n_bins": 8,
        "n_trials": 240,
        "ranks": {"sa": 2, "sb": 1, "choice": 2},
        "drop_prob": 0.2,
        "seed": seed,
    }
    args.update(kw)
    return simulate(**args)


@functools.cache
def _svd_fitted(seed: int = 0, **kw: Any) -> tuple[mtdr.SimulatedData, MTDR]:
    s = _sim(seed=seed, **kw)
    model = MTDR(ranks=s.ranks, estimator="svd").fit(
        s.Y_masked, s.X, regressor_names=s.regressor_names
    )
    return s, model


@pytest.fixture(scope="module")
def sim() -> mtdr.SimulatedData:
    return _sim()


@pytest.fixture(scope="module")
def svd_model(sim: mtdr.SimulatedData) -> MTDR:
    return MTDR(ranks=sim.ranks, estimator="svd").fit(
        sim.Y_masked, sim.X, regressor_names=sim.regressor_names
    )


@pytest.fixture(scope="module")
def mmle_model(sim: mtdr.SimulatedData) -> MTDR:
    return MTDR(ranks=sim.ranks).fit(
        sim.Y_masked, sim.X, regressor_names=sim.regressor_names
    )


@pytest.fixture(scope="module")
def raw_model(sim: mtdr.SimulatedData) -> MTDR:
    return MTDR(ranks=sim.ranks, canonicalize=False).fit(
        sim.Y_masked, sim.X, regressor_names=sim.regressor_names
    )


@pytest.fixture(scope="module")
def clean() -> tuple[mtdr.SimulatedData, MTDR, np.ndarray]:
    """A model whose fitted objects are the truth, and noise-free responses."""
    s = _sim(seed=1, drop_prob=0.0)
    model = MTDR(ranks=s.ranks, estimator="svd").fit(s.Y, s.X, regressor_names=NAMES)
    model.B_ = {n: np.array(s.B[n]) for n in NAMES}
    model.W_ = {n: np.array(s.W[n]) for n in NAMES}
    model.S_ = {n: np.array(s.S[n]) for n in NAMES}
    model.intercept_ = np.array(s.intercept)
    model.noise_precision_ = np.array(s.noise_precision)
    Y = model.predict(s.X)
    return s, model, Y


# ================================================================= constructor


def test_defaults_match_the_api() -> None:
    params = MTDR().get_params()
    assert params == {
        "ranks": "aic",
        "estimator": "mmle",
        "condition_independent": True,
        "max_rank": None,
        "min_observations": 2,
        "ridge": 0.0,
        "basis_ridge": 0.0,
        "rank_search_init": "svd_weighted",
        "rank_search_threshold": 0.0,
        "rank_search_warm_start": True,
        "ecme_max_iter": 100,
        "ecme_tol": 1.0,
        "refine_max_iter": 10,
        "refine_tol": 1e-4,
        "convergence_eps": 1e-12,
        "optimizer_max_iter": 2000,
        "optimizer_tol": 1e-6,
        "optimizer_progtol": 1e-9,
        "canonicalize": True,
        "verbose": 0,
        "n_jobs": None,
        "basis_preconditioning": False,
    }
    assert repr(MTDR()) == "MTDR()"
    assert repr(MTDR(ranks=[1, 2], ridge=0.5, canonicalize=False)) == (
        "MTDR(ranks=[1, 2], ridge=0.5, canonicalize=False)"
    )
    assert repr(MTDR(ecme_tol=1, verbose=0)) == "MTDR()"  # 1 == 1.0


@pytest.mark.parametrize(
    "kw",
    [
        {"ranks": "auto"},
        {"ranks": [1, -1]},
        {"ranks": []},
        {"ranks": [1.5]},
        {"ranks": {"a": -1}},
        {"ranks": {1: 1}},
        {"ranks": 3},
        {"estimator": "pca"},
        {"condition_independent": 1},
        {"max_rank": 0},
        {"min_observations": 0},
        {"ridge": -1.0},
        {"basis_ridge": np.nan},
        {"rank_search_threshold": -0.1},
        {"rank_search_init": "two"},
        {"rank_search_init": [0, 1]},
        {"rank_search_init": []},
        {"rank_search_warm_start": 1},
        {"ecme_max_iter": 0},
        {"refine_max_iter": 1.5},
        {"optimizer_max_iter": True},
        {"ecme_tol": 0.0},
        {"refine_tol": -1.0},
        {"convergence_eps": np.inf},
        {"optimizer_tol": "a"},
        {"canonicalize": None},
        {"verbose": 3},
        {"n_jobs": 0},
        {"n_jobs": -2},
        {"n_jobs": 1.5},
        {"n_jobs": True},
        {"basis_preconditioning": 1},
        {"basis_preconditioning": "yes"},
    ],
)
def test_bad_hyper_parameters(kw: dict[str, Any]) -> None:
    with pytest.raises(ParameterError):
        MTDR(**kw)


def test_set_params_validates_and_rolls_back() -> None:
    model = MTDR()
    assert model.set_params(estimator="svd", max_rank=3) is model
    with pytest.raises(ParameterError, match="unknown MTDR parameters"):
        model.set_params(nope=1)
    with pytest.raises(ParameterError):
        model.set_params(max_rank=0, ridge=1.0)
    assert model.max_rank == 3
    assert model.ridge == 0.0


def test_clone_contract() -> None:
    # The constructor stores its arguments as given (sklearn's estimator
    # contract), so klass(**get_params(deep=False)) has identical parameters.
    ranks = {"x0": 1, "x1": 1}
    s = simulate(n_neurons=10, n_bins=4, n_trials=40, ranks=[1, 1], seed=0)
    model = MTDR(ranks=ranks, estimator="svd").fit(s.Y, s.X)
    params = model.get_params(deep=False)
    rebuilt = type(model)(**params)
    assert all(rebuilt.get_params()[k] is v for k, v in params.items())
    assert not hasattr(rebuilt, "ranks_")
    # sklearn.base.clone deep-copies non-estimator parameters, so the clone's
    # are equal, not identical, and it is unfitted. CI installs scikit-learn on
    # one test leg (ubuntu, Python 3.13).
    base = pytest.importorskip("sklearn.base")  # pragma: no cover - optional
    clone = base.clone(model)  # pragma: no cover
    got = clone.get_params(deep=False)  # pragma: no cover
    assert got["ranks"] == ranks  # pragma: no cover
    assert got["ranks"] is not ranks  # pragma: no cover
    assert got == params  # pragma: no cover
    assert not hasattr(clone, "ranks_")  # pragma: no cover


def test_unfitted_model() -> None:
    model = MTDR()
    for name in ("ranks_", "B_", "aic_", "converged_"):
        assert not hasattr(model, name)
        with pytest.raises(NotFittedError, match="not fitted"):
            getattr(model, name)
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        model.nope  # noqa: B018
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((1, 1)))
    with pytest.raises(NotFittedError):
        model.aic()


# ================================================================= fit


def test_svd_fit_equals_the_functional_api(
    sim: mtdr.SimulatedData, svd_model: MTDR
) -> None:
    stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    ref = fit_svd(stats, [2, 1, 2])
    for p, name in enumerate(sim.regressor_names):
        np.testing.assert_allclose(svd_model.B_[name], ref.B[p], rtol=0, atol=1e-12)
    assert ref.intercept is not None
    assert svd_model.intercept_ is not None
    np.testing.assert_allclose(svd_model.intercept_, ref.intercept, atol=1e-12)
    np.testing.assert_allclose(svd_model.noise_precision_, ref.noise_precision)
    assert svd_model.log_likelihood_ == ref.log_likelihood
    assert svd_model.aic_ == ref.aic == svd_model.aic()
    assert svd_model.objective_ == svd_model.log_likelihood_
    assert svd_model.n_parameters_ == ref.n_parameters
    assert svd_model.estimator_ == "svd"
    assert svd_model.converged_
    assert svd_model.n_iter_ == {}
    assert svd_model.weight_posterior_cov_ is None
    assert svd_model.rank_search_history_ is None
    assert svd_model.n_observations_.tolist() == sim.mask.sum(axis=0).tolist()
    assert (svd_model.n_trials_, svd_model.n_neurons_, svd_model.n_bins_) == (
        240,
        40,
        8,
    )
    assert svd_model.regressor_names_ == NAMES
    assert svd_model.n_regressors_ == 3
    assert svd_model.ranks_ == {"sa": 2, "sb": 1, "choice": 2}
    assert svd_model.total_rank_ == 5
    assert svd_model.condition_independent_


def test_mmle_fit_equals_the_functional_api(
    sim: mtdr.SimulatedData, raw_model: MTDR, mmle_model: MTDR
) -> None:
    stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    ref = fit_mmle(stats, [2, 1, 2])
    for p, name in enumerate(sim.regressor_names):
        np.testing.assert_array_equal(raw_model.S_[name], ref.S[p])
        np.testing.assert_array_equal(raw_model.W_[name], ref.W[p])
        np.testing.assert_allclose(mmle_model.B_[name], ref.B[p], atol=1e-12)
    np.testing.assert_array_equal(raw_model.weight_posterior_cov_, ref.W_cov)
    for model in (raw_model, mmle_model):
        assert model.log_likelihood_ == ref.log_likelihood
        assert model.aic_ == ref.aic
        assert model.n_parameters_ == n_parameters_mmle([2, 1, 2], 40, 8)  # (M38a)
        assert model.converged_ == ref.converged
        assert model.n_iter_ == dict(ref.n_iter)
        assert model.estimator_ == "mmle"


def test_six_regressor_mmle_fit_equals_the_functional_api() -> None:
    # The class at the paper's number of regressors, in the default suite
    # (otherwise only the slow paper-scale search fits six regressors).
    ranks = [2, 2, 1, 2, 1, 1]
    s = simulate(
        n_neurons=60, n_bins=15, n_trials=300, ranks=ranks, drop_prob=0.3, seed=2
    )
    stats = sufficient_statistics(s.Y_masked, s.X, s.mask)
    ref, ref_caught = pf.run(lambda: fit_mmle(stats, ranks))
    model, caught = pf.run(
        lambda: MTDR(estimator="mmle", ranks=ranks, canonicalize=False).fit(
            s.Y_masked, s.X, s.mask
        )
    )
    assert caught.reasons == ref_caught.reasons
    assert model.n_regressors_ == 6
    assert model.total_rank_ == 9
    for p, name in enumerate(model.regressor_names_):
        np.testing.assert_array_equal(model.S_[name], ref.S[p])
        np.testing.assert_array_equal(model.W_[name], ref.W[p])
        np.testing.assert_array_equal(model.B_[name], ref.B[p])
    np.testing.assert_array_equal(model.noise_precision_, ref.noise_precision)
    np.testing.assert_array_equal(model.intercept_, ref.intercept)
    np.testing.assert_array_equal(model.weight_posterior_cov_, ref.W_cov)
    assert model.log_likelihood_ == ref.log_likelihood
    assert model.aic_ == ref.aic
    assert model.converged_ == ref.converged
    # log_likelihood_ is the marginal likelihood (M24) at the fitted parameters.
    marginal = mmle.marginal_log_likelihood(
        [model.S_[n] for n in model.regressor_names_],
        model.noise_precision_,
        model.intercept_,
        stats,
    )
    assert model.log_likelihood_ == pytest.approx(marginal, rel=1e-13)


def test_recovery_of_the_coefficient_matrices(
    sim: mtdr.SimulatedData, mmle_model: MTDR
) -> None:
    for name in sim.regressor_names:
        r = np.corrcoef(mmle_model.B_[name].ravel(), sim.B[name].ravel())[0, 1]
        assert r > 0.95, (name, r)
    lam = np.corrcoef(np.log(mmle_model.noise_precision_), np.log(sim.noise_precision))
    assert lam[0, 1] > 0.95


def test_canonical_frame(mmle_model: MTDR, raw_model: MTDR) -> None:
    # The canonical frame: W_ orthonormal in PC order, largest entry positive,
    # S_ = V Sigma; B_ and plug-in quantities unchanged; covariance R' C R.
    blocks = []
    for name in NAMES:
        W, S, B = mmle_model.W_[name], mmle_model.S_[name], mmle_model.B_[name]
        r = W.shape[1]
        np.testing.assert_allclose(W.T @ W, np.eye(r), atol=1e-12)
        np.testing.assert_allclose(W @ S.T, B, atol=1e-10)
        np.testing.assert_allclose(B, raw_model.B_[name], atol=1e-12)
        sigma = np.linalg.svd(B, compute_uv=False)[:r]
        np.testing.assert_allclose(np.linalg.norm(S, axis=0), sigma, rtol=1e-10)
        assert np.all(np.diff(sigma) <= 0)
        lead = W[np.argmax(np.abs(W), axis=0), np.arange(r)]
        assert np.all(lead > 0)
        blocks.append(np.linalg.lstsq(raw_model.W_[name], W, rcond=None)[0])
    R = scipy.linalg.block_diag(*blocks)
    cov_c = mmle_model.weight_posterior_cov_
    cov_r = raw_model.weight_posterior_cov_
    assert cov_c is not None
    assert cov_r is not None
    np.testing.assert_allclose(
        cov_c,
        R.T @ cov_r @ R,
        atol=1e-12,
    )
    assert mmle_model.log_likelihood_ == raw_model.log_likelihood_


# ------------------------------------------------------------ canonical_factors


def test_canonical_factors_is_the_fits_frame(mmle_model: MTDR) -> None:
    # The public PC frame (mtdr.canonical_factors) is the one the fit reports.
    for name in NAMES:
        r = mmle_model.ranks_[name]
        W, S = mtdr.canonical_factors(mmle_model.B_[name], r)
        np.testing.assert_array_equal(W, mmle_model.W_[name])
        np.testing.assert_array_equal(S, mmle_model.S_[name])


def test_canonical_factors_of_a_simulated_truth(sim: mtdr.SimulatedData) -> None:
    # rank=None keeps the numerical rank: the generating rank of a true B_p.
    for name in NAMES:
        B = sim.B[name]
        W, S = mtdr.canonical_factors(B)
        r = sim.ranks[name]
        assert W.shape == (B.shape[0], r)
        assert S.shape == (B.shape[1], r)
        np.testing.assert_allclose(W.T @ W, np.eye(r), atol=1e-12)
        np.testing.assert_allclose(W @ S.T, B, atol=1e-12)
        sigma = np.linalg.svd(B, compute_uv=False)[:r]
        np.testing.assert_allclose(np.linalg.norm(S, axis=0), sigma, rtol=1e-12)
        lead = W[np.argmax(np.abs(W), axis=0), np.arange(r)]
        assert np.all(lead > 0)
        # The fitted and true subspaces are compared in the same frame.
        assert W.flags.writeable
        assert S.flags.writeable


def test_canonical_factors_truncates_and_handles_rank_zero() -> None:
    B = np.random.default_rng(3).normal(size=(6, 4))
    W2, S2 = mtdr.canonical_factors(B, 2)
    W4, S4 = mtdr.canonical_factors(B, 4)
    np.testing.assert_array_equal(W2, W4[:, :2])
    np.testing.assert_array_equal(S2, S4[:, :2])
    np.testing.assert_allclose(W4 @ S4.T, B, atol=1e-12)
    W0, S0 = mtdr.canonical_factors(B, 0)
    assert W0.shape == (6, 0)
    assert S0.shape == (4, 0)
    Wz, Sz = mtdr.canonical_factors(np.zeros((3, 5)))  # numerical rank 0
    assert Wz.shape == (3, 0)
    assert Sz.shape == (5, 0)


def test_canonical_factors_sign_rule_and_ties() -> None:
    # The entry of largest absolute value of each W column is positive; on a
    # tie in absolute value, the first such entry (lowest row) decides.
    B = np.outer([0.0, -2.0, 1.0], [1.0, 2.0])
    W, S = mtdr.canonical_factors(B)
    assert W[:, 0].tolist() == pytest.approx([0.0, 2 / 5**0.5, -1 / 5**0.5])
    np.testing.assert_allclose(W @ S.T, B, atol=1e-14)
    # Entries equal in absolute value in exact arithmetic come out of the SVD
    # unequal at rounding level, so the sign of such a column is arbitrary
    # (documented): only W S^T = B and the argmax rule hold.
    tie = np.outer([1.0, -1.0], [3.0, 4.0])
    Wt, St = mtdr.canonical_factors(tie)
    assert abs(abs(Wt[0, 0]) - abs(Wt[1, 0])) < 1e-15
    assert Wt[np.argmax(np.abs(Wt[:, 0])), 0] > 0
    np.testing.assert_allclose(Wt @ St.T, tie, atol=1e-14)


@settings(deadline=None, max_examples=25)
@given(seed=st.integers(0, 2**31 - 1), sign=st.sampled_from([-1.0, 1.0]))
def test_property_canonical_factors_ignore_the_factorisation(
    seed: int, sign: float
) -> None:
    # B = W S^T has one canonical form whatever the factors were.
    rng = np.random.default_rng(seed)
    W_raw, S_raw = rng.normal(size=(7, 2)), rng.normal(size=(5, 2))
    R = rng.normal(size=(2, 2)) + 3 * np.eye(2)
    a = mtdr.canonical_factors(W_raw @ S_raw.T, 2)
    b = mtdr.canonical_factors((W_raw @ R) @ (S_raw @ np.linalg.inv(R).T).T, 2)
    c = mtdr.canonical_factors(sign * W_raw @ S_raw.T, 2)
    for x, y in [(a, b), (a, (c[0], sign * c[1]))]:
        np.testing.assert_allclose(x[0], y[0], atol=1e-8)
        np.testing.assert_allclose(x[1], y[1], atol=1e-8)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((np.ones(3),), "2-D"),
        ((np.ones((0, 3)),), "non-empty"),
        (([[1.0, 2.0], [3.0]],), "2-D real"),
        ((np.array([[np.nan, 1.0]]),), "finite"),
        ((np.array([["a"]]),), "real"),
        ((np.ones((3, 2)), 3), "between 0 and 2"),
        ((np.ones((3, 2)), -1), "between 0 and 2"),
        ((np.ones((3, 2)), 1.5), "integer"),
        ((np.ones((3, 2)), True), "integer"),
    ],
)
def test_canonical_factors_errors(args: tuple[Any, ...], match: str) -> None:
    with pytest.raises(ParameterError, match=match):
        mtdr.canonical_factors(*args)


def test_fixed_ranks_mapping_sequence_and_zero(sim: mtdr.SimulatedData) -> None:
    m = MTDR(ranks=[2, 0, 1], estimator="svd").fit(sim.Y_masked, sim.X)
    assert m.ranks_ == {"x0": 2, "x1": 0, "x2": 1}
    assert m.W_["x1"].shape == (40, 0)
    assert m.S_["x1"].shape == (8, 0)
    assert not m.B_["x1"].any()
    mm = MTDR(ranks={"b": 0, "a": 1, "c": 1}).fit(
        sim.Y_masked, sim.X, regressor_names=["a", "b", "c"]
    )
    assert mm.ranks_ == {"a": 1, "b": 0, "c": 1}
    assert mm.weight_posterior_cov_ is not None
    assert mm.weight_posterior_cov_.shape == (40, 2, 2)


@pytest.mark.parametrize(
    ("kw", "fit_kw", "match"),
    [
        ({"ranks": {"sa": 1}}, {"regressor_names": NAMES}, "name exactly"),
        ({"ranks": [1, 1]}, {}, "has 2 entries"),
        ({"ranks": [1, 1, 5], "max_rank": 4}, {}, "above max_rank"),
        ({"max_rank": 9}, {}, "above min"),
        ({"rank_search_init": [1, 1]}, {}, "rank_search_init has 2"),
        ({"rank_search_init": [1, 1, 5], "max_rank": 4}, {}, "above max_rank"),
        ({"estimator": "svd"}, {"regressor_names": ["a", "a", "b"]}, "unique"),
        ({"estimator": "svd"}, {"regressor_names": ["a", "total", "b"]}, "reserved"),
    ],
)
def test_fit_argument_errors(
    sim: mtdr.SimulatedData, kw: dict[str, Any], fit_kw: dict[str, Any], match: str
) -> None:
    with pytest.raises(ParameterError, match=match):
        MTDR(**kw).fit(sim.Y_masked, sim.X, **fit_kw)


def test_the_mmle_observation_floor_is_p_plus_2(sim: mtdr.SimulatedData) -> None:
    # Four trials suffice for "svd" (min_observations=2) but not for "mmle",
    # which needs residual degrees of freedom: with three regressors and the
    # intercept, P + 2 = 5 trials.
    mask = np.array(sim.mask)
    mask[:, 0] = False
    mask[np.arange(4), 0] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DesignWarning)
        MTDR(ranks=[1, 1, 1], estimator="svd").fit(sim.Y, sim.X, mask)
    # The message says where the floor comes from.
    floor = r'fewer than 5 trials, the "mmle" floor n_regressors \+ 2 \(above '
    with pytest.raises(ValidationError, match=floor + r"min_observations = 2\)"):
        MTDR(ranks=[1, 1, 1]).fit(sim.Y, sim.X, mask)
    with pytest.raises(ValidationError, match=r"fewer than 6 trials \(observed on"):
        MTDR(ranks=[1, 1, 1], min_observations=6).fit(sim.Y, sim.X, mask)


def test_bad_data_reaches_validate_inputs(sim: mtdr.SimulatedData) -> None:
    X = np.array(sim.X)
    X[:, 2] = 1.0
    with pytest.raises(ValidationError, match="constant"):
        MTDR().fit(sim.Y_masked, X)
    with pytest.raises(ValidationError, match="2-D"):
        MTDR().fit(sim.Y_masked, X[:, 0])


def test_an_uncentred_continuous_column_warns_under_mmle(
    sim: mtdr.SimulatedData,
) -> None:
    # Under "mmle", a continuous column (more than two values) whose mean is
    # more than one standard deviation from zero gets a fit-time DesignWarning,
    # since the marginal likelihood depends on where its zero is.
    X = np.array(sim.X)
    X[:, 0] += 3.0  # levels -2..2: sd 1.4, mean about 3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # (The shifted coding also slows the refinement to its cap here.)
        MTDR(ranks=[2, 1, 2]).fit(sim.Y_masked, X, regressor_names=NAMES)
    design = [str(w.message) for w in caught if w.category is DesignWarning]
    assert len(design) == 1
    assert design[0].startswith("continuous column(s) ['sa'] have a mean more than")
    # Not under "svd", whose fit does not depend on it; not for a column within
    # one standard deviation; never for a binary column (0/1 or +-1 codings)
    # or the intercept.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DesignWarning)
        # Off-centre codings slow the refinement (a cap, a basis line search);
        # this test is about the DesignWarning only.
        warnings.simplefilter("ignore", ConvergenceWarning)
        MTDR(ranks=[2, 1, 2], estimator="svd").fit(sim.Y_masked, X)
        near = np.array(sim.X)
        near[:, 0] += 0.9 * near[:, 0].std()
        near[:, 2] = (near[:, 2] > 0).astype(float)  # a 0/1 coding: mean 0.5
        MTDR(ranks=[2, 1, 2]).fit(sim.Y_masked, near)
        shifted = np.array(sim.X)
        shifted[:, 2] = (shifted[:, 2] > 0) + 5.0  # two values: not continuous
        MTDR(ranks=[2, 1, 2]).fit(sim.Y_masked, shifted)


def test_dataframe_column_names_name_the_regressors(sim: mtdr.SimulatedData) -> None:
    # With regressor_names=None, string column names of a DataFrame X name the
    # regressors, in order (duck-typed `columns`); other columns fall back to
    # x0, x1, ...; duplicates raise as regressor_names would; an explicit
    # regressor_names wins.
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(np.array(sim.X), columns=["stim_a", "stim_b", "choice"])
    model = MTDR(ranks={"stim_a": 2, "stim_b": 1, "choice": 2}, estimator="svd")
    model.fit(sim.Y_masked, df)
    assert model.regressor_names_ == ("stim_a", "stim_b", "choice")
    numbered = pd.DataFrame(np.array(sim.X))  # integer column labels
    fallback = MTDR(ranks=[2, 1, 2], estimator="svd").fit(sim.Y_masked, numbered)
    assert fallback.regressor_names_ == ("x0", "x1", "x2")
    twice = pd.DataFrame(np.array(sim.X), columns=["a", "a", "b"])
    with pytest.raises(ParameterError, match="must be unique"):
        MTDR(ranks=[2, 1, 2], estimator="svd").fit(sim.Y_masked, twice)
    named = MTDR(ranks=[2, 1, 2], estimator="svd").fit(
        sim.Y_masked, df, regressor_names=["p", "q", "r"]
    )
    assert named.regressor_names_ == ("p", "q", "r")
    # Duck typing: no usable `columns` gives the default names.
    for odd in (np.zeros((2, 2)), types.SimpleNamespace(columns=3)):
        assert mtdr.model._column_names(odd) is None
    assert mtdr.model._column_names(types.SimpleNamespace(columns=[])) is None


@pytest.mark.parametrize(
    ("Y", "X", "name"),
    [
        ([[[1.0], [2.0, 3.0]]], [[1.0]], "Y"),
        (np.zeros((3, 2, 2)), [[1.0], [2.0, 3.0], [1.0]], "X"),
    ],
)
def test_ragged_inputs_are_validation_errors(Y: Any, X: Any, name: str) -> None:
    # A ValidationError, not NumPy's raw inhomogeneous-shape ValueError.
    with pytest.raises(ValidationError, match=rf"^{name} is ragged"):
        MTDR(ranks=[1]).fit(Y, X)


def test_no_intercept(sim: mtdr.SimulatedData) -> None:
    Y = np.array(sim.Y) - np.array(sim.intercept)[None]
    model = MTDR(ranks=[2, 1, 2], condition_independent=False).fit(Y, sim.X, sim.mask)
    assert model.intercept_ is None
    assert not model.condition_independent_
    np.testing.assert_allclose(
        model.predict(sim.X[:2]),
        np.einsum("kp,pit->kit", sim.X[:2], np.stack(list(model.B_.values()))),
    )
    assert "intercept" not in model.explained_variance(Y, sim.X, sim.mask)
    assert model.n_parameters_ == n_parameters_mmle(
        [2, 1, 2], 40, 8, condition_independent=False
    )
    assert np.isfinite(model.project(Y, "x0", mask=sim.mask)).all()


def test_soft_design_warnings_at_fit(sim: mtdr.SimulatedData) -> None:
    X = np.array(sim.X)
    rng = np.random.default_rng(0)
    X[:, 1] = X[:, 0] + 0.05 * rng.normal(size=len(X))
    with pytest.warns(DesignWarning, match="near-collinear"):
        MTDR(ranks=[1, 1, 1], estimator="svd").fit(sim.Y_masked, X)
    # Scale disparities warn only when basis_ridge > 0.
    X = np.array(sim.X) * [1.0, 1.0, 100.0]
    MTDR(ranks=[1, 1, 1], estimator="svd").fit(sim.Y_masked, X)  # no warning
    with pytest.warns(DesignWarning, match="basis_ridge > 0"):
        MTDR(ranks=[2, 1, 2], basis_ridge=0.1).fit(sim.Y_masked, X)


def test_rank_deficient_neurons_warn_once_per_fit(sim: mtdr.SimulatedData) -> None:
    # fit_svd warns per candidate fit; MTDR.fit aggregates them into one.
    mask = np.array(sim.mask)
    mask[:, 3] = np.array(sim.X)[:, 0] == 2  # x0 constant over neuron 3's trials
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = MTDR(ranks="aic", estimator="svd", max_rank=4).fit(sim.Y, sim.X, mask)
    design = [w for w in caught if issubclass(w.category, DesignWarning)]
    assert len(design) == 1, [str(w.message) for w in caught]
    assert "neurons [3]" in str(design[0].message)
    assert model.rank_search_history_ is not None


def test_other_warnings_of_the_fits_pass_through(
    sim: mtdr.SimulatedData, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = fit_svd

    def noisy(*args: Any, **kw: Any) -> Any:
        warnings.warn("from inside the fit", UserWarning, stacklevel=1)
        return real(*args, **kw)

    monkeypatch.setattr(mtdr.model, "fit_svd", noisy)
    with pytest.warns(UserWarning, match="from inside the fit"):
        MTDR(ranks=[1, 1, 1], estimator="svd").fit(sim.Y_masked, sim.X)


def test_convergence_warnings_of_a_fixed_fit_pass_through(
    sim: mtdr.SimulatedData,
) -> None:
    with pytest.warns(
        ConvergenceWarning, match=r"refinement: hit its iteration cap \(max_iter = 1;"
    ):
        model = MTDR(ranks=[1, 1, 1], refine_max_iter=1, refine_tol=1e-300).fit(
            sim.Y_masked, sim.X
        )
    assert not model.converged_


def test_convergence_warnings_of_a_search_are_aggregated(
    sim: mtdr.SimulatedData,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = MTDR(ranks="aic", refine_max_iter=1, refine_tol=1e-300, max_rank=3).fit(
            sim.Y_masked, sim.X
        )
    conv = [str(w.message) for w in caught if w.category is ConvergenceWarning]
    assert len(conv) == 1
    assert conv[0].startswith("rank search: ")
    assert (
        "the selected fit did: refinement: hit its iteration cap (max_iter = 1;"
        in (conv[0])
    )
    assert not model.converged_
    assert model.rank_search_history_ is not None
    assert not any(model.rank_search_history_.converged)


def test_search_with_only_rejected_failures_says_so() -> None:
    runner = mtdr.model._Runner.__new__(mtdr.model._Runner)
    runner.deficient = set()
    runner.failed = [("mmle", (2, 1), ["refinement: x" + "y" * 300, "other"])]
    runner.n_fits = {"svd": 30, "mmle": 7}
    runner.searched = True
    runner.selected = ("mmle", (1, 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.report()
    (message,) = [str(w.message) for w in caught]
    # The denominator counts the estimator's fits only, and each failed fit is
    # listed with its first reason, cut to about 200 chars.
    head = "rank search: 1 of 7 mmle fits reported non-convergence: mmle [2, 1] ("
    reason = "refinement: x" + "y" * 184 + "..."
    assert message == (
        head + reason + "); the selected fit converged; rejected candidates' "
        "failures can affect which ranks were chosen"
    )
    assert len(reason) == 200
    runner.failed = [("mmle", (k, 1), ["x"]) for k in range(25)]
    with pytest.warns(ConvergenceWarning, match="and 5 more"):
        runner.report()


# ----------------------------------------------------------------- the search


def test_svd_search(sim: mtdr.SimulatedData) -> None:
    model = MTDR(ranks="aic", estimator="svd", rank_search_init="svd_weighted").fit(
        sim.Y_masked, sim.X, regressor_names=NAMES
    )
    h = model.rank_search_history_
    assert h is not None
    assert h.estimator == "svd"
    assert h.svd_stage is None
    assert h.init_ranks.tolist() == [1, 1, 1]  # the strings coincide for "svd"
    assert model.ranks_ == h.final_ranks()
    assert model.aic_ == h.aic[-1]
    start = MTDR(ranks="aic", estimator="svd", rank_search_init=[2, 1, 1]).fit(
        sim.Y_masked, sim.X
    )
    assert start.rank_search_history_ is not None
    assert start.rank_search_history_.init_ranks.tolist() == [2, 1, 1]


@pytest.mark.parametrize("init", ["svd_weighted", "svd", "ones", [1, 1, 2]])
def test_mmle_search_inits(sim: mtdr.SimulatedData, init: Any) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = MTDR(ranks="aic", rank_search_init=init, max_rank=4).fit(
            sim.Y_masked, sim.X, regressor_names=NAMES
        )
    # At most one aggregated warning, about rejected candidates (an
    # under-ranked candidate can stop on the refinement's iteration cap).
    assert len(caught) <= 1
    assert all("the selected fit converged" in str(w.message) for w in caught)
    h = model.rank_search_history_
    assert h is not None
    assert h.estimator == "mmle"
    if isinstance(init, list):
        assert h.svd_stage is None
        assert h.init_ranks.tolist() == init
    elif init == "ones":
        assert h.svd_stage is None
        assert h.init_ranks.tolist() == [1, 1, 1]
    else:
        assert h.svd_stage is not None
        assert h.svd_stage.estimator == "svd"
        assert h.init_ranks.tolist() == h.svd_stage.ranks[-1].tolist()
    # The search selects with (M38); aic_ reports (M38a).
    ll = model.log_likelihood_
    ranks = list(model.ranks_.values())
    count = n_parameters_mmle(ranks, 40, 8, formula="reference")
    assert h.aic[-1] == pytest.approx(aic_value(ll, count), rel=1e-14)
    assert model.aic_ == aic_value(ll, n_parameters_mmle(ranks, 40, 8))
    assert model.ranks_ == {"sa": 2, "sb": 1, "choice": 2}


def test_weighted_seed_is_the_weighted_svd_search(sim: mtdr.SimulatedData) -> None:
    model = MTDR(ranks="aic", max_rank=4).fit(sim.Y_masked, sim.X)
    stats = sufficient_statistics(sim.Y, sim.X, sim.mask)
    from mtdr.rank_search import greedy_aic

    _, ref = greedy_aic(
        lambda r: fit_svd(stats, r, precision_weighted=True),
        lambda f, r: f.aic,
        [1, 1, 1],
        4,
    )
    assert model.rank_search_history_ is not None
    stage = model.rank_search_history_.svd_stage
    assert stage is not None
    np.testing.assert_array_equal(stage.ranks, ref.ranks)
    np.testing.assert_array_equal(stage.aic, ref.aic)


def test_warm_search_starts_candidates_from_the_accepted_fit(
    sim: mtdr.SimulatedData, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The start fit is cold; each candidate starts from the accepted fit's
    # S, lambda and b, plus the candidate SVD fit's last column for the raised p.
    calls: list[tuple[tuple[int, ...], Any, Any]] = []

    def spy(stats: Any, ranks: Any, init: Any = None, **kw: Any) -> Any:
        out = fit_mmle(stats, ranks, init, **kw)
        calls.append((tuple(int(r) for r in ranks), init, out))
        return out

    monkeypatch.setattr(mtdr.model, "fit_mmle", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = MTDR(
            ranks="aic",
            rank_search_init="ones",
            max_rank=4,
            rank_search_warm_start=True,
        ).fit(sim.Y_masked, sim.X)
    h = model.rank_search_history_
    assert h is not None
    assert len(h.accepted) >= 2
    assert calls[0][1] is None
    fits = {ranks: out for ranks, _, out in calls}
    assert len(fits) == len(calls)
    stats = sufficient_statistics(sim.Y_masked, sim.X, sim.mask)
    for ranks, init, _ in calls[1:]:
        assert isinstance(init, mtdr.MMLEFit)
        # Candidates of round k raise the sum of the accepted ranks[k] by one.
        k = sum(ranks) - int(h.init_ranks.sum()) - 1
        accepted = fits[tuple(int(r) for r in h.ranks[k])]
        (p,) = np.flatnonzero(np.array(ranks) != np.array(accepted.ranks))
        svd = fit_svd(stats, list(ranks))
        for q in range(len(ranks)):
            want = accepted.S[q]
            if q == p:
                want = np.column_stack([want, svd.S[q][:, -1]])
            np.testing.assert_array_equal(init.S[q], want)
        np.testing.assert_array_equal(init.noise_precision, accepted.noise_precision)
        np.testing.assert_array_equal(init.intercept, accepted.intercept)
        np.testing.assert_array_equal(
            init.rank_deficient_neurons, svd.rank_deficient_neurons
        )


def test_warm_search_matches_the_cold_search(sim: mtdr.SimulatedData) -> None:
    kw: dict[str, Any] = {"ranks": "aic", "rank_search_init": "ones", "max_rank": 4}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        cold = MTDR(rank_search_warm_start=False, **kw).fit(sim.Y_masked, sim.X)
        warm = MTDR(rank_search_warm_start=True, **kw).fit(sim.Y_masked, sim.X)
    a, b = cold.rank_search_history_, warm.rank_search_history_
    assert a is not None
    assert b is not None
    np.testing.assert_array_equal(a.ranks, b.ranks)
    assert a.accepted == b.accepted
    # Measured: 3.5e-10 and 3.0e-7.
    np.testing.assert_allclose(b.aic, a.aic, rtol=1e-8)
    for x, y in zip(a.candidates, b.candidates, strict=True):
        assert x.keys() == y.keys()
        for name in x:
            assert y[name] == pytest.approx(x[name], rel=1e-6)
    assert warm.ranks_ == cold.ranks_


@pytest.mark.parametrize(
    "kw",
    [
        {"ranks": "aic", "estimator": "svd", "max_rank": 3},
        {"ranks": [2, 1, 2]},
    ],
)
def test_warm_start_leaves_other_fits_alone(
    sim: mtdr.SimulatedData, kw: dict[str, Any]
) -> None:
    cold = MTDR(rank_search_warm_start=False, **kw).fit(sim.Y_masked, sim.X)
    warm = MTDR(rank_search_warm_start=True, **kw).fit(sim.Y_masked, sim.X)
    assert warm.log_likelihood_ == cold.log_likelihood_
    for name in cold.B_:
        np.testing.assert_array_equal(warm.B_[name], cold.B_[name])


@pytest.mark.parametrize("warm", [True, False])
@pytest.mark.parametrize("on", [True, False])
def test_basis_preconditioning_reaches_cold_fits_only(
    sim: mtdr.SimulatedData, monkeypatch: pytest.MonkeyPatch, warm: bool, on: bool
) -> None:
    # Every fit started from its own SVD fit gets BASIS_SPAN_SCALE when the
    # option is on; a warm-started candidate never does.
    calls: list[tuple[Any, float]] = []

    def spy(stats: Any, ranks: Any, init: Any = None, **kw: Any) -> Any:
        calls.append((init, kw["basis_span_scale"]))
        return fit_mmle(stats, ranks, init, **kw)

    monkeypatch.setattr(mtdr.model, "fit_mmle", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        MTDR(
            ranks="aic",
            rank_search_init="ones",
            max_rank=3,
            rank_search_warm_start=warm,
            basis_preconditioning=on,
        ).fit(sim.Y_masked, sim.X)
        MTDR(ranks=sim.ranks, basis_preconditioning=on).fit(
            sim.Y_masked, sim.X, regressor_names=sim.regressor_names
        )
    assert len(calls) > 2
    for init, scale in calls:
        cold = init is None
        assert scale == (mmle.BASIS_SPAN_SCALE if on and cold else 1.0)
    assert any(init is not None for init, _ in calls) == warm


def test_basis_preconditioning_finds_the_same_fits(sim: mtdr.SimulatedData) -> None:
    # The same search and, at the optimiser's tolerance, the same fit.
    kw: dict[str, Any] = {
        "ranks": "aic",
        "rank_search_init": "ones",
        "max_rank": 4,
        "rank_search_warm_start": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        plain = MTDR(**kw).fit(sim.Y_masked, sim.X)
        scaled = MTDR(basis_preconditioning=True, **kw).fit(sim.Y_masked, sim.X)
    a, b = plain.rank_search_history_, scaled.rank_search_history_
    assert a is not None
    assert b is not None
    np.testing.assert_array_equal(a.ranks, b.ranks)
    assert a.accepted == b.accepted
    np.testing.assert_allclose(b.aic, a.aic, rtol=1e-9)
    assert scaled.ranks_ == plain.ranks_
    assert scaled.log_likelihood_ >= plain.log_likelihood_ - 1e-9 * abs(
        plain.log_likelihood_
    )
    for name in plain.B_:
        np.testing.assert_allclose(
            scaled.B_[name],
            plain.B_[name],
            rtol=0,
            atol=1e-5 * np.abs(plain.B_[name]).max(),
        )


def test_basis_preconditioning_leaves_the_svd_estimator_alone(
    sim: mtdr.SimulatedData,
) -> None:
    kw: dict[str, Any] = {"ranks": "aic", "estimator": "svd", "max_rank": 3}
    plain = MTDR(**kw).fit(sim.Y_masked, sim.X)
    scaled = MTDR(basis_preconditioning=True, **kw).fit(sim.Y_masked, sim.X)
    assert scaled.log_likelihood_ == plain.log_likelihood_
    for name in plain.B_:
        np.testing.assert_array_equal(scaled.B_[name], plain.B_[name])


@pytest.mark.parametrize(
    ("warm", "verbose", "precondition"),
    [(True, 2, False), (False, 0, False), (False, 0, True)],
)
def test_n_jobs_search_is_identical(
    sim: mtdr.SimulatedData,
    capsys: pytest.CaptureFixture[str],
    warm: bool,
    verbose: int,
    precondition: bool,
) -> None:
    # The candidates fitted in worker processes give the same search, fit,
    # warnings and verbose=2 lines (a round's lines at its end, in regressor
    # order) as fitting them in turn. refine_max_iter=1 makes every fit fail,
    # so the aggregated warning lists candidates from the workers. The workers'
    # BLAS runs on one thread and the caller's may not, so numbers are compared
    # to rounding (equal bit for bit on the builds measured). With
    # basis_preconditioning, the workers' cold candidates are scaled.
    kw: dict[str, Any] = {
        "ranks": "aic",
        "rank_search_init": "ones",
        "max_rank": 3,
        "rank_search_warm_start": warm,
        "basis_preconditioning": precondition,
        "refine_max_iter": 1,
        "refine_tol": 1e-300,
        "verbose": verbose,
    }
    before = {name: os.environ.get(name) for name in mtdr.model._BLAS_THREADS}
    runs = []
    for n_jobs in (1, 2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = MTDR(n_jobs=n_jobs, **kw).fit(sim.Y_masked, sim.X)
        printed = capsys.readouterr().out
        runs.append((model, [(w.category, str(w.message)) for w in caught], printed))
        assert {n: os.environ.get(n) for n in before} == before
    (one, one_warnings, one_out), (two, two_warnings, two_out) = runs
    a, b = one.rank_search_history_, two.rank_search_history_
    assert a is not None
    assert b is not None
    assert len(a.accepted) >= 1
    np.testing.assert_array_equal(a.ranks, b.ranks)
    assert a.accepted == b.accepted
    np.testing.assert_allclose(b.aic, a.aic, rtol=1e-10)
    for x, y in zip(a.candidates, b.candidates, strict=True):
        assert x.keys() == y.keys()
        for name in x:
            assert y[name] == pytest.approx(x[name], rel=1e-10)
    assert a.converged == b.converged
    assert a.n_fits == b.n_fits
    assert two.log_likelihood_ == pytest.approx(one.log_likelihood_, rel=1e-10)
    for name in one.B_:
        np.testing.assert_allclose(two.B_[name], one.B_[name], rtol=1e-8, atol=1e-10)

    def digitless(text: str) -> str:
        return re.sub(r"\d", "#", text)

    assert [(c, digitless(m)) for c, m in one_warnings] == [
        (c, digitless(m)) for c, m in two_warnings
    ]
    assert any("rank search: " in message for _, message in one_warnings)
    assert digitless(one_out) == digitless(two_out)
    if verbose == 2:
        assert "mmle [2, 1, 1]: refine: iteration 1" in one_out
    else:
        assert one_out == ""


def test_n_jobs_resolves() -> None:
    assert MTDR(n_jobs=-1)._params()["n_jobs"] == mtdr.model._cpu_count() >= 1
    assert MTDR()._params()["n_jobs"] == 1
    assert MTDR(n_jobs=3)._params()["n_jobs"] == 3


def test_candidate_pool_workers_and_environment(
    sim: mtdr.SimulatedData, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The workers start with the BLAS variables at 1; the caller's are set only
    # while they start, and restored. An error in a worker reaches the caller.
    stats = sufficient_statistics(sim.Y_masked, sim.X, sim.mask)
    options = mtdr.model._mmle_options(MTDR()._params())
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    bad = np.array([1, -1, 1], dtype=np.int64)
    with mtdr.model._candidate_pool(2) as pool:
        assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
        assert "MKL_NUM_THREADS" not in os.environ
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            assert pool.submit(os.getenv, name).result() == "1"
        future = pool.submit(
            mtdr.model._pool_fit, stats, bad, None, options, np.geterr()
        )
        with pytest.raises(ParameterError, match="ranks"):
            future.result()


def test_candidate_pool_terminates_its_workers_on_an_error() -> None:
    # An error or an interrupt does not wait for the running fits.
    def interrupted() -> None:
        with mtdr.model._candidate_pool(1) as pool:
            pool.submit(time.sleep, 60)
            raise KeyboardInterrupt

    start = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        interrupted()
    assert time.perf_counter() - start < 30


def test_gather_raises_the_first_error_in_job_order() -> None:
    # The error fitting in turn would raise (job 1's, not job 2's, which comes
    # first in time), without waiting for the slow job after it.
    def job(i: int) -> int:
        time.sleep({0: 0.2, 1: 0.4, 2: 0.0, 3: 3.0}[i])
        if i in (1, 2):
            raise ValueError(f"job {i}")
        return i

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        assert mtdr.model._gather(pool, job, [(0,), (3,)]) == [0, 3]
        start = time.perf_counter()
        with pytest.raises(ValueError, match="job 1"):
            mtdr.model._gather(pool, job, [(0,), (1,), (2,), (3,)])
        assert time.perf_counter() - start < 2.5


class _BrokenPool(concurrent.futures.Executor):
    """An executor whose every job fails as a dead worker's does."""

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        future.set_exception(BrokenProcessPool("a worker died"))
        return future


@pytest.mark.parametrize("failure", ["start", "broken"])
def test_n_jobs_falls_back_to_fitting_in_turn(
    sim: mtdr.SimulatedData, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    # A pool that cannot start (a daemonic process) or breaks (a script without
    # the __main__ guard, a dead worker) leaves the search to fit its candidates
    # in turn, with a RuntimeWarning, and the same result.
    kw: dict[str, Any] = {"ranks": "aic", "rank_search_init": "ones", "max_rank": 3}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        want = MTDR(**kw).fit(sim.Y_masked, sim.X)

    @contextlib.contextmanager
    def pool(n_workers: int) -> Any:
        if failure == "start":
            raise AssertionError("daemonic processes are not allowed to have children")
        yield _BrokenPool()

    monkeypatch.setattr(mtdr.model, "_candidate_pool", pool)
    reason = "could not start 2 worker" if failure == "start" else "stopped"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        with pytest.warns(RuntimeWarning, match=f"n_jobs: .*{reason}.* in turn"):
            got = MTDR(n_jobs=2, **kw).fit(sim.Y_masked, sim.X)
    a, b = want.rank_search_history_, got.rank_search_history_
    assert a is not None
    assert b is not None
    np.testing.assert_array_equal(a.ranks, b.ranks)
    assert a.candidates == b.candidates
    assert a.n_fits == b.n_fits
    assert got.log_likelihood_ == want.log_likelihood_


def test_n_jobs_pool_size_counts_the_movable_regressors(
    sim: mtdr.SimulatedData, monkeypatch: pytest.MonkeyPatch
) -> None:
    # From [3, 3, 1] with max_rank 3, a round has one candidate, so no pool is
    # started; from [1, 3, 1], two candidates, so two workers (n_jobs is 8).
    sizes: list[int] = []
    real = mtdr.model._candidate_pool

    def pool(n_workers: int) -> Any:
        sizes.append(n_workers)
        return real(n_workers)

    monkeypatch.setattr(mtdr.model, "_candidate_pool", pool)
    kw: dict[str, Any] = {"ranks": "aic", "max_rank": 3, "n_jobs": 8}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        MTDR(rank_search_init=[3, 3, 1], **kw).fit(sim.Y_masked, sim.X)
        assert sizes == []
        MTDR(rank_search_init=[1, 3, 1], **kw).fit(sim.Y_masked, sim.X)
    assert sizes == [2]


def test_pool_fit_returns_the_fit_its_warnings_and_lines(
    sim: mtdr.SimulatedData,
) -> None:
    # What a worker runs, here in this process: the fit `_mmle` makes, its
    # warnings without their source, and its prefixed verbose lines, all
    # picklable for the trip back.
    stats = sufficient_statistics(sim.Y_masked, sim.X, sim.mask)
    params = MTDR(refine_max_iter=1, refine_tol=1e-300, verbose=2)._params()
    options = mtdr.model._mmle_options(params)
    ranks = np.array([2, 1, 1], dtype=np.int64)
    out, caught, printed = pickle.loads(
        pickle.dumps(mtdr.model._pool_fit(stats, ranks, None, options, np.geterr()))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        want = fit_mmle(stats, ranks, **options)
    assert out.log_likelihood == want.log_likelihood
    assert [w.category for w in caught] == [ConvergenceWarning]
    assert caught[0].source is None
    assert printed.startswith("mmle [2, 1, 1]: ")
    assert "mmle [2, 1, 1]: refine: iteration 1" in printed


def test_verbose_prints_accepted_steps(
    sim: mtdr.SimulatedData, capsys: pytest.CaptureFixture[str]
) -> None:
    MTDR(ranks="aic", estimator="svd", verbose=1, max_rank=3).fit(sim.Y_masked, sim.X)
    out = capsys.readouterr().out
    assert out.startswith("rank search (svd): step 0:")
    MTDR(ranks=[2, 1, 2], verbose=2).fit(sim.Y_masked, sim.X)
    out = capsys.readouterr().out
    # Every line of a fit names its candidate.
    assert "mmle [2, 1, 2]: refine: iteration 1" in out
    assert all(line.startswith("mmle [2, 1, 2]: ") for line in out.splitlines())


def test_fitted_model_pickles(mmle_model: MTDR, sim: mtdr.SimulatedData) -> None:
    again = pickle.loads(pickle.dumps(mmle_model))
    np.testing.assert_array_equal(
        again.predict(sim.X[:3]), mmle_model.predict(sim.X[:3])
    )
    assert again.get_params() == mmle_model.get_params()


@settings(deadline=None, max_examples=15)
@given(perm_seed=st.integers(0, 2**31), seed=st.integers(0, 50))
def test_property_neuron_permutation_equivariance(perm_seed: int, seed: int) -> None:
    s = _sim(seed=seed, n_neurons=12, n_bins=5, n_trials=60)
    perm = np.random.default_rng(perm_seed).permutation(12)
    a = MTDR(ranks=[2, 1, 2], estimator="svd").fit(s.Y, s.X, s.mask)
    b = MTDR(ranks=[2, 1, 2], estimator="svd").fit(s.Y[:, perm], s.X, s.mask[:, perm])
    for name in a.regressor_names_:
        np.testing.assert_allclose(b.B_[name], a.B_[name][perm], atol=1e-9)
    np.testing.assert_allclose(b.noise_precision_, a.noise_precision_[perm], rtol=1e-9)
    assert b.aic_ == pytest.approx(a.aic_, rel=1e-12)


@settings(deadline=None, max_examples=10)
@given(seed=st.integers(0, 50), garbage=st.floats(-1e6, 1e6))
def test_property_unobserved_entries_are_ignored(seed: int, garbage: float) -> None:
    s = _sim(seed=seed, n_neurons=10, n_bins=4, n_trials=50, drop_prob=0.3)
    Y = np.array(s.Y)
    Y[~s.mask] = garbage
    a = MTDR(ranks=[2, 1, 2], estimator="svd").fit(s.Y_masked, s.X)
    b = MTDR(ranks=[2, 1, 2], estimator="svd").fit(Y, s.X, s.mask)
    assert a.aic_ == b.aic_
    assert a.log_likelihood(Y, s.X, s.mask) == b.log_likelihood(s.Y_masked, s.X)


@settings(deadline=None, max_examples=5)
@given(c=st.sampled_from([1e-3, 0.5, 3.0, 1e3]))
def test_property_scaling_y_mmle(c: float) -> None:
    # Y -> cY gives B -> cB and lambda -> lambda / c^2, the same ranks, to the
    # optimiser's tolerance (its gradient test is absolute, so where it stops
    # depends on the units of Y).
    s = _sim(seed=2, n_neurons=15, n_bins=5, n_trials=80)
    a = MTDR(ranks=[2, 1, 2]).fit(s.Y_masked, s.X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        b = MTDR(ranks=[2, 1, 2]).fit(c * s.Y_masked, s.X)
    for name in a.regressor_names_:
        np.testing.assert_allclose(
            b.B_[name], c * a.B_[name], rtol=0, atol=1e-3 * c * np.abs(a.B_[name]).max()
        )
    np.testing.assert_allclose(b.noise_precision_ * c**2, a.noise_precision_, rtol=1e-3)


# ================================================================= evaluation


def test_predict(svd_model: MTDR, sim: mtdr.SimulatedData) -> None:
    out = svd_model.predict(sim.X)
    assert svd_model.intercept_ is not None
    expected = svd_model.intercept_[None] + sum(
        sim.X[:, p, None, None] * svd_model.B_[n][None] for p, n in enumerate(NAMES)
    )
    np.testing.assert_allclose(out, expected, atol=1e-12)
    for bad, match in [
        (sim.X[:, :2], "columns"),
        (np.full((2, 3), np.nan), "finite"),
        (sim.X[:, 0], "2-D"),
        (np.full((2, 3), "a"), "2-D"),
        ([[1, 2], [3]], "ragged or not an array"),
    ]:
        with pytest.raises(ValidationError, match=match):
            svd_model.predict(bad)  # type: ignore[arg-type]


def test_log_likelihood_is_the_plug_in(
    svd_model: MTDR, sim: mtdr.SimulatedData
) -> None:
    # On the training data the svd plug-in equals log_likelihood_ (M21a).
    ll = svd_model.log_likelihood(sim.Y_masked, sim.X)
    assert ll == pytest.approx(svd_model.log_likelihood_, rel=1e-12)
    assert svd_model.score(sim.Y_masked, sim.X) == ll
    per = svd_model.log_likelihood(sim.Y_masked, sim.X, per_trial=True)
    assert per.shape == (240,)
    assert per.sum() == pytest.approx(ll, rel=1e-12)
    # An empty trial is NaN; a never-observed neuron contributes nothing.
    mask = np.array(sim.mask)
    mask[0] = False
    mask[:, 5] = False
    per = svd_model.log_likelihood(sim.Y, sim.X, mask, per_trial=True)
    assert np.isnan(per[0])
    assert np.isfinite(per[1:]).all()
    with pytest.raises(ValidationError, match="neurons"):
        svd_model.log_likelihood(sim.Y[:, :5], sim.X)
    with pytest.raises(ParameterError, match="per_trial"):
        svd_model.log_likelihood(sim.Y, sim.X, per_trial="yes")  # type: ignore[call-overload]


@settings(deadline=None, max_examples=25)
@given(k=st.integers(0, 239), i=st.integers(0, 39))
def test_property_masking_removes_exactly_one_term(k: int, i: int) -> None:
    s, model = _svd_fitted()
    mask = np.array(s.mask)
    mask[k, i] = True
    full = model.log_likelihood(s.Y, s.X, mask, per_trial=True)
    mask[k, i] = False
    less = model.log_likelihood(s.Y, s.X, mask, per_trial=True)
    lam = model.noise_precision_[i]
    resid = s.Y[k, i] - model.predict(s.X[k : k + 1])[0, i]
    term = 0.5 * 8 * (np.log(lam) - np.log(2 * np.pi)) - 0.5 * lam * resid @ resid
    if mask[k].any():
        assert full[k] - less[k] == pytest.approx(term, rel=1e-9, abs=1e-9)
    else:
        assert np.isnan(less[k])
        assert full[k] == pytest.approx(term, rel=1e-9)


def test_explained_variance(svd_model: MTDR, sim: mtdr.SimulatedData) -> None:
    ev = svd_model.explained_variance(sim.Y_masked, sim.X)
    assert list(ev) == [*NAMES, "intercept", "total"]
    m = sim.mask[:, :, None]
    Y = np.array(sim.Y)
    mean = (Y * m).sum(axis=(0, 2)) / (m.sum(axis=(0, 2)) * 8)
    base = (np.where(m, Y - mean[None, :, None], 0) ** 2).sum()
    pred = svd_model.predict(sim.X)
    total = 1 - (np.where(m, Y - pred, 0) ** 2).sum() / base
    assert ev["total"] == pytest.approx(total, rel=1e-12)
    sa = mean[None, :, None] + sim.X[:, 0, None, None] * svd_model.B_["sa"][None]
    assert ev["sa"] == pytest.approx(1 - (np.where(m, Y - sa, 0) ** 2).sum() / base)
    per = svd_model.explained_variance(sim.Y_masked, sim.X, per_bin=True)
    assert all(v.shape == (8,) for v in per.values())
    assert 0 < ev["total"] < 1
    assert ev["intercept"] > ev["sb"]
    flat = np.ones_like(sim.Y)
    with pytest.raises(ValidationError, match="over all bins"):
        svd_model.explained_variance(flat, sim.X)
    # Mean zero per neuron, bin 0 at the mean on every trial: no variance there.
    flat = np.zeros_like(Y)
    flat[:, :, 1] = 1.0
    flat[:, :, 2] = -1.0
    with pytest.raises(ValidationError, match=r"in bins \[0, 3, 4, 5, 6, 7\]"):
        svd_model.explained_variance(flat, sim.X, per_bin=True)


# ================================================================= project


def test_project_gls_matches_the_pseudo_inverse(
    svd_model: MTDR, sim: mtdr.SimulatedData
) -> None:
    # Full mask, weighted=False -> pinv(W) (Y - intercept).
    z = svd_model.project(sim.Y, "sa", weighted=False)
    W = svd_model.W_["sa"]
    assert svd_model.intercept_ is not None
    ref = np.einsum("ri,kit->ktr", np.linalg.pinv(W), sim.Y - svd_model.intercept_)
    np.testing.assert_allclose(z, ref, atol=1e-9)
    paper = svd_model.project(sim.Y, "sa", method="paper")
    lam = svd_model.noise_precision_
    ref = np.einsum("ir,i,kit->ktr", W, lam, sim.Y - svd_model.intercept_)
    np.testing.assert_allclose(paper, ref, rtol=1e-12, atol=1e-9)
    # GLS = (W'DW)^-1 applied to the paper projection with a full mask.
    gls = svd_model.project(sim.Y, "sa")
    G = W.T @ (lam[:, None] * W)
    np.testing.assert_allclose(gls, paper @ np.linalg.inv(G).T, rtol=1e-8, atol=1e-9)
    raw = svd_model.project(sim.Y, "sa", remove_intercept=False, weighted=False)
    np.testing.assert_allclose(
        raw, np.einsum("ri,kit->ktr", np.linalg.pinv(W), sim.Y), atol=1e-9
    )


def test_joint_projection_recovers_the_regressors(
    clean: tuple[mtdr.SimulatedData, MTDR, np.ndarray],
) -> None:
    # Noise-free data: the joint coordinates are X[k, p] S_p[t, :] exactly (M49a).
    s, model, Y = clean
    z = model.project(Y, None)
    assert isinstance(z, dict)
    for p, name in enumerate(NAMES):
        expected = s.X[:, p, None, None] * np.array(s.S[name])[None]
        np.testing.assert_allclose(z[name], expected, atol=1e-8)


@settings(deadline=None, max_examples=20)
@given(k=st.integers(0, 239), i=st.integers(0, 39), value=st.floats(-1e3, 1e3))
def test_property_project_ignores_unobserved_entries(
    k: int, i: int, value: float
) -> None:
    s, model = _svd_fitted()
    mask = np.array(s.mask)
    mask[k, i] = False
    Y = np.array(s.Y)
    a = model.project(Y, "sa", mask=mask)
    Y[k, i] = value
    b = model.project(Y, "sa", mask=mask)
    np.testing.assert_array_equal(a, b)


def test_project_singular_and_empty_trials(
    svd_model: MTDR, sim: mtdr.SimulatedData
) -> None:
    mask = np.ones((3, 40), dtype=bool)
    mask[1] = False
    mask[1, 0] = True  # one neuron for a rank-2 basis: singular
    mask[2] = False  # empty: NaN without a warning
    with pytest.warns(ProjectionWarning, match="1 trials"):
        z = svd_model.project(sim.Y[:3], "sa", mask=mask)
    assert np.isfinite(z[0]).all()
    assert np.isnan(z[1:]).all()
    paper = svd_model.project(sim.Y[:3], "sa", mask=mask, method="paper")
    assert np.isfinite(paper[:2]).all()
    assert np.isnan(paper[2]).all()


def test_project_bases(svd_model: MTDR, sim: mtdr.SimulatedData) -> None:
    Q = svd_model.orthogonalize(["choice", "sa"])
    z = svd_model.project(sim.Y, "sa", method="paper", basis=Q)
    assert z.shape == (240, 8, Q["sa"].shape[1])
    empty = svd_model.project(sim.Y, "sa", basis={"sa": np.zeros((40, 0))})
    assert empty.shape == (240, 8, 0)
    joint = svd_model.project(sim.Y, None, basis={"sb": np.zeros((40, 0))})
    assert joint["sb"].shape == (240, 8, 0)
    assert joint["sa"].shape == (240, 8, 2)
    shared = {"sb": svd_model.W_["sa"][:, :1]}
    with pytest.raises(SingularDesignError, match=r"\['sa', 'sb'\]"):
        svd_model.project(sim.Y, None, basis=shared)
    wide = {"sa": np.eye(40)}
    with pytest.raises(SingularDesignError, match="exceeds n_neurons"):
        svd_model.project(sim.Y, None, basis=wide)
    zero = {n: np.zeros((40, 0)) for n in NAMES}
    with pytest.raises(ParameterError, match="every regressor has rank 0"):
        svd_model.project(sim.Y, None, basis=zero)


@pytest.mark.parametrize(
    ("args", "kw", "match"),
    [
        (("nope",), {}, "unknown regressor"),
        ((None,), {"method": "paper"}, "no joint form"),
        (("sa",), {"method": "ols"}, "method"),
        (("sa",), {"basis": [1]}, "mapping"),
        (("sa",), {"basis": {"nope": np.ones((40, 1))}}, "unknown regressor"),
        (("sa",), {"basis": {"sa": np.ones((39, 1))}}, "n_neurons = 40"),
        (("sa",), {"basis": {"sa": np.full((40, 1), np.nan)}}, "finite"),
        (("sa",), {"weighted": 1}, "weighted"),
        (("sa",), {"remove_intercept": None}, "remove_intercept"),
    ],
)
def test_project_errors(
    svd_model: MTDR,
    sim: mtdr.SimulatedData,
    args: tuple[Any, ...],
    kw: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ParameterError, match=match):
        svd_model.project(sim.Y, *args, **kw)


def test_project_rank_zero_needs_a_basis(sim: mtdr.SimulatedData) -> None:
    m = MTDR(ranks=[1, 0, 1], estimator="svd").fit(sim.Y_masked, sim.X)
    with pytest.raises(ParameterError, match="rank 0"):
        m.project(sim.Y, "x1")
    assert m.project(sim.Y, None)["x1"].shape == (240, 8, 0)


# ================================================================= decode


def test_decode_continuous_recovers_noise_free_regressors(
    clean: tuple[mtdr.SimulatedData, MTDR, np.ndarray],
) -> None:
    s, model, Y = clean
    out = model.decode(Y)
    for p, name in enumerate(NAMES):
        np.testing.assert_allclose(out[name], s.X[:, p], atol=1e-8)
    known = model.decode(Y, "sa", known={"sb": s.X[:, 1], "choice": s.X[:, 2]})
    np.testing.assert_allclose(known["sa"], s.X[:, 0], atol=1e-8)


def _normal_matrix(
    model: MTDR, mask: np.ndarray, bins: Any = slice(None)
) -> np.ndarray:
    B = np.stack([model.B_[n][:, bins] for n in model.regressor_names_])
    w = mask * model.noise_precision_[None]
    out: np.ndarray = np.einsum("ki,pit,qit->kpq", w, B, B)
    return out


def test_llr_is_the_frisch_waugh_identity(
    svd_model: MTDR, sim: mtdr.SimulatedData
) -> None:
    # (M48c): LLR = 2 x_c / [(Xi' Lambda Xi)^-1]_cc for levels {-1, +1}.
    out, llr = svd_model.decode(
        sim.Y_masked, levels={"choice": [-1, 1]}, return_llr=True
    )
    free = svd_model.decode(sim.Y_masked)
    G = _normal_matrix(svd_model, sim.mask)
    cc = np.linalg.inv(G)[:, 2, 2]
    np.testing.assert_allclose(llr, 2 * free["choice"] / cc, rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(np.sign(llr), np.sign(free["choice"]))
    assert np.mean(out["choice"] == sim.X[:, 2]) > 0.9
    assert set(out) == {"sa", "sb", "choice"}
    prob = expit(llr)
    assert np.all((prob > 0.5) == (out["choice"] == 1))


@settings(deadline=None, max_examples=20)
@given(a=st.floats(-3, 3), k=st.integers(0, 239))
def test_property_llr_is_affine_in_y(a: float, k: int) -> None:
    s, model = _svd_fitted()
    Y1 = np.array(s.Y)[k : k + 1]
    Y2 = np.array(s.Y)[(k + 7) % 240 : (k + 7) % 240 + 1]
    m = np.ones((1, 40), dtype=bool)

    def llr(Y: np.ndarray) -> float:
        out = model.decode(Y, mask=m, levels={"choice": [-1, 1]}, return_llr=True)
        return float(out[1][0])

    mix = llr(a * Y1 + (1 - a) * Y2)
    assert mix == pytest.approx(a * llr(Y1) + (1 - a) * llr(Y2), rel=1e-9, abs=1e-7)


def test_discrete_enumeration_is_the_brute_force(
    svd_model: MTDR, sim: mtdr.SimulatedData
) -> None:
    # Two discrete unknowns and nothing continuous: the profile log-likelihoods
    # are the plug-in log-likelihood (M46) at every level combination.
    levels = {"sb": [-2.0, 0.0, 2.0], "choice": [-1.0, 1.0]}
    known = {"sa": sim.X[:, 0]}
    values, profile = svd_model.decode(
        sim.Y_masked,
        ["sb", "choice"],
        known=known,
        levels=levels,
        return_log_likelihood=True,
    )
    combos = list(itertools.product(levels["sb"], levels["choice"]))
    assert profile.shape == (240, len(combos))
    for c, (sb, ch) in enumerate(combos):
        X = np.column_stack([sim.X[:, 0], np.full(240, sb), np.full(240, ch)])
        ref = svd_model.log_likelihood(sim.Y_masked, X, per_trial=True)
        np.testing.assert_allclose(profile[:, c], ref, rtol=1e-10)
    best = np.argmax(profile, axis=1)
    np.testing.assert_array_equal(values["sb"], np.array(combos)[best, 0])
    np.testing.assert_array_equal(values["choice"], np.array(combos)[best, 1])


def test_profile_with_continuous_unknowns(
    svd_model: MTDR, sim: mtdr.SimulatedData
) -> None:
    # (M48a): for each level the continuous unknowns are the conditional MLE
    # (M47a), the value given that level as known.
    out, ll = svd_model.decode(
        sim.Y_masked,
        ["sa", "choice"],
        known={"sb": sim.X[:, 1]},
        levels={"choice": [-1, 1]},
        return_log_likelihood=True,
    )
    for c, level in enumerate([-1.0, 1.0]):
        cond, best = svd_model.decode(
            sim.Y_masked,
            "sa",
            known={"sb": sim.X[:, 1], "choice": np.full(240, level)},
            return_log_likelihood=True,
        )
        np.testing.assert_allclose(ll[:, c], best, rtol=1e-10)
        chosen = out["choice"] == level
        np.testing.assert_allclose(out["sa"][chosen], cond["sa"][chosen], rtol=1e-10)


def test_decode_bins(svd_model: MTDR, sim: mtdr.SimulatedData) -> None:
    a = svd_model.decode(sim.Y_masked, bins=3)
    b = svd_model.decode(sim.Y_masked, bins=[3])
    c = svd_model.decode(sim.Y_masked, bins=-5)
    d = svd_model.decode(sim.Y_masked, bins=slice(3, 4))
    for name in NAMES:
        np.testing.assert_array_equal(a[name], b[name])
        np.testing.assert_array_equal(a[name], c[name])
        np.testing.assert_array_equal(a[name], d[name])
    _, ll = svd_model.decode(sim.Y_masked, bins=[0, 1], return_log_likelihood=True)
    assert ll.shape == (240,)


def test_decode_singular_and_empty_trials(
    svd_model: MTDR, sim: mtdr.SimulatedData
) -> None:
    model = pickle.loads(pickle.dumps(svd_model))
    model.B_["sa"] = np.array(model.B_["sa"])
    model.B_["sa"][:, 0] = 0.0  # sa not encoded at bin 0: singular there
    mask = np.array(sim.mask)
    mask[0] = False
    with pytest.warns(DecodingWarning, match=r"239 trials .* bins \[0\]"):
        out, ll = model.decode(
            sim.Y_masked, mask=mask, bins=0, return_log_likelihood=True
        )
    assert all(np.isnan(v).all() for v in out.values())
    assert np.isnan(ll).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = model.decode(sim.Y_masked, mask=mask, bins=[0, 1])
    assert np.isnan(out["sa"][0])
    assert np.isfinite(out["sa"][1:]).all()


def test_decode_is_frame_independent(
    mmle_model: MTDR, raw_model: MTDR, sim: mtdr.SimulatedData
) -> None:
    a = mmle_model.decode(sim.Y_masked, levels={"choice": [-1, 1]})
    b = raw_model.decode(sim.Y_masked, levels={"choice": [-1, 1]})
    for name in NAMES:
        np.testing.assert_allclose(a[name], b[name], rtol=1e-8, atol=1e-10)


def test_decode_rank_zero_regressors_are_ignored(sim: mtdr.SimulatedData) -> None:
    m = MTDR(ranks=[1, 0, 1], estimator="svd").fit(sim.Y_masked, sim.X)
    out = m.decode(sim.Y_masked, ["x0", "x1", "x2"], levels={"x1": [0, 1]})
    assert set(out) == {"x0", "x2"}
    out = m.decode(sim.Y_masked, "x0", known={"x1": sim.X[:, 1], "x2": sim.X[:, 2]})
    assert set(out) == {"x0"}
    with pytest.raises(ParameterError, match="rank 0: nothing to decode"):
        m.decode(sim.Y_masked, "x1", known={"x0": sim.X[:, 0], "x2": sim.X[:, 2]})


def _tiny(ranks: tuple[int, int] = (1, 1)) -> MTDR:
    """Two neurons, one bin, unit precisions, zero intercept.

    `B_stim = (1, 1)`, `B_choice = (0, 1)`, so every likelihood is a hand sum.
    """
    s = simulate(n_neurons=2, n_bins=1, n_trials=40, ranks=[1, 1], seed=0)
    m = MTDR(ranks=list(ranks), estimator="svd").fit(
        s.Y, s.X, regressor_names=["stim", "choice"]
    )
    m.B_ = {"stim": np.array([[1.0], [1.0]]), "choice": np.array([[0.0], [1.0]])}
    if ranks[1] == 0:
        m.B_["choice"][:] = 0.0
    m.intercept_ = np.zeros((2, 1))
    m.noise_precision_ = np.ones(2)
    return m


@pytest.mark.parametrize("offset", [2.0, 1e8])
@pytest.mark.parametrize("profile", [False, True])
def test_decode_llr_by_hand_at_large_offsets(offset: float, profile: bool) -> None:
    # Y = (a, a + 1): with the stimulus known at a, the RSS is 4 under choice -1
    # and 0 under +1 (LLR 2); profiled, the best stimuli are a + 1 and a, RSS 2
    # and 0 (LLR 1, (M48c)). Every number is exact in float64; evaluating (M46)
    # by expanding the quadratic cancels at a = 1e8.
    m = _tiny()
    Y = np.array([[[offset], [offset + 1.0]]])
    known = None if profile else {"stim": np.array([offset])}
    out, ll, llr = m.decode(
        Y,
        ["stim", "choice"] if profile else "choice",
        known=known,
        levels={"choice": [-1, 1]},
        return_log_likelihood=True,
        return_llr=True,
    )
    np.testing.assert_allclose(llr, [1.0 if profile else 2.0], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(out["choice"], [1.0])
    rss = [2.0, 0.0] if profile else [4.0, 0.0]
    np.testing.assert_allclose(ll[0], [-np.log(2 * np.pi) - r / 2 for r in rss])
    if profile:
        np.testing.assert_array_equal(out["stim"], [offset])


@pytest.mark.parametrize("off", [0.0, 1e4])
def test_decode_likelihoods_are_residual_sums_for_an_uncentred_regressor(
    off: float,
) -> None:
    # At the demo's scale, a stimulus column shifted by `off` (an uncentred
    # coding). Evaluating (M46) by expanding the quadratic gives an LLR error
    # that grows as off**2 * eps (1.5e-4 at 1e4); the profile log-likelihoods
    # and the LLR must equal a brute-force residual computation of (M46).
    s = simulate(
        n_neurons=100, n_bins=15, n_trials=400, ranks=[2, 1, 3], drop_prob=0.3, seed=1
    )
    X = np.array(s.X)
    X[:, 0] += off
    m = MTDR(ranks=[2, 1, 3], estimator="svd").fit(
        s.Y, X, s.mask, regressor_names=["a", "b", "c"]
    )
    lo, hi = np.unique(X[:, 2])[[0, -1]]
    _, ll, llr = m.decode(
        s.Y,
        "c",
        s.mask,
        known={"a": X[:, 0], "b": X[:, 1]},
        levels={"c": [lo, hi]},
        return_log_likelihood=True,
        return_llr=True,
    )
    for j, level in enumerate([lo, hi]):
        Xc = np.column_stack([X[:, 0], X[:, 1], np.full(400, level)])
        direct = m.log_likelihood(s.Y, Xc, s.mask, per_trial=True)
        np.testing.assert_allclose(ll[:, j], direct, rtol=1e-12)
    direct_llr = np.diff(ll, axis=1)[:, 0]
    pred = {
        level: np.einsum(
            "kp,pit->kit",
            np.column_stack([X[:, 0], X[:, 1], np.full(400, level)]),
            np.stack(list(m.B_.values())),
        )
        for level in (lo, hi)
    }
    assert m.intercept_ is not None
    resid = {
        level: np.where(s.mask[:, :, None], s.Y - m.intercept_[None] - p, 0.0)
        for level, p in pred.items()
    }
    brute = -0.5 * np.einsum(
        "i,kit->k", m.noise_precision_, resid[hi] ** 2 - resid[lo] ** 2
    )
    scale = np.abs(brute).max()
    np.testing.assert_allclose(llr, brute, rtol=1e-9, atol=1e-9 * scale)
    np.testing.assert_allclose(direct_llr, brute, rtol=1e-9, atol=1e-9 * scale)


def test_decode_residual_blocks_do_not_change_the_result(
    svd_model: MTDR, sim: mtdr.SimulatedData, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The residuals are accumulated in blocks of trials; one trial per block
    # gives the same numbers as one block.
    kw: dict[str, Any] = {
        "regressor": ["sa", "choice"],
        "known": {"sb": sim.X[:, 1]},
        "levels": {"choice": [-1, 1]},
        "return_log_likelihood": True,
    }
    whole = svd_model.decode(sim.Y_masked, **kw)
    monkeypatch.setattr(mtdr.model, "_DECODE_BLOCK", 1)
    split = svd_model.decode(sim.Y_masked, **kw)
    np.testing.assert_array_equal(split[1], whole[1])
    for name in ("sa", "choice"):
        np.testing.assert_array_equal(split[0][name], whole[0][name])


def test_decode_enforces_the_combination_limit_before_enumerating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The 64-combination limit is checked before the Cartesian product is built.
    m = _tiny()

    def bomb(*args: Any) -> Any:
        raise RuntimeError("the product was built before its size guard")

    monkeypatch.setattr(itertools, "product", bomb)
    with pytest.raises(ParameterError, match="65 level combinations; at most 64"):
        m.decode(
            np.ones((1, 2, 1)),
            "choice",
            known={"stim": np.zeros(1)},
            levels={"choice": range(65)},
        )


def test_decode_ignores_known_values_of_rank_zero_regressors() -> None:
    # `known` entries of rank-0 regressors are ignored, values unchecked, also
    # when the name is decoded.
    m = _tiny((1, 0))
    Y = np.ones((1, 2, 1))
    assert set(m.decode(Y, known={"choice": "ignored"})) == {"stim"}
    out = m.decode(Y, ["stim", "choice"], known={"choice": np.array([0.0])})
    assert set(out) == {"stim"}
    with pytest.raises(ParameterError, match="unknown regressor 'zz'"):
        m.decode(Y, known={"zz": "ignored"})
    with pytest.raises(ParameterError, match="rank 0: nothing to decode"):
        m.decode(Y, "choice", known={"stim": np.zeros(1), "choice": np.zeros(1)})


@pytest.fixture(scope="module")
def tolerances() -> tuple[MTDR, mtdr.SimulatedData]:
    """A small fit for the tests of decision-bearing tolerances."""
    s = simulate(
        n_neurons=30,
        n_bins=6,
        n_trials=120,
        ranks={"sa": 2, "c": 1},
        levels=[[-2, -1, 0, 1, 2], [-1, 1]],
        drop_prob=0.2,
        seed=5,
    )
    model = MTDR(ranks=dict(s.ranks), estimator="svd").fit(
        s.Y_masked, s.X, regressor_names=s.regressor_names
    )
    return model, s


def test_decode_profile_loglik_on_a_bin_subset_is_the_plug_in(
    tolerances: tuple[MTDR, mtdr.SimulatedData],
) -> None:
    # The normalising constant counts |bins|, not n_bins.
    model, s = tolerances
    bins = [1, 3]
    _, ll = model.decode(
        s.Y_masked,
        ["sa", "c"],
        levels={"c": [-1, 1]},
        bins=bins,
        return_log_likelihood=True,
    )
    lam, b = model.noise_precision_, model.intercept_
    assert b is not None
    k, m = 0, s.mask[0]
    for j, c in enumerate([-1.0, 1.0]):
        cond, best = model.decode(
            s.Y_masked[[k]],
            "sa",
            known={"c": np.array([c])},
            bins=bins,
            return_log_likelihood=True,
        )
        pred = b + cond["sa"][0] * model.B_["sa"] + c * model.B_["c"]
        r = (s.Y[k] - pred)[:, bins]
        brute = np.sum(
            m[:, None]
            * (
                0.5 * np.log(lam)[:, None]
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * lam[:, None] * r * r
            )
        )
        assert ll[k, j] == pytest.approx(brute, rel=1e-12)
        assert best[0] == pytest.approx(brute, rel=1e-12)


def test_decode_ill_conditioned_system_is_nan(
    tolerances: tuple[MTDR, mtdr.SimulatedData],
) -> None:
    # The decoder's threshold is a condition number above eps**-1/2 (6.7e7),
    # not eps**-1: cond(G) ~ 1e10 at bin 0 is NaN.
    model, s = tolerances
    m = pickle.loads(pickle.dumps(model))
    m.B_["sa"] = np.array(m.B_["sa"])
    m.B_["sa"][:, 0] *= 1e-5
    with pytest.warns(DecodingWarning):
        out = m.decode(s.Y_masked, ["sa", "c"], bins=0)
    assert np.isnan(out["sa"]).all()


def test_project_keeps_a_well_posed_but_poorly_scaled_basis(
    tolerances: tuple[MTDR, mtdr.SimulatedData],
) -> None:
    # The GLS projection's singularity test is lambda_min <= eps * lambda_max,
    # so a Gram with eigenvalue ratio about 1e-12 is solved, without a warning.
    model, s = tolerances
    rng = np.random.default_rng(0)
    U = rng.normal(size=(30, 2))
    U[:, 1] = U[:, 0] + 1e-6 * rng.normal(size=30)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProjectionWarning)
        z = model.project(s.Y_masked, "sa", basis={"sa": U})
    assert np.isfinite(z[s.mask.any(axis=1)]).all()


def test_orthogonalize_drops_a_nearly_shared_direction_only_at_sqrt_eps(
    tolerances: tuple[MTDR, mtdr.SimulatedData],
) -> None:
    # A direction 1e-6 outside the earlier span is kept (1e-6 > sqrt(eps) =
    # 1.5e-8); an exactly shared one is dropped.
    model, _ = tolerances
    m = pickle.loads(pickle.dumps(model))
    U = m.W_["sa"]
    rng = np.random.default_rng(1)
    v = U[:, 0] + 1e-6 * rng.normal(size=U.shape[0])
    m.B_["c"] = np.outer(v / np.linalg.norm(v), np.ones(m.n_bins_))
    assert m.orthogonalize(["sa", "c"])["c"].shape[1] == 1
    m.B_["c"] = np.outer(U[:, 0], np.ones(m.n_bins_))
    assert m.orthogonalize(["sa", "c"])["c"].shape[1] == 0


@pytest.mark.parametrize(
    ("kw", "error", "match"),
    [
        ({"regressor": "sa", "known": {"sa": np.zeros(240)}}, ParameterError, "both"),
        ({"regressor": "sa"}, ParameterError, "neither decoded nor known"),
        ({"regressor": ["sa", "sa"]}, ParameterError, "twice"),
        ({"regressor": ["sa", 1]}, ParameterError, "unknown regressor"),
        ({"regressor": 3}, ParameterError, "a name, a sequence"),
        ({"levels": {"zz": [1]}}, ParameterError, "not decoded"),
        ({"levels": [1]}, ParameterError, "mapping"),
        ({"levels": {"sa": []}}, ParameterError, "finite real"),
        ({"levels": {"sa": [np.nan]}}, ParameterError, "finite real"),
        (
            {"levels": {"sa": range(5), "sb": range(5), "choice": range(3)}},
            ParameterError,
            "75 level combinations",
        ),
        ({"return_llr": True}, ParameterError, "return_llr"),
        ({"levels": {"choice": [-1, 0, 1]}, "return_llr": True}, ParameterError, "two"),
        ({"bins": 8}, ParameterError, "index the 8"),
        ({"bins": 1.5}, ParameterError, "bins must be"),
        ({"bins": slice(5, 5)}, ParameterError, "no time bin"),
        # Invalid slices are ParameterErrors, not NumPy's raw errors.
        ({"bins": slice(None, None, 0)}, ParameterError, "non-zero step"),
        ({"bins": slice(0.5, None)}, ParameterError, "slice"),
        ({"bins": slice(0, "a")}, ParameterError, "slice"),
        ({"known": [1]}, ParameterError, "mapping"),
        ({"known": {"zz": np.zeros(240)}}, ParameterError, "unknown regressor"),
        (
            {"regressor": "sa", "known": {"sb": np.zeros(3), "choice": np.zeros(240)}},
            ValidationError,
            "shape",
        ),
        (
            {
                "regressor": "sa",
                "known": {"sb": np.full(240, np.inf), "choice": np.zeros(240)},
            },
            ValidationError,
            "finite",
        ),
        ({"return_llr": 1}, ParameterError, "return_llr"),
    ],
)
def test_decode_errors(
    svd_model: MTDR,
    sim: mtdr.SimulatedData,
    kw: dict[str, Any],
    error: type,
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        svd_model.decode(sim.Y_masked, **kw)


# ================================================================= summaries


def test_orthogonalize(svd_model: MTDR) -> None:
    Q = svd_model.orthogonalize(["choice", "sa", "sb"])
    assert [q.shape[1] for q in Q.values()] == [2, 2, 1]
    stacked = np.concatenate(list(Q.values()), axis=1)
    np.testing.assert_allclose(stacked.T @ stacked, np.eye(5), atol=1e-12)
    U = mtdr.model._pc_basis(svd_model.B_["choice"], 2)[0]
    np.testing.assert_allclose(Q["choice"], U, atol=1e-14)  # first block unchanged
    assert list(svd_model.orthogonalize(["sb"])) == ["sb"]


def test_orthogonalize_drops_shared_directions(svd_model: MTDR) -> None:
    model = pickle.loads(pickle.dumps(svd_model))
    U, _, V = mtdr.model._pc_basis(model.B_["sa"], 2)
    model.B_["sb"] = np.outer(U[:, 1], V[:, 0])  # inside sa's subspace
    Q = model.orthogonalize(["sa", "sb", "choice"])
    assert Q["sb"].shape == (40, 0)
    assert Q["choice"].shape == (40, 2)
    rank0 = MTDR(ranks=[1, 0, 1], estimator="svd")
    rank0.__dict__.update(model.__dict__)
    rank0.ranks_ = {"sa": 2, "sb": 0, "choice": 2}
    assert list(rank0.orthogonalize(["sb", "sa"])) == ["sa"]
    for bad, match in [
        (["sa", "sa"], "repeated"),
        (["zz"], "unknown"),
        (3, "sequence"),
    ]:
        with pytest.raises(ParameterError, match=match):
            model.orthogonalize(bad)


def test_subspace_angles(
    mmle_model: MTDR, raw_model: MTDR, sim: mtdr.SimulatedData
) -> None:
    a = mmle_model.subspace_angles("sa", "choice")
    b = raw_model.subspace_angles("sa", "choice")
    np.testing.assert_allclose(a, b, atol=1e-8)  # frame-independent
    assert a.shape == (2,)
    assert np.all(np.diff(a) >= 0)
    ref = np.sort(
        scipy.linalg.subspace_angles(mmle_model.W_["sa"], mmle_model.W_["choice"])
    )
    np.testing.assert_allclose(a, ref, atol=1e-12)
    np.testing.assert_allclose(mmle_model.subspace_angles("sa", "sa"), 0, atol=1e-7)
    m0 = MTDR(ranks=[1, 0, 1], estimator="svd").fit(sim.Y_masked, sim.X)
    assert m0.subspace_angles("x0", "x1").shape == (0,)
    with pytest.raises(ParameterError, match="unknown"):
        mmle_model.subspace_angles("sa", "zz")


def test_to_xarray(mmle_model: MTDR) -> None:
    pytest.importorskip("xarray")
    ds = mmle_model.to_xarray()
    assert dict(ds.sizes) == {"regressor": 3, "neuron": 40, "time": 8, "component": 2}
    np.testing.assert_array_equal(
        ds["B"].sel(regressor="sb").values, mmle_model.B_["sb"]
    )
    assert np.isnan(ds["W"].sel(regressor="sb").values[:, 1]).all()
    np.testing.assert_array_equal(
        ds["W"].sel(regressor="sa").values, mmle_model.W_["sa"]
    )
    assert ds["rank"].values.tolist() == [2, 1, 2]
    assert ds.attrs["estimator"] == "mmle"
    assert ds.attrs["aic"] == mmle_model.aic_
    assert ds.attrs["mtdr_version"] == mtdr.__version__
    assert "intercept" in ds


def test_to_xarray_without_intercept_and_without_xarray(
    sim: mtdr.SimulatedData, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("xarray")
    Y = np.array(sim.Y) - np.array(sim.intercept)[None]
    m = MTDR(ranks=[1, 1, 1], estimator="svd", condition_independent=False).fit(
        Y, sim.X
    )
    assert "intercept" not in m.to_xarray()
    monkeypatch.setitem(sys.modules, "xarray", None)
    with pytest.raises(
        ImportError, match=r"pip install 'mtdr\[xarray\]' \(from a clone"
    ):
        m.to_xarray()


def test_decode_without_an_intercept(sim: mtdr.SimulatedData) -> None:
    Y = np.array(sim.Y) - np.array(sim.intercept)[None]
    model = MTDR(ranks=[2, 1, 2], estimator="svd", condition_independent=False).fit(
        Y, sim.X, sim.mask
    )
    out = model.decode(Y, mask=sim.mask)
    assert np.corrcoef(out["x0"], sim.X[:, 0])[0, 1] > 0.9
