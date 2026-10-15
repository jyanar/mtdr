"""`MTDR(groups=...)`: the estimator with regressor groups, `docs/model.md` § 11.

The construction- and fit-time rules of the argument, the fitted attributes
(latent quantities keyed by group, column quantities by column), every method
and plot on a grouped fit, pickles of grouped and of pre-groups models, the
grouped rank search, recovery of a shared subspace, coding invariance, and an
enumeration of the public API that fails when a callable has no grouped
exercise. Tolerances are declared next to each assertion.
"""

from __future__ import annotations

import inspect
import pickle
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import scipy.linalg
from numpy.typing import NDArray

import mtdr
import mtdr.aic
import mtdr.data
import mtdr.mmle
import mtdr.model
import mtdr.plot
import mtdr.rank_search
import mtdr.simulation
import mtdr.stats
import mtdr.svd_fit
import mtdr.xarray_io
from mtdr import (
    MTDR,
    ParameterError,
    SingularDesignError,
    ValidationError,
    categorical_design,
    simulate,
)
from mtdr.errors import ConvergenceWarning, DesignWarning
from mtdr.stats import sufficient_statistics

FloatArray = NDArray[np.float64]
LEVELS = [3, 6, 12, 24, 48]
#: The unequal level probabilities of the categorical simulations.
LEVEL_P = [0.3, 0.25, 0.2, 0.15, 0.1]


def categorical_sim(
    seed: int,
    n: int = 40,
    T: int = 6,
    N: int = 400,
    rank: int = 2,
    choice_rank: int = 1,
    drop_prob: float = 0.2,
    noise: float = 2.0,
) -> tuple[mtdr.SimulatedData, mtdr.CategoricalDesign, NDArray[Any]]:
    """Five Sa levels (unequal counts) as one group, and a binary choice."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(LEVELS, N, p=LEVEL_P)
    labels[:5] = LEVELS
    design = categorical_design(labels, levels=LEVELS, name="sa")
    choice = rng.choice([-1.0, 1.0], N)
    X = np.column_stack([design.X, choice])
    sim = simulate(
        n_neurons=n,
        n_bins=T,
        n_trials=N,
        ranks={"sa": rank, "choice": choice_rank},
        regressor_names=[*design.columns, "choice"],
        groups=design.groups,
        X=X,
        drop_prob=drop_prob,
        noise_precision=noise,
        seed=seed,
    )
    return sim, design, labels


NAMES = ["sa_1", "sa_2", "sa_3", "sa_4", "choice"]
GROUPS = {"sa": ["sa_1", "sa_2", "sa_3", "sa_4"]}


@pytest.fixture(scope="module")
def data() -> tuple[mtdr.SimulatedData, mtdr.CategoricalDesign, NDArray[Any]]:
    return categorical_sim(0)


@pytest.fixture(scope="module")
def model(data: Any) -> MTDR:
    sim = data[0]
    return _fit(sim, ranks={"sa": 2, "choice": 1}, groups=GROUPS)


@pytest.fixture(scope="module")
def svd_model(data: Any) -> MTDR:
    sim = data[0]
    return _fit(sim, ranks={"sa": 2, "choice": 1}, groups=GROUPS, estimator="svd")


@pytest.fixture(scope="module")
def svd_search(data: Any) -> MTDR:
    sim = data[0]
    return _fit(sim, ranks="aic", groups=GROUPS, max_rank=2, estimator="svd")


def _labels(design: mtdr.CategoricalDesign, values: Any) -> NDArray[Any]:
    """The level of each decoded row (decode returns the candidate rows exactly)."""
    rows = np.column_stack([values[c] for c in design.columns])
    hits = (rows[:, None, :] == design.basis[None]).all(axis=2)
    assert hits.any(axis=1).all()
    return design.levels[np.argmax(hits, axis=1)]


def _fit(sim: mtdr.SimulatedData, **kw: Any) -> MTDR:
    return MTDR(**kw).fit(
        sim.Y_masked, sim.X, regressor_names=list(sim.regressor_names)
    )


# ================================================================= the argument


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"groups": [["a"]]}, "groups must be a mapping"),
        ({"groups": {"": ["a"]}}, "non-empty strings"),
        ({"groups": {1: ["a"]}}, "non-empty strings"),
        ({"groups": {"total": ["a"]}}, "reserved"),
        ({"groups": {"g": "ab"}}, r"is the string 'ab'; give a sequence"),
        ({"groups": {"g": []}}, "non-empty sequence of column names"),
        ({"groups": {"g": 3}}, "non-empty sequence of column names"),
        ({"groups": {"g": ["a", 1]}}, "holds 1, which is not a column name"),
        ({"groups": {"g": ["a", "a"]}}, "appears twice in it"),
        ({"groups": {"g": ["a", "b"], "h": ["b"]}}, "in 'g' too"),
        ({"groups": GROUPS, "ranks": [2, 1]}, "pass ranks as a mapping"),
        ({"groups": GROUPS, "ranks": {"sa_1": 2}}, r"give ranks\['sa'\]"),
        ({"groups": GROUPS, "rank_search_init": [1, 1]}, "pass rank_search_init"),
        ({"groups": GROUPS, "rank_search_init": {"sa_2": 1}}, "a column of group"),
        ({"rank_search_init": {"a": 0}}, "map regressor .* positive ints"),
        ({"rank_search_init": {1: 1}}, "map regressor .* positive ints"),
    ],
)
def test_construction_checks(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ParameterError, match=match):
        MTDR(**kw)


def test_groups_are_a_parameter() -> None:
    model = MTDR(ranks={"sa": 2, "choice": 1}, groups=GROUPS)
    assert model.get_params()["groups"] is GROUPS
    assert repr(model).endswith("groups={'sa': ['sa_1', 'sa_2', 'sa_3', 'sa_4']})")
    # set_params checks the combination and rolls back.
    with pytest.raises(ParameterError, match="pass ranks as a mapping"):
        MTDR(ranks=[1, 2]).set_params(groups=GROUPS)
    plain = MTDR(ranks=[1, 2])
    with pytest.raises(ParameterError):
        plain.set_params(groups=GROUPS)
    assert plain.groups is None
    rebuilt = MTDR(**model.get_params())
    assert rebuilt.get_params() == model.get_params()


def test_clone_with_groups() -> None:  # pragma: no cover - optional
    base = pytest.importorskip("sklearn.base")
    model = MTDR(ranks={"sa": 2, "choice": 1}, groups=GROUPS)
    clone = base.clone(model)
    assert clone.get_params() == model.get_params()
    assert clone.get_params()["groups"] is not GROUPS


@pytest.mark.parametrize(
    ("kw", "names", "match"),
    [
        ({"groups": {"sa": ["sa_1", "sa_9"]}}, NAMES, r"names \['sa_9'\], which are"),
        ({"groups": {"choice": ["sa_1", "sa_2"]}}, NAMES, "no group lists"),
        (
            {"groups": {"sa_3": ["sa_1", "sa_2"], "g": ["sa_3", "sa_4"]}},
            NAMES,
            "is a column of group 'g'",
        ),
        ({"groups": GROUPS, "ranks": {"sa": 2}}, NAMES, "name exactly the groups"),
        ({"groups": GROUPS, "ranks": {"sa": 25, "choice": 1}}, NAMES, "cap of group"),
        ({"groups": GROUPS, "max_rank": 25}, NAMES, "largest group cap"),
        ({"groups": GROUPS, "rank_search_init": {"sa": 1}}, NAMES, "name exactly the"),
        (
            {
                "groups": GROUPS,
                "rank_search_init": {"sa": 5, "choice": 1},
                "max_rank": 4,
            },
            NAMES,
            r"rank_search_init\['sa'\] is 5, above the cap",
        ),
        ({"rank_search_init": {"x0": 1}}, None, "name exactly the regressors"),
    ],
)
def test_fit_checks(
    data: Any, kw: dict[str, Any], names: list[str] | None, match: str
) -> None:
    sim = data[0]
    with pytest.raises(ParameterError, match=match):
        MTDR(**kw).fit(sim.Y_masked, sim.X, regressor_names=names)


def test_resolve_groups_orders_by_column(data: Any) -> None:
    # Insertion order of the mapping is irrelevant; members follow the columns.
    swapped = {"sa": ["sa_4", "sa_2", "sa_3", "sa_1"]}
    names = ["choice", *NAMES[:4]]
    got = mtdr.resolve_groups(names, swapped)
    assert list(got) == ["choice", "sa"]
    assert got["sa"] == ("sa_1", "sa_2", "sa_3", "sa_4")
    later = mtdr.resolve_groups(["a", "b", "c", "d"], {"x": ["d", "b"], "y": ["c"]})
    assert list(later) == ["a", "x", "y"]
    # A group may be named after one of its own columns.
    own = mtdr.resolve_groups(NAMES, {"sa_1": NAMES[:4]})
    assert list(own) == ["sa_1", "choice"]
    with pytest.raises(ParameterError, match="regressor_names must be a sequence"):
        mtdr.resolve_groups("abc", None)
    with pytest.raises(ParameterError, match="unique"):
        mtdr.resolve_groups(["a", "a"], None)


def test_a_group_named_after_its_own_column(data: Any) -> None:
    sim = data[0]
    model = _fit(sim, ranks={"sa_1": 2, "choice": 1}, groups={"sa_1": NAMES[:4]})
    assert model.group_names_ == ("sa_1", "choice")
    assert model.W_["sa_1"].shape == (40, 2)
    # The name is the group in decode's regressor and levels.
    out = model.decode(sim.Y_masked, "sa_1", known={"choice": sim.X[:, 4]})
    assert sorted(out) == NAMES[:4]


# ================================================================= the default path


def test_explicit_singletons_are_the_default_bit_for_bit(data: Any) -> None:
    # groups=None and every column its own explicit group: identical outputs.
    sim = data[0]
    singles = {name: [name] for name in NAMES}
    for kw in (
        {"ranks": {"sa_1": 1, "sa_2": 1, "sa_3": 0, "sa_4": 1, "choice": 1}},
        {
            "ranks": {"sa_1": 1, "sa_2": 1, "sa_3": 1, "sa_4": 1, "choice": 1},
            "estimator": "svd",
        },
        {"ranks": "aic", "max_rank": 2, "estimator": "svd"},
    ):
        a = _fit(sim, **kw)
        b = _fit(sim, groups=singles, **kw)
        assert a.groups_ == b.groups_
        assert a.group_names_ == b.group_names_ == tuple(NAMES)
        # W, S, B, ranks, intercept and precisions, NaN padding included.
        assert a.to_xarray().identical(b.to_xarray())
        if a.weight_posterior_cov_ is not None:
            np.testing.assert_array_equal(
                a.weight_posterior_cov_, b.weight_posterior_cov_
            )
        assert (a.log_likelihood_, a.aic_, a.ranks_) == (
            b.log_likelihood_,
            b.aic_,
            b.ranks_,
        )
        np.testing.assert_array_equal(
            a.project(sim.Y_masked, "choice"), b.project(sim.Y_masked, "choice")
        )
        assert a.explained_variance(sim.Y_masked, sim.X) == b.explained_variance(
            sim.Y_masked, sim.X
        )
    assert a.groups_ == {name: (name,) for name in NAMES}


def test_a_renamed_single_column_group(data: Any) -> None:
    # A one-column group with another name is the ungrouped fit, keyed by group.
    sim = data[0]
    ranks = {"x": 1, "sa_2": 1, "sa_3": 1, "sa_4": 1, "choice": 1}
    a = _fit(sim, ranks=[1] * 5)
    b = _fit(sim, ranks=ranks, groups={"x": ["sa_1"]})
    np.testing.assert_array_equal(a.W_["sa_1"], b.W_["x"])
    np.testing.assert_array_equal(a.B_["sa_1"], b.B_["sa_1"])
    assert b.group_names_ == ("x", "sa_2", "sa_3", "sa_4", "choice")
    assert "group" in b.to_xarray().dims
    with pytest.raises(ParameterError, match="column of group 'x'"):
        b.project(sim.Y_masked, "sa_1")


# ================================================================= fitted attributes


def test_fitted_attributes_are_keyed_by_group_and_column(model: MTDR) -> None:
    assert model.groups_ == {"sa": tuple(NAMES[:4]), "choice": ("choice",)}
    assert model.group_names_ == ("sa", "choice")
    assert model.regressor_names_ == tuple(NAMES)
    assert model.ranks_ == {"sa": 2, "choice": 1}
    assert model.total_rank_ == 3
    assert {k: v.shape for k, v in model.W_.items()} == {
        "sa": (40, 2),
        "choice": (40, 1),
    }
    assert {k: v.shape for k, v in model.S_.items()} == {
        **dict.fromkeys(NAMES[:4], (6, 2)),
        "choice": (6, 1),
    }
    assert model.weight_posterior_cov_ is not None
    assert model.weight_posterior_cov_.shape == (40, 3, 3)
    for c in NAMES[:4]:
        np.testing.assert_allclose(model.B_[c], model.W_["sa"] @ model.S_[c].T)
    assert model.n_parameters_ == mtdr.aic.n_parameters_mmle(
        [2, 1], 40, 6, groups=[[0, 1, 2, 3], [4]]
    )
    # The canonical frame: W_ orthonormal per group; S_ carries the scale.
    np.testing.assert_allclose(model.W_["sa"].T @ model.W_["sa"], np.eye(2), atol=1e-12)
    assert model.converged_


def test_the_canonical_frame_and_the_covariance_rotation(data: Any) -> None:
    sim = data[0]
    kw: dict[str, Any] = {"ranks": {"sa": 2, "choice": 1}, "groups": GROUPS}
    raw = _fit(sim, canonicalize=False, **kw)
    canon = _fit(sim, **kw)
    for c in NAMES:
        np.testing.assert_array_equal(raw.B_[c], canon.B_[c])
    assert raw.weight_posterior_cov_ is not None
    assert canon.weight_posterior_cov_ is not None
    np.testing.assert_array_equal(
        raw.weight_posterior_cov_,
        raw._fit_object.W_cov,  # type: ignore[union-attr]
    )
    R = scipy.linalg.block_diag(
        *(
            np.linalg.lstsq(raw.W_[g], canon.W_[g], rcond=None)[0]
            for g in ("sa", "choice")
        )
    )
    np.testing.assert_allclose(
        canon.weight_posterior_cov_, R.T @ raw.weight_posterior_cov_ @ R, atol=1e-12
    )
    # For an orthonormal coding the group frame is that of the level responses.
    design = data[1]
    levels = design.level_responses(canon)
    joined = np.concatenate(list(levels), axis=1)
    U, s, _ = np.linalg.svd(joined, full_matrices=False)
    U = U[:, :2] * np.sign(U[np.argmax(np.abs(U[:, :2]), axis=0), [0, 1]])
    np.testing.assert_allclose(canon.W_["sa"], U, atol=1e-10)
    norms = np.sqrt(sum(np.sum(canon.S_[c] ** 2, axis=0) for c in NAMES[:4]))
    np.testing.assert_allclose(norms, s[:2], rtol=1e-10)


# ================================================================= methods


def test_predict_likelihood_score_and_aic(model: MTDR, data: Any) -> None:
    sim = data[0]
    want = model.intercept_ + np.einsum(
        "kp,pit->kit", sim.X, np.stack([model.B_[c] for c in NAMES])
    )
    np.testing.assert_allclose(model.predict(sim.X), want)
    assert model.score(sim.Y_masked, sim.X) == model.log_likelihood(sim.Y_masked, sim.X)
    assert model.aic() == model.aic_


def test_project_by_group(model: MTDR, data: Any) -> None:
    sim = data[0]
    z = model.project(sim.Y_masked, "sa")
    assert z.shape == (400, 6, 2)
    joint = model.project(sim.Y_masked, None)
    # The shared group enters the joint basis once: width 2 + 1, not 4 * 2 + 1.
    assert {k: v.shape for k, v in joint.items()} == {
        "sa": (400, 6, 2),
        "choice": (400, 6, 1),
    }
    assert np.isfinite(joint["sa"]).all()
    basis = model.orthogonalize(["choice", "sa"])
    zo = model.project(sim.Y_masked, "sa", basis=basis)
    assert zo.shape[2] == basis["sa"].shape[1]
    with pytest.raises(ParameterError, match=r"regressor 'sa_2': it is a column of"):
        model.project(sim.Y_masked, "sa_2")
    with pytest.raises(ParameterError, match="basis key 'sa_1'"):
        model.project(sim.Y_masked, "sa", basis={"sa_1": basis["sa"]})
    with pytest.raises(ParameterError, match=r"unknown group 'nope'; groups"):
        model.project(sim.Y_masked, "nope")
    shared = {"sa": model.W_["choice"], "choice": model.W_["choice"]}
    with pytest.raises(SingularDesignError, match="groups involved"):
        model.project(sim.Y_masked, None, basis=shared)


def test_a_rank_zero_group(data: Any) -> None:
    sim = data[0]
    model = _fit(sim, ranks={"sa": 0, "choice": 1}, groups=GROUPS)
    assert model.W_["sa"].shape == (40, 0)
    assert all(model.S_[c].shape == (6, 0) for c in NAMES[:4])
    assert not any(model.B_[c].any() for c in NAMES[:4])
    with pytest.raises(ParameterError, match="group 'sa' has rank 0"):
        model.project(sim.Y_masked, "sa")
    # A rank-0 group's levels are ignored, as a rank-0 column's are.
    out = model.decode(sim.Y_masked, "choice", levels={"sa": [[0, 0, 0, 0]]})
    assert sorted(out) == ["choice"]
    assert model.subspace_angles("sa", "choice").size == 0


@pytest.mark.parametrize("estimator", ["mmle", "svd"])
def test_rank_limits(estimator: str) -> None:
    # Rank 0 everywhere (intercept only), a rank above T within min(n, |g| T),
    # a rank at the cap, and above it.
    sim, _, _ = categorical_sim(1, n=12, T=2, N=200, rank=3)
    kw: dict[str, Any] = {"groups": GROUPS, "estimator": estimator}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        none = _fit(sim, ranks={"sa": 0, "choice": 0}, **kw)
        above_t = _fit(sim, ranks={"sa": 3, "choice": 1}, **kw)
        at_cap = _fit(sim, ranks={"sa": 8, "choice": 2}, **kw)
    assert none.total_rank_ == 0
    assert np.isfinite(none.log_likelihood_)
    assert above_t.W_["sa"].shape == (12, 3)
    assert above_t.S_["sa_1"].shape == (2, 3)
    assert at_cap.ranks_ == {"sa": 8, "choice": 2}
    assert at_cap.S_["sa_2"].shape == (2, 8)
    with pytest.raises(ParameterError, match=r"is 9, above the cap of group 'sa'"):
        _fit(sim, ranks={"sa": 9, "choice": 1}, **kw)
    with pytest.raises(ParameterError, match=r"ranks\['choice'\] is 3, above the cap"):
        _fit(sim, ranks={"sa": 1, "choice": 3}, **kw)
    with pytest.raises(ParameterError, match="above the largest group cap"):
        _fit(sim, ranks="aic", max_rank=9, **kw)


def test_decode_group_candidate_rows(model: MTDR, data: Any) -> None:
    # The group's rows are enumerated as one unit; labels come back exactly.
    sim, design, labels = data
    out = model.decode(
        sim.Y_masked, None, levels={"sa": design.rows(), "choice": [-1, 1]}
    )
    assert sorted(out) == sorted(NAMES)
    assert np.mean(_labels(design, out) == labels) > 0.95
    assert np.mean(out["choice"] == sim.X[:, 4]) > 0.95
    out2, profile = model.decode(
        sim.Y_masked,
        "sa",
        known={"choice": sim.X[:, 4]},
        levels={"sa": design.rows()},
        return_log_likelihood=True,
    )
    assert profile.shape == (400, 5)
    best = np.argmax(profile, axis=1)
    np.testing.assert_array_equal(_labels(design, out2), np.array(LEVELS)[best])
    # The group's columns can also be decoded as continuous values.
    cont = model.decode(sim.Y_masked, "sa", known={"choice": sim.X[:, 4]})
    assert sorted(cont) == NAMES[:4]


def test_the_group_llr_is_the_two_row_profile_difference(
    model: MTDR, data: Any
) -> None:
    # With choice known and two candidate rows for the group, the LLR is the
    # difference of the plug-in log-likelihoods (M46) of the two designs.
    sim, design, _ = data
    rows = design.rows()[[0, 4]]
    _, llr = model.decode(
        sim.Y_masked,
        "sa",
        known={"choice": sim.X[:, 4]},
        levels={"sa": rows},
        return_llr=True,
    )
    by_hand = []
    for row in rows:
        X = np.array(sim.X)
        X[:, :4] = row
        by_hand.append(model.log_likelihood(sim.Y_masked, X, per_trial=True))
    np.testing.assert_allclose(llr, by_hand[1] - by_hand[0], rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"levels": {"sa_1": [0.0, 1.0]}}, "a column of group 'sa'"),
        ({"levels": {"sa": [[0, 0, 0]]}}, r"rows, each 4 finite"),
        ({"levels": {"sa": [0.0, 1.0]}}, r"rows, each 4 finite"),
        ({"levels": {"sa": [[0, 0, 0, np.nan]]}}, r"rows, each 4 finite"),
        ({"levels": {"sa": []}}, r"rows, each 4 finite"),
        ({"levels": {"sa": [["a", "b", "c", "d"]]}}, r"rows, each 4 finite"),
        ({"levels": {"sa": [[True, False, True, False]]}}, r"rows, each 4 finite"),
        ({"levels": {"sa": [[0, 0], [0, 0, 0, 0]]}}, r"rows, each 4 finite"),
        (
            {
                "regressor": "choice",
                "known": {"sa_1": np.zeros(400)},
                "levels": {"sa": [[0.0] * 4]},
            },
            r"its columns \['sa_1'\] are known",
        ),
        (
            {
                "regressor": ["choice", "sa_1"],
                "known": {c: np.zeros(400) for c in NAMES[1:4]},
                "levels": {"sa": [[0.0] * 4]},
            },
            "are known",
        ),
        (
            {"regressor": ["choice"], "levels": {"sa": [[0.0] * 4]}},
            r"columns \['sa_1', 'sa_2', 'sa_3', 'sa_4'\] are not decoded",
        ),
        ({"regressor": ["sa", "sa_2"]}, "'sa_2' is listed twice"),
        ({"regressor": "nope"}, "unknown regressor or group 'nope'"),
        ({"known": {"sa": np.zeros(400)}}, "is a group: give the values of its"),
        (
            {"levels": {"sa": np.zeros((13, 4)), "choice": [-1, 0, 1, 2, 3]}},
            "65 level combinations",
        ),
        (
            {"levels": {"sa": np.zeros((5, 4))}, "return_llr": True},
            "or exactly one group with two rows",
        ),
    ],
)
def test_decode_rejects(model: MTDR, data: Any, kw: dict[str, Any], match: str) -> None:
    sim = data[0]
    with pytest.raises(ParameterError, match=match):
        model.decode(sim.Y_masked, **kw)


def test_explained_variance_has_one_term_per_group(model: MTDR, data: Any) -> None:
    sim = data[0]
    ev = model.explained_variance(sim.Y_masked, sim.X)
    assert list(ev) == ["sa", "choice", "intercept", "total"]
    # The group's term is the prediction from all its columns at once.
    mask = sim.mask[:, :, None]
    Y = np.where(mask, sim.Y, 0.0)
    counts = sim.mask.sum(axis=0) * 6
    mean = Y.sum(axis=(0, 2)) / counts
    pred = mean[None, :, None] + np.einsum(
        "kp,pit->kit", sim.X[:, :4], np.stack([model.B_[c] for c in NAMES[:4]])
    )
    resid = np.where(mask, sim.Y - pred, 0.0)
    base = np.where(mask, sim.Y - mean[None, :, None], 0.0)
    want = 1 - np.sum(resid**2) / np.sum(base**2)
    assert ev["sa"] == pytest.approx(want, rel=1e-12)
    per_bin = model.explained_variance(sim.Y_masked, sim.X, per_bin=True)
    assert per_bin["sa"].shape == (6,)


def test_orthogonalize_and_subspace_angles_by_group(model: MTDR) -> None:
    Q = model.orthogonalize(["sa", "choice"])
    joined = np.concatenate([model.B_[c] for c in NAMES[:4]], axis=1)
    U = np.linalg.svd(joined, full_matrices=False)[0][:, :2]
    assert scipy.linalg.subspace_angles(Q["sa"], U).max() < 1e-10
    np.testing.assert_allclose(Q["sa"].T @ Q["choice"], 0, atol=1e-12)
    angles = model.subspace_angles("sa", "choice")
    assert angles.shape == (1,)
    assert model.subspace_angles("sa", "sa").max() < 1e-6
    with pytest.raises(ParameterError, match=r"order entry 'sa_3': it is a column"):
        model.orthogonalize(["sa_3"])
    with pytest.raises(ParameterError, match="repeated"):
        model.orthogonalize(["sa", "sa"])
    with pytest.raises(ParameterError, match=r"b = 'sa_4': it is a column"):
        model.subspace_angles("sa", "sa_4")


def test_to_xarray_with_groups(model: MTDR) -> None:
    pytest.importorskip("xarray")
    ds = model.to_xarray()
    assert dict(ds["W"].sizes) == {"group": 2, "neuron": 40, "component": 2}
    assert dict(ds["S"].sizes) == {"regressor": 5, "time": 6, "component": 2}
    assert ds["rank"].dims == ("group",)
    assert ds["rank"].values.tolist() == [2, 1]
    assert ds["group_of"].values.tolist() == ["sa"] * 4 + ["choice"]
    assert bool(ds["W"].isnull()[1, :, 1].all())
    np.testing.assert_array_equal(
        ds["B"].sel(regressor="sa_3").values, model.B_["sa_3"]
    )


def test_svd_estimator_with_groups(svd_model: MTDR, data: Any) -> None:
    sim = data[0]
    assert svd_model.W_["sa"].shape == (40, 2)
    assert svd_model.weight_posterior_cov_ is None
    assert svd_model.n_parameters_ == mtdr.aic.n_parameters_svd(
        [2, 1], 40, 6, groups=[[0, 1, 2, 3], [4]]
    )
    joined = np.concatenate([svd_model.B_[c] for c in NAMES[:4]], axis=1)
    assert np.linalg.matrix_rank(joined) == 2
    assert np.isfinite(svd_model.log_likelihood(sim.Y_masked, sim.X))


# ================================================================= pickles


def test_a_grouped_model_pickles(model: MTDR, data: Any) -> None:
    sim = data[0]
    again = pickle.loads(pickle.dumps(model))
    assert again.groups_ == model.groups_
    assert again.groups == GROUPS
    np.testing.assert_array_equal(again.predict(sim.X), model.predict(sim.X))


def test_a_model_pickled_before_groups_loads_as_singletons() -> None:
    # A 0.1.0 pickle has no `groups`, `groups_` or `group_names_`; deleting them
    # from a pickled state reproduces it without a binary fixture.
    sim = simulate(n_neurons=10, n_bins=4, n_trials=60, ranks=[1, 1], seed=0)
    fitted = MTDR(ranks=[1, 1]).fit(sim.Y, sim.X)
    state = dict(fitted.__dict__)
    for key in ("groups", "groups_", "group_names_"):
        del state[key]
    old = MTDR.__new__(MTDR)
    old.__setstate__(state)
    assert old.groups is None
    assert old.groups_ == {"x0": ("x0",), "x1": ("x1",)}
    assert old.group_names_ == ("x0", "x1")
    np.testing.assert_array_equal(old.project(sim.Y, "x0"), fitted.project(sim.Y, "x0"))
    restored = pickle.loads(pickle.dumps(old))
    assert restored.groups_ == old.groups_
    unfitted_state = dict(MTDR(ranks=[1]).__dict__)
    del unfitted_state["groups"]
    unfitted = MTDR.__new__(MTDR)
    unfitted.__setstate__(unfitted_state)
    assert unfitted.groups is None
    assert repr(unfitted) == "MTDR(ranks=[1])"
    with pytest.raises(mtdr.NotFittedError):
        unfitted.groups_  # noqa: B018


# ================================================================= data checks


def test_a_level_absent_from_the_fitted_trials(data: Any) -> None:
    # [1, X] is rank-deficient; the existing check says why, in these words.
    sim, _, labels = data
    keep = labels != 48
    with pytest.raises(ValidationError, match="categorical level absent"):
        MTDR(ranks={"sa": 1, "choice": 1}, groups=GROUPS).fit(
            sim.Y_masked[keep], sim.X[keep], regressor_names=NAMES
        )
    two = (labels != 48) & (labels != 24)
    with pytest.raises(ValidationError, match="categorical level absent"):
        MTDR(
            ranks={"sa": 1, "choice": 1}, groups=GROUPS, condition_independent=False
        ).fit(sim.Y_masked[two], sim.X[two], regressor_names=NAMES)


def test_a_grouped_fit_that_does_not_converge_says_so(data: Any) -> None:
    # The existing ConvergenceWarning and converged_=False, unchanged by groups.
    sim = data[0]
    with pytest.warns(ConvergenceWarning, match="iteration cap"):
        model = _fit(
            sim,
            ranks={"sa": 2, "choice": 1},
            groups=GROUPS,
            refine_max_iter=1,
            refine_tol=1e-300,
        )
    assert not model.converged_
    assert np.isfinite(model.log_likelihood_)


def test_neurons_without_a_group_direction_are_aggregated(data: Any) -> None:
    # A neuron never observed at one level lacks a direction of the group's
    # design; the prior regularises it, and one DesignWarning names it.
    sim, _, labels = data
    mask = np.array(sim.mask)
    mask[labels == 48, 3] = False
    with pytest.warns(DesignWarning, match=r"neurons \[3\] have a rank-deficient"):
        model = MTDR(ranks={"sa": 2, "choice": 1}, groups=GROUPS).fit(
            sim.Y, sim.X, mask, regressor_names=NAMES
        )
    assert np.isfinite(model.B_["sa_1"][3]).all()


# ================================================================= the search


def test_the_grouped_search_selects_the_planted_ranks() -> None:
    sim, _, _ = categorical_sim(3, n=60, T=8, N=500, rank=3, noise=4.0)
    # From ones the under-ranked candidates need more than the default 10
    # refinement iterations; the default start (below) does not.
    model = _fit(
        sim,
        ranks="aic",
        groups=GROUPS,
        max_rank=6,
        rank_search_init="ones",
        refine_max_iter=50,
    )
    assert model.ranks_ == {"sa": 3, "choice": 1}
    history = model.rank_search_history_
    assert history is not None
    assert history.regressor_names == ("sa", "choice")
    assert history.ranks.tolist() == [[1, 1], [2, 1], [3, 1]]
    assert history.final_ranks() == model.ranks_
    default = _fit(sim, ranks="aic", groups=GROUPS, max_rank=6)
    assert default.ranks_ == {"sa": 3, "choice": 1}
    assert default.rank_search_history_ is not None
    assert default.rank_search_history_.svd_stage is not None


def test_warm_and_cold_grouped_candidates_take_the_same_path() -> None:
    sim, _, _ = categorical_sim(4, n=50, T=6, N=400, rank=2)
    kw: dict[str, Any] = {
        "ranks": "aic",
        "groups": GROUPS,
        "max_rank": 4,
        "rank_search_init": {"sa": 1, "choice": 1},
    }
    cold = _fit(sim, rank_search_warm_start=False, **kw)
    warm = _fit(sim, rank_search_warm_start=True, **kw)
    hc, hw = cold.rank_search_history_, warm.rank_search_history_
    assert hc is not None
    assert hw is not None
    np.testing.assert_array_equal(hc.ranks, hw.ranks)
    np.testing.assert_allclose(hc.aic, hw.aic, rtol=1e-8)
    for a, b in zip(hc.candidates, hw.candidates, strict=True):
        assert list(a) == list(b)
        np.testing.assert_allclose(list(a.values()), list(b.values()), rtol=1e-6)


def test_a_grouped_warm_start_extends_every_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Raising group g appends the last component of the grouped SVD fit at the
    # candidate ranks to every member's S_p.
    sim, _, _ = categorical_sim(5, n=40, T=6, N=300, rank=2)
    calls: list[tuple[tuple[int, ...], Any]] = []
    real = mtdr.mmle.fit_mmle

    def spy(stats: Any, ranks: Any, init: Any = None, **kw: Any) -> Any:
        calls.append((tuple(int(r) for r in ranks), init))
        return real(stats, ranks, init, **kw)

    monkeypatch.setattr(mtdr.model, "fit_mmle", spy)
    model = _fit(
        sim,
        ranks="aic",
        groups=GROUPS,
        max_rank=3,
        rank_search_init={"sa": 1, "choice": 1},
    )
    history = model.rank_search_history_
    assert history is not None
    stats = sufficient_statistics(sim.Y_masked, sim.X, sim.mask)
    raised_sa = [(r, init) for r, init in calls[1:] if r == (2, 1)]
    assert raised_sa
    ranks, init = raised_sa[0]
    svd = mtdr.svd_fit.fit_svd(stats, list(ranks), groups=[[0, 1, 2, 3], [4]])
    for p in range(4):
        np.testing.assert_array_equal(init.S[p][:, -1], svd.S[p][:, -1])
        assert init.S[p].shape == (6, 2)
    assert init.S[4].shape == (6, 1)
    assert init.groups == ((0, 1, 2, 3), (4,))


def test_n_jobs_with_groups() -> None:
    sim, _, _ = categorical_sim(6, n=30, T=5, N=300, rank=2)
    kw: dict[str, Any] = {
        "ranks": "aic",
        "groups": GROUPS,
        "max_rank": 3,
        "rank_search_init": {"sa": 1, "choice": 1},
    }
    one = _fit(sim, n_jobs=1, **kw)
    two = _fit(sim, n_jobs=2, **kw)
    assert one.ranks_ == two.ranks_
    h1, h2 = one.rank_search_history_, two.rank_search_history_
    assert h1 is not None
    assert h2 is not None
    np.testing.assert_array_equal(h1.ranks, h2.ranks)
    np.testing.assert_allclose(h1.aic, h2.aic, rtol=1e-10)
    for c in NAMES:
        np.testing.assert_allclose(one.B_[c], two.B_[c], rtol=1e-8, atol=1e-12)


def test_the_selection_count_is_the_grouped_reference_count(model: MTDR) -> None:
    # (M60)'s n + T sum |g| r_g + nT, which exceeds the identifiable count by
    # sum_g r_g (r_g - 1) / 2.
    ref = mtdr.aic.n_parameters_mmle(
        [2, 1], 40, 6, groups=[[0, 1, 2, 3], [4]], formula="reference"
    )
    assert ref == 40 + 6 * (4 * 2 + 1) + 240
    assert ref - model.n_parameters_ == 1


# ================================================================= sharing


@pytest.mark.parametrize("seed", range(4))
def test_a_shared_subspace_is_recovered(seed: int) -> None:
    # Four columns with one rank-2 W_g and distinct S_p: the grouped rank-2 fit
    # recovers the neural subspace and counts its basis once. Bound: the largest
    # principal angle below 0.1 rad (measured 0.008-0.016 over seeds 0-7).
    cols = ["a", "b", "c", "d"]
    sim = simulate(
        n_neurons=60,
        n_bins=8,
        n_trials=400,
        ranks={"g": 2},
        regressor_names=cols,
        groups={"g": cols},
        levels=[[-1.0, 0.0, 1.0]] * 4,
        noise_precision=2.0,
        drop_prob=0.2,
        seed=seed,
    )
    model = MTDR(ranks={"g": 2}, groups={"g": cols}).fit(
        sim.Y_masked, sim.X, regressor_names=cols
    )
    assert model.total_rank_ == 2
    angles = scipy.linalg.subspace_angles(sim.W["g"], model.W_["g"])
    assert angles.max() < 0.1
    separate = MTDR(ranks=[2] * 4).fit(sim.Y_masked, sim.X, regressor_names=cols)
    assert separate.total_rank_ == 8
    # Distinct temporal bases: the columns' S are not multiples of each other.
    assert np.linalg.matrix_rank(np.concatenate([sim.S[c] for c in cols], 1)) > 2


# ================================================================= coding invariance


def _recoded_fit(
    sim: mtdr.SimulatedData, R: FloatArray, **kw: Any
) -> tuple[MTDR, FloatArray]:
    X = np.array(sim.X)
    X[:, :4] = X[:, :4] @ R
    model = MTDR(ranks={"sa": 2, "choice": 1}, groups=GROUPS, **kw).fit(
        sim.Y_masked, X, regressor_names=NAMES
    )
    return model, X


#: Converged refits of orthogonal recodings (refine_tol=1e-8): level responses
#: (relative) and log-likelihood (absolute, nats) agree to this. Measured: 1.8e-7
#: and 3.4e-9 for a random orthogonal R, 6.2e-13 and 2.9e-11 for a permutation;
#: the refits stop after different numbers of refinement iterations, so this is
#: the optimiser's tolerance, not rounding.
REFIT_B_RTOL = 1e-5
REFIT_LL_ATOL = 1e-6


@pytest.mark.parametrize("kind", ["orthogonal", "permutation"])
def test_orthogonal_recodings_reach_the_same_fit(data: Any, kind: str) -> None:
    # A random orthogonal R, or a level permutation R = Q' P Q: the converged
    # level responses and log-likelihoods agree to optimizer tolerance.
    sim, design, _ = data
    Q = design.basis
    rng = np.random.default_rng(7)
    if kind == "orthogonal":
        R = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    else:
        P = np.eye(5)[[2, 0, 4, 1, 3]]
        R = Q.T @ P @ Q
        np.testing.assert_allclose(R.T @ R, np.eye(4), atol=1e-12)
    kw = {"refine_tol": 1e-8, "refine_max_iter": 50}
    base, _ = _recoded_fit(sim, np.eye(4), **kw)
    other, _ = _recoded_fit(sim, R, **kw)
    # X R codes level a by the row (Q R)[a]: level responses use that coding.
    lr_base = np.einsum("ap,pit->ait", Q, np.stack([base.B_[c] for c in NAMES[:4]]))
    lr_other = np.einsum(
        "ap,pit->ait", Q @ R, np.stack([other.B_[c] for c in NAMES[:4]])
    )
    assert np.linalg.norm(lr_other - lr_base) <= REFIT_B_RTOL * np.linalg.norm(lr_base)
    assert abs(other.log_likelihood_ - base.log_likelihood_) <= REFIT_LL_ATOL


def test_any_invertible_recoding_leaves_the_objective_unchanged(data: Any) -> None:
    # X_g -> X_g R with S_g mixed by R^{-1}: the per-trial designs are equal, so
    # the marginal likelihood is too, for any invertible R (basis_ridge = 0).
    # Only orthogonal R also preserves the least-squares start, the ridge and
    # the canonical frame, so for this R a refit takes another path.
    sim, design, _ = data
    rng = np.random.default_rng(8)
    stats = sufficient_statistics(sim.Y_masked, sim.X, sim.mask)
    groups = [[0, 1, 2, 3], [4]]
    S = [rng.normal(size=(6, 2)) for _ in range(4)] + [rng.normal(size=(6, 1))]
    lam = rng.uniform(0.5, 3.0, 40)
    b = np.array(stats.Y_mean)
    Q = design.basis
    for R in (
        np.linalg.qr(rng.normal(size=(4, 4)))[0],  # orthogonal
        Q.T @ np.eye(5)[[1, 4, 0, 3, 2]] @ Q,  # a level permutation
        rng.normal(size=(4, 4)) + 3 * np.eye(4),  # not orthogonal
    ):
        X = np.array(sim.X)
        X[:, :4] = X[:, :4] @ R
        stats_r = sufficient_statistics(sim.Y_masked, X, sim.mask)
        Rinv = np.linalg.inv(R)
        S_r = [sum(Rinv[q, p] * S[p] for p in range(4)) for q in range(4)] + [S[4]]
        a = mtdr.mmle.marginal_log_likelihood(S, lam, b, stats, groups=groups)
        c = mtdr.mmle.marginal_log_likelihood(S_r, lam, b, stats_r, groups=groups)
        assert abs(a - c) <= 1e-10 * abs(a)


# ================================================================= plots


def _first_label(ax: Any) -> str:
    legend = ax.get_legend()
    assert legend is not None
    return str(legend.get_texts()[0].get_text())


def test_plots_with_groups(model: MTDR, svd_search: MTDR, data: Any) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    sim, design, _ = data
    try:
        axes = mtdr.plot.bases(model)
        assert axes.shape == (5,)
        assert axes[0].get_title() == "sa_1"
        assert "\\|S[:, 1]\\|" in _first_label(axes[0])
        assert "sigma" in _first_label(axes[4])
        # Bases are per column: a group name is not one.
        with pytest.raises(ParameterError, match=r"unknown regressor 'sa'; names"):
            mtdr.plot.bases(model, ["sa"])
        assert mtdr.plot.weights(model).shape == (2,)
        assert mtdr.plot.weights(model, ["sa"], component=1).shape == (1,)
        with pytest.raises(ParameterError, match="component 1 does not exist"):
            mtdr.plot.weights(model, ["choice"], component=1)
        with pytest.raises(ParameterError, match=r"regressors entry 'sa_1'"):
            mtdr.plot.weights(model, ["sa_1"])
        assert len(mtdr.plot.coefficient_norms(model).lines) == 5
        ax = mtdr.plot.trajectories(
            model, sim.Y_masked, sim.X, "sa", by=list(design.columns)
        )
        assert len(ax.lines) == 10  # five levels, a line and a start marker each
        assert _first_label(ax).startswith("sa_1 = ")
        ax = mtdr.plot.trajectories(
            model, sim.Y_masked, sim.X, "sa", by=list(design.columns), color_by="sa_2"
        )
        assert _first_label(ax).startswith("sa_2 = ")
        with pytest.raises(ParameterError, match="color_by must name a column"):
            mtdr.plot.trajectories(model, sim.Y_masked, sim.X, "sa", color_by="sa")
        assert mtdr.plot.recovery(model, sim).shape == (4,)
        assert mtdr.plot.rank_search(svd_search).shape == (1,)
        separate = _fit(sim, ranks=[1] * 5)
        with pytest.raises(ParameterError, match="groups hold other columns"):
            mtdr.plot.recovery(separate, sim)
        raw = _fit(sim, ranks={"sa": 2, "choice": 1}, groups=GROUPS, canonicalize=False)
        text = _first_label(mtdr.plot.bases(raw, ["sa_1"])[0])
        assert text.startswith("comp. 1: |W||S|")
    finally:
        plt.close("all")


# ================================================================= the API surface


def _public_callables() -> set[str]:
    modules = (
        mtdr.model,
        mtdr.mmle,
        mtdr.svd_fit,
        mtdr.aic,
        mtdr.rank_search,
        mtdr.data,
        mtdr.simulation,
        mtdr.stats,
        mtdr.plot,
        mtdr.xarray_io,
    )
    out: set[str] = set()
    for module in modules:
        for name in module.__all__:
            obj = getattr(module, name)
            if inspect.isfunction(obj):
                out.add(f"{module.__name__}.{name}")
            elif inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if attr.startswith("_"):
                        continue
                    if isinstance(member, property | classmethod) or inspect.isfunction(
                        member
                    ):
                        out.add(f"{module.__name__}.{name}.{attr}")
    return out


def _exercises(
    model: MTDR, svd_model: MTDR, svd_search: MTDR, data: Any
) -> dict[str, Callable[[], object]]:
    """One grouped exercise per public callable, or its documented refusal.

    The fits come from the module's fixtures, which made them with groups
    (`MTDR.fit`, `fit_mmle`, `fit_svd`, `greedy_aic`, `simulate`).
    """
    sim, design, labels = data
    Y, X = sim.Y_masked, sim.X
    stats = sufficient_statistics(Y, X, sim.mask)
    groups = [[0, 1, 2, 3], [4]]
    fit = model._fit_object
    assert isinstance(fit, mtdr.MMLEFit)
    history = svd_search.rank_search_history_
    assert history is not None
    S, lam, b = list(fit.S), fit.noise_precision, fit.intercept

    def refuses(call: Callable[[], object], match: str) -> Callable[[], object]:
        def run() -> object:
            with pytest.raises(ParameterError, match=match):
                call()
            return None

        return run

    m = "mtdr.model.MTDR"
    return {
        f"{m}.get_params": lambda: model.get_params(),
        f"{m}.set_params": lambda: MTDR(groups=GROUPS).set_params(ranks={"sa": 1}),
        f"{m}.fit": lambda: model,
        f"{m}.predict": lambda: model.predict(X),
        f"{m}.log_likelihood": lambda: model.log_likelihood(Y, X),
        f"{m}.score": lambda: model.score(Y, X),
        f"{m}.aic": lambda: model.aic(),
        f"{m}.project": lambda: model.project(Y, "sa"),
        f"{m}.decode": lambda: model.decode(Y, None, levels={"sa": design.rows()}),
        f"{m}.explained_variance": lambda: model.explained_variance(Y, X),
        f"{m}.orthogonalize": lambda: model.orthogonalize(["sa", "choice"]),
        f"{m}.subspace_angles": lambda: model.subspace_angles("sa", "choice"),
        f"{m}.to_xarray": lambda: model.to_xarray(),
        "mtdr.model.canonical_factors": lambda: mtdr.canonical_factors(
            np.concatenate([model.B_[c] for c in NAMES[:4]], axis=1), 2
        ),
        "mtdr.mmle.marginal_log_likelihood": lambda: mtdr.mmle.marginal_log_likelihood(
            S, lam, b, stats, groups=groups
        ),
        "mtdr.mmle.marginal_nll_grad_S": lambda: mtdr.mmle.marginal_nll_grad_S(
            S, lam, b, stats, groups=groups
        ),
        "mtdr.mmle.marginal_nll_grad_noise": lambda: mtdr.mmle.marginal_nll_grad_noise(
            S, lam, b, stats, groups=groups
        ),
        "mtdr.mmle.posterior_weights": lambda: mtdr.mmle.posterior_weights(
            S, lam, b, stats, groups=groups
        ),
        "mtdr.mmle.update_intercept": lambda: mtdr.mmle.update_intercept(
            S, lam, stats, groups=groups
        ),
        "mtdr.mmle.ecme": lambda: mtdr.mmle.ecme(stats, fit, max_iter=1, groups=groups),
        "mtdr.mmle.refine": lambda: mtdr.mmle.refine(
            stats, fit, max_iter=1, groups=groups
        ),
        "mtdr.mmle.fit_mmle": lambda: fit,
        "mtdr.mmle.MMLEFit.from_parameters": lambda: mtdr.MMLEFit.from_parameters(
            stats, S, lam, b, groups=groups
        ),
        "mtdr.mmle.MMLEFit.B": lambda: fit.B,
        "mtdr.mmle.MMLEFit.n_neurons": lambda: fit.n_neurons,
        "mtdr.mmle.MMLEFit.n_bins": lambda: fit.n_bins,
        "mtdr.mmle.MMLEFit.total_rank": lambda: fit.total_rank,
        "mtdr.svd_fit.fit_svd": lambda: svd_model._fit_object,
        "mtdr.aic.n_parameters_svd": refuses(
            lambda: mtdr.aic.n_parameters_svd(
                [2, 1], 40, 6, groups=groups, formula="reference"
            ),
            "no regressor groups",
        ),
        "mtdr.aic.n_parameters_mmle": lambda: mtdr.aic.n_parameters_mmle(
            [2, 1], 40, 6, groups=groups
        ),
        "mtdr.aic.aic": lambda: mtdr.aic.aic(fit.log_likelihood, fit.n_parameters),
        "mtdr.rank_search.greedy_aic": lambda: history,
        "mtdr.rank_search.RankSearchHistory.final_ranks": history.final_ranks,
        "mtdr.rank_search.RankSearchHistory.n_fits": lambda: history.n_fits,
        "mtdr.rank_search.RankSearchHistory.margins": history.margins,
        "mtdr.rank_search.RankSearchHistory.near_ties": history.near_ties,
        "mtdr.rank_search.RankSearchHistory.to_frame": history.to_frame,
        "mtdr.data.validate_inputs": lambda: mtdr.validate_inputs(Y, X),
        "mtdr.data.check_design": lambda: mtdr.check_design(X, sim.mask),
        "mtdr.data.condition_average": lambda: mtdr.condition_average(
            Y, X, by=NAMES[:4], regressor_names=NAMES
        ),
        "mtdr.data.stack_sessions": lambda: mtdr.stack_sessions([(Y[:20], X[:20])]),
        "mtdr.data.split_trials": lambda: mtdr.split_trials(400, stratify=labels),
        "mtdr.data.resolve_groups": lambda: mtdr.resolve_groups(NAMES, GROUPS),
        "mtdr.data.categorical_design": lambda: categorical_design(labels, name="sa"),
        "mtdr.data.CategoricalDesign.groups": lambda: design.groups,
        "mtdr.data.CategoricalDesign.encode": lambda: design.encode(labels[:3]),
        "mtdr.data.CategoricalDesign.rows": design.rows,
        "mtdr.data.CategoricalDesign.level_responses": lambda: design.level_responses(
            model
        ),
        "mtdr.simulation.simulate": lambda: sim,
        "mtdr.simulation.SimulatedData.Y_masked": lambda: sim.Y_masked,
        "mtdr.simulation.SimulatedData.fit_kwargs": sim.fit_kwargs,
        "mtdr.stats.sufficient_statistics": lambda: sufficient_statistics(
            Y, X, sim.mask
        ),
        "mtdr.stats.SufficientStats.XtX": lambda: stats.XtX,
        "mtdr.stats.SufficientStats.XtY_raw": lambda: stats.XtY_raw,
        "mtdr.stats.SufficientStats.YtY_raw": lambda: stats.YtY_raw,
        "mtdr.stats.SufficientStats.n_neurons": lambda: stats.n_neurons,
        "mtdr.stats.SufficientStats.centered": lambda: stats.centered(b),
        "mtdr.plot.bases": lambda: mtdr.plot.bases(model),
        "mtdr.plot.weights": lambda: mtdr.plot.weights(model),
        "mtdr.plot.rank_search": lambda: mtdr.plot.rank_search(history),
        "mtdr.plot.coefficient_norms": lambda: mtdr.plot.coefficient_norms(model),
        "mtdr.plot.trajectories": lambda: mtdr.plot.trajectories(
            model, Y, X, "sa", by=NAMES[:4]
        ),
        "mtdr.plot.recovery": lambda: mtdr.plot.recovery(model, sim),
        "mtdr.xarray_io.to_dataset": lambda: mtdr.xarray_io.to_dataset(model),
    }


def test_every_public_callable_has_a_grouped_exercise(
    model: MTDR, svd_model: MTDR, svd_search: MTDR, data: Any
) -> None:
    # A callable added to the public API without a grouped exercise (or its
    # documented ParameterError) fails here.
    pytest.importorskip("matplotlib")
    pytest.importorskip("xarray")
    import matplotlib.pyplot as plt

    exercises = _exercises(model, svd_model, svd_search, data)
    public = _public_callables()
    assert public == set(exercises), (
        sorted(public - set(exercises)),
        sorted(set(exercises) - public),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for name in sorted(exercises):
                exercises[name]()
    finally:
        plt.close("all")
