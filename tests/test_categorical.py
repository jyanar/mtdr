"""`categorical_design` and `CategoricalDesign`: a categorical regressor as one group.

The polynomial basis is checked against the five-level columns used for Sa
(linear, quadratic, cubic, quartic, each of norm sqrt(10)) and against the QR
decomposition of the Vandermonde matrix; encoding and decoding round-trip;
`level_responses` is checked against a dense reconstruction.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest
import scipy.linalg

import mtdr
from mtdr import MTDR, ParameterError, ValidationError, categorical_design, simulate

#: The five-level columns of the polynomial Sa coding, each of norm sqrt(10).
ADAPTER = np.array(
    [
        [-2, -1, 0, 1, 2],
        np.array([2, -1, -2, -1, 2]) * np.sqrt(10 / 14),
        [-1, 2, 0, -2, 1],
        np.array([1, -4, 6, -4, 1]) * np.sqrt(10 / 70),
    ],
    dtype=float,
).T


def test_the_five_level_polynomial_basis_is_the_sa_coding() -> None:
    design = categorical_design([3, 6, 12, 24, 48], name="sa")
    np.testing.assert_allclose(np.linalg.norm(ADAPTER, axis=0), np.sqrt(10))
    np.testing.assert_allclose(design.basis, ADAPTER / np.sqrt(10), atol=1e-14)


@pytest.mark.parametrize("L", range(2, 9))
def test_the_polynomial_basis_is_the_vandermonde_qr(L: int) -> None:
    # The Q factor of the Vandermonde matrix of the indices without its constant
    # column, signed so each leading coefficient is positive (R's diagonal).
    x = np.arange(L, dtype=float)
    V = np.vander(x, L, increasing=True)
    Q, R = np.linalg.qr(V)
    Q = Q * np.sign(np.diag(R))
    basis = categorical_design(np.arange(L), name="c").basis
    np.testing.assert_allclose(basis, Q[:, 1:], atol=1e-10)
    np.testing.assert_allclose(basis.T @ basis, np.eye(L - 1), atol=1e-13)
    np.testing.assert_allclose(basis.sum(axis=0), 0, atol=1e-13)


def test_many_levels_stay_orthonormal() -> None:
    # At 60 levels the Vandermonde matrix's condition number is about 1e106; the
    # recurrence still gives an orthonormal zero-sum basis. (A QR of that matrix
    # stays orthonormal too, but drifts from the polynomial basis, by 7e-4 at
    # 20 levels, which the recurrence does not.)
    basis = categorical_design(np.arange(60), name="c").basis
    np.testing.assert_allclose(basis.T @ basis, np.eye(59), atol=1e-12)
    np.testing.assert_allclose(basis.sum(axis=0), 0, atol=1e-12)


def test_helmert_and_an_explicit_basis() -> None:
    helmert = categorical_design(["a", "b", "c"], name="h", basis="helmert")
    np.testing.assert_allclose(helmert.basis, scipy.linalg.helmert(3).T)
    Q = scipy.linalg.orth(np.eye(4) - 0.25)
    given = categorical_design([0, 1, 2, 3], name="q", basis=Q)
    np.testing.assert_array_equal(given.basis, Q)
    assert not given.basis.flags.writeable


@pytest.mark.parametrize(
    ("basis", "match"),
    [
        ("legendre", "basis must be 'polynomial', 'helmert'"),
        (np.ones((4, 4)), r"finite \(L, L - 1\) = \(4, 3\) array"),
        (np.full((4, 3), np.nan), "finite"),
        ([["a"] * 3] * 4, "not a real array"),
        (np.eye(4)[:, :3], r"scipy.linalg.orth\(C - C.mean"),
        (2 * scipy.linalg.helmert(4).T, "sum to zero"),
    ],
)
def test_a_bad_basis_is_refused(basis: Any, match: str) -> None:
    with pytest.raises(ParameterError, match=match):
        categorical_design([0, 1, 2, 3], name="c", basis=basis)


def test_codes_counts_and_groups() -> None:
    values = np.array([12, 3, 3, 48, 6, 24, 12, 3])
    design = categorical_design(values, name="sa")
    assert design.levels.tolist() == [3, 6, 12, 24, 48]
    assert design.counts.tolist() == [3, 1, 2, 1, 1]
    assert design.columns == ("sa_1", "sa_2", "sa_3", "sa_4")
    np.testing.assert_array_equal(design.X, design.basis[[2, 0, 0, 4, 1, 3, 2, 0]])
    assert design.groups == {"sa": ("sa_1", "sa_2", "sa_3", "sa_4")}
    assert design.groups is not design.groups  # a new dict each time
    assert repr(design) == (
        "CategoricalDesign(name='sa', levels=[3, 6, 12, 24, 48], counts=[3, 1, 2, "
        "1, 1], columns=['sa_1', 'sa_2', 'sa_3', 'sa_4'])"
    )
    for arr in (design.levels, design.counts, design.basis, design.X):
        assert not arr.flags.writeable
    again = pickle.loads(pickle.dumps(design))
    assert not again.X.flags.writeable
    np.testing.assert_array_equal(again.X, design.X)
    # Given levels set the order; absent levels are allowed.
    ordered = categorical_design(
        ["lo", "hi", "lo"], levels=["lo", "mid", "hi"], name="c"
    )
    assert ordered.counts.tolist() == [2, 0, 1]
    np.testing.assert_array_equal(ordered.X[1], ordered.basis[2])
    booleans = categorical_design([True, False, True], name="b")
    assert booleans.levels.tolist() == [False, True]


def test_encode_and_labels_round_trip() -> None:
    design = categorical_design(["a", "b", "c", "d"], name="g")
    values = np.array(["c", "a", "d", "d", "b"])
    X = design.encode(values)
    assert X.flags.writeable
    labels = design.labels_from_rows(X)
    assert isinstance(labels, np.ma.MaskedArray)
    assert labels.tolist() == values.tolist()
    # Levels absent from the values are allowed in encode.
    assert design.encode(["a"]).shape == (1, 3)
    # Decode's mapping, NaN rows masked; extra keys ignored.
    rows = dict(zip(design.columns, X.T, strict=True))
    rows["g_2"] = rows["g_2"].copy()
    for c in design.columns:
        rows[c] = rows[c].copy()
        rows[c][1] = np.nan
    rows["other"] = np.zeros(5)
    masked = design.labels_from_rows(rows)
    assert masked.mask.tolist() == [False, True, False, False, False]
    assert masked.compressed().tolist() == ["c", "d", "d", "b"]


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (["a", "e", "f"], r"labels \['e', 'f'\] that are not among the levels"),
        ([[0, 1], [1, 0]], "must be 1-D"),
        (np.array([0.0, np.nan]), r"missing labels \(NaN or None\) at \[1\]"),
        (np.array(["a", None], dtype=object), r"missing labels .* at \[1\]"),
        (np.array([1 + 1j]), "numbers, strings or booleans"),
        ([[1, 2], [3]], "ragged"),
    ],
)
def test_encode_rejects(values: Any, match: str) -> None:
    design = categorical_design(["a", "b", "c", "d"], name="g")
    with pytest.raises(ValidationError, match=match):
        design.encode(values)


@pytest.mark.parametrize(
    ("kw", "error", "match"),
    [
        ({"values": [1, 1, 1]}, ValidationError, "at least two levels"),
        ({"values": [1, 2], "levels": [1, 2, 1]}, ValidationError, "repeat a level"),
        ({"values": [1, 2, 3], "levels": [1, 2]}, ValidationError, r"labels \[3\]"),
        (
            {"values": np.array([1, "a"], dtype=object)},
            ValidationError,
            "cannot be sorted",
        ),
        ({"values": [1, 2], "name": ""}, ParameterError, "non-empty string"),
        ({"values": [1, 2], "name": 3}, ParameterError, "non-empty string"),
        ({"values": [1, 2], "name": "total"}, ParameterError, "reserved"),
        ({"values": [1, 2, 3], "name": "intercept"}, ParameterError, "reserved"),
    ],
)
def test_categorical_design_rejects(
    kw: dict[str, Any], error: type[Exception], match: str
) -> None:
    kw = {"name": "c", **kw}
    with pytest.raises(error, match=match):
        categorical_design(**kw)


def test_labels_from_rows_refuses_rows_that_match_no_level() -> None:
    design = categorical_design([0, 1, 2], name="c")
    X = design.encode([0, 2])
    X[1, 0] += 1e-6
    with pytest.raises(ValidationError, match=r"trials \[1\] match no level"):
        design.labels_from_rows(X)
    with pytest.raises(ValidationError, match="lacks the columns"):
        design.labels_from_rows({"c_1": X[:, 0]})
    with pytest.raises(ValidationError, match="one length"):
        design.labels_from_rows({"c_1": X[:, 0], "c_2": X[:1, 1]})
    with pytest.raises(ValidationError, match=r"real \(n_trials, 2\) array"):
        design.labels_from_rows(np.zeros((2, 3)))


def _fit(seed: int, groups: Any) -> tuple[MTDR, mtdr.CategoricalDesign]:
    rng = np.random.default_rng(seed)
    labels = rng.choice(["w", "x", "y", "z"], 300)
    design = categorical_design(labels, name="stim")
    sim = simulate(
        n_neurons=20,
        n_bins=5,
        n_trials=300,
        ranks={"stim": 2},
        regressor_names=list(design.columns),
        groups=design.groups,
        X=design.X,
        noise_precision=3.0,
        seed=seed,
    )
    ranks = {"stim": 2} if groups else dict.fromkeys(design.columns, 1)
    model = MTDR(ranks=ranks, groups=groups).fit(
        sim.Y, sim.X, regressor_names=list(design.columns)
    )
    return model, design


def test_level_responses_match_a_dense_reconstruction() -> None:
    model, design = _fit(0, {"stim": ["stim_1", "stim_2", "stim_3"]})
    responses = design.level_responses(model)
    assert responses.shape == (4, 20, 5)
    dense = np.zeros((4, 20, 5))
    for a in range(4):
        for p, column in enumerate(design.columns):
            dense[a] += design.basis[a, p] * model.B_[column]
    np.testing.assert_allclose(responses, dense, atol=1e-13)
    np.testing.assert_allclose(responses.sum(axis=0), 0, atol=1e-12)
    # The level means: the intercept plus each level's response.
    assert model.intercept_ is not None
    means = model.intercept_[None] + responses
    np.testing.assert_allclose(
        means[1], model.predict(design.encode(["x"]))[0], atol=1e-12
    )


def test_level_responses_need_the_coding_as_one_group() -> None:
    model, design = _fit(1, None)
    with pytest.raises(ParameterError, match="do not form one group"):
        design.level_responses(model)
    with pytest.raises(mtdr.NotFittedError):
        design.level_responses(MTDR())
