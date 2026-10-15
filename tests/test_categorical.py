"""`categorical_design` and `CategoricalDesign`: a categorical regressor as one group.

The polynomial basis is checked against the five-level columns used for Sa
(linear, quadratic, cubic, quartic, each of norm sqrt(10)) and against the QR
decomposition of the Vandermonde matrix; labels are validated before any
conversion could hide a missing value; `level_responses` is checked against a
dense reconstruction.
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


def test_an_explicit_basis() -> None:
    Q = scipy.linalg.orth(np.eye(4) - 0.25)
    given = categorical_design([0, 1, 2, 3], name="q", basis=Q)
    np.testing.assert_array_equal(given.basis, Q)
    assert not given.basis.flags.writeable


@pytest.mark.parametrize(
    ("basis", "match"),
    [
        ("helmert", "basis must be 'polynomial' or a finite real"),
        (np.ones((4, 4)), r"finite real \(L, L - 1\) = \(4, 3\) array; got shape"),
        (np.full((4, 3), np.nan), "finite"),
        ([["a"] * 3] * 4, "a non-real or ragged array"),
        ([[1.0] * 3] * 3 + [[1.0]], "a non-real or ragged array"),
        # A complex matrix whose real part is a valid basis: refused, not cast.
        ((1 + 1j) * scipy.linalg.helmert(4).T / np.sqrt(2), "non-real"),
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


def test_encode() -> None:
    design = categorical_design(["a", "b", "c", "d"], name="g")
    X = design.encode(np.array(["c", "a", "d", "d", "b"]))
    assert X.flags.writeable
    np.testing.assert_array_equal(X, design.basis[[2, 0, 3, 3, 1]])
    # Levels absent from the values are allowed in encode.
    assert design.encode(["a"]).shape == (1, 3)
    # The string "nan" is an ordinary label; so is an unmasked masked array.
    words = categorical_design(["nan", "low", "nan"], name="w")
    assert words.levels.tolist() == ["low", "nan"]
    assert words.encode(np.ma.array(["low"], mask=[False])).shape == (1, 1)


#: Missing or unsupported labels, refused where they were silently coded.
MISSING: list[tuple[Any, str]] = [
    (["low", np.nan, "high"], r"missing or unsupported labels at \[1\]: \[nan\]"),
    (np.array(["a", None], dtype=object), r"at \[1\]: \[None\]"),
    ([0.0, float("nan")], r"at \[1\]"),
    ([1.0, np.inf], r"at \[1\]"),
    ([1 + 1j, 2], r"at \[0\]"),
    ([{"a": 1}, "b"], r"at \[0\]"),
    ([[1, 2], [3]], r"at \[0, 1\]"),
    (
        np.ma.array(["A", "B", "C"], mask=[False, True, False]),
        r"masked labels at \[1\]",
    ),
    ([1, "a"], "mixes numbers and strings"),
    ([True, 0], "mixes booleans and numbers"),
    ([[0, 1], [1, 0]], "must be 1-D"),
]


@pytest.mark.parametrize(("values", "match"), MISSING)
def test_missing_or_unsupported_labels_are_refused(values: Any, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        categorical_design(values, name="c")
    with pytest.raises(ValidationError, match=match):
        categorical_design(["a", "b"], levels=values, name="c")
    design = categorical_design(["a", "b", "c", "d"], name="g")
    with pytest.raises(ValidationError, match=match):
        design.encode(values)


def test_pandas_missing_values_are_refused() -> None:
    pd = pytest.importorskip("pandas")
    labels = np.array([0, pd.NA], dtype=object)
    with pytest.raises(ValidationError, match=r"at \[1\]: \[<NA>\]"):
        categorical_design(labels, name="c")
    with pytest.raises(ValidationError, match=r"at \[1\]"):
        categorical_design([0, 1], levels=labels, name="c")
    with pytest.raises(ValidationError, match=r"at \[1\]"):
        categorical_design(pd.Series(["a", None, "b"]), name="c")


def test_encode_rejects_labels_outside_the_levels() -> None:
    design = categorical_design(["a", "b", "c", "d"], name="g")
    with pytest.raises(ValidationError, match=r"labels \['e', 'f'\] that are not"):
        design.encode(["a", "e", "f"])


@pytest.mark.parametrize(
    ("kw", "error", "match"),
    [
        ({"values": [1, 1, 1]}, ValidationError, "at least two levels"),
        ({"values": [1, 2], "levels": [1, 2, 1]}, ValidationError, "repeat a level"),
        ({"values": [1, 2, 3], "levels": [1, 2]}, ValidationError, r"labels \[3\]"),
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


def test_level_responses_of_separate_terms_and_missing_columns() -> None:
    # The sum over the coding is defined for separate-term fits as well.
    model, design = _fit(1, None)
    responses = design.level_responses(model)
    want = np.einsum("ap,pit->ait", design.basis, np.stack(list(model.B_.values())))
    np.testing.assert_allclose(responses, want, atol=1e-13)
    other = categorical_design(["w", "x", "y", "z", "v"], name="stim")
    with pytest.raises(ParameterError, match=r"no columns \['stim_4'\] of 'stim'"):
        other.level_responses(model)
    with pytest.raises(mtdr.NotFittedError):
        design.level_responses(MTDR())
