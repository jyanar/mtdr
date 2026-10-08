"""Tests for :mod:`mtdr.errors`: hierarchy, aliases, and pickling."""

from __future__ import annotations

import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import mtdr
from mtdr import errors

ERRORS: list[tuple[type[Exception], type[Exception]]] = [
    (errors.ValidationError, ValueError),
    (errors.ParameterError, ValueError),
    (errors.SingularDesignError, np.linalg.LinAlgError),
    (errors.NotFittedError, AttributeError),
]
WARNINGS: list[type[Warning]] = [
    errors.ConvergenceWarning,
    errors.DesignWarning,
    errors.DecodingWarning,
    errors.ProjectionWarning,
]


@pytest.mark.parametrize(("cls", "builtin"), ERRORS)
def test_error_is_catchable_as_builtin(
    cls: type[Exception], builtin: type[Exception]
) -> None:
    assert issubclass(cls, errors.MTDRError)
    with pytest.raises(builtin, match="boom"):
        raise cls("boom")


def test_singular_design_error_is_a_value_error() -> None:
    # LinAlgError subclasses ValueError; downstream code relying on either works.
    assert issubclass(errors.SingularDesignError, ValueError)


def test_not_fitted_error_makes_hasattr_false() -> None:
    class Unfitted:
        @property
        def ranks_(self) -> dict[str, int]:
            raise errors.NotFittedError("call fit() first")

    assert not hasattr(Unfitted(), "ranks_")


@pytest.mark.parametrize("cls", WARNINGS)
def test_warning_hierarchy(cls: type[Warning]) -> None:
    assert issubclass(cls, errors.MTDRWarning)
    assert issubclass(cls, UserWarning)
    with pytest.warns(cls, match="careful"):
        warnings.warn("careful", cls, stacklevel=1)


@pytest.mark.parametrize("cls", WARNINGS)
def test_root_warning_filter_escalates_every_warning(cls: type[Warning]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", errors.MTDRWarning)
        with pytest.raises(cls):
            warnings.warn("escalated", cls, stacklevel=1)


def test_errors_all_matches_module_contents() -> None:
    defined = {
        name
        for name, obj in vars(errors).items()
        if isinstance(obj, type)
        and issubclass(obj, (errors.MTDRError, errors.MTDRWarning))
    }
    assert defined == set(errors.__all__)


def test_top_level_aliases_are_the_same_objects() -> None:
    for name in errors.__all__:
        assert getattr(mtdr, name) is getattr(errors, name)
    assert set(errors.__all__) <= set(mtdr.__all__)


def test_version_is_a_string() -> None:
    assert isinstance(mtdr.__version__, str)
    assert mtdr.__version__


@given(message=st.text())
@pytest.mark.parametrize(("cls", "builtin"), ERRORS)
def test_errors_round_trip_through_pickle(
    cls: type[Exception], builtin: type[Exception], message: str
) -> None:
    err = cls(message)
    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is cls
    assert isinstance(restored, builtin)
    assert str(restored) == str(err)


def test_citation_version_matches_package() -> None:
    cff = Path(__file__).resolve().parent.parent / "CITATION.cff"
    if not cff.is_file():  # e.g. tests run against an installed wheel
        pytest.skip("CITATION.cff not available")
    match = re.search(r"^version:\s*(\S+)\s*$", cff.read_text(encoding="utf-8"), re.M)
    assert match is not None, "CITATION.cff has no version field"
    assert match.group(1) == mtdr.__version__
