"""The test suite's warning policy: `filterwarnings = ["error"]` alone.

No warning is ignored suite-wide. A rounding-level end of the precision step's
line search needs no filter: `mmle.refine` counts it as success when a scale-free
residual is below 1e-8 (`mmle.PRECISION_RESIDUAL_TOL`), and warns otherwise. So
every warning is an error unless a test records it explicitly (`pytest.warns`, or
`parity_fixtures.run` with declared reasons). These tests pin that policy, and the
`slow` marker's registration in `conftest.py`.
"""

from __future__ import annotations

import tomllib
import warnings
from pathlib import Path

import pytest

from mtdr.errors import ConvergenceWarning

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_suite_turns_every_warning_into_an_error() -> None:
    if not PYPROJECT.is_file():  # the wheel job copies tests/ without it
        pytest.skip("pyproject.toml is not next to the tests")
    options = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["pytest"][
        "ini_options"
    ]
    assert options["filterwarnings"] == ["error"]


def test_a_convergence_warning_is_an_error_here() -> None:
    message = (
        "refinement: iteration 1, noise precision: ABNORMAL: (L-BFGS-B status 2, "
        "nit 2, largest projected-gradient entry 4.88e-05); neurons with the "
        "largest |lambda_i E_i / (n_i T) - 1|: 12 (2.71e-07)"
    )
    if not PYPROJECT.is_file():
        pytest.skip("pyproject.toml is not next to the tests")
    with pytest.raises(ConvergenceWarning):
        warnings.warn(message, ConvergenceWarning, stacklevel=1)


def test_the_slow_marker_is_registered(pytestconfig: pytest.Config) -> None:
    markers = pytestconfig.getini("markers")
    assert any(str(m).startswith("slow:") for m in markers)
