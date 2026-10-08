"""Shared pytest configuration: hypothesis profiles and the test-helper path.

Select a profile with ``HYPOTHESIS_PROFILE=ci`` (set in GitHub Actions) or
``--hypothesis-profile=ci``. ``ci`` runs more examples, derandomized so a CI
failure reproduces locally; ``dev`` is the fast default.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

# Plots draw off-screen in tests and doctests (mtdr.plot).
os.environ.setdefault("MPLBACKEND", "Agg")

# `--import-mode=importlib` does not put tests/ on sys.path; the parity tests
# import the reference replicas from tests/matlab_compat.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

settings.register_profile("dev", max_examples=50)
settings.register_profile(
    "ci",
    max_examples=200,
    derandomize=True,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def pytest_configure(config: pytest.Config) -> None:
    """Register the `slow` marker here too.

    The wheel job runs the copied tests without `pyproject.toml`, where the
    marker is also declared; registering it here keeps that run quiet.
    """
    config.addinivalue_line(
        "markers",
        "slow: paper-scale MATLAB parity (minutes); deselected by default",
    )
