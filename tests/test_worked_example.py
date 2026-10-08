"""The worked example of `docs/getting-started.md` runs as written.

The test reads the ```python blocks of `docs/getting-started.md` in order and
executes them in one session, with matplotlib off-screen, under the suite's warning
policy (every warning an error). It skips only where `docs/` is not next to the
tests (the wheel job copies `tests/` alone). Measured values of the example are
printed for the record (`-s`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGE = DOCS / "getting-started.md"


def _example() -> str:
    text = PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert blocks, "docs/getting-started.md has no python blocks"
    return "\n".join(blocks)


def test_the_worked_example_runs_as_written() -> None:
    if not DOCS.is_dir():
        pytest.skip("docs/ is not next to the tests")
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    namespace: dict[str, object] = {"__name__": "getting_started_example"}
    try:
        exec(compile(_example(), str(PAGE), "exec"), namespace)
    finally:
        plt.close("all")
    model = namespace["model"]
    fixed = namespace["fixed"]
    sim = namespace["sim"]
    assert model.ranks_ == dict(sim.ranks)  # type: ignore[attr-defined]
    assert fixed.converged_  # type: ignore[attr-defined]
