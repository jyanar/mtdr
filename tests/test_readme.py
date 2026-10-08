"""The README's links into this repository name files that exist.

The README is also PyPI's project description, so it links absolutely. It skips
where the README is not next to the tests (the wheel job copies `tests/` alone).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPO_URL = "https://github.com/jyanar/mtdr"
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

pytestmark = pytest.mark.skipif(
    not README.is_file(), reason="README.md is not next to the tests"
)


def _link_targets(markdown: str) -> list[str]:
    """Targets of inline Markdown links, outside HTML comments."""
    return LINK.findall(COMMENT.sub("", markdown))


def test_the_readme_links_only_files_that_exist() -> None:
    prefixes = (f"{REPO_URL}/blob/main/", f"{REPO_URL}/tree/main/")
    targets = _link_targets(README.read_text(encoding="utf-8"))
    linked = [t for t in targets if t.startswith(prefixes)]
    assert linked, "the README links no file of the repository"
    for target in linked:
        for prefix in prefixes:
            if target.startswith(prefix):
                assert (ROOT / target.removeprefix(prefix)).exists(), target


def test_link_targets_ignore_comments_and_bare_names() -> None:
    text = "see [a](x/a.md) <!-- [b](x/b.md) --> and x/c.md"
    assert _link_targets(text) == ["x/a.md"]
