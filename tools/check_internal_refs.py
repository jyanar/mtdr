"""Fail on cross-references to working notes that are not part of this repository.

The package was developed against a log of design decisions, review notes and
experiments that is kept outside the repository. Their identifiers (``D12``,
``H4``, ``E11``, a review's ``R3``) and paths mean nothing to a reader of the
code, so the code, tests and docs explain themselves instead, and this check keeps
the identifiers from coming back. Equation labels ``(M1)``-``(M50)`` are allowed:
they refer to ``docs/model.md``.

Usage (from the repository root)::

    python tools/check_internal_refs.py            # every tracked text file
    python tools/check_internal_refs.py FILE ...   # only these (pre-commit)

Notebooks are checked cell source by cell source; their outputs are not read.
Binary files (``.mat``, images) are skipped.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

#: (pattern, what it is). Each matches an identifier or path of the notes.
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bD\d{1,3}[a-z]?\b"), "decision number"),
    (re.compile(r"\bH\d{1,2}\b"), "reference-defect number"),
    (re.compile(r"\bE\d{1,2}\b"), "experiment number"),
    # Only as a citation, "(R4)" or "(..., T7)": code names such as Q0 or R2 pass.
    (
        re.compile(r"(?:(?<=\()|(?<=[,;] ))[JQRT]\d{1,2}(?:\([a-z]\))?(?=[),;:])"),
        "review-item number",
    ),
    (
        re.compile(r"\b(?:CL|CM|CX|XM)-[A-Z]*\d+|\b(?:CL|CX|XM)\b|\bCM\b(?!-step)"),
        "reviewer tag",
    ),
    (re.compile(r"\b[Pp]hase[- ]\d"), "development phase"),
    (re.compile(r"\btriage\b", re.IGNORECASE), "review triage"),
    (re.compile(r"\bplanning/"), "path into the notes"),
    (
        re.compile(r"\b(?:DECISIONS|PORT_MAP|PLAN|MODEL|API)(?:\.md)?`?\s*§"),
        "section of a note",
    ),
    (re.compile(r"\b(?:DECISIONS|PORT_MAP|PLAN|MODEL|API)\.md\b"), "note file"),
    (re.compile(r"\b(?:API|PLAN|MODEL) section\b"), "section of a note"),
    (re.compile(r"Users[\\/]jorge|%TEMP%"), "local path"),
]

#: Never read: binary data, and this file, which spells the patterns out.
SKIP_SUFFIXES = {".mat", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".svg"}
SELF = Path(__file__).resolve()


def tracked_files() -> list[Path]:
    """Return the files git tracks, relative to the repository root."""
    out = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout
    return [Path(p) for p in out.split("\0") if p]


def text_lines(path: Path) -> list[tuple[int, str]]:
    """Return ``(line number, text)`` pairs to check in ``path``.

    For a notebook the line numbers count lines of cell source, in cell order.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return list(enumerate(text.splitlines(), start=1))
    source: list[str] = []
    for cell in json.loads(text).get("cells", []):
        src = cell.get("source", "")
        source.extend(("".join(src) if isinstance(src, list) else src).splitlines())
    return list(enumerate(source, start=1))


def check(paths: list[Path]) -> list[str]:
    """Return one ``path:line: what: match`` message per offending match."""
    problems: list[str] = []
    for path in paths:
        if path.suffix.lower() in SKIP_SUFFIXES or path.resolve() == SELF:
            continue
        if not path.is_file():
            continue
        try:
            lines = text_lines(path)
        except UnicodeDecodeError:
            continue
        for lineno, line in lines:
            for pattern, what in PATTERNS:
                for match in pattern.finditer(line):
                    problems.append(f"{path}:{lineno}: {what}: {match.group()!r}")
    return problems


def main(argv: list[str]) -> int:
    """Check ``argv`` (or every tracked file) and print each problem."""
    paths = [Path(a) for a in argv] if argv else tracked_files()
    problems = check(paths)
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} reference(s) to notes outside the repository.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
