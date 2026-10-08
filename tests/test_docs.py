"""The rendered API reference cannot mis-assign defaults or types.

mkdocstrings renders a grouped numpydoc entry (`a, b : int`) by giving every
name the first name's default and type, so one parameter's default can show on
another (`ecme_tol` with default 100). Every entry of a `Parameters`,
`Attributes` or `Returns` section therefore names one parameter, which this
test checks for every public object of the package. It also pins that
`MTDR` documents its fitted attributes and that `simulate`'s rendered defaults
are the module constants' values, not their private names.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections.abc import Iterator
from typing import Any

import mtdr
import mtdr.model
import mtdr.simulation

SECTIONS = ("Parameters", "Attributes", "Returns", "Other Parameters")
GROUPED = re.compile(r"^\*{0,2}\w+(, \*{0,2}\w+)+(\s*:.*)?$")


def _public_objects() -> Iterator[tuple[str, Any]]:
    for info in pkgutil.iter_modules(mtdr.__path__, "mtdr."):
        if info.name.rsplit(".", 1)[1].startswith("_"):
            continue
        module = importlib.import_module(info.name)
        yield info.name, module
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != info.name:
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            yield f"{info.name}.{name}", obj
            if inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if attr.startswith("_"):
                        continue
                    func = member.fget if isinstance(member, property) else member
                    func = getattr(func, "__func__", func)
                    if inspect.isfunction(func):
                        yield f"{info.name}.{name}.{attr}", func


def _grouped_entries(doc: str) -> list[str]:
    lines = doc.splitlines()
    section, out = None, []
    for k, line in enumerate(lines):
        if k + 1 < len(lines) and re.fullmatch(r"-{3,}", lines[k + 1].strip()):
            section = line.strip()
            continue
        if section in SECTIONS and GROUPED.match(line):
            out.append(line)
    return out


def test_no_numpydoc_entry_names_several_parameters() -> None:
    found = {
        name: grouped
        for name, obj in _public_objects()
        if (grouped := _grouped_entries(inspect.getdoc(obj) or ""))
    }
    assert not found, found


def test_the_walk_sees_the_api() -> None:
    names = {name for name, _ in _public_objects()}
    assert {"mtdr.model.MTDR", "mtdr.model.MTDR.decode", "mtdr.plot.trajectories"} <= (
        names
    )
    assert "mtdr.simulation.simulate" in names
    assert _grouped_entries("Parameters\n----------\nY, X, mask\n    As fit.") == [
        "Y, X, mask"
    ]
    assert _grouped_entries("Parameters\n----------\na, b : int\n    Two.") == [
        "a, b : int"
    ]
    assert _grouped_entries("Notes\n-----\nY, X, mask\n") == []


def test_mtdr_documents_its_fitted_attributes() -> None:
    # The 23 fitted attributes, in the class docstring the site renders.
    doc = inspect.getdoc(mtdr.MTDR) or ""
    section = doc.split("Attributes\n----------\n", 1)[1].split("\nRaises\n", 1)[0]
    listed = re.findall(r"^(\w+_) : ", section, re.MULTILINE)
    assert len(listed) == len(set(listed)) == 23
    assert set(listed) == set(mtdr.model._FITTED)
    assert "marginal" in section
    assert "plug-in" in section


def test_simulate_shows_literal_defaults() -> None:
    # The site renders a default as written in the source: a private constant's
    # name there is noise. The literals must equal the constants.
    params = inspect.signature(mtdr.simulate).parameters
    assert params["length_scale"].default == mtdr.simulation._DEFAULT_LENGTH_SCALE
    assert params["amplitude"].default == mtdr.simulation._DEFAULT_AMPLITUDE
    source = inspect.getsource(mtdr.simulate)
    header = source[: source.index("-> SimulatedData:")]
    assert "_DEFAULT" not in header
