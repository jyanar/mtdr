"""Read-only containers shared by the frozen result classes.

NumPy restores writeable arrays when a pickled object is loaded, and unpickling
a frozen dataclass does not run `__post_init__`, so every result class that
promises read-only arrays re-freezes them in `__setstate__` through
[`restore_frozen`][mtdr._frozen.restore_frozen].
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, TypeVar

import numpy as np

__all__ = ["ReadOnlyMapping", "freeze", "restore_frozen"]

_V = TypeVar("_V")


class ReadOnlyMapping(Mapping[str, _V]):
    """A read-only, picklable view of a dict.

    `types.MappingProxyType` cannot be pickled, so a result holding one could
    not be sent to a worker process; this thin wrapper can.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, _V]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> _V:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __getstate__(self) -> dict[str, _V]:
        return self._data

    def __setstate__(self, state: dict[str, _V]) -> None:
        self._data = state
        freeze(state)


def freeze(value: Any) -> None:
    """Make every NumPy array in `value` read-only, descending into containers."""
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict | ReadOnlyMapping):
        for item in value.values():
            freeze(item)
    elif isinstance(value, tuple | list):
        for item in value:
            freeze(item)


def restore_frozen(obj: object, state: dict[str, Any]) -> None:
    """`__setstate__` body for a frozen dataclass: restore `state`, re-freeze arrays."""
    freeze(state)
    obj.__dict__.update(state)
