"""Argument normalisation shared by the public functions.

NumPy scalars and 0-d arrays count as scalars and 1-D arrays as sequences;
`bool` is never accepted where a number is expected; strings, bytes, sets and
mappings are not sequences. Every failure is reported by the caller as a
[`ParameterError`][mtdr.errors.ParameterError]: the predicates here
return `None` on failure, and the `require_*` helpers raise.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import numpy as np

from mtdr.errors import ParameterError

__all__ = [
    "RESERVED_NAMES",
    "as_bool",
    "as_int",
    "as_list",
    "as_real",
    "positive_int",
    "require_int_vector",
    "require_names",
    "require_non_negative_real",
    "unwrap_0d",
]

#: Keys `MTDR.explained_variance` returns besides the regressor names, so never a
#: regressor name.
RESERVED_NAMES = frozenset({"intercept", "total"})


def unwrap_0d(value: object) -> object:
    """Return a 0-d array's element, and anything else unchanged."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def as_int(value: object) -> int | None:
    """`value` as an `int` if it is an integer (NumPy included), not a `bool`."""
    value = unwrap_0d(value)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    return int(value)


def as_real(value: object) -> float | None:
    """`value` as a `float` if it is a real scalar (NumPy included), not a `bool`."""
    value = unwrap_0d(value)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


def as_list(value: object) -> list[object] | None:
    """`value` as a list if it is a sequence or a 1-D array, else `None`.

    Strings, bytes, sets, mappings and arrays of any other dimension are not
    sequences here.
    """
    if isinstance(value, np.ndarray):
        return list(value) if value.ndim == 1 else None
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        return None
    return list(value)


def as_bool(name: str, value: object) -> bool:
    """`value` if it is a `bool` or `numpy.bool_`; anything else raises."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    raise ParameterError(f"{name} must be a bool; got {value!r}")


def positive_int(name: str, value: object) -> int:
    """`value` as a positive `int`; anything else raises."""
    as_integer = as_int(value)
    if as_integer is None or as_integer < 1:
        raise ParameterError(f"{name} must be a positive integer; got {value!r}")
    return as_integer


def require_non_negative_real(name: str, value: object) -> float:
    """`value` as a finite `float` `>= 0`; anything else raises."""
    as_float = as_real(value)
    if as_float is None or not np.isfinite(as_float) or as_float < 0:
        raise ParameterError(f"{name} must be a finite real number >= 0; got {value!r}")
    return as_float


def require_int_vector(name: str, value: object, *, minimum: int) -> list[int]:
    """Return a non-empty sequence (or 1-D array) of `int >= minimum`, or raise."""
    entries = as_list(value)
    if entries is None:
        raise ParameterError(
            f"{name} must be a sequence of int or a 1-D integer array; got {value!r}"
        )
    if not entries:
        raise ParameterError(f"{name} must have at least one entry")
    out: list[int] = []
    for p, entry in enumerate(entries):
        as_integer = as_int(entry)
        if as_integer is None or as_integer < minimum:
            kind = {0: "a non-negative integer", 1: "a positive integer"}.get(
                minimum, f"an integer >= {minimum}"
            )
            raise ParameterError(f"{name}[{p}] must be {kind}; got {entry!r}")
        out.append(as_integer)
    return out


def require_names(value: object, n_regressors: int) -> tuple[str, ...]:
    """Return validated names; `None` gives `("x0", "x1", ...)`."""
    if value is None:
        return tuple(f"x{p}" for p in range(n_regressors))
    entries = as_list(value)
    if entries is None:
        raise ParameterError(
            f"regressor_names must be a sequence of str; got {value!r}"
        )
    if len(entries) != n_regressors:
        raise ParameterError(
            f"regressor_names has {len(entries)} entries; expected {n_regressors}"
        )
    for name in entries:
        if not isinstance(name, str):
            raise ParameterError(f"regressor names must be str; got {name!r}")
        if name in RESERVED_NAMES:
            raise ParameterError(f"regressor name {name!r} is reserved")
    names = tuple(str(name) for name in entries)
    if len(set(names)) != len(names):
        raise ParameterError(f"regressor names must be unique; got {names}")
    return names
