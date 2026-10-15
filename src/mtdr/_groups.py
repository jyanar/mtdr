r"""Regressor groups in the positional layers: which columns share one latent weight.

The columns $p$ of a group $g$ share one neural basis, $B_p=W_gS_p^\top$
(`docs/model.md` § 11). [`mtdr.mmle`][mtdr.mmle], [`mtdr.svd_fit`][mtdr.svd_fit]
and [`mtdr.aic`][mtdr.aic] take `groups` as a sequence of sequences of column
indices, checked here; the name-level rules of
[`MTDR(groups=...)`][mtdr.model.MTDR] live in
[`resolve_groups`][mtdr.data.resolve_groups].
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mtdr._args import as_int, as_list, require_int_vector
from mtdr.errors import ParameterError

__all__ = ["Grouping", "as_grouping", "check_grouping", "require_ranks"]


@dataclass(frozen=True)
class Grouping:
    """A partition of `range(n_columns)`: `groups[g]` holds group `g`'s columns.

    Members are increasing and groups are ordered by their smallest column, so
    the order is a function of the partition; build one with `check_grouping`.
    """

    groups: tuple[tuple[int, ...], ...]

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_columns(self) -> int:
        return sum(self.sizes)

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(len(g) for g in self.groups)

    @property
    def group_of(self) -> tuple[int, ...]:
        """The group of each column."""
        out = [0] * self.n_columns
        for g, members in enumerate(self.groups):
            for p in members:
                out[p] = g
        return tuple(out)

    @property
    def trivial(self) -> bool:
        """Whether every group is one column: the ungrouped model."""
        return max(self.sizes) == 1

    def column_ranks(self, group_ranks: Sequence[int]) -> list[int]:
        """Each column's width, its group's rank; one rank per group."""
        if len(group_ranks) != self.n_groups:
            raise ParameterError(
                f"{len(group_ranks)} ranks for {self.n_groups} groups; expected one "
                "per group"
            )
        return [int(group_ranks[g]) for g in self.group_of]

    @classmethod
    def singletons(cls, n_columns: int) -> Grouping:
        """Every column alone, kept explicit (to force the tie in tests)."""
        return cls(tuple((p,) for p in range(n_columns)))


def check_grouping(groups: object, n_columns: int | None) -> Grouping:
    """Return the `Grouping` of `groups`, or raise a `ParameterError` naming the entry.

    `groups` must be non-empty sequences of int column indices that partition
    `range(n_columns)` (of their total size when `n_columns` is `None`),
    ordered by smallest column: positional ranks follow that order, so it is
    checked, not changed. Members are sorted.
    """
    entries = as_list(groups)
    if not entries:
        raise ParameterError(
            "groups must be a non-empty sequence of sequences of column indices; got "
            f"{groups!r}"
        )
    out: list[tuple[int, ...]] = []
    seen: dict[int, int] = {}
    for g, entry in enumerate(entries):
        members = as_list(entry)
        if not members:
            raise ParameterError(
                f"groups[{g}] must be a non-empty sequence of column indices; got "
                f"{entry!r}"
            )
        for member in members:
            p = as_int(member)
            if p is None or p < 0 or (n_columns is not None and p >= n_columns):
                bound = "" if n_columns is None else f" in [0, {n_columns})"
                raise ParameterError(
                    f"groups[{g}] holds {member!r}, which is not a column index{bound}"
                )
            if p in seen:
                where = "twice" if seen[p] == g else f"in groups {seen[p]} and {g}"
                raise ParameterError(f"column {p} appears {where} in groups")
            seen[p] = g
        out.append(tuple(sorted(p for p, h in seen.items() if h == g)))
    total = len(seen) if n_columns is None else n_columns
    missing = sorted(set(range(total)) - set(seen))
    if missing:
        raise ParameterError(
            f"groups must partition the {total} columns; columns {missing} are in no "
            "group (a column of its own is a group of one)"
        )
    ordered = sorted(out, key=lambda members: members[0])
    if ordered != out:
        raise ParameterError(
            "groups must be ordered by their smallest column, the order ranks follow; "
            f"got {[list(m) for m in out]}, expected {[list(m) for m in ordered]}"
        )
    return Grouping(tuple(out))


def as_grouping(groups: object, n_columns: int | None) -> Grouping | None:
    """`check_grouping`, but `None` for `None` or every column alone.

    The positional layers treat such a grouping exactly as `groups=None`: the
    same code path, numbers and fit objects.
    """
    if groups is None:
        return None
    grouping = check_grouping(groups, n_columns)
    return None if grouping.trivial else grouping


def require_ranks(
    ranks: object,
    grouping: Grouping | None,
    n_regressors: int | None,
    n_neurons: int,
    n_bins: int,
) -> list[int]:
    r"""Return one rank per group, each in $[0,\min(n,\lvert g\rvert T)]$, or raise.

    Without a grouping the number of ranks is checked against `n_regressors`
    (when given) and the messages are those of the ungrouped layers.
    """
    rank_list = require_int_vector("ranks", ranks, minimum=0)
    if grouping is None:
        if n_regressors is not None and len(rank_list) != n_regressors:
            raise ParameterError(
                f"ranks has {len(rank_list)} entries; expected one per regressor "
                f"({n_regressors}), without the intercept"
            )
        cap = min(n_neurons, n_bins)
        for p, r in enumerate(rank_list):
            if r > cap:
                raise ParameterError(
                    f"ranks[{p}] is {r}, above min(n_neurons, n_bins) = {cap}"
                )
        return rank_list
    if len(rank_list) != grouping.n_groups:
        raise ParameterError(
            f"ranks has {len(rank_list)} entries; expected one per group "
            f"({grouping.n_groups})"
        )
    for g, (r, size) in enumerate(zip(rank_list, grouping.sizes, strict=True)):
        cap = min(n_neurons, size * n_bins)
        if r > cap:
            raise ParameterError(
                f"ranks[{g}] (the group of columns {list(grouping.groups[g])}) is {r}, "
                f"above min(n_neurons, {size} * n_bins) = {cap}"
            )
    return rank_list
