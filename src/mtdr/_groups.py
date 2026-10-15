r"""Regressor groups in the positional layers: which columns share one latent weight.

A regressor group is a set of design columns whose coefficient matrices share one
neural basis, $B_p=W_gS_p^\top$ for every column $p$ of group $g$
(`docs/model.md` § 11). The positional layers ([`mtdr.mmle`][mtdr.mmle],
[`mtdr.svd_fit`][mtdr.svd_fit], [`mtdr.aic`][mtdr.aic]) take `groups` as a
sequence of sequences of column indices; [`Grouping`][mtdr._groups.Grouping]
checks it. The name-level rules of [`MTDR(groups=...)`][mtdr.model.MTDR] live in
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
    """A partition of the columns `range(n_columns)` into ordered groups.

    `groups[g]` holds group `g`'s columns in increasing order, and the groups
    are ordered by their smallest column, so the order is a function of the
    partition alone. Build one with [`check_grouping`][mtdr._groups.check_grouping].

    Attributes
    ----------
    groups : tuple of tuple of int
        The columns of each group.
    """

    groups: tuple[tuple[int, ...], ...]

    @property
    def n_groups(self) -> int:
        """Number of groups."""
        return len(self.groups)

    @property
    def n_columns(self) -> int:
        """Number of columns the groups partition."""
        return sum(len(g) for g in self.groups)

    @property
    def sizes(self) -> tuple[int, ...]:
        r"""$\lvert g\rvert$, the number of columns of each group."""
        return tuple(len(g) for g in self.groups)

    @property
    def group_of(self) -> tuple[int, ...]:
        """The group of each column, `(n_columns,)`."""
        out = [0] * self.n_columns
        for g, members in enumerate(self.groups):
            for p in members:
                out[p] = g
        return tuple(out)

    @property
    def trivial(self) -> bool:
        """Whether every group is a single column: the ungrouped model."""
        return all(len(g) == 1 for g in self.groups)

    def column_ranks(self, group_ranks: Sequence[int]) -> list[int]:
        """Return each column's width, the rank of its group.

        Parameters
        ----------
        group_ranks : sequence of int
            One rank per group, in group order.

        Returns
        -------
        list of int
            `(n_columns,)`, the rank of each column's group.
        """
        if len(group_ranks) != self.n_groups:
            raise ParameterError(
                f"{len(group_ranks)} ranks for {self.n_groups} groups; expected one "
                "per group"
            )
        return [int(group_ranks[g]) for g in self.group_of]

    @classmethod
    def singletons(cls, n_columns: int) -> Grouping:
        """Every column its own group (the ungrouped model, kept explicit).

        Parameters
        ----------
        n_columns : int
            Number of columns, positive.

        Returns
        -------
        Grouping
            `((0,), (1,), ...)`.
        """
        return cls(tuple((p,) for p in range(n_columns)))


def check_grouping(groups: object, n_columns: int | None) -> Grouping:
    """Check `groups` and return its `Grouping`, trivial or not.

    `groups` must be a sequence of non-empty sequences of column indices (ints,
    not bools; a string is not a sequence) that partition `range(n_columns)`
    (`range` of their total size when `n_columns` is `None`), ordered by their
    smallest column. Members are sorted within each group; the order of the
    groups is not changed, because positional ranks follow it.

    Parameters
    ----------
    groups : object
        The `groups` argument of a positional layer.
    n_columns : int or None
        The number of design columns, when known.

    Returns
    -------
    Grouping
        The checked partition.

    Raises
    ------
    ParameterError
        For any violation, naming the offending entry.
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
        columns: list[int] = []
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
            columns.append(p)
        out.append(tuple(sorted(columns)))
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
    """Check `groups`; `None` for `None` or a partition into single columns.

    The positional layers treat a trivial grouping exactly as `groups=None`:
    the same code path, the same numbers and fit objects whose `groups` is
    `None`.

    Parameters
    ----------
    groups : object
        `None`, or a sequence of sequences of column indices.
    n_columns : int or None
        The number of design columns, when known.

    Returns
    -------
    Grouping or None
        The grouping, `None` when it is the ungrouped model.
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
    r"""Check per-group ranks against their caps $\min(n,\lvert g\rvert T)$.

    Without a grouping each column is its own group and the messages are those
    of the ungrouped layers.

    Parameters
    ----------
    ranks : object
        The `ranks` argument.
    grouping : Grouping or None
        The groups, `None` for single columns.
    n_regressors : int or None
        The number of columns, when the caller knows it (checked against the
        number of ranks without a grouping).
    n_neurons : int
        $n$.
    n_bins : int
        $T$.

    Returns
    -------
    list of int
        The ranks, one per group.

    Raises
    ------
    ParameterError
        For a wrong number of ranks or a rank outside $[0,\min(n,\lvert
        g\rvert T)]$, naming the group's columns.
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
    for g, (r, members) in enumerate(zip(rank_list, grouping.groups, strict=True)):
        cap = min(n_neurons, len(members) * n_bins)
        if r > cap:
            raise ParameterError(
                f"ranks[{g}] (the group of columns {list(members)}) is {r}, above "
                f"min(n_neurons, {len(members)} * n_bins) = {cap}"
            )
    return rank_list
