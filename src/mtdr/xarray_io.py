"""The `MTDR.to_xarray` backend (optional `xarray` extra).

[`to_dataset`][mtdr.xarray_io.to_dataset] packs a fitted
[`MTDR`][mtdr.model.MTDR] into an `xarray.Dataset`. `xarray` is imported when
the function is called, so `import mtdr` never needs it; without it the call
raises `ImportError` with the install hint. Coordinates are positional
(`neuron = arange(n_neurons_)`, `time = arange(n_bins_)`): the package never
sees unit IDs or bin times, and the caller can relabel them afterwards. There
is no `xarray` accessor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mtdr._version import __version__

if TYPE_CHECKING:
    import xarray

    from mtdr.model import MTDR

__all__ = ["to_dataset"]


def to_dataset(model: MTDR) -> xarray.Dataset:
    """Return a fitted model as an `xarray.Dataset`.

    Dims `regressor`, `neuron`, `time`, `component`. Variables: `B`
    `(regressor, neuron, time)`, the coefficient matrices $B_p$ of (M37);
    `W` `(regressor, neuron, component)` and `S` `(regressor, time,
    component)`, in the model's frame, with `component` running to
    `max(ranks_)` and `NaN` beyond each regressor's rank; `rank`
    `(regressor,)`; `intercept` `(neuron, time)` (absent without one); and
    `noise_precision` `(neuron,)`. Attributes: `estimator`, `aic`,
    `log_likelihood`, `n_parameters`, `mtdr_version`.

    Parameters
    ----------
    model : MTDR
        A fitted model.

    Returns
    -------
    xarray.Dataset
        The dataset.

    Raises
    ------
    ImportError
        If xarray is not installed.

    Examples
    --------
    >>> from mtdr import MTDR, simulate
    >>> from mtdr.xarray_io import to_dataset
    >>> sim = simulate(n_neurons=10, n_bins=5, n_trials=80, ranks=[2, 1], seed=0)
    >>> ds = to_dataset(MTDR(ranks=[2, 1], estimator="svd").fit(sim.Y, sim.X))
    >>> ds["rank"].values.tolist(), bool(ds["W"].isnull()[1, :, 1].all())
    ([2, 1], True)
    """
    try:
        import xarray as xr
    except ImportError as err:
        raise ImportError(
            "MTDR.to_xarray needs xarray: pip install 'mtdr[xarray]' (from a clone "
            "of the repository: pip install '.[xarray]' in it, or pip install "
            "xarray pandas)"
        ) from err
    names = list(model.regressor_names_)
    ranks = [model.ranks_[name] for name in names]
    n, T, P = model.n_neurons_, model.n_bins_, len(names)
    width = max(ranks)
    W = np.full((P, n, width), np.nan)
    S = np.full((P, T, width), np.nan)
    for p, name in enumerate(names):
        r = ranks[p]
        W[p, :, :r] = model.W_[name]
        S[p, :, :r] = model.S_[name]
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "B": (("regressor", "neuron", "time"), np.stack([model.B_[m] for m in names])),
        "W": (("regressor", "neuron", "component"), W),
        "S": (("regressor", "time", "component"), S),
        "rank": (("regressor",), np.array(ranks, dtype=np.int64)),
    }
    if model.intercept_ is not None:
        data_vars["intercept"] = (("neuron", "time"), np.array(model.intercept_))
    data_vars["noise_precision"] = (("neuron",), np.array(model.noise_precision_))
    return xr.Dataset(
        data_vars,
        coords={
            "regressor": names,
            "neuron": np.arange(n),
            "time": np.arange(T),
            "component": np.arange(width),
        },
        attrs={
            "estimator": model.estimator_,
            "aic": model.aic_,
            "log_likelihood": model.log_likelihood_,
            "n_parameters": model.n_parameters_,
            "mtdr_version": __version__,
        },
    )
