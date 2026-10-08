# mtdr

Model-based targeted dimensionality reduction (mTDR) for neural population data,
in Python.

mTDR models population activity on trial $k$ as a sum of low-rank,
time-varying regression coefficient matrices, one per task variable:

$$
Y_k = b + \sum_{p=1}^{P} x_k^{(p)}\, W_p S_p^{\top} + E_k,
\qquad \operatorname{rank}(W_p S_p^{\top}) = r_p ,
$$

and chooses each rank $r_p$ by AIC. A per-trial observation mask lets neurons
recorded in different sessions be fitted jointly.

Start with [Getting started](getting-started.md) to fit, evaluate and decode data
with [`MTDR`][mtdr.model.MTDR]. [The model and its estimation](model.md) explains
the assumptions and outputs; [Differences from MATLAB](differences-from-matlab.md)
describes validation and departures from the reference. Data utilities, plotting,
simulation and functional estimators are documented in the API reference.

!!! note "Status: alpha"
    The API may still change between minor versions before 1.0; changes are listed in
    the [Changelog](changelog.md).

## Installation

```bash
pip install mtdr                  # the core: NumPy and SciPy
pip install "mtdr[plot,xarray]"   # with the optional extras
```

The core needs only NumPy and SciPy (Python >= 3.11). The extras are optional:

- `plot`: matplotlib, for `mtdr.plot`;
- `xarray`: xarray and pandas, for `MTDR.to_xarray` and
  `RankSearchHistory.to_frame`.

For development (tests, linters, this site): `pip install -e ".[dev]"`.

## Notation and references

- **(Mk)** is equation k of [The model and its estimation](model.md), the derivation the
  code implements; the docstrings cite these numbers.
- **NeurIPS supplement** and **Nature Neuroscience supplement** refer to the
  supplements of [Aoi & Pillow (2018)](https://proceedings.neurips.cc/paper/2018/hash/8a1ee9f2b7abe6e88d1a479ab6a42c5e-Abstract.html)
  and [Aoi, Mante & Pillow (2020)](https://doi.org/10.1038/s41593-020-0696-5).
  Citations specify a section (§), equation (eq.) or algorithm within the supplement.

## Credit

mTDR was introduced by Aoi & Pillow (NeurIPS 2018) and Aoi, Mante & Pillow
(*Nature Neuroscience* 2020). This package is an independent Python port of the
authors' MATLAB demo, [`pillowlab/mTDRdemo`](https://github.com/pillowlab/mTDRdemo),
made by reproducing that code's output; it is not affiliated with or endorsed by the
authors. Please cite both papers along with this software.
[Differences from MATLAB](differences-from-matlab.md) explains how the port was checked,
where its defaults and results differ from the demo's, and which options restore the
reference's behaviour.
