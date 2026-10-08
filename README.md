# mtdr

[![CI](https://github.com/jyanar/mtdr/actions/workflows/ci.yml/badge.svg)](https://github.com/jyanar/mtdr/actions/workflows/ci.yml)

Model-based targeted dimensionality reduction (mTDR) for neural population data,
in Python.

mTDR fits trial-by-neuron-by-time activity as a sum of low-rank, time-varying
regression coefficient matrices, one per task variable, and chooses each matrix's rank
by AIC. It handles neurons that were not recorded simultaneously through a per-trial
observation mask. The package provides the estimator `mtdr.MTDR` (fixed ranks or an AIC
rank search; predict, score, project, decode), data utilities, plots (`mtdr.plot`,
optional matplotlib) and a ground-truth simulator (`mtdr.simulate`).

Documentation: <https://jyanar.github.io/mtdr/>

## Installation

```bash
pip install mtdr                  # the core
pip install "mtdr[plot,xarray]"   # with the optional extras
```

The core needs only NumPy and SciPy (Python >= 3.11). The extras are optional:

- `plot`: matplotlib, for `mtdr.plot`;
- `xarray`: xarray and pandas, for `MTDR.to_xarray` and
  `RankSearchHistory.to_frame`.

A worked example is on the documentation's
[Getting started](https://jyanar.github.io/mtdr/getting-started/) page.

## Credit

mTDR is the method of

- M. C. Aoi, V. Mante & J. W. Pillow (2020). Prefrontal cortex exhibits
  multidimensional dynamic encoding during decision-making. *Nature
  Neuroscience* 23, 1410–1420. <https://doi.org/10.1038/s41593-020-0696-5>
- M. C. Aoi & J. W. Pillow (2018). Model-based targeted dimensionality reduction
  for neuronal population data. *Advances in Neural Information Processing
  Systems* 31.

This package is a Python port of the authors' MATLAB demo,
[`pillowlab/mTDRdemo`](https://github.com/pillowlab/mTDRdemo). It is not affiliated
with or endorsed by the authors; the method and the reference implementation are
theirs. If you use it, please cite the papers above as well as this software (see
[`CITATION.cff`](https://github.com/jyanar/mtdr/blob/main/CITATION.cff)).

## How this port was checked

`mtdr` was ported by reproducing the output of the published MATLAB code. The scripts
in [`tests/fixtures/matlab/`](https://github.com/jyanar/mtdr/tree/main/tests/fixtures/matlab) run the reference on its two
demos and on simulated datasets up to the scale of the 2020 paper, and save what it
computes. The test suite checks recorded quantities and reference-compatible rank
paths in Python within numerical tolerance;
default Python fits are checked separately for likelihood agreement.
Where the port deliberately departs from the MATLAB code (some defaults, a few fixed
bugs, SciPy's optimisers in place of `minFunc`), each difference and, where available,
a compatibility option is listed in
[`docs/differences-from-matlab.md`](https://github.com/jyanar/mtdr/blob/main/docs/differences-from-matlab.md). The model and
its estimation are described in [`docs/model.md`](https://github.com/jyanar/mtdr/blob/main/docs/model.md),
with technical appendices for the equations cited by the code.

## Development

```bash
pip install -e ".[dev]"   # the test, lint and docs tools
pre-commit install
ruff check . && ruff format --check .
mypy
pytest --cov=mtdr
mkdocs serve
```

## License

MIT (see [`LICENSE`](https://github.com/jyanar/mtdr/blob/main/LICENSE)).
