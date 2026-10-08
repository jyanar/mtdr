"""Benchmark `fit_mmle` and the MMLE rank search at the paper's scale.

Usage (from the repository root)::

    python benchmarks/bench_mmle.py                  # fits only, 5 seeds
    python benchmarks/bench_mmle.py --search         # also the two-stage search
    python benchmarks/bench_mmle.py --profile        # cProfile one fit

Two data sources at the paper's scale ($n=800$, $T=15$, ranks
$(5,4,3,4,2,2)$, $r_{tot}=20$):

* ``simulate``: `mtdr.simulate(n_neurons=800, n_bins=15, n_trials=1000,
  drop_prob=0.3)` at each seed (each neuron sees about 700 trials, three
  regressors' worth of the demo's Exponential precisions);
* ``paper``: the paper-scale parity dataset of `tests/paper_data.py` (100
  sessions of 8 neurons, 91-288 trials per neuron, six correlated regressors),
  which has no seed: it is fitted once.

For every fit it prints the wall time of each stage, the iteration counts
(`MMLEFit.n_iter`), `converged`, and the first words of any
`ConvergenceWarning`. With ``--search`` it runs the default rank search of
`MTDR` (the SVD AIC search on the precision-weighted truncation from all ones,
then the MMLE AIC search from its ranks, selecting with the reference count
(M38) as `MTDR` does) and prints the chosen ranks, the number of fits and the
time. Pure NumPy/SciPy; no new dependency. Times are for one process on an
otherwise idle machine; run nothing else meanwhile.
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import pstats
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

import mtdr
from mtdr.aic import aic, n_parameters_mmle
from mtdr.errors import ConvergenceWarning
from mtdr.mmle import MMLEFit, ecme, fit_mmle, refine
from mtdr.rank_search import greedy_aic
from mtdr.stats import SufficientStats, sufficient_statistics
from mtdr.svd_fit import fit_svd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import paper_data

RANKS = [5, 4, 3, 4, 2, 2]
R = TypeVar("R")


def _timed(func: Callable[[], R]) -> tuple[R, float, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        out = func()
        elapsed = time.perf_counter() - t0
    return out, elapsed, [str(w.message) for w in caught]


def _data(source: str, seed: int) -> tuple[SufficientStats, float]:
    if source == "paper":
        d = paper_data.paper_data()
        Y, X, mask = d.Y, d.X, d.mask
    else:
        sim = mtdr.simulate(
            n_neurons=800,
            n_bins=15,
            n_trials=1000,
            ranks=RANKS,
            drop_prob=0.3,
            seed=seed,
        )
        Y, X, mask = sim.Y_masked, sim.X, sim.mask
    stats, t_stats, _ = _timed(lambda: sufficient_statistics(Y, X, mask))
    return stats, t_stats


def bench_fit(source: str, seed: int) -> dict[str, Any]:
    """Time one `fit_mmle` at the true ranks, stage by stage."""
    stats, t_stats = _data(source, seed)
    svd, t_svd, _ = _timed(lambda: fit_svd(stats, RANKS))
    start = MMLEFit.from_parameters(stats, svd.S, svd.noise_precision, stats.Y_mean)
    (warm, _trace), t_ecme, _w_ecme = _timed(lambda: ecme(stats, start))
    fit, t_refine, _w_refine = _timed(lambda: refine(stats, warm))
    whole, t_fit, w_fit = _timed(lambda: fit_mmle(stats, RANKS))
    assert whole.log_likelihood == fit.log_likelihood
    return {
        "source": source,
        "seed": seed,
        "stats": t_stats,
        "svd": t_svd,
        "ecme": t_ecme,
        "refine": t_refine,
        "fit_mmle": t_fit,
        "n_iter": dict(whole.n_iter),
        "converged": whole.converged,
        "warnings": [m.split(";")[0][:90] for m in w_fit],
    }


def bench_search(source: str, seed: int) -> dict[str, Any]:
    """Time the two-stage search: weighted SVD from ones, then MMLE."""
    stats, _ = _data(source, seed)
    t0 = time.perf_counter()
    _, svd_hist = greedy_aic(
        functools.partial(fit_svd, stats, precision_weighted=True),
        lambda f, r: f.aic,
        [1] * len(RANKS),
        15,
    )
    t_svd = time.perf_counter() - t0
    n_fits = 0
    non_converged = 0

    def selection_aic(f: MMLEFit, r: Any) -> float:
        # The count MTDR's search selects with, (M38).
        count = n_parameters_mmle(r, stats.n_neurons, stats.n_bins, formula="reference")
        return aic(f.log_likelihood, count)

    def fit(r: Any) -> MMLEFit:
        nonlocal n_fits, non_converged
        out, _, _caught = _timed(lambda: fit_mmle(stats, r))
        n_fits += 1
        non_converged += int(not out.converged)
        return out

    (_best, hist), t_mmle, _ = _timed(
        lambda: greedy_aic(fit, selection_aic, svd_hist.ranks[-1], 15)
    )
    return {
        "source": source,
        "seed": seed,
        "svd_ranks": svd_hist.ranks[-1].tolist(),
        "mmle_ranks": hist.ranks[-1].tolist(),
        "svd_seconds": t_svd,
        "mmle_seconds": t_mmle,
        "n_fits": n_fits,
        "non_converged": non_converged,
        "accepted_converged": list(hist.converged),
    }


def profile_fit(source: str, seed: int) -> None:
    """CProfile one `fit_mmle`; print the 15 most expensive calls (cumulative)."""
    stats, _ = _data(source, seed)
    prof = cProfile.Profile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        prof.enable()
        fit_mmle(stats, RANKS)
        prof.disable()
    pstats.Stats(prof).sort_stats("cumulative").print_stats(15)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--no-paper", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    print(f"mtdr {mtdr.__version__}, numpy {np.__version__}")
    if args.profile:
        profile_fit("simulate", args.seeds[0])
        return
    runs = [("simulate", s) for s in args.seeds]
    if not args.no_paper:
        runs.append(("paper", 0))
    print(
        "| data | seed | stats | fit_svd | ecme | refine | fit_mmle | n_iter "
        "| converged | warnings |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for source, seed in runs:
        r = bench_fit(source, seed)
        print(
            f"| {source} | {seed} | {r['stats']:.2f} s | {r['svd'] * 1e3:.0f} ms "
            f"| {r['ecme']:.2f} s | {r['refine']:.1f} s | {r['fit_mmle']:.1f} s "
            f"| {r['n_iter']} | {r['converged']} | {r['warnings']} |",
            flush=True,
        )
    if args.search:
        print(
            "\n| data | seed | weighted SVD ranks | MMLE ranks | SVD time "
            "| MMLE time | MMLE fits | not converged |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for source, seed in runs:
            r = bench_search(source, seed)
            print(
                f"| {source} | {seed} | {r['svd_ranks']} | {r['mmle_ranks']} "
                f"| {r['svd_seconds']:.1f} s | {r['mmle_seconds']:.0f} s "
                f"| {r['n_fits']} | {r['non_converged']} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
