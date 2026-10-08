"""CI benchmark with a regression threshold.

Usage (from the repository root)::

    python benchmarks/ci_bench.py                    # measure, compare (1: regression)
    python benchmarks/ci_bench.py --json out.json    # also write the measurement
    python benchmarks/ci_bench.py --combine a.json b.json c.json d.json
        --source "CI runs ..."                        # write the baseline from runs

Two gates, against the baseline stored for the platform
(`benchmarks/baseline.json`, keyed by `platform.system()`):

* **time**: each workload's raw seconds (the median of its repeats) must stay
  within `THRESHOLD` (1.5) times the baseline's, which is the per-workload
  median over several runs of the CI job. Four ubuntu-latest runs differed
  from their median by at most a factor of 1.23, so a doubling reads at least
  1.8 and fails, while a false alarm needs a runner 1.5 times slower than the
  median.
* **work**, independent of the runner's speed: the paper-scale `fit_mmle`'s
  outer iteration counts (`n_iter`) must equal the baseline's, its total
  number of inner L-BFGS-B iterations (counted by wrapping
  `scipy.optimize.minimize`) must stay within `THRESHOLD` times the
  baseline's (inner counts are CPU-dependent, so not exact), and the fit must
  report `converged=True`.

A platform without a baseline passes with a message, except Linux, where CI
runs: there a missing baseline fails. The measurement is printed as JSON, so
a new baseline can be combined from the job's logs (`--combine`); re-record
when the runner image changes, or in the commit of a deliberate slowdown.
The workloads:

* `stats_svd_paper`: `sufficient_statistics` and `fit_svd` at the paper's
  scale ($n=800$, $T=15$, $N=1000$, ranks $(5,4,3,4,2,2)$);
* `fit_mmle_demo`: one `fit_mmle` at the demo's scale ($n=100$, $T=15$,
  $N=400$, ranks $(2,1,3)$), at the package defaults;
* `search_demo`: `MTDR(ranks="aic").fit` there (the default two-stage search);
* `fit_mmle_paper`: one `fit_mmle` at the paper's scale, once.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize

import mtdr
from mtdr import MTDR
from mtdr.errors import ConvergenceWarning
from mtdr.mmle import fit_mmle
from mtdr.stats import sufficient_statistics
from mtdr.svd_fit import fit_svd

BASELINE = Path(__file__).resolve().parent / "baseline.json"
THRESHOLD = 1.5
PAPER_RANKS = [5, 4, 3, 4, 2, 2]
DEMO_RANKS = [2, 1, 3]
#: Platforms on which a missing baseline fails (CI's).
REQUIRED = ("Linux",)


def _median_time(func: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    times = []
    out = None
    for _ in range(repeats):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            t0 = time.perf_counter()
            out = func()
            times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def _counting_lbfgs(func: Callable[[], Any]) -> tuple[Any, int]:
    """Run `func`, returning its result and the L-BFGS-B iterations it made."""
    real = scipy.optimize.minimize
    total = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal total
        res = real(*args, **kwargs)
        total += int(getattr(res, "nit", 0))
        return res

    scipy.optimize.minimize = counted
    try:
        out = func()
    finally:
        scipy.optimize.minimize = real
    return out, total


def measure() -> dict[str, Any]:
    """Time every workload; return the seconds and the work done."""
    paper = mtdr.simulate(
        n_neurons=800,
        n_bins=15,
        n_trials=1000,
        ranks=PAPER_RANKS,
        drop_prob=0.3,
        seed=0,
    )
    demo = mtdr.simulate(
        n_neurons=100, n_bins=15, n_trials=400, ranks=DEMO_RANKS, drop_prob=0.3, seed=0
    )
    demo_stats = sufficient_statistics(demo.Y_masked, demo.X, demo.mask)
    paper_stats = sufficient_statistics(paper.Y_masked, paper.X, paper.mask)

    def stats_svd() -> Any:
        st = sufficient_statistics(paper.Y_masked, paper.X, paper.mask)
        return fit_svd(st, PAPER_RANKS)

    results: dict[str, dict[str, Any]] = {}
    t, _ = _median_time(stats_svd, 5)
    results["stats_svd_paper"] = {"seconds": t, "work": {}}
    t, fit = _median_time(lambda: fit_mmle(demo_stats, DEMO_RANKS), 5)
    results["fit_mmle_demo"] = {"seconds": t, "work": {"n_iter": dict(fit.n_iter)}}
    t, model = _median_time(lambda: MTDR(ranks="aic").fit(demo.Y_masked, demo.X), 3)
    results["search_demo"] = {
        "seconds": t,
        "work": {"ranks": list(model.ranks_.values())},
    }
    t, (fit, nit) = _median_time(
        lambda: _counting_lbfgs(lambda: fit_mmle(paper_stats, PAPER_RANKS)), 1
    )
    results["fit_mmle_paper"] = {
        "seconds": t,
        "work": {
            "n_iter": dict(fit.n_iter),
            "lbfgs_iterations": nit,
            "converged": bool(fit.converged),
        },
    }
    return {
        "platform": platform.system(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "workloads": results,
    }


def compare(measured: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Return the regressions of the time and work gates."""
    failures = []
    for name, entry in measured["workloads"].items():
        ref = baseline["seconds"].get(name)
        if ref is None:
            failures.append(f"{name}: no baseline time")
            continue
        ratio = entry["seconds"] / ref
        entry["ratio"] = ratio
        if ratio > THRESHOLD:
            failures.append(
                f"{name}: {ratio:.2f} x the baseline median time (threshold "
                f"{THRESHOLD})"
            )
    work = measured["workloads"]["fit_mmle_paper"]["work"]
    ref_work = baseline["work"]["fit_mmle_paper"]
    if work["n_iter"] != ref_work["n_iter"]:
        failures.append(
            f"fit_mmle_paper: n_iter {work['n_iter']}, baseline {ref_work['n_iter']}"
        )
    ratio = work["lbfgs_iterations"] / ref_work["lbfgs_iterations"]
    if ratio > THRESHOLD:
        failures.append(
            f"fit_mmle_paper: {work['lbfgs_iterations']} L-BFGS-B iterations, "
            f"{ratio:.2f} x the baseline's {ref_work['lbfgs_iterations']} (threshold "
            f"{THRESHOLD})"
        )
    if not work["converged"]:
        failures.append("fit_mmle_paper: converged=False")
    return failures


def combine(paths: list[Path], source: str) -> dict[str, Any]:
    """Return a baseline entry: per-workload medians over several measurements."""
    runs = [json.loads(path.read_text()) for path in paths]
    systems = {run["platform"] for run in runs}
    if len(systems) != 1:
        raise SystemExit(f"measurements from several platforms: {sorted(systems)}")
    names = list(runs[0]["workloads"])
    seconds = {
        name: statistics.median(run["workloads"][name]["seconds"] for run in runs)
        for name in names
    }
    n_iters = [run["workloads"]["fit_mmle_paper"]["work"]["n_iter"] for run in runs]
    if any(n != n_iters[0] for n in n_iters):
        raise SystemExit(f"fit_mmle_paper's n_iter differs between runs: {n_iters}")
    if not all(run["workloads"]["fit_mmle_paper"]["work"]["converged"] for run in runs):
        raise SystemExit("a run's fit_mmle_paper did not converge")
    nits = [
        run["workloads"]["fit_mmle_paper"]["work"]["lbfgs_iterations"] for run in runs
    ]
    spread = {
        name: [run["workloads"][name]["seconds"] / seconds[name] for run in runs]
        for name in names
    }
    return {
        "source": source,
        "runs": len(runs),
        "python": runs[0]["python"],
        "numpy": runs[0]["numpy"],
        "seconds": seconds,
        "seconds_over_median": spread,
        "work": {
            "fit_mmle_paper": {
                "n_iter": n_iters[0],
                "lbfgs_iterations": int(statistics.median(nits)),
                "lbfgs_iterations_runs": nits,
            }
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; the exit status is 1 on a regression."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--combine", type=Path, nargs="+")
    parser.add_argument("--source", default="")
    args = parser.parse_args(argv)
    stored = json.loads(BASELINE.read_text()) if BASELINE.is_file() else {}
    if args.combine:
        entry = combine(args.combine, args.source)
        system = json.loads(args.combine[0].read_text())["platform"]
        stored[system] = entry
        BASELINE.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
        print(json.dumps(entry, indent=2, sort_keys=True))
        return 0
    measured = measure()
    system = measured["platform"]
    failures: list[str] = []
    if system in stored:
        failures = compare(measured, stored[system])
    elif system in REQUIRED:
        failures = [f"no baseline for {system}, where CI runs"]
    else:
        print(f"no baseline for {system}: nothing to compare (passes)")
    print(f"mtdr {mtdr.__version__}, numpy {measured['numpy']}, {system}")
    print("| workload | seconds | x baseline | work |")
    print("|---|---|---|---|")
    for name, entry in measured["workloads"].items():
        ratio = entry.get("ratio")
        shown = f"{ratio:.2f}" if ratio is not None else "-"
        print(f"| {name} | {entry['seconds']:.3f} | {shown} | {entry['work']} |")
    text = json.dumps(measured, indent=2, sort_keys=True)
    print("MEASUREMENT-JSON-BEGIN")
    print(text)
    print("MEASUREMENT-JSON-END")
    if args.json:
        args.json.write_text(text + "\n")
    for failure in failures:
        print(f"REGRESSION: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
