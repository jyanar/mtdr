"""The inner stopping rules of `refine`: progtol, the caps and rounding-level ends.

The basis step of `refine` stops on an absolute function change, minFunc's
`progTol`, implemented as L-BFGS-B `ftol = progtol / max(|f0|, 1)` with `f0`
the objective at the run's start; the precision step keeps a fixed relative
`ftol = 1e-11`. Both are capped at 2000 iterations by default.

An L-BFGS-B end with status 2 (a line search that found no decrease) counts
as convergence when it is at rounding level. A basis-step end does when its
last accepted decrease is at most `100 max(progtol, eps M)`, `M` the sum of
the objective's term magnitudes. A precision-step end is first polished by at
most three diagonal Newton steps, and does when every scale-free residual
`|lambda_i E_i / (n_i T) - 1|` is then at most `PRECISION_RESIDUAL_TOL = 1e-8`.
Caps, rejected points and non-finite values stay failures. The tests inject
`scipy.optimize.minimize` results to reach each branch.
"""

from __future__ import annotations

import inspect
import re
import types
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import scipy.optimize
from hypothesis import given
from hypothesis import strategies as st

import mtdr
import parity_fixtures as pf
from mtdr import mmle
from mtdr.errors import ConvergenceWarning, ParameterError
from mtdr.stats import SufficientStats, sufficient_statistics

REAL_MINIMIZE = scipy.optimize.minimize


def _stats(seed: int = 3) -> SufficientStats:
    sim = mtdr.simulate(n_neurons=20, n_bins=6, n_trials=80, ranks=[1, 2], seed=seed)
    return sufficient_statistics(sim.Y, sim.X, sim.mask)


def _fit(stats: SufficientStats) -> mmle.MMLEFit:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return mmle.fit_mmle(stats, [1, 2])


# ----------------------------------------------------------------- defaults


def test_inner_optimizer_defaults() -> None:
    for func in (mmle.refine, mmle.fit_mmle, mtdr.MTDR.__init__):
        params = inspect.signature(func).parameters
        assert params["optimizer_max_iter"].default == 2000
        assert params["optimizer_progtol"].default == 1e-9
        assert params["optimizer_tol"].default == 1e-6
    assert mmle.PRECISION_RESIDUAL_TOL == 1e-8
    assert mmle._LBFGS_FTOL == 1e-11


@given(
    f0=st.floats(-1e12, 1e12, allow_nan=False),
    progtol=st.floats(1e-15, 1.0),
)
def test_ftol_is_the_absolute_change_over_the_start(f0: float, progtol: float) -> None:
    ftol = mmle._ftol(f0, progtol)
    assert ftol == pytest.approx(progtol / max(abs(f0), 1.0), rel=1e-15)
    # The decrease that stops L-BFGS-B, ftol * max(|f|, 1), is progtol at f0.
    assert ftol * max(abs(f0), 1.0) == pytest.approx(progtol, rel=1e-15)


def test_ftol_edge_cases() -> None:
    assert mmle._ftol(5.0, None) == mmle._LBFGS_FTOL
    assert mmle._ftol(np.inf, 1e-9) == 1e-9
    assert mmle._ftol(0.25, 1e-9) == 1e-9


@pytest.mark.parametrize("bad", [0.0, -1e-9, np.nan, np.inf, True, "1e-9"])
def test_optimizer_progtol_is_validated(bad: Any) -> None:
    stats = _stats()
    fit = _fit(stats)
    with pytest.raises(ParameterError, match="optimizer_progtol"):
        mmle.refine(stats, fit, optimizer_progtol=bad)
    with pytest.raises(ParameterError, match="optimizer_progtol"):
        mtdr.MTDR(optimizer_progtol=bad)


def _recording(calls: list[dict[str, Any]]) -> Callable[..., Any]:
    def fake(fun: Any, x0: Any, **kw: Any) -> Any:
        f0, _ = fun(x0)
        calls.append({"bounds": kw.get("bounds"), "f0": f0, **kw["options"]})
        return REAL_MINIMIZE(fun, x0, **kw)

    return fake


def test_the_basis_step_uses_the_absolute_progtol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats = _stats()
    fit = _fit(stats)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(scipy.optimize, "minimize", _recording(calls))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mmle.refine(stats, fit, max_iter=1, optimizer_progtol=1e-7)
    basis = [c for c in calls if c["bounds"] is None]
    precision = [c for c in calls if c["bounds"] is not None]
    assert len(basis) == len(precision) == 1
    assert basis[0]["ftol"] == pytest.approx(1e-7 / abs(basis[0]["f0"]), rel=1e-14)
    assert basis[0]["maxiter"] == 2000
    assert precision[0]["ftol"] == 1e-11  # the precision step's relative ftol
    assert precision[0]["maxiter"] == 2000


def test_a_looser_progtol_stops_the_basis_step_earlier() -> None:
    stats = _stats(5)
    sim_start = mmle.MMLEFit.from_parameters(
        stats,
        mtdr.svd_fit.fit_svd(stats, [1, 2]).S,
        mtdr.svd_fit.fit_svd(stats, [1, 2]).noise_precision,
        stats.Y_mean,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        tight = mmle.refine(stats, sim_start, max_iter=1, optimizer_progtol=1e-9)
        loose = mmle.refine(stats, sim_start, max_iter=1, optimizer_progtol=1e-1)
    assert tight.log_likelihood >= loose.log_likelihood - 1e-9


# ----------------------------------------------------------------- precision ends


def _precision_optimum(fun: Any, x0: np.ndarray) -> np.ndarray:
    """Solve the separable precision gradient to rounding by Newton's method."""
    theta = np.array(x0, dtype=float)
    h = 1e-5
    for _ in range(40):
        _, g = fun(theta)
        _, gp = fun(theta + h)
        _, gm = fun(theta - h)
        step = g / ((gp - gm) / (2 * h))
        theta = theta - step
        if np.max(np.abs(step)) < 1e-14:
            break
    return theta


def _injected(
    *,
    precision_shift: float | None = None,
    precision_status: int = 2,
    basis_status: int | None = None,
) -> Callable[..., Any]:
    def fake(fun: Any, x0: Any, **kw: Any) -> Any:
        if kw.get("bounds") is None:
            res = REAL_MINIMIZE(fun, x0, **kw)
            if basis_status is not None:
                return types.SimpleNamespace(
                    x=res.x,
                    success=False,
                    status=basis_status,
                    message="ABNORMAL: ",
                    nit=3,
                )
            return res
        if precision_shift is None:
            return REAL_MINIMIZE(fun, x0, **kw)
        theta = _precision_optimum(fun, x0) + precision_shift
        return types.SimpleNamespace(
            x=theta,
            success=False,
            status=precision_status,
            message="ABNORMAL: " if precision_status == 2 else "STOP: TOTAL NO.",
            nit=3,
        )

    return fake


def _refine_once(stats: SufficientStats, fit: mmle.MMLEFit) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        out = mmle.refine(stats, fit, max_iter=1, tol=1e300)
    return out, [str(w.message) for w in caught]


def test_a_stationary_status_2_precision_end_is_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats = _stats()
    fit = _fit(stats)
    monkeypatch.setattr(scipy.optimize, "minimize", _injected(precision_shift=0.0))
    out, messages = _refine_once(stats, fit)
    assert messages == []
    assert out.converged == fit.converged


def test_a_status_2_precision_end_is_polished_to_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A status-2 end at a residual of about 2e-7 (above the 1e-8 tolerance) is
    # polished by diagonal Newton steps below 1e-8: a success.
    stats = _stats()
    fit = _fit(stats)
    monkeypatch.setattr(scipy.optimize, "minimize", _injected(precision_shift=2e-7))
    out, messages = _refine_once(stats, fit)
    assert messages == []
    assert out.converged == fit.converged


def test_a_polish_that_raises_the_objective_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A polished point whose objective rises is discarded and the residual
    # test judges the unpolished end, here at a residual of about 1e-6: a
    # failure.
    stats = _stats()
    fit = _fit(stats)
    monkeypatch.setattr(scipy.optimize, "minimize", _injected(precision_shift=1e-6))
    monkeypatch.setattr(mmle, "_newton_polish", lambda *a: a[4] + 1.0)
    out, messages = _refine_once(stats, fit)
    assert len(messages) == 1
    assert "noise precision: ABNORMAL" in messages[0]
    gaps = [float(g) for g in re.findall(r"\(([-0-9.e+]+)\)", messages[0])]
    assert max(abs(g) for g in gaps) > 5e-7
    assert not out.converged


def _precision_problem(
    stats: SufficientStats, fit: mmle.MMLEFit
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """K, u, ups and counts of the precision step at `fit` (as refine builds them)."""
    blocks = mmle._blocks(fit.S)
    xi, ups = stats.centered(fit.intercept)
    K = mmle._expand(stats.XtX, blocks.group) * (blocks.S.T @ blocks.S)
    u = mmle._project(xi, blocks)
    counts = stats.n_obs.astype(np.float64) * stats.n_bins
    return K, u, ups, counts


def test_the_polish_reaches_the_target_and_leaves_box_neurons() -> None:
    # From 0.05 off in log precision the polish reaches residuals below
    # 1e-10 in its three steps; a neuron on the box (here neuron 0's lower
    # bound) or with a non-positive curvature takes no step.
    stats = _stats()
    fit = _fit(stats)
    K, u, ups, counts = _precision_problem(stats, fit)
    theta0 = np.log(fit.noise_precision) + 0.05
    lower, upper = theta0 - 23.0, theta0 + 23.0
    lower[0] = theta0[0]
    out = mmle._newton_polish(K, u, ups, counts, theta0, lower, upper)
    assert out[0] == theta0[0]
    grad, _ = mmle._precision_derivatives(out, K, u, ups, counts)
    residual = np.abs(2 * grad / counts)
    assert residual[1:].max() <= 1e-10
    assert residual[0] > 1e-3
    # At the target already: no step at all.
    full = mmle._newton_polish(K, u, ups, counts, theta0, theta0 - 23, theta0 + 23)
    again = mmle._newton_polish(K, u, ups, counts, full, theta0 - 23, theta0 + 23)
    np.testing.assert_array_equal(again, full)


def test_the_precision_hessian_is_the_derivative_of_the_gradient() -> None:
    # h = (lambda / 2) (E + dE/dtheta) against a central difference of the
    # analytic gradient (separable: every theta shifted at once).
    stats = _stats()
    fit = _fit(stats)
    K, u, ups, counts = _precision_problem(stats, fit)
    theta = np.log(fit.noise_precision) + np.linspace(-0.3, 0.3, stats.n_neurons)
    grad, hess = mmle._precision_derivatives(theta, K, u, ups, counts)
    h = 1e-5
    gp, _ = mmle._precision_derivatives(theta + h, K, u, ups, counts)
    gm, _ = mmle._precision_derivatives(theta - h, K, u, ups, counts)
    np.testing.assert_allclose(hess, (gp - gm) / (2 * h), rtol=1e-6)
    lam = np.exp(theta)
    Cinv, mu, _, _ = mmle._posterior(lam, K, u)
    E = mmle._expected_residual(K, u, mu, Cinv, ups)
    np.testing.assert_allclose(grad, 0.5 * (lam * E - counts), rtol=1e-12)


def test_only_status_2_is_forgiven(monkeypatch: pytest.MonkeyPatch) -> None:
    stats = _stats()
    fit = _fit(stats)
    monkeypatch.setattr(
        scipy.optimize,
        "minimize",
        _injected(precision_shift=0.0, precision_status=1),
    )
    out, messages = _refine_once(stats, fit)
    assert len(messages) == 1
    assert "noise precision: STOP" in messages[0]
    assert not out.converged


def _basis_end(*, maxiter: int | None = None, x: Any = None, nit: int = 3) -> Any:
    """Inject a basis-step L-BFGS-B result with status 2."""

    def fake(fun: Any, x0: Any, **kw: Any) -> Any:
        if kw.get("bounds") is not None:
            return REAL_MINIMIZE(fun, x0, **kw)
        if x is not None:  # a returned point of our choosing, no iterate
            return types.SimpleNamespace(
                x=x(x0), success=False, status=2, message="ABNORMAL: ", nit=0
            )
        options = dict(kw["options"])
        if maxiter is not None:
            options["maxiter"] = maxiter
        res = REAL_MINIMIZE(fun, x0, **{**kw, "options": options})
        return types.SimpleNamespace(
            x=res.x, success=False, status=2, message="ABNORMAL: ", nit=nit
        )

    return fake


def test_a_rounding_level_basis_end_is_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real run stops at a decrease of about progtol; reported as a
    # line-search end, it is minFunc's normal "changing by less than progTol".
    stats = _stats()
    fit = _fit(stats)
    monkeypatch.setattr(scipy.optimize, "minimize", _basis_end())
    out, messages = _refine_once(stats, fit)
    assert messages == []
    assert out.converged == fit.converged


def test_a_basis_end_with_no_iterate_at_its_start_is_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # nit 0 with f_end = f_0: the last decrease is 0.
    stats = _stats()
    fit = _fit(stats)
    monkeypatch.setattr(scipy.optimize, "minimize", _basis_end(x=np.array))
    out, messages = _refine_once(stats, fit)
    assert messages == []
    assert out.converged == fit.converged


def test_a_basis_end_far_from_rounding_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # After one iteration the last decrease is far above the rounding bound
    # 100 max(progtol, eps M): still a failure, reported with its numbers.
    stats = _stats()
    svd = mtdr.svd_fit.fit_svd(stats, [1, 2])
    start = mmle.MMLEFit.from_parameters(
        stats, svd.S, svd.noise_precision, stats.Y_mean
    )  # far from the optimum, so one iteration still gains a lot
    monkeypatch.setattr(scipy.optimize, "minimize", _basis_end(maxiter=1))
    out, messages = _refine_once(stats, start)
    assert len(messages) == 1
    found = re.search(
        r"bases: ABNORMAL: \(L-BFGS-B status 2, nit \d+, largest projected-gradient "
        r"entry \S+\); last accepted decrease (\S+) \(rounding (\S+), progtol "
        r"1e-09\)(; |$)",
        messages[0],
    )
    assert found
    assert float(found[1]) > 10 * max(1e-9, float(found[2]))
    assert not out.converged


def test_a_rejected_basis_end_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    stats = _stats()
    fit = _fit(stats)
    monkeypatch.setattr(scipy.optimize, "minimize", _basis_end(x=lambda x0: x0 + 10.0))
    out, messages = _refine_once(stats, fit)
    assert len(messages) == 1
    assert "so the step was not taken" in messages[0]
    assert not out.converged


def test_a_basis_cap_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    stats = _stats()
    fit = _fit(stats)
    monkeypatch.setattr(scipy.optimize, "minimize", _injected(basis_status=1))
    out, messages = _refine_once(stats, fit)
    assert len(messages) == 1
    assert "`MTDR(optimizer_max_iter=...)`" in messages[0]
    assert not out.converged


@pytest.mark.parametrize(
    ("decrease", "magnitude", "status", "accepted", "finite", "ok"),
    [
        (2e-9, 3e5, 2, True, True, True),  # 2 progtol: rounding
        (5e-8, 3e5, 2, True, True, True),  # 50 progtol: within c = 100
        (1e-6, 3e5, 2, True, True, False),
        (0.0, 3e5, 2, True, True, True),  # nit 0 at the start
        (2e-6, 1e8, 2, True, True, True),  # eps M = 2.2e-8 dominates progtol
        (2e-9, 3e5, 1, True, True, False),  # the cap
        (2e-9, 3e5, 2, False, True, False),  # a rejected point
        (2e-9, 3e5, 2, True, False, False),  # a non-finite gradient
        (2e-9, np.inf, 2, True, True, False),  # a non-finite rounding scale
    ],
)
def test_the_basis_rule(
    decrease: float,
    magnitude: float,
    status: int,
    accepted: bool,
    finite: bool,
    ok: bool,
) -> None:
    grad = np.array([1.0, np.inf if not finite else 2.0])
    inner = mmle._Inner(np.zeros(2), grad, "ABNORMAL", status, accepted, 1e5, decrease)
    assert mmle._basis_end_is_rounding(inner, 1e-9, magnitude) is ok
    assert mmle.ROUNDING_PROGRESS_FACTOR == 100.0


def _demo_scale_fit(seed: int) -> tuple[SufficientStats, mmle.MMLEFit]:
    sim = mtdr.simulate(
        n_neurons=100,
        n_bins=15,
        n_trials=400,
        ranks=[2, 1, 3],
        drop_prob=0.3,
        seed=seed,
    )
    stats = sufficient_statistics(sim.Y_masked, sim.X, sim.mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return stats, mmle.fit_mmle(stats, [2, 2, 3])


@pytest.mark.parametrize(
    ("seed", "decrease"),
    [(55, 1.35e-8), (48, 1.49e-8)],  # Windows CI, Linux CI
)
def test_the_ci_basis_ends_are_rounding(
    seed: int, decrease: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On CI runners these demo-scale problems at ranks (2, 2, 3) ended a basis
    # step with status 2 after last accepted decreases of 1.35e-8 (Windows) and
    # 1.49e-8 (Linux), above a bound of 10 max(1e-9, eps |f|) = 1e-8, which is
    # below the objective's rounding. Reproduced at the fitted point: two
    # accepted iterates whose objectives differ by that decrease, then a
    # status-2 end at the second. A success under 100 max(progtol, eps M).
    stats, fit = _demo_scale_fit(seed)
    seen: list[float] = []

    def fake(fun: Any, x0: Any, **kw: Any) -> Any:
        if kw.get("bounds") is not None:
            return REAL_MINIMIZE(fun, x0, **kw)
        f_end, grad = fun(x0)
        direction = grad / np.linalg.norm(grad)

        def gain(t: float) -> float:
            return float(fun(x0 + t * direction)[0] - f_end)

        lo, hi = 0.0, decrease / np.linalg.norm(grad)
        while gain(hi) < decrease:  # f rises along +grad: bracket, then bisect
            hi *= 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if gain(mid) < decrease else (lo, mid)
        before = x0 + hi * direction
        kw["callback"](before)
        kw["callback"](np.array(x0))
        seen.append(fun(before)[0] - f_end)
        return types.SimpleNamespace(
            x=np.array(x0), success=False, status=2, message="ABNORMAL: ", nit=2
        )

    monkeypatch.setattr(scipy.optimize, "minimize", fake)
    out, messages = _refine_once(stats, fit)
    assert seen[0] == pytest.approx(decrease, rel=0.1)  # f's rounding: ~1e-9
    assert seen[0] > 1.1e-8  # above a bound of 1e-8
    assert messages == []
    assert out.converged == fit.converged


def test_a_polish_judged_within_the_objective_rounding_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A status-2 precision end at a residual of about 5e-8 is polished to the
    # optimum, but the polished objective, evaluated with rounding of about
    # eps M (M the sum of the objective's term magnitudes), can read above
    # f_end (seen on macOS and with SciPy 1.13). Reproduced by adding
    # 20 eps |f_end| to the polished evaluation: below the allowance 10 eps M
    # (here M is over 4 |f|) but above 10 eps |f_end|, an allowance that would
    # reject the polish and report the 5e-8 residual.
    stats = _stats()
    fit = _fit(stats)
    state: dict[str, Any] = {"returned": False, "after": []}
    injected = _injected(precision_shift=5e-8)

    def fake(fun: Any, x0: Any, **kw: Any) -> Any:
        res = injected(fun, x0, **kw)
        if kw.get("bounds") is not None:
            state["returned"] = True
        return res

    real = mmle._finite_or_inf

    def noisy(value: float, grad: np.ndarray) -> Any:
        if state["returned"]:
            state["after"].append(value)
            if len(state["after"]) == 2:  # the polished point's evaluation
                value = value + 20 * np.finfo(float).eps * abs(state["after"][0])
        return real(value, grad)

    monkeypatch.setattr(scipy.optimize, "minimize", fake)
    monkeypatch.setattr(mmle, "_finite_or_inf", noisy)
    out, messages = _refine_once(stats, fit)
    assert len(state["after"]) >= 2
    assert messages == []
    assert out.converged == fit.converged


def test_the_precision_rounding_scale_exceeds_the_value() -> None:
    # On the polish test's problem the precision objective's rounding scale M
    # is well above |f| (11.5 times here; eps M against eps |f| measured
    # 2.4e-11 against 5.6e-13 on a similar problem), which the polish test's
    # injected 20 eps |f| needs (at least 2 |f|).
    stats = _stats()
    fit = _fit(stats)
    K, u, ups, counts = _precision_problem(stats, fit)
    theta = np.log(fit.noise_precision)
    lam = np.exp(theta)
    _, _, explained, logdet = mmle._posterior(lam, K, u)
    value = 0.5 * float(np.sum(-counts * theta + logdet + lam * ups - explained))
    magnitude = mmle._precision_magnitude(theta, K, u, ups, counts)
    assert magnitude > 4 * abs(value)


# ----------------------------------------------------------------- last decrease


def test_the_last_accepted_decrease_is_read_from_the_iterates() -> None:
    # f_{k-1} - f_k over the accepted iterates, recorded independently
    # here with the same options and a callback of our own.
    A = np.diag(np.arange(1.0, 7.0))

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(0.5 * x @ A @ x + 3.0), A @ x

    x0 = np.ones(6)
    inner = mmle._lbfgs(objective, x0, None, 1e-12, 50, 1.0, 1e-9)
    seen: list[float] = []
    REAL_MINIMIZE(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        callback=lambda xk: seen.append(objective(xk)[0]),
        options={
            "maxiter": 50,
            "gtol": 1e-12,
            "ftol": mmle._ftol(objective(x0)[0], 1e-9),
            "maxcor": mmle._LBFGS_MEMORY,
        },
    )
    assert len(seen) >= 2
    assert inner.last_decrease == seen[-2] - seen[-1]
    assert inner.value == objective(inner.x)[0]


def test_with_no_accepted_iterate_the_decrease_is_from_the_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # After no accepted iterate (nit 0) the decrease is f_0 - f_end.
    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(x @ x), 2 * x

    def stuck(fun: Any, x0: Any, **kw: Any) -> Any:
        return types.SimpleNamespace(
            x=np.array(x0) * 0.5, success=False, status=2, message="ABNORMAL: ", nit=0
        )

    monkeypatch.setattr(scipy.optimize, "minimize", stuck)
    inner = mmle._lbfgs(objective, np.ones(2), None, 1e-6, 10, 1.0, 1e-9)
    assert inner.accepted
    assert inner.last_decrease == 2.0 - 0.5
    assert inner.value == 0.5


def test_no_basis_status_2_end_is_classified_as_rounding() -> None:
    # The package forgives rounding-level basis ends, so the test classifier
    # has no `basis_rounding` class: a basis status-2 warning is a genuine
    # failure, and no test may expect one.
    assert "basis_rounding" not in pf.REASONS
    for tail in ("", "; last accepted decrease 2e-09 (eps*|f| 6.9e-11, progtol 1e-09)"):
        message = (
            "refinement: iteration 1, bases: ABNORMAL: (L-BFGS-B status 2, nit 0, "
            "largest projected-gradient entry 1.00e+10)" + tail
        )
        with pytest.raises(AssertionError, match="unexpected reason"):
            pf.reasons(message)
