"""MATLAB parity through the `MTDR` class, on the demo and paper-scale fixtures.

`tests/test_parity_demo.py` and `tests/test_parity_paper.py` reach the reference
through the functional API (`fit_mmle`, `greedy_aic`). Here the same comparisons
run through `MTDR(...).fit`, so the class's wiring (statistics, estimator
options, the selection count (M38), the raw frame with `canonicalize=False`,
the two-stage search) is pinned to the reference too:

* `demoLearning.m`'s fit at its ranks: the class's raw-frame fit is
  `fit_mmle`'s exactly, its marginal NLL at least as good as the reference's to
  1e-7 (the gate for a fit from its own start), its estimates within the
  printed 1e-2 bound;
* `mTDRdemo.m`'s MMLE search from the reference's SVD ranks: the class selects
  with the reference count (M38), so it reproduces the reference's rank path
  exactly, with every accepted AIC decoded with the literal count
  $n+T\\sum r+nT$ and its NLL one-sided at 1e-7;
* the class's default pipeline on the demo data ends at the true ranks, its
  precision-weighted SVD seed there too;
* at the paper's scale (`slow`): the class's MMLE search from the true ranks
  takes the reference's path (it stays), one-sided 1e-7 per accepted row.

Each runs with and without `basis_preconditioning`, which changes the
optimiser's path but must pass the same other-path gates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import paper_data as pdm
import parity_fixtures as pf
from mtdr import MTDR, mmle
from mtdr.errors import DesignWarning
from mtdr.rank_search import RankSearchHistory
from mtdr.stats import sufficient_statistics

OTHER_PATH_NLL_RTOL = 1e-7
OTHER_PATH_GAP = 1e-2
ALLOWED = ("refine_cap", "precision_rounding", pf.SEARCH_REJECTED)


@pytest.fixture(scope="module")
def demo() -> pf.DemoFixture:
    assert pf.DEMO_FIXTURE.is_file(), "tests/fixtures/demo.mat is missing"
    return pf.load_demo()


def _quiet(func: Any) -> Any:
    out, _ = pf.run(func, ALLOWED)
    return out


def _reference_count(n: int, T: int, ranks: Any) -> int:
    # BTDR_AIC_S_lamb_b_wrapper's count, written out rather than taken from the
    # package, so that a wrong package count cannot shift the decoded reference
    # NLL with it.
    return n + T * int(sum(ranks)) + n * T


def _search_rises(
    history: RankSearchHistory,
    funhist: Any,
    constant: float,
    n: int,
    T: int,
) -> np.ndarray:
    """(port NLL - reference NLL) / |reference NLL| per accepted row."""
    out = []
    for ranks, port_aic, ref_aic in zip(
        history.ranks, history.aic, funhist, strict=True
    ):
        k = _reference_count(n, T, ranks)
        ref_nll = ref_aic / 2 - k
        port_nll = port_aic / 2 - k - constant
        out.append((port_nll - ref_nll) / abs(ref_nll))
    return np.array(out)


def _candidate_rises(
    history: RankSearchHistory,
    search: dict[str, Any],
    constant: float,
    n: int,
    T: int,
) -> np.ndarray:
    """(port NLL - reference NLL) / |reference NLL| per reference candidate call.

    The reference records every fit of its search (`calls_ranks`,
    `calls_values`); each call after the start is a candidate of round $k$, one
    regressor raised from the accepted `history.ranks[k]`, whose port score is in
    `history.candidates[k]` (so warm starts, the default, are checked candidate
    by candidate, not only on the accepted path).
    """
    start = int(np.sum(search["rest0"]))
    out = []
    for ranks, ref_aic in zip(
        search["calls_ranks"][1:], search["calls_values"][1:], strict=True
    ):
        r = np.array(ranks, dtype=np.int64)
        k = int(r.sum()) - start - 1
        (p,) = np.flatnonzero(r != history.ranks[k])
        port_aic = history.candidates[k][history.regressor_names[p]]
        count = _reference_count(n, T, r)
        ref_nll = ref_aic / 2 - count
        port_nll = port_aic / 2 - count - constant
        out.append((port_nll - ref_nll) / abs(ref_nll))
    return np.array(out)


@pytest.mark.parametrize("precondition", [False, True])
def test_learn_fit_through_the_class(demo: pf.DemoFixture, precondition: bool) -> None:
    le = demo.learn
    names = ["x0", "x1", "x2"]
    model = _quiet(
        lambda: MTDR(
            ranks=le["ranks"], canonicalize=False, basis_preconditioning=precondition
        ).fit(demo.Y, demo.X, demo.mask, regressor_names=names)
    )
    scale = mmle.BASIS_SPAN_SCALE if precondition else 1.0
    ref = _quiet(lambda: mmle.fit_mmle(demo.stats, le["ranks"], basis_span_scale=scale))
    assert model.log_likelihood_ == ref.log_likelihood
    for p, name in enumerate(names):
        np.testing.assert_array_equal(model.S_[name], ref.S[p])
        np.testing.assert_array_equal(model.W_[name], ref.W[p])
    fit = model._fit_object
    nll = pf.nll_without_constant(fit, demo.stats)
    rise = (nll - le["nll_final"]) / abs(le["nll_final"])
    gaps = pf.final_gaps(fit, le["final"], demo.stats)
    print(
        f"\n[demoLearning through MTDR, precondition={precondition}] "
        f"(port - reference)/|NLL| = {rise:.2e}; gaps "
        + ", ".join(f"{k} {v:.1e}" for k, v in gaps.items())
    )
    assert rise <= OTHER_PATH_NLL_RTOL
    assert all(v < OTHER_PATH_GAP for v in gaps.values())
    canonical = _quiet(
        lambda: MTDR(ranks=le["ranks"], basis_preconditioning=precondition).fit(
            demo.Y, demo.X, demo.mask
        )
    )
    for p, name in enumerate(names):
        np.testing.assert_allclose(canonical.B_[name], ref.B[p], atol=1e-10)


@pytest.mark.parametrize("precondition", [False, True])
@pytest.mark.parametrize("warm", [False, True])
def test_mmle_search_through_the_class(
    demo: pf.DemoFixture, warm: bool, precondition: bool
) -> None:
    # mTDRdemo.m's second search, seeded with the reference's SVD ranks (the
    # package's corrected SVD AIC would seed it elsewhere): the class selects
    # with (M38), so the path is the reference's. Cold is the reference's
    # procedure (M29); warm starts must reach the same path and likelihoods.
    mm = demo.mmle
    model = _quiet(
        lambda: MTDR(
            ranks="aic",
            rank_search_init=list(mm["rest0"]),
            max_rank=demo.maxrank,
            rank_search_warm_start=warm,
            basis_preconditioning=precondition,
        ).fit(demo.Y, demo.X, demo.mask)
    )
    h = model.rank_search_history_
    assert h is not None
    assert h.svd_stage is None
    np.testing.assert_array_equal(h.ranks, mm["rhist"])
    assert list(model.ranks_.values()) == mm["rest"] == demo.true_ranks
    constant = mmle._constant(demo.stats.n_obs, demo.stats.n_bins)
    rise = _search_rises(h, mm["funhist"], constant, demo.n, demo.T)
    cand = _candidate_rises(h, mm, constant, demo.n, demo.T)
    print(
        f"\n[demo MMLE search through MTDR, warm={warm}, precondition="
        f"{precondition}] {h.ranks.tolist()}; "
        "(port - reference)"
        f"/|NLL| per accepted row in [{rise.min():.1e}, {rise.max():.1e}], per "
        f"candidate in [{cand.min():.1e}, {cand.max():.1e}]; "
        f"near ties {h.near_ties()}"
    )
    assert np.all(rise <= OTHER_PATH_NLL_RTOL)
    assert np.all(cand <= OTHER_PATH_NLL_RTOL)
    # aic_ reports the identifiable count (M38a).
    assert model.aic_ < h.aic[-1]


def test_default_pipeline_on_the_demo_data(demo: pf.DemoFixture) -> None:
    model = _quiet(lambda: MTDR().fit(demo.Y, demo.X, demo.mask))
    h = model.rank_search_history_
    assert h is not None
    assert h.svd_stage is not None
    assert h.svd_stage.ranks[-1].tolist() == demo.true_ranks
    assert list(model.ranks_.values()) == demo.true_ranks


@pytest.mark.slow
@pytest.mark.parametrize("precondition", [False, True])
@pytest.mark.parametrize("warm", [False, True])
def test_paper_mmle_search_through_the_class(warm: bool, precondition: bool) -> None:
    assert pf.PAPER_FIXTURE.is_file(), "tests/fixtures/paper.mat is missing"
    ref = pf.load_paper()
    sr = ref.mmle
    assert sr, "paper.mat was generated without the MMLE search"
    data = pdm.paper_data()
    # The paper data has trial slots no session recorded: fit keeps them and
    # says so.
    with pytest.warns(DesignWarning, match="have no observed neuron"):
        model = _quiet(
            lambda: MTDR(
                ranks="aic",
                rank_search_init=list(sr["rest0"]),
                max_rank=15,
                rank_search_warm_start=warm,
                basis_preconditioning=precondition,
            ).fit(data.Y, data.X, data.mask)
        )
    h = model.rank_search_history_
    assert h is not None
    np.testing.assert_array_equal(h.ranks, sr["rhist"])
    stats = sufficient_statistics(data.Y, data.X, data.mask)
    constant = mmle._constant(stats.n_obs, stats.n_bins)
    rise = _search_rises(h, sr["funhist"], constant, ref.n, ref.T)
    cand = _candidate_rises(h, sr, constant, ref.n, ref.T)
    print(
        f"\n[paper MMLE search through MTDR, warm={warm}, precondition="
        f"{precondition}] {h.ranks.tolist()}; "
        "(port - reference)"
        f"/|NLL| per accepted row in [{rise.min():.1e}, {rise.max():.1e}], per "
        f"candidate in [{cand.min():.1e}, {cand.max():.1e}]; "
        f"converged {model.converged_}; near ties {h.near_ties()}"
    )
    assert np.all(rise <= OTHER_PATH_NLL_RTOL)
    assert np.all(cand <= OTHER_PATH_NLL_RTOL)
