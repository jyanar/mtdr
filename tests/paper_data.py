r"""Paper-scale parity data, regenerated bit for bit from integer streams.

The paper-scale fixture `tests/fixtures/paper.mat` (generator
`tests/fixtures/matlab/make_paper_fixture.m`) does not store its responses: at
$n=800$ neurons, $T=15$ bins and $N=1000$ trial slots they would take tens of
megabytes. Instead both the generator and this module build them from the same
specification, using only operations that IEEE 754 arithmetic rounds the same
way in MATLAB and NumPy (integer arithmetic below $2^{53}$, `+`, `-`, `*`, `/`,
`sqrt`, `floor`, all evaluated in the same order), and the test checks two exact
integer checksums that the generator records.

The specification (every index 1-based in the MATLAB, 0-based here):

* **Uniforms.** The Park-Miller minimal-standard generator
  $x_j = 16807\,x_{j-1} \bmod (2^{31}-1)$, $x_0=$ the stream's seed, gives
  $u_j=x_j/(2^{31}-1)$ for $j=1,2,\dots$ One stream per quantity (seeds 1001-1009).
* **Normals.** Irwin-Hall: each normal is $u_1+\dots+u_{12}-6$ over twelve
  consecutive uniforms, added left to right.
* **Regressors** ($P=6$, stream 1001, four uniforms $a,c,g,h$ per trial):
  motion $x_1$ and colour $x_2$ coherences from
  $\{-0.5,-0.15,-0.05,0.05,0.15,0.5\}$ (index $\lfloor 6a\rfloor$, $\lfloor 6c\rfloor$);
  context $\pm1$ ($+1$ iff $g<0.5$); choice $+1$ iff the relevant coherence
  ($x_1$ in context $+1$, else $x_2$) $+0.6(h-0.5)>0$, else $-1$; then
  $x_1\cdot$context and $x_2\cdot$context. So the columns are correlated as in a
  context-dependent task, and centred.
* **Missingness** (non-simultaneous recordings): 100 sessions of 8 consecutive
  neurons. Session $s$ (stream 1002, two uniforms) records $L_s=100+\lfloor 201u\rfloor$
  consecutive trial slots starting at slot $\lfloor (N-L_s+1)u'\rfloor$; within its
  window each neuron misses a trial when its uniform from stream 1003 (index
  $iN+k$) is below 0.05. Each neuron sees 91-288 trials.
* **Truth.** Ranks $(5,4,3,4,2,2)$, $r_{tot}=20$. Weights: normals (stream 1004)
  in an $n\times r_{tot}$ column-major matrix, block $p$ scaled by
  $\rho=(3,3,1,1,2,2)$. Bases: normals (stream 1005) in a $(T+8)\times r_{tot}$
  matrix, smoothed twice by the binomial filter $(1,4,6,4,1)/16$ and scaled by
  2.25 (unit variance, no linear solve). $B_p=\sum_j \rho_p g_j s_j^\top$, summed
  over the block's components in order. Intercept
  $b=2+\sum_{j\le3} w^0_j (s^0_j)^\top$ (streams 1008, 1009). Noise precisions
  $d_i=0.1+(2.4u_i)u_i$ (stream 1006), in $[0.1, 2.5]$.
* **Responses.** For neuron $i$ and its observed trials $k$ in increasing order,
  $v_{ikt}=b_{it}+\sum_p x_{kp}B_{p,it}+e_{ikt}/\sqrt{d_i}$, the sum taken in
  regressor order, with $e$ the next normals of stream 1007 (time fastest, then
  trials, then neurons), rounded to the grid $2^{-16}$ by
  $\lfloor 2^{16}v+0.5\rfloor/2^{16}$.

The checksums are $\sum_j z_j$ and $\sum_j z_j(j \bmod 997+1)$ over the grid
integers $z_j=2^{16}y_j$ of the **completed** array, visited neuron by neuron,
each neuron's observed trials in ascending order, bins fastest ($j$ 1-based),
exact in double precision. That traversal is the generation order, so
they also equal the running sums of the drawn values; `array_checksums`
computes them from the returned array, and the parity test recomputes them
again from `Y` independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

MODULUS = 2_147_483_647  # 2^31 - 1
MULTIPLIER = 16_807
N_NEURONS, N_BINS, N_TRIALS, N_SESSIONS = 800, 15, 1000, 100
RANKS = (5, 4, 3, 4, 2, 2)
RHO = (3.0, 3.0, 1.0, 1.0, 2.0, 2.0)
LEVELS = np.array([-0.5, -0.15, -0.05, 0.05, 0.15, 0.5])
GRID = 65536.0
_BLOCK = 4096
_CHUNK_BLOCKS = 256


def lcg(seed: int, count: int) -> FloatArray:
    r"""The first `count` uniforms $x_j/(2^{31}-1)$ of the stream with seed `seed`.

    Vectorised by jumping ahead in blocks: $x_{kB+j}=(a^j \bmod m)\,x_{kB}\bmod m$,
    exact in int64 because both factors are below $2^{31}$.
    """
    powers = np.empty(_BLOCK, dtype=np.int64)  # a^1 ... a^B mod m
    acc = 1
    for j in range(_BLOCK):
        acc = acc * MULTIPLIER % MODULUS
        powers[j] = acc
    n_blocks = -(-count // _BLOCK)
    starts = np.empty(n_blocks, dtype=np.int64)  # x_{kB}
    x = seed
    jump = int(powers[-1])
    for k in range(n_blocks):
        starts[k] = x
        x = x * jump % MODULUS
    out = np.empty(n_blocks * _BLOCK, dtype=np.float64)
    for lo in range(0, n_blocks, _CHUNK_BLOCKS):
        hi = min(lo + _CHUNK_BLOCKS, n_blocks)
        block = (starts[lo:hi, None] * powers[None, :]) % MODULUS
        out[lo * _BLOCK : hi * _BLOCK] = block.ravel() / MODULUS
    return out[:count]


def normals(seed: int, count: int) -> FloatArray:
    """Irwin-Hall normals: twelve consecutive uniforms added in order, minus 6."""
    U = lcg(seed, 12 * count).reshape(count, 12)
    z = U[:, 0].copy()
    for r in range(1, 12):
        z = z + U[:, r]
    return z - 6.0


def smooth_bases(seed: int, n_bins: int, rank: int) -> FloatArray:
    """`(n_bins, rank)` bases: normals smoothed twice by (1, 4, 6, 4, 1) / 16."""
    z = normals(seed, (n_bins + 8) * rank).reshape(rank, n_bins + 8).T
    for _ in range(2):
        z = (z[:-4] + 4 * z[1:-3] + 6 * z[2:-2] + 4 * z[3:-1] + z[4:]) / 16
    return 2.25 * z


@dataclass
class PaperData:
    """The paper-scale dataset in the port's layouts, with its ground truth."""

    Y: FloatArray  # (N, n, T), NaN where unobserved
    X: FloatArray  # (N, P)
    mask: NDArray[np.bool_]  # (N, n)
    B: list[FloatArray]  # (n, T) per regressor
    S: list[FloatArray]  # (T, r_p)
    W: list[FloatArray]  # (n, r_p)
    intercept: FloatArray  # (n, T)
    precision: FloatArray  # (n,)
    checksums: tuple[int, int]


def regressors(n_trials: int = N_TRIALS) -> FloatArray:
    """The `(N, 6)` design of the specification."""
    u = lcg(1001, 4 * n_trials).reshape(n_trials, 4)
    x1 = LEVELS[np.floor(6 * u[:, 0]).astype(int)]
    x2 = LEVELS[np.floor(6 * u[:, 1]).astype(int)]
    ctx = np.where(u[:, 2] < 0.5, 1.0, -1.0)
    relevant = np.where(ctx == 1, x1, x2)
    choice = np.where(relevant + 0.6 * (u[:, 3] - 0.5) > 0, 1.0, -1.0)
    return np.column_stack([x1, x2, choice, ctx, x1 * ctx, x2 * ctx])


def observation_mask(
    n_neurons: int = N_NEURONS, n_trials: int = N_TRIALS, n_sessions: int = N_SESSIONS
) -> NDArray[np.bool_]:
    """The `(N, n)` mask: session windows with 5 % random trial loss."""
    u = lcg(1002, 2 * n_sessions)
    drop = lcg(1003, n_neurons * n_trials).reshape(n_neurons, n_trials)
    per = n_neurons // n_sessions
    mask = np.zeros((n_trials, n_neurons), dtype=bool)
    for s in range(n_sessions):
        length = 100 + int(np.floor(201 * u[2 * s]))
        start = int(np.floor((n_trials - length + 1) * u[2 * s + 1]))
        window = np.zeros(n_trials, dtype=bool)
        window[start : start + length] = True
        for i in range(s * per, (s + 1) * per):
            mask[:, i] = window & (drop[i] >= 0.05)
    return mask


def _outer_sum(G: FloatArray, S: FloatArray, scale: float) -> FloatArray:
    out = np.zeros((G.shape[0], S.shape[0]))
    for j in range(G.shape[1]):
        out = out + np.outer(scale * G[:, j], S[:, j])
    return out


def array_checksums(Y: FloatArray, mask: NDArray[np.bool_]) -> tuple[int, int]:
    """The two integer checksums of the observed entries of `Y` (module docstring).

    `Y` is `(N, n, T)` on the grid $2^{-16}$, `mask` `(N, n)`.
    """
    sums = [0, 0]
    pos = 0
    for i in range(Y.shape[1]):
        kk = np.flatnonzero(mask[:, i])
        z = Y[kk, i, :] * GRID  # (trials, bins): bins fastest in ravel()
        if not np.array_equal(z, np.round(z)):
            raise ValueError(f"neuron {i}'s responses are not on the grid 2^-16")
        flat = z.ravel().astype(np.int64)
        weights = (np.arange(pos + 1, pos + flat.size + 1) % 997) + 1
        sums[0] += int(flat.sum())
        sums[1] += int((flat * weights).sum())
        pos += flat.size
    return sums[0], sums[1]


def paper_data() -> PaperData:
    """Regenerate the paper-scale dataset (about 1 s and 150 MB)."""
    n, T, N = N_NEURONS, N_BINS, N_TRIALS
    X = regressors()
    mask = observation_mask()
    rtot = sum(RANKS)
    G = normals(1004, n * rtot).reshape(rtot, n).T
    Sb = smooth_bases(1005, T, rtot)
    c = np.concatenate([[0], np.cumsum(RANKS)])
    W = [RHO[p] * G[:, c[p] : c[p + 1]] for p in range(len(RANKS))]
    S = [Sb[:, c[p] : c[p + 1]] for p in range(len(RANKS))]
    B = [_outer_sum(G[:, c[p] : c[p + 1]], S[p], RHO[p]) for p in range(len(RANKS))]
    W0 = normals(1008, n * 3).reshape(3, n).T
    S0 = smooth_bases(1009, T, 3)
    intercept = 2.0 + _outer_sum(W0, S0, 1.0)
    u = lcg(1006, n)
    precision = 0.1 + 2.4 * u * u
    n_obs = mask.sum(axis=0)
    e = normals(1007, int(n_obs.sum()) * T)
    Y = np.full((N, n, T), np.nan)
    sums = [0, 0]
    pos = 0
    for i in range(n):
        kk = np.flatnonzero(mask[:, i])
        m = kk.size * T
        noise = e[pos : pos + m].reshape(kk.size, T)  # time fastest
        sig = np.broadcast_to(intercept[i], (kk.size, T)).copy()
        for p in range(len(RANKS)):
            sig = sig + np.outer(X[kk, p], B[p][i])
        v = sig + noise / np.sqrt(precision[i])
        z = np.floor(v * GRID + 0.5)
        flat = z.ravel()
        weights = (np.arange(pos + 1, pos + m + 1) % 997) + 1
        sums[0] += int(flat.astype(np.int64).sum())
        sums[1] += int((flat.astype(np.int64) * weights).sum())
        Y[kk, i, :] = z / GRID
        pos += m
    checksums = array_checksums(Y, mask)
    if checksums != (sums[0], sums[1]):  # pragma: no cover - a bug in this module
        raise AssertionError("the completed array is not the drawn one")
    return PaperData(
        Y=Y,
        X=X,
        mask=mask,
        B=B,
        S=S,
        W=W,
        intercept=intercept,
        precision=precision,
        checksums=checksums,
    )
