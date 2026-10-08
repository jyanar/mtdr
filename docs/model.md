# The model and its estimation

This page describes the fitted model, estimation, rank selection and interpretation of
the outputs. The [technical appendices](#technical-appendices) retain the numbered
equations **(M1)**–**(M50)** cited by the code. For validation evidence and
implementation differences, see [Differences from MATLAB](differences-from-matlab.md).

**NeurIPS supplement** and **Nature Neuroscience supplement** refer to the
supplements of [Aoi & Pillow (2018)](https://proceedings.neurips.cc/paper/2018/hash/8a1ee9f2b7abe6e88d1a479ab6a42c5e-Abstract.html)
and [Aoi, Mante & Pillow (2020)](https://doi.org/10.1038/s41593-020-0696-5).
The authors' [`mTDRdemo`](https://github.com/pillowlab/mTDRdemo) supplies the MATLAB implementation.
The Nature Neuroscience supplement writes $B_p=W_pS_p$; here $S_p$ is transposed,
so $B_p=W_pS_p^\top$. Its $D=\mathrm{diag}(\lambda)$ is the noise precision matrix.

## 1. Notation and array layouts

### 1.1 Symbols

| Symbol | Meaning |
|---|---|
| $N$ | Number of trials |
| $k\in\{1,\ldots,N\}$ | Trial index |
| $n$ | Number of neurons, indexed by $i=1,\ldots,n$ |
| $T$ | Number of time bins per trial, indexed by $t=1,\ldots,T$ |
| $P_b$ | Number of task regressors, indexed by $p=1,\ldots,P_b$ |
| $x_k\in\mathbb R^{P_b}$ | Vector of task-regressor values on trial $k$; $x_{kp}$ is the value of regressor $p$ |
| $r_p$, $r_{tot}=\sum_p r_p$ | Rank of regressor $p$ and total rank |
| $h_{ki}$, $\mathcal K_i$, $n_i$ | Observation mask, observed trials and trial count for neuron $i$ |
| $W_p$, $S_p$, $B_p=W_pS_p^\top$ | Neuron weights, temporal bases and coefficient matrix |
| $b_i$, $\lambda_i$ | Neuron $i$'s intercept time course and noise precision (inverse variance) |
| $\gamma$, $g$ | SVD ridge (`ridge`) and marginal-likelihood basis ridge (`basis_ridge`), both zero by default |

### 1.2 Array layouts

| Public input/output | Shape | Meaning |
|---|---|---|
| `Y` | `(N, n, T)` | Responses; masked entries may contain `NaN` |
| `X` | `(N, P_b)` | Task regressors; with the default intercept, omit a constant column |
| `mask` | `(N, n)`, bool | Whether all bins of a neuron-trial are observed |
| `W_[p]` | `(n, r_p)` | Encoding basis for regressor `p` |
| `S_[p]` | `(T, r_p)` | Corresponding temporal coordinates |
| `B_[p]` | `(n, T)` | Fitted coefficient matrix |
| `intercept_` | `(n, T)` | Fitted intercept; `None` when disabled |
| `noise_precision_` | `(n,)` | Fitted inverse noise variances |

Per-regressor outputs are dictionaries keyed by name, in `X` column order.
Functional modules use lists in the same order. Without an explicit mask, any
`NaN` bin excludes the whole neuron-trial. Internal arrays and packing conventions
are defined in [Appendix A](#appendix-a-notation-and-statistics).

## 2. Generative model

$$
Y_k=\sum_{p=1}^{P_b}x_{kp}\,W_pS_p^\top+b^\top+E_k,\qquad
Y_k,\;E_k\in\mathbb R^{n\times T},\quad k=1,\dots,N,\tag{M1}
$$

$$
E_k[i,t]\overset{\text{ind}}{\sim}\mathcal N(0,\lambda_i^{-1})\quad\text{over }i,t,k,\tag{M2}
$$

$$
\text{observation: } Z_k[i,:]=h_{ki}\,Y_k[i,:];\ \text{the likelihood contains only entries with }h_{ki}=1,\tag{M3}
$$

$$
\text{prior (MMLE only): } w_i\overset{\text{iid}}{\sim}\mathcal N(0,I_{r_{tot}}),\quad
w_i=\big(W_1[i,:],\dots,W_{P_b}[i,:]\big)^\top,\ i=1,\dots,n.\tag{M4}
$$

Remarks.

- **The condition-independent term** is an unconstrained $b_i\in\mathbb R^T$ per
  neuron, as in the [Nature Neuroscience supplement, eqs. (8)–(11)][nn-supp], fitted by conditional maximisation of the
  marginal likelihood (M33). The demo's SVD stage instead appends a constant column to
  $X$ and treats the term as one more coefficient matrix of full rank $\min(n,T)$; the
  package uses that only to initialise. With `condition_independent=True`, a constant
  column in `X` is a `ValidationError`.
- **Which factor is marginalised.** The prior (M4) is on the neuron weights: $W$ is
  integrated out and $(S,\lambda,b)$ are the parameters. Marginalising $W$ rather than
  $S$ is what makes the marginal likelihood factorise over neurons ([§4](#4-marginal-likelihood-with-w-integrated-out)), which is what
  allows neurons that were not recorded simultaneously.

Noise is independent across neurons, bins and trials, with one variance per
neuron. Missingness is at the neuron-trial level; the model does not estimate
noise correlations between neurons. The per-neuron form and sufficient statistics
are given in [Appendix A](#appendix-a-notation-and-statistics).

## 3. The least-squares (SVD) estimator and its AIC

### 3.1 Least squares and truncation

The SVD estimator fits each neuron by least squares, then retains $r_p$ components per
regressor: a fit for `estimator="svd"` or a starting point for MMLE. With an intercept,
$X_i^f=[X_i\ \mathbf1]$ is the observed design;
[§A.6](#a6-statistics-of-the-least-squares-stage-mksuffstats_bilinreg_sims) defines
$A_i^f$ and $\xi_i^f$. The normal equations are

$$
(A^f_i\otimes I_T)\,\beta_i=\xi^f_i,\qquad
\beta_i=\big(B_1[i,:],\dots,B_{P_b}[i,:],\,b^{\rm full}_i\big)^\top,\tag{M15}
$$

With ridge $\gamma$,

$$
\hat\beta_i=(A^f_i\otimes I_T+\gamma I)^{-1}\xi^f_i\quad\forall i.\tag{M16}
$$

Each block is then truncated by its SVD:

$$
\hat B_p^{\top}=U_p\Sigma_pV_p^\top,\qquad
\hat S^{svd}_p=U_p[:,1{:}r_p]\,\Sigma_p[1{:}r_p]^{1/2},\quad
\hat W^{svd}_p=V_p[:,1{:}r_p]\,\Sigma_p[1{:}r_p]^{1/2},\tag{M17}
$$

$$
\hat B^{(r_p)}_p=\hat W^{svd}_p\hat S^{svd\top}_p\quad(\text{the best rank-}r_p\text{ approximation of }\hat B_p).\tag{M18}
$$

Components are ordered by decreasing singular value, hence by variance of $\hat B_p$
explained. Each weight column's largest absolute entry is positive; the canonical frame
([§10.1](#101-identifiability-and-the-canonical-frame)) also makes the weight columns
orthonormal.

With a fitted intercept and `ridge=0`, the SVD fit is invariant to shifts of regressors
or responses. A rank-deficient observed design uses a minimum-norm solution and emits a
[`DesignWarning`][mtdr.errors.DesignWarning]; see
[the numerical form](#b1-svd-numerical-form).

### 3.2 Noise precision and intercept

Truncation changes the predicted mean, so the intercept is refitted. Each neuron's
precision is the inverse residual variance. With intercept ridge $\gamma$,

$$
\hat b_i=\frac{n_i}{n_i+\gamma}\Big(\bar y_i-\sum_{p=1}^{P_b}\bar x_{ip}\hat B^{(r_p)}_p[i,:]^\top\Big),\qquad
\hat\lambda^{svd}_i=\frac{n_iT}{\mathrm{RSS}_i},\qquad
\mathrm{RSS}_i=\sum_{j=1}^{n_i}\sum_{t=1}^T\Big(z_i[t,j]-\hat b_i[t]-\sum_{p=1}^{P_b}x_{k_jp}\hat B^{(r_p)}_p[i,t]\Big)^2 .\tag{M19b}
$$

At $\gamma=0$ the refitted intercept maximises the plug-in likelihood at the truncated
coefficients; a positive ridge shrinks it towards zero.

### 3.3 The SVD-stage AIC

AIC balances fit against the number of parameters when choosing ranks: lower is better.
The SVD-stage score uses the Gaussian plug-in likelihood at
$(\hat B^{(r)},\hat b,\hat\lambda)$ of (M19b),

$$
-2\ell_{svd}=\sum_{i=1}^n\big[-n_iT\log\hat\lambda_i+\hat\lambda_i\mathrm{RSS}_i\big]
=\sum_{i=1}^n n_iT\Big[\log\frac{\mathrm{RSS}_i}{n_iT}+1\Big],\tag{M21a}
$$

$$
K_{svd}(r)=\sum_{p=1}^{P_b}r_p\,(n+T-r_p)+n\;\big[+\,nT\big],\qquad
\mathrm{AIC}_{svd}(r)=-2\ell_{svd}+2K_{svd}(r).\tag{M21b}
$$

The count includes the rank-$r_p$ coefficient matrices, $n$ precisions and, when fitted,
$nT$ intercept entries. The Gaussian constant omitted in (M21a) is included in public
log-likelihoods.

Truncated SVD need not maximise the rank-constrained likelihood, particularly with
unequal precisions or masks. Use this score to compare SVD ranks or seed the MMLE
search; it is not comparable with the marginal-likelihood AIC (M38). A Gaussian AIC can
be negative.

**Precision-weighted initialisation.** By default, the MMLE rank search is seeded by
precision-weighted SVD (`rank_search_init="svd_weighted"`), reducing the influence of
noisy neurons:

$$
\hat B^{(r),w}_p=D^{-1}\,\mathrm{trunc}_{r_p}\!\big(D\hat B_p\big),\qquad
D=\mathrm{diag}(d_1,\dots,d_n),\quad d_i=\sqrt{\hat\lambda^{\mathrm{OLS}}_i\,n_i},\quad
\hat\lambda^{\mathrm{OLS}}_i=\frac{n_iT}{\mathrm{RSS}^{\mathrm{full}}_i},\tag{M18w}
$$

Here $\mathrm{trunc}_r$ is the rank-$r$ truncated SVD and $\mathrm{RSS}^{\rm full}_i$ is
the unconstrained fit's residual. The weights are scaled to unit root mean square. The
intercept, residual and AIC are then computed as above. `estimator="svd"` uses
unweighted truncation (M18).

## 4. Marginal likelihood with $W$ integrated out

MMLE estimates $(S,\lambda,b)$ while integrating over uncertainty in the neuron weights
$W$. Each neuron contributes independently, so neurons can have different observed
trials. From (M9), $\zeta_i-\mathbf 1\otimes b_i\sim\mathcal N(0,\Sigma_i)$ with
$\Sigma_i=\lambda_i^{-1}I_{n_iT}+\Phi_i\Phi_i^\top$. Define

$$
K_i=A_i\otimes I_T,\qquad
C_i=I_{r_{tot}}+\lambda_i\Phi_i^\top\Phi_i=I_{r_{tot}}+\lambda_i\,\mathbf SK_i\mathbf S^\top\in\mathbb R^{r_{tot}\times r_{tot}}.\tag{M22}
$$

The [per-neuron model](#a3-per-neuron-form) defines $\Phi_i$ and $\zeta_i$. With
$u_i=\mathbf S\xi_i(b)$ and centred statistics (M13)–(M14), the negative marginal
log-likelihood, omitting the Gaussian constant and including a basis ridge, is

$$
\mathcal L(\lambda,s;b)=\frac12\sum_{i=1}^n\Big[-n_iT\log\lambda_i+\log\lvert C_i\rvert+\lambda_i\,\upsilon_i(b)-\lambda_i^2\,q_i\Big]+\frac g2\,s^\top s,
\qquad q_i=u_i^\top C_i^{-1}u_i .\tag{M24}
$$

Efficient evaluation and gradients are in
[Appendix C](#appendix-c-likelihood-and-estimation).

For the fitted model, [`log_likelihood_`][mtdr.model.MTDR] is the marginal training
log-likelihood, including the Gaussian constant and excluding the ridge. In contrast,
[`log_likelihood(Y, X, mask)`][mtdr.model.MTDR.log_likelihood] evaluates the plug-in
likelihood (M46), using the fitted coefficient matrices; use it for held-out
comparisons. With `basis_ridge>0`, [`objective_`][mtdr.model.MTDR] includes the penalty.

The posterior of each neuron's weights is

$$
w_i\mid\zeta_i,S,\lambda,b\ \sim\ \mathcal N\big(\mu_i,\;C_i^{-1}\big),\qquad
\mu_i=\lambda_iC_i^{-1}u_i,\qquad
\Omega_i:=\mathbb E[w_iw_i^\top]=\mu_i\mu_i^\top+C_i^{-1}.\tag{M25}
$$

A finite likelihood maximum requires residual degrees of freedom; see
[§10.3](#103-degenerate-neurons).

## 5. ECME and coordinate ascent

At fixed ranks, [`fit_mmle`][mtdr.mmle.fit_mmle] uses two stages: ECME's closed-form
updates quickly approach a local optimum, but slow near it; coordinate ascent then
refines the marginal likelihood directly. An SVD fit starts ECME, which stops at a loose
tolerance ([Nature Neuroscience supplement, §§3–3.1][nn-supp];
[NeurIPS supplement, Algorithm 1][neurips-paper]).

Check [`converged_`][mtdr.model.MTDR] and any
[`ConvergenceWarning`][mtdr.errors.ConvergenceWarning]. The
[API guide](api/model.md#converged_false-what-it-means-and-what-to-do) explains which
limit to raise and how to assess the fit. Convergence does not certify a global optimum.
See [§C.5](#c5-initialisation-ecmeregress_wrapper)
–[§C.7](#c7-coordinate-ascent-on-the-marginal-likelihood-estpars_coordascent_lambi_s_b)
for the steps.

## 6. Posterior of $W$ and the coefficient matrices

### 6.1 Posterior weights

After fitting the shared temporal bases, each neuron's posterior gives its weights and
their uncertainty, conditional on $(\hat{\mathbf S},\hat\lambda,\hat b)$:

$$
\hat w_i=\mathbb E[w_i\mid\zeta_i,\hat S,\hat\lambda,\hat b]=\hat\lambda_i\,\hat C_i^{-1}\hat{\mathbf S}\,\xi_i(\hat b)=\mu_i\ \text{of (M25)},\qquad
\mathrm{Cov}[w_i\mid\cdot]=\hat C_i^{-1},\tag{M36}
$$

[`posterior_weights`][mtdr.mmle.posterior_weights] returns these means and covariances
in the estimation frame ([Nature Neuroscience supplement, eqs. (16)–(17)][nn-supp]).

### 6.2 Coefficient matrices

The coefficient matrix describes the population response per unit of regressor $p$. It
combines the posterior mean weights with the fitted temporal basis:

$$
\hat W_p=\big(\hat w_{1[p]},\dots,\hat w_{n[p]}\big)^\top\in\mathbb R^{n\times r_p},\qquad
\hat B_p=\hat W_p\hat S_p^\top\in\mathbb R^{n\times T},\qquad
\text{intercept}=\hat b .\tag{M37}
$$

[`B_[p]`][mtdr.model.MTDR] holds this reconstruction
([Nature Neuroscience supplement, §4.1][nn-supp]). By default,
[`W_[p]`][mtdr.model.MTDR] and [`S_[p]`][mtdr.model.MTDR] express it in the canonical
display frame ([§10.1](#101-identifiability-and-the-canonical-frame)); their product is
unchanged.

## 7. Rank selection

### 7.1 The MMLE AIC

The MMLE rank search penalises extra temporal-basis parameters to balance likelihood
gains against model size. Its selection score uses (M24) with $g=0$:

$$
\mathrm{AIC}_{mml}(r)=2\,\mathcal L(\hat\lambda,\hat s;\hat b)+2\,K_{mml}(r),\qquad
K_{mml}(r)=\mathrm{numel}(\theta)=n+T\,r_{tot}+nT .\tag{M38}
$$

The count includes $n$ precisions, $Tr_{tot}$ basis entries and $nT$ intercept entries
(omit $nT$ without an intercept); the integrated-out $W$ is not counted. Neither paper
specifies this count; (M38) follows the MATLAB demo. Because (M24) is invariant under
$S_p\to S_pQ_p$ for orthogonal $Q_p$
([§10.1](#101-identifiability-and-the-canonical-frame)), $S_p$ carries only
$Tr_p-r_p(r_p-1)/2$ identifiable parameters:

$$
K_{mml}^{\rm id}(r)=n+\sum_{p=1}^{P_b}\Big(T\,r_p-\tfrac12r_p(r_p-1)\Big)+nT .\tag{M38a}
$$

[`MTDR`][mtdr.model.MTDR] **selects** with (M38):
[`rank_search_history_.aic`][mtdr.rank_search.RankSearchHistory] includes the Gaussian
constant and uses that count. [`n_parameters_`][mtdr.model.MTDR] and
[`aic_`][mtdr.model.MTDR] **report** (M38a), so the selection score exceeds
[`aic_`][mtdr.model.MTDR] at the same fit by $\sum_pr_p(r_p-1)$. To try another count,
pass an `objective` to [`greedy_aic`][mtdr.rank_search.greedy_aic]. Both AICs exclude
the basis ridge; [`objective_`][mtdr.model.MTDR] includes it.

### 7.2 Greedy search

An exhaustive search grows exponentially with the number of regressors. Greedy search
instead tries raising each rank by one and keeps the best improvement. For AIC $F(r)$,
starting ranks $r^{(0)}$, cap `max_rank` and improvement threshold $\delta\ge0$
(`rank_search_threshold`):

$$
\begin{aligned}
&r\leftarrow r^{(0)},\quad F_0\leftarrow F(r)\\
&\textbf{repeat:}\\
&\quad\mathcal P\leftarrow\{p:\ r_p<\texttt{max\_rank}\};\ \textbf{stop if }\mathcal P=\emptyset\\
&\quad\textbf{for }p\in\mathcal P:\ F_p\leftarrow F(r+e_p)\\
&\quad\textbf{stop if }\min_{p\in\mathcal P}F_p\ge F_0-\delta\\
&\quad p^*\leftarrow\arg\min_{p\in\mathcal P}F_p\ \ (\text{lowest index on ties}),\quad
r_{p^*}\leftarrow r_{p^*}+1,\quad F_0\leftarrow F_{p^*}
\end{aligned}\tag{M39}
$$

A step must lower AIC by more than $\delta$ (default 0). Ranks never decrease, so the
start matters: by default, MMLE starts at ranks selected by precision-weighted SVD
(M18w), searching from all ones.
[`RankSearchHistory`][mtdr.rank_search.RankSearchHistory] records the initial and
accepted fits and every round's candidate scores, including a rejected final round.

**Warm starts** (`rank_search_warm_start=True`, the default). Each candidate starts from
the accepted fit, adding an SVD component for the raised regressor. The first fit starts
from its own SVD estimate. Set `rank_search_warm_start=False` to start every candidate
from its own SVD estimate.

## 8. Simulation model

[`simulate`][mtdr.simulation.simulate] returns data with known ranks and coefficients
for checking recovery. It draws smooth temporal bases and heterogeneous noise
precisions, or accepts a supplied design, bases or weights. These choices do not
constrain [`MTDR.fit`][mtdr.model.MTDR.fit]. Its mask guarantees two observations per
neuron, which may be insufficient for MMLE ([§10.3](#103-degenerate-neurons)). Draws and
mask conditioning are described in
[Appendix D](#appendix-d-simulation-and-decoder-identities).

## 9. Decoding and projection

Decoding estimates task variables; projection shows activity within encoding subspaces.
Both use the fitted model, without a decoder prior on $x$
([Nature Neuroscience supplement, §6][nn-supp]).

### 9.1 Maximum-likelihood decoding

Decoding finds the task values whose predicted activity best matches a trial, weighting
neurons by precision. For observed neurons $\mathcal I$, responses
$Y\in\mathbb R^{n\times T}$ and bins $\mathcal T\subseteq\{1,\dots,T\}$, the
log-likelihood of $x\in\mathbb R^{P_b}$ is

$$
\ell(x)=-\frac12\sum_{i\in\mathcal I}\hat\lambda_i\sum_{t\in\mathcal T}\Big(Y[i,t]-\hat b[i,t]-\sum_{p=1}^{P_b}x_p\hat B_p[i,t]\Big)^2+\text{const},\tag{M46}
$$

The Gaussian constant is included by
[`MTDR.log_likelihood`][mtdr.model.MTDR.log_likelihood]. Decode over all bins, a single
bin (instantaneous decoding), or a selected window.

**Unconditional decoding** ([Nature Neuroscience supplement, eqs. (20)-(22)][nn-supp]).
With $\Xi\in\mathbb R^{\lvert\mathcal I\rvert\lvert\mathcal T\rvert\times P_b}$, column
$p=\mathrm{vec}(\hat B_p[\mathcal I,\mathcal T])$,
$\Lambda=\mathrm{diag}(\hat\lambda_i)$ repeated over bins, and
$y=\mathrm{vec}(Y[\mathcal I,\mathcal T]-\hat b[\mathcal I,\mathcal T])$,

$$
\hat x=(\Xi^\top\Lambda\,\Xi)^{-1}\Xi^\top\Lambda\,y .\tag{M47}
$$

The $P_b\times P_b$ matrix is singular or nearly so when few entries are observed or a
regressor's $\hat B_p$ is near zero on the selected bins (a stimulus before it is
shown). If it is singular or its condition number exceeds $\epsilon^{-1/2}$, that
trial's values are `NaN` and one [`DecodingWarning`][mtdr.errors.DecodingWarning] counts
such trials; [`decode`][mtdr.model.MTDR.decode] never raises for this.

**Conditional decoding** ([Nature Neuroscience supplement, eq. (23)][nn-supp]). With
unknown regressors $\mathcal U$ and known $\mathcal K$ at values $x_{\mathcal K}$,

$$
\hat x_{\mathcal U}\mid x_{\mathcal K}=(\Xi_{\mathcal U}^\top\Lambda\,\Xi_{\mathcal U})^{-1}\Xi_{\mathcal U}^\top\Lambda\,\big(y-\Xi_{\mathcal K}x_{\mathcal K}\big).\tag{M47a}
$$

Every positive-rank regressor must be either decoded or given a value; none is silently
set to zero. Rank-zero regressors are ignored.

**Discrete regressors by log-likelihood ratio**
([Nature Neuroscience supplement, §6.3][nn-supp]). For a binary regressor $c$ with
levels $\pm1$, re-estimating continuous unknowns under each hypothesis allows decoding
without knowing their true values. Compare the profile log-likelihoods,

$$
x^{+}=\hat x_{-c}\mid x_c=+1,\qquad x^{-}=\hat x_{-c}\mid x_c=-1\quad\text{by (M47a)},\tag{M48a}
$$

$$
\mathrm{LLR}_c=\ell\big(x_c=+1,\,x_{-c}=x^{+}\big)-\ell\big(x_c=-1,\,x_{-c}=x^{-}\big),\qquad
P(x_c=+1\mid Y)=\frac{e^{\mathrm{LLR}_c}}{1+e^{\mathrm{LLR}_c}} .\tag{M48b}
$$

The sigmoid in (M48b) normalises the two profile likelihoods with equal weights; it is
not a Bayesian posterior over nuisance regressors. The returned LLR is the second
supplied level's profile log-likelihood minus the first's. With several discrete
unknowns, the decoder enumerates their level combinations and re-solves only continuous
unknowns; ties go to the first combination. See [`decode`][mtdr.model.MTDR.decode] for
options and return shapes.

Use `known=` for measured task values and `levels=` for discrete unknowns. The paper
decodes stimuli given discrete values, and choice or context by re-estimating continuous
unknowns ([Nature Neuroscience supplement, §6.3][nn-supp]). The package also accepts
known continuous values when decoding discrete variables. The paper's pseudo-trial
resampling is not implemented.

### 9.2 Trajectories

Trajectories show how population activity evolves within an encoding subspace. Write
$y(t)=Y[:,t]$ and $\hat b(t)=\hat b[:,t]$. With every neuron observed, the paper's
precision-weighted projection is

$$
v_p(t)=\hat W_p^\top\,\hat D\,\big(y(t)-\hat b(t)\big)\in\mathbb R^{r_p},\qquad \hat D=\mathrm{diag}\hat\lambda,\tag{M49}
$$

This is [`project(method="paper")`][mtdr.model.MTDR.project]
([Nature Neuroscience supplement, §7][nn-supp]). The time-dependent readout (M50)
converts it to the instantaneous decoder ([§D.5](#d5-decoder-identities)).

With missing neurons, (M49) changes scale with the observed population, so trajectories
are not comparable across trials that observed different neurons. The default
[`project(method="gls")`][mtdr.model.MTDR.project] normalises for the observed basis and
returns weighted least-squares coordinates in the subspace over the neurons
$\mathcal I_k$ observed on trial $k$, with
$D_k=\mathrm{diag}(\hat\lambda_i)_{i\in\mathcal I_k}$ and
$U_{p,k}=\hat W_p[\mathcal I_k,:]$:

$$
z_{p,k}(t)=\big(U_{p,k}^\top D_kU_{p,k}\big)^{-1}U_{p,k}^\top D_k\big(y_k(t)-\hat b(t)\big)[\mathcal I_k]\in\mathbb R^{r_p}.\tag{M49a}
$$

With a full mask this is the fixed map $(\hat W_p^\top\hat D\hat W_p)^{-1}$ applied to
(M49), so the two span the same subspace. It is a single-subspace projection: other
regressors' contributions leak into it unless the subspaces are orthogonal in the
weighted inner product over the observed neurons;
[`project(regressor=None)`][mtdr.model.MTDR.project] solves the joint problem over all
the bases at once. If a trial's observed rows do not have full column rank to numerical
tolerance, its coordinates are `NaN` and a
[`ProjectionWarning`][mtdr.errors.ProjectionWarning] counts such trials; if the joint
basis itself is rank-deficient, [`project(regressor=None)`][mtdr.model.MTDR.project]
raises [`SingularDesignError`][mtdr.errors.SingularDesignError]. `weighted=False`
replaces $D_k$ by the identity in either method; `basis=` substitutes another basis,
which may have a different width (for example from
[`orthogonalize`][mtdr.model.MTDR.orthogonalize],
[§10.1](#101-identifiability-and-the-canonical-frame) ).

### 9.3 Subspace overlap

Subspace overlap measures how much two task variables share encoding directions.
[`MTDR.subspace_angles`][mtdr.model.MTDR.subspace_angles] returns principal angles
between the column spaces of $\hat W_a$ and $\hat W_b$; their cosines are the canonical
correlations ([Nature Neuroscience supplement, §8.2][nn-supp]). Use the fitted bases:
[`orthogonalize`][mtdr.model.MTDR.orthogonalize] removes the overlap being measured. The
paper's permutation null is not implemented.

## 10. Identifiability, scale and degenerate data

### 10.1 Identifiability and the canonical frame

Individual factors are not unique, so compare fits through their coefficient matrices
and subspaces rather than component by component. Rotating $S_p\to S_pQ_p$ and $w_{i[p]}\to Q_p^\top w_{i[p]}$ with
**orthogonal** $Q_p$ preserves (M24) and the prior. Any invertible change $W_p\to W_pR$,
$S_p\to S_pR^{-\top}$ preserves $\hat B_p$, but only orthogonal $R$ preserves the
marginal likelihood and prior. The canonical frame is therefore a **display frame**:
[`log_likelihood_`][mtdr.model.MTDR], [`aic_`][mtdr.model.MTDR] and
[`mtdr.mmle`][mtdr.mmle] use the raw fitted $(S_p,\lambda,b)$. Do not pass canonical
[`S_`][mtdr.model.MTDR] to functional MMLE routines.

For display, the SVD $\hat B_p=U_p\Sigma_pV_p^\top$ orders axes by variance of
$\hat B_p$ explained: the **PC orientation**
([Nature Neuroscience supplement, §4.1][nn-supp]).
[`MTDR(canonicalize=True)`][mtdr.model.MTDR], the default, sets $W_p=U_p$ and
$S_p=V_p\Sigma_p$ after fitting, with the entry of largest absolute value in each column
of $U_p$ positive. Components with (nearly) tied singular values are determined only up
to a rotation within the tied block.

[`orthogonalize(order=...)`][mtdr.model.MTDR.orthogonalize] removes each subspace's
overlap with earlier subspaces, in the chosen order
([Nature Neuroscience supplement, §4.2][nn-supp]). The returned width can shrink to
zero. Use these bases with [`project(basis=...)`][mtdr.model.MTDR.project] for display;
they cannot reconstruct [`B_`][mtdr.model.MTDR].

### 10.2 Regressor scale and origin

Centre continuous regressors and code binary regressors as $\pm1$. The marginal
likelihood depends on regressor origin, so compare alternative codings using held-out
plug-in likelihood. Global centring does not generally centre each neuron's observed
design under a mask.

At `basis_ridge=0`, rescaling a regressor can be offset by inversely scaling its
coefficient matrix, leaving the marginal model unchanged. Numerical optimisation still
depends on conditioning and tolerances. The package does not standardise inputs
automatically; [`check_design`][mtdr.data.check_design] reports scale disparities and
uncentred continuous regressors. See [Appendix E](#appendix-e-scale-and-degenerate-data)
for the identities.

### 10.3 Degenerate neurons

Too few observations or an exact fit can leave noise precision unbounded.
[`fit`][mtdr.model.MTDR.fit] rejects, with a
[`ValidationError`][mtdr.errors.ValidationError] naming them, neurons observed on fewer
than `min_observations` trials (under `"mmle"` at least $P_b+2$, or $P_b+1$ without the
intercept), and the MMLE stage also rejects any neuron whose unconstrained least-squares
residual is zero to rounding. A rank-deficient neuron that keeps residual degrees of
freedom (a regressor constant within a long session) is well posed and fitted; its
least-squares initialiser uses the minimum-norm solution
([§3.1](#31-least-squares-and-truncation)).

See [§E.2](#e2-residual-degrees-of-freedom)
–[§E.3](#e3-rounding-level-ends-of-the-inner-optimisers) for residual degrees of freedom
and optimiser rounding rules.

## Technical appendices

These appendices give the notation, derivations and numerical details behind the
estimators; §§1–10 above cover use and interpretation.

## Appendix A. Notation and statistics

### A.1 Internal notation

The symbols of [§1](#1-notation-and-array-layouts) apply throughout. Additional symbols
used in the derivations:

| Symbol | Meaning |
|---|---|
| $w_i$ | Concatenated neuron weights, (M4) |
| $\mathbf S$ | Block-diagonal temporal bases, (M7) |
| $\mathcal R_p$ | Row block $\sum_{q<p}r_q+1,\ldots,\sum_{q\le p}r_q$ of $\mathbf S$ |
| $\mathcal C_p$ | Column block $(p-1)T+1,\ldots,pT$ of $\mathbf S$ |
| $s$, $\theta$ | Packed bases (M19) and parameters (M28) |

Functional `S`, `W`, `B` are lists of `(T, r_p)`, `(n, r_p)`, `(n, T)` arrays. The
statistic arrays `xi`, `A`, `n_obs` have shapes `(n, P_b, T)`, `(n, P_b, P_b)`, `(n,)`.
The mathematical intercept $b$ is $T\times n$, whereas the public `intercept` array is
`(n, T)`.

### A.2 Conventions

These conventions specify how the equations stack arrays and index their blocks.

- $\mathrm{vec}(M)$ stacks the columns of $M$: for $M\in\mathbb R^{a\times b}$,
  $[\mathrm{vec}M]_{u+a(v-1)}=M_{uv}$. Kronecker products put the left factor's index
  outermost, so $\mathrm{vec}(AMB)=(B^\top\otimes A)\,\mathrm{vec}(M)$.
- Every $P_bT$-vector is ordered **time fastest, regressor slowest**: position
  $t+T(p-1)$.
- Index sets are 1-based in the equations; the code subtracts one.
- Blocks: for $M\in\mathbb R^{r_{tot}\times r_{tot}}$, $M_{[pq]}=M[\mathcal R_p,\mathcal R_q]$;
  for $v\in\mathbb R^{r_{tot}}$, $v_{[p]}=v[\mathcal R_p]$; for $u\in\mathbb R^{P_bT}$,
  $u_{\langle p\rangle}=u[\mathcal C_p]\in\mathbb R^T$.

### A.3 Per-neuron form

Stacking each neuron's observed trials makes its independent likelihood explicit. For
neuron $i$ let $k_1<\dots<k_{n_i}$ enumerate $\mathcal K_i$ and

$$
z_i\in\mathbb R^{T\times n_i},\quad z_i[t,j]=Z_{k_j}[i,t];\qquad
\zeta_i=\mathrm{vec}(z_i)\in\mathbb R^{Tn_i},\tag{M5}
$$

$$
X_i\in\mathbb R^{n_i\times P_b},\ X_i[j,:]=x_{k_j}^\top;\qquad A_i=X_i^\top X_i\in\mathbb R^{P_b\times P_b},\tag{M6}
$$

$$
\mathbf S=\mathrm{blkdiag}(S_1^\top,\dots,S_{P_b}^\top)\in\mathbb R^{r_{tot}\times P_bT},\tag{M7}
$$

$$
\Phi_i=(X_i\otimes I_T)\,\mathbf S^\top\in\mathbb R^{n_iT\times r_{tot}},\qquad
\Phi_i[\text{row block }j,\,\mathcal R_p]=x_{k_jp}\,S_p .\tag{M8}
$$

Then (M1)-(M4) read, per neuron,

$$
\zeta_i=\mathbf 1_{n_i}\otimes b_i+\Phi_iw_i+\varepsilon_i,\qquad
\varepsilon_i\sim\mathcal N(0,\lambda_i^{-1}I_{n_iT}),\qquad w_i\sim\mathcal N(0,I_{r_{tot}}),\tag{M9}
$$

and neurons are independent given $(S,\lambda,b)$.

### A.4 Sufficient statistics (`MkSuffStatsBTDR_IncompObs_uneqvar_S_fast`)

Sufficient statistics retain the information needed for fitting without repeatedly
reading the responses. With $\zeta_i$ the raw responses,

$$
\xi_i=(X_i^\top\otimes I_T)\,\zeta_i=\mathrm{vec}(z_iX_i)\in\mathbb R^{P_bT},\qquad
[\xi_i]_{t+T(p-1)}=\sum_{j=1}^{n_i}x_{k_jp}\,z_i[t,j],\tag{M10}
$$

$$
R_i=\xi_i\xi_i^\top,\qquad
\upsilon_i=\zeta_i^\top\zeta_i=\lVert z_i\rVert_F^2,\qquad
n_i=\sum_kh_{ki},\tag{M11}
$$

$$
\bar x_i=\frac1{n_i}\sum_{k\in\mathcal K_i}x_k\in\mathbb R^{P_b},\qquad
\bar y_i=\frac1{n_i}\sum_{k\in\mathcal K_i}Z_k[i,:]^\top\in\mathbb R^{T}.\tag{M12}
$$

$R_i$ is rank one and never formed; every use is written in terms of $\xi_i$. Unobserved
entries of `Y` do not contribute to these sums.

### A.5 Intercept-centred statistics (`ECMEsuffstat`)

Each intercept update changes the residual responses used in the likelihood. Given $b$,
write $\tilde z_i=z_i-b_i\mathbf 1_{n_i}^\top$:

$$
\xi_i(b)=\mathrm{vec}\big(\tilde z_iX_i\big)=\xi_i-n_i\,(\bar x_i\otimes b_i),\tag{M13}
$$

$$
\upsilon_i(b)=\lVert\tilde z_i\rVert_F^2=\upsilon_i-2n_i\,\bar y_i^\top b_i+n_i\lVert b_i\rVert^2,\qquad
R_i(b)=\xi_i(b)\xi_i(b)^\top.\tag{M14}
$$

The implementation accumulates moments about each neuron's means to avoid cancellation
when subtracting a large baseline. See [`SufficientStats`][mtdr.stats.SufficientStats]
for the stored arrays.

### A.6 Statistics of the least-squares stage (`MkSuffStats_BilinReg_Sims`)

To fit the intercept alongside the coefficients, least squares appends a constant:
$X^f_i=[X_i\ \mathbf 1]\in\mathbb R^{n_i\times(P_b+1)}$, $A^f_i=X_i^{f\top}X^f_i$ and
$\xi^f_i=\mathrm{vec}(z_iX^f_i)$, i.e. (M10) with the full design. The solve is
independent for each neuron ([§3.1](#31-least-squares-and-truncation)).

## Appendix B. SVD implementation and reference formulas

### B.1 SVD numerical form

Centred moments avoid losing precision when responses have a large baseline. Eliminating
the intercept from (M16) gives the exact Schur complement,

$$
\big(\tilde A_i+\gamma I+c_i\,\bar x_i\bar x_i^\top\big)\,\beta^{b}_i=\tilde\xi_i+c_i\,\bar x_i\bar y_i^\top,
\qquad c_i=\frac{n_i\gamma}{n_i+\gamma},\qquad
\hat b^{\rm full}_i=\frac{n_i}{n_i+\gamma}\big(\bar y_i-\beta^{b\top}_i\bar x_i\big),
$$

Here $\beta_i^b$ is the $P_b\times T$ block of regressor coefficients.

With $\gamma=0$, this form is shift-invariant in $X$ and $Y$. A singular system uses the
minimum-norm $\beta_i^b$ with the intercept profiled out and emits a
[`DesignWarning`][mtdr.errors.DesignWarning]; a ridge large enough relative to the Gram
matrix removes this case. A positive ridge penalises the intercept too and breaks shift
invariance.

The packed SVD bases are

$$
s^{svd}=\big(\mathrm{vec}\,\hat S^{svd}_1;\dots;\mathrm{vec}\,\hat S^{svd}_{P_b}\big).\tag{M19}
$$

### B.2 Reference residuals

These formulas describe the MATLAB implementation, not the package estimator. For the
consequences, see
[Differences from MATLAB](differences-from-matlab.md#reference-defects-that-are-fixed).

*What the reference computes.* `SVDRegress_S_Vdata` subtracts the prediction from the
responses in two different orders (trial-fastest against time-fastest), so its residual
pairs mismatched entries:

$$
\hat\lambda^{svd,\text{ref}}_i=\frac{n_iT}{\lVert\rho^{\text{ref}}_i\rVert^2},\qquad
\rho^{\text{ref}}_i=\mathrm{vec}\big(z_i^\top\big)-\mathrm{vec}\big(M_iX_i^{f\top}\big),\quad
M_i=\big[\hat B^{(r_1)}_1[i,:]^\top,\dots,\hat B^{(r_{P_b})}_{P_b}[i,:]^\top,\ \hat b^{\rm full}_i\big].\tag{M19a}
$$

Aligning the residual but retaining the full-rank intercept instead gives

$$
\mathrm{RSS}^{\rm joint}_i=\big\lVert z_i-M_iX_i^{f\top}\big\rVert_F^2 ,\tag{M19b-ref}
$$

### B.3 Reference SVD AIC

These formulas explain why MATLAB SVD scores differ from the scores in
[§3.3](#33-the-svd-stage-aic). `SVDRegB_AIC` reshapes the coefficient array so that its
rows are not the neurons' coefficients, misaligns the residual as in (M19a), divides by
the regressor count $P=P_b+1$ instead of the observation count and flips the sign of the
log-precision term:

$$
\tilde B=\mathrm{reshape}(\texttt{wsvd},\,n,\,TP),\quad
\beta^{\text{ref}}_i=\tilde B[i,:]^\top,\quad
\rho^{\text{ref}}_i=\mathrm{vec}(z_i^\top)-\mathrm{vec}\big(\mathrm{reshape}(\beta^{\text{ref}}_i,T,P)\,X_i^{f\top}\big),\tag{M20a}
$$

$$
\hat\lambda^{\text{ref}}_i=\frac{PT}{\lVert\rho^{\text{ref}}_i\rVert^2},\quad
\mathrm{NLL2}^{\text{ref}}=\sum_{i=1}^n\Big[\lVert\rho^{\text{ref}}_i\rVert^2\hat\lambda^{\text{ref}}_i+n_iT\log\hat\lambda^{\text{ref}}_i\Big],\quad
K^{\text{ref}}=\Big(nP+TP-\textstyle\sum_pr_p\Big)\textstyle\sum_pr_p,\tag{M20b}
$$

$$
\mathrm{AIC}^{\text{ref}}_{svd}(r)=\mathrm{NLL2}^{\text{ref}}+2K^{\text{ref}}.\tag{M20c}
$$

The count $K^{\text{ref}}$ over-penalises by roughly a factor $P$.
[`mtdr.aic.n_parameters_svd(formula="reference")`][mtdr.aic.n_parameters_svd] gives it;
the tests reproduce the rest.

## Appendix C. Likelihood and estimation

### C.1 Blockwise likelihood evaluation

Low-rank blocks reduce likelihood evaluation to $r_{tot}\times r_{tot}$ systems per neuron.
With $\Gamma_{pq}=S_p^\top S_q\in\mathbb R^{r_p\times r_q}$ shared across neurons,

$$
(\mathbf SK_i\mathbf S^\top)_{[pq]}=[A_i]_{pq}\,\Gamma_{pq},\qquad
u_i:=\mathbf S\,\xi_i(b),\quad (u_i)_{[p]}=S_p^\top\,\xi_i(b)_{\langle p\rangle}\in\mathbb R^{r_p},\tag{M23}
$$

so neither $\mathbf S$ nor $K_i$ nor $\Phi_i$ is ever formed. By the matrix determinant
lemma and Woodbury, $\log\lvert\Sigma_i\rvert=-n_iT\log\lambda_i+\log\lvert C_i\rvert$,
$\Sigma_i^{-1}=\lambda_iI-\lambda_i^2\Phi_iC_i^{-1}\Phi_i^\top$.

The marginal covariance is [Nature Neuroscience supplement, eq. (12)][nn-supp], with
$F_i=\Phi_i$. The public likelihood includes the Gaussian constant. Cholesky solves
evaluate (M24) and (M25); see
[`marginal_log_likelihood`][mtdr.mmle.marginal_log_likelihood].

### C.2 Gradient with respect to the bases (`neglogLikBTDR_IncompObs_uneqvar_Sonly`)

The basis gradient guides refinement of the shared temporal patterns. With $\lambda$
fixed,

$$
\frac{\partial\mathcal L}{\partial\mathbf S}=\sum_{i=1}^n\Big[\lambda_iC_i^{-1}\mathbf SK_i-\lambda_i^2\,C_i^{-1}\mathbf SR_i(b)\big(I_{P_bT}-\lambda_i\mathbf S^\top C_i^{-1}\mathbf SK_i\big)\Big]+g\,\mathbf S,\tag{M26a}
$$

keeping only the active blocks $(\mathcal R_p,\mathcal C_p)$. Using
$\lambda_i^2C_i^{-1}\mathbf SR_i(b)=\lambda_i\mu_i\xi_i(b)^\top$ gives the equivalent
blockwise form used by the estimator:

$$
\frac{\partial\mathcal L}{\partial S_p}=\sum_{i=1}^n\lambda_i\Big[\sum_{q=1}^{P_b}[A_i]_{pq}\,S_q\,(\Omega_i)_{[qp]}-\xi_i(b)_{\langle p\rangle}\,(\mu_i)_{[p]}^\top\Big]+g\,S_p\in\mathbb R^{T\times r_p}.\tag{M26b}
$$

### C.3 Gradient with respect to the precisions (`neglogLikBTDR_IncompObs_uneqvar_lambonly`)

The precision gradient balances each neuron's expected residual against its observation
count. With $\mathbf S$ fixed,

$$
\frac{\partial\mathcal L}{\partial\lambda_i}=\frac12\Big[-\frac{n_iT}{\lambda_i}+\mathrm{tr}\big(C_i^{-1}\mathbf SK_i\mathbf S^\top\big)+\upsilon_i(b)-2\lambda_iq_i+\lambda_i^2\,\mathrm{tr}\big(C_i^{-1}\mathbf SK_i\mathbf S^\top C_i^{-1}\mathbf SR_i(b)\mathbf S^\top\big)\Big],\tag{M27a}
$$

or equivalently, with (M25),

$$
\frac{\partial\mathcal L}{\partial\lambda_i}=\frac12\Big[-\frac{n_iT}{\lambda_i}+\mathcal E_i\Big],\qquad
\mathcal E_i:=\upsilon_i(b)-2\,u_i^\top\mu_i+\mathrm{tr}\big(\mathbf SK_i\mathbf S^\top\Omega_i\big)
=\mathbb E_{w_i\mid\cdot}\big\lVert\tilde\zeta_i-\Phi_iw_i\big\rVert^2,\tag{M27b}
$$

with
$\mathrm{tr}(\mathbf SK_i\mathbf S^\top\Omega_i)=\sum_{p,q}[A_i]_{pq}\,\mathrm{tr}\big(\Gamma_{pq}(\Omega_i)_{[qp]}\big)$.

### C.4 Parameter packing

Parameter packing defines the vector used for counts and convergence:

$$
\theta=\big(\lambda_1,\dots,\lambda_n;\ \mathrm{vec}S_1;\dots;\mathrm{vec}S_{P_b};\ \mathrm{vec}\,b\big),\qquad
\text{length }n+Tr_{tot}+Tn .\tag{M28}
$$

### C.5 Initialisation (`ECMEregress_wrapper`)

Without a supplied fit, MMLE starts from SVD bases and precisions:

$$
\text{run (M16)-(M19b) at ranks }r;\qquad
\lambda^{(0)}=\hat\lambda^{svd},\quad S^{(0)}_p=\hat S^{svd}_p,\quad b^{(0)}_i=\bar y_i .\tag{M29}
$$

The least-squares intercept is discarded; $b$ starts at the neuron's mean response
(M12).

### C.6 ECME (`ECMEtdr`)

ECME alternates posterior evaluation with closed-form parameter updates to initialise
refinement ([§5](#5-ecme-and-coordinate-ascent)). One iteration maps
$(\lambda,S,b)\mapsto(\lambda',S',b')$:

**Step 0.** $\xi_i(b)$, $\upsilon_i(b)$ from (M13)-(M14).

**Step 1 (E-step at $(\lambda,S,b)$).** $C_i$, $u_i$, $\mu_i$, $\Omega_i$ from
(M22)-(M25).

**Step 2 (CM-step for $\lambda$).**

$$
\lambda'_i=\frac{n_iT}{\mathcal E_i},\qquad
\mathcal E_i\ \text{ as in (M27b)}.\tag{M30}
$$

**Step 3 (CM-step for $S$, multicycle).** Redo the E-step at $(\lambda',S,b)$:
$C'_i=I+\lambda'_i\mathbf SK_i\mathbf S^\top$, $\mu'_i=\lambda'_iC_i'^{-1}u_i$,
$\Omega'_i=\mu'_i\mu_i'^\top+C_i'^{-1}$; then maximise
$\sum_i\lambda'_i\,\mathbb E_{w\mid\lambda',S}\big[-\lVert\tilde\zeta_i-\Phi_i(S')w_i\rVert^2\big]$
over $S'$. Setting the derivative to zero gives one $r_{tot}\times r_{tot}$ linear
system; with $\mathcal S'=[S'_1,\dots,S'_{P_b}]\in\mathbb R^{T\times r_{tot}}$,

$$
\mathcal G\,\mathcal S'^{\top}=\mathcal M,\qquad
\mathcal G_{[pq]}=\sum_{i=1}^n\lambda'_i\,[A_i]_{pq}\,(\Omega'_i)_{[pq]},\qquad
\mathcal M[\mathcal R_p,:]=\sum_{i=1}^n\lambda'_i\,(\mu'_i)_{[p]}\,\xi_i(b)_{\langle p\rangle}^\top .\tag{M31}
$$

The default uses the multicycle update of
[NeurIPS supplement, Algorithm 1][neurips-paper].
[`ecme(matlab_compat=True)`][mtdr.mmle.ecme] reproduces the reference implementation for
parity checks; see [Differences from MATLAB](differences-from-matlab.md).

**Step 4 (CM-step for $b$ on the marginal likelihood, `MMLE_b`).** With $(\lambda',S')$
fixed, $\mathcal L$ is quadratic in $b_i$, and its minimiser is the GLS estimate of the
mean of $\zeta_i\sim\mathcal N(\mathbf 1\otimes b_i,\Sigma_i)$. With

$$
\Psi_i=(\bar x_i^\top\otimes I_T)\,\mathbf S'^{\top}=\big[\bar x_{i1}S'_1,\;\dots,\;\bar x_{iP_b}S'_{P_b}\big]\in\mathbb R^{T\times r_{tot}},\tag{M32}
$$

$$
b'_i=\Big[I_T-\lambda'_in_i\,\Psi_iC_i'^{-1}\Psi_i^\top\Big]^{-1}\Big(\bar y_i-\lambda'_i\,\Psi_iC_i'^{-1}\,\mathbf S'\xi_i\Big),
\qquad\xi_i\ \text{the raw (M10)},\tag{M33}
$$

with $C'_i=I+\lambda'_i\mathbf S'K_i\mathbf S'^\top$ rebuilt from the new blocks.

The implementation evaluates the equivalent centred form; see
[`update_intercept`][mtdr.mmle.update_intercept].

**Step 5 (convergence).** With $\theta$ at the start of the iteration and $\theta'$ at
the end (all of $\lambda$, $s$ and $b$),

$$
\epsilon=\max_j\frac{(\theta'_j-\theta_j)^2}{\theta_j^2}.\tag{M34}
$$

The package adds `convergence_eps` ($10^{-12}$) to the denominator to handle zero
parameters. ECME stops below `ecme_tol=1.0`, with `ecme_max_iter=100`.

The default ECME checks that the unpenalised marginal negative log-likelihood does not
increase, to a rounding allowance. The reference complete-data diagnostic is

$$
Q(\theta';\theta)=\sum_{i=1}^n\Big[n_iT\log\lambda'_i-\lambda'_i\upsilon_i+2\lambda'_i\,u_i(S')^\top\mu_i-\mathrm{tr}\big(C'_i\Omega_i\big)\Big]
=2\,\mathbb E_{w\mid\theta}\big[\log p(\tilde\zeta,w\mid\theta')\big]+\text{const}\tag{M35}
$$

It is not monotone under the multicycle step and is not used.

### C.7 Coordinate ascent on the marginal likelihood (`Estpars_CoordAscent_lambi_S_b`)

Refinement directly improves the marginal likelihood after ECME's loose stopping test.
From its output, repeat (at most `refine_max_iter` $=10$ times):

1. the statistics $\xi_i(b)$, $\upsilon_i(b)$ at the current $b$, (M13)-(M14);
2. $s\leftarrow\arg\min_s\mathcal L(\lambda,s;b)$ with the gradient (M26b);
3. $\lambda\leftarrow\arg\min_\lambda\mathcal L(\lambda,s;b)$ with the gradient (M27b);
4. $b\leftarrow$ (M33) at the new $(S,\lambda)$;
5. stop when the change of [§C.6](#c6-ecme-ecmetdr) over $(\lambda;s;\mathrm{vec}\,b)$ is below
   `refine_tol` $=10^{-4}$.

The package uses SciPy L-BFGS-B for the basis and precision steps, with precisions
optimised on a logarithmic scale. `basis_ridge` applies in this refinement stage.
Acceptance requires a finite objective no worse than the current point, to a rounding
allowance. Defaults and reference differences are listed in
[Differences from MATLAB](differences-from-matlab.md#optimisers-replaced-not-reproducible-exactly);
see [§E.3](#e3-rounding-level-ends-of-the-inner-optimisers) for rounding-level ends.

## Appendix D. Simulation and decoder identities

The simulation formulas describe the MATLAB demo's data-generating process, used by
[`simulate`][mtdr.simulation.simulate] ([§8](#8-simulation-model)). The NeurIPS
paper's simulations differ: iid $\mathcal N(0,1)$ bases and weights, noise variances
rather than precisions drawn from an Exponential, and a different observation
probability.

### D.1 Weights and bases (`SimWeights`)

Weight amplitude and temporal length scale control signal strength and smoothness. For
each regressor $p$,

$$
\Sigma^{(p)}_S[t,t']=\exp\Big(-\frac{(t-t')^2}{2\ell_p^2}\Big),\qquad
W_p=\rho_p\cdot G_p,\ G_p\in\mathbb R^{n\times r_p}\text{ iid }\mathcal N(0,1),\tag{M40}
$$

$$
S_p=\mathrm{flip}\big(\texttt{mvnrnd}(0,\Sigma^{(p)}_S,r_p)^\top\big)\in\mathbb R^{T\times r_p}
\quad(\text{each column a draw from }\mathcal N(0,\Sigma^{(p)}_S),\text{ time-reversed}),\tag{M41}
$$

$$
\texttt{BB}=\big[W_1S_1^\top;\ \dots;\ W_PS_P^\top\big]\in\mathbb R^{Pn\times T}.\tag{M42}
$$

$\rho_p$ scales $W_p$ (a standard deviation); the kernel has unit variance. The bases
are raw Gaussian-process draws, not orthonormalised. Demo: $\ell_p=2$, $\rho_p=1$,
$r_p\sim\mathrm{Uniform}\{1,\dots,6\}$.

### D.2 Conditions (`SimConditions`)

Conditions are sampled independently, so their trial counts need not be equal:

$$
x_k\sim\mathrm{Uniform}\big(\mathrm{levels}_1\times\dots\times\mathrm{levels}_{P_b}\big)\ \text{iid},\tag{M43}
$$

drawn with replacement from the grid of all level combinations. Demo levels:
$\{-2,\dots,2\}$, $\{-2,\dots,2\}$, $\{-1,1\}$.

### D.3 Responses (`SimPopData`)

Responses combine the task-dependent signal, intercept and independent noise:

$$
Y_k=\sum_{p}x_{kp}\,B_p+b^\top+\mathrm{diag}(d)^{-1/2}\,G_k,\qquad G_k\in\mathbb R^{n\times T}\text{ iid }\mathcal N(0,1),\tag{M44}
$$

Here $d_i$ is the noise **precision** $\lambda_i$,
[`simulate`][mtdr.simulation.simulate]'s `noise_precision`. By default,
$d_i\sim\mathrm{Exp}(\text{mean }1.25)$. The intercept $b^\top=W_0S_0^\top$ is drawn
like the demo's constant term, a term of rank $T$ under (M40)-(M41).
[`simulate`][mtdr.simulation.simulate] also accepts a design `X`, bases `S` or weights
`W` in place of the corresponding draws.

### D.4 Mask

Missingness removes whole neuron-trials. The demo uses

$$
h_{ki}\sim\mathrm{Bernoulli}(1-p_{drop})\ \text{iid},\quad p_{drop}=0.3 .\tag{M45}
$$

[`simulate`][mtdr.simulation.simulate] defaults to `drop_prob=0`. It re-draws $h$ until
every trial has an observed neuron and every neuron at least two trials. This
conditioning can raise the observed fraction in sparse data; it does not ensure MMLE
validity ([§10.3](#103-degenerate-neurons)).

### D.5 Decoder identities

These identities connect continuous estimates, binary evidence and projected
trajectories. Vectorisation puts neurons fastest and bins slowest. When all other
unknown regressors are continuous, the binary LLR obeys

$$
\mathrm{LLR}_c=\frac{2\,\hat x_c}{\big[(\Xi^\top\Lambda\Xi)^{-1}\big]_{cc}}\quad\text{for levels }\{+1,-1\},\tag{M48c}
$$

Here the sign convention is positive-level minus negative-level, as in (M48b),
regardless of the supplied level order. The package returns second level minus first.
The identity does not apply when other discrete regressors are enumerated.

With $\hat s_p(t)=\hat S_p[t,:]^\top$, the time-dependent readout of (M49) is

$$
\hat x(t)=\Big[\hat B(t)^\top\hat D\,\hat B(t)\Big]^{-1}\mathrm{blkdiag}\big(\hat s_1(t)^\top,\dots,\hat s_{P_b}(t)^\top\big)\begin{pmatrix}v_1(t)\\\vdots\\v_{P_b}(t)\end{pmatrix},
\qquad\hat B(t)=\big(\hat B_1[:,t],\dots,\hat B_{P_b}[:,t]\big),\tag{M50}
$$

This equals (M47) at a single bin ([Nature Neuroscience supplement, eq. (27)][nn-supp]).

## Appendix E. Scale and degenerate data

### E.1 Scale and origin identities

These identities distinguish changes of units from changes of regressor origin
([§10.2](#102-regressor-scale-and-origin)). At $g=0$, $X_{:p}\to a_pX_{:p}$ with
$S_p\to S_p/a_p$ leaves $\Phi_i$ and (M24) unchanged. Under $Y\to cY$, $S\to cS$,
$b\to cb$ and $\lambda\to\lambda/c^2$, the posterior and whitened products
$\lambda_i u_i$, $\lambda_i\mathbf SK_i\mathbf S^\top$ are unchanged; (M24) shifts by
$\sum_i n_iT\log|c|$.

Shifting a regressor changes the marginal covariance. At fixed $(S,\lambda)$ with the
intercept optimised, the origin dependence is

$$
\ell=\ell_{\rm within}(S,\lambda)-\tfrac12\sum_i\log\big\lvert
I_T+\lambda_in_i\Psi_i\tilde C_i^{-1}\Psi_i^\top\big\rvert,\qquad
\Psi_i=\big[\bar x_{i1}S_1,\dots,\bar x_{iP_b}S_{P_b}\big],
$$

Here $\ell_{\rm within}$ depends only on moments about each neuron's means,
$\tilde C_i=I+\lambda_i\mathbf S(\tilde A_i\otimes I_T)\mathbf S^\top$, and
$\tilde A_i$, $\tilde\xi_i$ are the centred moments used by
[`SufficientStats`][mtdr.stats.SufficientStats]. The determinant penalty vanishes when
each observed design has zero mean. With a fitted intercept and `ridge=0`, the SVD stage
is shift-invariant; its ridge penalises the intercept too.

### E.2 Residual degrees of freedom

Without residual degrees of freedom, exact fits can drive precision to infinity. A
necessary condition for a bounded likelihood is
$n_i>\operatorname{rank}[X_i\ \mathbf 1]$ (omit the constant without an intercept). The
package also checks the unconstrained residual against rounding; see
[`fit_mmle`][mtdr.mmle.fit_mmle].

### E.3 Rounding-level ends of the inner optimisers

Near an optimum, rounding can stop a line search before its gradient test passes. An
accepted basis-step line-search end with finite value and gradient counts as converged
when its last decrease is below $100\max(\texttt{optimizer\_progtol},\epsilon M)$, with
$M$ the objective's rounding scale. A precision-step line-search end is polished by at
most three diagonal Newton steps and accepted only when every neuron's scale-free
residual $|\lambda_i\mathcal E_i/(n_iT)-1|\le10^{-8}$. See [`refine`][mtdr.mmle.refine]
for the acceptance rules and the
[convergence guide](api/model.md#converged_false-what-it-means-and-what-to-do) for how
to assess a warning.

[nn-supp]: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-020-0696-5/MediaObjects/41593_2020_696_MOESM1_ESM.pdf
[neurips-paper]: https://proceedings.neurips.cc/paper/2018/hash/8a1ee9f2b7abe6e88d1a479ab6a42c5e-Abstract.html
