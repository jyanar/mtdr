function make_svd_fixture(refdir, simfile, outfile)
% MAKE_SVD_FIXTURE  Parity fixture for mtdr.stats, mtdr.svd_fit and
% mtdr.rank_search.
%
%   make_svd_fixture(refdir, simfile, outfile)
%
% Two datasets, each run through the reference SVD path and saved as a struct:
%
%   A  the simulation fixture (Z, X, hk of SIMFILE), so the fixtures chain:
%      n = 20 neurons, T = 15 bins, N = 40 trials.
%   B  a demo-scale draw (n = 100, T = 15, N = 40) made with the reference
%      SimWeights / SimConditions / SimPopData and the mTDRdemo.m mask step at
%      rng(1), on which the reference rank search moves (on A it does not).
%      Its responses are rounded to single precision before any computation
%      and stored as single, so the file stays small and Python sees exactly
%      the values MATLAB used.
%
% For each dataset: the reference sufficient statistics
% (MkSuffStats_BilinReg_Sims, MkSuffStatsBTDR_IncompObs_uneqvar_S_fast,
% ECMEsuffstat at a fixed intercept, the mTDRdemo.m xbari/Ybar loop); the
% unconstrained solution XX\XY; SVDRegressB and SVDRegB_AIC on a list of rank
% vectors; SVDRegress_S_Vdata at the true ranks; and EstRankGreedily with
% SVDRegB_AIC exactly as mTDRdemo.m calls it, from several starting vectors,
% logging every objective evaluation. As an independent second implementation
% of what the port computes, it also evaluates the corrected quantities
% ((M19b) aligned residuals, (M21) fully normalised plug-in log-likelihood,
% textbook count and AIC) from the per-neuron residuals, and runs
% EstRankGreedily on that corrected objective. The corrected quantities follow
% the port, not the reference: each neuron's coefficients come from least
% squares on its centred observed design, each task block is truncated by its
% own SVD, and the intercept is refitted given the truncated blocks,
% b_i = ybar_i - sum_p xbar_ip B_p(i,:).
%
%   refdir   path to the mTDRdemo directory (read-only reference)
%   simfile  tests/fixtures/simulation.mat (make_simulation_fixture.m)
%   outfile  .mat file to write (tests/fixtures/svd.mat)
%
% Run from the repository root with
%   matlab -batch "addpath('tests/fixtures/matlab'); make_svd_fixture('path/to/mTDRdemo', 'tests/fixtures/simulation.mat', 'tests/fixtures/svd.mat')"

addpath(fullfile(refdir, 'functionFiles'), fullfile(refdir, 'functionFiles', 'tools_kron'));

% ---- dataset A: the simulation fixture -------------------------------------
sim = load(simfile);
% Every task block has r_p ~= T (a transposed layout fails); row 4 has a
% rank-0 block; the constant term is appended at maxrank, as in the demo.
rank_list = [1 1 1; 2 1 3; 3 2 1; 0 2 1; 5 3 7; 15 1 15];
starts = [1 1 1; 1 4 1; 2 1 3];
A = run_dataset(sim.Z, sim.X, sim.hk, rank_list, starts, 2);
% A's data are not repeated here: the tests read them from SIMFILE.

% ---- dataset B: a demo-scale draw on which the reference search moves -----
rng(1);
n = 100; T = 15; N = 40; P = 4; rP = [2 1 3];
[~, ~, BB] = SimWeights(n, T, P, [rP T], 2 * ones(P, 1), ones(P, 1));
d = exprnd(1 / .8, [n 1]);
X = SimConditions({-2:2, -2:2, [-1 1], 1}, N);
Y = SimPopData(X, BB, d, n, T, N);
hk = zeros(N, n);
Z = zeros(n, T, N);
for k = 1:N
    hk(k, :) = binornd(1, 1 - 0.3, [1 n]);
    Z(:, :, k) = spdiags(hk(k, :)', 0, n, n) * squeeze(Y(:, :, k));
end
Z = double(single(Z));
rank_list = [1 1 1; 2 1 3; 4 2 5];
starts = [1 1 1];
B = run_dataset(Z, X, hk, rank_list, starts, 2);
B = rmfield(B, {'XX', 'XY', 'b0_vdata'});   % XX is 6000 x 6000; A tests these
B.Z = single(Z);
B.X = int8(X);
B.hk = uint8(hk);
B.rP = rP;

matlab_version = version;
% Re-running reproduces every array and the serialised payload exactly; the
% 128-byte MAT header records the creation time, so the file differs there only.
save(outfile, 'A', 'B', 'matlab_version');
fprintf('wrote %s\n', outfile);
end


function out = run_dataset(Z, X, hk, rank_list, starts, k_factors)
[n, T, N] = size(Z);
P = size(X, 2);          % the last column is the constant term
Pb = P - 1;
maxrank = min(n, T);     % as mTDRdemo.m
ridgeparam = 0;
opts.MaxIter = 500;
opts.Display = 'off';
assert(all(X(:, P) == 1), 'the last column of X must be the constant term');

% Sufficient statistics.
[XX, XY, Yn, allstim] = MkSuffStats_BilinReg_Sims(Z, X, hk);
Xb = X(:, 1:Pb);
[~, Ai, zzi, ni, Xi, Xzetai0, zetai0] = MkSuffStatsBTDR_IncompObs_uneqvar_S_fast(Xb, Z, hk);
xbari = zeros(n, Pb);
Ybar = zeros(T, n);
for ii = 1:n
    xbari(ii, :) = mean(Xi{ii}, 1);
    Ybar(:, ii) = squeeze(sum(Z(ii, :, :), 3)) / ni(ii);
end
% Centred statistics at a fixed intercept that is not the mean.
b_test = Ybar + 0.25 * cos((1:T)' * (1:n) / 7);
[~, zzi_c, Xzetai_c] = ECMEsuffstat(zetai0, Xi, b_test);

% Unconstrained least squares and truncations.
w0 = XX \ XY;
rank_list = [rank_list, maxrank * ones(size(rank_list, 1), 1)];
K = size(rank_list, 1);
wsvd_all = zeros(T, n, P, K);
aic_ref = zeros(K, 1);
rss_aligned = zeros(n, K);
loglik_corrected = zeros(K, 1);
nparam_textbook = zeros(K, 1);
nparam_reference = zeros(K, 1);
for k = 1:K
    r = rank_list(k, :);
    [wsvd, wt, wx] = SVDRegressB(XX, XY, [T n], r, ridgeparam, opts);
    wsvd_all(:, :, :, k) = wsvd;
    aic_ref(k) = SVDRegB_AIC(Yn, allstim, wsvd, r);
    [rss_aligned(:, k), loglik_corrected(k), nparam_textbook(k)] = ...
        corrected_score(centred_truncation(Yn, allstim, n, T, Pb, r), r, ...
        Yn, allstim, n, T, Pb, maxrank);
    nparam_reference(k) = (n * P + T * P - sum(r)) * sum(r);
    if k == k_factors
        % Factors: wt{p} is T x r_p, wx{p} is n x r_p (constant term last).
        out.wt = wt;
        out.wx = wx;
    end
end
aic_corrected = 2 * nparam_textbook - 2 * loglik_corrected;

% SVDRegress_S_Vdata at rank_list(k_factors, :): the code's (misaligned)
% lambda and the packed bases s = [vec(wt{1}); ...; vec(wt{P})].
[pars_vdata, b0_vdata] = SVDRegress_S_Vdata(XX, XY, Yn, allstim, T, n, ...
    rank_list(k_factors, :), ridgeparam, opts);

% Greedy searches, as mTDRdemo.m: the reference objective from each start,
% then the corrected objective from all ones.
SVDRegPars = @(r) SVDRegressB(XX, XY, [T n], r, ridgeparam, opts);
histdir = tempname;
mkdir(histdir);
log_ranks = zeros(0, P);
log_values = zeros(0, 1);
searches = cell(1, size(starts, 1));
for s = 1:size(starts, 1)
    log_ranks = zeros(0, P);
    log_values = zeros(0, 1);
    histfile = fullfile(histdir, sprintf('RankEst_SVD_%d', s));
    [rest, rhist, funhist, parhist] = EstRankGreedily(@logged_reference_aic, ...
        SVDRegPars, [starts(s, :)'; maxrank], maxrank, [], histfile);
    searches{s} = struct('start', starts(s, :), 'rest', rest, 'rhist', rhist, ...
        'funhist', funhist, 'parhist', cat(4, parhist{:}), ...
        'calls_ranks', log_ranks, 'calls_values', log_values);
end
log_ranks = zeros(0, P);
log_values = zeros(0, 1);
histfile = fullfile(histdir, 'RankEst_SVD_corrected');
CorrectedPars = @(r) centred_truncation(Yn, allstim, n, T, Pb, r);
[rest, rhist, funhist] = EstRankGreedily(@logged_corrected_aic, CorrectedPars, ...
    [ones(Pb, 1); maxrank], maxrank, [], histfile);
corrected_search = struct('rest', rest, 'rhist', rhist, 'funhist', funhist, ...
    'calls_ranks', log_ranks, 'calls_values', log_values);
rmdir(histdir, 's');

out.n = n; out.T = T; out.N = N; out.maxrank = maxrank;
out.XX = XX; out.XY = XY;
out.Ai = Ai; out.zzi = zzi; out.ni = ni; out.Xzetai0 = Xzetai0;
out.xbari = xbari; out.Ybar = Ybar;
out.b_test = b_test; out.zzi_c = zzi_c; out.Xzetai_c = Xzetai_c;
out.w0 = w0;
out.rank_list = rank_list; out.wsvd_all = wsvd_all; out.aic_ref = aic_ref;
out.rss_aligned = rss_aligned; out.loglik_corrected = loglik_corrected;
out.nparam_textbook = nparam_textbook; out.nparam_reference = nparam_reference;
out.aic_corrected = aic_corrected;
out.k_factors = k_factors;
out.pars_vdata = pars_vdata; out.b0_vdata = b0_vdata;
out.searches = searches;
out.corrected_search = corrected_search;

    % Nested so that the call log is shared with run_dataset's workspace.
    function aic = logged_reference_aic(pars, r)
        aic = SVDRegB_AIC(Yn, allstim, pars, r);
        log_ranks(end + 1, :) = r(:)';
        log_values(end + 1, 1) = aic;
    end

    function aic = logged_corrected_aic(pars, r)
        [~, ll, kk] = corrected_score(pars, r, Yn, allstim, n, T, Pb, maxrank);
        aic = 2 * kk - 2 * ll;
        log_ranks(end + 1, :) = r(:)';
        log_values(end + 1, 1) = aic;
    end
end


function Btr = centred_truncation(Yn, allstim, n, T, Pb, r)
% The port's task blocks, written independently of the port: per neuron,
% least squares of its centred responses on its centred observed regressors
% (the constant column dropped), then the rank-r(p) truncation of each n x T
% block B_p by its SVD. Returns T x n x Pb, as wsvd without the constant term.
Bfull = zeros(n, T, Pb);
for ii = 1:n
    Xi = allstim{ii}(:, 1:Pb);                 % n_i x Pb
    Yi = Yn{ii}';                              % n_i x T
    assert(rank(Xi - mean(Xi, 1)) == Pb, 'a rank-deficient neuron');
    Bi = (Xi - mean(Xi, 1)) \ (Yi - mean(Yi, 1));   % Pb x T
    Bfull(ii, :, :) = reshape(Bi', [1 T Pb]);
end
Btr = zeros(T, n, Pb);
for p = 1:Pb
    [U, S, V] = svd(Bfull(:, :, p), 'econ');
    k = r(p);
    Btr(:, :, p) = (U(:, 1:k) * S(1:k, 1:k) * V(:, 1:k)')';
end
end


function [rss, loglik, nparam] = corrected_score(Btr, r, Yn, allstim, n, T, Pb, maxrank)
% The port's (M19b), (M21a) and (M21b), written independently of the port:
% the intercept refitted given the truncated task blocks Btr, each observed
% entry against its own prediction, the fully normalised plug-in Gaussian
% log-likelihood, and the textbook parameter count with the constant term at
% full rank maxrank.
rss = zeros(n, 1);
nobs = zeros(n, 1);
for ii = 1:n
    Bi = reshape(Btr(:, ii, :), T, Pb);        % T x Pb, column p = B_p(i, :)'
    Xi = allstim{ii}(:, 1:Pb);                 % n_i x Pb
    bi = mean(Yn{ii}, 2) - Bi * mean(Xi, 1)';  % T x 1, given Btr
    resid = Yn{ii} - Bi * Xi' - bi;            % T x n_i
    rss(ii) = sum(resid(:) .^ 2);
    nobs(ii) = size(allstim{ii}, 1);
end
m = nobs * T;
loglik = -0.5 * sum(m .* (log(rss ./ m) + 1 + log(2 * pi)));
rb = r(1:Pb);
nparam = sum(rb .* (n + T - rb)) + n + maxrank * (n + T - maxrank);
end
