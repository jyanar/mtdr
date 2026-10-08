function make_paper_fixture(refdir, outfile, run_mmle_search)
% MAKE_PAPER_FIXTURE  Paper-scale parity fixture.
%
%   make_paper_fixture(refdir, outfile)
%   make_paper_fixture(refdir, outfile, run_mmle_search)
%
% A dataset at the scale of Aoi, Mante & Pillow (2020): n = 800 neurons in
% 100 sessions of 8, T = 15 bins, N = 1000 trial slots, P = 6 correlated
% task regressors plus the constant, true ranks [5 4 3 4 2 2] (rtot = 20),
% heterogeneous noise precisions, and non-simultaneous recordings (each
% session records a window of 100-300 consecutive trials, with 5 % random
% trial loss, so every neuron sees 91-288 trials).
%
% The responses are not stored. They are built from integer random streams
% by operations that IEEE arithmetic rounds identically in MATLAB and NumPy,
% in a fixed order, and rounded to the grid 2^-16; tests/paper_data.py
% builds the same array from the same specification (its docstring states
% it) and checks the two exact integer checksums recorded here. Nothing in
% this file depends on MATLAB's rng.
%
% The reference is then run on the data exactly as mTDRdemo.m runs it:
%
%   stats    MkSuffStats_BilinReg_Sims, MkSuffStatsBTDR_IncompObs_uneqvar_S_fast
%            and the xbari/Ybar loop (a few per-neuron values are saved);
%   svd      EstRankGreedily with SVDRegB_AIC from all ones, every
%            objective evaluation logged;
%   fits     at fixed rank vectors (the truth, one under-ranked, one
%            over-ranked): SVDRegress_S_Vdata -> ECMEregress_wrapper ->
%            ECMEtdr('converge', 1) -> Estpars_CoordAscent_lambi_S_b, with the
%            start's precisions and bases, the post-ECME estimate, the
%            coordinate-ascent trace (the full NLL after every outer
%            iteration, minFunc's and fminunc's iteration counts and exit
%            flags) and the final estimate, NLL and AIC; and the time taken;
%   posterior  EBpost_W_uneqvar's posterior mean of the weights at the
%            true-rank fit's final estimate;
%   mmle     optionally (RUN_MMLE_SEARCH, default false; about an hour),
%            EstRankGreedily with the MMLE objective from the true ranks, the
%            reference's count, every evaluation logged.
%
%   refdir           path to the mTDRdemo directory (read-only reference)
%   outfile          .mat file to write (tests/fixtures/paper.mat)
%
% Run from the repository root, with MATLAB's default computation-thread
% setting (see `matlab_threads` below), with
%   matlab -batch "addpath('tests/fixtures/matlab'); make_paper_fixture('path/to/mTDRdemo', 'tests/fixtures/paper.mat', true)"

if nargin < 3
    run_mmle_search = false;
end
addpath(fullfile(refdir, 'functionFiles'), fullfile(refdir, 'functionFiles', 'tools_kron'));
addpath(genpath(fullfile(refdir, 'minFunc_2012')));
mmx_available = exist('mmx_mkl_single'); %#ok<EXIST>

% ---- the data -----------------------------------------------------------------
[Z, Xb, hk, truth, checksums] = paper_data();
[n, T, N] = size(Z);
P = size(Xb, 2) + 1;
Pb = P - 1;
X = [Xb ones(N, 1)];
maxrank = min(n, T);
ridgeparam = 0;
opts.MaxIter = 500;
opts.Display = 'off';
g = 0;

% ---- statistics, as mTDRdemo.m --------------------------------------------------
[XX, XY, Yn, allstim] = MkSuffStats_BilinReg_Sims(Z, X, hk);
[~, Ai, zzi, ni, Xi, Xzetai0, zetai0] = MkSuffStatsBTDR_IncompObs_uneqvar_S_fast(Xb, Z, hk);
xbari = zeros(n, Pb);
Ybar = zeros(T, n);
for ii = 1:n
    xbari(ii, :) = mean(Xi{ii}, 1);
    Ybar(:, ii) = squeeze(sum(Z(ii, :, :), 3)) / ni(ii);
end
stats = struct('ni', ni, 'zzi', zzi, 'xbari', xbari, 'Ybar_sum', sum(Ybar, 2), ...
    'Ai_first', Ai(:, :, 1), 'Xzetai0_first', Xzetai0(:, 1));

% ---- the SVD rank search, as mTDRdemo.m -----------------------------------------
histdir = tempname;
mkdir(histdir);
SVDRegPars = @(r) SVDRegressB(XX, XY, [T n], r, ridgeparam, opts);
log_ranks = zeros(0, P);
log_values = zeros(0, 1);
t0 = tic;
[rEstSVD, rhist, funhist] = EstRankGreedily(@logged_svd_aic, SVDRegPars, ...
    [ones(Pb, 1); maxrank], maxrank, [], fullfile(histdir, 'RankEst_SVD'));
svd = struct('rest', rEstSVD, 'rhist', rhist, 'funhist', funhist, ...
    'calls_ranks', log_ranks, 'calls_values', log_values, 'seconds', toc(t0));
fprintf('SVD search: [%s] in %.1f s\n', num2str(rEstSVD(:)'), svd.seconds);

% ---- the MMLE pipeline at fixed ranks -------------------------------------------
svdregress = @(r) SVDRegress_S_Vdata(XX, XY, Yn, allstim, T, n, r, ridgeparam, opts);
EMregressfun = @(r, pars0) ECMEtdr('converge', 1e-0, pars0, Ai, Xi, r, ni, zetai0, xbari, Ybar, Xzetai0);
EMparsfun = @(r) ECMEregress_wrapper(r, svdregress, EMregressfun, Ybar);
MMLEParfun = @(lamb0, s0, bhat0, r) Estpars_CoordAscent_lambi_S_b(lamb0, s0, bhat0, r, Ai, Xi, zetai0, ni, xbari, Ybar, Xzetai0);
MMLE_EMregressfun = @(r) MMLE_CoordAscentWrapper(EMparsfun, MMLEParfun, r, n, T);
rank_list = [5 4 3 4 2 2; 4 4 3 4 2 2; 5 4 3 4 2 3];
assert(isequal(rank_list(1, :), truth.rP));
fits = cell(1, size(rank_list, 1));
for k = 1:size(rank_list, 1)
    r = rank_list(k, :);
    L = n + sum(r) * T;
    fit = struct('r', r);
    t0 = tic;
    pars_svd = svdregress([r maxrank]);
    pars0 = [pars_svd(1:L); vec(Ybar)];
    fit.start_lam = pars0(1:n);
    fit.start_s = pars0(n+1:L);
    [ecme_pars, ~, ecme_parerr, ecme_nll] = ECMEtdr('converge', 1e-0, pars0, Ai, Xi, r, ni, ...
        zetai0, xbari, Ybar, Xzetai0);
    assert(isequal(ecme_pars, EMparsfun(r)), 'ECMEregress_wrapper differs');
    fit.ecme_pars = ecme_pars;
    fit.ecme_n_sweeps = numel(ecme_parerr);
    fit.ecme_nll = ecme_nll;
    lamb0 = ecme_pars(1:n);
    s0 = ecme_pars(n+1:L);
    bhat0 = reshape(ecme_pars(L+1:end), T, n);
    [lambhat, shat, bhat, fit.ca] = ca_traced(lamb0, s0, bhat0, r, Ai, Xi, zetai0, ni, ...
        xbari, Ybar, Xzetai0);
    pars_final = [lambhat; shat; vec(bhat)];
    fit.seconds = toc(t0);
    % The reference coordinate ascent itself, so that the traced copy is pinned.
    [l2, s2, ~, ~, b2] = MMLEParfun(lamb0, s0, bhat0, r);
    assert(isequal(l2, lambhat) && isequal(s2, shat) && isequal(b2, bhat), ...
        'ca_traced differs from Estpars_CoordAscent_lambi_S_b');
    fit.pars_final = pars_final;
    [~, zzi_f, Xzetai_f] = ECMEsuffstat(zetai0, Xi, bhat);
    fit.nll_final = neglogLikBTDR_IncompObs_uneqvar_S_nllonly(pars_final(1:L), Ai, Xzetai_f, zzi_f, r, ni, g);
    fit.aic = BTDR_AIC_S_lamb_b_wrapper(pars_final, Ai, Xi, zetai0, r, ni, g);
    fprintf('r = [%s]: %d ECME sweeps, %d CA iterations, NLL %.6f, %.1f s\n', num2str(r), ...
        fit.ecme_n_sweeps, numel(fit.ca.nll), fit.nll_final, fit.seconds);
    fits{k} = fit;
    if k == 1
        % The posterior mean of the weights at the true-rank final estimate
        % (EBpost_W_uneqvar, as demoLearning.m forms Bhat), an independent
        % oracle for the port's posterior at this scale. Kept out of `fits`
        % so that every earlier array is unchanged.
        Shat = mat2cell(reshape(shat, T, sum(r))', r, T);
        Shatblock = blkdiag(Shat{:});
        posterior = struct('r', r, 'Wt', EBpost_W_uneqvar(Shatblock, lambhat, Ai, Xzetai_f, sum(r)));
    end
end

% ---- optionally, the MMLE rank search from the true ranks ------------------------
mmle = struct();
if run_mmle_search
    log_ranks = zeros(0, Pb);
    log_values = zeros(0, 1);
    t0 = tic;
    [rest, rhist, funhist] = EstRankGreedily(@logged_mmle_aic, MMLE_EMregressfun, ...
        truth.rP, maxrank, [], fullfile(histdir, 'RankEstMMLE'));
    mmle = struct('rest0', truth.rP, 'rest', rest, 'rhist', rhist, 'funhist', funhist, ...
        'calls_ranks', log_ranks, 'calls_values', log_values, 'seconds', toc(t0));
    fprintf('MMLE search from [%s]: [%s] in %.1f s\n', num2str(truth.rP), num2str(rest), mmle.seconds);
end
rmdir(histdir, 's');

% ---- optimiser settings, as the reference resolves them --------------------------
mf = struct('display', 'none');
[~, ~, ~, ~, mf_maxFunEvals, mf_maxIter, mf_optTol, mf_progTol] = minFunc_processInputOptions(mf);
minfunc_options = struct('maxFunEvals', mf_maxFunEvals, 'maxIter', mf_maxIter, ...
    'optTol', mf_optTol, 'progTol', mf_progTol);

rP = truth.rP;
d = truth.d;
matlab_version = version;
% The computation-thread setting the fits ran with. The data, statistics,
% starts and rank paths do not depend on it, but the fits' iterates do: the
% committed fixture was made with MATLAB's default (one thread per physical
% core, 8 on the machine that made it), and a re-run with that setting on that
% machine reproduces every array except the fields named `seconds` (and the
% 128-byte MAT header's creation time). With -singleCompThread the optimiser
% traces and final parameter vectors change at the 1e-4 level and the final
% NLLs at the 1e-12 level.
matlab_threads = maxNumCompThreads;
save(outfile, 'n', 'T', 'N', 'rP', 'd', 'checksums', 'stats', 'svd', 'rank_list', 'fits', ...
    'mmle', 'mmx_available', 'minfunc_options', 'matlab_version', 'matlab_threads', 'posterior');
fprintf('wrote %s\n', outfile);

    % Nested so that the call logs are shared with the workspace above.
    function aic = logged_svd_aic(pars, r)
        aic = SVDRegB_AIC(Yn, allstim, pars, r);
        log_ranks(end + 1, :) = r(:)';
        log_values(end + 1, 1) = aic;
    end

    function aic = logged_mmle_aic(pars, r)
        aic = BTDR_AIC_S_lamb_b_wrapper(pars, Ai, Xi, zetai0, r, ni, g);
        log_ranks(end + 1, :) = r(:)';
        log_values(end + 1, 1) = aic;
        fprintf('  MMLE candidate [%s]: AIC %.6f\n', num2str(r(:)'), aic);
    end
end


% =============================================================================
% The data. tests/paper_data.py implements the same specification; keep the
% two in step (the checksums catch any difference).
% =============================================================================

function [Z, X, hk, truth, checksums] = paper_data()
n = 800; T = 15; N = 1000; nsess = 100;
rP = [5 4 3 4 2 2];
rho = [3 3 1 1 2 2];
rtot = sum(rP);
P = numel(rP);

% Regressors: four uniforms per trial (stream 1001).
u = lcg(1001, 4 * N);
levels = [-0.5 -0.15 -0.05 0.05 0.15 0.5];
X = zeros(N, P);
for k = 1:N
    a = u(4*k - 3); c = u(4*k - 2); cx = u(4*k - 1); ch = u(4*k);
    x1 = levels(floor(6 * a) + 1);
    x2 = levels(floor(6 * c) + 1);
    if cx < 0.5, ctx = 1; else, ctx = -1; end
    if ctx == 1, rel = x1; else, rel = x2; end
    if rel + 0.6 * (ch - 0.5) > 0, choice = 1; else, choice = -1; end
    X(k, :) = [x1, x2, choice, ctx, x1 * ctx, x2 * ctx];
end

% Sessions (stream 1002) and 5 % trial loss (stream 1003, index (i-1)*N + k).
u = lcg(1002, 2 * nsess);
drop = lcg(1003, n * N);
hk = zeros(N, n);
per = n / nsess;
for s = 1:nsess
    L = 100 + floor(201 * u(2*s - 1));
    st = floor((N - L + 1) * u(2*s));
    win = false(N, 1);
    win(st+1 : st+L) = true;
    for i = (s-1)*per + 1 : s*per
        hk(:, i) = win & (drop((i-1)*N + 1 : i*N) >= 0.05);
    end
end

% Truth: weights (1004), bases (1005), intercept (1008, 1009), precisions (1006).
G = reshape(normals(1004, n * rtot), n, rtot);
Sb = smooth_bases(1005, T, rtot);
c = [0 cumsum(rP)];
B = cell(1, P);
W = cell(1, P);
S = cell(1, P);
for p = 1:P
    W{p} = rho(p) * G(:, c(p)+1 : c(p+1));
    S{p} = Sb(:, c(p)+1 : c(p+1));
    B{p} = outer_sum(G(:, c(p)+1 : c(p+1)), S{p}, rho(p));
end
W0 = reshape(normals(1008, n * 3), n, 3);
S0 = smooth_bases(1009, T, 3);
b = 2 + outer_sum(W0, S0, 1);
ud = lcg(1006, n);
d = 0.1 + 2.4 * ud .* ud;

% Responses (noise: stream 1007, time fastest, then trials, then neurons).
ni = sum(hk, 1);
e = normals(1007, sum(ni) * T);
Z = zeros(n, T, N);
pos = 0;
c1 = 0;
c2 = 0;
for i = 1:n
    kk = find(hk(:, i));
    m = numel(kk) * T;
    noise = reshape(e(pos+1 : pos+m), T, numel(kk));
    sig = repmat(b(i, :)', 1, numel(kk));
    for p = 1:P
        sig = sig + B{p}(i, :)' * X(kk, p)';
    end
    v = sig + noise / sqrt(d(i));
    zint = floor(v * 65536 + 0.5);
    w = mod((pos+1 : pos+m)', 997) + 1;
    c1 = c1 + sum(zint(:));
    c2 = c2 + sum(zint(:) .* w);
    Z(i, :, kk) = reshape(zint / 65536, 1, T, numel(kk));
    pos = pos + m;
end
assert(abs(c2) < 2^53, 'checksum not exact');
% The checksums are those of the completed array: neuron by neuron,
% observed trials in ascending order, bins fastest; the w-th value so visited
% (from 1) has weight mod(w, 997) + 1. tests/paper_data.py recomputes them
% from the array it returns in the same traversal. They equal the draws'
% running sums above, which pins the traversal to the generation order.
checksums = array_checksums(Z, hk);
assert(isequal(checksums, [c1 c2]), 'the completed array is not the drawn one');
truth = struct('rP', rP, 'd', d, 'W', {W}, 'S', {S}, 'b', b);
end

function c = array_checksums(Z, hk)
% Integer checksums of the observed entries of Z (n x T x N), see above.
[n, T, ~] = size(Z);
c = [0 0];
pos = 0;
for i = 1:n
    kk = find(hk(:, i));
    zint = reshape(Z(i, :, kk), T, numel(kk)) * 65536;
    assert(isequal(zint, round(zint)), 'Z is not on the grid');
    m = numel(zint);
    w = mod((pos+1 : pos+m)', 997) + 1;
    c = c + [sum(zint(:)) sum(zint(:) .* w)];
    pos = pos + m;
end
assert(abs(c(2)) < 2^53, 'checksum not exact');
end

function u = lcg(seed, count)
% Park-Miller minimal standard: x <- 16807 x mod (2^31 - 1), u = x / (2^31 - 1).
m = 2147483647;
x = seed;
u = zeros(count, 1);
for j = 1:count
    x = mod(16807 * x, m);
    u(j) = x / m;
end
end

function z = normals(seed, count)
% Irwin-Hall: twelve consecutive uniforms added left to right, minus 6.
U = reshape(lcg(seed, 12 * count), 12, count);
z = U(1, :);
for r = 2:12
    z = z + U(r, :);
end
z = (z - 6)';
end

function S = smooth_bases(seed, T, r)
% Normals smoothed twice by the binomial filter (1, 4, 6, 4, 1) / 16.
z = reshape(normals(seed, (T + 8) * r), T + 8, r);
for pass = 1:2
    z = (z(1:end-4, :) + 4 * z(2:end-3, :) + 6 * z(3:end-2, :) + 4 * z(4:end-1, :) + z(5:end, :)) / 16;
end
S = 2.25 * z;
end

function out = outer_sum(G, S, scale)
% sum_j (scale * G(:, j)) * S(:, j)', component by component in order.
out = zeros(size(G, 1), size(S, 1));
for j = 1:size(G, 2)
    out = out + (scale * G(:, j)) * S(:, j)';
end
end


% =============================================================================
% The coordinate ascent with diagnostics (as in make_mmle_fixture.m).
%
% Attribution: ca_traced below reproduces code from
% functionFiles/Estpars_CoordAscent_lambi_S_b.m of mTDRdemo
% (https://github.com/pillowlab/mTDRdemo) by M. C. Aoi and J. W. Pillow. That code
% is theirs; it is included here only to record diagnostics the reference
% function does not return.
% =============================================================================

function [lambhat, shat, bhat, tr] = ca_traced(lambhat0, shat0, bhat0, r, Ai, Xi, zetai, ni, xbari, Ybar, Xzetai0)
% Estpars_CoordAscent_lambi_S_b.m (mTDRdemo, Aoi & Pillow) step for step, with
% minFunc's and fminunc's diagnostic outputs requested (they do not change the
% iterates) and the full marginal NLL recorded after every outer iteration. The
% final estimate is asserted equal to the reference function's by the caller.
P = length(r);
rtot = sum(r);
[T, n] = size(Ybar);
TP = T * P;
maxsteps = 10;
stopcrit = 1e-4;
k = 1;
tr = struct('nll', [], 'parerr', [], 'minfunc_exitflag', [], 'minfunc_iterations', [], ...
    'fminunc_exitflag', [], 'fminunc_iterations', []);
while true
    [Ri, zzi, ~] = ECMEsuffstat(zetai, Xi, bhat0);
    options.display = 'none';
    [shat, ~, flags, outs] = minFunc(@(s) neglogLikBTDR_IncompObs_uneqvar_Sonly(s, lambhat0, Ai, Ri, zzi, r, ni, 0), shat0, options);
    Shat = mat2cell(reshape(shat, T, rtot)', r, T);
    Shatblock = blkdiag(Shat{:});
    Hesspat = sparse(n, n);
    Hesspat(logical(eye(size(Hesspat)))) = ones(numel(lambhat0), 1);
    fminopts = optimset('gradobj', 'on', 'display', 'notify', 'Hessian', 'off', ...
        'HessPattern', Hesspat, 'algorithm', 'trust-region-reflective', 'maxfunevals', 1000, 'maxiter', 1000);
    [lambhat, ~, flagl, outl] = fminunc(@(lamb) neglogLikBTDR_IncompObs_uneqvar_lambonly(lamb, Shatblock, Ai, Ri, zzi, r, ni, 0), lambhat0, fminopts);
    S2 = reshape(Shatblock, [], P);
    AiIS = permute(reshape(S2 * reshape(Ai, P, P*n), rtot, TP, n), [2 1 3]);
    SAiIS = reshape(Shatblock * reshape(AiIS, TP, rtot*n), rtot, rtot, n);
    Ci = bsxfun(@plus, bsxfun(@times, SAiIS, permute(lambhat, [3 2 1])), eye(rtot));
    bhat = MMLE_b(Ci, Shatblock, lambhat, ni, xbari, Ybar, Xzetai0, r);
    newpars = [lambhat; shat; vec(bhat)];
    oldpars = [lambhat0; shat0; vec(bhat0)];
    parerr = max((newpars - oldpars).^2 ./ oldpars.^2);
    [~, zzn, Xzn] = ECMEsuffstat(zetai, Xi, bhat);
    tr.nll(end+1, 1) = neglogLikBTDR_IncompObs_uneqvar_S_nllonly([lambhat; shat], Ai, Xzn, zzn, r, ni, 0);
    tr.parerr(end+1, 1) = parerr;
    tr.minfunc_exitflag(end+1, 1) = flags;
    tr.minfunc_iterations(end+1, 1) = outs.iterations;
    tr.fminunc_exitflag(end+1, 1) = flagl;
    tr.fminunc_iterations(end+1, 1) = outl.iterations;
    if parerr < stopcrit || k >= maxsteps
        break
    end
    lambhat0 = lambhat;
    shat0 = shat;
    bhat0 = bhat;
    k = k + 1;
end
end
