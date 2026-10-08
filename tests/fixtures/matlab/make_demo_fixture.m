function make_demo_fixture(refdir, outfile)
% MAKE_DEMO_FIXTURE  Parity fixture: the two shipped demos end to end.
%
%   make_demo_fixture(refdir, outfile)
%
% Copies the reference directory to a scratch folder (the demo scripts save
% their histories under EstimatedPars/, so the reference itself is never
% written) and runs, unmodified and at rng(0):
%
%   mTDRdemo.m      the SVD rank search (SVDRegB_AIC from all ones), then the
%                   MMLE rank search (BTDR_AIC_S_lamb_b_wrapper) from the SVD
%                   search's ranks; and
%   demoLearning.m  the MMLE fit at the true ranks rP.
%
% Both scripts make the same random draws in the same order (randi for rP,
% SimWeights, exprnd, SimConditions, SimPopData, the binornd mask loop), so at
% the same seed they see the same data; that is asserted. The fixture saves
% rP with demoLearning.m's fit, which the shipped LearnDemoMMLE.mat lacks.
% The shipped file is not this run's output (its ranks sum to 12, as
% rP = [5 6 1] does, but its precisions and intercept differ); the
% differences are recorded (learn.shipped_*), not asserted.
%
% The scripts do not return their intermediates, so the estimation is then
% replayed with the same reference functions, logging every rank-search
% objective evaluation and recording the demoLearning fit's start, post-ECME
% estimate, coordinate-ascent trace and posterior; the replay is asserted
% equal (isequal) to the scripts' own histories and estimates.
%
% Saved: the draws and data (Z, X, hk, d, rP, Wtrue, Strue, BB; Z as the
% double values of its observed entries, Zobs, in MATLAB's column-major order
% of Z(i, t, k) over hk(k, i) = 1), the reference statistics, both searches
% (rhist, FunHist, parhist, and every call's ranks and value), and the
% demoLearning fit. The scripts' responses are used exactly: nothing is
% rounded.
%
%   refdir   path to the mTDRdemo directory (read-only reference)
%   outfile  .mat file to write (tests/fixtures/demo.mat)
%
% Run from the repository root with
%   matlab -batch "addpath('tests/fixtures/matlab'); make_demo_fixture('path/to/mTDRdemo', 'tests/fixtures/demo.mat')"

seed = 0;
scratch = tempname;
copyfile(refdir, scratch);
shipped_learn = load(fullfile(refdir, 'EstimatedPars', 'LearnDemoMMLE.mat'));

% ---- the scripts themselves -------------------------------------------------
demo = run_script(fullfile(scratch, 'mTDRdemo.m'), seed);
demo_svd = load(fullfile(scratch, 'EstimatedPars', 'RankEstDemo_SVD.mat'));
demo_mmle = load(fullfile(scratch, 'EstimatedPars', 'RankEstDemoMMLE.mat'));
learn = run_script(fullfile(scratch, 'demoLearning.m'), seed);
for name = {'rP', 'Wtrue', 'Strue', 'BB', 'd', 'X', 'Y', 'hk', 'Z'}
    assert(isequal(demo.(name{1}), learn.(name{1})), ...
        'the two demos drew different %s at the same seed', name{1});
end
fprintf('mTDRdemo.m: SVD search -> [%s], MMLE search -> [%s]; true ranks [%s]\n', ...
    num2str(demo.rEstSVD(:)'), num2str(demo.rEstMMLE_EM(:)'), num2str(demo.rP));

% Paths for the replay: the scratch copy's functions (identical to refdir's).
addpath(fullfile(scratch, 'functionFiles'), fullfile(scratch, 'functionFiles', 'tools_kron'));
addpath(genpath(fullfile(scratch, 'minFunc_2012')));
mmx_available = exist('mmx_mkl_single'); %#ok<EXIST>

Z = demo.Z;
X = demo.X;
hk = demo.hk;
[n, T, N] = size(Z);
P = size(X, 2);
Pb = P - 1;
maxrank = min(n, T);
ridgeparam = 0;
opts.MaxIter = 500;
opts.Display = 'off';
g = 0;

% ---- statistics, as mTDRdemo.m ----------------------------------------------
[XX, XY, Yn, allstim] = MkSuffStats_BilinReg_Sims(Z, X, hk);
Xb = X(:, 1:Pb);
[~, Ai, zzi, ni, Xi, Xzetai0, zetai0] = MkSuffStatsBTDR_IncompObs_uneqvar_S_fast(Xb, Z, hk);
xbari = zeros(n, Pb);
Ybar = zeros(T, n);
for ii = 1:n
    xbari(ii, :) = mean(Xi{ii}, 1);
    Ybar(:, ii) = squeeze(sum(Z(ii, :, :), 3)) / ni(ii);
end

% ---- replay of mTDRdemo.m's two searches, every evaluation logged ----------
histdir = tempname;
mkdir(histdir);
SVDRegPars = @(r) SVDRegressB(XX, XY, [T n], r, ridgeparam, opts);
log_ranks = zeros(0, P);
log_values = zeros(0, 1);
[rEstSVD, rhist, funhist, parhist] = EstRankGreedily(@logged_svd_aic, SVDRegPars, ...
    [ones(Pb, 1); maxrank], maxrank, [], fullfile(histdir, 'RankEstDemo_SVD'));
assert(isequal(rEstSVD, demo.rEstSVD) && isequal(rhist, demo_svd.rhist) && ...
    isequal(funhist, demo_svd.FunHist) && isequal(parhist, demo_svd.parhist), ...
    'the SVD search replay differs from mTDRdemo.m');
svd = struct('rest', rEstSVD, 'rhist', rhist, 'funhist', funhist, ...
    'parhist', cat(4, parhist{:}), 'calls_ranks', log_ranks, 'calls_values', log_values);

svdregress = @(r) SVDRegress_S_Vdata(XX, XY, Yn, allstim, T, n, r, ridgeparam, opts);
EMregressfun = @(r, pars0) ECMEtdr('converge', 1e-0, pars0, Ai, Xi, r, ni, zetai0, xbari, Ybar, Xzetai0);
EMparsfun = @(r) ECMEregress_wrapper(r, svdregress, EMregressfun, Ybar);
MMLEParfun = @(lamb0, s0, bhat0, r) Estpars_CoordAscent_lambi_S_b(lamb0, s0, bhat0, r, Ai, Xi, zetai0, ni, xbari, Ybar, Xzetai0);
MMLE_EMregressfun = @(r) MMLE_CoordAscentWrapper(EMparsfun, MMLEParfun, r, n, T);
rest0 = rEstSVD(1:Pb);
log_ranks = zeros(0, Pb);
log_values = zeros(0, 1);
[rest, rhist, funhist, parhist] = EstRankGreedily(@logged_mmle_aic, MMLE_EMregressfun, ...
    rest0, maxrank, [], fullfile(histdir, 'RankEstDemoMMLE'));
assert(isequal(rest, demo.rEstMMLE_EM) && isequal(rhist, demo_mmle.rhist) && ...
    isequal(funhist, demo_mmle.FunHist) && isequal(parhist, demo_mmle.parhist), ...
    'the MMLE search replay differs from mTDRdemo.m');
mmle = struct('rest0', rest0, 'rest', rest, 'rhist', rhist, 'funhist', funhist, ...
    'calls_ranks', log_ranks, 'calls_values', log_values);
mmle.parhist = parhist;

% ---- replay of demoLearning.m's fit, with its intermediates ----------------
% demoLearning.m builds the MMLE statistics inline with the formulas of
% MkSuffStatsBTDR_IncompObs_uneqvar_S_fast; the statistics above are those,
% so its fit is MMLE_CoordAscentWrapper(..., rP).
r = learn.rP;
L = n + sum(r) * T;
fit = struct('r', r);
pars_svd = svdregress([r maxrank]);
pars0 = [pars_svd(1:L); vec(Ybar)];
fit.pars0 = pars0;
[fit.ecme_pars, ~, ecme_parerr, fit.ecme_nll] = ECMEtdr('converge', 1e-0, pars0, Ai, Xi, r, ni, ...
    zetai0, xbari, Ybar, Xzetai0);
assert(isequal(fit.ecme_pars, EMparsfun(r)), 'ECMEregress_wrapper differs');
fit.ecme_n_sweeps = numel(ecme_parerr);
lamb0 = fit.ecme_pars(1:n);
s0 = fit.ecme_pars(n+1:L);
bhat0 = reshape(fit.ecme_pars(L+1:end), T, n);
[lambhat, shat, Shat, Shatblock, bhat] = MMLEParfun(lamb0, s0, bhat0, r);
pars_final = [lambhat; shat; vec(bhat)];
assert(isequal(pars_final, learn.parhist), 'the demoLearning.m replay differs from the script');
fit.pars_final = pars_final;
fit.Shat = Shat;
[~, zzi_f, Xzetai_f] = ECMEsuffstat(zetai0, Xi, bhat);
fit.Wt = EBpost_W_uneqvar(Shatblock, lambhat, Ai, Xzetai_f, sum(r));
for p = 1:Pb
    assert(isequal(learn.Bhat{p}, fit.Wt(sum(r(1:p-1))+1:sum(r(1:p)), :)' * Shat{p}), ...
        'Bhat of demoLearning.m is not the posterior mean times Shat');
end
fit.Bhat = learn.Bhat(1:Pb);       % demoLearning.m's own (centred, the correct call)
fit.nll_final = neglogLikBTDR_IncompObs_uneqvar_S_nllonly(pars_final(1:L), Ai, Xzetai_f, zzi_f, r, ni, g);
fit.aic = BTDR_AIC_S_lamb_b_wrapper(pars_final, Ai, Xi, zetai0, r, ni, g);
% Against the shipped LearnDemoMMLE.mat: the precisions and the intercept, and
% the bases up to the sign of each column (which the marginal likelihood does
% not see, docs/model.md § 10.1).
ship = shipped_learn.parhist(:);
assert(numel(ship) == numel(pars_final), 'the shipped LearnDemoMMLE.mat has another length');
fit.shipped_max_abs_diff = max(abs(ship - pars_final));
fit.shipped_diff_lambda = max(abs(ship(1:n) - pars_final(1:n)));
fit.shipped_diff_b = max(abs(ship(L+1:end) - pars_final(L+1:end)));
s_ship = reshape(ship(n+1:L), T, []);
s_ours = reshape(pars_final(n+1:L), T, []);
flips = sign(sum(s_ship .* s_ours, 1));
fit.shipped_diff_S_signed = max(max(abs(s_ship - s_ours .* flips)));
fit.shipped_sign_flips = sum(flips < 0);
fprintf(['demoLearning.m at rng(%d): rP = [%s]; against the shipped LearnDemoMMLE.mat: ' ...
    'max |difference| %g; lambda %g, b %g, S up to %d column signs %g\n'], seed, num2str(r), ...
    fit.shipped_max_abs_diff, fit.shipped_diff_lambda, fit.shipped_diff_b, ...
    fit.shipped_sign_flips, fit.shipped_diff_S_signed);
rmdir(histdir, 's');
rmpath(fullfile(scratch, 'functionFiles'), fullfile(scratch, 'functionFiles', 'tools_kron'));
rmpath(genpath(fullfile(scratch, 'minFunc_2012')));
clear mex   % minFunc's MEX files stay locked while loaded
fileattrib(scratch, '+w', '', 's');
[removed, msg] = rmdir(scratch, 's');
if ~removed
    warning('make_demo_fixture:scratch', 'could not remove %s: %s', scratch, msg);
end

% ---- save --------------------------------------------------------------------
mask3 = repmat(reshape(logical(hk'), n, 1, N), [1 T 1]);
assert(all(Z(~mask3) == 0), 'unobserved entries of Z are not zero');
Zobs = Z(mask3);
rP = learn.rP;
d = learn.d;
X = int8(X);
hk = uint8(hk);
Wtrue = learn.Wtrue;
Strue = learn.Strue;
BB = learn.BB;
stats = struct('Ai', Ai, 'zzi', zzi, 'ni', ni, 'Xzetai0', Xzetai0, 'xbari', xbari, 'Ybar', Ybar);
learn = fit;
matlab_version = version;
% Re-running reproduces every array exactly; the 128-byte MAT header records
% the creation time, so the file differs there only.
save(outfile, 'seed', 'n', 'T', 'N', 'maxrank', 'rP', 'd', 'X', 'hk', 'Zobs', ...
    'Wtrue', 'Strue', 'BB', 'stats', 'svd', 'mmle', 'learn', 'mmx_available', 'matlab_version');
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
    end
end


function ws = run_script(path, seed)
% Run a demo script unmodified at rng(seed) and return its workspace. The
% scripts start with `clear all`, which clears this function's arguments
% too; `clear all` does not reset the random number generator.
rng(seed);
run(path);
close all force;
names = who;
ws = struct();
for j = 1:numel(names)
    ws.(names{j}) = eval(names{j});
end
end
