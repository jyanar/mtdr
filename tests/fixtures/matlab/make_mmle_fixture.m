function make_mmle_fixture(refdir, simfile, svdfile, outfile)
% MAKE_MMLE_FIXTURE  Parity fixture for the marginal likelihood, ECME, the
% coordinate ascent, the posterior weights and the MMLE rank search.
%
%   make_mmle_fixture(refdir, simfile, svdfile, outfile)
%
% The two datasets of the SVD fixture, so the fixtures chain:
%
%   A  the simulation fixture (Z, X, hk of SIMFILE): n = 20 neurons,
%      T = 15 bins, N = 40 trials, true ranks [2 1 3].
%   B  the demo-scale draw of SVDFILE (B.Z, B.X, B.hk; n = 100, T = 15,
%      N = 40, true ranks B.rP = [2 1 3]), read back rather than redrawn.
%
% For each dataset, with the reference's own statistics
% (MkSuffStats_BilinReg_Sims, MkSuffStatsBTDR_IncompObs_uneqvar_S_fast, the
% mTDRdemo.m xbari/Ybar loop):
%
%   points     random parameter points (rng fixed, every draw saved) at which
%              the three NLL files, both gradients, EBpost_W_uneqvar, MMLE_b
%              and one ECMEtdr sweep are evaluated;
%   fits       the mTDRdemo.m MMLE pipeline at fixed rank vectors:
%              SVDRegress_S_Vdata at [r maxrank] -> ECMEregress_wrapper ->
%              ECMEtdr('converge', 1e-0) with every iterate saved ->
%              Estpars_CoordAscent_lambi_S_b with a per-outer-iteration trace
%              -> EBpost_W_uneqvar, MakeBhat_data, BTDR_AIC_S_lamb_b_wrapper;
%   search     EstRankGreedily with the MMLE objective exactly as mTDRdemo.m
%              calls it, seeded from the ranks of the reference SVD search
%              (SVDRegB_AIC) recorded in SVDFILE, logging every objective
%              evaluation.
%
% ECMEtdr does not return its iterates; iterate k is the output of a fresh
% ECMEtdr('steps', k) run from the same start, which executes the same
% sweeps (the stop rule is the only difference), and the last one is
% asserted equal to the 'converge' run. Estpars_CoordAscent_lambi_S_b does not
% return minFunc's or fminunc's diagnostics; ca_traced below repeats its loop
% line for line with the extra output arguments, and its result is asserted
% equal to the reference function's.
%
% Attribution: ca_traced, fminunc_opts and consistent_ci below reproduce code from
% functionFiles/Estpars_CoordAscent_lambi_S_b.m of mTDRdemo
% (https://github.com/pillowlab/mTDRdemo) by M. C. Aoi and J. W. Pillow. That code
% is theirs; it is included here only to record diagnostics the reference
% function does not return.
%
% The reference's mmx_mkl_single MEX is not shipped for this platform; every
% `try mmx_mkl_single ... catch slow* end` in the reference therefore runs the
% slowMult / slowBackslash / slowChol branch (recorded as mmx_available = 0).
% minFunc's compiled lbfgs MEX files (minFunc_2012/minFunc/compiled/*.mexw64)
% are shipped and used (useMex = 1 by default).
%
%   refdir   path to the mTDRdemo directory (read-only reference)
%   simfile  tests/fixtures/simulation.mat (make_simulation_fixture.m)
%   svdfile  tests/fixtures/svd.mat (make_svd_fixture.m)
%   outfile  .mat file to write (tests/fixtures/mmle.mat)
%
% Run from the repository root with
%   matlab -batch "addpath('tests/fixtures/matlab'); make_mmle_fixture('path/to/mTDRdemo', 'tests/fixtures/simulation.mat', 'tests/fixtures/svd.mat', 'tests/fixtures/mmle.mat')"

addpath(fullfile(refdir, 'functionFiles'), fullfile(refdir, 'functionFiles', 'tools_kron'));
addpath(genpath(fullfile(refdir, 'minFunc_2012')));
mmx_available = exist('mmx_mkl_single'); %#ok<EXIST>

% ---- dataset A: the simulation fixture -------------------------------------
sim = load(simfile);
svd = load(svdfile);
rank_list = [2 1 3; 2 1 2; 3 1 3];
svd_ranks_A = svd.A.searches{1}.rest;     % mTDRdemo.m's start r0 = ones
A = run_dataset(sim.Z, sim.X, sim.hk, rank_list, svd_ranks_A, 11);
A.rP = double(sim.ranks);

% ---- dataset B: the SVD fixture's demo-scale draw --------------------------
Z = double(svd.B.Z);
X = double(svd.B.X);
hk = double(svd.B.hk);
svd_ranks_B = svd.B.searches{1}.rest;
B = run_dataset(Z, X, hk, rank_list, svd_ranks_B, 12);
B.rP = double(svd.B.rP);

% ---- optimiser settings, as the reference resolves them --------------------
mf = struct('display', 'none');           % Estpars_CoordAscent_lambi_S_b.m:16
% Outputs 5-13, 29, 32, 33 of minFunc_processInputOptions.
[~, ~, ~, ~, mf_maxFunEvals, mf_maxIter, mf_optTol, mf_progTol, mf_method, ...
    mf_corrections, mf_c1, mf_c2, mf_LS_init, ~, ~, ~, ~, ~, ~, ~, ~, ~, ...
    ~, ~, ~, ~, ~, ~, mf_useMex, ~, ~, mf_LS_type, mf_LS_interp] = ...
    minFunc_processInputOptions(mf);
minfunc_options = struct('passed', 'options.display = ''none''', ...
    'method_code', mf_method, 'maxFunEvals', mf_maxFunEvals, ...
    'maxIter', mf_maxIter, 'optTol', mf_optTol, 'progTol', mf_progTol, ...
    'corrections', mf_corrections, 'c1', mf_c1, 'c2', mf_c2, ...
    'LS_init', mf_LS_init, 'LS_type', mf_LS_type, 'LS_interp', mf_LS_interp, ...
    'useMex', mf_useMex);
fo = fminunc_opts(4);
fd = optimset('fminunc');                 % fminunc's defaults for unset fields
fminunc_options = struct('passed', ['optimset(''gradobj'',''on'',''display'',''notify'',' ...
    '''Hessian'',''off'',''HessPattern'',speye(n),''algorithm'',' ...
    '''trust-region-reflective'',''maxfunevals'',1000,''maxiter'',1000)'], ...
    'Algorithm', fo.Algorithm, 'GradObj', fo.GradObj, 'Hessian', fo.Hessian, ...
    'Display', fo.Display, 'MaxFunEvals', fo.MaxFunEvals, 'MaxIter', fo.MaxIter, ...
    'TolFun_default', fd.TolFun, 'TolX_default', fd.TolX, ...
    'algorithm_reported', A.fits{1}.ca.fminunc_algorithm);

matlab_version = version;
% Re-running reproduces every array exactly; the 128-byte MAT header records
% the creation time, so the file differs there only.
save(outfile, 'A', 'B', 'mmx_available', 'minfunc_options', ...
    'fminunc_options', 'matlab_version');
fprintf('wrote %s\n', outfile);
end


function fminopts = fminunc_opts(n)
% Estpars_CoordAscent_lambi_S_b.m:23-26, verbatim (mTDRdemo, Aoi & Pillow).
Hesspat = sparse(n,n);
Hesspat(logical(eye(size(Hesspat)))) = ones(n,1);
fminopts = optimset('gradobj','on','display','notify','Hessian','off',...
    'HessPattern',Hesspat,'algorithm','trust-region-reflective','maxfunevals',1000,'maxiter',1000);
end


function out = run_dataset(Z, X, hk, rank_list, svd_ranks, seed)
[n, T, N] = size(Z);
P = size(X, 2);          % the last column is the constant term
Pb = P - 1;
maxrank = min(n, T);     % as mTDRdemo.m
ridgeparam = 0;
opts.MaxIter = 500;
opts.Display = 'off';
g = 0;
assert(all(X(:, P) == 1), 'the last column of X must be the constant term');

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

% ---- random parameter points ----------------------------------------------
rng(seed);
pt_ranks = [2 1 3; 1 2 1; 3 1 2; 2 2 2; 2 1 3];
pt_g = [0 0 0 0 0.7];
points = cell(1, size(pt_ranks, 1));
for k = 1:size(pt_ranks, 1)
    r = pt_ranks(k, :);
    rtot = sum(r);
    gk = pt_g(k);
    s = 0.5 * randn(T * rtot, 1);
    lam = 0.5 + rand(n, 1);
    b = Ybar + 0.2 * randn(T, n);
    pars = [lam; s];
    [Ri_c, zzi_c, Xzetai_c] = ECMEsuffstat(zetai0, Xi, b);
    Shat = mat2cell(reshape(s, T, rtot)', r, T);   % r_p x T blocks (loader test)
    Sblock = blkdiag(Shat{:});
    pt = struct('r', r, 'g', gk, 's', s, 'lam', lam, 'b', b);
    pt.Shat = Shat;
    pt.zzi_c = zzi_c;
    pt.Xzetai_c = Xzetai_c;
    pt.nll_nllonly = neglogLikBTDR_IncompObs_uneqvar_S_nllonly(pars, Ai, Xzetai_c, zzi_c, r, ni, gk);
    pt.nll_nllonly_raw = neglogLikBTDR_IncompObs_uneqvar_S_nllonly(pars, Ai, Xzetai0, zzi, r, ni, gk);
    [pt.nll_Sonly, pt.grad_Sonly] = neglogLikBTDR_IncompObs_uneqvar_Sonly(s, lam, Ai, Ri_c, zzi_c, r, ni, gk);
    [pt.nll_lambonly, pt.grad_lambonly] = neglogLikBTDR_IncompObs_uneqvar_lambonly(lam, Sblock, Ai, Ri_c, zzi_c, r, ni, gk);
    [pt.Wt, pt.Ci_post] = EBpost_W_uneqvar(Sblock, lam, Ai, Xzetai_c, rtot);
    pt.Ci = consistent_ci(Sblock, lam, Ai, r);
    pt.b_mmle = MMLE_b(pt.Ci, Sblock, lam, ni, xbari, Ybar, Xzetai0, r);
    [pt.ecme_parhat, pt.ecme_Q, pt.ecme_parerr, pt.ecme_nll] = ECMEtdr('steps', 1, ...
        [lam; s; vec(b)], Ai, Xi, r, ni, zetai0, xbari, Ybar, Xzetai0);
    points{k} = pt;
end

% ---- the mTDRdemo.m MMLE pipeline at fixed ranks ----------------------------
svdregress = @(r) SVDRegress_S_Vdata(XX, XY, Yn, allstim, T, n, r, ridgeparam, opts);
EMregressfun = @(r, pars0) ECMEtdr('converge', 1e-0, pars0, Ai, Xi, r, ni, zetai0, xbari, Ybar, Xzetai0);
EMparsfun = @(r) ECMEregress_wrapper(r, svdregress, EMregressfun, Ybar);
MMLEParfun = @(lamb0, s0, bhat0, r) Estpars_CoordAscent_lambi_S_b(lamb0, s0, bhat0, r, Ai, Xi, zetai0, ni, xbari, Ybar, Xzetai0);
MMLE_EMregressfun = @(r) MMLE_CoordAscentWrapper(EMparsfun, MMLEParfun, r, n, T);
histdir = tempname;
mkdir(histdir);
fits = cell(1, size(rank_list, 1));
for k = 1:size(rank_list, 1)
    r = rank_list(k, :);
    rtot = sum(r);
    L = n + rtot * T;
    fit = struct('r', r);
    t0 = tic;
    % ECMEregress_wrapper.m, unrolled so that the start is saved.
    fit.pars_svd = svdregress([r maxrank]);
    pars0 = [fit.pars_svd(1:L); vec(Ybar)];
    fit.pars0 = pars0;
    fit.ecme_pars = EMparsfun(r);                       % the reference wrapper itself
    [p4, fit.ecme_Q, fit.ecme_parerr, fit.ecme_nll] = ECMEtdr('converge', 1e-0, ...
        pars0, Ai, Xi, r, ni, zetai0, xbari, Ybar, Xzetai0);
    assert(isequal(p4, fit.ecme_pars), 'ECMEtdr outputs depend on nargout');
    K = numel(fit.ecme_parerr);
    iterates = zeros(numel(pars0), K + 1);
    iterates(:, 1) = pars0;
    for j = 1:K
        iterates(:, j + 1) = ECMEtdr('steps', j, pars0, Ai, Xi, r, ni, zetai0, xbari, Ybar, Xzetai0);
    end
    assert(isequal(iterates(:, end), fit.ecme_pars), 'steps run differs from converge run');
    fit.ecme_iterates = iterates;
    fit.ecme_n_sweeps = K;

    % Coordinate ascent: the reference, then the traced replica.
    lamb0 = fit.ecme_pars(1:n);
    s0 = fit.ecme_pars(n+1:L);
    bhat0 = reshape(fit.ecme_pars(L+1:end), T, n);
    [lambhat, shat, Shat, Shatblock, bhat] = MMLEParfun(lamb0, s0, bhat0, r);
    [lt, st, bt, fit.ca] = ca_traced(lamb0, s0, bhat0, r, Ai, Xi, zetai0, ni, xbari, Ybar, Xzetai0);
    assert(isequal(lt, lambhat) && isequal(st, shat) && isequal(bt, bhat), ...
        'ca_traced differs from Estpars_CoordAscent_lambi_S_b');
    pars_final = [lambhat; shat; vec(bhat)];
    assert(isequal(pars_final, MMLE_EMregressfun(r)), 'MMLE_CoordAscentWrapper differs');
    fit.pars_final = pars_final;
    fit.Shat = Shat;

    % Posterior, Bhat, AIC at the final estimate.
    [~, zzi_f, Xzetai_f] = ECMEsuffstat(zetai0, Xi, bhat);
    fit.Xzetai_f = Xzetai_f;
    fit.zzi_f = zzi_f;
    [fit.Wt, fit.Ci_post] = EBpost_W_uneqvar(Shatblock, lambhat, Ai, Xzetai_f, rtot);
    histfile = fullfile(histdir, sprintf('fit_%d.mat', k));
    parhist = {pars_final}; rhist = r; %#ok<NASGU>
    save(histfile, 'parhist', 'rhist');
    [fit.Bhat, ~, fit.What, fit.Shat_mb, fit.lambhat_mb] = MakeBhat_data(histfile, Ai, Xzetai_f, r, []);
    % mTDRdemo.m:155 passes the uncentred Xzetai0, a reference defect (see
    % docs/differences-from-matlab.md); the port uses the centred statistics.
    [fit.Bhat_raw, ~, fit.What_raw] = MakeBhat_data(histfile, Ai, Xzetai0, r, []);
    fit.nll_final = neglogLikBTDR_IncompObs_uneqvar_S_nllonly(pars_final(1:L), Ai, Xzetai_f, zzi_f, r, ni, g);
    fit.aic = BTDR_AIC_S_lamb_b_wrapper(pars_final, Ai, Xi, zetai0, r, ni, g);
    seconds = toc(t0);   % printed only, so that a re-run reproduces the file
    fprintf('r = [%s]: %d ECME sweeps, %d CA iterations, %.1f s\n', num2str(r), K, ...
        numel(fit.ca.nll), seconds);
    fits{k} = fit;
end

% ---- EstRankGreedily with the MMLE objective, as mTDRdemo.m ---------------
rest0 = svd_ranks(1:Pb);
log_ranks = zeros(0, Pb);
log_values = zeros(0, 1);
log_pars = {};
histfile = fullfile(histdir, 'RankEstMMLE');
t0 = tic;
[rest, rhist, funhist, parhist] = EstRankGreedily(@logged_mmle_aic, MMLE_EMregressfun, ...
    rest0, maxrank, [], histfile);
search = struct('rest0', rest0, 'rest', rest, 'rhist', rhist, 'funhist', funhist, ...
    'calls_ranks', log_ranks, 'calls_values', log_values);
search.parhist = parhist;
search.calls_pars = log_pars;
fprintf('MMLE rank search from [%s]: %.1f s\n', num2str(rest0), toc(t0));
rmdir(histdir, 's');

out.n = n; out.T = T; out.N = N; out.maxrank = maxrank;
out.Ai = Ai; out.zzi = zzi; out.ni = ni; out.Xzetai0 = Xzetai0;
out.xbari = xbari; out.Ybar = Ybar;
out.points = points;
out.rank_list = rank_list;
out.fits = fits;
out.svd_ranks = svd_ranks;
out.search = search;

    % Nested so that the call log is shared with run_dataset's workspace.
    function aic = logged_mmle_aic(pars, r)
        aic = BTDR_AIC_S_lamb_b_wrapper(pars, Ai, Xi, zetai0, r, ni, g);
        log_ranks(end + 1, :) = r(:)';
        log_values(end + 1, 1) = aic;
        log_pars{end + 1} = pars;
    end
end


function Ci = consistent_ci(Shatblock, lambhat, Ai, r)
% Estpars_CoordAscent_lambi_S_b.m:31-35, verbatim (mTDRdemo, Aoi & Pillow).
P = length(r);
rtot = sum(r);
TP = size(Shatblock, 2);
n = numel(lambhat);
S2 = reshape(Shatblock,[],P);
AiIS = permute(reshape(S2*reshape(Ai,P,P*n),rtot,TP,n),[2 1 3]);
SAiIS = reshape(Shatblock*reshape(AiIS,TP,rtot*n),rtot,rtot,n);
lambiSAiS = bsxfun(@times,SAiIS,permute(lambhat,[3 2 1]));
Ci = bsxfun(@plus,lambiSAiS,eye(rtot));
end


function [lambhat, shat, bhat, tr] = ca_traced(lambhat0,shat0,bhat0,r,Ai,Xi,zetai,ni,xbari,Ybar,Xzetai0)
% Estpars_CoordAscent_lambi_S_b.m (mTDRdemo, Aoi & Pillow) line for line, with
% minFunc's and fminunc's diagnostic outputs requested (they do not change the
% iterates) and the full marginal NLL recorded after every outer iteration.
P = length(r);
rtot = sum(r);
[T,n] = size(Ybar);
TP = T*P;
maxsteps = 10;
stopvar = true;
stopcrit = 1e-4;
k=1;
tr = struct('nll', [], 'parerr', [], 'minfunc_f', [], 'minfunc_exitflag', [], ...
    'minfunc_iterations', [], 'minfunc_funcCount', [], 'fminunc_f', [], ...
    'fminunc_exitflag', [], 'fminunc_iterations', [], 'fminunc_funcCount', [], ...
    'fminunc_algorithm', '', 'pars', zeros(numel(lambhat0) + numel(shat0) + numel(bhat0), 0));
while stopvar
    [Ri,zzi,~] = ECMEsuffstat(zetai,Xi,bhat0);
    options.display = 'none';
    loglikS_lamb = @(s)neglogLikBTDR_IncompObs_uneqvar_Sonly(s,lambhat0,Ai,Ri,zzi,r,ni,0);
    [shat, fs, flags, outs] = minFunc(loglikS_lamb,shat0,options);
    Shat = mat2cell(reshape(shat,T,rtot)',r,T);
    Shatblock = blkdiag(Shat{:});
    Hesspat = sparse(n,n);
    Hesspat(logical(eye(size(Hesspat)))) = ones(numel(lambhat0),1);
    fminopts = optimset('gradobj','on','display','notify','Hessian','off',...
        'HessPattern',Hesspat,'algorithm','trust-region-reflective','maxfunevals',1000,'maxiter',1000);
    loglikfunLambda = @(lamb)neglogLikBTDR_IncompObs_uneqvar_lambonly(lamb,Shatblock,Ai,Ri,zzi,r,ni,0);
    [lambhat, fl, flagl, outl] = fminunc(loglikfunLambda,lambhat0,fminopts);
    S2 = reshape(Shatblock,[],P);
    AiIS = permute(reshape(S2*reshape(Ai,P,P*n),rtot,TP,n),[2 1 3]);
    SAiIS = reshape(Shatblock*reshape(AiIS,TP,rtot*n),rtot,rtot,n);
    lambiSAiS = bsxfun(@times,SAiIS,permute(lambhat,[3 2 1]));
    Ci = bsxfun(@plus,lambiSAiS,eye(rtot));
    bhat = MMLE_b(Ci,Shatblock,lambhat,ni,xbari,Ybar,Xzetai0,r);
    newpars = [lambhat;shat;vec(bhat)];
    oldpars = [lambhat0;shat0;vec(bhat0)];
    parerr = max((newpars-oldpars).^2./oldpars.^2);
    % trace (not in the reference)
    [~, zzn, Xzn] = ECMEsuffstat(zetai, Xi, bhat);
    tr.nll(end+1, 1) = neglogLikBTDR_IncompObs_uneqvar_S_nllonly([lambhat; shat], Ai, Xzn, zzn, r, ni, 0);
    tr.parerr(end+1, 1) = parerr;
    tr.minfunc_f(end+1, 1) = fs;
    tr.minfunc_exitflag(end+1, 1) = flags;
    tr.minfunc_iterations(end+1, 1) = outs.iterations;
    tr.minfunc_funcCount(end+1, 1) = outs.funcCount;
    tr.fminunc_f(end+1, 1) = fl;
    tr.fminunc_exitflag(end+1, 1) = flagl;
    tr.fminunc_iterations(end+1, 1) = outl.iterations;
    tr.fminunc_funcCount(end+1, 1) = outl.funcCount;
    tr.fminunc_algorithm = outl.algorithm;
    tr.pars(:, end+1) = newpars;
    % end trace
    if parerr<stopcrit||k>=maxsteps; stopvar = false; end
    lambhat0 = lambhat;
    shat0 = shat;
    bhat0 = bhat;
    k = k + 1;
end
end
