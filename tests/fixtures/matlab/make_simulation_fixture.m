function make_simulation_fixture(refdir, outfile)
% MAKE_SIMULATION_FIXTURE  Parity fixture for mtdr.simulate.
%
%   make_simulation_fixture(refdir, outfile)
%
% Runs the reference simulation functions (SimWeights, SimConditions,
% SimPopData, and the mTDRdemo.m mask step) at rng(0), then replays the
% generator to recover every standard-normal draw they consumed, checks that
% the replay reproduces the reference outputs exactly, and saves draws and
% outputs to OUTFILE. The Python test injects the draws into the port's
% deterministic transforms and compares outputs (docs/model.md § 8). RNG
% streams are never compared.
%
%   refdir   path to the mTDRdemo directory (read-only reference)
%   outfile  .mat file to write (tests/fixtures/simulation.mat)
%
% Run from the repository root with
%   matlab -batch "addpath('tests/fixtures/matlab'); make_simulation_fixture('path/to/mTDRdemo', 'tests/fixtures/simulation.mat')"

addpath(fullfile(refdir, 'functionFiles'), fullfile(refdir, 'functionFiles', 'tools_kron'));

% Sizes: r_p ~= T in every task block, so a transposed S fails. Per-regressor
% length scales and weight scales differ, so a mis-assigned entry fails too.
% The last entry of len/rho is the condition-independent term's.
n = 20; T = 15; N = 40;
rP = [2 1 3];
P = numel(rP) + 1;
len = [2 3 1.5 2.5]';
rho = [1 0.5 2 1.5]';
var_uniq = {-2:2, -2:2, [-1 1], 1};
mstdnse = 1/.8;
pdrop = 0.3;

% ---- reference run --------------------------------------------------------
rng(0);
s0 = rng;
[W, S, BB] = SimWeights(n, T, P, [rP T], len, rho);
d = exprnd(mstdnse, [n 1]);
[X, Xcond] = SimConditions(var_uniq, N);
s_noise = rng;
Y = SimPopData(X, BB, d, n, T, N);
hk = zeros(N, n);
Z = zeros(n, T, N);
for k = 1:N
    hk(k, :) = binornd(1, 1 - pdrop, [1 n]);
    Z(:, :, k) = spdiags(hk(k, :)', 0, n, n) * squeeze(Y(:, :, k));
end

% ---- replay the normals ---------------------------------------------------
rng(s0);
G = cell(1, P);       % weight normals, n x r_p
Zs = cell(1, P);      % mvnrnd normals, r_p x T
Rs = cell(1, P);      % cholcov factors, T x T (upper)
for p = 1:P
    rp = [rP T];
    G{p} = randn(n, rp(p));
    Zs{p} = randn(rp(p), T);
    Scov = toeplitz(exp(-((0:T-1) / len(p)).^2 / 2));
    [R, err] = cholcov(Scov);
    assert(err == 0 && isequal(size(R), [T T]), 'kernel %d is not positive definite', p);
    Rs{p} = R;
    assert(isequal(rho(p) * G{p}, W{p}), 'W replay mismatch, p = %d', p);
    assert(isequal(flip((Zs{p} * R)'), S{p}), 'S replay mismatch, p = %d', p);
end
rng(s_noise);
noise = zeros(n, T, N);
for k = 1:N
    noise(:, :, k) = randn(n, T);
end
Yreplay = zeros(n, T, N);
for k = 1:N
    Yreplay(:, :, k) = kronmult({eye(n), X(k, :)}, BB) + diag(1 ./ sqrt(d)) * noise(:, :, k);
end
assert(isequal(Yreplay, Y), 'Y replay mismatch');

[found, cond_idx] = ismember(X, Xcond, 'rows');
assert(all(found), 'X row not in the condition grid');

% Cells as separate variables so scipy.io.loadmat needs no object arrays.
G1 = G{1}; G2 = G{2}; G3 = G{3}; G0 = G{4};
Z1 = Zs{1}; Z2 = Zs{2}; Z3 = Zs{3}; Z0 = Zs{4};
W1 = W{1}; W2 = W{2}; W3 = W{3}; W0 = W{4};
S1 = S{1}; S2 = S{2}; S3 = S{3}; S0 = S{4};
R1 = Rs{1}; R2 = Rs{2}; R3 = Rs{3}; R0 = Rs{4};
ranks = rP;
cond_idx = cond_idx - 1;   % 0-based for Python
matlab_version = version;
save(outfile, 'n', 'T', 'N', 'ranks', 'len', 'rho', 'pdrop', ...
     'G1', 'G2', 'G3', 'G0', 'Z1', 'Z2', 'Z3', 'Z0', ...
     'W1', 'W2', 'W3', 'W0', 'S1', 'S2', 'S3', 'S0', 'R1', 'R2', 'R3', 'R0', 'BB', ...
     'd', 'X', 'Xcond', 'cond_idx', 'noise', 'Y', 'hk', 'Z', 'matlab_version');
fprintf('wrote %s\n', outfile);
end
