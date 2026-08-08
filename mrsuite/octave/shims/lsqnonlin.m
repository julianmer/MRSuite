function [x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqnonlin(fun, x0, lb, ub, options)
% LSQNONLIN  Octave stand-in for MATLAB's Optimization Toolbox lsqnonlin.
%
%   x = lsqnonlin(fun, x0)
%   [x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqnonlin(fun, x0, lb, ub, options)
%
% Minimises sum(fun(x).^2) subject to lb <= x <= ub. `fun` returns the residual vector,
% not a scalar. Bounds may be [] for unbounded.
%
% Part of MRSuite's Octave compatibility shims. Osprey calls this from its multiplex fit
% and its edited sub-spectra alignment; Octave-forge "optim" offers nonlin_residmin with
% a different signature, so a MATLAB-shaped wrapper is easier to drop in than rewriting
% the call sites.
%
% Solver is the shared bounded Levenberg-Marquardt in mrsuite_lm.m.

    if nargin < 3, lb = []; end
    if nargin < 4, ub = []; end
    if nargin < 5 || isempty(options), options = optimset('lsqnonlin'); end

    getopt = @(n, d) local_get(options, n, d);
    maxiter   = getopt('MaxIter', 400);
    tolfun    = getopt('TolFun', 1e-8);
    tolx      = getopt('TolX', 1e-8);
    derivstep = getopt('FinDiffRelStep', eps^(1/3));

    [x, residual, jacobian, iter, exitflag] = ...
        mrsuite_lm(fun, x0, lb, ub, maxiter, tolfun, tolx, derivstep);

    resnorm = sum(residual(:).^2);
    output = struct('iterations', iter, 'funcCount', NaN, ...
                    'algorithm', 'levenberg-marquardt (MRSuite shim)', ...
                    'firstorderopt', norm(jacobian' * residual(:), inf), ...
                    'message', 'MRSuite Octave shim for lsqnonlin');
    lambda = struct('lower', [], 'upper', []);
end


function v = local_get(o, name, default)
    if isstruct(o) && isfield(o, name) && ~isempty(o.(name))
        v = o.(name);
    else
        v = default;
    end
end
