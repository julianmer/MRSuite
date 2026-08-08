function [x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqcurvefit(fun, x0, xdata, ydata, lb, ub, options)
% LSQCURVEFIT  Octave stand-in for MATLAB's Optimization Toolbox lsqcurvefit.
%
%   x = lsqcurvefit(fun, x0, xdata, ydata, lb, ub, options)
%
% Fits fun(x, xdata) to ydata in the least-squares sense. This is lsqnonlin on the
% residual fun(x,xdata) - ydata, which is exactly how MATLAB defines it, so the two
% shims share one solver rather than duplicating the algorithm.
%
% Part of MRSuite's Octave compatibility shims; Osprey uses it for frequency referencing
% (osp_CrChoReferencing, osp_XReferencing).

    if nargin < 5, lb = []; end
    if nargin < 6, ub = []; end
    if nargin < 7, options = []; end

    yv = ydata(:);
    resfun = @(p) reshape(fun(p, xdata), [], 1) - yv;

    [x, resnorm, residual, exitflag, output, lambda, jacobian] = ...
        lsqnonlin(resfun, x0, lb, ub, options);
end
