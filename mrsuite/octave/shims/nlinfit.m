function [beta, R, J, CovB, MSE, ErrorModelInfo] = nlinfit(X, y, modelfun, beta0, options)
% NLINFIT  Octave stand-in for MATLAB's Statistics Toolbox nlinfit.
%
%   beta = nlinfit(X, y, modelfun, beta0)
%   [beta, R, J, CovB, MSE] = nlinfit(X, y, modelfun, beta0, options)
%
% Fits modelfun(beta, X) to y by unconstrained nonlinear least squares using
% Levenberg-Marquardt with a forward-difference Jacobian.
%
%   beta  fitted parameters, in the shape of beta0 (MATLAB preserves orientation)
%   R     residuals, y - modelfun(beta, X)
%   J     Jacobian at the solution
%   CovB  parameter covariance, MSE * inv(J'*J)
%   MSE   residual variance, sum(R.^2) / (n - p)
%
% Part of MRSuite's Octave compatibility shims. Osprey's frequency-referencing code
% calls nlinfit, which is MATLAB-only; Octave-forge "statistics" has no equivalent.
%
% Note on fidelity: this is a different implementation from MathWorks', not a
% reproduction of it. On well-conditioned problems it lands in the same place, but
% on ill-conditioned ones the two can converge to different local minima. Fits that
% matter should be validated against a MATLAB reference.

    if nargin < 5 || isempty(options)
        options = statset('nlinfit');
    end

    function v = opt(name, default)
        if isfield(options, name) && ~isempty(options.(name))
            v = options.(name);
        else
            v = default;
        end
    end

    maxiter   = opt('MaxIter',   200);
    tolfun    = opt('TolFun',    1e-8);
    tolx      = opt('TolX',      1e-8);
    derivstep = opt('DerivStep', eps^(1/3));

    shape = size(beta0);
    b  = beta0(:);
    yv = y(:);
    p  = numel(b);

    % model evaluated as a column vector, with beta restored to its original shape
    fun = @(bb) reshape(modelfun(reshape(bb, shape), X), [], 1);

    r = yv - fun(b);
    if numel(r) ~= numel(yv)
        error('nlinfit: modelfun returned %d values but y has %d.', ...
              numel(r), numel(yv));
    end
    S = r' * r;

    lambda = 1e-2;      % LM damping; falls on success, rises on rejection
    for iter = 1:maxiter
        Jc = local_jacobian(fun, b, derivstep, numel(yv));
        A  = Jc' * Jc;
        g  = Jc' * r;
        dA = diag(A);
        dA(dA == 0) = 1;        % keep the damping term finite for dead parameters

        stepTaken = false;
        for attempt = 1:12      % raise damping until the step actually improves S
            H = A + lambda * diag(dA);
            warnstate = warning('off', 'all');
            delta = H \ g;
            warning(warnstate);

            if any(~isfinite(delta))
                lambda = lambda * 10;
                continue;
            end

            bNew = b + delta;
            rNew = yv - fun(bNew);
            SNew = rNew' * rNew;

            if isfinite(SNew) && SNew < S
                dS = S - SNew;
                b = bNew; r = rNew; S = SNew;
                lambda = max(lambda / 10, 1e-12);
                stepTaken = true;
                % converged once neither the objective nor the step moves much
                if dS <= tolfun * (S + tolfun) || ...
                   norm(delta) <= tolx * (norm(b) + tolx)
                    stepTaken = false;      % break the outer loop below
                end
                break;
            end
            lambda = lambda * 10;
        end

        if ~stepTaken
            break;
        end
    end

    J = local_jacobian(fun, b, derivstep, numel(yv));
    R = yv - fun(b);
    n = numel(yv);
    dof = max(n - p, 1);
    MSE = (R' * R) / dof;

    warnstate = warning('off', 'all');
    CovB = MSE * pinv(J' * J);      % pinv, so a rank-deficient fit still returns
    warning(warnstate);

    beta = reshape(b, shape);
    if size(y, 2) == 1 || size(y, 1) == 1
        R = reshape(R, size(y));
    end
    ErrorModelInfo = struct('ErrorModel', 'constant', 'ErrorParameters', MSE, ...
                            'ErrorVariance', MSE, 'MSE', MSE, ...
                            'ScheffeSimPred', NaN, 'WeightFunction', false, ...
                            'FixedWeights', false, 'RobustWeightFunction', false);
end


function J = local_jacobian(fun, b, derivstep, m)
% Forward-difference Jacobian, d fun_i / d b_j. The step scales with each parameter
% so that the same relative perturbation is used regardless of magnitude.
    p = numel(b);
    J = zeros(m, p);
    f0 = fun(b);
    for j = 1:p
        h = derivstep * max(abs(b(j)), 1);
        bp = b; bp(j) = bp(j) + h;
        J(:, j) = (fun(bp) - f0) / h;
    end
end
