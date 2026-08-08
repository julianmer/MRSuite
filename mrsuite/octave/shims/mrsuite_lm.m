function [x, resid, J, iter, exitflag] = mrsuite_lm(resfun, x0, lb, ub, maxiter, tolfun, tolx, derivstep)
% MRSUITE_LM  Bounded Levenberg-Marquardt on a residual vector.
%
% Shared core behind the lsqnonlin / lsqcurvefit shims. Minimises sum(resfun(x).^2)
% subject to lb <= x <= ub, using a forward-difference Jacobian and adaptive damping.
%
%   resfun     function of x returning a residual column vector
%   x0         starting point
%   lb, ub     bounds; [] means unbounded on that side
%   exitflag   1 converged, 0 hit maxiter
%
% Bounds are handled by projection: each trial point is clamped into the box before the
% residual is evaluated, and the damping is raised if the clamped step does not improve.
% That is adequate for the mild boxes MRS fitting uses (linewidths, shifts, phases) and
% avoids pulling in an active-set solver.
%
% Not used by the nlinfit shim, which keeps its own unbounded loop: that one is verified
% against MATLAB to ~1e-8 and is deliberately left undisturbed.

    x = x0(:);
    n = numel(x);
    if isempty(lb), lb = -inf(n, 1); else, lb = lb(:); end
    if isempty(ub), ub =  inf(n, 1); else, ub = ub(:); end
    lb(~isfinite(lb)) = -inf;
    ub(~isfinite(ub)) =  inf;

    clamp = @(v) min(max(v, lb), ub);
    x = clamp(x);

    resid = resfun(x);
    resid = resid(:);
    S = resid' * resid;

    lambda = 1e-2;
    exitflag = 0;
    for iter = 1:maxiter
        J = local_jac(resfun, x, derivstep, numel(resid));
        A = J' * J;
        g = -J' * resid;                 % steepest-descent direction for 0.5*||r||^2
        dA = diag(A);
        dA(dA == 0) = 1;

        improved = false;
        for attempt = 1:12
            warnstate = warning('off', 'all');
            delta = (A + lambda * diag(dA)) \ g;
            warning(warnstate);

            if any(~isfinite(delta))
                lambda = lambda * 10;
                continue;
            end

            xNew = clamp(x + delta);
            rNew = resfun(xNew);
            rNew = rNew(:);
            SNew = rNew' * rNew;

            if isfinite(SNew) && SNew < S
                dS = S - SNew;
                step = norm(xNew - x);
                x = xNew; resid = rNew; S = SNew;
                lambda = max(lambda / 10, 1e-12);
                improved = true;
                if dS <= tolfun * (S + tolfun) || step <= tolx * (norm(x) + tolx)
                    exitflag = 1;
                    improved = false;    % converged; stop the outer loop
                end
                break;
            end
            lambda = lambda * 10;
        end

        if ~improved
            if exitflag == 0 && lambda >= 1e10
                exitflag = 1;            % damping saturated: no further progress
            end
            break;
        end
    end

    J = local_jac(resfun, x, derivstep, numel(resid));
    x = reshape(x, size(x0));
end


function J = local_jac(resfun, x, derivstep, m)
    p = numel(x);
    J = zeros(m, p);
    f0 = resfun(x); f0 = f0(:);
    for j = 1:p
        h = derivstep * max(abs(x(j)), 1);
        xp = x; xp(j) = xp(j) + h;
        fj = resfun(xp);
        J(:, j) = (fj(:) - f0) / h;
    end
end
