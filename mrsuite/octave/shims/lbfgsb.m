function [x, f, info] = lbfgsb(fcn, l, u, opts)
% LBFGSB  Osprey's L-BFGS-B-C driver, without the per-call bookkeeping.
%
%   [x, f, info] = lbfgsb(fcn, l, u, opts)
%
% The same interface, option defaults and checks as the lbfgsb.m bundled with Osprey
% (L-BFGS-B-C, Stephen Becker, BSD licence), and the same call into the same MEX, so
% the iterates are the same. What is left out is fcn_wrapper, which the original puts
% between the MEX and the objective to keep a history of f and norm(g, Inf) for
% printing and for info.err. Osprey sets printEvery = 0 and discards info, and in
% Octave the wrapper was the largest single cost of a fit: the MEX calls back into the
% interpreter about a million times per spectrum, and every extra layer is paid on
% each call. info.err is therefore not returned.
%
% Osprey only ever hands this a bound-constrained least-squares problem, as
% @(x) fminunc_wrapper(x, fcn, grad) with fcn = @(x) norm(A + M*x - b)^2 (A optional) and
% grad = @(x) 2*(H*x - c). Those are recognised and run by lbfgsb_quadratic, the same loop
% with f and g computed in C, so there is no callback at all. Its arithmetic is Octave's,
% operation for operation, and it is only used once it has been confirmed to reproduce
% this Octave's own norm bit for bit (see calibrate); anything else takes the path above.
%
% Osprey calls this some ten thousand times per spectrum, and every step of the
% interpreter costs tens of microseconds, so the work per call is kept to the minimum:
% options are read inline, and a handle's text is parsed only the first time it is seen.

    if nargin < 4 || isempty(opts), opts = struct(); end

    n = length(l);
    if length(u) ~= n, error('l and u must be same length'); end
    if isfield(opts, 'x0'), x = opts.x0 + 0; else, x = zeros(n, 1); end
    if size(x, 2) ~= 1, error('x0 must be a column vector'); end
    if size(l, 2) ~= 1, error('l must be a column vector'); end
    if size(u, 2) ~= 1, error('u must be a column vector'); end
    if size(x, 1) ~= n, error('x0 and l have mismatchig sizes'); end
    if size(u, 1) ~= n, error('u and l have mismatchig sizes'); end

    if isfield(opts, 'm'), m = opts.m; else, m = 5; end
    if isfield(opts, 'factr'), factr = opts.factr; else, factr = 1e7; end
    if isfield(opts, 'pgtol'), pgtol = opts.pgtol; else, pgtol = 1e-5; end
    if isfield(opts, 'maxIts'), maxIts = opts.maxIts; else, maxIts = 100; end
    if isfield(opts, 'maxTotalIts'), maxTotalIts = opts.maxTotalIts; else, maxTotalIts = 5e3; end
    if isfield(opts, 'verbose'), iprint = opts.verbose; else, iprint = -1; end
    if any(m < 0) || any(factr < 0) || any(pgtol < 0) || any(maxIts < 1)
        error('Value is too small');
    end

    % 0 unbounded, 1 lower only, 2 both, 3 upper only
    nbd = isfinite(l) + isfinite(u) + 2 * isinf(l) .* isfinite(u);
    if ispc, nbd = int32(nbd); else, nbd = int64(nbd); end

    fused = calibrate();
    q = [];
    if fused >= 0, q = quadratic(fcn); end
    if ~isempty(q)
        [f, x, task, outer, total] = lbfgsb_quadratic(m, x, l, u, nbd, q{:}, factr, pgtol, ...
                                                      iprint, maxIts, maxTotalIts, fused);
    else
        if iscell(fcn)
            callF = @(x) fminunc_wrapper(x, fcn{1}, fcn{2});    % separate f and g
        else
            callF = fcn;
        end
        % the MEX calls feval(fcn, x, iteration); the iteration is only for the history
        [f, x, task, outer, total] = lbfgsb_wrapper(m, x, l, u, nbd, @(x, ~) callF(x), ...
                                                    factr, pgtol, iprint, maxIts, maxTotalIts);
    end

    info.iterations      = outer;
    info.totalIterations = total;
    info.task            = task;
end


function q = quadratic(fcn)
% Osprey's least-squares problem as {M, A, b, H, c}, or [] if fcn is anything else.
%
% The shapes are parsed from the handles' text once per distinct text and remembered:
% Osprey builds a new handle for every call, but only ever from a few lines of code.
% Sizes and types are checked by lbfgsb_quadratic, where it costs nothing.
    q = [];
    if ~isa(fcn, 'function_handle'), return; end

    outer = functions(fcn);
    p = remember(outer.function, @parse_outer);
    if isempty(p), return; end
    ws = outer.workspace{1};
    fi = functions(ws.(p{1}));
    gi = functions(ws.(p{2}));
    pf = remember(fi.function, @parse_norm);
    pg = remember(gi.function, @parse_grad);
    if isempty(pf) || isempty(pg), return; end

    fw = fi.workspace{1}; gw = gi.workspace{1};
    A = [];
    if ~isempty(pf{1}), A = fw.(pf{1}); end
    q = {fw.(pf{2}), A, fw.(pf{3}), gw.(pg{1}), gw.(pg{2})};
end


function tokens = remember(text, parse)
% parse(text without spaces), computed once per distinct text. The three kinds of text
% never coincide, so one cache serves all three parsers.
    persistent texts parsed
    if isempty(texts), texts = {}; parsed = {}; end
    k = find(strcmp(text, texts), 1);
    if isempty(k)
        texts{end + 1} = text;
        parsed{end + 1} = parse(strrep(text, ' ', ''));
        k = numel(texts);
    end
    tokens = parsed{k};
end


function t = parse_outer(body)
% {fcn, grad} from '@(x)fminunc_wrapper(x,fcn,grad)', or [].
    t = regexp(body, '^@\(x\)fminunc_wrapper\(x,(\w+),(\w+)\)$', 'tokens', 'once');
    if numel(t) ~= 2, t = []; else, t = reshape(t, 1, []); end
end


function t = parse_norm(body)
% {A, M, b} from '@(x)norm(A+M*x-b)^2', A '' when there is no offset, or [].
% Two patterns rather than an optional group: Octave drops an unmatched group's token.
    t = regexp(body, '^@\(x\)norm\((\w+)\+(\w+)\*x-(\w+)\)\^2$', 'tokens', 'once');
    if numel(t) ~= 3
        t = regexp(body, '^@\(x\)norm\((\w+)\*x-(\w+)\)\^2$', 'tokens', 'once');
        if numel(t) ~= 2, t = []; return; end
        t = [{''}, reshape(t, 1, [])];
    else
        t = reshape(t, 1, []);
    end
end


function t = parse_grad(body)
% {H, c} from '@(x)2*(H*x-c)', or [].
    t = regexp(body, '^@\(x\)2\*\((\w+)\*x-(\w+)\)$', 'tokens', 'once');
    if numel(t) ~= 2, t = []; else, t = reshape(t, 1, []); end
end


function fused = calibrate()
% Which of lbfgsb_quadratic's two norms this Octave's norm is, or -1 if neither.
%
% The same Octave computes norm differently depending on the compiler it was built with:
% gcc fuses its accumulator into multiply-adds, clang does not. Decided once per session
% on fixed vectors spanning sixteen orders of magnitude, without touching the random
% number generator, whose state the fit may depend on.
    persistent known
    if isempty(known)
        known = -1;
        if exist('lbfgsb_quadratic', 'file') == 3
            k = (1:4000)';
            same = [true true];
            for j = 1:40
                v = sin(k * j + j) .* 10 .^ (mod(k * j, 17) - 8);
                v = v(1:(100 * j));
                for mode = [0 1]
                    same(mode + 1) = same(mode + 1) && norm(v) == lbfgsb_quadratic('norm', v, mode);
                end
            end
            % only a verdict the vectors actually told apart
            if xor(same(1), same(2)), known = find(same) - 1; end
        end
    end
    fused = known;
end
