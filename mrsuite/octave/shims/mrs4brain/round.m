function y = round(x, n, type)
% ROUND  MRS4Brain-scoped wrapper of Octave's round adding MATLAB's round(x, n):
% n digits after the decimal point (negative n: left of it), or with
% 'significant', n significant digits. round(x) is Octave's built-in.
    if nargin == 1, y = builtin('round', x); return; end
    if nargin == 3 && strcmpi(type, 'significant')
        n = n - floor(log10(abs(x))) - 1; n(~isfinite(n)) = 0;
    end
    f = 10 .^ n;
    y = builtin('round', x .* f) ./ f;
end
