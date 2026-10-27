function y = round(x, n)
% ROUND  Octave's round, and MATLAB's round(x, n) to n digits after the point.
%
%   y = round(x)
%   y = round(x, n)
%
% SimnTG's Sim_UFATG_PRESS rounds its TE series with round(x, 1), which Octave 9.4
% refuses ("Invalid call to round"). On the path only while MRSuite calls SimnTG, so
% every other toolbox keeps the built-in, which a wrapper would slow on every call.
%
% Part of MRSuite's Octave compatibility shims.

    if nargin < 2
        y = builtin('round', x);
    else
        y = builtin('round', x .* 10^n) ./ 10^n;
    end
end
