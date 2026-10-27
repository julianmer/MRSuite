function tf = allfinite(x)
% ALLFINITE  Octave stand-in for MATLAB's allfinite (R2022a).
%
%   tf = allfinite(x)
%
% As MATLAB's: true when every element of x is finite, neither Inf nor NaN; for a
% sparse x only the stored entries can be anything else, so only they are checked.
%
% Part of MRSuite's Octave compatibility shims: Spinach checks its operators with it.

    if issparse(x)
        tf = all(isfinite(nonzeros(x)));
    else
        tf = all(isfinite(x(:)));
    end
end
