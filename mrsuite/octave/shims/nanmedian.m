function m = nanmedian(x, varargin)
% NANMEDIAN  Octave stand-in for MATLAB's nanmedian.
%
%   m = nanmedian(x)
%   m = nanmedian(x, dim)
%
% The median ignoring NaNs, along the first non-singleton dimension or along dim; NaN
% where every value is NaN. Of an even count it takes (a+b)/2 where MATLAB takes
% a+(b-a)/2, so the two can differ in the last digit.
%
% Part of MRSuite's Octave compatibility shims: Osprey's spectral registration
% (op_robustSpecReg, op_SpecRegFreqRestrict, op_probabSpecReg) calls it, and Octave has
% it only in its statistics package, which MRSuite does not load.

    m = median(x, varargin{:}, 'omitnan');
end
