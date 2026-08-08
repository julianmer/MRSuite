function opts = optimoptions(varargin)
% OPTIMOPTIONS  Octave stand-in for MATLAB's optimoptions.
%
%   opts = optimoptions(@lsqnonlin)
%   opts = optimoptions(opts, 'Name', value, ...)
%
% MATLAB returns an options *object*; this returns the same plain struct optimset does,
% which is all the shimmed solvers read. Osprey uses both entry points interchangeably.

    opts = optimset(varargin{:});
end
