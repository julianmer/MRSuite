function r = days(x)
% DAYS  Octave stand-in for MATLAB's days: a duration of x days.
%
%   d = days(x)
%
% As MATLAB's: days(x) for a number is a duration of x days. days(d) of a duration,
% the number of days, is a method of the duration stand-in.
%
% Part of MRSuite's Octave compatibility shims: Gannet compares times with it.

    r = duration(24 * x, 0, 0);
end
