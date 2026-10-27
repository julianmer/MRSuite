classdef duration
% DURATION  Octave stand-in for MATLAB's duration, a length of time.
%
%   d = duration(h, m, s)        d = duration([h m s])
%
% As MATLAB's for one length of time: built from hours, minutes and seconds; compared
% with another duration (or a number of days, as Octave's datenum arithmetic gives
% them); added and subtracted; days(d) and hours(d) give the number. Arrays of
% durations and display formats are not modelled.
%
% Part of MRSuite's Octave compatibility shims: what the datetime stand-in's
% difference is, and what days() returns.

    properties
        d   % length in days
    end

    methods
        function obj = duration(h, m, s)
            if nargin == 1, s = h(3); m = h(2); h = h(1); end
            obj.d = (h + m / 60 + s / 3600) / 24;
        end
        function r = days(obj),  r = obj.d; end
        function r = hours(obj), r = obj.d * 24; end
        function r = gt(a, b), r = duration.n(a) > duration.n(b); end
        function r = lt(a, b), r = duration.n(a) < duration.n(b); end
        function r = ge(a, b), r = duration.n(a) >= duration.n(b); end
        function r = le(a, b), r = duration.n(a) <= duration.n(b); end
        function r = plus(a, b),  r = duration(24 * (duration.n(a) + duration.n(b)), 0, 0); end
        function r = minus(a, b), r = duration(24 * (duration.n(a) - duration.n(b)), 0, 0); end
        function disp(obj), printf('%g days\n', obj.d); end
    end

    methods (Static, Hidden)
        function v = n(x)
            if isa(x, 'duration'), v = x.d; else, v = x; end
        end
    end
end
