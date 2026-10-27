classdef datetime
% DATETIME  Octave stand-in for MATLAB's datetime (a point in time).
%
%   t = datetime('now')  or  datetime()        current local time
%   t = datetime(str, 'InputFormat', fmt)      parse text (MATLAB/ICU format codes)
%   t = datetime(x, 'ConvertFrom', 'posixtime')
%   ..., 'Format', fmt                         display format (MATLAB default dd-MMM-yyyy HH:mm:ss)
%
% As MATLAB's for one point in time: char and string give it as text in its Format;
% t1 - t2 is a duration, t + d and t - d (a duration, or a number of days) a datetime;
% t1 < t2 and the other comparisons; posixtime, year, month, day, hour, minute, second;
% datestr and datenum take it. Not modelled: time zones, arrays of datetimes, leap
% seconds, and datetime(y, m, d).
%
% Part of MRSuite's Octave compatibility shims: Gannet stamps every stage with it,
% MRS4Brain reads acquisition times with it.

    properties
        num                                 % serial date number (datenum convention)
        Format = 'dd-MMM-yyyy HH:mm:ss';
    end

    methods
        function t = datetime(varargin)
            if nargin == 0 || (ischar(varargin{1}) && strcmpi(varargin{1}, 'now'))
                t.num = now();
                opts = varargin(2:end);
            elseif nargin >= 3 && ischar(varargin{2}) && strcmpi(varargin{2}, 'ConvertFrom')
                switch lower(varargin{3})
                    case 'posixtime', t.num = datenum(1970, 1, 1) + double(varargin{1}) / 86400;
                    case 'datenum',   t.num = double(varargin{1});
                    otherwise, error('datetime: ConvertFrom ''%s'' is not supported', varargin{3});
                end
                opts = varargin(4:end);
            elseif ischar(varargin{1})
                opts = varargin(2:end);
                k = find(strcmpi(opts(1:2:end), 'InputFormat'));
                if isempty(k)
                    t.num = datenum(varargin{1});
                else
                    t.num = datenum(varargin{1}, datetime.icu2datestr(opts{2 * k}));
                end
            else
                error('datetime: unsupported call');
            end
            k = find(strcmpi(opts(1:2:end), 'Format'));
            if ~isempty(k), t.Format = opts{2 * k}; end
        end

        function s = char(t),   s = datestr(t.num, datetime.icu2datestr(t.Format)); end
        function s = string(t), s = char(t); end
        function disp(t),       disp(char(t)); end
        function s = datestr(t, varargin), s = datestr(t.num, varargin{:}); end
        function n = datenum(t), n = t.num; end
        function p = posixtime(t), p = (t.num - datenum(1970, 1, 1)) * 86400; end
        function v = year(t),   v = datetime.part(t, 1); end
        function v = month(t),  v = datetime.part(t, 2); end
        function v = day(t),    v = datetime.part(t, 3); end
        function v = hour(t),   v = datetime.part(t, 4); end
        function v = minute(t), v = datetime.part(t, 5); end
        function v = second(t), v = datetime.part(t, 6); end

        function r = minus(a, b)
            if isa(b, 'datetime'), r = duration(24 * (a.num - b.num), 0, 0);
            else, r = a; r.num = a.num - duration.n(b); end
        end
        function r = plus(a, b)
            if isa(a, 'datetime'), r = a; r.num = a.num + duration.n(b);
            else, r = b; r.num = b.num + duration.n(a); end
        end
        function r = lt(a, b), r = datenum(a) < datenum(b); end
        function r = gt(a, b), r = datenum(a) > datenum(b); end
        function r = le(a, b), r = datenum(a) <= datenum(b); end
        function r = ge(a, b), r = datenum(a) >= datenum(b); end
        function r = eq(a, b), r = datenum(a) == datenum(b); end
    end

    methods (Static, Hidden)
        function v = part(t, k)
            v = datevec(t.num); v = v(:, k);
        end
        function f = icu2datestr(f)
            % MATLAB/ICU codes -> datestr codes: y yyyy MMM MM dd HH mm ss SSS a
            out = ''; i = 1;
            while i <= numel(f)
                c = f(i); j = i;
                while j < numel(f) && f(j + 1) == c, j = j + 1; end
                n = j - i + 1;
                switch c
                    case 'y', out = [out 'yyyy'];
                    case 'M', if n >= 4, out = [out 'mmmm']; elseif n == 3, out = [out 'mmm']; else, out = [out 'mm']; end
                    case 'd', out = [out 'dd'];
                    case 'H', out = [out 'HH'];
                    case 'h', out = [out 'HH'];
                    case 'm', out = [out 'MM'];
                    case 's', out = [out 'SS'];
                    case 'S', out = [out 'FFF'];
                    case 'a', out = [out 'PM'];
                    otherwise, out = [out f(i:j)];
                end
                i = j + 1;
            end
            f = out;
        end
    end
end
