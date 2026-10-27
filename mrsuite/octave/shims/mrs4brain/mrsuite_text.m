classdef mrsuite_text
% MRSUITE_TEXT  Minimal stand-in for a MATLAB string scalar.
%
% Octave reads "..." literals as char, so MATLAB's string arithmetic on them
% ("abc" + newline, "N=" + 5) becomes numeric. Where a scoped shim can hand the
% toolbox a string instead (newline, makeValidName), this object restores MATLAB's
% behaviour: + concatenates (numbers via string()), s{1} and char(s) give the
% characters, strfind/strlength/strcmp/strsplit/str2double/contains/eq work on the
% text, and concatenating it into char ([... s]) yields char.

    properties
        t = ''   % the characters
    end

    methods
        function s = mrsuite_text(t)
            if nargin > 0, s.t = char(t); end
        end
        function c = char(s), c = s.t; end
        function c = cellstr(s), c = {s.t}; end
        function disp(s), disp(s.t); end
        function r = plus(a, b), r = mrsuite_text([mrsuite_text.txt(a) mrsuite_text.txt(b)]); end
        function r = horzcat(varargin)
            parts = cellfun(@mrsuite_text.txt, varargin, 'UniformOutput', false);
            r = [parts{:}];
        end
        function r = strfind(a, b, varargin), r = strfind(mrsuite_text.txt(a), mrsuite_text.txt(b), varargin{:}); end
        function n = strlength(s), n = numel(s.t); end
        function r = strcmp(a, b), r = strcmp(mrsuite_text.txt(a), mrsuite_text.txt(b)); end
        function r = strcmpi(a, b), r = strcmpi(mrsuite_text.txt(a), mrsuite_text.txt(b)); end
        function r = eq(a, b), r = strcmp(mrsuite_text.txt(a), mrsuite_text.txt(b)); end
        function r = ne(a, b), r = ~eq(a, b); end
        function r = contains(a, b, varargin), r = contains(mrsuite_text.txt(a), mrsuite_text.txt(b), varargin{:}); end
        function r = strsplit(s, varargin), r = strsplit(s.t, varargin{:}); end
        function r = str2double(s), r = str2double(s.t); end
        function r = isempty(s), r = isempty(s.t); end
        function r = fieldname_text(s), r = s.t; end
        function v = subsref(s, idx)
            switch idx(1).type
                case '{}', v = s.t;
                case '.',  v = s.(idx(1).subs);
                otherwise, v = s;
            end
            if numel(idx) > 1, v = subsref(v, idx(2:end)); end
        end
    end

    methods (Static, Hidden)
        function c = txt(x)
            if isa(x, 'mrsuite_text'), c = x.t;
            elseif ischar(x), c = x;
            elseif isnumeric(x) || islogical(x), c = string(x);
            else, c = char(x); end
        end
    end
end
