classdef mrsuite_strings
% MRSUITE_STRINGS  Stand-in for a MATLAB string array, a column of text.
%
% Holds a cell array of text. As MATLAB's string array: s == "x" and s ~= "x" compare
% each element's text (a logical array of s's size), s(i) indexes, s{i} is one element's
% characters, s + t joins element by element, cellstr(s) and char(s) convert.
%
% Part of MRSuite's Octave compatibility shims: the text columns of the table stand-in.

    properties
        c = {};   % cellstr
    end
    methods
        function s = mrsuite_strings(c), if nargin > 0, s.c = cellstr(c); end, end
        function c = cellstr(s), c = s.c; end
        function c = char(s), c = char(s.c); end
        function r = eq(a, b), r = mrsuite_strings.cmp(a, b); end
        function r = ne(a, b), r = ~mrsuite_strings.cmp(a, b); end
        function r = size(s, varargin), r = size(s.c, varargin{:}); end
        function r = isempty(s), r = isempty(s.c); end
        function disp(s), disp(s.c); end
        function v = subsref(s, idx)
            switch idx(1).type
                case '()', v = mrsuite_strings(s.c(idx(1).subs{:}));
                case '{}', v = s.c{idx(1).subs{:}};
                otherwise, v = s.(idx(1).subs);
            end
            if numel(idx) > 1, v = subsref(v, idx(2:end)); end
        end
        function s = subsasgn(s, idx, v)
            if strcmp(idx(1).type, '()'), s.c(idx(1).subs{:}) = cellstr(char(v));
            else, error('mrsuite_strings: unsupported assignment'); end
        end
        function r = plus(a, b)
            % element-wise text concatenation, as MATLAB's string +
            ca = mrsuite_strings.cells(a); cb = mrsuite_strings.cells(b);
            r = mrsuite_strings(strcat(ca, cb));
        end
    end
    methods (Static, Hidden)
        function r = cmp(a, b)
            if isa(a, 'mrsuite_strings') && isa(b, 'mrsuite_strings'), r = strcmp(a.c, b.c);
            elseif isa(a, 'mrsuite_strings'), r = strcmp(a.c, char(b));
            else, r = strcmp(char(a), b.c); end
        end
        function c = cells(x)
            if isa(x, 'mrsuite_strings'), c = x.c; else, c = {char(x)}; end
        end
    end
end
