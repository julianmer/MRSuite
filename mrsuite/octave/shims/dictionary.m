classdef dictionary
% DICTIONARY  Octave stand-in for MATLAB's dictionary (R2022b), with text keys.
%
%   d = dictionary(keys, values)     d(key)   isKey(d, key)   keys(d)   values(d)
%   lookup(d, key, FallbackValue=v)  the value, or v when the key is absent
%
% As MATLAB's for text keys (char, cellstr or the string-array stand-in): each key maps
% to the value at its place; d(key) gives it and raises for a key it lacks. Octave 9
% parses the Name=value call syntax but passes only the value, so a lone third argument
% of lookup is read as the FallbackValue; lookup(d, key, 'FallbackValue', v) works too.
%
% Part of MRSuite's Octave compatibility shims: Spinach looks its isotopes up with it.

    properties (Hidden)
        map
    end

    methods
        function d = dictionary(k, v)
            d.map = containers.Map('KeyType', 'char', 'ValueType', 'any');
            if nargin == 0, return; end
            k = dictionary.keylist(k);
            if ~iscell(v), v = num2cell(v); end
            for i = 1:numel(k), d.map(k{i}) = v{i}; end
        end
        function v = lookup(d, key, varargin)
            key = char(key);
            if d.map.isKey(key), v = d.map(key); return; end
            if numel(varargin) == 1, v = varargin{1};
            elseif numel(varargin) == 2 && strcmpi(varargin{1}, 'FallbackValue'), v = varargin{2};
            else, error('dictionary: key ''%s'' not found', key); end
        end
        function tf = isKey(d, key), tf = d.map.isKey(char(key)); end
        function k = keys(d), k = d.map.keys().'; end
        function v = values(d), v = d.map.values().'; end
        function n = numEntries(d), n = d.map.Count; end
        function v = subsref(d, s)
            if strcmp(s(1).type, '()'), v = d.map(char(s(1).subs{1}));
            else, error('dictionary stand-in: only d(key) indexing is supported'); end
            if numel(s) > 1, v = subsref(v, s(2:end)); end
        end
    end

    methods (Static, Hidden)
        function k = keylist(k)
            if isa(k, 'mrsuite_strings'), k = cellstr(k); elseif ischar(k), k = cellstr(k); end
        end
    end
end
