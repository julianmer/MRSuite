function tf = contains(str, pat, varargin)
% CONTAINS  Octave stand-in for MATLAB's contains.
%
%   tf = contains(str, pat)
%   tf = contains(str, pat, 'IgnoreCase', true)
%
% Accepts char arrays or cellstr for either argument. With a cellstr haystack the
% result is a logical array; with a cellstr needle, true means *any* pattern matched,
% matching MATLAB's semantics.
%
% Part of MRSuite's Octave compatibility shims: Osprey uses modern MATLAB string
% functions that Octave does not implement.

    ignore = false;
    for k = 1:2:numel(varargin)
        if strcmpi(varargin{k}, 'IgnoreCase')
            ignore = logical(varargin{k + 1});
        end
    end

    if ischar(str), strs = {str}; else, strs = cellstr(str); end
    if ischar(pat), pats = {pat}; else, pats = cellstr(pat); end

    if ignore
        strs = lower(strs);
        pats = lower(pats);
    end

    tf = false(size(strs));
    for i = 1:numel(strs)
        for j = 1:numel(pats)
            if ~isempty(strfind(strs{i}, pats{j}))   %#ok<STREMP>
                tf(i) = true;
                break;
            end
        end
    end

    if ischar(str)
        tf = tf(1);
    end
end
