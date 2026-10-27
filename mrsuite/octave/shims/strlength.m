function n = strlength(s)
% STRLENGTH  Octave stand-in for MATLAB's strlength (R2016b).
%
%   n = strlength(s)
%
% As MATLAB's: the number of characters of a text, or of each entry of a cell array of
% them, in the cell array's shape.
%
% Part of MRSuite's Octave compatibility shims: MRS4Brain reads Bruker headers with it.

    if iscell(s), n = cellfun(@numel, s); else, n = numel(s); end
end
