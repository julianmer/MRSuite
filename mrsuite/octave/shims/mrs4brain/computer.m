function [c, maxsize, endian] = computer(varargin)
% COMPUTER  MRS4Brain-scoped stand-in for MATLAB's computer: MATLAB's platform
% codes ('GLNXA64', 'MACI64', 'MACA64', 'PCWIN64') instead of Octave's GNU
% triplet. MRS4Brain chooses its LCModel call by this code.
    if ispc, c = 'PCWIN64';
    elseif ismac
        [~, m] = system('uname -m'); if strncmp(strtrim(m), 'arm', 3), c = 'MACA64'; else, c = 'MACI64'; end
    else, c = 'GLNXA64'; end
    maxsize = 2^48 - 1; endian = 'L';
    if nargin > 0 && strcmpi(varargin{1}, 'arch'), c = lower(c); end
end
