function [val, count, errmsg, nextindex] = sscanf(varargin)
% SSCANF  Octave/MATLAB compatibility wrapper around the built-in sscanf.
%
% Clears errmsg in exactly one situation: the scan stopped because only trailing
% whitespace was left. Everything else, including every returned value, passes
% through the built-in untouched.
%
% Why this exists
% ---------------
% Measured against MATLAB R2025a:
%
%   sscanf(' 0.6169E-01  0.1588E+01\n', '%f', inf)
%       MATLAB  -> count = 2, errmsg = ''
%       Octave  -> count = 2, errmsg = 'sscanf: format failed to match'
%
%   sscanf(' 0.6169E-01  0.1588E+01', '%f', inf)     % same, no newline
%       both    -> count = 2, errmsg = ''
%
% Octave treats a trailing newline as a failed conversion attempt; MATLAB does not.
% Since fgets returns the newline, every data line read that way trips it. FID-A's
% io_readlcmraw_basis scans each line with
%
%     [A, count, errmsg] = sscanf(line, '%f', inf);
%     if ~isempty(errmsg), error(...); end
%
% and so aborts on the first data line under Octave while reading the same file
% correctly under MATLAB.
%
% Scope
% -----
% Genuine trailing junk is left alone: for ' 1.5 2.5 junk' the remainder is not
% whitespace, so errmsg survives and the caller still sees the error, exactly as
% MATLAB reports it. nextindex is what makes this distinguishable.
%
% Verified by tests/octave/test_matlab_parity.py, which runs these cases through both
% engines and compares. Widening the rule would mask genuine read errors: MATLAB
% reports non-numeric text as an error too.

    [val, count, errmsg, nextindex] = builtin('sscanf', varargin{:});

    if isempty(errmsg) || isempty(strfind(lower(errmsg), 'format failed to match'))  %#ok<STREMP>
        return;
    end

    str = varargin{1};
    if ~ischar(str)
        return;
    end

    if nextindex >= 1 && nextindex <= numel(str)
        remainder = str(nextindex:end);
    else
        remainder = '';      % scan consumed the whole string
    end

    if isempty(remainder) || all(isspace(remainder))
        errmsg = '';
    end
end
