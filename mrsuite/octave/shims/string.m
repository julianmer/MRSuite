function s = string(x)
% STRING  Octave stand-in for MATLAB's string conversion (R2016b).
%
%   s = string(x)
%
% MATLAB converts to its string type. Octave has none, so text stays text: a char row
% as it is, a cell array of them as it is. A number converts as MATLAB's string(number)
% writes it (five significant digits), an array of them to a cell array of such text;
% an object through its char method (the datetime stand-in's).
%
% Part of MRSuite's Octave compatibility shims.

    if ischar(x) || iscellstr(x)
        s = x;
    elseif isnumeric(x) || islogical(x)
        if isscalar(x), s = num2str(x, 5);
        else, s = arrayfun(@(v) num2str(v, 5), x, 'UniformOutput', false); end
    elseif isobject(x)
        s = char(x);
    else
        error('string: cannot convert a %s', class(x));
    end
end
