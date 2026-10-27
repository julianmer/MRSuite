function r = strcmp(a, b)
% STRCMP  MRS4Brain-scoped wrapper of Octave's strcmp: MATLAB compares string
% elements by their text, so mrsuite_text stand-ins (also inside cell arrays) are
% read as their characters before Octave's strcmp runs.
    r = builtin('strcmp', plain(a), plain(b));
end

function x = plain(x)
    if isa(x, 'mrsuite_text'), x = char(x);
    elseif iscell(x)
        k = cellfun(@(e) isa(e, 'mrsuite_text'), x);
        x(k) = cellfun(@char, x(k), 'UniformOutput', false);
    end
end
