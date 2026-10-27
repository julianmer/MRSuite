function T = cell2table(C, varargin)
% CELL2TABLE  Octave stand-in for MATLAB's cell2table, on MRSuite's table stand-in.
%
%   T = cell2table(C)
%   T = cell2table(C, 'VariableNames', names)
%
% As MATLAB's: column j of C becomes a variable of T; a column of numeric or logical
% scalars becomes one numeric column, any other column stays a cell array. Variables
% are named Var1, Var2, ... unless named; MATLAB names them after C's own name where it
% has one (C1, C2, ...), which only display shows.
%
% Part of MRSuite's Octave compatibility shims: Oryx and MRS4Brain tabulate with it.

    cols = cell(1, size(C, 2));
    for j = 1:size(C, 2)
        c = C(:, j);
        if all(cellfun(@(x) (isnumeric(x) || islogical(x)) && isscalar(x), c))
            cols{j} = cell2mat(c);
        else
            cols{j} = c;
        end
    end
    T = table(cols{:}, varargin{:});
end
