classdef table
% TABLE  Octave stand-in for MATLAB's table, for the calls the toolboxes make.
%
%   T = table(col1, col2, ..., 'VariableNames', names)
%   T.name            a column;   T.name = v / T.name(i) = v   assign
%   T.Properties.UserData / VariableNames / RowNames / ...     metadata
%   height(T), width(T), size(T)
%   T(rows, :)        a table of those rows;   T(rows, :) = []   delete rows
%   [T1; T2]          stack tables with the same variables
%
% Columns are stored as given (numeric or logical arrays with one row per table
% row, cell arrays, text stand-ins). Covers the calls the toolboxes make; joins,
% variable indexing T(:, vars) and MATLAB's display formatting are not provided.
%
% Part of MRSuite's Octave compatibility shims: Spinach keeps its isotopes and
% interactions in tables, Oryx and MRS4Brain build them with cell2table.

    properties (Hidden)
        cols = struct();
        meta = struct('VariableNames', {{}}, 'UserData', [], 'RowNames', {{}}, ...
                      'VariableUnits', {{}}, 'VariableDescriptions', {{}}, 'Description', '');
    end

    methods
        function T = table(varargin)
            k = find(cellfun(@(a) ischar(a) && strcmp(a, 'VariableNames'), varargin), 1);
            if isempty(k)
                data = varargin; names = arrayfun(@(i) sprintf('Var%d', i), 1:numel(data), 'UniformOutput', false);
            else
                data = varargin(1:k - 1); names = cellstr(varargin{k + 1});
            end
            for i = 1:numel(data), T.cols.(names{i}) = data{i}; end
            T.meta.VariableNames = names;
        end

        function n = height(T)
            f = fieldnames(T.cols);
            if isempty(f), n = 0; else, n = size(T.cols.(f{1}), 1); end
        end
        function n = width(T), n = numel(fieldnames(T.cols)); end
        function varargout = size(T, d)
            s = [height(T) width(T)];
            if nargin > 1, s = s(d); end
            if nargout <= 1, varargout = {s}; else, varargout = num2cell(s); end
        end

        function v = subsref(T, s)
            switch s(1).type
                case '.'
                    if strcmp(s(1).subs, 'Properties'), v = T.meta; else, v = T.cols.(s(1).subs); end
                case '()'
                    v = table.rows(T, s(1).subs{1});
                otherwise
                    error('table stand-in: T{...} indexing is not supported');
            end
            if numel(s) > 1, v = subsref(v, s(2:end)); end
        end

        function T = vertcat(varargin)
            T = varargin{1}; f = fieldnames(T.cols);
            for k = 2:numel(varargin)
                for j = 1:numel(f), T.cols.(f{j}) = [T.cols.(f{j}); varargin{k}.cols.(f{j})]; end
            end
        end

        function T = subsasgn(T, s, v)
            if strcmp(s(1).type, '()') && numel(s) == 1 && isempty(v) && isnumeric(v)
                keep = true(height(T), 1); keep(s(1).subs{1}) = false;
                T = table.rows(T, keep); return
            end
            if ~strcmp(s(1).type, '.'), error('table stand-in: only T.name = value and T(rows,:) = [] are supported'); end
            if strcmp(s(1).subs, 'Properties')
                if numel(s) == 1, T.meta = v; else, T.meta = subsasgn(T.meta, s(2:end), v); end
            else
                name = s(1).subs;
                if numel(s) == 1, T.cols.(name) = v;
                else, T.cols.(name) = subsasgn(T.cols.(name), s(2:end), v); end
                if ~any(strcmp(T.meta.VariableNames, name)), T.meta.VariableNames{end + 1} = name; end
            end
        end
    end

    methods (Static, Hidden)
        function R = rows(T, r)
            R = T; f = fieldnames(T.cols);
            for j = 1:numel(f)
                c = T.cols.(f{j});
                if isa(c, 'mrsuite_strings'), R.cols.(f{j}) = subsref(c, substruct('()', {r}));
                else, R.cols.(f{j}) = c(r, :); end
            end
        end
    end
end
