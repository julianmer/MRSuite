function r = sum(x, varargin)
% SUM  Spinach-scoped wrapper of Octave's built-in sum adding MATLAB's
% sum(x, 'all') (and 'all' followed by a type such as 'double'); every other call
% is the built-in.
    if ~isempty(varargin) && ischar(varargin{1}) && strcmpi(varargin{1}, 'all')
        r = builtin('sum', x(:), varargin{2:end});
    else
        r = builtin('sum', x, varargin{:});
    end
end
