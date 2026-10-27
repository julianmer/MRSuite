function r = all(x, varargin)
% ALL  Spinach-scoped wrapper of Octave's built-in all adding MATLAB's
% all(x, 'all') (over all elements, R2018b); every other call is the built-in.
    if numel(varargin) == 1 && ischar(varargin{1}) && strcmpi(varargin{1}, 'all')
        r = builtin('all', x(:));
    else
        r = builtin('all', x, varargin{:});
    end
end
