function r = any(x, varargin)
% ANY  Spinach-scoped wrapper of Octave's built-in any adding MATLAB's
% any(x, 'all') (over all elements, R2018b); every other call is the built-in.
    if numel(varargin) == 1 && ischar(varargin{1}) && strcmpi(varargin{1}, 'all')
        r = builtin('any', x(:));
    else
        r = builtin('any', x, varargin{:});
    end
end
