function varargout = max(varargin)
% MAX  Spinach-scoped wrapper of Octave's built-in max adding MATLAB's
% max(x, [], 'all') (over all elements; the index output is linear, as in MATLAB);
% every other call is the built-in.
    nout = nargout; if nout < 1, nout = 1; end
    if nargin == 3 && isempty(varargin{2}) && ischar(varargin{3}) && strcmpi(varargin{3}, 'all')
        x = varargin{1};
        [varargout{1:nout}] = builtin('max', x(:));
    else
        [varargout{1:nout}] = builtin('max', varargin{:});
    end
end
