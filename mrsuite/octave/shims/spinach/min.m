function varargout = min(varargin)
% MIN  Spinach-scoped wrapper of Octave's built-in min adding MATLAB's
% min(x, [], 'all') (over all elements; the index output is linear, as in MATLAB);
% every other call is the built-in.
    nout = nargout; if nout < 1, nout = 1; end
    if nargin == 3 && isempty(varargin{2}) && ischar(varargin{3}) && strcmpi(varargin{3}, 'all')
        x = varargin{1};
        [varargout{1:nout}] = builtin('min', x(:));
    else
        [varargout{1:nout}] = builtin('min', varargin{:});
    end
end
