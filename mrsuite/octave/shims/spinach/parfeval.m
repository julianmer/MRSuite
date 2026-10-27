function f = parfeval(varargin)
% PARFEVAL  Spinach-scoped serial stand-in: evaluates the function at once in the
% client and returns a completed future (fetchOutputs gives the results).
    k = 1; if ~isa(varargin{1}, 'function_handle'), k = 2; end
    fn = varargin{k}; nout = varargin{k + 1}; args = varargin(k + 2:end);
    out = cell(1, nout);
    [out{:}] = fn(args{:});
    f = mrsuite_serial_future(out);
end
