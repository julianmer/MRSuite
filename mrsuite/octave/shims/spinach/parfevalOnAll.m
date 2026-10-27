function f = parfevalOnAll(pool, fn, nout, varargin)
% PARFEVALONALL  Spinach-scoped serial stand-in: runs FN once, in the client (the
% only "worker"), and returns a completed future.
    out = cell(1, nout);
    [out{:}] = fn(varargin{:});
    f = mrsuite_serial_future(out);
end
