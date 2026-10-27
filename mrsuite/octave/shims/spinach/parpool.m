function p = parpool(varargin)
% PARPOOL  Spinach-scoped serial stand-in: "starts" a one-worker pool on the given
% (or a new) cluster and makes it current. Work runs in the client process.
    c = [];
    if nargin > 0 && isa(varargin{1}, 'mrsuite_serial_cluster'), c = varargin{1}; end
    if isempty(c), c = parcluster(); end
    p = mrsuite_serial_pool(c);
    mrsuite_serial_registry('set', p);
end
