function p = gcp(opt)
% GCP  Spinach-scoped serial stand-in for MATLAB's gcp: returns the current pool
% or, with 'nocreate', [] when none exists; without 'nocreate' it starts one.
    p = mrsuite_serial_registry('get');
    if ~isempty(p)
        try, p.NumWorkers; catch, p = []; mrsuite_serial_registry('clear'); end   % deleted with delete(pool)
    end
    if isempty(p) && ~(nargin > 0 && strcmpi(opt, 'nocreate'))
        p = parpool();
    end
end
