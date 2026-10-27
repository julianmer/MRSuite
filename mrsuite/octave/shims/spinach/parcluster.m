function c = parcluster(profile)
% PARCLUSTER  Spinach-scoped serial stand-in: a cluster object with one worker.
    c = mrsuite_serial_cluster();
    if nargin > 0, c.Profile = profile; end
end
