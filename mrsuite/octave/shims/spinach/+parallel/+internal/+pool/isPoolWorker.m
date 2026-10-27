function tf = isPoolWorker()
% Stand-in for MATLAB's parallel.internal.pool.isPoolWorker: Octave runs no pool
% workers, so code always runs on the client.
    tf = false;
end
