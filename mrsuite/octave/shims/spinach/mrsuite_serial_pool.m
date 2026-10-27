classdef mrsuite_serial_pool < handle
% Pool object of the serial Parallel Computing Toolbox stand-in. parfor and spmd
% run in the client (Octave executes them serially), so the pool has one worker.
    properties
        Cluster; NumWorkers = 1; IdleTimeout = Inf; ValueStore; Connected = true;
    end
    methods
        function p = mrsuite_serial_pool(c)
            p.Cluster = c; p.NumWorkers = c.NumWorkers; p.ValueStore = mrsuite_serial_store();
        end
    end
end
