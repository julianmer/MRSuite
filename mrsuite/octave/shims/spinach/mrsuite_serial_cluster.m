classdef mrsuite_serial_cluster < handle
% Cluster object of the serial Parallel Computing Toolbox stand-in: one worker.
    properties
        Profile = 'Processes'; NumWorkers = 1; NumThreads = 1;
        JobStorageLocation = ''; AdditionalProperties = struct();
    end
end
