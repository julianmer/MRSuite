classdef DataQueue < handle
% Stand-in for parallel.pool.DataQueue: with work running in the client, send(q, x)
% calls the afterEach callbacks at once with x (MATLAB queues them to the client,
% where they run in order; here that order is the order of the sends).
    properties (Hidden)
        callbacks = {};
    end
    methods
        function afterEach(q, fn), q.callbacks{end + 1} = fn; end
        function send(q, varargin)
            for k = 1:numel(q.callbacks), q.callbacks{k}(varargin{:}); end
        end
    end
end
