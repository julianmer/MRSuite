classdef mrsuite_serial_future < handle
% Completed future of the serial parfeval stand-ins.
    properties
        State = 'finished'; Error = []; out = {};
    end
    methods
        function f = mrsuite_serial_future(out), f.out = out; end
        function wait(f, varargin), end
        function varargout = fetchOutputs(f), varargout = f.out; end
        function cancel(f), end
    end
end
