classdef Constant < handle
% Stand-in for parallel.pool.Constant: with one serial worker the value is simply held.
    properties
        Value
    end
    methods
        function c = Constant(v, varargin)
            if isa(v, 'function_handle'), c.Value = v(); else, c.Value = v; end
        end
    end
end
