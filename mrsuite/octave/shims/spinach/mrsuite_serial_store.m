classdef mrsuite_serial_store < handle
% ValueStore of the serial pool stand-in: a key-value map (MATLAB ValueStore API
% subset: keys, remove, isKey, put/get via subsref is not needed by Spinach).
    properties (Hidden)
        map
    end
    methods
        function s = mrsuite_serial_store(), s.map = containers.Map(); end
        function k = keys(s), k = s.map.keys(); end
        function remove(s, k)
            k = cellstr(k); k = k(cellfun(@(x) s.map.isKey(x), k));
            if ~isempty(k), s.map.remove(k); end
        end
        function tf = isKey(s, k), tf = s.map.isKey(k); end
    end
end
