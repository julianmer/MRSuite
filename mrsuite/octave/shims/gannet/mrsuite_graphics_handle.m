classdef mrsuite_graphics_handle
% MRSUITE_GRAPHICS_HANDLE  MATLAB-style (HG2) view of an Octave graphics handle.
%
%   h.Parent, h.Position             read a property (MATLAB dot access)
%   h.Position = p                   set it
%   get(h, name), set(h, name, v)    with MATLAB's property-name abbreviations
%   double(h)                        the Octave handle
%
% MATLAB accepts any unambiguous abbreviation of a property name and, for legacy
% abbreviations that newer properties made ambiguous ('pos' for Position next to
% PositionConstraint), still resolves to the original property; here an
% ambiguous prefix resolves to the shortest matching name, which is that property.

    properties
        h   % Octave graphics handle
    end

    methods
        function obj = mrsuite_graphics_handle(h), obj.h = h; end
        function d = double(obj), d = obj.h; end

        function v = get(obj, varargin)
            if numel(varargin) >= 1, varargin{1} = mrsuite_graphics_handle.resolve(obj.h, varargin{1}); end
            v = get(obj.h, varargin{:});
        end

        function set(obj, varargin)
            for k = 1:2:numel(varargin)
                varargin{k} = mrsuite_graphics_handle.resolve(obj.h, varargin{k});
            end
            set(obj.h, varargin{:});
        end

        function v = subsref(obj, s)
            if strcmp(s(1).type, '.')
                v = get(obj.h, mrsuite_graphics_handle.resolve(obj.h, s(1).subs));
                if numel(s) > 1, v = subsref(v, s(2:end)); end
            else
                error('mrsuite_graphics_handle: only dot access is supported');
            end
        end

        function obj = subsasgn(obj, s, v)
            if numel(s) == 1 && strcmp(s.type, '.')
                set(obj.h, mrsuite_graphics_handle.resolve(obj.h, s.subs), v);
            else
                error('mrsuite_graphics_handle: only h.Property = value is supported');
            end
        end
    end

    methods (Static, Hidden)
        function name = resolve(h, name)
            if ~ischar(name), return; end
            props = fieldnames(get(h));
            hit = props(strncmpi(props, name, numel(name)));
            exact = hit(strcmpi(hit, name));
            if ~isempty(exact), name = exact{1};
            elseif ~isempty(hit), [~, i] = min(cellfun(@numel, hit)); name = hit{i}; end
        end
    end
end
