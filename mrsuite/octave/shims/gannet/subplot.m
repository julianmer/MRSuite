function varargout = subplot(varargin)
% SUBPLOT  Gannet-scoped wrapper of Octave's subplot.
% MATLAB's subplot returns a graphics object: h.Parent, h.Position = ... and
% get/set with abbreviated property names work on it. Octave returns a plain
% numeric handle, so the returned handle is wrapped in mrsuite_graphics_handle,
% which forwards those calls to Octave's get/set. Gannet only uses subplot's
% output with get, set and dot access.
    orig = mrsuite_original('subplot', fileparts(mfilename('fullpath')));
    if nargout == 0
        orig(varargin{:});
    else
        varargout{1} = mrsuite_graphics_handle(orig(varargin{:}));
    end
end
