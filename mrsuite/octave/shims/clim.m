function varargout = clim(varargin)
% CLIM  Octave stand-in for MATLAB's clim (R2022a), the colour limits of an axes.
%
%   clim([lo hi])      lim = clim()      clim(ax, ...)      clim('auto')
%
% MATLAB's clim is the newer name of caxis, with the same calls; Octave 9 has caxis
% only, so the call is caxis's.
%
% Part of MRSuite's Octave compatibility shims: Oryx and MRS4Brain draw maps with it.

    [varargout{1:nargout}] = caxis(varargin{:});
end
