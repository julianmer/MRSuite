function map = parula(n)
% PARULA  Octave stand-in for MATLAB's parula colormap.
%
%   map = parula()      map = parula(n)
%
% MATLAB's default colormap, n rows of RGB. Octave does not ship it; its viridis is
% the closely related perceptual map with the same call, so the colours are close but
% not MATLAB's.
%
% Part of MRSuite's Octave compatibility shims: Gannet colours its maps with it.

    if nargin < 1, map = viridis(); else, map = viridis(n); end
end
