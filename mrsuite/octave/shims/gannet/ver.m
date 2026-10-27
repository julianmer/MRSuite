function v = ver(varargin)
% VER  Gannet-scoped wrapper of Octave's ver.
% Gannet's ToolboxCheck asks MATLAB's ver for four MathWorks toolboxes by name.
% Under Octave the functions Gannet uses from them come from Octave itself and
% MRSuite's shims, so the listing adds those four names (Version '(Octave)');
% everything else is Octave's own ver.
    orig = mrsuite_original('ver', fileparts(mfilename('fullpath')));
    if nargout == 0 && nargin == 0, orig(); return; end
    v = orig(varargin{:});
    if nargin == 0
        names = {'Image Processing Toolbox', 'Optimization Toolbox', ...
                 'Signal Processing Toolbox', 'Statistics and Machine Learning Toolbox'};
        for k = 1:numel(names)
            v(end + 1) = struct('Name', names{k}, 'Version', '(Octave)', 'Release', '', 'Date', '');
        end
    end
end
