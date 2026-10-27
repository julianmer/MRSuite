function v = feature(name, varargin)
% FEATURE  Octave stand-in for MATLAB's undocumented feature, for the queries asked of it.
%
%   pid = feature('getpid')      n = feature('numcores')
%
% MATLAB's feature('getpid') is the process id, feature('numcores') the number of
% cores; here Octave's getpid and nproc. Any other feature raises, naming it.
%
% Part of MRSuite's Octave compatibility shims: Spinach and Oryx ask it.

    switch lower(name)
        case 'getpid',   v = getpid();
        case 'numcores', v = nproc();
        otherwise, error('feature: ''%s'' is not provided by the Octave shims', name);
    end
end
