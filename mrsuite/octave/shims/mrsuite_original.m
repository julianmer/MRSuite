function f = mrsuite_original(name, shimdir)
% MRSUITE_ORIGINAL  A handle to the function NAME that a shim in SHIMDIR stands in front of.
%
%   f = mrsuite_original('imread', fileparts(mfilename('fullpath')))
%
% NAME is resolved with SHIMDIR off the path, so the handle binds to what the shim
% shadows (an Octave m-file or built-in), and kept for later calls. A shim that adds
% MATLAB behaviour around one of Octave's own functions calls through it.
%
% Part of MRSuite's Octave compatibility shims.

    persistent cache
    if isempty(cache), cache = containers.Map(); end
    key = [shimdir filesep name];
    if ~isKey(cache, key)
        p = path();
        rmpath(shimdir);
        cache(key) = str2func(name);
        path(p);
    end
    f = cache(key);
end
