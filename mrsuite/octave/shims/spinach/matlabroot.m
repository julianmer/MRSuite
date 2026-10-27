function r = matlabroot()
% MATLABROOT  Spinach-scoped stand-in for MATLAB's matlabroot.
% Spinach's startup check (existentials) requires toolbox/parallel, nnet, rl,
% optim, stats and map under matlabroot. Under Octave the parts Spinach calls are
% MRSuite's shims, so matlabroot names a folder holding those six (empty) toolbox
% folders. Nothing reads files under it.
    r = fullfile(fileparts(mfilename('fullpath')), 'matlabroot-tree');
end
