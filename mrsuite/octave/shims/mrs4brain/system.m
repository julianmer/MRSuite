function varargout = system(cmd, varargin)
% SYSTEM  MRS4Brain-scoped wrapper of Octave's system.
% On Linux MRS4Brain runs LCModel at the fixed path of its developers' set-up,
% /home/lcmodel/.lcmodel/bin/lcmodel, after cd-ing into the configured
% LCModel_path. MRSuite maps that path to the lcmodel in the current folder (the
% configured LCModel_path), as a container volume mapping would; any other command
% runs unchanged.
    cmd = strrep(cmd, '/home/lcmodel/.lcmodel/bin/lcmodel', './lcmodel');
    [varargout{1:max(nargout, 1)}] = builtin('system', cmd, varargin{:});
    if nargout == 0, varargout = {}; end
end
