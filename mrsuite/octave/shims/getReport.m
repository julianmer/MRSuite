function s = getReport(ME, varargin)
% GETREPORT  Octave stand-in for MATLAB's MException getReport.
%
%   s = getReport(ME)          s = getReport(ME, 'basic')
%
% As MATLAB's: the error message followed by its call stack ('extended', the default),
% or the message alone ('basic'). The 'hyperlinks' option only changes MATLAB's command
% window and is ignored.
%
% Part of MRSuite's Octave compatibility shims: Gannet reports caught errors with it.

    s = ME.message;
    if ~any(strcmpi(varargin, 'basic'))
        for k = 1:numel(ME.stack)
            s = sprintf('%s\nError in %s (line %d)', s, ME.stack(k).name, ME.stack(k).line);
        end
    end
end
