classdef mrsuite_progress_dialog < handle
% MRSUITE_PROGRESS_DIALOG  The object uiprogressdlg's stand-in returns, with no window.
%
% The properties of MATLAB's ProgressDialog (Value, Message, Title, Indeterminate,
% Cancelable, ShowPercentage, CancelRequested, Icon), set from name-value pairs, which
% toolbox code sets, reads and deletes; nothing is drawn.
%
% Part of MRSuite's Octave compatibility shims.

    properties
        Value = 0; Message = ''; Title = ''; Indeterminate = 'off';
        Cancelable = 'off'; ShowPercentage = 'off'; CancelRequested = false; Icon = '';
    end
    methods
        function d = mrsuite_progress_dialog(varargin)
            for k = 1:2:numel(varargin), d.(varargin{k}) = varargin{k + 1}; end
        end
    end
end
