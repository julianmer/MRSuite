function d = uiprogressdlg(varargin)
% UIPROGRESSDLG  Octave stand-in for MATLAB's uiprogressdlg (R2018a), without a window.
%
%   d = uiprogressdlg(fig, 'Title', t, 'Message', m, ...)
%
% MATLAB draws a progress dialog in the uifigure FIG and returns its ProgressDialog.
% Headless, this returns an object with the same properties (mrsuite_progress_dialog),
% which the toolbox sets, reads and deletes; FIG is ignored and nothing is drawn.
%
% Part of MRSuite's Octave compatibility shims: Oryx reports its stages through it.

    d = mrsuite_progress_dialog(varargin{2:end});
end
