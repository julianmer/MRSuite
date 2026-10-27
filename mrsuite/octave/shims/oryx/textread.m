function varargout = textread(varargin)
% TEXTREAD  Oryx-scoped wrapper of Octave's textread.
% With the default whitespace delimiter, MATLAB's textread treats a run of spaces
% and line ends as one separator, so a '%s' read never yields empty strings.
% Octave's textread returns an empty string per blank line; for a single '%s'
% field without a 'delimiter' option those are removed, otherwise the call is
% passed on unchanged.
    orig = mrsuite_original('textread', fileparts(mfilename('fullpath')));
    [varargout{1:max(nargout, 1)}] = orig(varargin{:});
    fmt = '';
    if numel(varargin) >= 2 && ischar(varargin{2}), fmt = strtrim(varargin{2}); end
    opts = varargin(3:end);
    hasdelim = any(cellfun(@(a) ischar(a) && strcmpi(a, 'delimiter'), opts));
    if strcmp(fmt, '%s') && ~hasdelim && iscell(varargout{1})
        varargout{1} = varargout{1}(~cellfun(@isempty, varargout{1}));
    end
end
