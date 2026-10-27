function varargout = makeValidName(str, varargin)
% MAKEVALIDNAME  MRS4Brain-scoped wrapper of Octave's matlab.lang.makeValidName.
% MATLAB returns a string when given a string. MRS4Brain passes "..." literals
% (strings in MATLAB, char in Octave) and indexes the result as varname{1}; the
% result is therefore handed back as a string stand-in (mrsuite_text), whose {1}
% gives the characters. The name is computed by Octave's own implementation
% (__make_valid_name__, what Octave's makeValidName calls).
    [varargout{1:max(nargout, 1)}] = __make_valid_name__(str, varargin{:});
    if ischar(varargout{1}), varargout{1} = mrsuite_text(varargout{1}); end
end
