function s = pad(s, n, varargin)
% PAD  Octave stand-in for MATLAB's pad (R2016b), text padded to a width.
%
%   s = pad(s)      s = pad(s, n)      s = pad(s, n, side)      s = pad(s, n, side, ch)
%
% As MATLAB's: text padded with spaces (or CH) to N characters, on the 'right' (the
% default), 'left' or 'both' sides; a cell array's entries to its longest one unless N
% is given; text longer than N is left as it is.
%
% Part of MRSuite's Octave compatibility shims: Spinach lays out its reports with it.

    side = 'right'; ch = ' ';
    for k = 1:numel(varargin)
        if any(strcmpi(varargin{k}, {'left', 'right', 'both'})), side = lower(varargin{k});
        else, ch = varargin{k}; end
    end
    if iscell(s)
        if nargin < 2 || isempty(n), n = max(cellfun(@numel, s)); end
        s = cellfun(@(x) pad(x, n, side, ch), s, 'UniformOutput', false);
        return
    end
    if nargin < 2, n = numel(s); end
    m = max(0, n - numel(s));
    switch side
        case 'right', s = [s repmat(ch, 1, m)];
        case 'left',  s = [repmat(ch, 1, m) s];
        case 'both',  s = [repmat(ch, 1, floor(m / 2)) s repmat(ch, 1, ceil(m / 2))];
    end
end
