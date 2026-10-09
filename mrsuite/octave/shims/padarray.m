function B = padarray(A, padsize, varargin)
% PADARRAY  Octave stand-in for MATLAB's padarray (Image Processing Toolbox).
%
%   B = padarray(A, padsize)
%   B = padarray(A, padsize, padval)
%   B = padarray(A, padsize, direction)
%   B = padarray(A, padsize, padval, direction)
%
% A with padsize(k) copies of padval (0 by default) added along dimension k: before it
% ('pre'), after it ('post') or on both sides ('both', the default). Constant padding
% only; the 'circular', 'replicate', 'symmetric' and 'reflect' patterns are refused.
%
% Part of MRSuite's Octave compatibility shims: FID-A's op_zeropad zero-fills with it,
% and so op_autophase, op_ppmref and op_getLW, and Octave has it only in its image
% package, which MRSuite does not load.

    if ~isnumeric(padsize) || ~isvector(padsize) || any(padsize < 0) || any(padsize ~= fix(padsize))
        error('padarray: PADSIZE must be a vector of non-negative integers');
    end

    padval = 0;
    direction = 'both';
    for k = 1:numel(varargin)
        v = varargin{k};
        if ischar(v) && any(strcmpi(v, {'pre', 'post', 'both'}))
            direction = lower(v);
        elseif ischar(v) && strcmpi(v, 'zeros')
            padval = 0;
        elseif ~ischar(v) && isscalar(v)
            padval = v;
        else
            error('padarray: MRSuite''s shim pads with a constant value only');
        end
    end

    n = max(numel(padsize), ndims(A));
    sz = size(A);
    sz(end + 1:n) = 1;
    pad = padsize(:).';
    pad(end + 1:n) = 0;
    before = pad * any(strcmp(direction, {'pre', 'both'}));
    after = pad * any(strcmp(direction, {'post', 'both'}));

    B = repmat(cast(padval, class(A)), sz + before + after);
    at = arrayfun(@(d) before(d) + (1:sz(d)), 1:n, 'UniformOutput', false);
    B(at{:}) = A;
end
