function y = fftshift(x, dim)
% FFTSHIFT  Octave's fftshift without its per-call bookkeeping.
%
%   y = fftshift(x)
%   y = fftshift(x, dim)
%
% The same indexing as Octave's own, which only moves data and does no arithmetic, so
% the result is the same bit for bit. Octave's builds its index through a cell and
% repmat on every call; Osprey's model evaluation calls fftshift on every basis function
% and every spectrum, some fifty thousand times per fit, where that bookkeeping was a
% tenth of the fit (measured on 2016 dataset 1: fftshift 6.6 s and its repmat 4.6 s of
% 115 s).
%
% Part of MRSuite's Octave compatibility shims.

    % Octave's own checks, so what it refuses is still refused
    if ~(isnumeric(x) || islogical(x) || ischar(x))
        error('fftshift: X must be a vector or matrix');
    end
    if nargin == 2
        if ~(isscalar(dim) && dim > 0 && dim == fix(dim))
            error('fftshift: dimension DIM must be a positive integer');
        end
        sz = size(x);
        n = sz(dim);
        h = ceil(n / 2);
        k = [h+1:n, 1:h];
        if ndims(x) == 2 && dim == 1
            y = x(k, :);
        elseif ndims(x) == 2 && dim == 2
            y = x(:, k);
        else
            idx = repmat({':'}, ndims(x), 1);
            idx{dim} = k;
            y = x(idx{:});
        end
    elseif isvector(x)
        n = length(x);
        h = ceil(n / 2);
        y = x([h+1:n, 1:h]);
    else
        sz = size(x);
        idx = cell(1, numel(sz));
        for i = 1:numel(sz)
            h = ceil(sz(i) / 2);
            idx{i} = [h+1:sz(i), 1:h];
        end
        y = x(idx{:});
    end
end
