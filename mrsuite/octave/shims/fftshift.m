function y = fftshift(x, dim)
% FFTSHIFT  The zero frequency moved to the centre: each dimension's halves swapped.
%
%   y = fftshift(x)          along a vector, or along every dimension of an array
%   y = fftshift(x, dim)     along dimension dim
%
% Only an index permutation, no arithmetic, so the result is Octave's own bit for bit
% (tests/octave/test_shims.py holds it to that). Written for speed: Octave's builds its
% index through a cell and repmat on every call, and Osprey's model evaluation calls
% fftshift on every basis function and every spectrum, some fifty thousand times per
% fit, where that bookkeeping was a tenth of the fit (measured on 2016 dataset 1:
% fftshift 6.6 s and its repmat 4.6 s of 115 s). The common forms, a vector and a
% matrix along one of its two dimensions, index directly.
%
% Part of MRSuite's Octave compatibility shims.

    if ~(isnumeric(x) || islogical(x) || ischar(x))
        error('MRSuite:fftshift', 'fftshift: takes a numeric, logical or char array');
    end
    if nargin < 2
        if isvector(x)
            y = x(halves(numel(x)));
        else
            index = cell(1, ndims(x));
            for d = 1:ndims(x)
                index{d} = halves(size(x, d));
            end
            y = x(index{:});
        end
        return
    end
    if ~(isscalar(dim) && dim == fix(dim) && dim >= 1 && dim <= ndims(x))
        error('MRSuite:fftshift', 'fftshift: dim must be one of the array''s dimensions');
    end
    if ismatrix(x) && dim == 1
        y = x(halves(size(x, 1)), :);
    elseif ismatrix(x)
        y = x(:, halves(size(x, 2)));
    else
        index = cell(1, ndims(x));
        index(:) = {':'};
        index{dim} = halves(size(x, dim));
        y = x(index{:});
    end
end


function k = halves(n)
% The second half of 1:n, then the first, whose length is the larger for odd n.
    h = ceil(n / 2);
    k = [h + 1:n, 1:h];
end
