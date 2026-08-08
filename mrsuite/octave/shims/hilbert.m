function y = hilbert(x, n)
% HILBERT  Octave stand-in for MATLAB's hilbert (Signal Processing Toolbox).
%
%   y = hilbert(x)
%   y = hilbert(x, n)
%
% Analytic signal of a real sequence: the input as the real part, its Hilbert
% transform as the imaginary part. Columns are transformed independently, and a row
% vector is treated as a single signal, matching MATLAB.
%
% Follows MATLAB's documented four steps: FFT the sequence, keep DC and Nyquist, double
% the positive frequencies and zero the negative ones, then inverse FFT.
%
% Part of MRSuite's Octave compatibility shims. Octave ships hilbert only in the forge
% `signal` package, and Osprey's spline baseline (fit_makeSplineBasis) and multiplex
% model call it on every fit, so the fit path cannot run without it.

    if ~isreal(x)
        x = real(x);
    end

    [x, nshifts] = shiftdim(x);          % work along the first non-singleton dimension
    if nargin < 2 || isempty(n)
        n = size(x, 1);
    end
    if n == 0
        y = shiftdim(zeros(0, size(x, 2)), -nshifts);
        return;
    end

    h = zeros(n, 1);
    if mod(n, 2) == 0
        h([1, n / 2 + 1]) = 1;           % DC and Nyquist
        h(2:n / 2) = 2;                  % positive frequencies
    else
        h(1) = 1;
        h(2:(n + 1) / 2) = 2;
    end

    y = ifft(fft(x, n, 1) .* h, [], 1);
    y = shiftdim(y, -nshifts);
end
