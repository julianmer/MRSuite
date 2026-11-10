function out = mrsuite_l2_removal(fid, bandwidth, kind, band, damping, components, beta, ...
                                  echo_position, seed)
% MRSUITE_L2_REMOVAL  Water or lipid signal taken out of spectra by the toolbox's L2 regularisation.
%
%   out = mrsuite_l2_removal(fid, bandwidth, kind, band, damping, components, beta, ...
%                             echo_position, seed)
%
% FID is complex (x, y, z, points), as NIfTI-MRS holds it; BANDWIDTH in Hz. KIND is 'water'
% or 'lipid': the basis is the toolbox's synthesised one, generator_water over +-BAND(1) Hz
% about the carrier, or generator_lipid over BAND(1)..BAND(2) Hz in the toolbox's frame,
% where positive Hz run upfield (ppm = centre - Hz / f0). DAMPING is its [damp, damp_range],
% COMPONENTS the number of random components, BETA its regularisation exponent (10^-beta),
% ECHO_POSITION its echoposition (0 for an FID from the echo top), SEED the random seed, so a
% call repeats.
%
% The spectra are made as the toolbox's script makes them, flip(fftshift(fft)) along the
% spectral axis with no zero-filling, and handed over as its single-voxel case, one column
% per voxel (its grid cases drop a trailing singleton dimension for one voxel or one slice);
% the result goes back to FIDs the same way.
%
% MRSuite's driver code for Wat_Lip_Removal_L2, on the path under every runtime.

    shape = size(fid);
    n = shape(4);
    spectra = flip(fftshift(fft(fid, n, 4), 4), 4);
    spectra = reshape(permute(spectra, [4 1 2 3]), n, []);
    rng(seed);
    if strcmp(kind, 'water')
        basis = generator_water(bandwidth, n, n, band(1), damping(1), damping(2), components, ...
                                echo_position);
    else
        basis = generator_lipid(bandwidth, n, n, band(1), band(2), damping(1), damping(2), ...
                                components, echo_position);
    end
    spectra = watersup_sim(spectra, real(basis), beta);
    spectra = reshape(spectra, [n shape(1:3)]);
    out = ifft(ifftshift(flip(permute(spectra, [2 3 4 1]), 4), 4), [], 4);
end
