function out = mrsuite_water_shift(fid, reference)
% MRSUITE_WATER_SHIFT  Every voxel shifted so its water, found on the reference, is at the carrier.
%
%   out = mrsuite_water_shift(fid, reference)
%
% FID and REFERENCE are complex (x, y, z, points), as NIfTI-MRS holds them, the reference an
% unsuppressed acquisition of the same voxels. The toolbox's watershift works slice by
% slice on spectra in its frame, flip(fftshift(fft)), and shifts a voxel only within a tenth
% of the bandwidth; the shifted spectra go back to FIDs the same way.
%
% MRSuite's driver code for Wat_Lip_Removal_L2, on the path under every runtime.

    n = size(fid, 4);
    spectra = flip(fftshift(fft(permute(fid, [1 2 4 3]), n, 3), 3), 3);
    water = flip(fftshift(fft(permute(reference, [1 2 4 3]), n, 3), 3), 3);
    for z = 1:size(spectra, 4)
        [~, spectra(:, :, :, z)] = watershift(water(:, :, :, z), spectra(:, :, :, z));
    end
    out = permute(ifft(ifftshift(flip(spectra, 3), 3), [], 3), [1 2 4 3]);
end
