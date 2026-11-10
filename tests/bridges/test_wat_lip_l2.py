####################################################################################################
#                                        test_wat_lip_l2.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 09/11/26                                                                                #
#                                                                                                  #
# Purpose: Water and lipid removal by L2 regularisation, and the water shift. Synthetic spectra   #
#          on NIfTI-MRS's conventions pin what each step takes out and what it leaves: water or   #
#          lipid gone, NAA kept; an MRSI grid keeps its shape; the refusals come before Octave.    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite.bridges import wat_lip_l2
from tests.conftest import needs_octave

N, DWELL, F0, CENTRE = 1024, 1 / 2000.0, 127.8, 4.7


def _fid(lines, n=N, dwell=DWELL, f0=F0, shift=0.0):
    """Lorentzian lines (ppm, amplitude, damping) on NIfTI-MRS's conventions, the water at
    the carrier, all shifted by ``shift`` ppm."""
    t = np.arange(n) * dwell
    return sum(a * np.exp((2j * np.pi * (ppm + shift - CENTRE) * f0 - d) * t)
               for ppm, a, d in lines)


def _grid(fids, dwell=DWELL, f0=F0):
    """FIDs as one NIfTI-MRS: a single voxel, or a 2 x 2 grid of four."""
    stack = np.array(fids)
    shape = (1, 1, 1, stack.shape[-1]) if stack.ndim == 1 else (2, 2, 1, stack.shape[-1])
    out = gen_nifti_mrs(stack.reshape(shape).astype(np.complex64), dwell, f0, nucleus="1H")
    out.add_hdr_field("SpecFreqChemShift", CENTRE)
    return out


def _band(item, low, high):
    """The summed magnitude of every spectrum between two ppm."""
    data = np.asarray(item[:]).reshape(-1, item.shape[3])
    spectra = np.abs(np.fft.fftshift(np.fft.fft(data, axis=1), axes=1))
    ppm = CENTRE + np.fft.fftshift(np.fft.fftfreq(item.shape[3], item.dwelltime)) / F0
    return spectra[:, (ppm > low) & (ppm < high)].sum(axis=1)


WATER = (CENTRE, 50.0, 15.0)
NAA = (2.01, 1.0, 8.0)
LIPID = (1.3, 10.0, 60.0)


#*************#
#   refused   #
#*************#
def test_coils_and_transients_are_refused_before_octave_is_needed():
    coils = gen_nifti_mrs(np.ones((1, 1, 1, 64, 2), complex), DWELL, F0,
                          dim_tags=["DIM_COIL", None, None])
    with pytest.raises(ValueError, match="DIM_COIL"):
        wat_lip_l2.water_removal(coils)
    with pytest.raises(ValueError, match="DIM_COIL"):
        wat_lip_l2.water_shift(coils, coils)
    odd = _grid(_fid([WATER], n=N + 1))
    with pytest.raises(ValueError, match="even"):
        wat_lip_l2.water_shift(odd, odd)


#***********#
#   water   #
#***********#
@needs_octave
def test_water_goes_and_naa_stays():
    """What is left at NAA is compared with NAA alone, since the water's tail reaches it."""
    data, naa = _grid(_fid([WATER, NAA])), _grid(_fid([NAA]))

    out = wat_lip_l2.water_removal(data)

    assert out.shape == data.shape
    assert _band(out, 4.4, 5.0)[0] < 0.05 * _band(data, 4.4, 5.0)[0]
    assert _band(out, 1.9, 2.1)[0] == pytest.approx(_band(naa, 1.9, 2.1)[0], rel=0.1)


@needs_octave
def test_the_same_seed_repeats_and_another_differs():
    data = _grid(_fid([WATER, NAA]))

    one, same = wat_lip_l2.water_removal(data, seed=1), wat_lip_l2.water_removal(data, seed=1)
    other = wat_lip_l2.water_removal(data, seed=2)

    assert np.allclose(np.asarray(one[:]), np.asarray(same[:]))
    assert not np.allclose(np.asarray(one[:]), np.asarray(other[:]))


#***********#
#   lipid   #
#***********#
@needs_octave
def test_lipid_goes_from_every_voxel_of_a_grid_and_naa_stays():
    """A 2 x 2 grid with a broad lipid at 1.3 ppm: the band given in ppm reaches the lipid
    in the toolbox's frame, where positive Hz run upfield."""
    data = _grid([_fid([NAA, LIPID]), _fid([NAA, LIPID]), _fid([NAA]), _fid([NAA, LIPID])])
    naa = _grid([_fid([NAA])] * 4)

    out = wat_lip_l2.lipid_removal(data, band=(0.9, 1.7))

    assert out.shape == (2, 2, 1, N)
    with_lipid = [0, 1, 3]
    assert (_band(out, 1.0, 1.6)[with_lipid] < 0.1 * _band(data, 1.0, 1.6)[with_lipid]).all()
    assert np.allclose(_band(out, 1.9, 2.1), _band(naa, 1.9, 2.1), rtol=0.15)


#***********#
#   shift   #
#***********#
@needs_octave
def test_the_water_shift_puts_each_voxels_water_on_the_carrier():
    """Four voxels, three of them off by up to 0.15 ppm, the reference's water likewise:
    after the shift every voxel's NAA sits at 2.01 ppm."""
    shifts = (0.0, 0.1, -0.15, 0.05)
    data = _grid([_fid([NAA], shift=s) for s in shifts])
    reference = _grid([_fid([WATER], shift=s) for s in shifts])

    out = wat_lip_l2.water_shift(data, reference)

    spectra = np.abs(np.fft.fftshift(np.fft.fft(np.asarray(out[:]).reshape(-1, N), axis=1), axes=1))
    ppm = CENTRE + np.fft.fftshift(np.fft.fftfreq(N, DWELL)) / F0
    assert np.allclose(ppm[spectra.argmax(axis=1)], 2.01, atol=0.02)
    with pytest.raises(ValueError, match="reference's grid"):
        wat_lip_l2.water_shift(data, _grid(_fid([WATER])))
