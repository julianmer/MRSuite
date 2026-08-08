####################################################################################################
#                                          test_codec.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Round-trip fidelity for FIDACodec. A converter that does not return what it was         #
#          given makes every result downstream of it meaningless, and does so silently, so         #
#          the round trip is the gate on the whole FID-A bridge.                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsuite.bridges.fid_a import FIDACodec


#***********#
#   codec   #
#***********#
@pytest.fixture(scope="module")
def codec():
    """The codec is pure NumPy, so these tests need no Octave."""
    return FIDACodec()


#****************#
#   round trip   #
#****************#
def test_round_trip_preserves_samples(codec, spectrum):
    """nifti -> FID-A -> nifti must return the original samples."""
    back = codec.fida2nifti(codec.nifti2fida(spectrum))

    orig = np.asarray(spectrum[:]).squeeze()
    got = np.asarray(back[:]).squeeze()
    assert got.shape == orig.shape
    assert np.allclose(got, orig, rtol=1e-10, atol=1e-12), \
        f"max deviation {np.max(np.abs(got - orig)):.3e}"


#********************#
#   shape and tags   #
#********************#
def test_round_trip_preserves_shape_and_dim_tags(codec, spectrum):
    """Compares the full shape, not the squeezed one.

    FID-A has no representation for a length-1 coil or average axis: it squeezes them
    and marks the slot 0. A converter that does not restore them silently drops the
    dimension and its tag, which a squeezed comparison would not notice.
    """
    back = codec.fida2nifti(codec.nifti2fida(spectrum))

    assert back.shape == spectrum.shape
    assert list(back.dim_tags) == list(spectrum.dim_tags)


#****************#
#   parameters   #
#****************#
def test_round_trip_preserves_acquisition_parameters(codec, spectrum):
    back = codec.fida2nifti(codec.nifti2fida(spectrum))

    assert np.isclose(back.dwelltime, spectrum.dwelltime, rtol=1e-12)
    assert np.isclose(back.spectrometer_frequency[0],
                      spectrum.spectrometer_frequency[0], rtol=1e-12)
    assert back.nucleus == spectrum.nucleus


#*****************#
#   conjugation   #
#*****************#
def test_conjugation_is_undone(codec, spectrum):
    """nifti2fida conjugates on the way in.

    Forgetting to undo it inverts every imaginary part while leaving the real part and
    all magnitudes intact: invisible to a magnitude check, fatal to phasing.
    """
    back = codec.fida2nifti(codec.nifti2fida(spectrum))

    orig = np.asarray(spectrum[:]).squeeze()
    got = np.asarray(back[:]).squeeze()
    assert np.allclose(got.imag, orig.imag, rtol=1e-10, atol=1e-12)
    assert not np.allclose(got.imag, -orig.imag, atol=1e-12)


#*******************#
#   no provenance   #
#*******************#
def test_rejects_a_struct_without_provenance(codec):
    """A struct built by hand cannot be converted back, and should say so plainly."""
    with pytest.raises(KeyError, match="nii_mrs"):
        codec.fida2nifti({"fids": np.zeros(8), "dims": {"t": 1}})
