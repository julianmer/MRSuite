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
def test_naa_sits_where_naa_is_on_fid_as_axis(codec, spectrum):
    """The struct describes the spectrum, not its mirror: on FID-A's own descending axis
    with its ifft, the challenge spectrum's largest peak is NAA's, at 2.01 ppm. One
    conjugation too many puts it near 7.3 ppm, which FID-A's phasing and linewidth on
    creatine then miss."""
    struct = codec.nifti2fida(spectrum)
    ppm = np.asarray(struct["ppm"]).ravel()
    spec = np.asarray(struct["specs"]).ravel()

    band = (ppm > 0.5) & (ppm < 4.2)
    assert abs(ppm[band][np.argmax(np.abs(spec[band]))] - 2.01) < 0.1


def test_conjugation_is_undone(codec, spectrum):
    """The samples come back as they went in, imaginary parts and all: a conjugation
    on one way and not the other inverts every imaginary part while leaving the real part
    and all magnitudes intact, invisible to a magnitude check, fatal to phasing."""
    back = codec.fida2nifti(codec.nifti2fida(spectrum))

    orig = np.asarray(spectrum[:]).squeeze()
    got = np.asarray(back[:]).squeeze()
    assert np.allclose(got.imag, orig.imag, rtol=1e-10, atol=1e-12)
    assert not np.allclose(got.imag, -orig.imag, atol=1e-12)


def test_single_precision_reaches_fid_a_as_double(codec, spectrum):
    """FID-A's own loaders give MATLAB doubles, and Octave solves no sparse system in single
    precision (Osprey's op_robustSpecReg, its whittaker): data stored as complex64, as Twix
    read by spec2nii is, arrive as complex128."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs

    single = gen_nifti_mrs(np.asarray(spectrum[:]).astype(np.complex64), spectrum.dwelltime,
                           spectrum.spectrometer_frequency[0])
    struct = codec.nifti2fida(single)

    assert np.asarray(struct["fids"]).dtype == np.complex128
    assert np.asarray(struct["specs"]).dtype == np.complex128


#*******************#
#   no provenance   #
#*******************#
def test_rejects_a_struct_without_provenance(codec):
    """A struct built by hand cannot be converted back, and should say so plainly."""
    with pytest.raises(KeyError, match="nii_mrs"):
        codec.fida2nifti({"fids": np.zeros(8), "dims": {"t": 1}})
