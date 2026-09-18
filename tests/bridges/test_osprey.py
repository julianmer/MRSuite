####################################################################################################
#                                         test_osprey.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: The Osprey bridge, and the dialect its bundled FID-A has drifted into. The              #
#          dialect is pinned against the data rather than against a loader: every one of           #
#          its differences fails silently, as a mirrored or shifted spectrum.                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#

import os
import re
from pathlib import Path

import numpy as np
import pytest

from mrsuite.bridges.fid_a import FIDACodec
from mrsuite.bridges.osprey import OSPREY_ROOT, OspreyCodec
from mrsuite.core.result import FitResult

from tests.conftest import CHALLENGE_BASIS, needs_octave

BASIS = (OSPREY_ROOT / "fit" / "basissets" / "3T" / "siemens" / "unedited"
         / "press" / "30" / "basis_siemens_press30.mat")


#***********#
#   codec   #
#***********#
@pytest.fixture(scope="module")
def codec():
    """Pure NumPy, so the dialect can be checked without starting Octave."""
    return OspreyCodec()


#*********#
#   naa   #
#*********#
def test_the_dialect_puts_naa_where_naa_is(codec, spectrum):
    """The check that decides the whole dialect.

    Conjugation, transform direction and axis direction are three switches that only
    agree in one combination. Get any of them wrong and the spectrum is mirrored, which
    no type check catches: NAA lands near 7.3 ppm instead of 2.01.
    """
    struct = codec.nifti2fida(spectrum)
    ppm = np.asarray(struct["ppm"]).ravel()
    spec = np.asarray(struct["specs"]).ravel()

    band = (ppm > 0.5) & (ppm < 4.2)
    assert abs(ppm[band][np.argmax(np.abs(spec[band]))] - 2.01) < 0.1


#**********#
#   axis   #
#**********#
def test_the_ppm_axis_ascends_from_centre_freq(codec, spectrum):
    """Osprey's formula, against upstream's descending -f/(Bo*42.577) + 4.65."""
    ppm = np.asarray(codec.nifti2fida(spectrum)["ppm"]).ravel()

    assert ppm[0] < ppm[-1], "Osprey's axis ascends"
    assert abs((ppm[0] + ppm[-1]) / 2 - OspreyCodec.CENTRE_PPM) < 1e-6


#***************#
#   transform   #
#***************#
def test_specs_use_fft_not_ifft(codec, spectrum):
    struct = codec.nifti2fida(spectrum)
    fids = np.asarray(struct["fids"]).ravel()

    assert np.allclose(np.asarray(struct["specs"]).ravel(),
                       np.fft.fftshift(np.fft.fft(fids)))


#***************#
#   conjugate   #
#***************#
def test_the_fids_are_not_conjugated(codec, spectrum):
    """Upstream conjugates on load and Osprey does not, so the two dialects hold
    mirror-image FIDs for the same spectrum."""
    ours = np.asarray(codec.nifti2fida(spectrum)["fids"]).ravel()
    upstream = np.asarray(FIDACodec().nifti2fida(spectrum)["fids"]).ravel()

    assert np.allclose(ours, np.conj(upstream))
    assert np.allclose(ours, np.asarray(spectrum[:]).ravel())


#***********#
#   units   #
#***********#
def test_times_are_in_milliseconds(codec, spectrum):
    """Osprey works in ms and the shipped basis sets record te as 30, not 0.03."""
    upstream = FIDACodec().nifti2fida(spectrum)
    ours = codec.nifti2fida(spectrum)

    assert ours["te"] == pytest.approx(float(upstream["te"]) * 1e3)
    assert ours["tr"] == pytest.approx(float(upstream["tr"]) * 1e3)


#****************#
#   round trip   #
#****************#
def test_round_trip_preserves_samples(codec, spectrum):
    back = codec.fida2nifti(codec.nifti2fida(spectrum))

    assert np.allclose(np.asarray(back[:]).squeeze(),
                       np.asarray(spectrum[:]).squeeze(), rtol=1e-10, atol=1e-12)


#****************#
#   provenance   #
#****************#
def test_the_struct_carries_what_osprey_reads_unguarded(codec, spectrum):
    """Every Osprey step records provenance and reads these without checking, so a
    missing one surfaces as an error from deep inside the fit."""
    struct = codec.nifti2fida(spectrum)

    assert "ProcessingSoftwareVersion" in struct["nii_mrs"]["hdr_ext"]
    assert struct["flags"]["isUnEdited"] == 1
    assert struct["names"] == ("A",)


#*****************#
#   fit options   #
#*****************#
def test_fit_options_fill_what_fit_runFit_leaves_out():
    """fit_runFit sets three fields; fit_Osprey reads more, normally from a job file."""
    from mrsuite.bridges.osprey import bridge

    opts = bridge.fit_options()
    assert set(opts) >= {"range", "rangeWater", "bLineKnotSpace", "GAP"}
    assert bridge.fit_options(range=[0.5, 4.0])["range"] == [0.5, 4.0]


#***************#
#   discovery   #
#***************#
@needs_octave
def test_the_whole_toolbox_is_reachable():
    from mrsuite.bridges.osprey import bridge

    names = bridge.available()
    assert len(names) > 400, f"only {len(names)} discovered"
    for expected in ("fit_runFit", "osp_fitInitialise", "op_zeropad",
                     "osp_recalculate_basis_specs"):
        assert expected in names


#***********#
#   basis   #
#***********#
@needs_octave
@pytest.mark.skipif(not BASIS.exists(), reason="osprey submodule not checked out")
def test_loading_a_basis_adds_the_spectra_and_axis():
    """The shipped .mat holds only FIDs; fit_resampleBasis reads a ppm axis that
    osp_recalculate_basis_specs has to add first."""
    from mrsuite.bridges.osprey import bridge

    basis = bridge.basis(str(BASIS))
    assert "ppm" in basis and "specs" in basis
    assert np.asarray(basis["fids"]).shape[1] == int(basis["nMets"]) + int(basis["nMM"])


#*******************#
#   lcmodel basis   #
#*******************#
needs_challenge_basis = pytest.mark.skipif(not os.path.exists(CHALLENGE_BASIS),
                                           reason="challenge submodule not checked out")


def peak_ppm(basis, name, lo=1.8, hi=2.2):
    """Where a basis function peaks, on the container's own ppm axis."""
    ppm = np.asarray(basis["ppm"]).ravel()
    names = [str(n) for n in np.asarray(basis["name"]).ravel()]
    spec = np.real(np.asarray(basis["specs"])[:, names.index(name)])
    band = (ppm > lo) & (ppm < hi)
    return ppm[band][np.argmax(spec[band])]


@needs_octave
@needs_challenge_basis
def test_a_basis_without_ppmsep_is_read_at_the_given_centre(tmp_path):
    """MakeBasis states the centre as PPMSEP in a $NMUSED block, and without one
    fit_makeBasis puts it at 3 ppm, 1.65 ppm away from the data. Written outside the
    repo on purpose: the container sees only the copy that is read."""
    from mrsuite.bridges.osprey import bridge

    text = re.sub(r"^[ \t]*\$NMUSED.*?\$END[ \t]*\n", "", Path(CHALLENGE_BASIS).read_text(),
                  flags=re.S | re.M)
    assert "PPMSEP" not in text
    bare = tmp_path / "no_ppmsep.BASIS"
    bare.write_text(text)

    stated = bridge.basis_from_lcmodel(CHALLENGE_BASIS, add_mm=False)
    missing = bridge.basis_from_lcmodel(str(bare), add_mm=False, centre_ppm=4.65)
    for basis in (stated, missing):
        ppm = np.asarray(basis["ppm"]).ravel()
        assert abs(ppm[len(ppm) // 2] - 4.65) < 0.01
        assert abs(peak_ppm(basis, "NAA") - 2.01) < 0.02


def basis_blocks(path):
    """The spectrum every $BASIS block of a .BASIS file holds, by metabolite."""
    blocks = {}
    for m in re.finditer(r"\$BASIS\s(.*?)\$END(.*?)(?=\$|\Z)", Path(path).read_text(), re.S):
        name = re.search(r"METABO\s*=\s*'([^']*)'", m.group(1)).group(1).strip()
        values = np.array(m.group(2).split(), dtype=float)
        blocks[name] = values[0::2] + 1j * values[1::2]
    return blocks


@needs_octave
@needs_challenge_basis
def test_a_basis_is_read_without_a_frequency_shift():
    """A .BASIS block is the FFT of the FID in LCModel's orientation, the conjugate of
    NIfTI-MRS'. io_readlcmraw_basis reorders it with flipud(fftshift(conj(.))), which
    for an even length lands one bin off: every function 0.98 Hz low here, since
    fit_makeBasis' op_dccorr changes only the first point."""
    from mrsuite.bridges.osprey import bridge

    basis = bridge.basis_from_lcmodel(CHALLENGE_BASIS, add_mm=False)
    names = [str(n) for n in np.asarray(basis["name"]).ravel()]
    fids = np.asarray(basis["fids"])
    for name, spec in basis_blocks(CHALLENGE_BASIS).items():
        want = np.conj(np.fft.ifft(spec))[1:]
        got = fids[1:, names.index(name)]
        scale = np.vdot(want, got) / np.vdot(want, want)
        assert np.linalg.norm(got - scale * want) / np.linalg.norm(got) < 1e-6, name


#*********#
#   fit   #
#*********#
@needs_octave
@pytest.mark.skipif(not BASIS.exists(), reason="osprey submodule not checked out")
def test_a_fit_runs_end_to_end(spectrum):
    """Osprey needs no fit adapter: the basis is an ordinary argument, so the whole
    toolbox including fit_runFit is reachable by dispatch.

    Asserted on the ordering rather than on values: a mirrored or misreferenced
    spectrum still fits, and still returns finite numbers, but not with NAA on top.
    """
    from mrsuite.bridges.osprey import bridge

    basis = bridge.basis(str(BASIS))
    params, res_basis = bridge.fit_runFit(spectrum, basis, "Osprey",
                                          bridge.fit_options(), nout=2)

    ampl = np.asarray(params["ampl"]).ravel()
    names = [str(n) for n in np.asarray(res_basis["name"]).ravel()]
    assert len(ampl) == len(names) and np.all(np.isfinite(ampl))

    ranked = [n for n, _ in sorted(zip(names, ampl), key=lambda p: -p[1])]
    assert ranked[0] == "NAA", f"NAA should dominate, got {ranked[:4]}"
    assert set(ranked[:6]) & {"Cr", "PCr", "Glu", "mI"}, f"unphysiological: {ranked[:6]}"


#***************#
#   to result   #
#***************#
@needs_octave
@pytest.mark.skipif(not BASIS.exists(), reason="osprey submodule not checked out")
def test_a_fit_becomes_a_uniform_result(spectrum):
    """What lets Osprey be compared against FSL-MRS and LCModel at all: without this
    its output is a MATLAB struct that nothing downstream can read."""
    from mrsuite.bridges import osprey

    out = osprey.fit(spectrum, osprey.bridge.basis(str(BASIS)))

    assert isinstance(out, FitResult) and out.method == "osprey:Osprey"
    tcr = out.conc("internal")
    assert 1.0 < tcr["NAA"] < 2.0, f"NAA/tCr = {tcr['NAA']}, physiologically ~1.5"
    assert out.qc["fwhm"] > 0


#***********#
#   scale   #
#***********#
@needs_octave
@pytest.mark.skipif(not BASIS.exists(), reason="osprey submodule not checked out")
def test_the_fit_does_not_depend_on_the_data_scale(spectrum):
    """osp_fitInitialise divides the data by max(real(data)) / max(real(basis)) before
    fitting, and the soft priors are weighed against the unnormalised residual: without
    that step a spectrum and the same spectrum times 1024 fit differently.

    A power of two, so the scaled data are the same bits: this spectrum sits on a knife
    edge, and a factor of 1000 moves the fit through rounding alone (1e-16 relative).
    """
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    from mrsuite.bridges import osprey

    basis = osprey.bridge.basis(str(BASIS))
    louder = gen_nifti_mrs(np.asarray(spectrum[:]) * 1024.0, spectrum.dwelltime,
                           spectrum.spectrometer_frequency[0], nucleus=spectrum.nucleus[0])
    quiet, loud = osprey.fit(spectrum, basis), osprey.fit(louder, basis)

    for name, value in quiet.conc("internal").items():
        assert loud.conc("internal")[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name
    assert loud.conc("raw")["NAA"] == pytest.approx(1024 * quiet.conc("raw")["NAA"], rel=1e-9)


#*****************#
#   referencing   #
#*****************#
def test_concentrations_are_referenced_to_total_creatine():
    """Osprey leaves referencing to its quantify stage, so the adapter does what
    OspreyQuantify does: divide by Cr + PCr."""
    from mrsuite.bridges.osprey import to_result

    out = to_result({"ampl": [[6.0], [3.0], [1.0]]},
                    {"name": ["NAA", "Cr", "PCr"]})

    assert out.conc("raw")["NAA"] == 6.0
    assert out.conc("internal")["NAA"] == pytest.approx(6.0 / 4.0)


#*****************#
#   no creatine   #
#*****************#
def test_without_creatine_only_raw_is_reported():
    """A fit that collapsed tCr to zero cannot be referenced, and saying so beats
    dividing by it - that is what turned three challenge datasets into 1e15."""
    from mrsuite.bridges.osprey import to_result

    out = to_result({"ampl": [[6.0]]}, {"name": ["NAA"]})

    assert "raw" in out.concentrations
    assert "internal" not in out.concentrations
