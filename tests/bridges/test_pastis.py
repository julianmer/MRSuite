####################################################################################################
#                                          test_pastis.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 09/11/26                                                                                #
#                                                                                                  #
# Purpose: PASTIS's fit and processing steps. The result mapping is asserted on arrays, the        #
#          refusals happen before its repository is fetched, a spectrum made of its own basis      #
#          functions comes back with their amplitudes, and its transient steps keep NIfTI-MRS's    #
#          dimensions: coils combined away, transients realigned or rejected.                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite.bridges import pastis
from mrsuite.core.basis import Basis
from mrsuite.core.result import FitResult
from mrsuite.simulators import bases
from tests.simulators.test_bases import needs_pastis, needs_pygamma


#*************#
#   mapping   #
#*************#
def test_result_maps_the_parameters():
    """Amplitudes land raw and over the reference metabolite, the bounds raw and as a
    share; a quality measure the spectrum could not give is left out."""
    names = ["NAA", "Cr_CH3", "Water"]
    params = np.array([[10.0, 8.0, 0.5, 0.01], [5.0, 7.0, 0.2, 0.01], [0.0, 8.0, 0.0, 0.0]])
    bounds = np.array([[0.1, 1.0, 0.1, 0.01], [0.2, 1.0, 0.1, 0.01], [0.0, 0.0, 0.0, 0.0]])

    out = pastis._result(names, ["NAA", "Cr_CH3"], "NAA", params, bounds,
                         {"r2_time": 0.9, "fqn": float("nan")})

    assert isinstance(out, FitResult) and out.method == "pastis:fit_pastis"
    assert out.metabolites == ["NAA", "Cr_CH3"]
    assert out.conc("raw") == {"NAA": 10.0, "Cr_CH3": 5.0}
    assert out.conc("internal")["Cr_CH3"] == 0.5
    assert out.crlb("raw")["NAA"] == 0.1 and out.crlb("percentage")["NAA"] == pytest.approx(1.0)
    assert out.qc == {"r2_time": 0.9}
    assert out.raw["columns"] == ("amplitude", "damping", "shift", "phase")


def test_non_finite_amplitudes_are_refused():
    with pytest.raises(ValueError, match="non-finite"):
        pastis._result(["NAA"], ["NAA"], "NAA", np.array([[np.nan, 1.0, 0.0, 0.0]]),
                       np.zeros((1, 4)), {})


#*************#
#   refused   #
#*************#
def test_what_cannot_be_fitted_is_refused_before_the_library_is_needed():
    coils = gen_nifti_mrs(np.ones((1, 1, 1, 64, 2), complex), 1 / 2000, 123.2,
                          dim_tags=["DIM_COIL", None, None])
    one = gen_nifti_mrs(np.ones((1, 1, 1, 64), complex), 1 / 2000, 123.2)
    foreign = Basis(["NAA"], np.ones((1, 64), complex), 2000.0, 123.2)

    with pytest.raises(ValueError, match="single spectra"):
        pastis.fit(coils, foreign)
    with pytest.raises(ValueError, match="bases.pastis"):
        pastis.fit(one, foreign)


#************#
#   signal   #
#************#
def _transients(fid, dwell, f0, count=8, coils=0, shifted=(), shift=0.8):
    """``count`` noisy copies of a FID as NIfTI-MRS transients, some shifted by ``shift``
    ppm, with ``coils`` receive channels of differing phase when asked."""
    rng = np.random.default_rng(0)
    t = np.arange(fid.size) * dwell
    stack = []
    for k in range(count):
        one = fid * np.exp(2j * np.pi * shift * f0 * t) if k in shifted else fid
        stack.append(one + 1e-3 * np.abs(fid).max() * (rng.normal(size=fid.size)
                                                        + 1j * rng.normal(size=fid.size)))
    data = np.array(stack).T                                            # (points, transients)
    tags = ["DIM_DYN", None, None]
    if coils:
        data = data[:, :, None] * np.exp(1j * np.linspace(0, 1, coils))[None, None, :]
        tags = ["DIM_DYN", "DIM_COIL", None]
    out = gen_nifti_mrs(data.reshape(1, 1, 1, *data.shape).astype(np.complex64), dwell, f0,
                        nucleus="1H", dim_tags=tags)
    out.add_hdr_field("RepetitionTime", 2.0)
    return out


def _singlets(n, dwell, f0, lines=((4.7, 5.0), (2.01, 1.0)), damping=40.0, centre=4.7):
    t = np.arange(n) * dwell
    return sum(a * np.exp((2j * np.pi * (ppm - centre) * f0 - damping) * t) for ppm, a in lines)


#*********#
#   fit   #
#*********#
@needs_pastis
@needs_pygamma
def test_a_fit_returns_the_amplitudes_of_its_own_basis_functions():
    """NAA and creatine from PASTIS's own basis, 2:1 with a common damping, on the
    NIfTI-MRS conventions: PASTIS's model is these functions times an amplitude, so the
    amplitudes come back as given, and the creatine over the reference NAA as a half."""
    n, dwell, f0 = 2048, 1 / 2500.0, 127.7
    basis = bases.pastis(30.0, "press", frequency=f0, points=n, bandwidth=1 / dwell)
    t = np.arange(n) * dwell
    naa, cr = (basis.fids[basis.names.index(name)] for name in ("NAA", "Cr_CH3"))
    fid = (2.0 * naa + 1.0 * cr) * np.exp(-8.0 * t)
    rng = np.random.default_rng(0)
    fid += 1e-3 * np.abs(fid).max() * (rng.normal(size=n) + 1j * rng.normal(size=n))
    data = gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), dwell, f0, nucleus="1H")

    out = pastis.fit(data, basis, metabolites=["NAA", "Cr_CH3"])

    assert out.conc("raw")["NAA"] == pytest.approx(2.0, rel=0.05)
    assert out.conc("internal")["Cr_CH3"] == pytest.approx(0.5, rel=0.05)
    assert 0.0 < out.crlb("percentage")["NAA"] < 10.0
    assert 0.9 < out.qc["r2_time"] <= 1.0
    assert out.raw["parameters"][basis.names.index("NAA"), 1] == pytest.approx(8.0, rel=0.1)


@needs_pastis
@needs_pygamma
def test_coupled_metabolites_are_fitted_without_the_reference_among_them():
    """PASTIS integrates the reference metabolite's singlet before it fits, so a fit of
    glutamate and myo-inositol alone still runs; without the reference fitted there is no
    internal scaling."""
    n, dwell, f0 = 2048, 1 / 2500.0, 127.7
    basis = bases.pastis(30.0, "press", frequency=f0, points=n, bandwidth=1 / dwell)
    t = np.arange(n) * dwell
    glu, ins = (basis.fids[basis.names.index(name)] for name in ("Glu", "mI"))
    fid = (3.0 * glu + 2.0 * ins) * np.exp(-8.0 * t)
    data = gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), dwell, f0, nucleus="1H")

    out = pastis.fit(data, basis, metabolites=["Glu", "mI"])

    assert out.conc("raw")["Glu"] == pytest.approx(3.0, rel=0.05)
    assert out.conc("raw")["mI"] == pytest.approx(2.0, rel=0.05)
    assert "internal" not in out.concentrations


@needs_pastis
@needs_pygamma
def test_a_31p_basis_is_fitted_though_its_template_has_no_water():
    n, dwell, f0 = 1024, 1 / 5000.0, 120.3
    basis = bases.pastis(1.0, "fid", frequency=f0, points=n, bandwidth=1 / dwell, nucleus="31P")
    first, second = basis.names[:2]
    t = np.arange(n) * dwell
    fid = (2.0 * basis.fids[0] + basis.fids[1]) * np.exp(-8.0 * t)
    data = gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), dwell, f0, nucleus="31P")

    out = pastis.fit(data, basis, metabolites=[first, second])

    assert out.conc("raw")[first] == pytest.approx(2.0, rel=0.05)
    assert out.conc("raw")[second] == pytest.approx(1.0, rel=0.05)


@needs_pastis
@needs_pygamma
def test_a_metabolite_or_option_pastis_lacks_is_refused():
    n, dwell, f0 = 1024, 1 / 2500.0, 127.7
    basis = bases.pastis(30.0, "press", frequency=f0, points=n, bandwidth=1 / dwell)
    data = gen_nifti_mrs(np.ones((1, 1, 1, n), np.complex64), dwell, f0, nucleus="1H")

    with pytest.raises(ValueError, match=r"not in PASTIS's template 'Brain_7T': \['tCr'\]"):
        pastis.fit(data, basis, metabolites=["tCr"])
    with pytest.raises(ValueError, match="not a setting of PASTIS's fit tool"):
        pastis.fit(data, basis, opts={"baseline": 2})


#***********#
#   steps   #
#***********#
@needs_pastis
def test_coils_are_combined_away_and_the_transients_kept():
    n, dwell, f0 = 2048, 1 / 2500.0, 127.7
    data = _transients(_singlets(n, dwell, f0), dwell, f0, count=4, coils=2)

    out = pastis.combine(data)

    assert out.shape == (1, 1, 1, n, 4) and list(out.dim_tags)[0] == "DIM_DYN"
    peak = np.abs(np.fft.fft(np.asarray(out[:])[0, 0, 0, :, 0]))
    assert peak.argmax() == np.abs(np.fft.fft(np.asarray(data[:])[0, 0, 0, :, 0, 0])).argmax()


@needs_pastis
def test_a_spectrum_without_coils_is_not_combined():
    n, dwell, f0 = 1024, 1 / 2500.0, 127.7
    data = _transients(_singlets(n, dwell, f0), dwell, f0, count=4)

    assert pastis.combine(data) is data


@needs_pastis
def test_phasing_keeps_every_dimension():
    n, dwell, f0 = 2048, 1 / 2500.0, 127.7
    data = _transients(_singlets(n, dwell, f0) * np.exp(0.5j), dwell, f0, count=4, coils=2)

    out = pastis.phase(data)

    assert out.shape == data.shape and list(out.dim_tags)[:2] == ["DIM_DYN", "DIM_COIL"]
    first = np.asarray(out[:])[0, 0, 0, 0, :, 0]
    assert np.abs(np.angle(first)).max() < 0.1


@needs_pastis
def test_realigning_takes_the_shift_out_of_the_shifted_transients():
    """One of four transients 0.1 ppm (13 Hz, within PASTIS's 25 Hz) off: its peak comes
    back onto the others' bin."""
    n, dwell, f0 = 2048, 1 / 2500.0, 127.7
    data = _transients(_singlets(n, dwell, f0), dwell, f0, count=4, shifted=(1,), shift=0.1)
    before = np.abs(np.fft.fft(np.asarray(data[:])[0, 0, 0], axis=0)).argmax(axis=0)

    out = pastis.align(data)

    assert out.shape == data.shape and list(out.dim_tags)[0] == "DIM_DYN"
    after = np.abs(np.fft.fft(np.asarray(out[:])[0, 0, 0], axis=0)).argmax(axis=0)
    assert np.ptp(before) > 5 and np.ptp(after) <= 1


@needs_pastis
def test_rejecting_drops_the_transients_that_stray():
    """Two of eight transients shifted by 0.8 ppm, beyond PASTIS's 0.5 ppm bound on the water
    peak's chemical shift (and its linewidth lost): six come back, still transients. The
    phase bound, a share of the phase spread the two outliers dominate, is let out of the
    way through PASTIS's own option."""
    n, dwell, f0 = 2048, 1 / 2500.0, 127.7
    data = _transients(_singlets(n, dwell, f0), dwell, f0, count=8, shifted=(2, 5))
    bounds = {"amplitude (%)": None, "linewidth (Hz)": [5.0, 30.0], "chemical shift (ppm)": 0.5,
              "phase std. factor (%)": 1000.0}

    out = pastis.reject(data, peak_properties_ranges=bounds)

    assert out.shape == (1, 1, 1, n, 6) and list(out.dim_tags)[0] == "DIM_DYN"
    with pytest.raises(ValueError, match="not a rejection criterion"):
        pastis.reject(data, criteria=["snr"])


@needs_pastis
def test_transient_steps_refuse_coils_and_other_dimensions():
    n, dwell, f0 = 512, 1 / 2500.0, 127.7
    with_coils = _transients(_singlets(n, dwell, f0), dwell, f0, count=2, coils=2)
    edited = gen_nifti_mrs(np.ones((1, 1, 1, n, 2, 2), np.complex64), dwell, f0,
                           dim_tags=["DIM_DYN", "DIM_EDIT", None])

    with pytest.raises(ValueError, match="combine the coils first"):
        pastis.align(with_coils)
    with pytest.raises(ValueError, match="DIM_EDIT"):
        pastis.reject(edited)
