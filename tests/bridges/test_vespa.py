####################################################################################################
#                                           test_vespa.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-29                                                                              #
#                                                                                                  #
# Purpose: Fitting with Vespa. The result mapping is asserted on a chain-shaped parameter          #
#          vector, refusals happen before the library is needed, and a synthetic                   #
#          two-singlet spectrum pins the conjugate orientation and the ppm conventions             #
#          the way the physiology tests do for TARQUIN and pyAMARES.                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite.bridges import vespa
from mrsuite.core.result import FitResult


needs_vespa = pytest.mark.skipif(vespa._version() is None, reason="Vespa not installed")

# NAA's and tCr's singlets as Vespa line lists
SINGLETS = {"NAA": {"ppms": [2.008], "areas": [1.0], "phases": [0.0]},
            "tCr": {"ppms": [3.027], "areas": [1.0], "phases": [0.0]}}


#*************#
#   mapping   #
#*************#
def test_result_maps_the_parameter_vector():
    """Areas, both uncertainty kinds and the tCr scaling should all land; Vespa's area
    bounds are percentages already. The lineshape, frequency and phase parameters after
    the areas are the method's own business."""
    params = np.array([8.33, 6.0, 2.008, 3.027, 0.5, 0.2, 0.1, 5.0])
    bounds = np.array([3.0, 4.0, 1e-5, 1e-5, 1e-3, 1e-4, 1e-3, 0.1])

    out = vespa._result(["NAA", "tCr"], params, bounds, scale=2.0)

    assert isinstance(out, FitResult) and out.method == "vespa:Voigt"
    assert out.conc("raw")["NAA"] == pytest.approx(16.66)
    assert out.conc("internal")["NAA"] == pytest.approx(8.33 / 6.0)
    assert out.crlb("percentage")["NAA"] == 3.0
    assert out.crlb("raw")["NAA"] == pytest.approx(16.66 * 0.03)


#***************#
#   no bounds   #
#***************#
def test_bounds_vespa_did_not_compute_are_no_uncertainty():
    """Vespa leaves its Cramer-Rao bounds at zero when switched off or singular; zero
    would read as perfect certainty, so the result carries none."""
    params = np.array([8.33, 6.0, 2.008, 3.027, 0.5, 0.2, 0.1, 5.0])

    out = vespa._result(["NAA", "tCr"], params, np.zeros_like(params))

    assert out.uncertainties == {} and out.conc("raw")["NAA"] == pytest.approx(8.33)


#***********#
#   empty   #
#***********#
def test_non_finite_areas_are_refused():
    with pytest.raises(ValueError, match="non-finite"):
        vespa._result(["NAA"], np.array([np.nan, 2.0]), np.array([0.1, 0.1]))


#*************#
#   refused   #
#*************#
def test_what_cannot_be_fitted_is_refused_before_the_library_is_needed():
    coils = gen_nifti_mrs(np.ones((1, 1, 1, 64, 2), complex), 1 / 2000, 123.2,
                          dim_tags=["DIM_COIL", None, None])
    with pytest.raises(ValueError, match="single spectra"):
        vespa.fit(coils, prior={})


#****************#
#   physiology   #
#****************#
@needs_vespa
def test_a_fit_finds_naa_where_naa_is():
    """Two singlets at NAA's and tCr's shifts, amplitudes 2:1, on the NIfTI-MRS
    conventions: a spectrum handed over unconjugated cannot return this ratio at these
    ppm, so this pins Vespa's orientation the way the dpt does TARQUIN's."""
    n, dwell, f0, shift = 2048, 1 / 2500.0, 127.7, 4.65
    t = np.arange(n) * dwell
    fid = np.zeros(n, complex)
    for ppm, amp in ((2.008, 2.0), (3.027, 1.0)):
        fid += amp * np.exp((2j * np.pi * (ppm - shift) * f0 - 8.0) * t)
    rng = np.random.default_rng(0)
    fid += 0.001 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    data = gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), dwell, f0,
                         nucleus="1H")

    out = vespa.fit(data, SINGLETS, opts={"baseline_method": ""})

    assert out.conc("internal")["NAA"] == pytest.approx(2.0, rel=0.05)
    assert 0.0 < out.crlb("percentage")["NAA"] < 20.0


#******************#
#   noise window   #
#******************#
@needs_vespa
def test_a_noise_window_on_the_metabolites_is_refused():
    """At a narrow sweep width both tails hold signal (the metabolites upfield, the water
    downfield); the Cramer-Rao noise window would read fit residual as noise."""
    n, dwell, f0 = 512, 1 / 500.0, 127.7
    data = gen_nifti_mrs(np.ones((1, 1, 1, n), np.complex64), dwell, f0, nucleus="1H")

    with pytest.raises(ValueError, match="noise window"):
        vespa.fit(data, SINGLETS)


@needs_vespa
def test_the_noise_window_moves_to_the_free_tail():
    """At 1000 Hz the upfield tail holds the metabolites but the downfield one is free,
    so the fit runs and its bounds are finite."""
    n, dwell, f0 = 1024, 1 / 1000.0, 127.7
    data = gen_nifti_mrs(_singlets(n, dwell, f0), dwell, f0, nucleus="1H")

    out = vespa.fit(data, SINGLETS, opts={"baseline_method": ""})

    assert out.conc("internal")["NAA"] == pytest.approx(2.0, rel=0.05)
    assert 0.0 < out.crlb("percentage")["NAA"] < 20.0


@needs_vespa
def test_the_noise_window_takes_the_upfield_tail_when_it_is_free():
    """A line downfield of the water at 7 ppm and none below 3 ppm: at 1000 Hz only the
    upfield tail is free."""
    n, dwell, f0, shift = 1024, 1 / 1000.0, 127.7, 4.65
    t = np.arange(n) * dwell
    lines = {"A": {"ppms": [3.2], "areas": [1.0], "phases": [0.0]},
             "B": {"ppms": [7.0], "areas": [1.0], "phases": [0.0]}}
    fid = sum(a * np.exp((2j * np.pi * (ppm - shift) * f0 - 8.0) * t)
              for ppm, a in ((3.2, 2.0), (7.0, 1.0)))
    fid += 0.001 * np.random.default_rng(0).normal(size=n)
    data = gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), dwell, f0,
                         nucleus="1H")

    out = vespa.fit(data, lines, opts={"baseline_method": ""})

    assert out.conc("raw")["A"] == pytest.approx(2.0 * out.conc("raw")["B"], rel=0.05)


#*************#
#   wavelet   #
#*************#
@needs_vespa
def test_the_wavelet_baseline_on_other_than_a_power_of_2_is_refused():
    """Vespa's wavelet filter returns the largest power of 2 below the length, and the
    fit then fails to broadcast (the MRSI challenge's 384 points)."""
    n, dwell, f0 = 384, 1 / 1200.0, 123.2
    data = gen_nifti_mrs(_singlets(n, dwell, f0), dwell, f0, nucleus="1H")

    with pytest.raises(ValueError, match="power of 2"):
        vespa.fit(data, SINGLETS)


#*************#
#   sampled   #
#*************#
@needs_vespa
def test_sampled_basis_functions_fit_as_line_lists_do():
    """Sampled functions in NIfTI-MRS orientation, one of them a doublet closer than the
    FID resolves into lines of its own width: their amplitudes come back, which a
    function handed over unconjugated or unchopped cannot give."""
    n, dwell, f0, shift = 2048, 1 / 2500.0, 127.7, 4.65
    t = np.arange(n) * dwell

    def line(ppm):
        return np.exp((2j * np.pi * (ppm - shift) * f0 - 6.0) * t)

    basis = {"NAA": line(2.008), "tCr": line(3.027), "Glx": line(2.34) + line(2.37)}
    amps = {"NAA": 2.0, "tCr": 1.0, "Glx": 1.5}
    fid = sum(a * basis[m] for m, a in amps.items()) * np.exp(-4.0 * t)
    fid += 0.001 * (np.random.default_rng(0).normal(size=n)
                    + 1j * np.random.default_rng(1).normal(size=n))
    data = gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), dwell, f0,
                         nucleus="1H")

    out = vespa.fit(data, {m: {"fid": f} for m, f in basis.items()},
                    opts={"baseline_method": ""})

    assert out.conc("internal")["NAA"] == pytest.approx(2.0, rel=0.05)
    assert out.conc("internal")["Glx"] == pytest.approx(1.5, rel=0.05)

    with pytest.raises(ValueError, match="points"):
        vespa.fit(data, {"NAA": {"fid": basis["NAA"][:1024]}})


#***********#
#   scale   #
#***********#
@needs_vespa
def test_a_prior_on_another_scale_gives_the_same_fit():
    """Vespa starts the areas from NAA's peak for a proton of area 1 and searches 50-150%
    around the start, so a basis on a scale of its own (here 0.05 a proton) would be
    fitted at a bound: the amplitudes come back on the basis' scale all the same."""
    n, dwell, f0, shift = 2048, 1 / 2500.0, 127.7, 4.65
    t = np.arange(n) * dwell

    def line(ppm):
        return 0.15 * np.exp((2j * np.pi * (ppm - shift) * f0 - 6.0) * t)

    basis = {"NAA": line(2.008), "Cr": line(3.027), "Glx": line(2.34) + line(2.37)}
    amps = {"NAA": 40.0, "Cr": 30.0, "Glx": 30.0}
    fid = sum(a * basis[m] for m, a in amps.items()) * np.exp(-4.0 * t)
    fid += 0.001 * (np.random.default_rng(0).normal(size=n)
                    + 1j * np.random.default_rng(1).normal(size=n))
    data = gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), dwell, f0,
                         nucleus="1H")

    out = vespa.fit(data, {m: {"fid": f} for m, f in basis.items()},
                    opts={"baseline_method": ""})

    for name, amp in amps.items():
        assert out.conc("raw")[name] == pytest.approx(amp, rel=0.1), name

    # Vespa's Area Scale Factor moves a start, and with it the 50-150% the area may take:
    # Glx started at a fifth of the literature's cannot reach its amplitude
    low = vespa.fit(data, {m: {"fid": f, **({"area_scale": 0.2} if m == "Glx" else {})}
                           for m, f in basis.items()}, opts={"baseline_method": ""})

    assert low.conc("raw")["Glx"] < 0.5 * amps["Glx"]
    assert low.conc("raw")["NAA"] == pytest.approx(amps["NAA"], rel=0.1)


#************#
#   proton   #
#************#
def test_a_proton_is_measured_on_the_reference_singlet():
    """The prior's scale is read off NAA's singlet, 3 protons in Vespa's table, for a
    sampled function as for a line list; else off Cr's; a prior with neither is left."""
    class Table:
        abbreviations, spins = ["naa", "cr"], [3.0, 3.0]

        def get_abbreviation(self, name):
            return {"naa": "naa", "tcr": "cr"}.get(name, name)

    n, dwell, f0 = 2048, 1 / 2500.0, 127.7
    t = np.arange(n) * dwell
    ppm = 4.65 + np.fft.fftfreq(n, dwell) / f0
    naa = 0.6 * np.exp((2j * np.pi * (2.008 - 4.65) * f0 - 6.0) * t)

    assert vespa._proton({"NAA": {"fid": naa}}, ppm, Table()) == pytest.approx(0.2, rel=0.1)
    lines = {"ppms": [2.008, 2.6], "areas": [1.5, 1.0]}
    assert vespa._proton({"Glu": {"ppms": [2.35], "areas": [9.0]}, "NAA": lines}, ppm,
                         Table()) == pytest.approx(0.5)
    assert vespa._proton({"tCr": {"ppms": [3.027], "areas": [6.0]}}, ppm, Table()) == 2.0
    assert vespa._proton({"Glu": {"ppms": [2.35], "areas": [9.0]}}, ppm, Table()) == 1.0


#**************#
#   singlets   #
#**************#
def _singlets(n, dwell, f0, shift=4.65):
    """NAA's and tCr's singlets, amplitudes 2:1, as a (1, 1, 1, n) array."""
    t = np.arange(n) * dwell
    fid = sum(a * np.exp((2j * np.pi * (ppm - shift) * f0 - 8.0) * t)
              for ppm, a in ((2.008, 2.0), (3.027, 1.0)))
    fid += 0.001 * (np.random.default_rng(0).normal(size=n)
                    + 1j * np.random.default_rng(1).normal(size=n))
    return fid.reshape(1, 1, 1, -1).astype(np.complex64)
