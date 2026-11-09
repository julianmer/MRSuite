####################################################################################################
#                                           test_oxsa.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 09/11/26                                                                                #
#                                                                                                  #
# Purpose: Fitting with OXSA's AMARES. The result mapping is asserted on what the driver           #
#          returns, the refusals happen before Octave is needed, and synthetic 31P peaks pin the   #
#          orientation and the ppm convention: a shift counted from the carrier, as OXSA counts.   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite.bridges import oxsa
from mrsuite.core.result import FitResult
from tests.conftest import needs_octave

# two 31P peaks as OXSA's three tables: PCr at the carrier, the reference, and Pi 4.8 ppm up
TWO_PEAKS = {
    "bounds": [
        {"peakName": "PCr", "chemShift": [-1.0, 1.0], "linewidth": [5.0, 100.0],
         "amplitude": [0.0, np.inf], "phase": [0.0, 360.0], "chemShiftDelta": [],
         "amplitudeRatio": []},
        {"peakName": "Pi", "chemShift": [3.5, 6.0], "linewidth": [5.0, 100.0],
         "amplitude": [0.0, np.inf], "phase": [0.0, 360.0], "chemShiftDelta": [],
         "amplitudeRatio": []}],
    "initialValues": [
        {"peakName": "PCr", "chemShift": 0.0, "linewidth": 20.0, "amplitude": 1.0, "phase": 0.0},
        {"peakName": "Pi", "chemShift": 4.8, "linewidth": 20.0, "amplitude": 1.0, "phase": 0.0}],
    "priorKnowledge": [
        {"peakName": "PCr", "multiplet": [], "chemShiftDelta": [], "amplitudeRatio": [],
         "G_linewidth": [], "G_amplitude": [], "G_phase": 1, "G_chemShiftDelta": [], "refPeak": 1},
        {"peakName": "Pi", "multiplet": [], "chemShiftDelta": [], "amplitudeRatio": [],
         "G_linewidth": [], "G_amplitude": [], "G_phase": 1, "G_chemShiftDelta": [], "refPeak": 0}],
}


# NAA's singlet on 1H data centred at 4.65 ppm: 2.01 ppm is -2.64 ppm from the carrier
NAA_PRIOR = {
    "bounds": [{"peakName": "NAA", "chemShift": [-3.0, -2.3], "linewidth": [2.0, 30.0],
                "amplitude": [0.0, np.inf], "phase": [0.0, 360.0], "chemShiftDelta": [],
                "amplitudeRatio": []}],
    "initialValues": [{"peakName": "NAA", "chemShift": -2.64, "linewidth": 8.0, "amplitude": 1.0,
                       "phase": 0.0}],
    "priorKnowledge": [{"peakName": "NAA", "multiplet": [], "chemShiftDelta": [],
                        "amplitudeRatio": [], "G_linewidth": [], "G_amplitude": [], "G_phase": 1,
                        "G_chemShiftDelta": [], "refPeak": 1}],
}


def _peaks(n, dwell, f0, lines, damping=20.0):
    """Lorentzian peaks at ppm from the carrier, in NIfTI-MRS's orientation, as 31P data."""
    t = np.arange(n) * dwell
    fid = sum(a * np.exp((2j * np.pi * ppm * f0 - np.pi * damping) * t) for ppm, a in lines)
    fid += 1e-3 * np.random.default_rng(0).normal(size=n)
    return gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), dwell, f0, nucleus="31P")


#*************#
#   mapping   #
#*************#
def test_result_maps_what_the_driver_returns():
    out = {"names": ["PCr", "Pi"],
           "results": {"amplitude": np.array([2.0, 1.0]), "chemShift": np.array([0.1, 4.9]),
                       "linewidth": np.array([20.0, 25.0]), "phase": np.array([10.0, 10.0])},
           "crb": {"amplitude": np.array([0.02, 0.05]), "chemShift": np.array([0.001, 0.002]),
                   "linewidth": np.array([0.5, 0.8]), "phase": np.array([1.0, 1.0]),
                   "covariance": np.eye(7)},
           "reference": np.array([[1.0]]), "exitflag": 1.0, "iterations": 7.0,
           "relative_norm": 0.01, "residual_norm": 3.0}

    result = oxsa._result(out, centre=0.0)

    assert isinstance(result, FitResult) and result.method == "oxsa:AMARES"
    assert result.conc("raw") == {"PCr": 2.0, "Pi": 1.0}
    assert result.conc("internal")["Pi"] == 0.5
    assert result.crlb("raw")["Pi"] == 0.05
    assert result.crlb("percentage")["Pi"] == pytest.approx(5.0)
    assert result.qc == {"relative_norm": 0.01, "iterations": 7.0, "exitflag": 1.0}
    assert result.raw["ppm"]["Pi"] == pytest.approx(4.9)


def test_one_peak_and_no_reference_still_map():
    """A single peak comes back from Octave as scalars and a bare name; no reference peak,
    no internal scaling; a bound OXSA could not compute is no percentage."""
    out = {"names": "Peak1", "results": {"amplitude": 3.0, "chemShift": 0.0, "linewidth": 9.0,
                                         "phase": 0.0},
           "crb": {"amplitude": np.nan, "chemShift": np.nan, "linewidth": np.nan, "phase": np.nan,
                   "covariance": np.full((4, 4), np.nan)},
           "reference": np.zeros((0, 0)), "exitflag": 1.0, "iterations": 3.0,
           "relative_norm": 0.2, "residual_norm": 1.0}

    result = oxsa._result(out, centre=4.65)

    assert result.metabolites == ["Peak1"] and result.conc("raw") == {"Peak1": 3.0}
    assert "internal" not in result.concentrations and "percentage" not in result.uncertainties
    assert result.raw["ppm"]["Peak1"] == 4.65


def test_non_finite_amplitudes_are_refused():
    out = {"names": ["A"], "results": {"amplitude": np.array([np.nan]), "chemShift": [0.0],
                                       "linewidth": [1.0], "phase": [0.0]},
           "crb": {"amplitude": [0.0]}, "reference": [], "exitflag": 1.0, "iterations": 1.0,
           "relative_norm": 1.0, "residual_norm": 1.0}
    with pytest.raises(ValueError, match="non-finite"):
        oxsa._result(out)


#*************#
#   refused   #
#*************#
def test_what_cannot_be_fitted_is_refused_before_octave_is_needed():
    coils = gen_nifti_mrs(np.ones((1, 1, 1, 64, 2), complex), 1 / 2000, 120.3,
                          dim_tags=["DIM_COIL", None, None])
    with pytest.raises(ValueError, match="single spectra"):
        oxsa.fit(coils, "PK_SinglePeak")


#**********#
#   fits   #
#**********#
@needs_octave
def test_two_peaks_come_back_with_their_amplitudes_and_shifts():
    """PCr at the carrier and Pi 4.8 ppm up, 2:1: the shifts are counted from the carrier
    as OXSA counts them, and the ratio over the reference PCr is a half."""
    n, dwell, f0 = 2048, 1 / 5000.0, 120.3
    data = _peaks(n, dwell, f0, ((0.0, 2.0), (4.8, 1.0)))

    out = oxsa.fit(data, TWO_PEAKS)

    assert out.metabolites == ["PCr", "Pi"]
    assert out.conc("raw")["PCr"] == pytest.approx(2.0, rel=0.02)
    assert out.conc("internal")["Pi"] == pytest.approx(0.5, rel=0.02)
    assert out.raw["parameters"]["chemShift"][1] == pytest.approx(4.8, abs=0.02)
    assert out.raw["parameters"]["linewidth"][0] == pytest.approx(20.0, rel=0.1)
    assert 0.0 < out.crlb("percentage")["Pi"] < 5.0
    assert out.qc["relative_norm"] < 0.01


@needs_octave
def test_a_shipped_prior_fits_a_single_peak_and_the_history_names_oxsa():
    n, dwell, f0 = 1024, 1 / 5000.0, 120.3
    data = _peaks(n, dwell, f0, ((0.3, 1.5),))

    out = oxsa.fit(data, "PK_SinglePeak")

    assert out.metabolites == ["Peak1"]
    assert out.conc("raw")["Peak1"] == pytest.approx(1.5, rel=0.02)
    assert out.raw["parameters"]["chemShift"][0] == pytest.approx(0.3, abs=0.02)
    assert out.provenance[-1]["Program"] == "OXSA"


@needs_octave
def test_naa_is_found_where_naa_is_on_real_data(spectrum):
    """The challenge spectrum as stored, centred at 4.65 ppm: OXSA's shift from the carrier
    puts NAA at 2.01 ppm, which pins the orientation against NIfTI-MRS data, not against
    a signal made with the same formula."""
    out = oxsa.fit(spectrum, NAA_PRIOR)

    assert out.raw["ppm"]["NAA"] == pytest.approx(2.01, abs=0.03)
    assert out.conc("raw")["NAA"] > 0


@needs_octave
def test_priors_offered_stand_on_their_own():
    offered = oxsa.priors()
    assert {"PK_3T_Cardiac", "PK_7T_Cardiac", "PK_7T_Liver", "PK_SinglePeak"} <= set(offered)
    assert not any(name.endswith("_chooseInit") for name in offered)


@needs_octave
def test_a_prior_oxsa_lacks_is_refused_with_its_list():
    data = _peaks(512, 1 / 5000.0, 120.3, ((0.0, 1.0),))
    with pytest.raises(RuntimeError, match="no prior knowledge 'PK_Brain'.*PK_7T_Cardiac"):
        oxsa.fit(data, "PK_Brain")
