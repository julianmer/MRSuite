####################################################################################################
#                                         test_tarquin.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Fitting with TARQUIN, through PyTARQUIN. TARQUIN is an external binary, so fits skip    #
#          when none is resolvable; the mapping is asserted on results laid out as TARQUIN's       #
#          ExportTxtResults writes them, and what is refused is refused before the binary runs.    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os

import numpy as np
import pytest

from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite.bridges import tarquin
from mrsuite.core.result import FitResult

from tests.conftest import CHALLENGE_BASIS

# as src/common/export_data.cpp writes them: the table, then 'key : value' blocks
RESULTS_TXT = """\
Signal     Conc (au)       %SD   SD (au)
----------------------------------------
NAA          8.330     3.000    0.2500
Cr           4.100     5.000    0.2100
PCr          1.900     9.000    0.1700
TCr          6.000     4.000    0.2400

Fit quality
-----------
Metab FWHM (PPM) : 0.04100
Metab FWHM (Hz)  : 5.240
SNR residual     : 25.30
SNR max          : 35.00
Q                : 1.230

Fitting parameters
------------------
Initial beta : 57.50
Max iters    : 100
"""


#***********#
#   ready   #
#***********#
def _available():
    """Whether a TARQUIN binary can be resolved without downloading or a container."""
    try:
        from tarquin_wrapper import binaries
        binaries.resolve_executable(allow_download=False, allow_docker=False)
        return True
    except Exception:                          # noqa: BLE001 - any failure means absent
        return False


needs_tarquin = pytest.mark.skipif(not _available(), reason="no TARQUIN binary")


#***************#
#   to result   #
#***************#
def test_a_report_maps_onto_the_result(tmp_path):
    """Concentrations, both uncertainty kinds and the fit quality block should all land."""
    path = tmp_path / "result.txt"
    path.write_text(RESULTS_TXT)

    from tarquin_wrapper import read_results   # here: conformance imports this module
    out = tarquin.to_result(read_results(path))

    assert isinstance(out, FitResult) and out.method == "tarquin:TARQUIN"
    assert out.conc("raw")["NAA"] == 8.33 and "TCr" in out.metabolites
    assert out.conc("internal")["NAA"] == pytest.approx(8.33 / 6.0)   # /(Cr+PCr)
    assert out.crlb("percentage")["NAA"] == 3.0
    assert out.crlb("raw")["NAA"] == 0.25
    assert out.qc == {"metab_fwhm_ppm": 0.041, "metab_fwhm_hz": 5.24, "snr_residual": 25.3,
                      "snr_max": 35.0, "q": 1.23}
    assert out.raw["report"].sections["fitting_parameters"]["max_iters"] == 100   # whole


#**************#
#   molarity   #
#**************#
def test_water_scaled_results_are_filed_as_molarity(tmp_path):
    """With a water reference TARQUIN's table is in mM, which is where LCModel and FSL-MRS
    file water-scaled values: comparing conc('raw') across them would mix units."""
    path = tmp_path / "result.txt"
    path.write_text(RESULTS_TXT.replace("Conc (au)", "Conc (mM)").replace("SD (au)", "SD (mM)"))

    from tarquin_wrapper import read_results
    out = tarquin.to_result(read_results(path))

    assert "molarity" in out.concentrations and "raw" not in out.concentrations
    assert out.crlb("molarity")["NAA"] == 0.25


#*************#
#   refused   #
#*************#
def test_what_cannot_be_fitted_is_refused_before_the_binary_runs(spectrum):
    """The internal basis is simulated at the echo time, which jMRUI text does not carry;
    and a FID per coil is not one spectrum."""
    with pytest.raises(ValueError, match="echo time"):
        tarquin.fit(spectrum)

    coils = gen_nifti_mrs(np.ones((1, 1, 1, 64, 2), complex), 1 / 2000, 123.2,
                          dim_tags=["DIM_COIL", None, None])
    with pytest.raises(ValueError, match="single spectra"):
        tarquin.fit(coils, te=0.03)
    single = gen_nifti_mrs(np.ones((1, 1, 1, 64), complex), 1 / 2000, 123.2)
    with pytest.raises(ValueError, match="single spectra"):           # averages in the water
        tarquin.fit(single, te=0.03, water=coils)


#*********#
#   fit   #
#*********#
@needs_tarquin
def test_fit_returns_a_result_per_spectrum(spectra):
    if not os.path.exists(CHALLENGE_BASIS):
        pytest.skip("challenge submodule not checked out")

    outs = tarquin.fit(spectra, CHALLENGE_BASIS)

    assert len(outs) == len(spectra.list())
    for out in outs:
        assert isinstance(out, FitResult)
        assert out.conc("raw") and out.crlb("percentage")


#****************#
#   physiology   #
#****************#
@needs_tarquin
def test_a_fit_finds_naa_where_naa_is(spectrum):
    """Asserted on physiology, as for LCModel: a mirrored spectrum - the FID handed over
    in the wrong orientation - still fits, but not with these ratios."""
    if not os.path.exists(CHALLENGE_BASIS):
        pytest.skip("challenge submodule not checked out")

    tcr = tarquin.fit(spectrum, CHALLENGE_BASIS).conc("internal")

    assert 1.0 < tcr["NAA"] < 2.0, f"NAA/tCr = {tcr['NAA']}, physiologically ~1.5"


#************#
#   curves   #
#************#
@needs_tarquin
def test_a_fit_keeps_the_curves_tarquin_drew(spectrum):
    """TARQUIN writes its fit as a CSV when asked; the fit asks, reads it back and keeps
    the series: the whole spectrum, as TARQUIN exports it, with the fit on the data."""
    out = tarquin.fit(spectrum, CHALLENGE_BASIS)

    curves = out.curves()
    ppm, data, fit = curves["ppm"], curves["data"], curves["fit"]
    assert len(ppm) == len(data) == len(fit) == len(curves["baseline"]) >= spectrum.shape[3]
    assert ppm[0] < 0.0 < 4.2 < ppm[-1]
    within = (ppm > 0.5) & (ppm < 4.2)
    assert np.linalg.norm(data[within] - fit[within]) < 0.2 * np.linalg.norm(data[within])
    assert np.abs(fit[within]).max() > 0.5 * np.abs(data).max()   # the fit is where the data are
