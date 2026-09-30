####################################################################################################
#                                         test_tarquin.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Fitting with TARQUIN. TARQUIN is an external binary, so fits skip when none is          #
#          resolvable; the parser is asserted on results laid out as TARQUIN's ExportTxtResults    #
#          writes them, and what is refused is refused before the binary is needed.                #
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
    try:
        tarquin._executable()
        return True
    except FileNotFoundError:
        return False


needs_tarquin = pytest.mark.skipif(not _available(), reason="no TARQUIN binary")


#***********#
#   parse   #
#***********#
def test_parse_maps_a_results_file(tmp_path):
    """Concentrations, both uncertainty kinds and the fit quality block should all land."""
    path = tmp_path / "result.txt"
    path.write_text(RESULTS_TXT)

    out = tarquin._parse(str(path))

    assert isinstance(out, FitResult) and out.method == "tarquin:TARQUIN"
    assert out.conc("raw")["NAA"] == 8.33 and "TCr" in out.metabolites
    assert out.conc("internal")["NAA"] == pytest.approx(8.33 / 6.0)   # /(Cr+PCr)
    assert out.crlb("percentage")["NAA"] == 3.0
    assert out.crlb("raw")["NAA"] == 0.25
    assert out.qc == {"metab_fwhm_ppm": 0.041, "metab_fwhm_hz": 5.24, "snr_residual": 25.3,
                      "snr_max": 35.0, "q": 1.23}


#**************#
#   molarity   #
#**************#
def test_water_scaled_results_are_filed_as_molarity(tmp_path):
    """With a water reference TARQUIN's table is in mM, which is where LCModel and FSL-MRS
    file water-scaled values: comparing conc('raw') across them would mix units."""
    path = tmp_path / "result.txt"
    path.write_text(RESULTS_TXT.replace("Conc (au)", "Conc (mM)").replace("SD (au)", "SD (mM)"))

    out = tarquin._parse(str(path))

    assert "molarity" in out.concentrations and "raw" not in out.concentrations
    assert out.crlb("molarity")["NAA"] == 0.25


#***********#
#   empty   #
#***********#
def test_results_without_signals_are_refused(tmp_path):
    path = tmp_path / "result.txt"
    path.write_text(RESULTS_TXT.split("NAA")[0])

    with pytest.raises(ValueError, match="no signals"):
        tarquin._parse(str(path))


#****************#
#   unreadable   #
#****************#
def test_a_row_run_together_is_refused(tmp_path):
    """TARQUIN's columns are fixed-width: a value that fills its column runs into the
    next, and such a row must not silently drop its metabolite."""
    path = tmp_path / "result.txt"
    path.write_text(RESULTS_TXT.replace("NAA          8.330     3.000    0.2500",
                                        "Ins       -0.0001234-4.052e+04   0.05000"))

    with pytest.raises(ValueError, match="cannot be read"):
        tarquin._parse(str(path))


#*************#
#   version   #
#*************#
def test_the_release_is_read_from_the_splash():
    """TARQUIN prints 'TARQUIN Version X' once data is loaded; --help prints only usage."""
    assert tarquin._banner("\nTARQUIN Version 4.3.11\nLoading data...\n") == "4.3.11"
    assert tarquin._banner("Usage: tarquin --input ...") is None


#*********#
#   dpt   #
#*********#
def test_the_dpt_puts_naa_where_tarquin_reads_it(tmp_path):
    """TARQUIN reads the samples with a forward FFT and ppm = ref - f/ft: a singlet made
    at 2.008 ppm in NIfTI-MRS must land there."""
    n, dwell, f0, ref = 2048, 1 / 2500.0, 127.7, 4.65
    t = np.arange(n) * dwell
    fid = np.exp((2j * np.pi * (2.008 - ref) * f0 - 8.0) * t)
    item = gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), dwell, f0)

    lines = open(tarquin._dpt(item, 0.03, str(tmp_path / "in.dpt"))).read().splitlines()
    header = dict(line.split("\t")[:2] for line in lines if "\t" in line and
                  not line.startswith("Real_FID"))
    samples = np.array([complex(*map(float, line.split())) for line in lines[9:]])
    spectrum = np.fft.fftshift(np.fft.fft(samples))
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / float(header["Sampling_frequency"])))
    ppm = float(header["PPM_reference"]) - f / float(header["Transmitter_frequency"]) * 1e6

    assert ppm[np.argmax(np.abs(spectrum))] == pytest.approx(2.008, abs=0.01)


#*************#
#   options   #
#*************#
def test_options_reach_the_command_line_as_tarquin_reads_them(spectrum, monkeypatch):
    """TARQUIN reads only the exact string 'true' as true; a Python True must arrive so."""
    import subprocess
    calls = []
    monkeypatch.setattr(tarquin, "_executable", lambda: "tarquin")
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 1, "", ""))

    with pytest.raises(RuntimeError, match="no result"):
        tarquin.fit.__wrapped__(spectrum, str(CHALLENGE_BASIS),
                                opts={"rescale_lcm_basis": True, "auto_phase": False})

    cmd = calls[-1]
    assert cmd[cmd.index("--rescale_lcm_basis") + 1] == "true"
    assert cmd[cmd.index("--auto_phase") + 1] == "false"


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
