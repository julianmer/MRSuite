####################################################################################################
#                                         test_pyamares.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-29                                                                              #
#                                                                                                  #
# Purpose: Fitting with pyAMARES. The result mapping is asserted on a table laid out as            #
#          pyAMARES' report builds it, refusals happen before the library is needed, and           #
#          a synthetic two-singlet spectrum pins the orientation and ppm conventions the           #
#          way the physiology tests do for TARQUIN and LCModel.                                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite.bridges import pyamares
from mrsuite.core.result import FitResult


needs_pyamares = pytest.mark.skipif(pyamares._version() is None, reason="pyAMARES not installed")

# as util.report builds result_sum: peaks by name, amplitudes and SDs summed over a
# multiplet's components, but CRLB(%) the first component's (NAA's here disagrees)
RESULT_SUM = pd.DataFrame(
    {"amplitude": [8.33, 6.0], "sd": [0.25, 0.24], "CRLB(%)": [1.0, 4.0],
     "chem shift(ppm)": [2.008, 3.027], "SNR": [35.0, 30.0]},
    index=pd.Index(["NAA", "tCr"], name="name"))

# a prior-knowledge file of NAA's and tCr's singlets, as pyAMARES reads them: initial
# values, then bounds; an open upper bound is written without its closing parenthesis
SINGLETS = ("Index,NAA,tCr\n"
            "Initial Values,,\n"
            "amplitude,1,1\n"
            "chemicalshift,2.008,3.027\n"
            "linewidth,3,3\n"
            "phase,0,0\n"
            "g,0,0\n"
            "Bounds,,\n"
            'amplitude,"(0, ","(0, "\n'
            'chemicalshift,"(1.9, 2.1)","(2.93, 3.13)"\n'
            'linewidth,"(0.5, 15)","(0.5, 15)"\n'
            'phase,"(-180, 180)","(-180, 180)"\n'
            'g,"(0, 0.001)","(0, 0.001)"\n')


#*************#
#   mapping   #
#*************#
def test_result_maps_the_summed_table():
    """Amplitudes, both uncertainty kinds, the tCr scaling and the SNR should all land;
    the percentage is SD over amplitude, whatever the table's CRLB(%) says, and the
    fit's scale comes back out."""
    out = pyamares._result(SimpleNamespace(result_sum=RESULT_SUM), scale=2.0)

    assert isinstance(out, FitResult) and out.method == "pyamares:AMARES"
    assert out.conc("raw")["NAA"] == pytest.approx(16.66)
    assert out.conc("internal")["NAA"] == pytest.approx(8.33 / 6.0)
    assert out.crlb("percentage")["NAA"] == pytest.approx(100 * 0.25 / 8.33)
    assert out.crlb("raw")["NAA"] == pytest.approx(0.5)
    assert out.qc == {"snr_max": 35.0}


#***********#
#   empty   #
#***********#
def test_a_fit_without_peaks_is_refused():
    with pytest.raises(ValueError, match="no result table"):
        pyamares._result(SimpleNamespace(result_sum=RESULT_SUM.iloc[:0]))


#*************#
#   refused   #
#*************#
def test_what_cannot_be_fitted_is_refused_before_the_library_is_needed():
    coils = gen_nifti_mrs(np.ones((1, 1, 1, 64, 2), complex), 1 / 2000, 123.2,
                          dim_tags=["DIM_COIL", None, None])
    with pytest.raises(ValueError, match="single spectra"):
        pyamares.fit(coils, prior="unused.csv")


#****************#
#   physiology   #
#****************#
@needs_pyamares
def test_a_fit_finds_naa_where_naa_is(tmp_path):
    """Two singlets at NAA's and tCr's shifts, amplitudes 2:1, on the NIfTI-MRS
    conventions: a mirrored or unreferenced axis cannot return this ratio at these ppm."""
    n, dwell, f0, shift = 2048, 1 / 2500.0, 127.7, 4.65
    t = np.arange(n) * dwell
    fid = np.zeros(n, complex)
    for ppm, amp in ((2.008, 2.0), (3.027, 1.0)):
        fid += amp * np.exp((2j * np.pi * (ppm - shift) * f0 - 8.0) * t)
    rng = np.random.default_rng(0)
    fid += 0.001 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    data = gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), dwell, f0,
                         nucleus="1H")

    prior = tmp_path / "prior.csv"
    prior.write_text(SINGLETS)

    out = pyamares.fit(data, str(prior))

    assert out.conc("internal")["NAA"] == pytest.approx(2.0, rel=0.05)
    assert 0.0 < out.crlb("percentage")["NAA"] < 20.0

    # scanner data sits anywhere from 1e-5 to 1e9; the fit must not depend on it
    for scale in (1e-5, 1e5, 1e9):
        scaled = gen_nifti_mrs((scale * fid).reshape(1, 1, 1, -1).astype(np.complex128),
                               dwell, f0, nucleus="1H")
        again = pyamares.fit(scaled, str(prior))
        assert again.conc("internal")["NAA"] == pytest.approx(2.0, rel=0.05), scale
        assert again.conc("raw")["NAA"] == pytest.approx(scale * out.conc("raw")["NAA"],
                                                         rel=0.02), scale


#***********#
#   prior   #
#***********#
@needs_pyamares
def test_a_prior_from_basis_functions_returns_their_amplitudes(tmp_path):
    """Basis functions become peak groups with one amplitude each, divided back to unit
    concentration: a doublet comes back as one metabolite, a name with digits (which
    pyAMARES reads as a multiplet suffix) under its own name, each at its amplitude."""
    n, dwell, f0, shift = 2048, 1 / 2500.0, 127.7, 4.65
    t = np.arange(n) * dwell

    def line(ppm):
        return np.exp((2j * np.pi * (ppm - shift) * f0 - 6.0) * t)

    basis = {"NAA": line(2.008), "Cr39": line(3.927), "Glx": line(2.34) + line(2.37)}
    amps = {"NAA": 2.0, "Cr39": 1.0, "Glx": 3.0}
    fid = sum(a * basis[m] for m, a in amps.items()) * np.exp(-4.0 * t)
    fid += 0.001 * (np.random.default_rng(0).normal(size=n)
                    + 1j * np.random.default_rng(1).normal(size=n))
    data = gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64), dwell, f0,
                         nucleus="1H")

    path = pyamares.prior(str(tmp_path / "prior.csv"), basis, dwell, f0, shift)
    out = pyamares.fit(data, path)

    assert set(out.conc("raw")) == set(amps)
    for name, amp in amps.items():
        assert out.conc("raw")[name] == pytest.approx(amp, rel=0.05), name


#********************#
#   reconstruction   #
#********************#
@needs_pyamares
def test_a_prior_as_pyamares_reads_it_is_the_basis(tmp_path):
    """Components of distinct phases and dampings, simulated by pyAMARES itself from the
    file at unit concentration: the basis back, one global phase apart at most."""
    import pyAMARES
    from pyAMARES.kernel.fid import multieq6, uninterleave

    n, dwell, f0, shift = 2048, 1 / 2500.0, 127.7, 4.65
    t = np.arange(n) * dwell

    def line(ppm, phase, rate):
        return np.exp((2j * np.pi * (ppm - shift) * f0 - rate) * t + 1j * phase)

    basis = {"A": line(2.0, 0.0, 6.0) + 0.6 * line(2.1, 0.9, 11.0),
             "B": line(3.0, -0.4, 8.0) + 0.5 * line(3.2, -1.3, 5.0)}
    path = pyamares.prior(str(tmp_path / "prior.csv"), basis, dwell, f0, shift)
    params = pyAMARES.initialize_FID(np.ones(n, complex), priorknowledgefile=path, MHz=f0,
                                     sw=1 / dwell, deadtime=0.0, carrier=shift, g_global=False)
    initial = params.initialParams.copy()
    for name, p in initial.items():
        if name.startswith("dk_") and p.expr is None:
            p.set(value=p.min)                  # no extra broadening: the functions' own
    initial.update_constraints()

    model = uninterleave(multieq6(initial, params.timeaxis))
    want = sum(basis.values()) * np.exp(2j * np.pi * shift * f0 * t)
    gain = np.vdot(model, want) / np.vdot(model, model)
    assert abs(gain) == pytest.approx(1.0, rel=0.01)
    assert np.linalg.norm(want - gain * model) < 0.01 * np.linalg.norm(want)


#***********#
#   names   #
#***********#
def test_names_are_spelt_as_pyamares_reads_them(tmp_path):
    """Digits as letters, anything else as an underscore; names spelt alike refused."""
    t = np.arange(256) / 2500.0
    fid = np.exp((2j * np.pi * 100.0 - 8.0) * t)

    path = pyamares.prior(str(tmp_path / "p.csv"), {"MM0.9": fid, "GPC+PCh": 2 * fid},
                          1 / 2500.0, 127.7, 4.65)
    assert pyamares._metabolites(path)[0] == {"MMa_j": "MM0.9", "GPC_PCh": "GPC+PCh"}

    with pytest.raises(ValueError, match="alike"):
        pyamares.prior(str(tmp_path / "q.csv"), {"A+B": fid, "A-B": fid}, 1 / 2500.0, 127.7,
                       4.65)
