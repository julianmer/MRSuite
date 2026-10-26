####################################################################################################
#                                      test_mr_science_lab.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: The MR SCIENCE Lab's basis sets and synthetic spectra read the right way round, named  #
#          as the repository names them; fetched once, so these run on what is already here.      #
#                                                                                                  #
####################################################################################################
import numpy as np
import pytest

from mrsuite.datasets import mr_science_lab as lab
from mrsuite.gui.graph import traces

needs_set = pytest.mark.skipif(
    not (lab.DATA / "basis_sets/press/siemens/SummedSpins_for_PRESSSiemens_TE_30_BW_4000_NPts_2048"
         / "NAA.mat").is_file(), reason="the Siemens PRESS TE 30 set is not fetched")
needs_spectrum = pytest.mark.skipif(
    not (lab.DATA / "synthetic/ImperfectModelSpectra/GB20Hz2.mat").is_file(),
    reason="the imperfect 20 Hz² spectrum is not fetched")


@needs_set
def test_a_marss_basis_set_reads_the_right_way_round():
    basis = lab.basis("press", "siemens", 30, 4000, 2048, metabolites=["NAA", "Cr", "Cho"])
    ppm, specs = basis.ppm(), basis.spectra()
    peaks = {n: ppm[np.abs(s).argmax()] for n, s in zip(basis.names, specs)}
    assert abs(peaks["NAA"] - 2.01) < 0.05 and abs(peaks["Cr"] - 3.03) < 0.05, peaks
    assert basis.fids.shape == (3, 2048) and basis.bandwidth == 4000 and 123 < basis.frequency < 124
    assert basis.te == 30 and basis.seq == "PRESS" and "mrs-basis-sets" in basis.source["repository"]


@needs_spectrum
def test_a_synthetic_spectrum_reads_as_nifti_mrs_with_its_source():
    spectra = lab.spectrum("imperfect", 20)
    item = spectra.list()[0]
    ppm, ys, _ = traces(item)
    assert item.shape == (1, 1, 1, 2048) and abs(ppm[np.argmax(ys[0])] - 2.01) < 0.05
    assert item.hdr_ext["SpecFreqChemShift"] == lab.CENTRE
    step = item.hdr_ext["ProcessingApplied"][-1]
    assert step["Method"].endswith("mr_science_lab.spectrum") and "INSPECTOR" in step["Details"]


def test_names_follow_the_repository():
    assert lab._plain(16.5) == "16.5" and lab._plain(30) == "30" and lab._plain(4000.0) == "4000"
    with pytest.raises(KeyError):
        lab.basis("mega", "siemens")
