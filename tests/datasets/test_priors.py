####################################################################################################
#                                         test_priors.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-04                                                                              #
#                                                                                                  #
# Purpose: The challenges' basis functions as sampled FIDs, and the Vespa and pyAMARES priors      #
#          made of them: the FIDs are the .BASIS every other fitter reads (the 2016 one less its   #
#          reference peak), and each prior names every basis function.                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsuite.bridges import pyamares
from mrsuite.datasets import challenge2016, mrsi_challenge
from tests.conftest import needs_fsl_mrs
from tests.datasets.test_challenges import MRSI_SUBJECT, needs_2016, needs_mrsi


#***************#
#   2016 fids   #
#***************#
@needs_2016
@needs_fsl_mrs
def test_the_2016_fids_are_the_basis_less_one_reference_peak():
    """The .BASIS holds the same functions at 1/64, each with the reference peak at 0 ppm
    the data lack: what the FIDs leave out is one and the same signal for all of them."""
    from fsl_mrs.utils import mrs_io

    fids = challenge2016.basis_fids()
    basis = mrs_io.read_basis(str(challenge2016.BASIS))
    assert set(basis.names) == set(fids) - {"Ace"}          # the .BASIS lacks Ace

    left = {name: 64 * basis.original_basis_array[:2048, basis.names.index(name)] - fids[name]
            for name in basis.names}
    peak = left["Cr"]
    ppm = 4.65 + np.fft.fftfreq(2048, 2.5e-4) / 123.2
    assert abs(ppm[np.argmax(np.abs(np.fft.fft(peak)))]) < 0.05
    for name, rest in left.items():
        assert np.linalg.norm(rest - peak) < 1e-3 * np.linalg.norm(fids[name]), name


#************#
#   priors   #
#************#
CHALLENGES = [pytest.param(challenge2016, {}, id="ismrm2016", marks=needs_2016),
              pytest.param(mrsi_challenge, {"subject": MRSI_SUBJECT}, id="mrsi",
                           marks=needs_mrsi)]


@pytest.mark.parametrize("challenge, where", CHALLENGES)
def test_each_prior_names_every_basis_function(challenge, where, tmp_path):
    names = set(challenge.basis_fids(**where))

    assert set(challenge.vespa_prior(**where)) == names
    path = challenge.pyamares_prior(folder=tmp_path, **where)
    assert set(pyamares._metabolites(path)[0].values()) == names
