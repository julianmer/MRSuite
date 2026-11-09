####################################################################################################
#                                       test_conformance.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Every kind of method MRSuite ships, run through mrsuite.testing.check: processing       #
#          from each bridge, every fitter, and both models. Each skips without its runtime.        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import importlib

import pytest

from mrsuite import r
from mrsuite.datasets import challenge2016
from mrsuite.models import MLP, WAND
from mrsuite.testing import check

from tests.bridges.test_lcmodel import needs_lcmodel
from tests.bridges.test_mrspeclab import needs_mrspeclab
from tests.bridges.test_oxsa import NAA_PRIOR
from tests.bridges.test_pastis import needs_pastis
from tests.bridges.test_pyamares import SINGLETS as AMARES_SINGLETS, needs_pyamares
from tests.bridges.test_tarquin import needs_tarquin
from tests.bridges.test_vespa import SINGLETS as VESPA_SINGLETS, needs_vespa
from tests.conftest import needs_fsl_mrs, needs_octave
from tests.simulators.test_bases import needs_pygamma
from tests.models.test_models import (MLP_ACQUISITION, MLP_METABOLITES, WAND_ACQUISITION,
                                      WAND_METABOLITES, _spectra)

BASIS = str(challenge2016.BASIS)

needs_r = pytest.mark.skipif(not r.available(), reason="Rscript not on PATH")


#***************#
#   challenge   #
#***************#
@pytest.fixture(scope="module")
def challenge():
    """Two different challenge spectra and their water references."""
    if not challenge2016.BASIS.exists():
        pytest.skip("challenge submodule not checked out")
    data, water, _ = challenge2016.load(2)
    return data, water


#****************#
#   processing   #
#****************#
@needs_fsl_mrs
def test_fsl_mrs_processing(challenge):
    from mrsuite import fsl_mrs
    check(fsl_mrs.apodize, challenge[0], amount=[5.0])


#***************#
#   reference   #
#***************#
@needs_fsl_mrs
def test_fsl_mrs_processing_with_a_reference(challenge):
    from mrsuite import fsl_mrs
    check(fsl_mrs.ecc, *challenge)


#***********#
#   fid-a   #
#***********#
@needs_octave
def test_fid_a_processing(challenge):
    from mrsuite import fid_a
    check(fid_a.op_addphase, challenge[0], 30.0)


#************#
#   osprey   #
#************#
@needs_octave
def test_osprey_processing(challenge):
    from mrsuite import osprey
    check(osprey.op_addphase, challenge[0], 30.0)


#***********#
#   spant   #
#***********#
@needs_r
def test_spant_processing(challenge):
    from mrsuite import spant
    check(spant.lb, challenge[0], 5)


#***************#
#   mrspeclab   #
#***************#
@needs_mrspeclab
def test_mrspeclab_processing(challenge):
    from mrsuite import mrspeclab
    check(mrspeclab.LineBroadening, challenge[0], factor=3)


#**********#
#   fits   #
#**********#
@pytest.mark.parametrize("name", [
    pytest.param("fsl_mrs", id="fsl_mrs", marks=needs_fsl_mrs),
    pytest.param("lcmodel", id="lcmodel", marks=needs_lcmodel),
    pytest.param("osprey", id="osprey", marks=needs_octave),
    pytest.param("spant", id="spant", marks=needs_r),
    pytest.param("tarquin", id="tarquin", marks=needs_tarquin),
])
def test_fits(name, challenge):
    check(importlib.import_module(f"mrsuite.bridges.{name}").fit, challenge[0], BASIS)


#******************#
#   pyamares fit   #
#******************#
@needs_pyamares
def test_pyamares_fit(challenge, tmp_path):
    """pyAMARES takes a prior-knowledge file rather than a basis."""
    from mrsuite import pyamares
    prior = tmp_path / "prior.csv"
    prior.write_text(AMARES_SINGLETS)
    check(pyamares.fit, challenge[0], str(prior))


#***************#
#   vespa fit   #
#***************#
@needs_vespa
def test_vespa_fit(challenge):
    """Vespa takes metabolite line lists rather than a basis."""
    from mrsuite import vespa
    check(vespa.fit, challenge[0], VESPA_SINGLETS)


#**************#
#   oxsa fit   #
#**************#
@needs_octave
def test_oxsa_fit(challenge):
    """OXSA takes a prior-knowledge set rather than a basis: NAA on the 1H challenge data."""
    from mrsuite import oxsa
    check(oxsa.fit, challenge[0], NAA_PRIOR)


#****************#
#   pastis fit   #
#****************#
@needs_pastis
@needs_pygamma
def test_pastis_fit(challenge):
    """PASTIS fits the model it simulates: its basis names the simulation."""
    from mrsuite import pastis
    from mrsuite.simulators import bases
    data = challenge[0]
    basis = bases.pastis(30.0, "press", frequency=float(data.spectrometer_frequency[0]),
                         points=data.shape[3], bandwidth=1.0 / data.dwelltime)
    check(pastis.fit, data, basis, metabolites=["NAA", "Cr_CH3", "Cr_CH2", "mI", "Glu"])


#***********#
#   water   #
#***********#
@needs_fsl_mrs
def test_a_water_scaled_fit(challenge):
    from mrsuite import fsl_mrs
    data, water = challenge
    check(fsl_mrs.fit, data, BASIS, water=water, te=0.030, tr=2.0)


#************#
#   models   #
#************#
def test_a_quantifying_model():
    check(MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION), _spectra(3))


#*******************#
#   decomposition   #
#*******************#
# WAND's default normalises with the batch's statistics, as its paper ran it and
# reproduces exactly; that makes a spectrum's result depend on its batch by design, so
# the contract is checked with the stored averages, where it holds
def test_a_decomposing_model(challenge):
    check(WAND(WAND_METABOLITES, 1792, 2036, WAND_ACQUISITION, batch_statistics=False),
          challenge[0])


#*********************#
#   processing step   #
#*********************#
def test_a_decomposing_model_as_a_processing_step(challenge):
    check(WAND(WAND_METABOLITES, 1792, 2036, WAND_ACQUISITION, batch_statistics=False),
          challenge[0], remove=["artifacts"])
