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
from tests.conftest import needs_octave
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
def test_fsl_mrs_processing(challenge):
    from mrsuite import fsl_mrs
    check(fsl_mrs.apodize, challenge[0], amount=[5.0])


#***************#
#   reference   #
#***************#
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


#**********#
#   fits   #
#**********#
@pytest.mark.parametrize("name", [
    pytest.param("fsl_mrs", id="fsl_mrs"),
    pytest.param("lcmodel", id="lcmodel", marks=needs_lcmodel),
    pytest.param("osprey", id="osprey", marks=needs_octave),
    pytest.param("spant", id="spant", marks=needs_r),
])
def test_fits(name, challenge):
    check(importlib.import_module(f"mrsuite.bridges.{name}").fit, challenge[0], BASIS)


#***********#
#   water   #
#***********#
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
def test_a_decomposing_model(challenge):
    check(WAND(WAND_METABOLITES, 1792, 2036, WAND_ACQUISITION), challenge[0])


#*********************#
#   processing step   #
#*********************#
def test_a_decomposing_model_as_a_processing_step(challenge):
    check(WAND(WAND_METABOLITES, 1792, 2036, WAND_ACQUISITION), challenge[0],
          remove=["artifacts"])
