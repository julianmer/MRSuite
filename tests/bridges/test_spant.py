####################################################################################################
#                                          test_spant.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: The spant bridge. spant writes NIfTI-MRS other readers do not accept as it is, so       #
#          what a step hands back is checked against the spectrum that went in: its shape, its     #
#          ppm reference and its history.                                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsuite import r

pytestmark = pytest.mark.skipif(not r.available(), reason="Rscript not on PATH")


#*************#
#   a step    #
#*************#
def test_a_step_keeps_the_spectrum_what_it_was(spectrum):
    from mrsuite import spant

    source = spectrum.copy()
    source.add_hdr_field("SpecFreqChemShift", 4.70)
    out = spant.lb(source, 5)

    assert out.shape == source.shape and out.dim_tags == source.dim_tags
    assert out.SpecFreqChemShift == 4.70
    assert not np.allclose(out[:], source[:])                # it did broaden the lines
    assert out.hdr_ext["ProcessingApplied"][-1]["Program"] == "spant"


#*************#
#   unknown   #
#*************#
def test_an_unknown_name_is_rejected_at_lookup():
    from mrsuite import spant

    with pytest.raises(AttributeError, match="no function"):
        spant.not_a_spant_function


#***************#
#   untouched   #
#***************#
def test_the_callers_spectrum_is_left_alone(spectrum):
    """Saving for R must not repoint the caller's image at a file that is then deleted."""
    from mrsuite import spant

    source = spectrum.copy()
    spant.lb(source, 5)
    spant.lb(source, 5)                                       # would fail on a stale file

    assert np.array_equal(source[:], spectrum[:])
