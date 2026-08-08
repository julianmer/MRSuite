####################################################################################################
#                                         test_result.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: The uniform analysis result, and the FSL-MRS adapter that fills it. Filled from         #
#          a real fit rather than a hand-built stand-in: the point of the adapter is that          #
#          it survives what a toolbox actually returns.                                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pytest

from mrsuite.bridges.fsl_mrs import to_result
from mrsuite.core.result import FitResult


#**********#
#   conc   #
#**********#
def test_conc_and_crlb_read_the_named_scaling():
    res = FitResult(method="t", metabolites=["NAA"],
                    concentrations={"raw": {"NAA": 1.0}, "internal": {"NAA": 2.0}},
                    uncertainties={"percentage": {"NAA": 5.0}})

    assert res.conc("raw") == {"NAA": 1.0}
    assert res.conc("internal") == {"NAA": 2.0}
    assert res.crlb() == {"NAA": 5.0}


#***************#
#   not there   #
#***************#
def test_an_absent_scaling_says_what_is_there():
    """A method reports only the scalings its fit was set up for, so the error has to
    name the alternatives rather than just fail."""
    res = FitResult(method="t", concentrations={"raw": {}})

    with pytest.raises(KeyError, match=r"\['raw'\]"):
        res.conc("molarity")
    with pytest.raises(KeyError, match="no 'percentage' uncertainties"):
        res.crlb()


#*************#
#   adapter   #
#*************#
def test_the_fsl_mrs_adapter_fills_the_uniform_fields(fsl_mrs_fit):
    out = to_result(fsl_mrs_fit)

    assert out.method == "fsl_mrs:Newton"
    assert "NAA" in out.metabolites
    assert out.conc("raw")["NAA"] > 0
    assert 0 < out.crlb("percentage")["NAA"] < 100
    assert out.qc["mse"] > 0
    assert out.raw is fsl_mrs_fit, "the method's own object must be kept"


#**************#
#   scalings   #
#**************#
def test_only_the_scalings_the_fit_calculated_are_reported(fsl_mrs_fit):
    """getConc raises for a scaling that was never calculated, so the adapter probes.

    This fit has no water reference, so molarity and molality cannot exist; claiming
    them would be worse than omitting them.
    """
    out = to_result(fsl_mrs_fit)

    assert "raw" in out.concentrations
    assert "molarity" not in out.concentrations
    assert "molality" not in out.concentrations


#***********#
#   names   #
#***********#
def test_every_metabolite_has_a_concentration(fsl_mrs_fit):
    out = to_result(fsl_mrs_fit)

    assert set(out.conc("raw")) == set(out.metabolites)
