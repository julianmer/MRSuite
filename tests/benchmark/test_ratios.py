####################################################################################################
#                                           test_ratios.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: Concentrations as ratios to total creatine, a result of their own.                      #
#                                                                                                  #
####################################################################################################
import numpy as np
import pytest

from mrsuite.benchmark import scoring
from mrsuite.core.result import FitResult


#************#
#   ratios   #
#************#
def test_ratios_divide_by_creatine_and_phosphocreatine_together():
    fit = FitResult(method="lcmodel:LCModel", metabolites=["NAA", "Cr", "PCr", "Cr39"],
                    concentrations={"raw": {"NAA": 12.0, "Cr": 4.0, "PCr": 4.0}},
                    uncertainties={"percentage": {"NAA": 3.0}}, qc={"snr": 20.0}, raw=object())

    out, = scoring.ratios([fit])

    assert out.concentrations == {"tCr": {"NAA": 1.5, "Cr": 0.5, "PCr": 0.5}}
    assert out.method == fit.method and out.qc == fit.qc and out.raw is None
    assert out.uncertainties == {}


def test_ratios_read_a_water_scaled_fit_without_raw_concentrations():
    """LCModel with a water reference reports 'internal' and 'molarity' only."""
    fit = FitResult(method="lcmodel:LCModel", metabolites=["NAA", "Cr"],
                    concentrations={"internal": {"NAA": 1.2, "Cr": 1.0},
                                    "molarity": {"NAA": 9.0, "Cr": 6.0}})
    out, = scoring.ratios([fit])
    assert out.concentrations["tCr"] == {"NAA": 1.5, "Cr": 1.0}
    assert scoring.ratios([fit], scaling="internal")[0].concentrations["tCr"]["NAA"] == 1.2


def test_ratios_need_creatine_and_only_know_it():
    fit = FitResult(method="m:M", metabolites=["NAA"], concentrations={"raw": {"NAA": 1.0}})
    with pytest.raises(KeyError, match="no Cr or PCr"):
        scoring.ratios([fit])
    with pytest.raises(ValueError, match="tCr"):
        scoring.ratios([fit], to="water")
    assert np.isnan(scoring.ratios([FitResult(method="m:M", metabolites=["Cr", "NAA"],
                                              concentrations={"raw": {"Cr": 2.0, "NAA": np.nan}})]
                                   )[0].concentrations["tCr"]["NAA"])
