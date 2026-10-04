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
#          it survives what a toolbox actually returns. Saved and loaded back to the bit,          #
#          every fitter's own.                                                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import importlib
import os

import numpy as np
import pandas as pd
import pytest

from mrsuite import r
from mrsuite.bridges.fsl_mrs import to_result
from mrsuite.core.result import FitResult, load, save
from mrsuite.datasets import challenge2016

from tests.bridges.test_lcmodel import needs_lcmodel
from tests.bridges.test_tarquin import needs_tarquin
from tests.conftest import needs_octave


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


#***********#
#   exact   #
#***********#
def _exact(value):
    """A structure with each float as its bits and every other value with its type, so
    equality means equal to the bit."""
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, dict):
        return {k: _exact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_exact(v) for v in value]
    return type(value).__name__, value


#**********#
#   same   #
#**********#
def _same(a, b) -> bool:
    """Two of a tool's own objects equal to the bit, through containers, objects, arrays
    and pandas."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray) and a.dtype == object:       # Octave's cells, say
        return a.shape == b.shape and _same(a.ravel().tolist(), b.ravel().tolist())
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return _same(a.to_dict(), b.to_dict()) and a.dtypes.equals(b.dtypes)
    if isinstance(a, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.generic)):
        return np.asarray(a).tobytes() == np.asarray(b).tobytes()
    if hasattr(a, "__dict__") and not callable(a):
        return _same(vars(a), vars(b))
    return a == b


#****************#
#   to the bit   #
#****************#
def test_save_and_load_keep_every_value_to_the_bit(tmp_path):
    values = {"NAA": 0.1 + 0.2, "Cr": -0.0, "GABA": float("nan"), "Lac": float("inf"),
              "Ins": 5e-324, "Glu": -1.7976931348623157e308}
    res = FitResult(method="t", metabolites=list(values), concentrations={"raw": values},
                    uncertainties={"percentage": {"NAA": 3.0}}, qc={"snr": 12.5},
                    provenance=[{"Time": "now", "Details": "{}"}], raw={"own": [1, 2]})

    save([res, res], str(tmp_path / "r.json"))
    back = load(str(tmp_path / "r.json"))

    assert len(back) == 2
    assert _exact(back[0].to_dict()) == _exact(res.to_dict())
    assert back[1].raw == {"own": [1, 2]}
    assert load(str(tmp_path / "r.json"), raw=False)[1].raw is None


#************#
#   no raw   #
#************#
def test_without_raw_no_pickle_is_written_or_read(tmp_path):
    """Nor a pickle of an earlier save read back with results it does not belong to."""
    path = str(tmp_path / "r.json")
    save([FitResult(method="t", raw="old")], path)
    save([FitResult(method="t", raw="new")], path, raw=False)

    assert not os.path.exists(path + ".pkl")
    assert load(path)[0].raw is None


#******************#
#   every fitter   #
#******************#
@pytest.mark.skipif(not challenge2016.BASIS.exists(), reason="challenge submodule not checked out")
@pytest.mark.parametrize("fitter", [
    pytest.param("fsl_mrs", id="fsl_mrs"),
    pytest.param("lcmodel", id="lcmodel", marks=needs_lcmodel),
    pytest.param("tarquin", id="tarquin", marks=needs_tarquin),
    pytest.param("spant", id="spant",
                 marks=pytest.mark.skipif(not r.available(), reason="Rscript not on PATH")),
    pytest.param("osprey", id="osprey", marks=needs_octave)])
def test_every_fitters_result_loads_back_exactly(fitter, tmp_path):
    """A real fit, water-scaled where the fitter takes water: the uniform fields to the bit,
    and the tool's own object, value for value. Not its pickle's bytes: pandas lays out
    a fresh frame's blocks unlike an unpickled one's."""
    data, water, _ = challenge2016.load(1)
    fit = importlib.import_module(f"mrsuite.bridges.{fitter}").fit
    basis = str(challenge2016.BASIS)
    res = (fit(data.list()[0], basis, water=water.list()[0], **challenge2016.WATER[fitter])
           if fitter in challenge2016.WATER else fit(data.list()[0], basis))

    save([res], str(tmp_path / "r.json"))
    back, = load(str(tmp_path / "r.json"))

    assert _exact(back.to_dict()) == _exact(res.to_dict())
    assert _same(back.raw, res.raw)


#******************#
#   other pickle   #
#******************#
def test_a_pickle_saved_with_other_results_is_refused(tmp_path):
    """As one left by a save cut short after the pickle, before its JSON."""
    path = str(tmp_path / "r.json")
    save([FitResult(method="t", raw="old")], path)
    old = open(path, "rb").read()
    save([FitResult(method="t", raw="new"), FitResult(method="u", raw="new")], path)
    with open(path, "wb") as fh:
        fh.write(old)

    with pytest.raises(ValueError, match="other results"):
        load(path)
    assert load(path, raw=False)[0].raw is None
