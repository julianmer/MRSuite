####################################################################################################
#                                         test_parallel.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: Fitting a batch across processes. Spreading the work must not change it: the same       #
#          results, in the batch's order, with the provenance each spectrum carried.               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os

import pytest

from mrsuite.core.method import method
from mrsuite.core.parallel import PartlyFailed
from mrsuite.core.result import FitResult
from mrsuite.datasets import challenge2016


#*****************#
#   first point   #
#*****************#
@method("test", parallel=True)
def first_point(spectrum):
    """'NAA' is the spectrum's first point; fails for the values in MRSUITE_TEST_FAIL,
    which reach the workers as the environment does."""
    value = float(abs(spectrum[:].ravel()[0]))
    if str(value) in os.environ.get("MRSUITE_TEST_FAIL", "").split(","):
        raise ValueError(f"cannot fit {value}")
    return FitResult("test", ["NAA"], {"raw": {"NAA": value}})


#*************#
#   workers   #
#*************#
@pytest.mark.skipif(not challenge2016.BASIS.exists(), reason="challenge submodule not checked out")
def test_workers_change_nothing_but_the_time():
    from mrsuite.bridges import fsl_mrs

    data, _, _ = challenge2016.load(3)
    serial = fsl_mrs.fit(data, str(challenge2016.BASIS))
    spread = fsl_mrs.fit(data, str(challenge2016.BASIS), workers=2)

    assert [r.conc("raw") for r in spread] == [r.conc("raw") for r in serial]
    assert [r.provenance[0]["Details"] for r in spread] == \
        [r.provenance[0]["Details"] for r in serial]


#***********#
#   water   #
#***********#
@pytest.mark.skipif(not challenge2016.BASIS.exists(), reason="challenge submodule not checked out")
def test_each_worker_gets_the_references_of_its_own_spectra():
    """Spectra cannot be pickled, so references travel as files, cut like the data."""
    from mrsuite.bridges import fsl_mrs

    data, water, _ = challenge2016.load(3)
    serial = fsl_mrs.fit(data, str(challenge2016.BASIS), water=water, te=0.030, tr=2.0)
    spread = fsl_mrs.fit(data, str(challenge2016.BASIS), water=water, te=0.030, tr=2.0,
                         workers=2)

    assert [r.conc("molarity") for r in spread] == [r.conc("molarity") for r in serial]


#************#
#   single   #
#************#
def test_one_worker_or_one_spectrum_runs_in_process(spectrum):
    """No processes for a single spectrum: it would only add their start-up."""
    seen = []

    @method("test", parallel=True)
    def fit(data):
        seen.append(data)
        return FitResult("test")

    assert fit(spectrum, workers=8).method == "test" and seen == [spectrum]


#*************#
#   failing   #
#*************#
@pytest.mark.skipif(not challenge2016.BASIS.exists(), reason="challenge submodule not checked out")
def test_what_the_other_workers_gave_is_kept_when_one_fails(monkeypatch):
    data, _, _ = challenge2016.load(3)
    values = [float(abs(s[:].ravel()[0])) for s in data.list()]

    monkeypatch.setenv("MRSUITE_TEST_FAIL", str(values[1]))
    with pytest.raises(PartlyFailed) as caught:
        first_point(data, workers=3)
    assert {i: r.conc("raw")["NAA"] for i, r in caught.value.outputs.items()} == \
        {0: values[0], 2: values[2]}
    assert list(caught.value.errors) == [1] and isinstance(caught.value.errors[1], ValueError)
    assert caught.value.__cause__ is caught.value.errors[1]       # its traceback kept

    monkeypatch.setenv("MRSUITE_TEST_FAIL", ",".join(map(str, values)))
    with pytest.raises(ValueError):                 # nothing to keep: the error itself
        first_point(data, workers=3)
