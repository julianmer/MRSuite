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
import pytest

from mrsuite.core.method import method
from mrsuite.core.result import FitResult
from mrsuite.datasets import challenge2016


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
