####################################################################################################
#                                       test_provenance.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: What a step records about itself. The record has to be enough to redo the step,         #
#          so these check that it names exact versions, keeps small arguments as values, and       #
#          identifies large ones - arrays, reference spectra - by what they contain.               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json

import numpy as np

from mrsuite.core import provenance


#*************#
#   entries   #
#*************#
def test_an_entry_has_the_fields_nifti_mrs_defines():
    item = provenance.entry("FID-A", "V1.2", "op_addphase", {"args": [30.0]})

    assert set(item) == {"Time", "Program", "Version", "Method", "Details"}
    details = json.loads(item["Details"])
    assert details["args"] == [30.0]
    assert details["mrsuite"].startswith(__import__("mrsuite").__version__)


#*************#
#   unknown   #
#*************#
def test_a_version_that_cannot_be_told_says_so():
    assert provenance.entry("X", None, "m", {})["Version"] == "unknown"


#*********************#
#   small and large   #
#*********************#
def test_small_arguments_are_values_and_large_ones_checksums():
    assert provenance.describe((4, "t", 2.5)) == [4, "t", 2.5]
    assert provenance.describe(np.arange(3)) == [0, 1, 2]

    large = provenance.describe(np.arange(1000.0))
    assert large["array"]["shape"] == [1000]
    assert large["array"]["sha256"] == provenance.checksum(np.arange(1000.0))
    assert large["array"]["sha256"] != provenance.checksum(np.arange(1000.0) + 1)


#****************#
#   a spectrum   #
#****************#
def test_a_reference_spectrum_is_recorded_with_its_own_history(spectrum):
    """The water reference passed to a step shaped the result as much as the step did."""
    ref = spectrum.copy()
    provenance.append(ref, provenance.entry("FSL-MRS", "2.5.0", "coilcombine", {}))

    record = provenance.describe({"water": ref})["water"]["nifti_mrs"]
    assert record["sha256"] == provenance.checksum(ref[:])
    assert [s["Method"] for s in record["ProcessingApplied"]] == ["coilcombine"]
    json.dumps(record)                                      # storable as it is


#************#
#   append   #
#************#
def test_append_keeps_the_history_in_order(spectrum):
    s = spectrum.copy()
    for method in ("first", "second"):
        provenance.append(s, provenance.entry("P", "1", method, {}))

    assert [e["Method"] for e in provenance.history(s)] == ["first", "second"]


#*****************#
#   git version   #
#*****************#
def test_versions_come_from_the_checkout(tmp_path):
    assert provenance.git_version(str(provenance.REPO))
    assert provenance.git_version(str(tmp_path)) is None


#***************#
#   fit steps   #
#***************#
def test_every_fitter_is_a_step_under_the_same_name():
    """Swapping the fitter in a pipeline is swapping one name."""
    from mrsuite.bridges import fsl_mrs, lcmodel, osprey

    for module in (fsl_mrs, lcmodel, osprey):
        assert module.step.fit("basis.BASIS").__name__ == "fit"
