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
import subprocess
from pathlib import Path

import numpy as np

from mrsuite.core import provenance
from tests.conftest import needs_fsl_mrs


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


#***************#
#   byteorder   #
#***************#
def test_the_same_values_read_back_from_a_file_keep_their_checksum():
    """Native and explicitly little-endian complex arrays hold the same bytes; only the
    dtype's repr differs ('complex128' vs '<c16')."""
    values = np.arange(4.0) + 1j
    read = values.astype(np.dtype("c16").newbyteorder("<"))   # as nibabel hands it out
    assert repr(read.dtype) != repr(values.dtype)
    assert provenance.checksum(values) == provenance.checksum(read)
    assert provenance.checksum(values) != provenance.checksum(values.astype("<c8"))


#************#
#   a file   #
#************#
def test_a_file_is_recorded_by_its_bytes(tmp_path):
    """A basis is passed by its path, and the path alone would not show a changed file."""
    path = tmp_path / "basis.BASIS"
    path.write_text("one")
    before = provenance.describe({"basis": str(path)})["basis"]
    path.write_text("three")
    after = provenance.describe({"basis": path})["basis"]

    assert before["path"] == after["path"] == str(path)
    assert before["sha256"] != after["sha256"] == provenance.file_checksum(path)["sha256"]


#**************#
#   portable   #
#**************#
def test_a_shared_path_never_says_where_it_lies_on_this_machine(tmp_path):
    """Inside the checkout relative to it, anywhere else by the file's name alone."""
    inside = provenance.REPO / "data" / "basis_sets" / "some.BASIS"
    (tmp_path / "some.BASIS").write_text("one")

    assert provenance.portable(inside) == str(Path("data") / "basis_sets" / "some.BASIS")
    assert provenance.portable(tmp_path / "some.BASIS") == "some.BASIS"
    assert provenance.portable_file(tmp_path / "some.BASIS") == {
        "path": "some.BASIS",
        "sha256": provenance.file_checksum(tmp_path / "some.BASIS")["sha256"]}


#*****************#
#   entry files   #
#*****************#
def test_an_entry_records_files_portably(tmp_path):
    """A header travels with its spectrum, so an entry's files carry no machine path."""
    path = tmp_path / "basis.BASIS"
    path.write_text("one")
    # as a method records its call: the arguments as given, positional and by name
    details = json.loads(provenance.entry("LCModel", "6.3", "fit", {
        "args": (str(path),), "kwargs": {"basis": path}})["Details"])

    file = {"path": "basis.BASIS", "sha256": provenance.file_checksum(path)["sha256"]}
    assert details["args"] == [file]
    assert details["kwargs"] == {"basis": file}


def test_an_entry_records_folders_and_paths_to_be_portably(tmp_path):
    """A fit's output folder, given or yet to be made, is a path too."""
    details = json.loads(provenance.entry("LCModel", "6.3", "fit", {"kwargs": {
        "save_path": str(tmp_path), "out": tmp_path / "later"}})["Details"])

    assert details["kwargs"] == {"save_path": tmp_path.name, "out": "later"}
    assert str(tmp_path) not in json.dumps(details)


#*****************#
#   basis files   #
#*****************#
def test_a_basis_records_its_files_once(monkeypatch):
    """Its source already holds each file's record; a portable path that happens to lie
    under the working folder is not read as a file a second time."""
    from mrsuite.core.basis import Basis

    monkeypatch.chdir(provenance.REPO)
    source = {"files": [provenance.portable_file(provenance.REPO / "pyproject.toml")],
              "format": "LCModel .BASIS", "reader": "BasiC 90329e3"}
    basis = Basis(["NAA"], np.ones((1, 4), complex), 4000.0, 123.2, source=source)

    assert provenance.describe(basis, portable=True)["basis"]["source"] == source


#*********************#
#   mrsuite version   #
#*********************#
def test_an_installed_mrsuite_never_takes_another_repositorys_commit(tmp_path, monkeypatch):
    """Installed, the package lies in site-packages, which may sit inside any repository."""
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t",
           "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "--quiet"], check=True)
    subprocess.run(git + ["commit", "--quiet", "--allow-empty", "-m", "1"], check=True)
    monkeypatch.setattr(provenance, "REPO", tmp_path)

    assert provenance.git_version(str(tmp_path)), "the other repository has a commit"
    assert provenance.mrsuite_version() == __import__("mrsuite").__version__


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
@needs_fsl_mrs
def test_every_fitter_is_a_step_under_the_same_name():
    """Swapping the fitter in a pipeline is swapping one name."""
    from mrsuite.bridges import fsl_mrs, lcmodel, osprey

    for module in (fsl_mrs, lcmodel, osprey):
        assert module.step.fit("basis.BASIS").__name__ == "fit"


#****************#
#   byte order   #
#****************#
def test_a_checksum_is_the_values_whatever_their_byte_order():
    values = np.arange(6, dtype=complex).reshape(2, 3) * (1 + 2j)
    assert provenance.checksum(values) == provenance.checksum(values.astype(">c16"))
    assert provenance.checksum(values) != provenance.checksum(values.astype("<c8"))

