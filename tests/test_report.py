####################################################################################################
#                                          test_report.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-02                                                                              #
#                                                                                                  #
# Purpose: The report of spectra and fits: the MRSinMRS checklist filled from what they record and #
#          nothing else, every step, a trained model's MRSsynMRS table, and each tool cited once.  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import re
from pathlib import Path

import numpy as np
import pytest
from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite import report as report_
from mrsuite.core import provenance
from mrsuite.core.result import FitResult
from mrsuite.report import report
from mrsuite.training import recipe

CHECKOUT = Path(__file__).resolve().parents[1]


#**************#
#   spectrum   #
#**************#
def spectrum(te: float = 0.03, transients: int = 1, **header):
    """A 3 T 1H spectrum of a 20 mm cube, its header as given."""
    data = np.ones((1, 1, 1, 512, transients) if transients > 1 else (1, 1, 1, 512), complex)
    nifti = gen_nifti_mrs(data, 1 / 2000, 123.2, affine=np.diag([20.0, 20.0, 20.0, 1.0]),
                          dim_tags=["DIM_DYN" if transients > 1 else None, None, None])
    for key, value in {"EchoTime": te, **header}.items():
        nifti.add_hdr_field(key, value)
    return nifti


#**********#
#   rows   #
#**********#
def rows(text: str) -> dict:
    """The checklist's items, by their letter and section: '2d' -> its value."""
    found, section = {}, None
    for line in text.splitlines():
        head = re.match(r"\| (\d)\. ", line)
        item = re.match(r"\| ([a-i])\. [^|]*\| (.*) \|$", line)
        if head:
            section = head.group(1)
        elif item and section:
            found[section + item.group(1)] = item.group(2)
    return found


#***************#
#   checklist   #
#***************#
def test_what_the_header_records_fills_its_item_and_the_rest_says_not_recorded():
    nifti = spectrum(RepetitionTime=2.0, Manufacturer="Siemens", ManufacturersModelName="Prisma",
                     SoftwareVersions="XA30", SequenceName="svs_se", WaterSuppressed=True,
                     WaterSuppressionType="CHESS", TxCoil="Body", RxCoil="Head_32")
    got = rows(report(nifti))

    assert got["1a"].startswith("2.89 T (123.200 MHz, 1H)")
    assert (got["1b"], got["1c"]) == ("Siemens", "Prisma, software XA30")
    assert got["1d"] == "nucleus 1H, transmit Body, receive Head_32"
    assert got["2a"] == "svs_se" and got["2d"] == "TR 2000 ms, TE 30 ms"
    assert got["2c"] == "20 × 20 × 20 mm (8 cm³)"
    assert got["2f"] == "spectral width 2000 Hz, 512 points" and got["2g"] == "suppressed, CHESS"
    for item in ("1e", "2b", "2e", "2h", "2i", "3c", "3d", "4a", "4b", "4c", "4d"):
        assert got[item] == "not recorded", item


#**************#
#   unknowns   #
#**************#
def test_a_file_without_geometry_has_no_voi_size():
    """spec2nii writes a 10 m cube for a format without geometry; that is no voxel size."""
    nifti = gen_nifti_mrs(np.ones((1, 1, 1, 512), complex), 1 / 2000, 123.2,
                          affine=np.diag([10000.0, 10000.0, 10000.0, 1.0]))

    assert rows(report(nifti))["2c"] == "not recorded"


#*************#
#   batches   #
#*************#
def test_a_batch_is_one_report_listing_what_differs_per_spectrum():
    got = rows(report([spectrum(te=0.03, transients=32), spectrum(te=0.144, transients=32)]))

    assert got["2d"] == "1: TE 30 ms<br>2: TE 144 ms"
    assert got["2e"] == "32 transients in the data (DIM_DYN)"


#**********#
#   fits   #
#**********#
def test_a_fit_fills_its_items_and_each_tool_is_cited_once():
    nifti = spectrum()
    entries = [provenance.entry("FSL-MRS", "2.5.0", "coilcombine", {}),
               provenance.entry("FSL-MRS", "2.5.0", "align", {"kwargs": {"dim": "DIM_DYN"}}),
               provenance.entry("LCModel", "6.3-1R", "LCModel", {"kwargs": {"basis": "a.BASIS"}})]
    fit = FitResult(method="lcmodel:LCModel", metabolites=["NAA"],
                    concentrations={"raw": {"NAA": 1.0}, "molarity": {"NAA": 12.0}},
                    uncertainties={"percentage": {"NAA": 3.0}}, qc={"SNR": 61.8},
                    provenance=entries)
    text = report(nifti, fit)
    got = rows(text)

    assert got["3a"] == "FSL-MRS 2.5.0, LCModel 6.3-1R"
    assert got["3c"] == "lcmodel:LCModel: raw, molarity"
    assert got["3d"] == "LCModel LCModel: basis=a.BASIS"
    assert (got["4a"], got["4c"]) == ("SNR 61.8", "CRLB [%]: NAA 3")
    assert "2. **FSL-MRS** 2.5.0: `align` (dim=DIM_DYN)" in text
    references = text.split("## References")[1]
    assert references.count(report_.CITATIONS["FSL-MRS"]) == 1
    assert report_.CITATIONS["LCModel"] in references and report_.STANDARD in references


#************#
#   counts   #
#************#
def test_as_many_fits_as_spectra_are_asked():
    with pytest.raises(ValueError, match="2 fits for 1 spectra"):
        report(spectrum(), [None, None])


#*****************#
#   undeclared   #
#*****************#
def test_a_tool_without_a_citation_says_so():
    nifti = spectrum()
    provenance.append(nifti, provenance.entry("MyLab denoiser", "1.2", "denoise", {}))

    assert "MyLab denoiser: citation not declared" in report(nifti)


#***************#
#   free text   #
#***************#
def test_another_programs_free_text_details_are_shown_as_they_are():
    """spec2nii's conversion entry has Details as text, not MRSuite's JSON."""
    nifti = spectrum()
    provenance.append(nifti, {"Time": "2026-11-04T12:00:00.000000", "Program": "spec2nii",
                              "Version": "0.8.15", "Method": "Conversion",
                              "Details": "Converted from sub-01.dat"})

    assert "1. **spec2nii** 0.8.15: `Conversion` (Converted from sub-01.dat)" in report(nifti)


#***************#
#   synthetic   #
#***************#
def test_a_trained_model_in_the_chain_adds_its_mrssynmrs_table(monkeypatch):
    """Read as it was loaded, from the Hub's cache; one from a folder is named, not read."""
    from mrsuite.models import base
    made = {"MRSsynMRS": recipe.report()}
    monkeypatch.setattr(base, "_read_recipe", lambda name, revision, hub:
                        made if name == "MRSuite/mlp-test" else None)
    nifti = spectrum()
    for name in ("MRSuite/mlp-test", "my-folder"):
        provenance.append(nifti, provenance.entry("MRSuite", "0.0.3", "MLP", {
            "model": "mrsuite.models.mlp.MLP", "source": {"name": name}}))
    text = report(nifti)

    assert f"### MRSuite/mlp-test\n\n{recipe.table(made)}" in text
    assert "### my-folder\n\nLoaded from its own folder; its recipe.json lies beside it." in text
    assert text.split("## References")[1].count(report_.OWN["mrsuite.models.mlp.MLP"]) == 1
    assert report_.SYNTHETIC in text.split("## References")[1]


#****************#
#   every tool   #
#****************#
def test_every_tool_mrsuite_records_has_a_citation():
    """Read from the code: every Program a history entry or method names, cited by name."""
    named = set()
    for path in (CHECKOUT / "mrsuite").rglob("*.py"):
        source = path.read_text()
        named |= set(re.findall(r'provenance\.entry\(\s*["\']([^"\']+)["\']', source))
        named |= set(re.findall(r'@method\(\s*["\']([^"\']+)["\']', source))
        named |= set(re.findall(r'^\s+program = ["\']([^"\']+)["\']', source, re.M))
        named |= set(re.findall(r'Toolbox\(\s*["\'][^"\']+["\'],\s*["\']([^"\']+)["\']', source))
    named -= {"MRSuite"}                               # its own, cited as MRSuite

    assert named and not named - set(report_.CITATIONS), sorted(named - set(report_.CITATIONS))
