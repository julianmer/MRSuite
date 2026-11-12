####################################################################################################
#                                         test_specvis.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 10/11/26                                                                                #
#                                                                                                  #
# Purpose: The SpecVis bridge. The table MRSuite writes from fit results names the sums as         #
#          SpecVis's importers do and pairs the rows; on SpecVis's own example data (Big GABA,    #
#          three vendors by Osprey, LCModel and TARQUIN) its importers, every plot and its        #
#          statistics run through R, and the statistics give the numbers its Example.md prints.   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import glob
import os

import numpy as np
import pandas as pd
import pytest

from mrsuite import externals, r
from mrsuite.bridges import specvis
from mrsuite.core.result import FitResult

needs_r = pytest.mark.skipif(not r.available(), reason="Rscript not on PATH")

MEASURES = ["tNAA", "tCho", "Ins", "Glx"]
LIMITS = [(0.75, 2.1), (0.05, 0.33), (0.4, 1.22), (0.75, 2.75)]


def _fit(method, **conc):
    return FitResult(method=method, metabolites=list(conc), concentrations={"raw": conc})


#***********#
#   table   #
#***********#
def test_the_table_names_the_sums_as_specvis_does_and_pairs_the_rows():
    """LCModel reports the sums under its own names, FSL-MRS only the singles: one table."""
    lcm = [_fit("lcmodel", **{"NAA": 10.0, "NAAG": 1.0, "Cr": 6.0, "PCr": 2.0,
                              "NAA+NAAG": 11.0, "Cr+PCr": 8.0, "GPC+PCh": 2.0, "GPC": 1.5,
                              "PCh": 0.5})]
    fsl = [_fit("fsl", NAA=9.0, NAAG=2.0, Cr=5.0, PCr=3.0, GPC=1.0, PCho=1.0)]   # the basis's name
    frame = specvis.table({"LCModel": lcm, "FSL-MRS": fsl},
                          covariates=pd.DataFrame({"age (years)": [30]}))

    assert list(frame.Group) == ["LCModel", "FSL-MRS"]
    assert np.allclose(frame.tNAA, [11 / 8, 11 / 8]) and np.allclose(frame.tCr, 1)
    assert np.allclose(frame.tCho, [2 / 8, 2 / 8]) and list(frame["age (years)"]) == [30, 30]
    assert "NAA+NAAG" not in frame.columns and "Cr+PCr" not in frame.columns


def test_the_table_on_another_scaling_keeps_the_values_and_names_the_sums():
    frame = specvis.table([_fit("fsl", NAA=9.0, NAAG=2.0, Cr=5.0, PCr=3.0)], scaling="raw")

    assert "Group" not in frame.columns
    assert frame.NAA[0] == 9.0 and frame.tNAA[0] == 11.0 and frame.tCr[0] == 8.0


def test_covariates_must_be_one_row_per_fit():
    with pytest.raises(ValueError, match="rows of covariates"):
        specvis.table([_fit("fsl", NAA=9.0, Cr=5.0)], covariates=pd.DataFrame({"age": [1, 2]}))


#*************#
#   refused   #
#*************#
def test_what_the_plots_cannot_pair_is_refused_before_r_is_needed():
    frame = pd.DataFrame({"tNAA": [1.0, 1.1, 1.2], "Group": ["A", "A", "B"],
                          "site": ["x", "y", "x"]})
    with pytest.raises(ValueError, match="no 'C'"):
        specvis.correlation(frame, "A", "C", ["tNAA"])
    with pytest.raises(ValueError, match="pairs them one to one"):
        specvis.correlation(frame, "A", "B", ["tNAA"])
    with pytest.raises(ValueError, match="no column 'tCho'"):
        specvis.correlation(frame.iloc[[0, 2]].assign(tNAA=[np.nan, 1.0]), "A", "B", ["tCho"])
    with pytest.raises(ValueError, match="no column 'vendor'"):
        specvis.bland_altman(frame.iloc[[0, 2]], "A", "B", ["tNAA"], by="vendor")
    with pytest.raises(ValueError, match="2 limits for 1 measures"):
        specvis.correlation(frame.iloc[[0, 2]], "A", "B", ["tNAA"], limits=LIMITS[:2])
    with pytest.raises(ValueError, match="go together"):
        specvis.bland_altman(frame.iloc[[0, 2]], "A", "B", ["tNAA"], limits=LIMITS[:1])
    with pytest.raises(ValueError, match="no Group column"):
        specvis.correlation(frame.drop(columns="Group"), "A", "B", ["tNAA"])
    with pytest.raises(ValueError, match=r"\.pdf, \.png or \.svg"):
        specvis.raincloud(frame, ["tNAA"], path="figure.eps")
    with pytest.raises(ValueError, match="no column 'tCho' in the table"):
        specvis.raincloud(frame, ["tNAA", "tCho"])
    with pytest.raises(ValueError, match="no column 'tCho' in the table"):
        specvis.correlation(frame.iloc[[0, 2]], "A", "B", ["tCho"])
    with pytest.raises(ValueError, match="no column 'tCho' in the table"):
        specvis.statistics(frame, ["tCho"])


#*************#
#   example   #
#*************#
@pytest.fixture(scope="module")
def example():
    """SpecVis's own example data: Big GABA, three sites, by three algorithms. Skipped where
    R lacks SpecVis's packages."""
    try:
        specvis.runtime()
    except RuntimeError as error:
        if "needs the R packages" not in str(error):
            raise
        pytest.skip("R without SpecVis's packages")
    return externals.ensure("specvis") / "examples" / "data"


@pytest.fixture(scope="module")
def frame(example):
    """Example.md's dfData: every vendor's results by every algorithm, with its stat.csv,
    the algorithm under Group and the vendor beside it."""
    parts = []
    for vendor in ("GE", "Philips", "Siemens"):
        stat = pd.read_csv(example / vendor / "stat.csv")
        for name, where in (("LCModel", "LCModelOutput"), ("Osprey", "off_tCr.csv"),
                            ("Tarquin", "TarquinOutput")):
            part = specvis.load(str(example / vendor / where))
            parts.append(pd.concat([part, stat], axis=1).assign(Group=name, vendor=vendor))
    return pd.concat(parts, ignore_index=True)


#**********#
#   load   #
#**********#
@needs_r
def test_the_three_formats_load_alike(example):
    osprey = specvis.load(str(example / "GE" / "off_tCr.csv"))
    lcmodel = specvis.load(str(example / "GE" / "LCModelOutput"))
    tarquin = specvis.load(str(example / "GE" / "TarquinOutput"))

    for table in (osprey, lcmodel, tarquin):
        assert len(table) == 12 and set(MEASURES) <= set(table.columns)
    assert np.allclose(lcmodel.tCr, 1) and np.allclose(tarquin.tCr, 1)
    assert not np.allclose(specvis.load(str(example / "GE" / "LCModelOutput"), ratios=False).tCr, 1)
    crlbs = specvis.load(str(example / "GE" / "LCModelOutput"), what="crlbs")
    diagnostics = specvis.load(str(example / "GE" / "TarquinOutput"), what="diagnostics")
    assert len(crlbs) == 12 and "tNAA" in crlbs.columns
    assert len(diagnostics) == 12 and len(diagnostics.columns) > 1
    with pytest.raises(ValueError, match="not 'crlbs'"):
        specvis.load(str(example / "GE" / "off_tCr.csv"), what="crlbs")


@needs_r
def test_mrsuites_table_from_the_coord_files_is_specvis_own_import(example):
    """The same .coord files through PyLCModel's parser and MRSuite's ratios give SpecVis's
    numbers, so fits made in MRSuite and results read by SpecVis sit in one table."""
    from lcmodel_wrapper import coord

    from mrsuite.bridges.lcmodel import to_result

    folder = example / "GE" / "LCModelOutput"
    files = sorted(glob.glob(os.path.join(folder, "*.coord")))
    mine = specvis.table([to_result(coord.read_coord(f, meta=True)) for f in files])
    theirs = specvis.load(str(folder))

    assert np.allclose(mine[MEASURES], theirs[MEASURES], rtol=1e-4)


#***********#
#   plots   #
#***********#
@needs_r
def test_the_distribution_plots_draw_and_write(frame, tmp_path):
    from matplotlib.figure import Figure

    made = specvis.raincloud(frame, MEASURES, limits=LIMITS, title="Distribution Analysis",
                             columns=4, size=(20, 5), path=str(tmp_path / "raincloud.pdf"))
    assert isinstance(made, Figure) and (tmp_path / "raincloud.pdf").stat().st_size > 0
    assert isinstance(specvis.boxplot(frame, MEASURES[:2], by="vendor", limits=LIMITS[:2]), Figure)
    assert isinstance(specvis.raincloud(frame, MEASURES[:2]), Figure)   # its limits from the data
    assert isinstance(specvis.violin(frame, MEASURES[:2], path=str(tmp_path / "v.png")), Figure)
    assert (tmp_path / "v.png").stat().st_size > 0


@needs_r
def test_the_comparison_plots_draw_pooled_and_by_group(frame, tmp_path):
    from matplotlib.figure import Figure

    pooled = specvis.correlation(frame, "Osprey", "LCModel", MEASURES, limits=LIMITS,
                                 columns=4)
    grouped = specvis.correlation(frame, "Osprey", "LCModel", MEASURES, by="vendor",
                                  hue="group name", limits=LIMITS, columns=4,
                                  path=str(tmp_path / "correlation.svg"))
    facets = specvis.correlation(frame, "Osprey", "Tarquin", MEASURES, by="vendor",
                                 hue="group name", facet=True, size=(30, 10))
    agreement = specvis.bland_altman(frame, "Osprey", "LCModel", MEASURES, by="vendor",
                                     hue="group name", limits=[(1, 2.1), (0.1, 0.33),
                                                               (0.5, 1.22), (1, 2.75)],
                                     differences=[(-0.5, 0.5), (-0.1, 0.1), (-0.5, 0.5),
                                                  (-1, 1)], columns=4)

    failed = frame.copy()                       # a fit that failed pairs with nothing
    failed.loc[failed.index[0], "tNAA"] = np.nan
    one = specvis.correlation(failed, "Osprey", "LCModel", ["tNAA"])
    ranged = specvis.bland_altman(frame, "Osprey", "LCModel", MEASURES[:2])   # limits from the data
    for made in (pooled, grouped, facets, agreement, one, ranged):
        assert isinstance(made, Figure)
    assert (tmp_path / "correlation.svg").read_text().startswith("<?xml")


#****************#
#   statistics   #
#****************#
@needs_r
def test_the_statistics_are_those_of_example_md(frame):
    stats = specvis.statistics(frame, MEASURES)
    descriptive = stats["Descriptive Statistics"]
    rows = descriptive.set_index("Group")

    assert rows.loc["tNAA LCModel", "N"] == 35
    assert np.allclose(rows.loc[["tNAA LCModel", "tNAA Osprey", "tNAA Tarquin"], "mean"],
                       [1.4788926, 1.5475494, 1.4922791], rtol=1e-4)
    assert np.allclose(rows.loc[["Ins LCModel", "Ins Osprey", "Ins Tarquin"], "CV"],
                       [0.09629097, 0.12090132, 0.13369120], rtol=1e-4)
    variance = stats["Variance Analysis"].set_index("VarsName")
    assert variance.loc["Ins", "method"] == "ANOVA" and variance.loc["tNAA", "method"] == \
        "KruskalWallis"
    assert isinstance(stats["Posthoc test Variance Analysis"], pd.DataFrame)


@needs_r
def test_the_runtime_says_what_it_has(example):
    found = specvis.runtime()

    assert found["runtime"] in ("local", "docker") and found["ggplot2"]
    assert {"spvs_RainCloud", "spvs_Correlation", "spvs_BlandAltman"} <= set(found["functions"])
    assert specvis.version() == "8e18409"


#**************#
#   defaults   #
#**************#
def test_the_gui_shows_the_measures_the_bridge_draws_by_default():
    from mrsuite.gui import blocks

    assert blocks._MEASURES == repr(list(specvis.MEASURES))
