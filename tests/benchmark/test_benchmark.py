####################################################################################################
#                                        test_benchmark.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: MRSuite's own part of a benchmark, without MRSValidate: fits paired with the ground     #
#          truth on each reference, on constructed fits where the right answer is known exactly;  #
#          the optimal scale behind MOSAE; what the fits cover; and the radar.                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pandas as pd
import pytest

from mrsuite.benchmark import coverage, metabolites, optimal_scale, pair, plot_scores, radar
from mrsuite.benchmark.radar import aspect
from mrsuite.core.result import FitResult


#***********#
#   built   #
#***********#
def built(**conc):
    """A fit carrying nothing but concentrations, so a test states its own truth."""
    return FitResult(method="test", metabolites=sorted(conc),
                     concentrations={"raw": dict(conc)})


#*********#
#   tcr   #
#*********#
def test_both_sides_are_divided_by_total_creatine():
    """Fits report amplitudes on a scale of their own; this one is perfect, 10x over."""
    true, fit, names = pair([built(Cr=8.0, PCr=2.0, NAA=15.0)],
                            [{"Cr": 0.8, "PCr": 0.2, "NAA": 1.5}], "tCr")

    assert fit == pytest.approx(true)
    assert true[:, names.index("NAA")] == pytest.approx([1.5])


#*************#
#   only cr   #
#*************#
def test_creatine_alone_as_tcr_is_not_scored_against_itself():
    """With no PCr, as in the MRSI challenge, Cr / tCr is 1 on both sides whatever the
    fit, and would pull every pooled score towards perfect."""
    _, _, names = pair([built(Cr=8.0, NAA=15.0)], [{"Cr": 0.8, "NAA": 1.5}], "tCr")

    assert names == ["NAA"]


#**************#
#   moieties   #
#**************#
def test_a_molecule_split_into_moieties_is_scored_once():
    """The MRSI challenge's truth gives Cr and Cr39 the same amplitude; a fit that shares
    it out unevenly between them still has the molecule right."""
    true, fit, names = pair([built(Cr=0.5, Cr39=1.5, PE40=0.7, NAA=3.0)],
                            [{"Cr": 1.0, "Cr39": 1.0, "PE": 0.7, "PE40": 0.7, "NAA": 3.0}],
                            "none")

    assert names == ["Cr", "NAA", "PE"]
    assert fit == pytest.approx(true)

    # a half the fit failed on leaves the molecule to the other half
    true, fit, _ = pair([built(Cr=1.0, Cr39=np.nan, NAA=3.0)],
                        [{"Cr": 1.0, "Cr39": 1.0, "NAA": 3.0}], "none")
    assert fit == pytest.approx(true)


#**********#
#   none   #
#**********#
def test_none_compares_the_values_as_given():
    true, fit, _ = pair([built(NAA=1.5)], [{"NAA": 1.0}], "none")

    assert (fit - true).ravel() == pytest.approx([0.5])


#************#
#   chosen   #
#************#
def test_the_caller_picks_metabolites_and_scaling():
    fit = FitResult(method="test", metabolites=["Cr", "NAA"],
                    concentrations={"raw": {"Cr": 9.0, "NAA": 9.0},
                                    "internal": {"Cr": 1.0, "NAA": 1.5}})
    _, est, names = pair([fit], [{"Cr": 1.0, "NAA": 1.5}], "none", scaling="internal",
                         metabolites=np.array(["NAA"]))

    assert names == ["NAA"]
    assert est.ravel() == pytest.approx([1.5])


#***********#
#   water   #
#***********#
def test_water_reads_the_water_scaled_concentrations():
    fit = FitResult(method="test", metabolites=["NAA"],
                    concentrations={"raw": {"NAA": 1e4}, "molarity": {"NAA": 9.0}})
    true, est, _ = pair([fit], [{"NAA": 10.0}], "water")

    assert est.ravel() == pytest.approx([9.0]) and true.ravel() == pytest.approx([10.0])


#*************#
#   optimal   #
#*************#
def test_the_optimal_scale_removes_any_scale_and_leaves_out_macromolecules():
    """MOSAE needs neither water nor tCr, and macromolecules fitted wrongly do not move
    the scale, as the paper leaves them out: 'Mac' in the 2016 challenge, 'mm_0_9' in
    the MRSI one."""
    true, fit, names = pair([built(Cr=14.0, NAA=21.0, Mac=100.0, mm_0_9=50.0)],
                            [{"Cr": 2.0, "NAA": 3.0, "Mac": 1.0, "mm_0_9": 1.0}], "optimal")

    assert names == ["Cr", "NAA"]
    assert fit == pytest.approx(true)


#*************#
#   minimum   #
#*************#
def test_the_optimal_scale_is_the_exact_l1_minimiser():
    """Eq. 15 of Merkofer et al., checked against a fine search like the paper's own,
    with estimates of either sign; a metabolite a fit failed on carries no weight."""
    rng = np.random.default_rng(0)
    estimate, truth = rng.uniform(-1, 5, (20, 18)), rng.uniform(0.1, 5, (20, 18))
    grid = np.linspace(0, 5, 50001)

    for w, e, t in zip(optimal_scale(estimate, truth), estimate, truth):
        assert np.abs(w * e - t).sum() <= np.abs(grid[:, None] * e - t).sum(-1).min() + 1e-12

    assert optimal_scale(np.array([[1.0, 2.0]]), np.array([[-1.0, -2.0]])) == [0.0]
    assert optimal_scale(np.zeros((1, 3)), np.ones((1, 3))) == [0.0]
    assert optimal_scale(np.array([[1.0, np.nan, 2.0, np.inf]]),
                         np.array([[2.0, 1.0, 4.0, 1.0]])) == [2.0]
    assert optimal_scale(np.ones((1, 3)), np.array([[np.nan, np.nan, 2.0]])) == [2.0]


#**********#
#   sums   #
#**********#
def test_sums_are_not_scored_whoever_reports_them():
    """FSL-MRS writes 'Cr+PCr', the challenge truth 'Cr + PCr', TARQUIN 'TNAA'; only the
    singles are scored, so nothing counts twice."""
    fit = built(Cr=1.0, PCr=1.0, NAA=1.0, **{"Cr+PCr": 2.0, "Cr + PCr": 2.0, "TNAA": 1.0})
    _, _, names = pair([fit], [{"Cr": 1.0, "PCr": 1.0, "NAA": 1.0, "Cr + PCr": 2.0}], "none")

    assert names == ["Cr", "NAA", "PCr"]


#****************#
#   mismatched   #
#****************#
def test_mismatches_are_rejected_saying_why():
    with pytest.raises(ValueError, match="line up spectrum for spectrum"):
        pair([built(NAA=1.0)], [{"NAA": 1.0}, {"NAA": 1.0}], "none")
    with pytest.raises(ValueError, match="unknown reference"):
        pair([built(NAA=1.0)], [{"NAA": 1.0}], "Cr")
    with pytest.raises(ValueError, match=r"nothing to score: every fit has \['naa'\]"):
        pair([built(naa=1.0)], [{"NAA": 1.0}], "none")
    with pytest.raises(ValueError, match=r"every fit of every method has \[\]"):
        metabolites({"a": [built(NAA=1.0)], "b": [built(naa=1.0)]}, [{"NAA": 1.0}], ["none"])


#*******************#
#   every method   #
#*******************#
def test_every_method_is_scored_on_what_all_of_them_fitted():
    """A method that skips Glu would otherwise pool over an easier set."""
    methods = {"a": [built(Cr=1.0, Glu=1.0, NAA=1.0)], "b": [built(Cr=1.0, NAA=1.0)]}

    assert metabolites(methods, [{"Cr": 1.0, "Glu": 1.0, "NAA": 1.0}], ["none"]) == ["Cr", "NAA"]
    assert metabolites(methods, [{"Cr": 1.0}], ["none"], metabolites=["NAA"]) == ["NAA"]


#***************#
#   reference   #
#***************#
def test_a_missing_reference_says_what_is_available():
    with pytest.raises(KeyError, match=r"fit 0 has no \['PCr'\]; it has \['Cr', 'NAA'\]"):
        pair([built(Cr=1.0, NAA=1.0)], [{"Cr": 1.0, "PCr": 1.0, "NAA": 1.0}], "tCr")
    with pytest.raises(KeyError, match=r"no Cr or PCr to make tCr; it has \['NAA'\]"):
        pair([built(NAA=1.0)], [{"NAA": 1.0}], "tCr")
    with pytest.raises(KeyError, match="no 'molarity'"):
        pair([built(NAA=1.0)], [{"NAA": 1.0}], "water")


#**************#
#   coverage   #
#**************#
def test_coverage_reports_what_was_not_fitted_everywhere():
    """Scores cover what every spectrum has on both sides, so a method that fits half
    the metabolites, or drops one now and then, would look strong without this."""
    truth = [{"NAA": 1.0, "Cr": 1.0, "Gln": 1.0, "GSH": 1.0}, {"NAA": 1.0, "Cr": 1.0}]
    fits = [built(NAA=1.0, Cr=1.0, GSH=1.0, TNAA=1.0), built(NAA=1.0, Glx=1.0)]

    assert coverage(fits, truth) == {"missing": ["Cr", "GSH", "Gln"], "extra": ["Glx", "TNAA"]}
    internal = FitResult(method="test", metabolites=["NAA"],
                         concentrations={"internal": {"NAA": 1.0}})
    assert coverage([internal], [{"NAA": 1.0}], scaling="internal") == {"missing": [],
                                                                        "extra": []}


#***********#
#   radar   #
#***********#
def test_radar_draws_one_polygon_per_method_best_at_the_rim():
    """Each spoke runs from the worst method to the best: MAE lower is better, bias by its
    magnitude, Lin's CCC higher is better; limits given override; a spoke where all score
    the same sits at the rim."""
    from matplotlib.figure import Figure

    values = pd.DataFrame({"mae": [0.1, 0.2], "bias": [-0.2, 0.1], "lin_ccc": [0.5, 1.0],
                           "rmse": [0.3, 0.3]}, index=["lcmodel", "tarquin"])
    ax = radar(values, ax=Figure().add_subplot(projection="polar"))
    lines = [line for line in ax.get_lines() if not line.get_label().startswith("_")]

    assert [line.get_label() for line in lines] == ["lcmodel", "tarquin"]
    assert [t.get_text().strip() for t in ax.texts] == ["0", "0.2", "0.4", "0.6", "0.8", "1"]
    assert list(lines[0].get_ydata()) == pytest.approx([1.0, 0.05, 0.05, 1.0, 1.0]), "floored"
    assert list(lines[1].get_ydata()) == pytest.approx([0.05, 1.0, 1.0, 1.0, 0.05])
    assert [t.get_text() for t in ax.get_xticklabels()] == \
        ["mae\n0.2 to 0.1", "bias\n0.2 to 0.1", "lin_ccc\n0.5 to 1", "rmse\n0.3 to 0.3"]

    ax = radar(values, limits={"mae": (0.4, 0.0)}, ax=Figure().add_subplot(projection="polar"))
    assert [line for line in ax.get_lines() if line.get_label() == "lcmodel"][0].get_ydata()[0] \
        == pytest.approx(0.75)


def test_an_empty_radar_waits_for_scores():
    ax = plot_scores({})

    assert not [line for line in ax.get_lines() if not line.get_label().startswith("_")], \
        "no polygon"
    assert len(ax.get_lines()) == 5 and "No scores yet" in ax.get_title(), "the empty web"
    assert [t.get_text().strip() for t in ax.texts][::5] == ["0", "1"], "its scale"
    assert ax.figure.get_facecolor() == ax.get_facecolor() == (0.0, 0.0, 0.0, 0.0), "no white box"
    assert aspect("mae", "mean") == "Errors"             # MRSuite's own, without MRSValidate


def test_a_radar_reads_each_analysis_its_own_way_round():
    """A slope is best at 1, Shapiro-Wilk's W the higher the better, and an equivalence
    test's intercept best at 0, whichever side of it."""
    rows = pd.DataFrame({"method": ["a", "b", "c"], "reference": "none", "metabolite": "all"})
    tables = {name: rows.assign(metric=name, analysis=name, score=scores) for name, scores in
              (("passing_bablok", [1.02, 0.6, 1.5]), ("shapiro_wilk", [0.99, 0.9, 0.95]),
               ("equivalence_test_1", [0.01, -1.0, 0.5]))}

    ax = plot_scores(tables, by="measure")

    assert [t.get_text() for t in ax.get_xticklabels()] == \
        ["|equivalence_test_1|", "|passing_bablok - 1|", "shapiro_wilk"]
    radius = {line.get_label(): list(line.get_ydata())[:3] for line in ax.get_lines()
              if line.get_label() in ("a", "b", "c")}
    assert radius["a"] == pytest.approx([1.0, 1.0, 1.0]), "a: best on all three"
    assert radius["b"][0] == radius["b"][2] == pytest.approx(0.05), "b: -1 and W 0.9 worst"
    assert radius["c"][1] == pytest.approx(0.05), "c: the slope furthest from 1"
