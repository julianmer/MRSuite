####################################################################################################
#                                        test_benchmark.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Scoring fits against ground truth. Mostly on constructed fits, where the right answer   #
#          is known exactly; then MRSValidate's own toy example, reproduced through MRSuite's      #
#          path; and one real FSL-MRS fit against the real 2016 challenge truth, where the naming  #
#          and scaling problems live.                                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import csv
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mrsvalidate")   # the 'benchmark' extra
from mrsvalidate.analysis import (agreement, descriptive, errors, method_comparison, regression,
                                  resampling)
from mrsvalidate.exceptions import AlignmentError

from mrsuite.benchmark import (analyses, bias, coverage, figure, mae, observations,
                               optimal_scale, plot_scores, radar, score)
from mrsuite.benchmark.radar import aspect
from mrsuite.core.result import FitResult

# MRSValidate's repository, for its toy example, which its package does not ship
TOY = (Path(os.environ.get("MRSUITE_MRSVALIDATE", "/nonexistent")) / "assets" / "publication"
       / "examples" / "toy_outputs")


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
    true, fit = observations([built(Cr=8.0, PCr=2.0, NAA=15.0)],
                             [{"Cr": 0.8, "PCr": 0.2, "NAA": 1.5}], "tCr")

    assert fit.values == pytest.approx(true.values)
    assert true.column("NAA") == pytest.approx([1.5])


#*************#
#   only cr   #
#*************#
def test_creatine_alone_as_tcr_is_not_scored_against_itself():
    """With no PCr, as in the MRSI challenge, Cr / tCr is 1 on both sides whatever the
    fit, and would pull every pooled score towards perfect."""
    true, fit = observations([built(Cr=8.0, NAA=15.0)], [{"Cr": 0.8, "NAA": 1.5}], "tCr")

    assert true.parameters.names == ("NAA",)


#**************#
#   moieties   #
#**************#
def test_a_molecule_split_into_moieties_is_scored_once():
    """The MRSI challenge's truth gives Cr and Cr39 the same amplitude; a fit that shares
    it out unevenly between them still has the molecule right."""
    true, fit = observations([built(Cr=0.5, Cr39=1.5, PE40=0.7, NAA=3.0)],
                             [{"Cr": 1.0, "Cr39": 1.0, "PE": 0.7, "PE40": 0.7, "NAA": 3.0}],
                             "none")

    assert true.parameters.names == ("Cr", "NAA", "PE")
    assert fit.values == pytest.approx(true.values)

    # a half the fit failed on leaves the molecule to the other half
    true, fit = observations([built(Cr=1.0, Cr39=np.nan, NAA=3.0)],
                             [{"Cr": 1.0, "Cr39": 1.0, "NAA": 3.0}], "none")
    assert fit.values == pytest.approx(true.values)


#**********#
#   none   #
#**********#
def test_none_compares_the_values_as_given():
    true, fit = observations([built(NAA=1.5)], [{"NAA": 1.0}], "none")

    assert fit.column("NAA") - true.column("NAA") == pytest.approx([0.5])


#************#
#   chosen   #
#************#
def test_the_caller_picks_metabolites_scaling_and_ids():
    fit = FitResult(method="test", metabolites=["Cr", "NAA"],
                    concentrations={"raw": {"Cr": 9.0, "NAA": 9.0},
                                    "internal": {"Cr": 1.0, "NAA": 1.5}})
    true, est = observations([fit], [{"Cr": 1.0, "NAA": 1.5}], "none", scaling="internal",
                             metabolites=np.array(["NAA"]), ids=["subject-7"])

    assert est.parameters.names == ("NAA",)
    assert est.column("NAA") == pytest.approx([1.5])
    assert list(true.sample_ids) == ["subject-7"]


#***********#
#   water   #
#***********#
def test_water_reads_the_water_scaled_concentrations_in_mm(tmp_path):
    fit = FitResult(method="test", metabolites=["NAA"],
                    concentrations={"raw": {"NAA": 1e4}, "molarity": {"NAA": 9.0}})
    true, est = observations([fit], [{"NAA": 10.0}], "water")

    assert est.column("NAA") == pytest.approx([9.0])
    assert {p.unit for p in true.parameters} == {"mM"}
    table = score({"m": [fit]}, [{"NAA": 10.0}], ["water"], {"mae": mae}, tmp_path)
    assert table.set_index("metabolite").loc["NAA", "value"] == pytest.approx(1.0)


#*************#
#   optimal   #
#*************#
def test_the_optimal_scale_removes_any_scale_and_leaves_out_macromolecules():
    """MOSAE needs neither water nor tCr, and macromolecules fitted wrongly do not move
    the scale, as the paper leaves them out: 'Mac' in the 2016 challenge, 'mm_0_9' in
    the MRSI one."""
    true, fit = observations([built(Cr=14.0, NAA=21.0, Mac=100.0, mm_0_9=50.0)],
                             [{"Cr": 2.0, "NAA": 3.0, "Mac": 1.0, "mm_0_9": 1.0}], "optimal")

    assert true.parameters.names == ("Cr", "NAA")
    assert fit.values == pytest.approx(true.values)


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


#************#
#   failed   #
#************#
def test_a_metabolite_a_fit_failed_on_does_not_cost_the_others(tmp_path):
    names = ["Cr", "Glu", "NAA"]
    fits = [built(Cr=2.0, Glu=np.nan, NAA=4.0), built(Cr=2.0, Glu=2.0, NAA=2.0)]
    table = score({"m": fits}, [dict.fromkeys(names, 1.0)] * 2, ["optimal"], {"mae": mae},
                  tmp_path)
    used = table.set_index("metabolite")["n_used"]

    assert used["NAA"] == 2 and used["Glu"] == 1
    assert table.set_index("metabolite").loc["NAA", "value"] == pytest.approx(0.0)


#**********#
#   sums   #
#**********#
def test_sums_are_not_scored_whoever_reports_them():
    """FSL-MRS writes 'Cr+PCr', the challenge truth 'Cr + PCr', TARQUIN 'TNAA'; only the
    singles are scored, so nothing counts twice."""
    fit = built(Cr=1.0, PCr=1.0, NAA=1.0, **{"Cr+PCr": 2.0, "Cr + PCr": 2.0, "TNAA": 1.0})
    true, _ = observations([fit], [{"Cr": 1.0, "PCr": 1.0, "NAA": 1.0, "Cr + PCr": 2.0}],
                           "none")

    assert true.parameters.names == ("Cr", "NAA", "PCr")


#****************#
#   mismatched   #
#****************#
def test_mismatches_are_rejected_saying_why():
    with pytest.raises(ValueError, match="line up spectrum for spectrum"):
        observations([built(NAA=1.0)], [{"NAA": 1.0}, {"NAA": 1.0}], "none")
    with pytest.raises(ValueError, match="unknown reference"):
        observations([built(NAA=1.0)], [{"NAA": 1.0}], "Cr")
    with pytest.raises(ValueError, match=r"nothing to score: every fit has \['naa'\]"):
        observations([built(naa=1.0)], [{"NAA": 1.0}], "none")
    with pytest.raises(ValueError, match=r"every fit of every method has \[\]"):
        score({"a": [built(NAA=1.0)], "b": [built(naa=1.0)]}, [{"NAA": 1.0}], ["none"],
              {"mae": mae}, "/nonexistent")


#***************#
#   reference   #
#***************#
def test_a_missing_reference_says_what_is_available():
    with pytest.raises(KeyError, match=r"fit 0 has no \['PCr'\]; it has \['Cr', 'NAA'\]"):
        observations([built(Cr=1.0, NAA=1.0)], [{"Cr": 1.0, "PCr": 1.0, "NAA": 1.0}], "tCr")
    with pytest.raises(KeyError, match=r"no Cr or PCr to make tCr; it has \['NAA'\]"):
        observations([built(NAA=1.0)], [{"NAA": 1.0}], "tCr")
    with pytest.raises(KeyError, match="no 'molarity'"):
        observations([built(NAA=1.0)], [{"NAA": 1.0}], "water")


#**********#
#   bias   #
#**********#
def test_bias_separates_a_systematic_offset_from_noise(tmp_path):
    """Off by +0.2 and -0.2 has the same MAE as off by +0.2 twice; only bias tells
    them apart."""
    truth = [{"NAA": 1.0}, {"NAA": 1.0}]
    methods = {"noisy": [built(NAA=1.2), built(NAA=0.8)],
               "biased": [built(NAA=1.2), built(NAA=1.2)]}
    table = score(methods, truth, ["none"], {"mae": mae, "bias": bias}, tmp_path)
    got = table.query("metabolite == 'NAA'").set_index(["method", "metric"])["value"]

    assert got["noisy", "mae"] == pytest.approx(got["biased", "mae"])
    assert got["noisy", "bias"] == pytest.approx(0.0)
    assert got["biased", "bias"] == pytest.approx(0.2)


#**************#
#   analysis   #
#**************#
def test_the_analysis_column_names_the_analysis_that_ran(tmp_path):
    """MRSValidate's summary writes a result's own 'method' field (a bootstrap's
    'percentile') over the analysis' name."""
    def difference(reference, prediction, sample_ids):
        return resampling.bootstrap_ci(lambda x, y: np.mean(y - x), reference, prediction,
                                       sample_ids=sample_ids, n_resamples=100, seed=0,
                                       method="percentile")

    table = score({"m": [built(NAA=1.0), built(NAA=2.0)]}, [{"NAA": 1.0}] * 2, ["none"],
                  {"mae": mae, "difference": difference}, tmp_path)

    assert dict(zip(table.metric, table.analysis)) == {"mae": "mean",
                                                       "difference": "bootstrap_ci"}


#**********#
#   same   #
#**********#
def test_every_method_is_scored_on_the_same_metabolites(tmp_path):
    """A method that skips Glu would otherwise pool over an easier set."""
    def fit(**conc):
        return FitResult(method="test", metabolites=sorted(conc),
                         concentrations={"internal": conc})

    truth = [{"Cr": 1.0, "Glu": 1.0, "NAA": 1.0}, {"Glu": 1.0, "NAA": 1.0}]
    methods = {"a": [fit(Cr=1.0, Glu=1.0, NAA=1.0)] * 2, "b": [fit(Cr=1.0, NAA=1.0)] * 2}
    table = score(methods, truth, ["none"], {"mae": mae}, tmp_path, scaling="internal")

    assert sorted(table.query("method == 'a'").metabolite) == ["NAA", "all"]
    assert sorted(table.query("method == 'b'").metabolite) == ["NAA", "all"]
    with pytest.raises(AlignmentError):   # the ids reach the pairing, which wants them unique
        score(methods, truth, ["none"], {"mae": mae}, tmp_path, scaling="internal",
              ids=["s", "s"])


#***********#
#   mosae   #
#***********#
def test_mae_pooled_on_the_optimal_reference_is_the_mosae(tmp_path):
    """Eq. 16: the mean of |w * estimate - truth| over spectra and metabolites."""
    rng = np.random.default_rng(1)
    names = ["Cr", "Glu", "NAA", "PCr"]
    estimate, truth = rng.uniform(1, 10, (5, 4)), rng.uniform(1, 10, (5, 4))
    table = score({"m": [built(**dict(zip(names, e))) for e in estimate]},
                  [dict(zip(names, t)) for t in truth], ["optimal"], {"mae": mae}, tmp_path)
    mosae = np.abs(optimal_scale(estimate, truth)[:, None] * estimate - truth).mean()

    assert sorted(table.metabolite) == ["Cr", "Glu", "NAA", "PCr", "all"]
    assert table.set_index("metabolite").loc["all", "value"] == pytest.approx(mosae)
    assert set(table.status) == {"valid"}


#**********#
#   runs   #
#**********#
def test_runs_never_share_a_folder_and_steps_never_a_name(tmp_path):
    """Names with dots in them would otherwise meet: (Cr, NAA.mae) and (Cr.NAA, mae)."""
    truth = [{"Cr": 1.0, "Cr.NAA": 1.0}, {"Cr": 2.0, "Cr.NAA": 2.0}]
    fits = [built(Cr=1.0, **{"Cr.NAA": 1.0}), built(Cr=2.0, **{"Cr.NAA": 2.0})]
    metrics = {"mae": mae, "NAA.mae": bias}
    first = score({"x": fits}, truth, ["none"], metrics, tmp_path)
    second = score({"x": fits}, truth, ["none"], metrics, tmp_path)

    assert len(first) == len(second) == 3 * 2
    assert len(list(tmp_path.iterdir())) == 2


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


#*********#
#   toy   #
#*********#
def test_mrsvalidates_toy_example_is_reproduced(tmp_path):
    """Figures 1 and 6 of MRSValidate's toy example score a truth against a prediction;
    through MRSuite's fits, references and plan the numbers must come out as its own."""
    if not TOY.exists():
        pytest.skip("set MRSUITE_MRSVALIDATE to MRSValidate's repository for its toy example")

    def read(name):
        with open(TOY / name) as f:
            return list(csv.DictReader(f))

    metrics = {"pearson_r": (regression.pearson_r, "r"), "lin_ccc": (agreement.lin_ccc, "rho_c"),
               "r_squared": (regression.r_squared_direct, "value"),
               "rmse_mM": (errors.rmse, "value"), "mean_error_mM": (bias, "value"),
               "mae_mM": (mae, "value")}
    for figure, data in (("figure01", "figure01_correlation_vs_agreement.csv"),
                         ("figure06", "figure06_matching_marginals_wrong_pairing.csv")):
        rows = read(f"data/{data}")
        for expected in read(f"analysis/{figure}_metrics.csv"):
            pairs = [r for r in rows if r["scenario"] == expected["scenario"]]
            asked = {m: metrics[m] for m in metrics if m in expected}
            table = score({"toy": [built(X=float(r["prediction_mM"])) for r in pairs]},
                          [{"X": float(r["ground_truth_mM"])} for r in pairs], ["none"],
                          {m: f for m, (f, _) in asked.items()},
                          tmp_path / figure / expected["scenario"],
                          ids=[r["sample_id"] for r in pairs])
            got = table.query("metabolite == 'X'").set_index("metric")

            for m, (_, field) in asked.items():   # the toy outputs are rounded to 4 places
                assert got.loc[m, field] == pytest.approx(float(expected[m]), abs=5e-5), \
                    (figure, expected["scenario"], m)


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
    assert list(lines[0].get_ydata()) == pytest.approx([1.0, 0.05, 0.05, 1.0, 1.0]), "floored"
    assert list(lines[1].get_ydata()) == pytest.approx([0.05, 1.0, 1.0, 1.0, 0.05])
    assert [t.get_text() for t in ax.get_xticklabels()] == \
        ["mae\n0.2 to 0.1", "bias\n0.2 to 0.1", "lin_ccc\n0.5 to 1", "rmse\n0.3 to 0.3"]

    ax = radar(values, limits={"mae": (0.4, 0.0)}, ax=Figure().add_subplot(projection="polar"))
    assert [line for line in ax.get_lines() if line.get_label() == "lcmodel"][0].get_ydata()[0] \
        == pytest.approx(0.75)


def test_an_empty_radar_waits_for_scores():
    ax = plot_scores({})

    assert ax.get_lines() == [] and "No scores yet" in ax.get_title()
    assert ax.figure.get_facecolor() == ax.get_facecolor() == (0.0, 0.0, 0.0, 0.0), "no white box"
    assert aspect("mae", "mean") == "Errors" and aspect("ccc", "lin_ccc") == "Agreement"
    assert aspect("deming", "deming") == "Method comparison"


def test_score_tables_draw_on_one_radar():
    """A spoke per metric, from each table's score column, a polygon per method; a
    reference not scored, or a metric twice, is said."""
    truth = [{"NAA": 1.0}, {"NAA": 2.0}, {"NAA": 1.5}, {"NAA": 3.0}]
    methods = {"a": [built(NAA=v) for v in (1.1, 2.2, 1.4, 3.1)],
               "b": [built(NAA=v) for v in (1.0, 2.5, 1.9, 2.6)]}
    tables = {"mae": score(methods, truth, ["none"], {"mae": mae}),
              "ccc": score(methods, truth, ["none"], {"ccc": agreement.lin_ccc})}

    ax = plot_scores(tables)

    polygons = [line for line in ax.get_lines() if line.get_label() in ("a", "b")]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Agreement", "Errors"]
    assert list(polygons[0].get_ydata()) == pytest.approx([1.0, 1.0, 1.0]), "a: best in both"
    assert list(polygons[1].get_ydata()) == pytest.approx([0.05, 0.05, 0.05])
    with pytest.raises(ValueError, match=r"given twice: mae \(none\)"):
        plot_scores({"x": tables["mae"], "y": tables["mae"]})

    more = {**tables, "mape": score(methods, truth, ["none"], {"mape": errors.mape}),
            "r": score(methods, truth, ["none"], {"pearson_r": regression.pearson_r})}
    ax = plot_scores(more)                       # a corner is the mean of its measures
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Agreement", "Errors", "Regression"]
    errors_ = [line for line in ax.get_lines() if line.get_label() == "b"][0].get_ydata()[1]
    assert errors_ == pytest.approx((0.05 + 0.05) / 2, abs=0.05) or errors_ <= 0.5


#**************#
#   analyses   #
#**************#
def test_every_analysis_runs_on_its_own_argument_names():
    """bland_altman takes the fits as 'candidate', deming as 'test' and needs arguments of
    its own, by partial; a describing one runs on the errors, fits minus truth; the score
    column is each result's first number; with no folder given, one of its own."""
    truth = [{"NAA": 1.0, "Cr": 2.0}, {"NAA": 2.0, "Cr": 1.0}, {"NAA": 1.5, "Cr": 1.5}]
    methods = {"m": [built(NAA=1.1, Cr=2.0), built(NAA=2.1, Cr=1.2), built(NAA=1.4, Cr=1.6)]}
    deming = partial(method_comparison.deming, reference_error_variance=1.0,
                     test_error_variance=1.0)

    table = score(methods, truth, ["none"], {"ba": method_comparison.bland_altman,
                                             "median": descriptive.median, "deming": deming})

    pooled = table.query("metabolite == 'all'").set_index("metric")
    true = np.array([2.0, 1.0, 1.5, 1.0, 2.0, 1.5])               # Cr, then NAA
    fit = np.array([2.0, 1.2, 1.6, 1.1, 2.1, 1.4])
    ids = np.array([f"{s}/{m}" for m in ("Cr", "NAA") for s in "012"])
    assert pooled.loc["ba", "score"] == pytest.approx(
        method_comparison.bland_altman(true, fit, ids).bias)
    assert pooled.loc["median", "score"] == pytest.approx(0.1)
    assert pooled.loc["deming", "score"] == pytest.approx(
        deming(true, fit, ids).slope) and pooled.loc["deming", "status"] == "valid"


def test_analyses_are_those_that_compare_fits_with_truth():
    found = analyses()

    assert {"errors.rmse", "agreement.lin_ccc", "method_comparison.bland_altman",
            "regression.regression_metrics", "descriptive.median",
            "normality.shapiro_wilk"} <= set(found)
    assert not {"error_consistency.error_consistency", "distribution.ecwas_kossler_mukherjee",
                "normality.normality_test_suite", "information.standardized_error",
                "monte_carlo.monte_carlo_condition"} & set(found)


def test_what_says_nothing_of_the_fits_has_no_score_and_leaves_nothing_behind(tmp_path,
                                                                            monkeypatch):
    """An interval has two numbers, so no one score; a radar refuses a missing score; a
    run without a folder of its own leaves none."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)
    truth = [{"NAA": 1.0}, {"NAA": 2.0}, {"NAA": 1.5}, {"NAA": 3.0}]
    methods = {"a": [built(NAA=v) for v in (1.1, 2.2, 1.4, 3.1)]}

    table = score(methods, truth, ["none"], {"tukey": descriptive.tukey_outlier_range})

    assert table["score"].isna().all() and table["low"].notna().all()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match=r"no score for tukey \(none\) \(a\)"):
        plot_scores(table)
    assert not {"standardization.standardize", "descriptive.cv"} & set(analyses())


def test_a_figure_draws_one_plot_per_method():
    from mrsvalidate.viz.agreement import plot_bland_altman

    truth = [{"NAA": 1.0}, {"NAA": 2.0}, {"NAA": 1.5}]
    methods = {"a": [built(NAA=v) for v in (1.1, 2.2, 1.4)],
               "b": [built(NAA=v) for v in (1.0, 2.5, 1.9)]}

    fig = figure(methods, truth, "none", method_comparison.bland_altman, plot_bland_altman)

    assert [ax.get_title() for ax in fig.axes][:2] == ["a · all", "b · all"]
    with pytest.raises(ValueError, match="no 'Glu' scored"):
        figure(methods, truth, "none", method_comparison.bland_altman, plot_bland_altman,
               metabolite="Glu")


#***************#
#   real data   #
#***************#
def test_a_real_fit_scores_against_the_real_truth(fsl_mrs_fit):
    """The naming and scaling problems only show up on the real pair."""
    from mrsuite.bridges.fsl_mrs import to_result
    from mrsuite.datasets import challenge2016

    truth = challenge2016.load(n=1)[2]
    true, fit = observations([to_result(fsl_mrs_fit)], truth, "tCr")

    assert len(true.parameters) > 15, f"only {len(true.parameters)} metabolites matched"
    assert not any("+" in m for m in true.parameters.names)
    assert np.abs(fit.values - true.values).max() < 1.0, "normalisation looks wrong"
    # Cr and PCr make tCr, so their ratios add up to 1 on either side by construction
    for side in (true, fit):
        assert side.column("Cr") + side.column("PCr") == pytest.approx([1.0])
