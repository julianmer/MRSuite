####################################################################################################
#                                       test_mrsvalidate.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 11/11/26                                                                                #
#                                                                                                  #
# Purpose: The MRSValidate bridge. Scoring fits against ground truth: on constructed fits, where   #
#          the right answer is known exactly; then MRSValidate's own toy example, reproduced       #
#          through MRSuite's path; the score tables on the radar; and one real FSL-MRS fit         #
#          against the real 2016 challenge truth, where the naming and scaling problems live.      #
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

pytest.importorskip("mrsvalidate")
from mrsvalidate.analysis import (agreement, descriptive, errors, method_comparison, regression,
                                  resampling)
from mrsvalidate.exceptions import AlignmentError

import mrsuite
from mrsuite.benchmark import optimal_scale, plot_scores
from mrsuite.benchmark.radar import aspect
from mrsuite.bridges.mrsvalidate import (analyses, bias, figure, mae, observations, score,
                                         version)
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


#**************#
#   a bridge   #
#**************#
def test_the_bridge_is_a_toolbox_of_mrsuite_with_a_version():
    assert mrsuite.mrsvalidate.score is score
    assert version() and version()[0].isdigit()


#******************#
#   observations   #
#******************#
def test_observations_carry_the_ids_the_unit_and_the_method(tmp_path):
    fit = FitResult(method="lcm", metabolites=["Cr", "NAA"],
                    concentrations={"raw": {"Cr": 9.0, "NAA": 9.0},
                                    "internal": {"Cr": 1.0, "NAA": 1.5},
                                    "molarity": {"Cr": 8.0, "NAA": 9.0}})
    true, est = observations([fit], [{"Cr": 1.0, "NAA": 1.5}], "none", scaling="internal",
                             metabolites=np.array(["NAA"]), ids=["subject-7"])

    assert est.parameters.names == ("NAA",)
    assert est.column("NAA") == pytest.approx([1.5])
    assert list(true.sample_ids) == ["subject-7"]
    assert (true.model_id, est.model_id) == ("truth", "lcm")

    true, est = observations([fit], [{"Cr": 10.0, "NAA": 10.0}], "water")
    assert est.column("NAA") == pytest.approx([9.0])
    assert {p.unit for p in true.parameters} == {"mM"}
    assert list(true.sample_ids) == ["0"]
    table = score({"m": [fit]}, [{"Cr": 10.0, "NAA": 10.0}], ["water"], {"mae": mae}, tmp_path)
    assert table.set_index("metabolite").loc["NAA", "value"] == pytest.approx(1.0)


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


#****************#
#   mismatched   #
#****************#
def test_mismatches_are_rejected_saying_why():
    with pytest.raises(ValueError, match="unknown reference"):
        observations([built(NAA=1.0)], [{"NAA": 1.0}], "Cr")
    with pytest.raises(ValueError, match=r"every fit of every method has \[\]"):
        score({"a": [built(NAA=1.0)], "b": [built(naa=1.0)]}, [{"NAA": 1.0}], ["none"],
              {"mae": mae}, "/nonexistent")


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
    for figure_, data in (("figure01", "figure01_correlation_vs_agreement.csv"),
                          ("figure06", "figure06_matching_marginals_wrong_pairing.csv")):
        rows = read(f"data/{data}")
        for expected in read(f"analysis/{figure_}_metrics.csv"):
            pairs = [r for r in rows if r["scenario"] == expected["scenario"]]
            asked = {m: metrics[m] for m in metrics if m in expected}
            table = score({"toy": [built(X=float(r["prediction_mM"])) for r in pairs]},
                          [{"X": float(r["ground_truth_mM"])} for r in pairs], ["none"],
                          {m: f for m, (f, _) in asked.items()},
                          tmp_path / figure_ / expected["scenario"],
                          ids=[r["sample_id"] for r in pairs])
            got = table.query("metabolite == 'X'").set_index("metric")

            for m, (_, field) in asked.items():   # the toy outputs are rounded to 4 places
                assert got.loc[m, field] == pytest.approx(float(expected[m]), abs=5e-5), \
                    (figure_, expected["scenario"], m)


#***********#
#   radar   #
#***********#
def test_score_tables_draw_on_one_radar():
    """A spoke per metric, from each table's score column, a polygon per method; a metric
    given twice is said."""
    truth = [{"NAA": 1.0}, {"NAA": 2.0}, {"NAA": 1.5}, {"NAA": 3.0}]
    methods = {"a": [built(NAA=v) for v in (1.1, 2.2, 1.4, 3.1)],
               "b": [built(NAA=v) for v in (1.0, 2.5, 1.9, 2.6)]}
    tables = {"mae": score(methods, truth, ["none"], {"mae": mae}),
              "ccc": score(methods, truth, ["none"], {"ccc": agreement.lin_ccc})}

    ax = plot_scores(tables)

    polygons = [line for line in ax.get_lines() if line.get_label() in ("a", "b")]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Agreement\nccc", "Errors\nMAE"]
    assert list(polygons[0].get_ydata()) == pytest.approx([1.0, 1.0, 1.0]), "a: best in both"
    assert list(polygons[1].get_ydata()) == pytest.approx([0.05, 0.05, 0.05])
    with pytest.raises(ValueError, match=r"given twice: mae \(none\)"):
        plot_scores({"x": tables["mae"], "y": tables["mae"]})

    more = {**tables, "mape": score(methods, truth, ["none"], {"mape": errors.mape}),
            "r": score(methods, truth, ["none"], {"pearson_r": regression.pearson_r})}
    ax = plot_scores(more)                       # a corner is the mean of its measures
    assert [t.get_text() for t in ax.get_xticklabels()] == \
        ["Agreement\nccc", "Errors\nMAE, MAPE", "Regression\nPearson r"]
    swapped = tables["mae"].assign(metric="rmse", analysis="rmse",   # b best here, worst on
                                   method=tables["mae"]["method"].map({"a": "b", "b": "a"}))
    ax = plot_scores({"ccc": tables["ccc"], "mae": tables["mae"], "rmse": swapped})
    corner = {line.get_label(): line.get_ydata()[1] for line in ax.get_lines()
              if line.get_label() in ("a", "b")}                       # MAE: so even
    assert corner["a"] == pytest.approx(corner["b"]) and 0.05 < corner["b"] < 1


def test_a_radar_by_measure_has_a_spoke_per_measure_with_its_range():
    """A spoke per measure, named, a signed one by its magnitude, every spoke 0 to 1; a
    measure on two references is labelled with each."""
    truth = [{"NAA": v} for v in (1.0, 2.0, 1.5, 3.0)]
    methods = {"a": [built(NAA=t["NAA"] + 0.1) for t in truth],
               "b": [built(NAA=t["NAA"] - 0.3) for t in truth]}
    table = score(methods, truth, ["none"], {"mae": mae, "bias": bias})

    ax = plot_scores(table, by="measure")
    assert [t.get_text() for t in ax.get_xticklabels()] == \
        ["|bias|", "MAE"]
    two = pd.concat([table.query("metric == 'mae'"),
                     table.query("metric == 'mae'").assign(reference="optimal")])
    assert [t.get_text() for t in plot_scores(two).get_xticklabels()] == \
        ["Errors\nMAE (none), MAE (optimal)"]
    both = [{"NAA": t["NAA"], "Cr": 2.0} for t in truth]           # no pooled rows: a spoke
    each = score({k: [built(NAA=t["NAA"] + d, Cr=2.0 + d) for t in both]   # per metabolite
                  for k, d in (("a", 0.1), ("b", -0.3))}, both, ["none"], {"mae": mae})
    each = each.query("metabolite != 'all'")
    assert [t.get_text() for t in plot_scores(each, by="measure").get_xticklabels()] == \
        ["MAE Cr", "MAE NAA"]
    with pytest.raises(ValueError, match="by 'aspect' or by 'measure'"):
        plot_scores(table, by="metric")


def test_a_bland_altman_bias_is_best_at_zero_on_the_radar():
    """Bland-Altman's score is its bias, signed: the method off by +0.05 beats the one off
    by -0.3, though -0.3 is the lower number."""
    truth = [{"NAA": v} for v in (1.0, 2.0, 1.5, 3.0)]
    methods = {"a": [built(NAA=t["NAA"] + 0.05) for t in truth],
               "b": [built(NAA=t["NAA"] - 0.3) for t in truth]}
    ax = plot_scores(score(methods, truth, ["none"], {"ba": method_comparison.bland_altman}))

    polygons = {line.get_label(): line for line in ax.get_lines()
                if line.get_label() in ("a", "b")}
    assert list(polygons["a"].get_ydata()) == pytest.approx([1.0, 1.0])
    assert list(polygons["b"].get_ydata()) == pytest.approx([0.05, 0.05])


def test_an_aspect_is_the_module_of_its_analysis():
    assert aspect("ccc", "lin_ccc") == "Agreement"
    assert aspect("deming", "deming") == "Method comparison"


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
