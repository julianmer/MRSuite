####################################################################################################
#                                             blocks.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: The building blocks the GUI offers, in groups after the MRS consensus papers: data in,  #
#          simulation, preprocessing (Near et al., NMR Biomed 2021), fitting - spectral analysis   #
#          (Near 2021; Kreis et al., NMR Biomed 2021, section 4.2) -, evaluation (quality and      #
#          accuracy), and other operations the toolboxes offer. A block is one step of the         #
#          workflow; where toolboxes each do it, it offers every one of them to swap between.      #
#          Nothing here knows NiceGUI.                                                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from mrsuite.gui import catalog
from mrsuite.gui.catalog import Field

# what a file to load is read as: NIfTI-MRS as it is, ground truth (io.truth), or spec2nii's
# formats that need no options of their own (its text and raw need dwell times and such)
FORMATS = ["nifti", "truth", "twix", "ge", "philips", "dicom", "rda", "bruker", "varian",
           "jmrui", "uih"]

# what flows along a wire; 'any' meets everything. A challenge's truth is concentrations per
# spectrum, as scoring takes it; a simulation's is MRS-Sim's own (amplitudes, signals); a
# basis is a basis set (mrsuite.core.basis); files are one written for another tool; scores
# are scoring.score's table
SPECTRA, RESULTS, TRUTH, SIMULATED, BASIS, FILES, SCORES, ANY = \
    "spectra", "results", "truth", "simulated", "basis", "files", "scores", "any"


#**************************************************************************************************#
#                                           Class Group                                            #
#**************************************************************************************************#
#                                                                                                  #
# A group of blocks, with its colour, taken from the logo.                                         #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Group:
    """A group of blocks, with its colour, taken from the logo.

    Attributes:
        key (str): Its name in code.
        title (str): Its name on screen.
        colour (str): CSS colour.
        help (str): What its blocks do, and where the grouping comes from.
    """

    key: str
    title: str
    colour: str
    help: str


GROUPS = {g.key: g for g in [
    Group("data", "Data", "#9fb0c6",
          "Load spectra and ground truth from files and folders, named as you like, or the "
          "challenges; and basis sets, in any format. Every block exports what it made; "
          "Convert basis set writes its files itself."),
    Group("simulate", "Simulate", "#8ea4ff", "Synthetic spectra with known contents."),
    Group("preprocess", "Preprocess", "#5274f0",
          "Prepare the spectra for analysis: correct imperfections, reduce dimensionality "
          "(Near et al., NMR Biomed 2021, Tables 2-4)."),
    Group("fit", "Fit", "#16bdc6",
          "Spectral analysis: model the spectrum to estimate its signals, in arbitrary units "
          "or, with a water reference, as the toolbox quantifies them (Near 2021; Kreis et "
          "al., NMR Biomed 2021, section 4.2)."),
    Group("models", "Models", "#3fcfae",
          "Trained networks from MRSuite's model zoo on the Hugging Face Hub: a checkpoint "
          "each, loaded by name. WAND is a processing step and gives spectra; the MLPs fit."),
    Group("evaluate", "Evaluate", "#7fe0e6",
          "Judge the spectra and the fits: spectral quality (SNR, linewidth), and fits against "
          "the ground truth with every analysis of MRSValidate that compares the two (Kreis et "
          "al., NMR Biomed 2021, section 4.3), its plots, and a radar of the scores."),
    Group("other", "Other", "#6f7f99",
          "Further operations the toolboxes offer: modelling peaks, cropping, conjugating, "
          "shifting, matching linewidths, resampling, adding and selecting."),
    Group("toolbox", "All functions", "#5a6577", "Every function of every toolbox."),
]}


#**************************************************************************************************#
#                                           Class Port                                             #
#**************************************************************************************************#
#                                                                                                  #
# Where a wire goes in or comes out.                                                               #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Port:
    """Where a wire goes in or comes out.

    Attributes:
        name (str): The argument it gives, or the output it is.
        kind (str): What flows: SPECTRA, RESULTS, TRUTH, SIMULATED, BASIS, FILES or ANY.
        many (bool): Takes any number of wires, as one dictionary by block name.
    """

    name: str
    kind: str = ANY
    many: bool = False


#**************************************************************************************************#
#                                           Class Block                                            #
#**************************************************************************************************#
#                                                                                                  #
# One kind of building block.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Block:
    """One kind of building block.

    A step takes spectra and runs as a pipeline step, so a toolbox's side results ride
    along; its options are the toolboxes' functions that do it, as (bridge, function),
    and its form is the chosen function's arguments. A call is any other Python call,
    written as ``call(...)`` with the fields it declares; a score is a call of
    scoring.score with one of its analyses, whose own arguments join its form.

    Attributes:
        key (str): Its name in code and in a saved graph.
        group (str): A key of GROUPS.
        title (str): Its name on screen.
        options (Tuple[Tuple[str, str], ...]): A step's (bridge, function) choices; the
            bridge may be a module path instead, for a plain function of MRSuite.
        out (str): What a step makes.
        call (str): A call's function, as the script writes it.
        imports (Tuple[str, ...]): The imports a call needs.
        fields (Tuple[Field, ...]): A call's arguments, set in its form.
        inputs (Tuple[Port, ...]): A call's arguments that come in by wire.
        outputs (Tuple[Port, ...]): What a call makes; several when it returns a tuple.
        fixed (Dict[str, str]): Arguments a call always gets, as code.
        wrap (Dict[str, str]): Code a field's value is put into, as '{}' marks.
        analyses (Tuple[Tuple[str, str], ...]): A score's choice of analysis, as
            (module, function).
        model (str): A trained model's class, e.g. 'mrsuite.models.wand.WAND'; the block
            loads a checkpoint of it by name and runs it as a step with its fields.
        folder (str): A fold of its group the palette lists it under, e.g. 'Basis sets';
            '' for the group itself.
        help (str): One line on what it does.
    """

    key: str
    group: str
    title: str
    folder: str = ""
    options: Tuple[Tuple[str, str], ...] = ()
    out: str = SPECTRA
    call: str = ""
    model: str = ""
    imports: Tuple[str, ...] = ()
    fields: Tuple[Field, ...] = ()
    inputs: Tuple[Port, ...] = ()
    outputs: Tuple[Port, ...] = (Port("out"),)
    fixed: Dict[str, str] = field(default_factory=dict)
    wrap: Dict[str, str] = field(default_factory=dict)
    analyses: Tuple[Tuple[str, str], ...] = ()
    help: str = ""

    #**********#
    #   step   #
    #**********#
    @property
    def step(self) -> bool:
        return bool(self.options)

    #************#
    #   offers   #
    #************#
    @property
    def offers(self) -> Tuple[Tuple[str, str], ...]:
        """What it offers to choose between: a step's toolboxes, or a score's analyses."""
        return self.options or self.analyses


#***********#
#   value   #
#***********#
def _value(name: str, default: str = "", positional: bool = False) -> Field:
    return Field(name, "value", [], default, positional)


#************#
#   choice   #
#************#
def _choice(name: str, choices: List[str], default: str = "", positional: bool = False) -> Field:
    return Field(name, "choice", choices, default, positional)


#**********#
#   pick   #
#**********#
def _pick(name: str, choices: Tuple[str, ...]) -> Field:
    return Field(name, "pick", list(choices), "", False)


#*******************#
#   preprocessing   #
#*******************#
# in Near et al.'s order: coils, imperfections, averages, residual water, then the operations
# their Table 4 advises against before fitting
_PREPROCESS = [
    ("coil-combination", "Coil combination",
     [("fsl_mrs", "coilcombine"), ("fid_a", "op_addrcvrs"), ("osprey", "op_addrcvrs"),
      ("mrspeclab", "CoilCombinationSVD"), ("mrspeclab", "CoilCombinationAdaptive")],
     "Combine the receive coils' signals into one."),
    ("eddy-current", "Eddy-current correction",
     [("fsl_mrs", "ecc"), ("fid_a", "op_ecc"), ("osprey", "op_eccKlose"),
      ("mrspeclab", "EddyCurrentCorrection")],
     "Correct the phase eddy currents cause, from the water reference."),
    ("outliers", "Outlier removal",
     [("fsl_mrs", "remove_unlike"), ("fid_a", "op_rmbadaverages"),
      ("mrspeclab", "RemoveBadAverages")],
     "Remove corrupted transients, e.g. from motion."),
    ("alignment", "Frequency & phase alignment",
     [("fsl_mrs", "align"), ("fid_a", "op_alignAverages"), ("osprey", "op_robustSpecReg"),
      ("spant", "align"), ("mrspeclab", "FreqPhaseAlignment")],
     "Correct frequency and phase drift between transients."),
    ("averaging", "Averaging",
     [("fsl_mrs", "average"), ("fid_a", "op_averaging"), ("osprey", "op_averaging"),
      ("spant", "mean_dyns"), ("mrspeclab", "Average")],
     "Average the transients."),
    ("subspectra-alignment", "Sub-spectra alignment",
     [("fsl_mrs", "aligndiff"), ("fid_a", "op_alignMPSubspecs")],
     "Align edited sub-spectra before subtraction."),
    ("subtraction", "Sub-spectra subtraction",
     [("fsl_mrs", "subtract"), ("fid_a", "op_combinesubspecs")],
     "Subtract edited sub-spectra."),
    ("water-removal", "Residual water removal",
     [("fsl_mrs", "remove_peaks"), ("osprey", "op_iterativeWaterFilter"),
      ("fid_a", "op_removeWater"), ("spant", "hsvd_filt")],
     "Remove the residual water peak (HLSVD)."),
    ("phasing", "Phasing",
     [("fsl_mrs", "phase_correct"), ("fid_a", "op_autophase"), ("spant", "auto_phase")],
     "Zero-order phase correction."),
    ("referencing", "Frequency referencing",
     [("fsl_mrs", "shift_to_reference"), ("fid_a", "op_ppmref"), ("osprey", "op_ppmref")],
     "Shift a known peak to its chemical shift."),
    ("baseline", "Baseline correction",
     [("spant", "bc_als"), ("spant", "bc_poly"), ("spant", "bc_spline"),
      ("spant", "bc_constant"), ("fid_a", "op_fddccorr")],
     "Remove a smooth baseline or offset; most fitting tools model the baseline instead."),
    ("apodization", "Apodization",
     [("fsl_mrs", "apodize"), ("fid_a", "op_filter"), ("spant", "lb"),
      ("mrspeclab", "LineBroadening")],
     "Line broadening; not recommended before fitting (Near 2021, Table 4)."),
    ("zero-filling", "Zero-filling",
     [("fsl_mrs", "truncate_or_pad"), ("fid_a", "op_zeropad"), ("spant", "zf"),
      ("mrspeclab", "ZeroPadding")],
     "Pad the FID with zeros; not recommended before fitting (Near 2021, Table 4)."),
]

# what the toolboxes offer beyond the steps above
_OTHER = [
    ("peak-model", "HSVD peak model",
     [("fsl_mrs", "hlsvd_model_peaks"), ("fid_a", "op_HSVDfit")],
     "Model the peaks in a range as decaying sinusoids (HSVD)."),
    ("crop", "Crop", [("fid_a", "op_freqrange"), ("osprey", "op_freqrange"),
                      ("spant", "crop_spec")],
     "Keep a ppm range of the spectrum."),
    ("conjugate", "Conjugate", [("fsl_mrs", "conjugate"), ("fid_a", "op_complexConj")],
     "Mirror the frequency axis, for data stored the other way round."),
    ("shift-phase", "Shift or phase",
     [("fsl_mrs", "fshift"), ("fid_a", "op_freqshift"), ("osprey", "op_freqshift"),
      ("spant", "shift"), ("fsl_mrs", "apply_fixed_phase"), ("fid_a", "op_addphase"),
      ("osprey", "op_addphase"), ("spant", "phase")],
     "A frequency shift or a phase given by hand."),
    ("match-linewidth", "Match linewidth",
     [("fid_a", "op_matchLW"), ("spant", "set_lw")],
     "Broaden to a reference's linewidth, or to a given one."),
    ("time-shift", "Time shift or resample",
     [("fsl_mrs", "tshift"), ("fid_a", "op_leftshift"), ("osprey", "op_leftshift"),
      ("spant", "downsample_mrs_td"), ("spant", "decimate_mrs_td")],
     "Shift the FID in time, drop its first points, or resample it."),
    ("combine", "Combine scans",
     [("fsl_mrs", "add"), ("fid_a", "op_addScans"), ("osprey", "op_addScans"),
      ("fid_a", "op_subtractScans")],
     "Add or subtract two scans."),
    ("select", "Select",
     [("fid_a", "op_takeaverages"), ("osprey", "op_takeaverages"), ("spant", "get_dyns"),
      ("spant", "rm_dyns"), ("fid_a", "op_takesubspec"), ("osprey", "op_takesubspec"),
      ("fid_a", "op_takecoils")],
     "Keep or drop chosen transients, sub-spectra or coils."),
]

_CHALLENGE = "mrsuite.datasets.challenge2016"

# MRSuite's model zoo on the Hub, by block, as pinned (tests/models/test_zoo.py checks each
# one's weights); the GUI lists the organisation's repos itself where it can reach the Hub
ZOO = {"wand": ("MRSuite/wand-ismrm2016",),
       "mlp": ("MRSuite/mlp-supervised-7t-slaser", "MRSuite/mlp-selfsupervised-7t-slaser")}

# MRSValidate's analyses that compare fits with the ground truth (scoring.analyses lists
# them; a test keeps the two the same), a block per module of it, MRSuite's MAE and bias first
_SCORING, _MRSV = "mrsuite.benchmark.scoring", "mrsvalidate.analysis"
_SCORES = [
    ("errors", "Errors",
     [(_SCORING, "mae"), (_SCORING, "bias"),
      *[(f"{_MRSV}.errors", f) for f in ("rmse", "rmese", "mpe", "mepe", "mape",
                                         "bias_corrected_mape", "meape")]],
     "How far the fits are from the truth: absolute, squared and percentage errors, and the "
     "bias (Kreis 2021, section 4.3: accuracy)."),
    ("agreement", "Agreement", [(f"{_MRSV}.agreement", f) for f in ("lin_ccc", "robust_ccc")],
     "How closely the fits follow the truth: Lin's concordance correlation, plain and robust."),
    ("regression", "Correlation & regression",
     [(f"{_MRSV}.regression", f) for f in ("pearson_r", "median_centered_correlation",
                                           "r_squared_direct", "adjusted_r_squared",
                                           "concise_score_original", "concise_score_manuscript",
                                           "theil_sen", "regression_metrics")],
     "The fits against the truth: correlation, R², slope and intercept."),
    ("method-comparison", "Method comparison",
     [(f"{_MRSV}.method_comparison", f) for f in ("bland_altman", "passing_bablok", "deming",
                                                  "equivalence_test_1", "equivalence_test_2",
                                                  "equivalence_test_3")],
     "The fits as a method compared with the truth: Bland-Altman limits of agreement, "
     "Passing-Bablok and Deming regression, equivalence tests."),
    ("error-spread", "Error distribution",
     [(f"{_MRSV}.descriptive", f) for f in ("mean", "median", "variance", "robust_variance",
                                            "robust_sd", "iqr", "tukey_outlier_range")],
     "The errors, fits minus truth: their centre, spread and outlier range."),
    ("normality", "Normality of errors",
     [(f"{_MRSV}.normality", f) for f in ("shapiro_wilk", "kolmogorov_smirnov", "lilliefors",
                                          "anderson_darling", "dagostino_pearson")],
     "Whether the errors, fits minus truth, are normally distributed."),
]
_REFERENCES = ["optimal", "tCr", "water", "none"]
_JUDGED = (Port("methods", RESULTS, many=True), Port("truth", TRUTH))
_SCORED = (Port("scores", SCORES),)
_SCORE = "from mrsuite.benchmark import scoring"

# what a toolbox's own standard pipeline passes where its function has no default of its own
# (FID-A at 1eaa207, Osprey at 98b2e35, plain single-voxel 1H): a new block's form starts with
# them, so it runs as the toolbox runs it, and the script says so. Zero-filling and line
# broadening are not here: FID-A's scripts apply them only to a copy they estimate from.
PIPELINE: Dict[Tuple[str, str], Tuple[Dict[str, Any], str]] = {
    ("fid_a", "op_autophase"): ({"ppmmin": 2.9, "ppmmax": 3.1},
                                "FID-A's run_pressproc_auto.m, line 332 (on creatine)"),
    ("fid_a", "op_ppmref"): ({"ppmmin": 2.9, "ppmmax": 3.1, "ppmrefval": 3.027},
                             "FID-A's run_pressproc_auto.m, line 348 (creatine to 3.027)"),
    ("fid_a", "op_combinesubspecs"): ({"mode": "diff"},
                                      "FID-A's run_megapressproc_auto.m, line 403"),
    ("osprey", "op_robustSpecReg"): ({"seqType": "unedited", "echo": 0},
                                     "Osprey's OspreyProcess.m, lines 61 and 258"),
}

BLOCKS: Dict[str, Block] = {b.key: b for b in [
    # data
    Block("load", "data", "Load data", folder="Files", call="collect.load",
          imports=("from mrsuite import collect",), outputs=(),
          help="Files and folders from anywhere, each file named as you like (data, water, mm, "
               "truth, ...); a folder is searched and its files named by BIDS or by words in "
               "their names. One output per name, in group order."),
    Block("ismrm2016", "data", "2016 fitting challenge", folder="Simulated datasets",
          call="challenge2016.load",
          imports=("from mrsuite.datasets import challenge2016",),
          fields=(_value("n", "None"),),
          outputs=(Port("spectra", SPECTRA), Port("water", SPECTRA), Port("truth", TRUTH)),
          help="The ISMRM 2016 fitting challenge: spectra, water references, ground truth."),
    Block("mrsi2024", "data", "MRSI challenge", folder="Simulated datasets",
          call="mrsi_challenge.load",
          imports=("from mrsuite.datasets import mrsi_challenge",),
          fields=(_value("subject", "", positional=True), _value("voxels", "None")),
          outputs=(Port("spectra", SPECTRA), Port("truth", TRUTH), Port("voxels")),
          help="One subject of the 2024 MRSI challenge, a spectrum per brain voxel."),
    Block("openneuro-ds005371", "data", "HERMES and HERCULES (ds005371)",
          folder="In vivo datasets", call="openneuro.load",
          imports=("from mrsuite.datasets import openneuro",),
          fixed={"dataset": "'ds005371'"}, fields=(_value("subject", "'sub-01'"),),
          outputs=(Port("spectra", SPECTRA), Port("water", SPECTRA)),
          help="OpenNeuro ds005371 (Mikkelsen): edited PRESS and sLASER at 3 T, one subject's "
               "sessions and acquisitions, with their water references."),
    Block("openneuro-ds006812", "data", "COWS (ds006812)", folder="In vivo datasets",
          call="openneuro.load", imports=("from mrsuite.datasets import openneuro",),
          fixed={"dataset": "'ds006812'"}, fields=(_value("subject", "'sub-01'"),),
          outputs=(Port("spectra", SPECTRA), Port("mm", SPECTRA)),
          help="OpenNeuro ds006812 (Igwe): sLASER at 3 T under three water suppression "
               "schemes, one subject's raw acquisitions and their macromolecule ones."),
    Block("basis", "data", "Load basis set", folder="Basis sets", call="io.basis",
          imports=("from mrsuite import io",),
          fields=(_value("files", positional=True), _value("format", "'auto'"),
                  _value("sw", "None"), _value("sf", "None"), _value("te", "None")),
          outputs=(Port("basis", BASIS),),
          help="A basis set in any format, told apart by BasiC, for any fit; sw (Hz), sf "
               "(MHz) and te (ms) where the files do not state them."),
    Block("basis-add", "data", "Add to basis set", folder="Basis sets", call="io.merge_basis",
          imports=("from mrsuite import io",),
          inputs=(Port("basis", BASIS), Port("more", ANY, many=True)),
          fields=(_value("keep", "None"),), outputs=(Port("basis", BASIS),),
          help="Other basis sets' functions, or measured spectra (e.g. a macromolecule "
               "acquisition processed like the data, named as its wire), on the basis' axis; "
               "Choose functions keeps which, one per name."),
    Block("basis-convert", "data", "Convert basis set", folder="Basis sets",
          call="io.convert_basis",
          imports=("from mrsuite import io",), inputs=(Port("basis", BASIS),),
          fields=(_value("folder"), _value("tool"), _value("output", "'default'")),
          outputs=(Port("files", FILES),),
          help="A basis set written for another tool by BasiC; tool is BasiC's id for it, "
               "e.g. 'osprey', 'fsLmrs', 'jmrui', 'niftimrs' (a wrong one is told the rest)."),
    # simulate
    Block("simulation", "simulate", "MRS-Sim spectra", call="mrs_sim.simulate",
          imports=("from mrsuite.simulators import mrs_sim",),
          fields=(_choice("config", ["coil_combined_with_artifacts", "denoising_dataset"],
                          positional=True),
                  _value("n", "", positional=True), _value("seed", "0")),
          outputs=(Port("spectra", SPECTRA), Port("truth", SIMULATED)),
          wrap={"config": "mrs_sim.config({})"},
          help="Spectra simulated by MRS-Sim from one of its configs, and their truth."),
    # preprocess
    *[Block(key, "preprocess", title, options=tuple(options), help=help)
      for key, title, options, help in _PREPROCESS],
    # fit
    Block("lcm-fit", "fit", "Linear-combination fit", out=RESULTS,
          options=(("fsl_mrs", "fit"), ("lcmodel", "fit"), ("osprey", "fit"), ("spant", "fit"),
                   ("tarquin", "fit")),
          help="Model the spectrum as a combination of basis spectra."),
    Block("peak-fit", "fit", "Peak fit (AMARES)", out=RESULTS, options=(("pyamares", "fit"),),
          help="Model the spectrum as peaks with prior knowledge."),
    # models
    Block("wand", "models", "WAND", model="mrsuite.models.wand.WAND",
          imports=("from mrsuite import models",),
          fields=(_pick("checkpoint", ZOO["wand"]), _value("remove", "None"),
                  _value("keep", "None")),
          inputs=(Port("data", SPECTRA),), outputs=(Port("out", SPECTRA),),
          help="Decomposes each spectrum into its metabolites, baseline and artifacts: the "
               "spectrum without the components removed (e.g. ['artifacts']), with only those "
               "kept ('metabolites' for all), or, with neither, every component apart."),
    Block("mlp", "models", "MLP", model="mrsuite.models.mlp.MLP", out=RESULTS,
          imports=("from mrsuite import models",), fields=(_pick("checkpoint", ZOO["mlp"]),),
          inputs=(Port("data", SPECTRA),), outputs=(Port("out", RESULTS),),
          help="Quantifies each spectrum with a multilayer perceptron, trained supervised on "
               "simulated spectra or self-supervised with a signal model."),
    # evaluate
    Block("snr", "evaluate", "SNR", out=ANY,
          options=((_CHALLENGE, "snr"), ("fid_a", "op_getSNR"), ("osprey", "op_getSNR"),
                   ("spant", "calc_spec_snr")),
          help="Signal-to-noise ratio, as each toolbox defines it."),
    Block("linewidth", "evaluate", "Linewidth", out=ANY,
          options=(("fid_a", "op_getLW"), ("osprey", "op_getLW")),
          help="Full width at half maximum."),
    Block("mosae", "evaluate", "MOSAE", call="scoring.score", imports=(_SCORE,),
          fields=(_value("output_dir", "None"),), inputs=_JUDGED, outputs=_SCORED,
          fixed={"references": "['optimal']", "metrics": "{'mosae': scoring.mae}"},
          help="Mean optimal-scale absolute error (Merkofer et al., arXiv:2511.23135, Eq. 16): "
               "each fit scaled to match its truth best, then the MAE; 'all' pools the "
               "metabolites."),
    *[Block(key, "evaluate", title, call="scoring.score", imports=(_SCORE,),
            fields=(_choice("references", _REFERENCES), _value("output_dir", "None")),
            inputs=_JUDGED, outputs=_SCORED, wrap={"references": "[{}]"},
            analyses=tuple(analyses), help=help)
      for key, title, analyses, help in _SCORES],
    Block("bland-altman", "evaluate", "Bland-Altman plot", call="scoring.figure",
          imports=(_SCORE, "from mrsvalidate.analysis import method_comparison",
                   "from mrsvalidate.viz.agreement import plot_bland_altman"),
          fields=(_choice("reference", _REFERENCES), _value("metabolite", "'all'")),
          inputs=_JUDGED, outputs=(Port("figure"),),
          fixed={"analysis": "method_comparison.bland_altman", "plot": "plot_bland_altman"},
          help="Each method's differences from the truth against the truth, with its bias and "
               "limits of agreement."),
    Block("regression-plot", "evaluate", "Regression plot", call="scoring.figure",
          imports=(_SCORE, "from mrsvalidate.analysis import regression",
                   "from mrsvalidate.viz.scatter import plot_regression_result"),
          fields=(_choice("reference", _REFERENCES), _value("metabolite", "'all'")),
          inputs=_JUDGED, outputs=(Port("figure"),),
          fixed={"analysis": "regression.regression_metrics", "plot": "plot_regression_result"},
          help="Each method's fits against the truth, with its regression line."),
    Block("radar", "evaluate", "Radar", call="plot_scores",
          imports=("from mrsuite.benchmark import plot_scores",),
          inputs=(Port("scores", SCORES, many=True),),
          fields=(_choice("reference", _REFERENCES), _value("metabolite", "'all'")),
          outputs=(Port("figure"),),
          help="Score tables on one radar: a spoke per metric, a polygon per method."),
    # other
    *[Block(key, "other", title, options=tuple(options), help=help)
      for key, title, options, help in _OTHER],
]}


#***********#
#   block   #
#***********#
def block(key: str) -> Block:
    """A block by its key: one of BLOCKS, or 'bridge:function' for any toolbox function."""
    if key in BLOCKS:
        return BLOCKS[key]
    bridge, _, function = key.partition(":")
    if bridge not in catalog.BRIDGES or not re.fullmatch(r"[A-Za-z_][\w.]*", function):
        raise KeyError(f"no block {key!r}")
    return Block(key, "toolbox", function, options=((bridge, function),), out=ANY)


#****************#
#   categories   #
#****************#
# where a toolbox keeps a function says what it is for: FID-A's and Osprey's folders, spant's
# name prefixes; what the curated blocks offer keeps their group, a fit is a fit, and the
# remaining steps (spectra in, spectra out) are preprocessing
FOLDERS = {"fid_a": {"inputOutput": "data", "simulationTools": "simulate"},
           "osprey": {"load": "data", "process": "preprocess", "fit": "fit", "quantify": "fit",
                      "plot": "evaluate", "overview": "evaluate"}}
PREFIXES = {"spant": {"read_": "data", "write_": "data", "sim_": "simulate", "fit_": "fit",
                      "calc_": "evaluate", "plot_": "evaluate"}}


def categories(bridge: str) -> Dict[str, str]:
    """Every function of a toolbox with the group it belongs to, a key of GROUPS: 'other'
    where nothing says."""
    curated = {o: b.group for b in BLOCKS.values() for o in b.options}
    steps, folders = set(catalog.functions(bridge)), catalog.folders(bridge)

    def category(name: str) -> str:
        if (bridge, name) in curated:
            return curated[(bridge, name)]
        if name == "fit":
            return "fit"
        group = FOLDERS.get(bridge, {}).get(folders.get(name, ""))
        if group:
            return group
        for prefix, group in PREFIXES.get(bridge, {}).items():
            if name.startswith(prefix):
                return group
        return "preprocess" if name in steps else "other"
    return {name: category(name) for name in catalog.functions(bridge, True)}


#***********#
#   label   #
#***********#
def label(option: Tuple[str, str]) -> str:
    """An option as a person reads it: the toolbox, and the function where a toolbox
    offers several."""
    where, function = option
    where = "MRSValidate" if where.startswith(f"{_MRSV}.") else where
    return {"fsl_mrs": "FSL-MRS", "fid_a": "FID-A", "osprey": "Osprey", "spant": "spant",
            "mrspeclab": "MRSpecLAB", "lcmodel": "LCModel", "tarquin": "TARQUIN",
            "pyamares": "pyAMARES", "vespa": "Vespa", _CHALLENGE: "2016 challenge",
            _SCORING: "MRSuite"}.get(where, where) + f" · {function}"


#***********#
#   ports   #
#***********#
def ports(b: Block, arguments: List[str]) -> Tuple[List[Port], List[Port]]:
    """
    A block's inputs and outputs: a step takes spectra, and each of its arguments, which
    a wire gives or else the form; it makes what it makes. An argument named basis takes
    a basis set (every fit's does); one named files takes files, as a block reads them.
    A load's outputs are its names, which the graph knows (``Graph.ports``).

    Args:
        b (Block): The block.
        arguments (List[str]): Its arguments, each an input of its own.
    """
    given = [Port(n, BASIS if n == "basis" else FILES if n == "files" else ANY)
             for n in arguments]
    if b.step:
        return [Port("data", SPECTRA), *given], [Port("out", b.out)]
    return [*b.inputs, *given], list(b.outputs)
