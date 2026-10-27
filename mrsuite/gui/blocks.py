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
from mrsuite.simulators import bases
from mrsuite.simulators.mrsandbox import SEQUENCES

# spant's pulse sequences with ideal pulses, as its simulators take them
SPANT_SEQUENCES = ["seq_press_ideal", "seq_slaser_ideal", "seq_steam_ideal",
                   "seq_steam_ideal_cof", "seq_steam_ideal_young", "seq_mega_press_ideal",
                   "seq_spin_echo_ideal", "seq_cpmg_ideal", "seq_pulse_acquire",
                   "seq_press_2d_shaped"]

# what a file to load is read as: NIfTI-MRS as it is, ground truth (io.truth), or spec2nii's
# formats that need no options of their own (its text and raw need dwell times and such)
FORMATS = ["nifti", "truth", "twix", "ge", "philips", "dicom", "rda", "bruker", "varian",
           "jmrui", "uih"]

# what flows along a wire; 'any' meets everything. A challenge's truth is concentrations per
# spectrum, as scoring takes it; a simulation's is MRS-Sim's own (amplitudes, signals); a
# basis is a basis set (mrsuite.core.basis); files are one written for another tool; scores
# are scoring.score's table; a sequence is a pulse sequence a simulator plays
SPECTRA, RESULTS, TRUTH, SIMULATED, BASIS, FILES, SCORES, SEQUENCE, ANY = \
    "spectra", "results", "truth", "simulated", "basis", "files", "scores", "sequence", "any"


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
    Group("simulate", "Simulate", "#8ea4ff",
          "Basis sets and spectra with known contents, by every simulator: FID-A's, spant's, "
          "FSL-MRS's and Vespa's density-matrix simulations, FSL-MRS's and spant's synthetic "
          "spectra, MRS-Sim, and MRSandbox's virtual scanner on its virtual brain; the pulse "
          "sequences they play, from MRSandbox's library or any Pulseq file, and the toolboxes' "
          "RF pulse tools under All functions."),
    Group("augment", "Augment", "#7a8ff6",
          "Vary the spectra, physically consistently, with Augmentrum: the signal (noise, line "
          "broadening, baselines, phase and frequency, amplitude), artifacts (residual water, "
          "macromolecules, eddy currents, spurious echoes, contaminant peaks), and what the scan "
          "sampled (coils, transients, k-space); a water reference wired in is varied alike."),
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
    Group("mrsi", "MRSI", "#3a9ac7",
          "Spectroscopic imaging on a spatial NIfTI-MRS: spant's k-space, mask and voxel "
          "tools and FSL-MRS's alignment, phasing and lipid removal across the voxels; then "
          "the voxels apart, so every step and fitter takes them, and the fits back on the "
          "grid as metabolite maps. FID-A's CSI functions are under All functions (they need "
          "the MATLAB runtime)."),
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
        kind (str): What flows: SPECTRA, RESULTS, TRUTH, SIMULATED, BASIS, FILES, SCORES,
            SEQUENCE or ANY.
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
      ("mrspeclab", "CoilCombinationSVD"), ("mrspeclab", "CoilCombinationAdaptive"),
      ("suspect", "combine_channels")],
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
      ("spant", "align"), ("mrspeclab", "FreqPhaseAlignment"), ("suspect", "alignment")],
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
      ("fid_a", "op_removeWater"), ("spant", "hsvd_filt"), ("suspect", "water_removal")],
     "Remove the residual water peak (HLSVD)."),
    ("phasing", "Phasing",
     [("fsl_mrs", "phase_correct"), ("fid_a", "op_autophase"), ("spant", "auto_phase"),
      ("suspect", "phasing")],
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
      ("mrspeclab", "LineBroadening"), ("suspect", "apodization")],
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
    ("conjugate", "Conjugate", [("fsl_mrs", "conjugate"), ("fid_a", "op_complexConj"),
                                ("nifti_mrs", "conjugate")],
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
    ("fid_a", "op_alignAverages"): ({"tmax": 0.2, "med": "n"},
                                    "FID-A's run_pressproc.m, lines 39 and 197 (tmaxin, 'n'): "
                                    "its own estimate of tmax fails where no point clears SNR 5"),
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
    Block("suspect-read", "data", "Read with suspect", folder="Files", call="suspect.read",
          imports=("from mrsuite.bridges import suspect",),
          fields=(_value("files", ""),
                  _choice("format", ["twix", "rda", "sdat", "siemens_dicom", "dicom", "bruker"],
                          "'twix'")),
          outputs=(Port("spectra", SPECTRA),),
          help="Raw scanner files read by suspect's own readers (Twix, RDA, SDAT, DICOM, "
               "Bruker), one spectrum each, as acquired; spec2nii (Load data) reads more."),
    Block("inspector-spectrum", "data", "INSPECTOR synthetic spectrum", folder="Simulated datasets",
          call="mr_science_lab.spectrum",
          imports=("from mrsuite.datasets import mr_science_lab",),
          fields=(_choice("model", ["perfect", "imperfect"], "'perfect'"),
                  _choice("broadening", [3, 8, 20], "8")),
          outputs=(Port("spectra", SPECTRA),),
          help="A brain spectrum simulated in INSPECTOR by the MR SCIENCE Lab (Siemens 3 T), "
               "perfect or with its macromolecules changed, at 3, 8 or 20 Hz² of broadening."),
    Block("agnostic", "data", "AGNOSTIC", folder="Simulated datasets", call="agnostic.load",
          imports=("from mrsuite.datasets import agnostic",),
          fields=(_value("shard", "21"), _value("n", "10"),
                  _value("components", "('Metab', 'MM', 'water', 'noise')")),
          outputs=(Port("spectra", SPECTRA), Port("truth", TRUTH)),
          help="AGNOSTIC's synthetic PRESS spectra (Gudmundson et al.; 1.4 to 3.1 T, TE 10 to "
               "80 ms), the chosen signals summed, every metabolite's amplitude as truth; a "
               "shard downloaded from Dryad into MRSuite's data folder."),
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
    Block("mrsi2024-volume", "data", "MRSI challenge volume", folder="Simulated datasets",
          call="mrsi_challenge.volume",
          imports=("from mrsuite.datasets import mrsi_challenge",),
          fields=(_value("subject", "", positional=True),
                  _choice("signal", ["all", "all-nuisance", "meta"], "'all'"), _value("z", "None")),
          outputs=(Port("spectra", SPECTRA), Port("truth"), Port("mask")),
          help="One subject of the 2024 MRSI challenge as it is, a whole volume or one slice "
               "of x by y voxels, for the MRSI steps; its truth as maps, and its brain mask."),
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
          outputs=(Port("spectra", SPECTRA), Port("water", SPECTRA), Port("mm", SPECTRA),
                   Port("mm_water", SPECTRA)),
          help="OpenNeuro ds006812 (Igwe): sLASER at 3 T under three water suppression "
               "schemes, one subject's raw acquisitions with their water references, and "
               "the macromolecule ones with theirs."),
    Block("marss-basis", "data", "MARSS basis set (MR SCIENCE Lab)", folder="Basis sets",
          call="mr_science_lab.basis",
          imports=("from mrsuite.datasets import mr_science_lab",),
          fields=(_choice("sequence", ["press", "slaser", "steam"], "'press'"),
                  _choice("vendor", ["siemens", "ge", "philips", "bruker"], "'siemens'"),
                  _value("te", "30"), _value("bandwidth", "4000"), _value("points", "2048"),
                  _value("tm", "None")),
          outputs=(Port("basis", BASIS),),
          help="The MR SCIENCE Lab's MARSS-simulated basis sets for the vendors' PRESS, sLASER "
               "and STEAM with their exact pulses and timings, fetched on first use."),
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
    Block("dimensions", "data", "Dimensions",
          options=(("nifti_mrs", "remove_dim"), ("nifti_mrs", "reorder"),
                   ("nifti_mrs", "reshape"), ("nifti_mrs", "split")),
          help="Drop, reorder, reshape or split the higher dimensions of NIfTI-MRS (coils, "
               "transients, sub-spectra), with nifti_mrs_tools."),
    # simulate: basis sets
    Block("fid-a-basis", "simulate", "FID-A basis set", folder="Basis set simulation", call="bases.fid_a",
          imports=("from mrsuite.simulators import bases",),
          fields=(_value("metabolites", repr(list(bases.METABOLITES))),
                  _choice("seq", ["press", "steam", "se", "laser"], "'press'"),
                  _value("te1", "11.0"), _value("te2", "19.0"), _value("points", "2048"),
                  _value("bandwidth", "2000.0"), _value("field", "3.0"),
                  _value("linewidth", "2.0")),
          outputs=(Port("basis", BASIS),),
          help="A basis set simulated by FID-A with ideal pulses, a density-matrix "
               "simulation per metabolite of its spin systems (sim_lcmrawbasis); te1 and "
               "te2 in ms (TE and TM for STEAM)."),
    Block("spant-basis", "simulate", "spant basis set", folder="Basis set simulation", call="bases.spant",
          imports=("from mrsuite.simulators import bases",),
          fields=(_choice("pul_seq", SPANT_SEQUENCES, "'seq_press_ideal'"),
                  _value("frequency", "123.2"), _value("bandwidth", "2000.0"),
                  _value("points", "1024"), _value("centre", "4.65"),
                  _choice("lcm_compat", [True, False], "False")),
          outputs=(Port("basis", BASIS),),
          help="spant's 1H brain basis set through one of its ideal sequences "
               "(sim_basis_1h_brain); the sequence's timings in seconds (TE1, TE2, TE3, TE, "
               "TM) under further arguments."),
    Block("fsl-basis", "simulate", "FSL-MRS basis simulation", folder="Basis set simulation",
          call="bases.fsl_mrs", imports=("from mrsuite.simulators import bases",),
          fields=(_value("sequence_json", positional=True),
                  _value("metabolites", repr(list(bases.METABOLITES))), _value("te", "None"),
                  _value("autophase", "None")),
          outputs=(Port("basis", BASIS),),
          help="A basis set simulated by FSL-MRS's fsl_mrs_sim from a sequence described in "
               "its JSON (RF pulses, gradients, delays, coherence filter); needs FSL-MRS "
               "with fsl-denmatsim."),
    # Vespa's own defaults: B0, points and sweep width of vespa/common/constants.py
    # (DEFAULT_B0, DEFAULT_SPECTRAL_POINTS, DEFAULT_SWEEP_WIDTH), the timings its
    # mrs_experiment.DEFAULT_LOOP, the metabolites its bundled 'Example - Spin-Echo sequence'
    Block("vespa-basis", "simulate", "Vespa basis set", folder="Basis set simulation",
          call="bases.vespa", imports=("from mrsuite.simulators import bases",),
          fields=(_value("metabolites", repr(list(bases.VESPA_METABOLITES))),
                  _choice("sequence", ["PRESS Ideal", "STEAM Ideal", "Spin-Echo"],
                          "'PRESS Ideal'"),
                  _value("timings", "None"), _value("frequency", "64.0"),
                  _value("points", "2048"), _value("bandwidth", "2000.0")),
          outputs=(Port("basis", BASIS),),
          help="A basis set simulated by Vespa's Simulation on PyGAMMA (built on first use, "
               "with make, g++ and SWIG): its bundled ideal sequences on its metabolites' spin "
               "systems, unbroadened; timings in ms, one per loop (PRESS: TE1, TE2; STEAM: TE, "
               "TM; Spin-Echo: TE), None for Vespa's default of 0 ms; frequency in MHz."),
    Block("mrsandbox-basis", "simulate", "MRSandbox basis set", folder="Basis set simulation",
          call="mrsandbox.basis", imports=("from mrsuite.simulators import mrsandbox",),
          fields=(_choice("sequence", list(SEQUENCES), "'press'"),
                  _value("brain", "'icbm152'"), _value("scanner", "'generic_3t'"),
                  _value("position", "(0.0, -45.0, 35.0)"), _value("size", "20.0"),
                  _value("seed", "0"), _value("points", "2048"), _value("bandwidth", "4000.0"),
                  _choice("imperfections", [True, False], "True")),
          outputs=(Port("basis", BASIS),),
          help="The basis set of MRSandbox's own simulation: every metabolite in the voxel "
               "rendered by its engine through the sequence, of its library or wired in, at "
               "the relaxation and transmit the seed draws."),
    # simulate: spectra
    Block("simulation", "simulate", "MRS-Sim spectra", folder="Spectrum simulation", call="mrs_sim.simulate",
          imports=("from mrsuite.simulators import mrs_sim",),
          fields=(_choice("config", ["ismrm2016", "coil_combined_with_artifacts",
                                     "denoising_dataset"], positional=True),
                  _value("n", "", positional=True), _value("seed", "0")),
          outputs=(Port("spectra", SPECTRA), Port("truth", SIMULATED)),
          wrap={"config": "mrs_sim.config({})"},
          help="Spectra simulated by MRS-Sim from one of its configs - the 2016 challenge's "
               "distribution, or its own with artifacts - and their truth."),
    Block("simntg-basis", "simulate", "SimnTG triglyceride", folder="Basis set simulation",
          call="bases.simntg", imports=("from mrsuite.simulators import bases",),
          fields=(_value("chain_length", "18"), _value("double_bonds", "(6, 9)"),
                  _choice("seq", ["press", "steam", "se", "laser"], "'press'"),
                  _value("te1", "11.0"), _value("te2", "19.0"), _value("points", "2048"),
                  _value("bandwidth", "2000.0"), _value("field", "3.0"),
                  _value("linewidth", "2.0")),
          outputs=(Port("basis", BASIS),),
          help="Lipid basis functions: SimnTG's triglyceride spin systems (chain length, double "
               "bonds) through FID-A's ideal sequence, one per system and their sum; needs the "
               "MATLAB runtime until SimnTG's round(x, 1) is fixed upstream."),
    Block("fsl-synthetic", "simulate", "FSL-MRS synthetic spectra", folder="Spectrum simulation",
          call="synthetic.fsl_mrs", imports=("from mrsuite.simulators import synthetic",),
          inputs=(Port("basis", BASIS),),
          fields=(_value("n", "1"), _value("concentrations", "None"),
                  _value("broadening", "(9.0, 0.0)"), _value("shift", "0.0"),
                  _value("phi0", "0.0"), _value("phi1", "0.0"), _value("noise", "0.1"),
                  _value("baseline", "None"), _value("seed", "0")),
          outputs=(Port("spectra", SPECTRA), Port("truth", TRUTH)),
          help="Spectra synthesised by FSL-MRS from a basis set: its signal model at given "
               "or standard concentrations, broadened, shifted, phased, with noise; the "
               "concentrations as truth."),
    Block("spant-brain", "simulate", "spant brain spectra", folder="Spectrum simulation",
          call="synthetic.spant", imports=("from mrsuite.simulators import synthetic",),
          fields=(_value("n", "1"), _value("type", "'normal_v2'"),
                  _choice("pul_seq", SPANT_SEQUENCES, "'seq_slaser_ideal'"),
                  _value("frequency", "123.2"), _value("bandwidth", "2000.0"),
                  _value("points", "1024"), _value("centre", "4.65"), _value("seed", "0")),
          outputs=(Port("spectra", SPECTRA), Port("truth", TRUTH)),
          help="Brain-like spectra simulated by spant (sim_brain_1h) through one of its ideal "
               "sequences, its amplitudes as truth; timings under further arguments."),
    Block("digital-phantom", "simulate", "Digital phantom spectrum", folder="Spectrum simulation",
          call="digital_phantom.scan", imports=("from mrsuite.simulators import digital_phantom",),
          fields=(_choice("sequence", ["PRESS", "sLASER"], "'PRESS'"), _value("te", "35"),
                  _value("size", "20.0"), _value("seed", "0")),
          outputs=(Port("spectra", SPECTRA), Port("truth", TRUTH)),
          help="A voxel of MRS-Digital-Phantom's digital brain (its MRiLab skeleton) as a "
               "spectrum: metabolites from its tissue tables, macromolecules, lipids, water and "
               "noise, on the Philips basis set of that sequence and TE (PRESS 30, 35, 68, 120; "
               "sLASER 10, 30, 35, 120 ms); its tissue-weighted concentrations as truth."),
    Block("mrsandbox-scan", "simulate", "MRSandbox scan", folder="Spectrum simulation",
          call="mrsandbox.scan", imports=("from mrsuite.simulators import mrsandbox",),
          fields=(_choice("sequence", list(SEQUENCES), "'press'"),
                  _value("brain", "'icbm152'"), _value("scanner", "'generic_3t'"),
                  _value("position", "(0.0, -45.0, 35.0)"), _value("size", "20.0"),
                  _value("n", "1"), _value("seed", "0"), _value("points", "2048"),
                  _value("bandwidth", "4000.0"), _value("averages", "1"),
                  _choice("imperfections", [True, False], "True"),
                  _choice("mm", [True, False], "True"), _choice("lipids", [True, False], "True")),
          outputs=(Port("spectra", SPECTRA), Port("truth", TRUTH), Port("basis", BASIS)),
          help="A voxel of MRSandbox's virtual brain acquired by its virtual scanner, every "
               "quantity drawn from its priors per seed, through a sequence of its library or "
               "one wired in; the drawn concentrations (mM) as truth, and the basis set of "
               "the very sequence it played."),
    Block("mrsandbox-mrsi", "simulate", "MRSandbox MRSI scan", folder="Spectrum simulation",
          call="mrsandbox.mrsi", imports=("from mrsuite.simulators import mrsandbox",),
          fields=(_choice("sequence", list(SEQUENCES), "'press'"),
                  _value("brain", "'icbm152'"), _value("scanner", "'generic_3t'"),
                  _value("centre", "(0.0, -20.0, 35.0)"), _value("matrix", "(16, 16)"),
                  _value("voxel", "10.0"), _value("seed", "0"), _value("points", "2048"),
                  _value("bandwidth", "4000.0"), _value("averages", "None"),
                  _choice("imperfections", [True, False], "True"),
                  _choice("mm", [True, False], "True"), _choice("lipids", [True, False], "True")),
          outputs=(Port("spectra", SPECTRA), Port("truth", TRUTH), Port("basis", BASIS)),
          help="A slice of MRSandbox's virtual brain acquired by its virtual scanner as one "
               "spatial NIfTI-MRS, phase encoded and decoded; the drawn concentrations (mM) as "
               "truth, and the basis set of the very sequence it played."),
    # simulate: RF pulses
    Block("rf-sinc", "simulate", "Sinc pulse", folder="RF pulses", call="pulses.sinc",
          imports=("from mrsuite.simulators import pulses",),
          fields=(_value("lobes", "4"), _value("points", "256"),
                  _choice("kind", ["exc", "ref", "inv"], "'exc'")),
          outputs=(Port("pulse"),),
          help="A sinc RF pulse by FID-A: its lobes and points, for excitation, refocusing or "
               "inversion, with its time-bandwidth product."),
    Block("rf-gauss", "simulate", "Gaussian pulse", folder="RF pulses", call="pulses.gauss",
          imports=("from mrsuite.simulators import pulses",),
          fields=(_value("duration", "2.0"), _value("bandwidth", "2000.0"), _value("points", "256"),
                  _choice("kind", ["exc", "ref", "inv"], "'exc'"), _value("offset", "0.0")),
          outputs=(Port("pulse"),),
          help="A Gaussian RF pulse by FID-A: duration (ms), bandwidth (Hz), points, kind and "
               "frequency offset; FID-A's other pulses (hyperbolic secant, GOIA, dual-band) are "
               "under All functions."),
    Block("rf-profile", "simulate", "Pulse profile", folder="RF pulses", call="pulses.profile",
          imports=("from mrsuite.simulators import pulses",),
          inputs=(Port("pulse"),),
          fields=(_value("duration", "5.0"), _value("span", "10.0"), _value("centre", "0.0")),
          outputs=(Port("profile"),),
          help="An RF pulse's frequency profile by FID-A's Bloch simulation: the pulse's "
               "duration (ms; FID-A has no default), the span and its centre (kHz, FID-A's "
               "10 and 0); |Mxy| and Mz across it, at the pulse's nominal peak B1."),
    # simulate: pulse sequences
    *[Block(f"seq-{kind}", "simulate", title, folder="Pulse sequences", call="mrsandbox.sequence",
            imports=("from mrsuite.simulators import mrsandbox",), fixed={"kind": repr(kind)},
            fields=tuple(_value(name, repr(value)) for name, value in SEQUENCES[kind][1].items()),
            outputs=(Port("sequence", SEQUENCE),),
            help=f"{title} from MRSandbox's library, its timings in seconds"
                 f"{'; a physical design, which a Pulseq file can hold' * localized}.")
      for kind, title, localized in [("press", "PRESS", False), ("steam", "STEAM", False),
                                     ("slaser", "sLASER", False),
                                     ("megapress", "MEGA-PRESS", False),
                                     ("hermes", "HERMES", False), ("hercules", "HERCULES", False),
                                     ("press-localized", "PRESS, localized", True),
                                     ("steam-localized", "STEAM, localized", True),
                                     ("slaser-localized", "sLASER, localized", True)]],
    Block("pulseq-read", "simulate", "Pulseq file", folder="Pulse sequences",
          call="mrsandbox.read_pulseq",
          imports=("from mrsuite.simulators import mrsandbox",),
          fields=(_value("path", positional=True), _value("centre", "4.65"),
                  _value("readout", "0")),
          outputs=(Port("sequence", SEQUENCE),),
          help="A sequence read from a Pulseq .seq file by MRSandbox, block by block, one "
               "readout of it."),
    Block("pulseq-write", "simulate", "Write Pulseq", folder="Pulse sequences",
          call="mrsandbox.write_pulseq",
          imports=("from mrsuite.simulators import mrsandbox",),
          inputs=(Port("sequence", SEQUENCE),),
          fields=(_value("path", positional=True), _value("points", "2048"),
                  _value("bandwidth", "4000.0"), _value("frequency", "127.73"),
                  _value("centre", "4.65")),
          outputs=(Port("files", FILES),),
          help="A physical sequence written as a Pulseq .seq file by MRSandbox through "
               "pypulseq; an idealized one is refused."),
    # preprocess
    *[Block(key, "preprocess", title, options=tuple(options), help=help)
      for key, title, options, help in _PREPROCESS],
    # augment
    Block("noise", "augment", "Noise", folder="Signal", options=(("augmentrum", "noise"),),
          help="Complex Gaussian acquisition noise, at an SNR or a sigma, absolute or relative."),
    Block("line-broadening", "augment", "Line broadening", folder="Signal",
          options=(("augmentrum", "line_broadening"),),
          help="Lorentzian, Gaussian or Voigt broadening of every line, in Hz."),
    Block("baseline-distortion", "augment", "Baseline", folder="Signal",
          options=(("augmentrum", "baseline"),),
          help="A baseline added: a random walk, a B-spline or a polynomial, scaled to the data."),
    Block("phase-shift", "augment", "Phase shift", folder="Signal",
          options=(("augmentrum", "phase"),),
          help="Zero- and first-order phase, in degrees."),
    Block("frequency-shift", "augment", "Frequency shift", folder="Signal",
          options=(("augmentrum", "frequency_shift"),), help="The spectrum shifted, in Hz."),
    Block("amplitude-scaling", "augment", "Amplitude scaling", folder="Signal",
          options=(("augmentrum", "amplitude"),),
          help="The whole signal scaled by a factor, or one drawn from a range."),
    Block("residual-water", "augment", "Residual water", folder="Artifacts",
          options=(("augmentrum", "residual_water"),),
          help="Residual water added at the centre, with its lobes."),
    Block("macromolecules", "augment", "Macromolecules", folder="Artifacts",
          options=(("augmentrum", "macromolecules"),),
          help="A macromolecular background added: parametrized, measured or supplied."),
    Block("eddy-currents", "augment", "Eddy currents", folder="Artifacts",
          options=(("augmentrum", "eddy_current"),),
          help="Eddy-current phase distortions along the FID."),
    Block("spurious-echoes", "augment", "Spurious echoes", folder="Artifacts",
          options=(("augmentrum", "spurious_echoes"),),
          help="Spurious echoes added along the FID, on all or a fraction of the transients."),
    Block("artificial-peaks", "augment", "Contaminant peaks", folder="Artifacts",
          options=(("augmentrum", "artificial_peaks"),),
          help="Contaminant peaks added at given or random positions."),
    Block("coil-sampling", "augment", "Coil sampling", folder="Sampling",
          options=(("augmentrum", "coil_sampling"),),
          help="A receive array drawn from the coils there, or synthesised."),
    Block("average-sampling", "augment", "Transient sampling", folder="Sampling",
          options=(("augmentrum", "average_sampling"),),
          help="A subset of the transients, as a shorter scan would have."),
    Block("transient-synthesis", "augment", "Transient synthesis", folder="Sampling",
          options=(("augmentrum", "transient_synthesis"),),
          help="Transients synthesised from one FID, with a real scan's drift, jitter and "
               "motion events."),
    Block("undersampling", "augment", "k-space undersampling", folder="Sampling",
          options=(("augmentrum", "undersampling"),),
          help="MRSI k-space undersampled on a Cartesian or a non-Cartesian trajectory."),
    Block("spatial", "augment", "Spatial augmentation", folder="Sampling",
          options=(("augmentrum", "spatial"),),
          help="Affine transformations of an MRSI volume and its coil sensitivities."),
    # fit
    Block("lcm-fit", "fit", "Linear-combination fit", out=RESULTS,
          options=(("fsl_mrs", "fit"), ("lcmodel", "fit"), ("osprey", "fit"), ("spant", "fit"),
                   ("tarquin", "fit")),
          help="Model the spectrum as a combination of basis spectra."),
    Block("peak-fit", "fit", "Peak fit (AMARES)", out=RESULTS, options=(("pyamares", "fit"),),
          help="Model the spectrum as peaks with prior knowledge."),
    Block("ratios", "fit", "Ratios to creatine", call="scoring.ratios", imports=(_SCORE,),
          inputs=(Port("results", RESULTS),),
          fields=(_choice("to", ["tCr"], "'tCr'"), _value("scaling", "None")),
          outputs=(Port("out", RESULTS),),
          help="Each fit's concentrations as ratios to total creatine, Cr + PCr, the most "
               "common reference (Near 2021), as a result of its own."),
    # models
    Block("wand", "models", "WAND", model="mrsuite.models.wand.WAND",
          imports=("from mrsuite import models",),
          fields=(_pick("checkpoint", ZOO["wand"]), _value("remove", "None"),
                  _value("keep", "None")),
          inputs=(Port("data", SPECTRA),), outputs=(Port("out", SPECTRA),),
          help="Decomposes each spectrum into its metabolites, baseline and artifacts: the "
               "spectrum without the components removed (e.g. ['artifacts']), with only those "
               "kept ('metabolites' for all), or, with neither, every component apart."),
    Block("mlp", "models", "QuantMLP", model="mrsuite.models.mlp.MLP", out=RESULTS,
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
          help="Full width at half maximum. FID-A and Osprey measure the water peak unless a "
               "range is given: ppmmin 1.8, ppmmax 2.2 for NAA on water-suppressed spectra."),
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
          fields=(_choice("by", ["aspect", "measure"], "'aspect'"),),
          inputs=(Port("scores", SCORES, many=True),), outputs=(Port("figure"),),
          help="Score tables on one radar, filled as they are wired in: a corner per aspect "
               "naming its measures, or by measure a spoke per measure, each scaled from the "
               "worst method to the best at the rim, a polygon per method, as bsuite draws "
               "its agents."),
    # other
    *[Block(key, "other", title, options=tuple(options), help=help)
      for key, title, options, help in _OTHER],
    # mrsi
    Block("mrsi-coils", "mrsi", "MRSI coil combination",
          options=(("spant", "comb_coils_mrsi_gls"), ("fsl_mrs", "coilcombine")),
          help="The coils of every voxel combined: spant's generalised least squares on the "
               "noise, or FSL-MRS's weights."),
    Block("kspace", "mrsi", "k-space",
          options=(("spant", "zf_xy"), ("spant", "apodise_xy"), ("spant", "grid_shift_xy"),
                   ("spant", "pg_extrap_xy"), ("spant", "img2kspace_xy"),
                   ("spant", "kspace2img_xy")),
          help="The slice in k-space: zero-filled, apodised, shifted, extrapolated, or "
               "transformed there and back, with spant."),
    Block("mrsi-alignment", "mrsi", "MRSI frequency alignment",
          options=(("fsl_mrs", "mrsi_freq_align"),),
          help="Every voxel aligned in frequency to a target or to their mean, with FSL-MRS."),
    Block("mrsi-phasing", "mrsi", "MRSI phase correction",
          options=(("fsl_mrs", "mrsi_phase_corr"),),
          help="Every voxel phased, with FSL-MRS."),
    Block("lipid-removal", "mrsi", "Lipid removal",
          options=(("fsl_mrs", "lipid_removal_l2"), ("spant", "ssp")),
          help="Lipids from the skull taken out of the voxels: FSL-MRS's L2 regularisation "
               "on a lipid mask or basis, or spant's signal-space projection."),
    Block("voxel-mask", "mrsi", "Voxel mask",
          options=(("spant", "mask_xy_ellipse"), ("spant", "mask_xy_corners"),
                   ("spant", "crop_xy")),
          help="Voxels outside an ellipse or in the corners masked, or the grid cropped."),
    Block("voxel", "mrsi", "Voxel",
          options=(("spant", "get_voxel"), ("spant", "get_slice")),
          help="One voxel, or one slice, out of the grid."),
    Block("voxels", "mrsi", "Voxels apart", call="mrsi.voxels",
          imports=("from mrsuite import mrsi",),
          inputs=(Port("spectra", SPECTRA),), fields=(_value("threshold", "0.0"),),
          outputs=(Port("spectra", SPECTRA), Port("index")),
          help="The grid's voxels as single-voxel spectra, one per voxel with signal, which "
               "every step and fitter takes; and their indices, for the maps."),
    Block("voxels-mask", "mrsi", "Voxels in a mask", call="mrsi.voxels",
          imports=("from mrsuite import mrsi",),
          inputs=(Port("spectra", SPECTRA), Port("mask")),
          outputs=(Port("spectra", SPECTRA), Port("index")),
          help="The grid's voxels inside a mask (the brain's) as single-voxel spectra, and "
               "their indices, for the maps."),
    Block("mrsi-fit", "mrsi", "MRSI fit", call="fsl_mrs.fit_mrsi",
          imports=("from mrsuite.bridges import fsl_mrs",),
          inputs=(Port("spectra", SPECTRA), Port("basis", BASIS), Port("mask")),
          fields=(_value("ppmlim", "(0.2, 4.2)"), _value("baseline_order", "2"),
                  _choice("algo", ["Newton", "MH"], "'Newton'"), _value("workers", "None")),
          outputs=(Port("results", RESULTS), Port("index")),
          help="FSL-MRS's fsl_mrsi on the whole grid: every voxel in the mask fitted inside "
               "FSL-MRS, in parallel; the fits and their indices, for the maps."),
    Block("maps", "mrsi", "Metabolite maps", call="mrsi.maps",
          imports=("from mrsuite import mrsi",),
          inputs=(Port("results", RESULTS), Port("index")),
          fields=(_value("metabolites", "None"), _value("scaling", "None")),
          outputs=(Port("figure"),),
          help="Fits of the voxels back on the grid, one map per metabolite."),
    Block("maps-truth", "mrsi", "Maps against the truth", call="mrsi.maps",
          imports=("from mrsuite import mrsi",),
          inputs=(Port("results", RESULTS), Port("index"), Port("truth")),
          fields=(_value("metabolites", "None"), _value("scaling", "None")),
          outputs=(Port("figure"),),
          help="Fits of the voxels back on the grid, each map over its true map (the MRSI "
               "challenge's), on the fitted voxels."),
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
# name prefixes, and what a name says; what the curated blocks offer keeps their group, a fit
# is a fit, and the remaining steps (spectra in, spectra out) are preprocessing
FOLDERS = {"fid_a": {"inputOutput": "data", "simulationTools": "simulate",
                     "rfPulseTools": "simulate"},
           "osprey": {"load": "data", "process": "preprocess", "fit": "fit", "quantify": "fit",
                      "plot": "evaluate", "overview": "evaluate"}}
PREFIXES = {"spant": {"read_": "data", "write_": "data", "sim_": "simulate", "seq_": "simulate",
                      "fit_": "fit", "calc_": "evaluate", "plot_": "evaluate"}}
# and what a name says before either: FID-A's CSI functions, spant's MRSI ones
PATTERNS = {"fid_a": {r"^(op|io)_CSI": "mrsi"},
            "spant": {r"mrsi|_xy(_|$)|slice|voi|psf|kspace": "mrsi"}}
# and a toolbox that does one thing: every module of Augmentrum augments
DEFAULTS = {"augmentrum": "augment"}
PREFIXES["suspect"] = {"load_": "data"}


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
        for pattern, group in PATTERNS.get(bridge, {}).items():
            if re.search(pattern, name):
                return group
        group = FOLDERS.get(bridge, {}).get(folders.get(name, ""))
        if group:
            return group
        for prefix, group in PREFIXES.get(bridge, {}).items():
            if name.startswith(prefix):
                return group
        if bridge in DEFAULTS:
            return DEFAULTS[bridge]
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
            "pyamares": "pyAMARES", "vespa": "Vespa", "nifti_mrs": "NIfTI-MRS tools",
            "augmentrum": "Augmentrum", "suspect": "suspect",
            _CHALLENGE: "2016 challenge", _SCORING: "MRSuite"}.get(where, where) + f" · {function}"


#***********#
#   ports   #
#***********#
def ports(b: Block, arguments: List[str]) -> Tuple[List[Port], List[Port]]:
    """
    A block's inputs and outputs: a step takes spectra, and each of its arguments, which
    a wire gives or else the form; it makes what it makes. An argument named basis takes
    a basis set (every fit's does); one named files takes files, as a block reads them.
    One named sequence takes a pulse sequence. A load's outputs are its names, which the
    graph knows (``Graph.ports``).

    Args:
        b (Block): The block.
        arguments (List[str]): Its arguments, each an input of its own.
    """
    kinds = {"basis": BASIS, "files": FILES, "sequence": SEQUENCE}
    given = [Port(n, kinds.get(n, ANY)) for n in arguments]
    if b.step:
        return [Port("data", SPECTRA), *given], [Port("out", b.out)]
    return [*b.inputs, *given], list(b.outputs)
