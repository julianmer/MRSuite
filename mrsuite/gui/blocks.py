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
from typing import Any, Dict, List, Optional, Tuple

from mrsuite.gui import catalog
from mrsuite.gui.catalog import Field

# spec2nii's formats that need no options of their own (its text and raw need dwell times
# and such); 'auto' reads NIfTI-MRS as it is and lets spec2nii tell the rest apart
FORMATS = ["auto", "twix", "ge", "philips", "dicom", "rda", "bruker", "varian", "jmrui", "uih"]

# what flows along a wire; 'any' meets everything. A challenge's truth is concentrations per
# spectrum, as scoring takes it; a simulation's is MRS-Sim's own (amplitudes, signals); a
# basis is a basis set (mrsuite.core.basis); files are one written for another tool
SPECTRA, RESULTS, TRUTH, SIMULATED, BASIS, FILES, ANY = \
    "spectra", "results", "truth", "simulated", "basis", "files", "any"


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
          "Read spectra in: scanner files, NIfTI-MRS, the challenges; and basis sets, in any "
          "format. Every block exports what it made; Convert basis set writes its files "
          "itself."),
    Group("simulate", "Simulate", "#8ea4ff", "Synthetic spectra with known contents."),
    Group("preprocess", "Preprocess", "#5274f0",
          "Prepare the spectra for analysis: correct imperfections, reduce dimensionality "
          "(Near et al., NMR Biomed 2021, Tables 2-4)."),
    Group("fit", "Fit", "#16bdc6",
          "Spectral analysis: model the spectrum to estimate its signals, in arbitrary units "
          "or, with a water reference, as the toolbox quantifies them (Near 2021; Kreis et "
          "al., NMR Biomed 2021, section 4.2)."),
    Group("evaluate", "Evaluate", "#7fe0e6",
          "Judge the spectra and the fits: spectral quality (SNR, linewidth) and fits against "
          "the ground truth."),
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
    written as ``call(...)`` with the fields it declares.

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
        help (str): One line on what it does.
    """

    key: str
    group: str
    title: str
    options: Tuple[Tuple[str, str], ...] = ()
    out: str = SPECTRA
    call: str = ""
    imports: Tuple[str, ...] = ()
    fields: Tuple[Field, ...] = ()
    inputs: Tuple[Port, ...] = ()
    outputs: Tuple[Port, ...] = (Port("out"),)
    fixed: Dict[str, str] = field(default_factory=dict)
    wrap: Dict[str, str] = field(default_factory=dict)
    help: str = ""

    #**********#
    #   step   #
    #**********#
    @property
    def step(self) -> bool:
        return bool(self.options)


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
    Block("files", "data", "Read files", call="io.load", imports=("from mrsuite import io",),
          fields=(_value("files"), _choice("format", FORMATS, "'auto'")),
          outputs=(), help="Raw scanner data through spec2nii, or NIfTI-MRS as it is."),
    Block("ismrm2016", "data", "2016 fitting challenge", call="challenge2016.load",
          imports=("from mrsuite.datasets import challenge2016",),
          fields=(_value("n", "None"),),
          outputs=(Port("spectra", SPECTRA), Port("water", SPECTRA), Port("truth", TRUTH)),
          help="The ISMRM 2016 fitting challenge: spectra, water references, ground truth."),
    Block("mrsi2024", "data", "MRSI challenge", call="mrsi_challenge.load",
          imports=("from mrsuite.datasets import mrsi_challenge",),
          fields=(_value("subject", "", positional=True), _value("voxels", "None")),
          outputs=(Port("spectra", SPECTRA), Port("truth", TRUTH), Port("voxels")),
          help="One subject of the 2024 MRSI challenge, a spectrum per brain voxel."),
    Block("basis", "data", "Basis set", call="io.basis", imports=("from mrsuite import io",),
          fields=(_value("files", positional=True), _value("format", "'auto'"),
                  _value("sw", "None"), _value("sf", "None"), _value("te", "None")),
          outputs=(Port("basis", BASIS),),
          help="A basis set in any format, told apart by BasiC, for any fit; sw (Hz), sf "
               "(MHz) and te (ms) where the files do not state them."),
    Block("basis-convert", "data", "Convert basis set", call="io.convert_basis",
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
    Block("model", "fit", "Trained model", out=ANY, imports=("from mrsuite import models",),
          fields=(_value("name"),), inputs=(Port("data", SPECTRA),),
          outputs=(Port("out", ANY),),                 # a fit, or spectra (e.g. WAND's)
          help="A trained network, by its name on the Hugging Face Hub, e.g. "
               "MRSuite/wand-mrsi2024."),
    # evaluate
    Block("snr", "evaluate", "SNR", out=ANY,
          options=((_CHALLENGE, "snr"), ("fid_a", "op_getSNR"), ("osprey", "op_getSNR"),
                   ("spant", "calc_spec_snr")),
          help="Signal-to-noise ratio, as each toolbox defines it."),
    Block("linewidth", "evaluate", "Linewidth", out=ANY,
          options=(("fid_a", "op_getLW"), ("osprey", "op_getLW")),
          help="Full width at half maximum."),
    Block("benchmark", "evaluate", "Benchmark score", call="scoring.score",
          imports=("from mrsuite.benchmark import scoring",),
          fields=(_choice("references", ["optimal", "tCr", "water", "none"]),
                  _value("output_dir")),
          inputs=(Port("methods", RESULTS, many=True), Port("truth", TRUTH)),
          fixed={"metrics": "{'mae': scoring.mae, 'bias': scoring.bias}"},
          wrap={"references": "[{}]"},
          help="MAE and bias of each fit against the ground truth, on the reference "
               "chosen (Kreis 2021, section 4.3: accuracy)."),
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


#***********#
#   label   #
#***********#
def label(option: Tuple[str, str]) -> str:
    """An option as a person reads it: the toolbox, and the function where a toolbox
    offers several."""
    where, function = option
    return {"fsl_mrs": "FSL-MRS", "fid_a": "FID-A", "osprey": "Osprey", "spant": "spant",
            "mrspeclab": "MRSpecLAB", "lcmodel": "LCModel", "tarquin": "TARQUIN",
            "pyamares": "pyAMARES", "vespa": "Vespa", _CHALLENGE: "2016 challenge"}.get(
        where, where) + f" · {function}"


#***********#
#   ports   #
#***********#
def ports(b: Block, arguments: List[str], outputs: Optional[List[str]] = None) \
        -> Tuple[List[Port], List[Port]]:
    """
    A block's inputs and outputs: a step takes spectra, and each of its arguments, which
    a wire gives or else the form; it makes what it makes; a file read makes the outputs
    it found. An argument named basis takes a basis set (every fit's does); one named
    files takes files, as a block reads them.

    Args:
        b (Block): The block.
        arguments (List[str]): Its arguments, each an input of its own.
        outputs (Optional[List[str]]): The outputs a read found.
    """
    given = [Port(n, BASIS if n == "basis" else FILES if n == "files" else ANY)
             for n in arguments]
    if b.step:
        return [Port("data", SPECTRA), *given], [Port("out", b.out)]
    found = [Port(n, SPECTRA) for n in outputs or []]
    return [*b.inputs, *given], found or list(b.outputs)
