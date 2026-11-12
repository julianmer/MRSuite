####################################################################################################
#                                            report.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-02                                                                              #
#                                                                                                  #
# Purpose: A result reports itself: the MRSinMRS checklist (Lin et al., NMR Biomed 2021, Table 1)  #
#          filled from what the spectra and fits carry - their NIfTI-MRS header and processing     #
#          history - and "not recorded" for what they do not, never guessed; every step as it was  #
#          recorded; the MRSsynMRS table of each trained model in the chain; and the references    #
#          of every tool used. Markdown.                                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from nifti_mrs.nifti_mrs import NIFTI_MRS

from mrsuite.core import provenance
from mrsuite.core.result import FitResult

NOT = "not recorded"

# the voxel spec2nii writes for a format that carries no geometry (jMRUI, text, Varian): a 10 m
# cube, its stand-in for none (spec2nii/jmrui.py, other_formats.py, varian_importer.py)
NO_GEOMETRY = (10000.0, 10000.0, 10000.0)

# the checklist itself, and MRSuite
STANDARD = ("Lin A, Andronesi O, Bogner W, et al. Minimum Reporting Standards for in vivo Magnetic "
            "Resonance Spectroscopy (MRSinMRS): Experts' consensus recommendations. NMR Biomed. "
            "2021;34(5):e4484. doi:10.1002/nbm.4484")
MRSUITE = "Merkofer JP. MRSuite. https://github.com/julianmer/MRSuite"
# MRSsynMRS, cited where a trained model's table is (its section 6, see mrsuite.training.recipe)
SYNTHETIC = ("LaMaster JT, Gudmundson AT, Abaei A, et al. Synthetic Data in MR Spectroscopy: "
             "Current Practices, Applications, and Considerations. arXiv. 2026. arXiv:2602.23463")

# by the Program a history entry names, as the bridges, simulators and readers write it; each as
# its authors ask to be cited (their README or citation file), the details from Crossref
_NIFTI_MRS = ("Clarke WT, Bell TK, Emir UE, et al. NIfTI-MRS: A standard data format for magnetic "
              "resonance spectroscopy. Magn Reson Med. 2022;88(6):2358-2370. doi:10.1002/mrm.29418")
CITATIONS: Dict[str, str] = {
    "FID-A": "Simpson R, Devenyi GA, Jezzard P, Hennessy TJ, Near J. Advanced processing and "
             "simulation of MRS data using the FID appliance (FID-A) - An open source, MATLAB-"
             "based toolkit. Magn Reson Med. 2017;77(1):23-33. doi:10.1002/mrm.26091",
    "Osprey": "Oeltzschner G, Zöllner HJ, Hui SCN, et al. Osprey: Open-source processing, "
              "reconstruction & estimation of magnetic resonance spectroscopy data. J Neurosci "
              "Methods. 2020;343:108827. doi:10.1016/j.jneumeth.2020.108827",
    "FSL-MRS": "Clarke WT, Stagg CJ, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis "
               "package. Magn Reson Med. 2021;85(6):2950-2964. doi:10.1002/mrm.28630",
    "Vespa": "Soher BJ, Semanchuk P, Todd D, et al. Vespa: Integrated applications for RF pulse "
             "design, spectral simulation and MRS data analysis. Magn Reson Med. "
             "2023;90(3):823-838. doi:10.1002/mrm.29686",
    "spant": "Wilson M. spant: An R package for magnetic resonance spectroscopy analysis. J Open "
             "Source Softw. 2021;6(67):3646. doi:10.21105/joss.03646",
    "LCModel": "Provencher SW. Estimation of metabolite concentrations from localized in vivo "
               "proton NMR spectra. Magn Reson Med. 1993;30(6):672-679. "
               "doi:10.1002/mrm.1910300604",
    "TARQUIN": "Wilson M, Reynolds G, Kauppinen RA, Arvanitis TN, Peet AC. A constrained least-"
               "squares approach to the automated quantitation of in vivo 1H magnetic resonance "
               "spectroscopy data. Magn Reson Med. 2011;65(1):1-12. doi:10.1002/mrm.22579",
    "spec2nii": _NIFTI_MRS,
    "nifti_mrs_tools": _NIFTI_MRS,
    "pyAMARES": "Xu J, Vaeggemose M, Schulte RF, et al. PyAMARES, an Open-Source Python Library "
                "for Fitting Magnetic Resonance Spectroscopy Data. Diagnostics (Basel). "
                "2024;14(23):2668. doi:10.3390/diagnostics14232668",
    "PASTIS": "Roussel T, Le Fur Y, Guye M, Viout P, Ranjeva JP, Callot V. Respiratory-triggered "
              "quantitative MR spectroscopy of the human cervical spinal cord at 7 T. Magn Reson "
              "Med. 2022;87(6):2600-2612. doi:10.1002/mrm.29182",
    "OXSA": "Purvis LAB, Clarke WT, Biasiolli L, Valkovic L, Robson MD, Rodgers CT. OXSA: An "
            "open-source magnetic resonance spectroscopy analysis toolbox in MATLAB. PLoS One. "
            "2017;12(9):e0185356. doi:10.1371/journal.pone.0185356; its AMARES: Vanhamme L, van "
            "den Boogaart A, Van Huffel S. Improved method for accurate and efficient "
            "quantification of MRS data with use of prior knowledge. J Magn Reson. "
            "1997;129(1):35-43. doi:10.1006/jmre.1997.1244",
    "Wat_Lip_Removal_L2": "Lin L, Považan M, Berrington A, Chen Z, Barker PB. Water removal in MR "
                          "spectroscopic imaging with L2 regularization. Magn Reson Med. "
                          "2019;82(4):1278-1287. doi:10.1002/mrm.27824",
    "SpecVis": "Zöllner HJ, Považan M, Hui SCN, Tapper S, Edden RAE, Oeltzschner G. Comparison of "
               "different linear-combination modeling algorithms for short-TE proton spectra. NMR "
               "Biomed. 2021;34(4):e4482. doi:10.1002/nbm.4482",
    "MRSpecLAB": "Xiao Y, Kaiser A, Kockisch M, et al. A graphical pipeline platform for MRS data"
                 " processing and analysis: MRSpecLAB. Front Neuroimaging. 2025;4:1610658. "
                 "doi:10.3389/fnimg.2025.1610658",
    "suspect": "suspect. https://github.com/openmrslab/suspect",
    "Gannet": "Edden RAE, Puts NAJ, Harris AD, Barker PB, Evans CJ. Gannet: A batch-processing "
              "tool for the quantitative analysis of gamma-aminobutyric acid-edited MR "
              "spectroscopy spectra. J Magn Reson Imaging. 2014;40(6):1445-1452. "
              "doi:10.1002/jmri.24478",
    "MRS4Brain": "Alves B, Phan TT, Briand G, et al. MRS4Brain: a software for preclinical proton"
                 " and deuterium-based MR spectroscopic imaging data. MAGMA. 2026;39(3):361-379. "
                 "doi:10.1007/s10334-026-01351-w; Simicic D, Alves B, Mosso J, et al. Fast High-"
                 "Resolution Metabolite Mapping in the rat Brain Using 1H-FID-MRSI at 14.1 T. NMR"
                 " Biomed. 2025;38(2):e5304. doi:10.1002/nbm.5304; Alves B, Simicic D, Mosso J, "
                 "et al. Noise-reduction techniques for 1H-FID-MRSI at 14.1 T: Monte Carlo "
                 "validation and in vivo application. NMR Biomed. 2024;37(11):e5211. "
                 "doi:10.1002/nbm.5211",
    "Oryx-MRSI": "Cengiz S, Yildirim M, Bas A, Ozturk-Isik E. ORYX-MRSI: A fully-automated open-"
                 "source software for proton magnetic resonance spectroscopic imaging data "
                 "analysis. Int J Imaging Syst Technol. 2022;32(4):1068-1083. "
                 "doi:10.1002/ima.22748",
    "Augmentrum": "LaMaster JT, Merkofer JP, Igwe KC. Augmentrum: A Data Augmentation Package for"
                  " MR Spectroscopy. Abstract 05685. ISMRM Annual Meeting; 2026; Cape Town, South"
                  " Africa.",
    "MRS-Sim": "LaMaster J, Oeltzschner G, Li Y. MRS-Sim: Open-Source Framework for Simulating In"
               " Vivo-Like Magnetic Resonance Spectra. NMR Biomed. 2025;38(10):e70130. "
               "doi:10.1002/nbm.70130",
    "MRS-Digital-Phantom": "van de Sande DMJ, Gudmundson AT, Murali-Manohar S, et al. A Digital "
                           "Phantom for MR Spectroscopy Data Simulation. Magn Reson Med. "
                           "2026;95(3):1345-1359. doi:10.1002/mrm.70138",
    "MRSandbox": "MRSandbox. https://github.com/julianmer/MRSandbox",
    "Spinach": "Hogben HJ, Krzystyniak M, Charnock GTP, Hore PJ, Kuprov I. Spinach - A software "
               "library for simulation of spin dynamics in large spin systems. J Magn Reson. "
               "2011;208(2):179-194. doi:10.1016/j.jmr.2010.11.008",
}

# MRSuite's own entries, by the dataset module (Method) or model class (Details' 'model') they name
OWN: Dict[str, str] = {
    "mrsuite.datasets.challenge2016":
        "Marjańska M, Deelchand DK, Kreis R; 2016 ISMRM MRS Study Group Fitting Challenge Team. "
        "Results and interpretation of a fitting challenge for MR spectroscopy set up by the MRS "
        "study group of ISMRM. Magn Reson Med. 2022;87(1):11-32. doi:10.1002/mrm.28942",
    "mrsuite.datasets.mrsi_challenge":
        "LaMaster J, Merkofer JP, van de Sande D, et al. The 2024 MRSI Data Processing and "
        "Quantification Challenge Synthetic Dataset. Zenodo. 2026. doi:10.5281/zenodo.21890222",
    "mrsuite.datasets.agnostic":
        "Gudmundson AT, Davies-Jenkins CW, Özdemir İ, et al. Application of a 1H brain MRS "
        "benchmark dataset to deep learning for out-of-voxel artifacts. Imaging Neurosci (Camb). "
        "2023;1:imag-1-00025. doi:10.1162/imag_a_00025",
    "mrsuite.models.wand.WAND":
        "Merkofer JP, van de Sande DMJ, Amirrajab S, et al. WAND: Wavelet Analysis-Based Neural "
        "Decomposition of MRS Signals for Artifact Removal. NMR Biomed. 2025;38(6):e70038. "
        "doi:10.1002/nbm.70038",
    "mrsuite.models.mlp.MLP": "Merkofer JP, Kaiser A, Schrantee A, Gurney-Champion OJ, van Sloun "
                              "RJG. Strategies to Minimize Out-of-Distribution Effects in Data-"
                              "Driven MRS Quantification. arXiv. 2025. arXiv:2511.23135",
}


#************#
#   report   #
#************#
def report(spectra: Any, fits: Union[FitResult, Sequence[FitResult], None] = None,
           path: Union[str, Path, None] = None) -> str:
    """
    The report of spectra, and of their fits where given, as Markdown.

    Args:
        spectra (Any): One spectrum (NIfTI-MRS) or a batch (NIfTI-MRS+, or a list).
        fits (Union[FitResult, Sequence[FitResult], None]): The fits of those spectra, one
            per spectrum, in order: their history includes the fit's own entry.
        path (Union[str, Path, None]): A file to write the report to as well.

    Returns:
        str: The report: the MRSinMRS checklist, the processing, MRSsynMRS tables, the
        references. An item that differs across a batch lists each spectrum's value.

    Raises:
        ValueError: If there are not as many fits as spectra.
    """
    items = spectra.list() if hasattr(spectra, "list") else \
        list(spectra) if isinstance(spectra, (list, tuple)) else [spectra]
    fits = [fits] if isinstance(fits, FitResult) else list(fits or [None] * len(items))
    if len(fits) != len(items):
        raise ValueError(f"{len(fits)} fits for {len(items)} spectra; one per spectrum")
    histories = [fit.provenance if fit is not None else provenance.history(item)
                 for item, fit in zip(items, fits)]

    text = "\n\n".join(["# Report", _checklist(items, fits, histories),
                        _processing(histories), _synthetic(histories),
                        _references(histories)])
    if path is not None:
        Path(path).write_text(text + "\n")
    return text


#***************#
#   checklist   #
#***************#
def _checklist(items: List[NIFTI_MRS], fits: List[Optional[FitResult]],
               histories: List[List[Dict[str, str]]]) -> str:
    """Table 1 of MRSinMRS, item by item, each filled for every spectrum."""
    lines = ["## MRSinMRS checklist", "",
             f"As Lin et al. (NMR Biomed 2021) list it; what the data does not record says so."]
    for section, entries in CHECKLIST:
        lines += ["", f"| {section} | |", "|:---|:---|"]
        for label, read in entries:
            values = [read(item, fit, history) or NOT
                      for item, fit, history in zip(items, fits, histories)]
            lines.append(f"| {label} | {_one(values)} |")
    return "\n".join(lines)


#*********#
#   one   #
#*********#
def _one(values: List[str]) -> str:
    """One value for a batch that agrees, else each spectrum's."""
    if len(set(values)) == 1:
        return _cell(values[0])
    return "<br>".join(f"{k + 1}: {_cell(v)}" for k, v in enumerate(values))


#**********#
#   cell   #
#**********#
def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


#**********#
#   ext   #
#**********#
def _ext(item: NIFTI_MRS, key: str) -> Any:
    """A header field, or None where the spectrum does not record it."""
    value = item.hdr_ext[key] if key in item.hdr_ext else None
    return value[0] if isinstance(value, list) and len(value) == 1 else value


#**********#
#   dims   #
#**********#
def _dim(item: NIFTI_MRS, tag: str) -> Optional[int]:
    """The size of a higher dimension, where the data still has it."""
    tags = list(item.dim_tags)
    return item.shape[4 + tags.index(tag)] if tag in tags else None


#************#
#   joined   #
#************#
def _joined(*parts: Optional[str]) -> Optional[str]:
    """What is recorded of an item's parts, or None if nothing is."""
    found = [p for p in parts if p]
    return ", ".join(found) if found else None


#********************#
#   field strength   #
#********************#
def _field(item, fit, history):
    strength = item.field_strength
    return None if math.isnan(strength) else \
        f"{strength:.2f} T ({item.spectrometer_frequency[0]:.3f} MHz, {item.nucleus[0]})"


#***********#
#   coils   #
#***********#
def _coils(item, fit, history):
    channels = _dim(item, "DIM_COIL")
    return _joined(f"nucleus {item.nucleus[0]}",
                   _ext(item, "TxCoil") and f"transmit {_ext(item, 'TxCoil')}",
                   _ext(item, "RxCoil") and f"receive {_ext(item, 'RxCoil')}",
                   channels and f"{channels} channels")


#***********#
#   model   #
#***********#
def _model(item, fit, history):
    return _joined(_ext(item, "ManufacturersModelName"),
                   _ext(item, "SoftwareVersions") and f"software {_ext(item, 'SoftwareVersions')}")


#**********#
#   size   #
#**********#
def _size(item, fit, history):
    x, y, z = item.header.get_zooms()[:3]
    if (x, y, z) == NO_GEOMETRY:
        return None
    voxel = f"{x:g} × {y:g} × {z:g} mm ({x * y * z / 1000:g} cm³)"
    grid = item.shape[:3]
    return voxel if grid == (1, 1, 1) else f"voxel {voxel}, matrix {' × '.join(map(str, grid))}"


#***********#
#   times   #
#***********#
def _times(item, fit, history):
    tr, te = _ext(item, "RepetitionTime"), _ext(item, "EchoTime")
    return _joined(tr is not None and f"TR {tr * 1000:g} ms",
                   te is not None and f"TE {te * 1000:g} ms")


#**************#
#   averages   #
#**************#
def _averages(item, fit, history):
    dynamics = _dim(item, "DIM_DYN")
    return dynamics and f"{dynamics} transients in the data (DIM_DYN)"


#*************************#
#   sequence parameters   #
#*************************#
def _sequence_more(item, fit, history):
    mixing, edit = _ext(item, "MixingTime"), _ext(item, "EditPulse")
    return _joined(f"spectral width {item.spectralwidth:g} Hz", f"{item.shape[3]} points",
                   mixing is not None and f"TM {mixing * 1000:g} ms",
                   edit and f"editing pulses {json.dumps(edit)}",
                   _ext(item, "kSpace") and f"k-space {_ext(item, 'kSpace')}")


#***********#
#   water   #
#***********#
def _water(item, fit, history):
    suppressed, kind = _ext(item, "WaterSuppressed"), _ext(item, "WaterSuppressionType")
    return _joined(suppressed is not None and ("suppressed" if suppressed else "not suppressed"),
                   kind)


#***************#
#   triggered   #
#***************#
def _triggered(item, fit, history):
    triggered = _ext(item, "SequenceTriggered")
    return None if triggered is None else ("triggered" if triggered else "not triggered")


#**************#
#   software   #
#**************#
def _software(item, fit, history):
    seen = dict.fromkeys(f"{e['Program']} {e['Version']}" for e in history)
    return ", ".join(seen) or None


#***********#
#   steps   #
#***********#
def _steps(item, fit, history):
    return history and "every step, as recorded, under Processing"


#************#
#   output   #
#************#
def _output(item, fit, history):
    return fit and ", ".join(fit.concentrations) and \
        f"{fit.method}: {', '.join(fit.concentrations)}"


#*************#
#   fitting   #
#*************#
def _fitting(item, fit, history):
    entry = fit and fit.provenance and fit.provenance[-1]
    return entry and f"{entry['Program']} {entry['Method']}: {_arguments(entry)}"


#***************#
#   variables   #
#***************#
def _variables(item, fit, history):
    return fit and fit.qc and ", ".join(f"{k} {_number(v)}" for k, v in fit.qc.items())


#*****************#
#   fit quality   #
#*****************#
def _fit_quality(item, fit, history):
    """The CRLBs in percent where the fit gives them, else every uncertainty it reports."""
    kinds = fit and ({"percentage": fit.uncertainties["percentage"]}
                     if "percentage" in fit.uncertainties else fit.uncertainties)
    return kinds and "<br>".join(
        f"{'CRLB [%]' if kind == 'percentage' else kind}: "
        + ", ".join(f"{m} {_number(v)}" for m, v in values.items())
        for kind, values in kinds.items())


#***************#
#   checklist   #
#***************#
Read = Callable[[NIFTI_MRS, Optional[FitResult], List[Dict[str, str]]], Optional[str]]
CHECKLIST: List[Tuple[str, List[Tuple[str, Read]]]] = [
    ("1. Hardware", [
        ("a. Field strength [T]", _field),
        ("b. Manufacturer", lambda item, fit, history: _ext(item, "Manufacturer")),
        ("c. Model (software version if available)", _model),
        ("d. RF coils: nuclei (transmit/receive), number of channels, type, body part", _coils),
        ("e. Additional hardware", lambda *_: None)]),
    ("2. Acquisition", [
        ("a. Pulse sequence", lambda item, fit, history:
            _joined(_ext(item, "SequenceName"), _ext(item, "ProtocolName"))),
        ("b. Volume of interest (VOI) locations", lambda *_: None),
        ("c. Nominal VOI size [cm³, mm³]", _size),
        ("d. Repetition time (TR), echo time (TE) [ms, s]", _times),
        ("e. Total number of excitations or acquisitions per spectrum", _averages),
        ("f. Additional sequence parameters (spectral width in Hz, number of spectral points, "
         "frequency offsets; STEAM: mixing time; MRSI: matrix)", _sequence_more),
        ("g. Water suppression method", _water),
        ("h. Shimming method, reference peak, and thresholds for “acceptance of shim” chosen",
         lambda *_: None),
        ("i. Triggering or motion correction method", _triggered)]),
    ("3. Data analysis methods and outputs", [
        ("a. Analysis software", _software),
        ("b. Processing steps deviating from quoted reference or product", _steps),
        ("c. Output measure (eg absolute concentration, institutional units, ratio)", _output),
        ("d. Quantification references and assumptions, fitting model assumptions", _fitting)]),
    ("4. Data quality", [
        ("a. Reported variables (SNR, linewidth (with reference peaks))", _variables),
        ("b. Data exclusion criteria", lambda *_: None),
        ("c. Quality measures of postprocessing model fitting (eg CRLB, goodness of fit, SD of "
         "residual)", _fit_quality),
        ("d. Sample spectrum", lambda *_: None)]),
]


#************#
#   number   #
#************#
def _number(value: Any) -> str:
    """A measure as recorded: a number to four significant digits, anything else as it is."""
    return f"{value:.4g}" if isinstance(value, (int, float)) else str(value)


#*************#
#   details   #
#*************#
def _details(entry: Dict[str, str]) -> Dict[str, Any]:
    """An entry's Details as MRSuite writes them, JSON; another program's free text, as
    spec2nii's conversion entry has, gives nothing."""
    try:
        details = json.loads(entry.get("Details") or "{}")
    except ValueError:
        return {}
    return details if isinstance(details, dict) else {}


#***************#
#   arguments   #
#***************#
def _arguments(entry: Dict[str, str]) -> str:
    """A step's arguments as its entry records them, each value shortened; free text as it is."""
    details = _details(entry)
    if not details and entry.get("Details"):
        return entry["Details"]
    shown = [_short(a) for a in details.get("args", [])] + \
        [f"{k}={_short(v)}" for k, v in details.get("kwargs", {}).items()]
    return ", ".join(shown) or "no arguments"


#***********#
#   short   #
#***********#
def _short(value: Any, limit: int = 60) -> str:
    """A value as recorded, a basis set by its files and functions, cut to ``limit``."""
    basis = value.get("basis") if isinstance(value, dict) else None
    if isinstance(basis, dict) and "names" in basis:
        files = [Path(f["path"]).name for f in basis.get("source", {}).get("files", [])]
        return f"{', '.join(files) or 'a basis set'} ({len(basis['names'])} functions)"
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit - 1] + "…"


#****************#
#   processing   #
#****************#
def _processing(histories: List[List[Dict[str, str]]]) -> str:
    """Every step as recorded: one list for a batch processed alike, else one per spectrum."""
    def steps(history):
        return [f"{k + 1}. **{e['Program']}** {e['Version']}: `{e['Method']}` ({_arguments(e)})"
                for k, e in enumerate(history)] or ["No step recorded."]

    lists = [steps(history) for history in histories]
    if all(one == lists[0] for one in lists):
        return "\n".join(["## Processing", "", *lists[0]])
    return "\n".join(["## Processing", *(f"\n### Spectrum {k + 1}\n\n" + "\n".join(one)
                                         for k, one in enumerate(lists))])


#***************#
#   synthetic   #
#***************#
def _synthetic(histories: List[List[Dict[str, str]]]) -> str:
    """The MRSsynMRS table of each trained model in the chain, from its recipe."""
    sources = {}
    for entry in (e for history in histories for e in history):
        source = _details(entry).get("source")
        if isinstance(source, dict) and source.get("name"):
            sources[(source["name"], source.get("revision"))] = None
    if not sources:
        return "## MRSsynMRS\n\nNo trained model in the chain."

    from mrsuite.models import base
    from mrsuite.training import recipe as recipe_
    parts = ["## MRSsynMRS"]
    for name, revision in sources:
        if "/" not in name:                            # a folder, recorded by its name alone
            found, why = None, f"loaded from its own folder; its {base.RECIPE} lies beside it"
        else:
            try:
                found = base._read_recipe(name, revision, {})
            except Exception as err:                   # noqa: BLE001 - said in the report
                found, why = None, f"its recipe could not be read: {err}"
            else:
                why = "saved without a recipe"
        said = recipe_.table(found) if found else f"{why.capitalize()}."
        parts.append(f"### {name}\n\n{said}")
    return "\n\n".join(parts)


#****************#
#   references   #
#****************#
def _references(histories: List[List[Dict[str, str]]]) -> str:
    """The checklist, MRSuite, then every tool the history names, each once."""
    cited = dict.fromkeys([STANDARD, MRSUITE])
    for entry in (e for history in histories for e in history):
        if isinstance(_details(entry).get("source"), dict):
            cited[SYNTHETIC] = None
        for name in _named(entry):
            cited[CITATIONS.get(name) or OWN.get(name) or f"{name}: citation not declared"] = None
    return "\n".join(["## References", "", *(f"{k + 1}. {c}" for k, c in enumerate(cited))])


#***********#
#   named   #
#***********#
def _named(entry: Dict[str, str]) -> List[str]:
    """What an entry used: its Program, or for MRSuite's own the dataset or model it names."""
    if entry["Program"] != "MRSuite":
        return [entry["Program"]]
    model = _details(entry).get("model")
    module = entry["Method"].rpartition(".")[0]
    return [found for found in (model, module) if found in OWN]
