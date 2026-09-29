####################################################################################################
#                                            recipe.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-29                                                                              #
#                                                                                                  #
# Purpose: Training recipes as the MRS community reports synthetic data: the MRSsynMRS minimum     #
#          reporting standard (LaMaster, Gudmundson et al., arXiv 2602.23463, section 6), whose    #
#          template spreadsheet has seven sections, stored as the JSON sidecar the working group   #
#          recommends beside NIfTI-MRS data. A simulator source fills the report; save writes      #
#          the sidecar; table renders it for a model card.                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Union

# The seven sections of the MRSsynMRS template spreadsheet (SyntheticMRS_Reporting_Standard.xlsx,
# linked from the paper's Figure 7), item for item, with the spreadsheet's units in brackets.
# Empty dicts hold one entry per metabolite; 'Number of Points' is an added row, which the
# standard invites where a method needs more. None means not reported.
TEMPLATE: Dict[str, Any] = {
    "Experiment": {"ID": None, "Nucleus": None},
    "Software": {
        "Basis Set Simulation": {"Name": None, "Repository (Link, DOI, or RRID)": None},
        "Signal Model": {"Name": None, "Repository (Link, DOI, or RRID)": None},
        "Additional Code": {"Name": None, "Repository (Link, DOI, or RRID)": None,
                            "Purpose": None}},
    "Pulse Sequence": {
        "Field Strength [T]": None,
        "Localization": None,
        "Spectral Width [Hz]": None,
        "Number of Points": None,
        "Voxel Size (X, Y, Z) [mm]": None,
        "Sequence Timing(s)": {"Repetition Time(s) [ms]": None, "Echo Time(s) [ms]": None,
                               "Mixing Time(s) [ms]": None, "Inversion Time(s) [ms]": None},
        "Water Suppression": {"Method": None, "Bandwidth(s) [Hz]": None},
        "Shimming Method": None,
        "Radio Frequency Pulse(s)": {"Shape": None, "Duration(s)": None, "Delay(s)": None},
        "Slice-Select Gradients": {"Shape": None, "Duration(s)": None, "Delay(s)": None},
        "Crusher/Rephasing Gradients": {"Shape": None, "Duration(s)": None, "Delay(s)": None},
        "Editing": {"Type of editing": None, "Number of subexperiments": None,
                    "Editing bands per subexperiment": None}},
    "Metabolites": {
        "Population": {"Age Range": None, "Sex": None, "Healthy included": None,
                       "Clinical included": None},
        "Region of interest": None,
        "Distribution": None,
        "Component dropout probability": None,
        "Units": None,
        "Sources of Parameter Values": {"Concentration Ranges (Link or DOI)": None,
                                        "T2 Relaxation Ranges (Link or DOI)": None,
                                        "T1 Relaxation Ranges (Link or DOI)": None},
        "Concentration Ranges": {},
        "T2 Relaxation Ranges": {},
        "T1 Relaxation Ranges": {},
        "Metabolite-Level Frequency Shifts": {}},
    "Components": {
        "Number Spectral Averages": None,
        "Number of Acquisitions per Excitation": None,
        "Dynamic Experiment": {"Diffusion B-values": None, "Diffusion Model (Name, DOI)": None,
                               "fMRS Target Metabolites and Percent Changes": None},
        "Macromolecule": {"Available in Isolation": None,
                          "Simulated, Modeled, Experimental, or Mixture": None,
                          "Description": None},
        "Spectral Baseline - nonMMBG": {"Model Used (Equation, Description, or DOI)": None,
                                        "Available in Isolation": None, "Applied": None},
        "Noise": {"Available in Isolation": None, "Distribution": None,
                  "Amplitude Scaling": None},
        "B0 Map": {"B0 Map Model Used": None, "Available in Isolation": None, "Applied": None},
        "Tissue Segmentation": {"Brain Map Model Used": None, "Voxel Percentage(s)": None,
                                "Applied": None}},
    "Artifacts": {
        "Zero-Order Phase": {"Phase Range": None, "Applied": None},
        "First-Order Phase": {"Phase Range": None, "Amplitude Range": None,
                              "PPM Reference [ppm]": None, "Applied": None},
        "Frequency Shifts - Global": {"Frequency Shift Range": None, "Applied": None},
        "Residual Water": {"Model Used (Equation, Description, or DOI)": None,
                           "If Analytical, Parameter Ranges": None,
                           "Available in Isolation": None, "Applied": None,
                           "Chemical Shift [ppm]": None, "Number of Subcomponents": None,
                           "Amplitude Scaling [a.u.]": None},
        "Unsuppressed Water": {"Model Used (Equation, Description, or DOI)": None,
                               "Available in Isolation": None, "Applied": None},
        "Lipid": {"Model Used (Equation, Description, or DOI)": None,
                  "Available in Isolation": None, "Applied": None},
        "Eddy Currents": {"Model Used (Equation, Description, or DOI)": None,
                          "If Analytical, Parameter Ranges": None,
                          "Available in Isolation": None, "Applied": None},
        "Out-of-Voxel": {"Model Used (Equation, Description, or DOI)": None,
                         "If Analytical, Parameter Ranges": None,
                         "Available in Isolation": None, "Applied": None},
        "Motion": {"Model Used (Equation, Description, or DOI)": None,
                   "If Analytical, Parameter Ranges": None,
                   "Available in Isolation": None, "Applied": None}},
    "Data Quality": {
        "Signal Noise Ratio Range [a.u.]": {"Range": None,
                                            "Metabolite-level SNRs recorded": None,
                                            "Definition of SNR used": None},
        "Linewidths": {"Metabolite-level Linewidths recorded": None,
                       "Reference Metabolite and Moiety": None,
                       "Reference Metabolite Linewidth Range [Hz]": None}},
}


#************#
#   report   #
#************#
def report() -> Dict[str, Any]:
    """A blank MRSsynMRS report, to fill: the template's seven sections, item for item."""
    return copy.deepcopy(TEMPLATE)


#**********#
#   save   #
#**********#
def save(recipe: Dict[str, Any], path: Union[str, Path]) -> str:
    """Write a recipe as the JSON sidecar, readable and diffable."""
    Path(path).write_text(json.dumps(recipe, indent=2, sort_keys=False) + "\n")
    return str(path)


#***********#
#   table   #
#***********#
def table(recipe: Dict[str, Any]) -> str:
    """
    The recipe's MRSsynMRS report as a Markdown table, for a model card.

    Args:
        recipe (Dict[str, Any]): A full recipe (its 'MRSsynMRS' entry is taken) or the
            report itself.

    Returns:
        str: One two-column table: the sections bold, nested items indented, ranges as
        'low – high', None as '—'.
    """
    report = recipe.get("MRSsynMRS", recipe)
    lines = ["| MRSsynMRS | Value |", "|:---|:---|"]
    for section, items in report.items():
        lines.append(f"| **{_text(section)}** | |")
        lines += _rows(items, 0)
    return "\n".join(lines)


#**********#
#   rows   #
#**********#
def _rows(items: Dict[str, Any], depth: int) -> List[str]:
    """One table row per item, nesting shown by indentation."""
    pad = "&nbsp;&nbsp;" * depth
    rows = []
    for name, value in items.items():
        if isinstance(value, dict):
            rows.append(f"| {pad}{_text(name)} | |")
            rows += _rows(value, depth + 1)
        else:
            rows.append(f"| {pad}{_text(name)} | {_value(value)} |")
    return rows


#***********#
#   value   #
#***********#
def _value(value: Any) -> str:
    """A cell: '—' for unreported, 'low – high' for a pair, plain text otherwise."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                   for v in value):
            return f"{_value(value[0])} – {_value(value[1])}"
        return ", ".join(_value(v) for v in value)
    return _text(str(value))


#**********#
#   text   #
#**********#
def _text(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")
