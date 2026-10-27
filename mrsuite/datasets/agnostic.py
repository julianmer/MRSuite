####################################################################################################
#                                            agnostic.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: AGNOSTIC (Gudmundson et al.), a synthetic PRESS dataset across fields of 1.4 to 3.1 T   #
#          and echo times of 10 to 80 ms: its spectra as NIfTI-MRS, the metabolite, macromolecule, #
#          water and noise signals as chosen, with every metabolite's amplitude as truth.          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import importlib.util
import re
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite import externals
from mrsuite.core import provenance

DOI = "https://doi.org/10.7280/D1RX1T"
DATA = externals.local("data") / "agnostic"
LICENCE = "CC0 1.0 (Dryad, doi:10.7280/D1RX1T)"
COMPONENTS = ("Metab", "MM", "water", "noise")
# the reference its own ppm axis is centred on
CENTRE = 4.7
GAMMA = 42.577478          # 1H, MHz/T


#**********#
#   load   #
#**********#
def load(shard: int = 21, n: Optional[int] = None,
         components: Sequence[str] = COMPONENTS) -> Tuple[NIfTI_MRS_Plus, List[Dict[str, float]]]:
    """
    Spectra of one of AGNOSTIC's 22 shards (``data0`` to ``data21``, 7200 spectra each), the
    chosen signals summed, each with its field, bandwidth and echo time, and its truth: the
    amplitude of every metabolite and macromolecule, as AGNOSTIC drew it.

    Dryad hands out its files only after a login, so a shard is not fetched: it is read
    from MRSuite's data folder, as ``data<shard>.tgz`` or unpacked to ``data<shard>.tar``.

    Args:
        shard (int): Which shard.
        n (Optional[int]): The first ``n`` spectra; all of the shard when None.
        components (Sequence[str]): Of 'Metab', 'MM', 'water' and 'noise', the ones summed.

    Returns:
        Tuple[NIfTI_MRS_Plus, List[Dict[str, float]]]: The spectra, their source in each
        one's history, and the amplitudes per spectrum, by metabolite.
    """
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"AGNOSTIC's signals are {COMPONENTS}, not {sorted(unknown)}")
    path = next((p for p in (DATA / f"data{shard}.tar", DATA / f"data{shard}.tgz") if p.exists()),
                None)
    if path is None:
        raise FileNotFoundError(f"AGNOSTIC's shard data{shard}.tgz is not in {DATA}: Dryad asks "
                                f"for a login, so download it from {DOI} and put it there")
    names = metabolites()
    items, truth = [], []
    with tarfile.open(path) as archive:
        members = sorted((m for m in archive if m.name.endswith(".npz")), key=lambda m: m.name)
        for member in members:
            with np.load(archive.extractfile(member)) as part:
                fids = sum(part[c] for c in components)
                for k in range(len(fids)):
                    if n is not None and len(items) == n:
                        break
                    field, te = part["Field_Str"][k] / 10, float(part["Echo_Times"][k])
                    item = gen_nifti_mrs(fids[k][None, None, None], 1.0 / float(part["sw"][k]),
                                         field * GAMMA)
                    item.add_hdr_field("SpecFreqChemShift", CENTRE)
                    item.add_hdr_field("EchoTime", te / 1e3)
                    provenance.append(item, provenance.entry(
                        "MRSuite", provenance.mrsuite_version(), "mrsuite.datasets.agnostic.load",
                        {"dataset": DOI, "file": f"data{shard}/{member.name}", "index": k,
                         "components": list(components), "licence": LICENCE}))
                    items.append(item)
                    truth.append({name: float(part["Amplitude"][k][i])
                                  for name, i in names.items()})
            if n is not None and len(items) == n:
                break
    return NIfTI_MRS_Plus(items), truth


#*****************#
#   metabolites   #
#*****************#
@lru_cache(maxsize=None)
def metabolites() -> Dict[str, int]:
    """
    Where each metabolite's amplitude lies among AGNOSTIC's 182 spin systems, by name.

    AGNOSTIC stores one amplitude per spin system, in the order its 02_metab_matrix.py
    builds them from its metab_dict: the metabolites sorted, but the generic multiplets
    and choline, each one's distinct shifts between 0 and 8 ppm. A metabolite simulated
    in parts (NAA1, NAA2) shares one amplitude across them, so the name drops the part.

    Returns:
        Dict[str, int]: The first spin system of each metabolite, by name.
    """
    spec = importlib.util.spec_from_file_location(
        "agnostic_metab_dict", externals.ensure("agnostic") / "metab_dict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = module.metab_dict
    index, out = 0, {}
    for key in sorted(set(table) - {"Doub", "Trip", "Quar", "Cho"}):
        for omega in set(np.asarray(table[key]["omega"])):
            if 0.0 < omega < 8.0:
                out.setdefault(re.sub(r"(?<=[A-Za-z])\d$", "", key), index)
                index += 1
    return out
