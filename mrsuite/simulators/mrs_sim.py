####################################################################################################
#                                            mrs_sim.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Simulating with MRS-Sim (LaMaster et al., NMR Biomed 2025). MRS-Sim runs in a process   #
#          of its own, through mrs_sim_driver.py, and hands back NIfTI-MRS with the whole truth:   #
#          concentrations, parameters, and every basis function's signal, the baseline, the        #
#          residual water and the noise, which add up to the data.                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nifti_mrs.create_nmrs import gen_nifti_mrs

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core import provenance
from mrsuite.octave import scratch
from mrsuite.training import source
from mrsuite.training.source import Truth

ROOT = Path(__file__).resolve().parents[2] / "externals" / "mrs-sim"
DRIVER = Path(__file__).with_name("mrs_sim_driver.py")

# the Python MRS-Sim runs in: this one unless it needs an environment of its own
PYTHON = os.environ.get("MRSUITE_MRS_SIM_PYTHON") or sys.executable


#************#
#   config   #
#************#
def config(name: str = "coil_combined_with_artifacts") -> Dict[str, Any]:
    """
    One of MRS-Sim's predefined configs, to adjust and simulate from.

    Args:
        name (str): The file in MRS-Sim's ``src/config/predefined``, without '.json'. At
            the pinned commit only 'coil_combined_with_artifacts' and 'denoising_dataset'
            match MRS-Sim's code; the others predate it.

    Returns:
        Dict[str, Any]: The config, as MRS-Sim reads it.
    """
    with open(ROOT / "src" / "config" / "predefined" / f"{name}.json") as fh:
        return json.load(fh)


#*************#
#   version   #
#*************#
def version() -> Optional[str]:
    """The MRS-Sim checkout's exact state, for the provenance."""
    return provenance.git_version(str(ROOT))


#**************#
#   simulate   #
#**************#
def simulate(config: Dict[str, Any], n: int, seed: int = 0,
             components: bool = True) -> Tuple[NIfTI_MRS_Plus, Truth]:
    """
    Simulate ``n`` spectra with MRS-Sim.

    Args:
        config (Dict[str, Any]): MRS-Sim's config (see ``config``); every parameter is drawn
            uniformly within its ranges.
        n (int): Spectra to simulate.
        seed (int): Seeds the draw, so the same seed gives the same spectra.
        components (bool): Return each signal on its own. Memory grows with the number of
            basis functions, so leave it off when concentrations are enough.

    Returns:
        Tuple[NIfTI_MRS_Plus, Truth]: The spectra, each with how it was made in its
        ProcessingApplied, and what they really hold.
    """
    r = _run(config, n, seed, components)
    entry = provenance.entry("MRS-Sim", version(), "PhysicsModel.forward",
                             {"config": config, "seed": seed})
    items = []
    for fid in r["data"]:
        item = gen_nifti_mrs(fid.reshape(1, 1, 1, -1), float(r["dwell"]), float(r["frequency"]))
        item.add_hdr_field("SpecFreqChemShift", float(r["centre"]))
        provenance.append(item, entry)
        items.append(item)
    return NIfTI_MRS_Plus(items), _truth(r)


#**************************************************************************************************#
#                                           Class Source                                           #
#**************************************************************************************************#
#                                                                                                  #
# MRS-Sim as a training source: spectra and their truth, simulated a chunk at a time.              #
#                                                                                                  #
#**************************************************************************************************#
class Source(source.Source):
    """MRS-Sim as a training source: spectra and their truth, simulated a chunk at a time.

    Starting MRS-Sim costs seconds, so it simulates ``chunk`` spectra per start and hands
    them out batch by batch. The first chunk is simulated when the source is made, which
    is also how it knows its acquisition. Chunk ``k`` is seeded from ``seed`` and ``k``,
    so a source replays exactly.

        source = mrs_sim.Source(mrs_sim.config(), components=True)
        wand.learn(source, steps=10_000)

    Args:
        config (Dict[str, Any]): MRS-Sim's config (see ``config``).
        seed (int): Seeds the stream.
        chunk (int): Spectra per MRS-Sim start.
        components (bool): Give each signal on its own, as a model learning components
            needs.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, config: Dict[str, Any], seed: int = 0, chunk: int = 128,
                 components: bool = False) -> None:
        self.config, self.seed, self.chunk, self.components = config, seed, chunk, components
        self._drawn, self._next = 0, 0
        self._fill()
        self.acquisition = {"points": self._data.shape[1], "bandwidth": 1 / self._dwell,
                            "frequency": self._frequency, "centre": self._centre}

    #**********#
    #   fill   #
    #**********#
    def _fill(self) -> None:
        """The next chunk, simulated."""
        seed = int(np.random.SeedSequence([self.seed, self._drawn]).generate_state(1)[0])
        r = _run(self.config, self.chunk, seed, self.components)
        self._data, self._truth, self._next = r["data"], _truth(r), 0
        self._dwell, self._frequency, self._centre = (float(r["dwell"]), float(r["frequency"]),
                                                      float(r["centre"]))
        self._drawn += 1

    #**********#
    #   draw   #
    #**********#
    def draw(self, n: int) -> Tuple[np.ndarray, Truth]:
        if n > self.chunk:
            raise ValueError(f"a batch of {n} is more than a chunk of {self.chunk}")
        if self._next + n > self.chunk:
            self._fill()
        part = slice(self._next, self._next + n)
        self._next += n
        t = self._truth
        return self._data[part], Truth(
            concentrations={k: v[part] for k, v in t.concentrations.items()},
            components={k: v[part] for k, v in t.components.items()},
            parameters={k: v[part] for k, v in t.parameters.items()})


#*********#
#   run   #
#*********#
def _run(config: Dict[str, Any], n: int, seed: int, components: bool) -> Dict[str, np.ndarray]:
    """One MRS-Sim job, in its own process, with the MRS-Sim checkout as its working folder."""
    if config.get("num_coils", 1) > 1:
        raise ValueError(f"MRS-Sim ({version()}) fails on several coils inside its own noise "
                         f"model (physics_model.py, generate_noise); simulate combined spectra, "
                         f"num_coils 1")
    # MRS-Sim's baseline smoothing squeezes a batch of one away (aux.batch_smooth), so a
    # single spectrum is simulated as two and the second dropped
    count = max(n, 2)
    folder = scratch("mrsuite_mrs_sim_")
    try:
        path, result = os.path.join(folder, "job.json"), os.path.join(folder, "result.npz")
        with open(path, "w") as fh:
            json.dump({"config": config, "n": count, "seed": seed, "components": components,
                       "folder": folder, "result": result}, fh)
        proc = subprocess.run([PYTHON, str(DRIVER), path], cwd=ROOT, capture_output=True,
                              text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if proc.returncode != 0 or not os.path.exists(result):
            raise RuntimeError(f"MRS-Sim failed:\n{proc.stderr[-3000:]}")
        with np.load(result) as r:
            out = {k: r[k] for k in r.files}
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    for key in ("data", "components", "concentrations", "parameters", "snr"):
        if len(out[key]):
            out[key] = out[key][:n]
    return out


#***********#
#   truth   #
#***********#
def _truth(r: Dict[str, np.ndarray]) -> Truth:
    """The driver's arrays as a Truth, by name."""
    metabolites = [str(m) for m in r["metabolites"]]
    concentrations = dict(zip(metabolites, r["concentrations"].T))
    skip = {m.lower() for m in metabolites}
    parameters = {str(k): v for k, v in zip(r["labels"], r["parameters"].T)
                  if str(k) not in skip}
    components = {}
    if len(r["components"]):
        components = dict(zip((str(k) for k in r["names"]), r["components"].transpose(1, 0, 2)))
    return Truth(concentrations=concentrations, components=components, parameters=parameters)
