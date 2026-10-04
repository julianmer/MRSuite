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
import copy
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from nifti_mrs.create_nmrs import gen_nifti_mrs

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core import provenance
from mrsuite.octave import scratch
from mrsuite.training import recipe as recipe_
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
            the pinned commit 'coil_combined_with_artifacts' and 'denoising_dataset'
            simulate; three others need several coils, which this driver does not take
            apart yet, and the rest predate MRS-Sim's code.

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


#*******************#
#   compile basis   #
#*******************#
def compile_basis(basis: Union[str, Path], out: Union[str, Path], te: float,
                  centre: float = 4.65, vendor: str = "", pulse_sequence: str = "",
                  points: Optional[int] = None) -> str:
    """
    MRS-Sim's basis .mat from an LCModel .BASIS, as src/compileBasisSet.py builds one.

    MRS-Sim's own compiler reads MARSS spin simulations; this builds the same file from
    a .BASIS, so any basis a fitter uses can also simulate. The basis functions keep
    the .BASIS' own scale and NIfTI-MRS' orientation, which FSL-MRS' reader gives, and
    MRS-Sim simulates on exactly this grid ('resample' off), so its spectra come out at
    the basis' bandwidth and points.

    Args:
        basis (Union[str, Path]): The .BASIS file.
        out (Union[str, Path]): The .mat to write.
        te (float): Echo time in ms, which the .BASIS does not state reliably.
        centre (float): The ppm the spectrum is centred on.
        vendor (str): For the header, if known.
        pulse_sequence (str): For the header, if known.
        points (Optional[int]): Cut the basis FIDs to this many points, to put a longer
            basis on the data's grid. More points than the basis has is refused.

    Returns:
        str: The path written.
    """
    from fsl_mrs.utils import mrs_io
    from scipy import io as scipy_io

    if Path(basis).suffix.lower() != ".basis":
        raise ValueError(f"{basis} is not an LCModel .BASIS; FSL-MRS reads other formats, "
                         f"such as jMRUI's, in the opposite orientation, which would simulate "
                         f"mirrored spectra")
    b = mrs_io.read_basis(str(basis))
    fids = b.original_basis_array                       # (points, n), NIfTI-MRS orientation
    n = points or fids.shape[0]
    if n > fids.shape[0]:
        raise ValueError(f"{basis} has {fids.shape[0]} points, not the {n} asked for; "
                         f"a basis is cut to the data's grid, never padded")
    fids, bandwidth, frequency = fids[:n], float(b.original_bw), float(b.cf)
    names = [str(name) for name in b.names]
    if len({name.lower() for name in names}) != len(names):
        raise ValueError(f"{basis} holds names that collide once lowercased, as MRS-Sim "
                         f"stores them: {sorted(names)}")

    dwell = 1 / bandwidth
    header = {"B0": frequency / 42.5774785, "carrier_frequency": frequency,
              "dwelltime": dwell, "linewidth": 1, "Ns": n,
              "ppm": np.linspace(-0.5 * bandwidth, 0.5 * bandwidth, n) / frequency + centre,
              "TE": te, "vendor": vendor, "t": (np.arange(0, n * dwell, dwell))[:n],
              "spectralwidth": bandwidth, "centerFreq": centre,
              "pulse_sequence": pulse_sequence,
              "notes": f"Compiled by MRSuite {provenance.mrsuite_version()} from "
                       f"{Path(basis).name} (sha256 "
                       f"{provenance.file_checksum(basis)['sha256']}), at its own scale."}
    metabolites = {name.lower(): {"fid": np.stack([fid.real, fid.imag]),
                                  "min": 0.0, "max": 1.0}
                   for name, fid in zip(names, fids.T)}
    scipy_io.savemat(str(out), {"metabolites": metabolites, "header": header})
    return str(out)


#***********#
#   basis   #
#***********#
def basis(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    The config's basis functions, by the config's names: complex FIDs on the simulation grid.

    As the .mat stores them, without the fixed 1 Hz broadening MRS-Sim adds inside its
    forward model.
    """
    metabolites = _basis_mat(config)[0]
    fids = {}
    for name in config["metabolites"]:
        if name.lower() not in metabolites:
            raise KeyError(f"the basis {config['PM_basis_set']} has no {name!r}; it has "
                           f"{sorted(metabolites)}")
        fid = np.asarray(metabolites[name.lower()]["fid"], float).reshape(2, -1)
        fids[name] = fid[0] + 1j * fid[1]
    return fids


#***************#
#   basis mat   #
#***************#
def _basis_mat(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The config's basis .mat, resolved as MRS-Sim resolves it: its metabolites and header."""
    from scipy import io as scipy_io

    path = Path(config["PM_basis_set"])
    if not path.is_absolute():
        path = ROOT / "src" / "basis_sets" / path
    mat = scipy_io.loadmat(str(path), simplify_cells=True)
    return mat["metabolites"], mat["header"]


#*******************#
#   distributions   #
#*******************#
def distributions(samples: Dict[str, np.ndarray], folder: Union[str, Path],
                  independent: Tuple[str, ...] = ()) -> Dict[str, str]:
    """
    MRS-Sim's own fitted distributions for parameter values seen in real data.

    MRS-Sim's pipeline (src/findParamDist.py, then its CopulaInVivoSampler) fits a
    marginal distribution to each parameter and the correlations between them, and
    simulates from both. It reads Osprey's parameter export, one file per spectrum,
    so the values are written that way, each parameter as its own key; put the result
    in a config as ``config["copula"]`` and the simulation samples from it.

    Args:
        samples (Dict[str, np.ndarray]): (n,) values per parameter, in MRS-Sim's own
            units and names: a basis function's name for its amplitude (relative to
            the config's reference metabolites), '<name>.d' for its Lorentzian width
            in 1/s (MRS-Sim adds a fixed 1/s of its own), 'snr' in MRS-Sim's terms (peak
            of the reference lines' real spectrum over the noise's time-domain SD), and
            'f_shift' in ppm.
        folder (Union[str, Path]): Where the files and MRS-Sim's fits go.
        independent (Tuple[str, ...]): Parameters to sample without correlations, by
            the names above.

    Returns:
        Dict[str, str]: The fitted distributions and the correlation matrix, as paths.
    """
    from scipy import io as scipy_io

    keys = {name: _osprey_key(name) for name in samples}
    n = {len(np.ravel(v)) for v in samples.values()}
    if len(n) != 1:
        raise ValueError(f"every parameter needs the same number of samples, not {sorted(n)}")
    folder = Path(folder)
    for own in ("parameters", "fits"):                  # only what an earlier call wrote
        shutil.rmtree(folder / own, ignore_errors=True)
    for i in range(n.pop()):
        spectrum = folder / "parameters" / f"{i:06d}"
        spectrum.mkdir(parents=True)
        scipy_io.savemat(str(spectrum / "fitting_parameters.mat"), {
            "header": {"names": np.array(list(samples), dtype=object)},
            **{keys[k]: float(np.ravel(v)[i]) for k, v in samples.items()}})

    proc = subprocess.run(
        [PYTHON, "src/findParamDist.py", "--paramPath", str(folder / "parameters"),
         "--savedir", str(folder / "fits"), "--paramKeys", ",".join(keys.values()),
         "--findCorr"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"})
    fitted, corr = (folder / "fits" / "parameter_distributions_best_fit.json",
                    folder / "fits" / "correlation_matrix.mat")
    if proc.returncode != 0 or not fitted.exists() or not corr.exists():
        raise RuntimeError(f"MRS-Sim's findParamDist.py failed:\n{proc.stderr[-3000:]}")

    mat = scipy_io.loadmat(str(corr), simplify_cells=True)
    names = [str(x) for x in np.ravel(mat["names"])]
    matrix = np.asarray(mat["corr"], float)
    for key in (keys[k] for k in independent):
        i = names.index(key)
        matrix[i, :] = matrix[:, i] = 0.0
        matrix[i, i] = 1.0
    scipy_io.savemat(str(corr), {"corr": matrix, "names": np.array(names, dtype=object)})
    return {"distributions": str(fitted), "correlation": str(corr)}


#****************#
#   osprey key   #
#****************#
def _osprey_key(name: str) -> str:
    """A parameter's name as findParamDist.py reads Osprey's export, and back as
    CopulaInVivoSampler resolves it (see the driver's maps): 'snr', 'f_shift',
    '<name>.d' or a basis function's name."""
    if name == "snr":
        return "SNR"
    if name == "f_shift":
        return "global_freqShift"
    if name.endswith(".d"):
        return f"{name[:-2]}_lorentzLB"
    if "." in name:
        raise ValueError(f"MRS-Sim's fitted distributions take amplitudes, '<name>.d', 'snr' "
                         f"and 'f_shift', not {name!r}")
    return f"{name}_ampl"


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
    so a source replays exactly, however many chunks are simulated ahead.

        source = mrs_sim.Source(mrs_sim.config(), components=True)
        wand.learn(source, steps=10_000)

    Args:
        config (Dict[str, Any]): MRS-Sim's config (see ``config``).
        seed (int): Seeds the stream.
        chunk (int): Spectra per MRS-Sim start.
        components (bool): Give each signal on its own, as a model learning components
            needs.
        ahead (int): Chunks simulated in the background while the current one is handed
            out, each in an MRS-Sim process of its own, so learning need not wait for
            the simulator; the spectra are the same either way. 0 by default, each chunk
            simulated when it is needed. ``close`` cancels those not yet needed.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, config: Dict[str, Any], seed: int = 0, chunk: int = 128,
                 components: bool = False, ahead: int = 0) -> None:
        self.config, self.seed, self.chunk, self.components = config, seed, chunk, components
        self.ahead = ahead
        self._drawn, self._next, self._coming = 0, 0, {}
        self._fill()
        self.acquisition = {"points": self._data.shape[1], "bandwidth": 1 / self._dwell,
                            "frequency": self._frequency, "centre": self._centre}

    #**********#
    #   fill   #
    #**********#
    def _fill(self) -> None:
        """The next chunk, simulated, then the ``ahead`` after it set going: never more
        than ``ahead`` MRS-Sim processes at once."""
        coming = self._coming.pop(self._drawn, None)
        r = coming.result() if coming else self._simulate(self._drawn)
        if self.ahead and "_pool" not in self.__dict__:
            self._pool = ThreadPoolExecutor(self.ahead)     # each thread waits on a process
        for k in range(self._drawn + 1, self._drawn + 1 + self.ahead):
            if k not in self._coming:
                self._coming[k] = self._pool.submit(self._simulate, k)
        self._data, self._truth, self._next = r["data"], _truth(r), 0
        self._dwell, self._frequency, self._centre = (float(r["dwell"]), float(r["frequency"]),
                                                      float(r["centre"]))
        self._ranges = {str(k): [float(lo), float(hi)] for k, (lo, hi)
                        in zip(r["labels"], r["parameter_ranges"])}
        self._concentration_ranges = {str(k): [float(lo), float(hi)] for k, (lo, hi)
                                      in zip(r["metabolites"], r["concentration_ranges"])}
        self._table, self._provenance = (json.loads(str(r["mrssynmrs"])),
                                         json.loads(str(r["provenance"])))
        self._drawn += 1

    #***********#
    #   close   #
    #***********#
    def close(self) -> None:
        """Cancel the chunks simulated ahead that have not started; a running one ends on
        its own."""
        if "_pool" in self.__dict__:
            self._pool.shutdown(wait=False, cancel_futures=True)
        self._coming.clear()

    #**************#
    #   simulate   #
    #**************#
    def _simulate(self, k: int) -> Dict[str, np.ndarray]:
        """Chunk ``k``, seeded from the source's seed and ``k`` alone."""
        seed = int(np.random.SeedSequence([self.seed, k]).generate_state(1)[0])
        return _run(self.config, self.chunk, seed, self.components)

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

    #***********#
    #   basis   #
    #***********#
    @property
    def basis(self) -> Dict[str, np.ndarray]:
        """The config's basis functions by its names, read once (see ``basis``)."""
        if "_basis" not in self.__dict__:
            self._basis = basis(self.config)
        return self._basis

    #************#
    #   recipe   #
    #************#
    def recipe(self) -> Dict[str, Any]:
        """How the spectra are made, as the MRSsynMRS report.

        Its first four sections are MRS-Sim's own table (src/mrssynmrs.py), which MRSuite
        only completes where its own sampling decides: the reference metabolites fixed
        at their ratios, basis functions the table misses, the number of points and the
        metabolite-level shifts. MRS-Sim's table stops at Metabolites, so MRSuite fills
        Components, Artifacts and Data Quality from the config and the ranges MRS-Sim
        really sampled. MRS-Sim's provenance record, the config and the seed go beside it.
        """
        c, r = self.config, self._ranges
        reference = [m.strip() for m in c["wrt_metab"].split(",")]
        ratios = c.get("wrt_metab_ratio") or [1 / len(reference)] * len(reference)
        names = list(self._concentration_ranges)
        background = [m for m in names
                      if "mm" in m.lower() or "lip" in m.lower() or "mac" in m.lower()]

        table = copy.deepcopy(self._table)
        report = recipe_.report()
        report.update({section: table[section] for section in
                       ("Experiment", "Software", "Pulse Sequence", "Metabolites")})

        report["Software"]["Additional Code"].append({
            "Name": f"MRSuite {provenance.mrsuite_version()}",
            "Repository": "https://github.com/julianmer/MRSuite",
            "Purpose": "the training source (mrsuite.simulators.mrs_sim): fixes the reference "
                       "metabolites, shares one Gaussian width per group, applies dropout"})
        # MRS-Sim drops the header's text fields when it loads a basis; the .mat keeps them
        _, header = _basis_mat(c)
        if not report["Software"]["Basis Set Simulation"].get("Name"):
            report["Software"]["Basis Set Simulation"]["Name"] = str(header.get("notes")) or None
        if not report["Pulse Sequence"].get("Localization"):
            report["Pulse Sequence"]["Localization"] = str(header.get("pulse_sequence")) or None
        report["Pulse Sequence"]["Number of Points"] = self.acquisition["points"]

        metabolites = report["Metabolites"]
        metabolites["Units"] = (f"ratio to {' + '.join(reference)}, fixed at "
                                f"{[float(x) for x in ratios]}")
        ranges = metabolites.get("Concentration Ranges") or {}
        for name in names:
            if name in reference:
                ranges[name] = f"fixed at {ratios[reference.index(name)]:g} (reference)"
            elif name not in ranges:
                ranges[name] = self._concentration_ranges[name]
        metabolites["Concentration Ranges"] = ranges
        metabolites["Metabolite-Level Frequency Shifts"] = (
            {m: r[f"{m}.f_shifts"] for m in names} if c["fshift_i"] else "not applied")
        # MRS-Sim's table gives its database T2s, but the config's 'd' is what simulates
        metabolites["T2 Relaxation Ranges"] = (
            "not applied (Gaussian lineshape)" if c["lineshape"] == "gaussian"
            else {m: r[f"{m}.d"] for m in names})
        metabolites.setdefault("Sources of Parameter Values", {})["T2 relaxation ranges"] = (
            "the config's 'd': MRS-Sim's Lorentzian decay rate in 1/s, exp(-(d + 1) t), "
            "not a T2 in ms")

        components = report["Components"]
        components.update({"Number Spectral Averages": 1,
                           "Number of Acquisitions per Excitation": 1})
        components["Macromolecule"].update({
            "Available in Isolation": self.components,
            "Simulated, Modeled, Experimental, or Mixture":
                "basis functions, sampled like the metabolites" if background
                else "none in the basis",
            "Description": ", ".join(background) or None})
        components["Spectral Baseline - nonMMBG"].update({
            "Model Used (Equation, Description, or DOI)":
                "MRS-Sim sample_baselines: bounded smoothed random splines (src/aux.py)",
            "Available in Isolation": self.components,
            "Applied": bool(c.get("baseline_cfg"))})
        components["Noise"].update({
            "Available in Isolation": self.components,
            "Distribution": "additive circular Gaussian in the time domain",
            "Amplitude Scaling": "set by the SNR range (see Data Quality)"})
        components["B0 Map"].update({"B0 Map Model Used":
                                     "MRS-Sim B0 inhomogeneities" if c["b0"] else None,
                                     "Applied": bool(c["b0"])})
        components["Tissue Segmentation"]["Applied"] = False

        artifacts = report["Artifacts"]
        artifacts["Zero-Order Phase"].update({"Phase Range": r["phi0"],
                                              "Applied": bool(c["phi0"])})
        artifacts["First-Order Phase"].update({"Phase Range": r["phi1"],
                                               "PPM Reference [ppm]": c.get("ppm_ref"),
                                               "Applied": bool(c["phi1"])})
        artifacts["Frequency Shifts - Global"].update({"Frequency Shift Range": r["f_shift"],
                                                       "Applied": bool(c["fshift_g"])})
        water = c.get("resWater_cfg")
        artifacts["Residual Water"].update({
            "Model Used (Equation, Description, or DOI)":
                "MRS-Sim sample_resWater: bounded random spline around the water (src/aux.py)",
            "If Analytical, Parameter Ranges": copy.deepcopy(water),
            "Available in Isolation": self.components,
            "Applied": bool(water),
            "Chemical Shift [ppm]": (water or {}).get("ppm_range"),
            "Amplitude Scaling [a.u.]": (water or {}).get("scale")})
        artifacts["Unsuppressed Water"]["Applied"] = False
        artifacts["Lipid"]["Applied"] = bool(c.get("mm_lip"))
        artifacts["Eddy Currents"].update({
            "Model Used (Equation, Description, or DOI)":
                "MRS-Sim first_order_eddy_currents (amplitude, time constant)",
            "If Analytical, Parameter Ranges": {"Amplitude": r.get("ecc[0]"),
                                                "Time Constant": r.get("ecc[1]")},
            "Applied": bool(c["eddy"])})
        artifacts["Out-of-Voxel"]["Applied"] = False
        artifacts["Motion"]["Applied"] = False

        quality = report["Data Quality"]
        quality["Signal Noise Ratio Range [a.u.]"].update({
            "Range": r["snr"],
            "Metabolite-level SNRs recorded": "per spectrum: 'snr' in the truth parameters",
            "Definition of SNR used": f"MRS-Sim's, on {c.get('snr_metab', c['wrt_metab'])}"})
        quality["Linewidths"].update({
            "Metabolite-level Linewidths recorded":
                "per spectrum: '<name>.d' and '<name>.g' in the truth parameters",
            "Reference Metabolite and Moiety": c["wrt_metab"]})

        path = Path(c["PM_basis_set"])
        if not path.is_absolute():
            path = ROOT / "src" / "basis_sets" / path
        # files as they can be shared, never where they lie on this machine
        basis = _as_mrs_sim_reads(path)
        config = {**copy.deepcopy(c), "PM_basis_set": basis}
        if c.get("copula"):
            config["copula"] = {k: provenance.portable_file(v) for k, v in c["copula"].items()}
        mrs_sim_provenance = copy.deepcopy(self._provenance)
        if "basis_set_name" in mrs_sim_provenance:          # as the driver handed it over
            mrs_sim_provenance["basis_set_name"] = basis
        return {"MRSsynMRS": report,
                "generation": {"simulator": {"name": "MRS-Sim", "version": version(),
                                             "repository":
                                                 "https://github.com/JohnLaMaster/MRS-Sim"},
                               "config": config, "seed": self.seed, "chunk": self.chunk,
                               "components": self.components,
                               "basis": provenance.portable_file(path),
                               "sampled_ranges": {**self._concentration_ranges, **r},
                               "mrs_sim_provenance": mrs_sim_provenance}}


#**********************#
#   as mrs-sim reads   #
#**********************#
def _as_mrs_sim_reads(path: Path) -> str:
    """A basis as a recipe records it: as MRS-Sim reads it, relative to its own
    src/basis_sets, when MRSuite's checkout holds it, so the config runs again from any
    checkout; else by the file's name alone, never where it lies on this machine."""
    path = os.path.abspath(path)
    if not Path(path).is_relative_to(provenance.REPO):
        return Path(path).name
    return Path(os.path.relpath(path, ROOT / "src" / "basis_sets")).as_posix()


#*********#
#   run   #
#*********#
def _run(config: Dict[str, Any], n: int, seed: int, components: bool) -> Dict[str, np.ndarray]:
    """One MRS-Sim job, in its own process, with the MRS-Sim checkout as its working folder."""
    if config.get("num_coils", 1) > 1:
        raise ValueError("MRSuite's MRS-Sim driver does not yet take several coils apart into "
                         "their truth; simulate combined spectra, num_coils 1")
    if os.path.isabs(config.get("PM_basis_set", "")):
        # MRS-Sim resolves the basis inside its checkout ('./src/basis_sets/' + name)
        config = {**config, "PM_basis_set": os.path.relpath(config["PM_basis_set"],
                                                            ROOT / "src" / "basis_sets")}
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
    concentrations = dict(zip((str(m) for m in r["metabolites"]), r["concentrations"].T))
    parameters = dict(zip((str(k) for k in r["labels"]), r["parameters"].T))
    components = {}
    if len(r["components"]):
        components = dict(zip((str(k) for k in r["names"]), r["components"].transpose(1, 0, 2)))
    return Truth(concentrations=concentrations, components=components, parameters=parameters)
