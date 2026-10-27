####################################################################################################
#                                             bases.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: Basis sets simulated by the toolboxes - FID-A's density-matrix simulation of ideal      #
#          sequences, spant's, FSL-MRS's fsl_mrs_sim from its sequence description, Vespa's        #
#          Simulation on PyGAMMA - as one Basis every fit takes, with how it was simulated in its  #
#          source.                                                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from mrsuite.core.basis import CENTRE, Basis

# the 1H brain metabolites FID-A holds spin systems for (simulationTools/metabolites), as it
# spells them
METABOLITES = ("Ala", "Asc", "Asp", "Cr", "GABA", "Glc", "Gln", "Glu", "GPC", "GSH", "Gly", "Ins",
               "Lac", "NAA", "NAAG", "PCh", "PCr", "PE", "Scyllo", "Tau")

# the metabolites of Vespa's bundled experiment 'Example - Spin-Echo sequence', as its
# metabolite database spells them
VESPA_METABOLITES = ("acetate", "aspartate", "choline-truncated", "creatine", "gaba", "glutamate",
                     "glutamine", "lactate", "myo-inositol", "n-acetylaspartate", "taurine")


#***********#
#   fid a   #
#***********#
def fid_a(metabolites: Sequence[str] = METABOLITES, seq: str = "press", te1: float = 11.0,
          te2: float = 19.0, points: int = 2048, bandwidth: float = 2000.0, field: float = 3.0,
          linewidth: float = 2.0) -> Basis:
    """
    A basis set simulated by FID-A with ideal pulses: one density-matrix simulation per
    metabolite (``sim_lcmrawbasis``, which loads FID-A's own spin systems), through Octave
    or MATLAB.

    Args:
        metabolites (Sequence[str]): As FID-A's simulationTools/metabolites spell them.
        seq (str): 'se' (spin echo), 'press', 'steam' or 'laser'.
        te1 (float): The first echo time in ms; TE for a spin echo, STEAM and LASER.
        te2 (float): The second echo time in ms for PRESS; TM for STEAM; unused otherwise.
        points (int): Points of the FID.
        bandwidth (float): Spectral width in Hz.
        field (float): Field strength in T.
        linewidth (float): Linewidth in Hz.

    Returns:
        Basis: The metabolites' FIDs, water at the centre as FID-A puts it.
    """
    from mrsuite.bridges.fid_a import bridge

    codes = {"se": "se", "press": "p", "steam": "st", "laser": "l"}
    if seq not in codes:
        raise ValueError(f"seq must be one of {sorted(codes)}, not {seq!r}")
    fids, out = [], None
    for name in metabolites:
        _, out = bridge.sim_lcmrawbasis(points, bandwidth, field, linewidth, name, te1, te2,
                                        "n", "n", codes[seq], nout=2)
        fids.append(np.asarray(out["fids"], dtype=complex).ravel())
    return Basis(list(metabolites), np.array(fids), float(out["spectralwidth"]),
                 float(out["txfrq"]) / 1e6, centre=CENTRE, te=float(out["te"]), seq=seq,
                 source={"simulated": {"program": "FID-A", "version": bridge.version(),
                                       "function": "sim_lcmrawbasis",
                                       "arguments": {"seq": seq, "te1": te1, "te2": te2,
                                                     "points": points, "bandwidth": bandwidth,
                                                     "field": field, "linewidth": linewidth}}})


#************#
#   simntg   #
#************#
def simntg(chain_length: int = 18, double_bonds: Sequence[int] = (6, 9), seq: str = "press",
           te1: float = 11.0, te2: float = 19.0, points: int = 2048, bandwidth: float = 2000.0,
           field: float = 3.0, linewidth: float = 2.0) -> Basis:
    """
    Triglyceride basis functions by SimnTG (Ouwerkerk, MRSHub) on FID-A: its spin systems
    for a triglyceride of three fatty-acid chains of ``chain_length`` carbons with double
    bonds at ``double_bonds`` (counted from the methyl end) and the glycerol backbone,
    simulated through FID-A's ideal-pulse sequence at MRSuite's grid; one function per
    spin system (the olefinic, diallylic, chain, methyl and glycerol protons) and their
    sum, the whole triglyceride, named by chain length and unsaturation (``TG18:2``).

    SimnTG's own run builds the systems and then a TE series with MATLAB's ``round(x, 1)``,
    which Octave lacks: under Octave a shim gives it, on the path for that call alone.

    Args:
        chain_length (int): Carbons per fatty-acid chain.
        double_bonds (Sequence[int]): Positions of the double bonds from the methyl end.
        seq (str): 'se', 'press', 'steam' or 'laser', as FID-A's ideal simulations.
        te1 (float): The first echo time in ms; TE for a spin echo, STEAM and LASER.
        te2 (float): The second echo time in ms for PRESS; TM for STEAM.
        points (int): Points of the FID.
        bandwidth (float): Spectral width in Hz.
        field (float): Field strength in T.
        linewidth (float): Linewidth in Hz.

    Returns:
        Basis: The spin systems' functions and ``TG<chain>:<bonds>``, their sum.
    """
    from mrsuite import externals
    from mrsuite.bridges.fid_a import bridge
    from mrsuite.octave import shim_path

    sims = {"se": "sim_spinecho(n, sw, B, lw, s, te1)", "press": "sim_press(n, sw, B, lw, s, te1, te2)",
            "steam": "sim_steam(n, sw, B, lw, s, te1, te2)", "laser": "sim_laser(n, sw, B, lw, s, te1)"}
    if seq not in sims:
        raise ValueError(f"seq must be one of {sorted(sims)}, not {seq!r}")
    octave = bridge.octave
    octave.eval(f"addpath('{externals.ensure('simntg')}'); set(0, 'defaultfigurevisible', 'off');")
    shims = os.path.join(shim_path(), "simntg") if bridge.runtime()["runtime"] != "matlab" else None
    if shims:                                                # Octave's round, with MATLAB's digits
        octave.eval(f"addpath('{shims}');")
    try:
        octave.eval(f"[~, simntg_sys] = Sim_UFATG_PRESS({int(chain_length)}, "
                    f"[{' '.join(str(int(d)) for d in double_bonds)}]);")
    finally:
        if shims:
            octave.eval(f"rmpath('{shims}');")
    octave.eval(f"n = {int(points)}; sw = {float(bandwidth)}; B = {float(field)}; lw = {float(linewidth)}; "
                f"te1 = {float(te1)}; te2 = {float(te2)}; simntg_names = {{simntg_sys.name}}; "
                f"simntg_fids = []; for k = 1:numel(simntg_sys); s = simntg_sys(k); "
                f"o = {sims[seq]}; simntg_fids(:, k) = o.fids(:); end; "
                f"s = simntg_sys; o = {sims[seq]}; simntg_fids(:, end + 1) = o.fids(:);")
    names = [str(n) for n in np.ravel(octave.pull("simntg_names"))]   # Octave's cell is a row
    seen, labels = {}, []
    for name in names:                                       # two chain blocks: FAchain, FAchain2
        seen[name] = seen.get(name, 0) + 1
        labels.append(name if seen[name] == 1 else f"{name}{seen[name]}")
    whole = f"TG{int(chain_length)}:{len(double_bonds)}"
    fids = np.asarray(octave.pull("simntg_fids"), dtype=complex).T
    out = octave.pull("o")
    return Basis([*labels, whole], fids, float(out["spectralwidth"]), float(out["txfrq"]) / 1e6,
                 centre=CENTRE, te=float(out["te"]), seq=seq,
                 source={"simulated": {"program": "SimnTG on FID-A", "version": bridge.version(),
                                       "function": f"Sim_UFATG_PRESS, {sims[seq].split('(')[0]}",
                                       "arguments": {"chain_length": chain_length,
                                                     "double_bonds": list(double_bonds), "seq": seq,
                                                     "te1": te1, "te2": te2, "points": points,
                                                     "bandwidth": bandwidth, "field": field,
                                                     "linewidth": linewidth}}})


#***********#
#   spant   #
#***********#
def spant(pul_seq: str = "seq_press_ideal", frequency: float = 123.2, bandwidth: float = 2000.0,
          points: int = 1024, centre: float = CENTRE, lcm_compat: bool = False,
          **timings: Any) -> Basis:
    """
    spant's 1H brain basis set (``sim_basis_1h_brain``): its metabolites, macromolecules
    and lipids through one of its ideal sequences, written by spant in LCModel's format
    and read back by FSL-MRS's reader (BasiC's refuses spant's file).

    Args:
        pul_seq (str): spant's sequence: 'seq_press_ideal', 'seq_slaser_ideal',
            'seq_steam_ideal', 'seq_mega_press_ideal', 'seq_spin_echo_ideal',
            'seq_pulse_acquire'; its timings by name in seconds (TE1, TE2, TE3, TE, TM).
        frequency (float): Spectrometer frequency in MHz.
        bandwidth (float): Spectral width in Hz.
        points (int): Points.
        centre (float): The ppm at the centre.
        lcm_compat (bool): spant's LCModel-compatible naming.

    Returns:
        Basis: The basis functions, the water at ``centre``.
    """
    from fsl_mrs.utils.mrs_io import read_basis

    from mrsuite import r
    from mrsuite.octave import scratch

    folder = scratch("mrsuite_spant_sim_")
    try:
        path = os.path.join(folder, "spant.basis")
        r.run({"task": "simulate", "func": "sim_basis_1h_brain", "pul_seq": pul_seq,
               "acq_paras": {"ft": frequency * 1e6, "fs": bandwidth, "N": points, "ref": centre},
               "args": [], "kwargs": {"lcm_compat": lcm_compat, **timings}, "output": path},
              folder)
        read = read_basis(path)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return Basis([str(n) for n in read.names], np.asarray(read.original_basis_array).T,
                 float(read.original_bw), float(read.cf), centre=centre, seq=pul_seq,
                 source={"simulated": {"program": "spant", "version": r.info()["spant"],
                                       "function": "sim_basis_1h_brain",
                                       "arguments": {"pul_seq": pul_seq, "frequency": frequency,
                                                     "bandwidth": bandwidth, "points": points,
                                                     "centre": centre, "lcm_compat": lcm_compat,
                                                     **timings}}})


#*************#
#   fsl mrs   #
#*************#
def fsl_mrs(sequence_json: str, metabolites: Sequence[str] = METABOLITES, te: float = None,
            autophase: float = None) -> Basis:
    """
    A basis set simulated by FSL-MRS's ``fsl_mrs_sim``: its density-matrix simulation of a
    sequence described in its JSON (RF pulses, gradients, delays and the coherence filter),
    one run per metabolite of its spin systems, read back from the files it writes.

    Args:
        sequence_json (str): The sequence description, a JSON file as fsl_mrs_sim takes it
            (its examples: examplePRESS.json, exampleSTEAM.json).
        metabolites (Sequence[str]): As FSL-MRS's spin systems spell them.
        te (float): Echo time in ms written into the files (no effect on the simulation).
        autophase (float): A ppm to simulate a singlet at, to phase the basis by.

    Returns:
        Basis: As ``mrsuite.io.basis`` reads FSL-MRS's files.

    Raises:
        RuntimeError: If fsl_mrs_sim is not there (it comes with FSL-MRS from the FSL
            conda channel, with fsl-denmatsim), or fails, with its own message.
    """
    from mrsuite import io
    from mrsuite.octave import scratch

    folder = scratch("mrsuite_fsl_sim_")
    try:
        listed = os.path.join(folder, "metabolites.txt")
        with open(listed, "w") as fh:
            fh.write("\n".join(metabolites) + "\n")
        out = os.path.join(folder, "basis")
        command = [sys.executable, "-c", "from fsl_mrs.scripts.fsl_mrs_sim import main; main()",
                   sequence_json, "-b", listed, "-o", out]
        command += ["-e", str(te)] if te is not None else []
        command += ["-p", str(autophase)] if autophase is not None else []
        run = subprocess.run(command, capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(
                "fsl_mrs_sim failed; it needs FSL-MRS with its simulator fsl-denmatsim, from "
                "the FSL conda channel (conda install -c conda-forge -c https://fsl.fmrib.ox."
                f"ac.uk/fsldownloads/fslconda/public/ fsl_mrs fsl-denmatsim):\n{run.stderr[-2000:]}")
        files = sorted(os.path.join(out, f) for f in os.listdir(out) if f.endswith(".json"))
        basis = io.basis(files)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    basis.source["simulated"] = {"program": "FSL-MRS", "function": "fsl_mrs_sim",
                                 "arguments": {"sequence_json": sequence_json, "te": te,
                                               "autophase": autophase}}
    return basis


#***********#
#   vespa   #
#***********#
def vespa(metabolites: Sequence[str] = VESPA_METABOLITES, sequence: str = "PRESS Ideal",
          timings: Optional[Sequence[float]] = None, frequency: float = 64.0, points: int = 2048,
          bandwidth: float = 2000.0) -> Basis:
    """
    A basis set simulated by Vespa's Simulation on PyGAMMA: one of its bundled pulse
    sequences run on its bundled metabolites' spin systems (``run_experiment``), their
    lines summed into FIDs (``build_basis_functions``) about 4.7 ppm, as Vespa centres 1H,
    and left unbroadened, as Vespa sums them. The defaults are Vespa's own
    (vespa/common/constants.py: DEFAULT_B0, DEFAULT_SPECTRAL_POINTS, DEFAULT_SWEEP_WIDTH;
    mrs_experiment.DEFAULT_LOOP).

    Vespa runs as published: its database built once from its bundled XML into MRSuite's
    folder, by a process of its own whose home is that folder, which so takes the logs it
    writes; and ``inspect.getargspec``, removed in Python 3.11 and still called by its
    experiment runner, given back as the first four of ``getfullargspec``. PyGAMMA is
    built on first use (``pygamma_build``).

    Args:
        metabolites (Sequence[str]): As Vespa's metabolite database spells them, e.g.
            'n-acetylaspartate'.
        sequence (str): A pulse sequence bundled with Vespa: 'PRESS Ideal', 'STEAM Ideal',
            'Spin-Echo', ...
        timings (Optional[Sequence[float]]): One per loop of the sequence in ms, in its
            order (PRESS Ideal: TE1, TE2; STEAM Ideal: TE, TM; Spin-Echo: TE); None for
            Vespa's default loop, 0 ms each.
        frequency (float): Spectrometer frequency, Vespa's B0, in MHz.
        points (int): Points of the FID.
        bandwidth (float): Spectral width in Hz.

    Returns:
        Basis: The metabolites in the order asked; TE the sum of the timings named TE.

    Raises:
        ImportError: If Vespa is not installed, or configobj, which it needs.
        ValueError: If the sequence or a metabolite is not in Vespa's database, or the
            timings are not one per loop.
        RuntimeError: If PyGAMMA cannot be built, or the simulation fails, with Vespa's
            message.
    """
    import inspect

    from mrsuite import externals
    from mrsuite.simulators import pygamma_build

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = lambda func: tuple(inspect.getfullargspec(func)[:4])
    try:
        import vespa.common.constants as constants
        import vespa.common.mrs_experiment as mrs_experiment
        import vespa.common.mrs_simulation as mrs_simulation
        import vespa.common.util.misc as util_misc
        import vespa.simulation.util_db as util_db
    except ImportError as err:
        raise ImportError(f"Vespa cannot be imported ({err}): pip install it from "
                          "https://github.com/vespa-mrs/vespa (the PyPI wheel lags) with "
                          "configobj") from err

    version = util_misc.get_vespa_version()
    db = util_db.Database(str(_vespa_database(version)))
    found = db.fetch_pulse_sequences_by_name(sequence)
    if not found:
        raise ValueError(f"no pulse sequence {sequence!r} in Vespa's database; it holds "
                         f"{sorted(s.name for s in db.fetch_pulse_sequences())}")
    seq = db.fetch_pulse_sequence(found[0].id)
    labels = list(seq.loop_labels)
    timings = [float(t) for t in timings] if timings is not None \
        else [mrs_experiment.DEFAULT_LOOP[0]] * len(labels)
    if len(timings) != len(labels):
        raise ValueError(f"{sequence} takes a timing per loop, {labels}, not {len(timings)}")
    known = {m.name: m for m in db.fetch_metabolites()}
    unknown = [m for m in metabolites if m not in known]
    if unknown:
        raise ValueError(f"not in Vespa's metabolite database: {unknown}; it holds "
                         f"{sorted(known)}")

    pygamma_build.ensure()               # the experiment runner and its sequences import pygamma
    import pygamma
    import vespa.simulation.build_basis_functions as build_basis_functions
    import vespa.simulation.run_experiment_controller as run_experiment_controller

    experiment = mrs_experiment.Experiment()
    experiment.name = "MRSuite"
    experiment.b0 = float(frequency)
    experiment.isotope = "1H"
    experiment.pulse_sequence = seq
    experiment.metabolites = [known[m] for m in metabolites]
    experiment.dims = [[t] for t in timings] + [mrs_experiment.DEFAULT_LOOP] * (3 - len(timings))
    results, failed = run_experiment_controller.run_experiment(experiment, local_cpu=1)
    if failed:
        raise RuntimeError("Vespa's simulation failed:\n"
                           + run_experiment_controller.exception_to_message(failed[0]))
    by_name = {result["metabolite"]: result for result in results}
    experiment.simulations = [mrs_simulation.Simulation({**by_name[m.name], "metabolite": m})
                              for m in experiment.metabolites]
    centre = constants.DEFAULT_PROTON_CENTER_PPM
    lines = build_basis_functions.build_basis_functions(experiment, int(points), float(bandwidth),
                                                        centre, local_cpu=1)
    # Vespa puts a line at sw / 2 - (ppm - centre) * B0 Hz (its ppm2pts): conjugated and moved
    # by half the bandwidth, a peak at f Hz sits at centre + f / frequency, as Basis holds it
    fids = np.conj(lines.reshape(len(metabolites), int(points))) * (-1.0) ** np.arange(points)
    return Basis(list(metabolites), fids.astype(complex), float(bandwidth), float(frequency),
                 centre=centre, seq=sequence,
                 te=sum(t for label, t in zip(labels, timings) if label.startswith("TE")),
                 source={"simulated": {"program": "Vespa Simulation", "version": version,
                                       "engine": {"program": "PyGAMMA",
                                                  "version": pygamma.__version__,
                                                  "commit": externals.REGISTRY["pygamma"][1]},
                                       "function": "run_experiment, build_basis_functions",
                                       "pulse_sequence": {"name": seq.name, "id": seq.id},
                                       "arguments": {"metabolites": list(metabolites),
                                                     "sequence": sequence,
                                                     "timings": dict(zip(labels, timings)),
                                                     "frequency": frequency, "points": points,
                                                     "bandwidth": bandwidth}}})


#********************#
#   vespa database   #
#********************#
def _vespa_database(version: str) -> Path:
    """Vespa's database, built once per Vespa version by its own create_database from its
    bundled XML, into MRSuite's folder; by a process whose home is that folder, so the logs
    Vespa writes into its data folder land there; built beside its place and moved in whole."""
    from mrsuite import externals

    folder = externals.local(".vespa")
    path = folder / f"vespa-{version}.sqlite"
    if path.is_file():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    part = Path(tempfile.mkdtemp(prefix=".part", dir=folder))
    code = ("import logging, os, sys; import vespa.common.util.misc as misc; "
            "os.makedirs(os.path.join(misc.get_data_dir(), 'logs'), exist_ok=True); "
            "from vespa.common.create_database import create_database; "
            "create_database(sys.argv[1], logging.getLogger('vespa'))")
    # without XDG_DATA_HOME, Vespa's data folder on Linux is ~/.Vespa, as on its first run
    env = {k: v for k, v in os.environ.items() if k != "XDG_DATA_HOME"} | {"HOME": str(folder)}
    try:
        run = subprocess.run([sys.executable, "-c", code, str(part / path.name)], env=env,
                             capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"Vespa could not build its database:\n{run.stderr[-2000:]}")
        os.replace(part / path.name, path)
    finally:
        shutil.rmtree(part, ignore_errors=True)
    return path
