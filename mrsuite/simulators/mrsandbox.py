####################################################################################################
#                                           mrsandbox.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: MRSandbox as a simulator of MRSuite: pulse sequences from its library or a Pulseq file, #
#          drawn as diagrams, acquisitions of its virtual scanner on its virtual brain as          #
#          NIfTI-MRS+ with the concentrations that made them as truth, and the basis set of the   #
#          same simulation. MRSandbox is Julian Merkofer's own package (github.com/julianmer/     #
#          MRSandbox); it says how to install itself when missing.                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite.core import provenance
from mrsuite.core.basis import CENTRE, Basis

Truth = List[Dict[str, float]]

# MRSandbox's library by a short name, with the timings its constructor asks for and the
# values MRSandbox's own README uses where it states none; the localized ones play shaped
# pulses under gradients (slower), the physical designs a Pulseq file can hold
SEQUENCES = {"press": ("Press", {"te1_s": 0.011, "te2_s": 0.019}),
             "steam": ("Steam", {"te_s": 0.020, "tm_s": 0.010}),
             "slaser": ("SemiLaser", {"te_s": 0.026}),
             "megapress": ("MegaPress", {"edit_ppm": 1.88}),
             "hermes": ("Hermes", {}),
             "hercules": ("Hercules", {}),
             "pulse-acquire": ("PulseAcquire", {}),
             "press-localized": ("LocalizedPress", {"te1_s": 0.011, "te2_s": 0.019,
                                                    "f0_mhz": 127.73, "size_mm": 20.0}),
             "steam-localized": ("LocalizedSteam", {"te_s": 0.020, "tm_s": 0.010,
                                                    "f0_mhz": 127.73, "size_mm": 20.0}),
             "slaser-localized": ("LocalizedSemiLaser", {"te_s": 0.026, "f0_mhz": 127.73,
                                                         "size_mm": 20.0})}


#**************#
#   imported   #
#**************#
def _mrsandbox() -> Any:
    """The package, or how to get it. Imported here, so Keras picks its backend when a
    sequence is first made (MRSandbox takes the first one installed, JAX before PyTorch,
    unless KERAS_BACKEND says)."""
    try:
        import mrsandbox
    except ImportError:
        raise ImportError("MRSandbox is not installed; pip install mrsandbox[jax] (or, from "
                          "a checkout, pip install -e <MRSandbox>)") from None
    return mrsandbox


#*************#
#   version   #
#*************#
def version() -> Optional[str]:
    return getattr(_mrsandbox(), "__version__", None)


#**************#
#   sequence   #
#**************#
def sequence(kind: str = "press", **timings: float) -> Any:
    """
    A pulse sequence of MRSandbox's library.

    Args:
        kind (str): One of SEQUENCES: 'press', 'steam', 'slaser', 'megapress', 'hermes',
            'hercules' or 'pulse-acquire'.
        **timings: The constructor's timings in seconds (te1_s, te2_s, te_s, tm_s, tr_s,
            edit_ppm, ...), over the defaults of SEQUENCES.

    Returns:
        mrsandbox.sequences.Sequence: As a scan takes it.
    """
    if kind not in SEQUENCES:
        raise ValueError(f"kind must be one of {sorted(SEQUENCES)}, not {kind!r}")
    name, defaults = SEQUENCES[kind]
    return getattr(_mrsandbox(), name)(**{**defaults, **timings})


#*****************#
#   read pulseq   #
#*****************#
def read_pulseq(path: str, centre: float = 4.65, readout: int = 0) -> Any:
    """
    A Pulseq .seq file as a sequence MRSandbox plays (``mrsandbox.io.pulseq.read_seq``):
    its RF, gradient and delay blocks, one readout of its ADC blocks.

    Args:
        path (str): The .seq file.
        centre (float): The ppm of the scanner's carrier, water at 4.65.
        readout (int): Which ADC block's readout the sequence carries, counted from 0.
    """
    from mrsandbox.io.pulseq import read_seq
    return read_seq(path, center_ppm=centre, readout=readout)


#******************#
#   write pulseq   #
#******************#
def write_pulseq(sequence: Any, path: str, points: int = 2048, bandwidth: float = 4000.0,
                 frequency: float = 127.73, centre: float = 4.65) -> List[str]:
    """
    A sequence written as a Pulseq .seq file (``mrsandbox.io.pulseq.write_seq``, through
    pypulseq): a physical design, as the localized sequences are; MRSandbox refuses an
    idealized one, which no scanner can play.

    Args:
        sequence: A sequence of ``sequence`` or ``read_pulseq``.
        path (str): The .seq file to write.
        points (int): Points of the readout.
        bandwidth (float): Spectral width in Hz.
        frequency (float): Spectrometer frequency in MHz.
        centre (float): The ppm of the carrier.

    Returns:
        List[str]: The file written, as a block hands files on.
    """
    from mrsandbox.io.pulseq import write_seq
    write_seq(sequence, path, points, 1.0 / bandwidth, frequency, center_ppm=centre)
    return [path]


#*************#
#   virtual   #
#*************#
def _virtual(sequence: Any, brain: str, scanner: str, timings: Dict[str, float],
             imperfections: bool = True, mm: bool = True, lipids: bool = True) -> Any:
    """MRSandbox's VirtualScan of a brain and scanner preset playing a sequence, given or
    named; without ``imperfections`` the scanner is ideal (no frequency or phase error, no
    eddy currents, no spurious echo, no water from outside the voxel), without ``mm`` or
    ``lipids`` the voxel has none."""
    sandbox = _mrsandbox()
    if isinstance(sequence, str):
        sequence = globals()["sequence"](sequence, **timings)
    options: Dict[str, Any] = {}
    if not imperfections:
        options["params"] = sandbox.Params(
            acquisition=dict(shift_hz=0.0, phase_deg=0.0, eddy_std_rad=0.0,
                             eddy_transient_rad=0.0),
            spurious=dict(amplitude=0.0), oov=dict(fraction=0.0))
    if not mm:
        options["mm"] = sandbox.NoMM()
    if not lipids:
        options["lipids"] = sandbox.NoLipids()
    return sandbox.VirtualScan(sandbox.Brain.load(brain), sandbox.Scanner.preset(scanner),
                               sequence, **options)


#***************#
#   echo time   #
#***************#
def _te_ms(sequence: Any) -> Optional[float]:
    """A sequence's echo time in ms from its timings: te_s, or te1_s + te2_s."""
    timings = vars(sequence)
    if "te_s" in timings:
        return 1000.0 * float(timings["te_s"])
    if "te1_s" in timings and "te2_s" in timings:
        return 1000.0 * float(timings["te1_s"] + timings["te2_s"])
    return None


#***********#
#   basis   #
#***********#
def basis(sequence: Any = "press", brain: str = "icbm152", scanner: str = "generic_3t",
          position: Sequence[float] = (0.0, -45.0, 35.0), size: float = 20.0, seed: int = 0,
          points: int = 2048, bandwidth: float = 4000.0, imperfections: bool = True,
          **timings: float) -> Basis:
    """
    The basis set of MRSandbox's own simulation: every metabolite present in the voxel,
    rendered by its engine through the sequence at the T2 and B1 the seed draws
    (``Voxel.metabolite_basis``), on the axis every basis set has (water at CENTRE; the
    engine renders at the chemical shifts themselves, so the FIDs are demodulated by it).
    A scan hands out the same basis beside its spectra; this is it without a scan.

    Args:
        sequence: As ``scan`` takes it.
        brain, scanner, position, size: As ``scan`` takes them.
        seed (int): The draw whose relaxation and transmit the basis is rendered at.
        points (int): Points of each FID.
        bandwidth (float): Spectral width in Hz.
        imperfections (bool): As ``scan`` takes it.

    Returns:
        Basis: One FID per metabolite present, named as MRSandbox names them.
    """
    virtual = _virtual(sequence, brain, scanner, timings, imperfections)
    return _basis(virtual, virtual.voxel(tuple(position), size), seed, points, bandwidth,
                  {"brain": brain, "scanner": scanner, "position": list(position),
                   "size": size, "imperfections": imperfections, **timings})


#******************#
#   basis of one   #
#******************#
def _basis(virtual: Any, voxel: Any, seed: int, points: int, bandwidth: float,
           arguments: Dict[str, Any]) -> Basis:
    """The basis of a voxel of a virtual scan at one seed's draw, demodulated onto the axis
    every basis set has (see ``basis``)."""
    from mrsandbox.scan.voxel import draw

    _, truth, _ = draw(virtual.params, seed)
    rendered = np.asarray(voxel.metabolite_basis(truth, points, 1.0 / bandwidth),
                          dtype=complex)
    names = list(truth.conc)
    present = [k for k in range(len(names)) if np.abs(rendered[k]).max() > 0]
    frequency = virtual.database.gamma("1H") * virtual.engine.b0_t          # MHz
    t = np.arange(points) / bandwidth
    fids = rendered[present] * np.exp(-2j * np.pi * CENTRE * frequency * t)
    played = virtual.sequence
    return Basis([names[k] for k in present], fids, bandwidth, float(frequency), centre=CENTRE,
                 te=_te_ms(played), seq=type(played).__name__,
                 source={"simulated": {"program": "MRSandbox", "version": version(),
                                       "function": "Voxel.metabolite_basis",
                                       "arguments": {"sequence": type(played).__name__,
                                                     "seed": seed, "points": points,
                                                     "bandwidth": bandwidth, **arguments}}})


#**********#
#   mrsi   #
#**********#
def mrsi(sequence: Any = "press", brain: str = "icbm152", scanner: str = "generic_3t",
         centre: Sequence[float] = (0.0, -20.0, 35.0), matrix: Sequence[int] = (16, 16),
         voxel: float = 10.0, seed: int = 0, points: int = 2048, bandwidth: float = 4000.0,
         averages: Optional[int] = None, imperfections: bool = True, mm: bool = True,
         lipids: bool = True, **timings: float) -> Tuple[NIfTI_MRS_Plus, Truth, Basis]:
    """
    A slice acquired by MRSandbox's virtual scanner: a lattice of voxels through its virtual
    brain, phase encoded and decoded as a scan would, as one spatial NIfTI-MRS (x by y
    voxels, each its spectrum) with the drawn truth in the header; and the basis set of the
    very sequence it played. One draw serves every voxel, as one subject has one set of
    concentrations: the tissue under each voxel varies the signal, not the truth.

    Args:
        sequence: A sequence (``sequence``, ``read_pulseq``), or a kind of SEQUENCES by
            name, with ``timings`` for it.
        brain (str): A brain preset of MRSandbox, e.g. 'icbm152'.
        scanner (str): A scanner preset, e.g. 'generic_3t'.
        centre (Sequence[float]): The slice's centre in mm, in the brain's frame.
        matrix (Sequence[int]): Voxels across and down.
        voxel (float): The voxel's edge in mm.
        seed (int): The draw.
        points (int): Points of each FID.
        bandwidth (float): Spectral width in Hz.
        averages (Optional[int]): Encodes averaged, weighted as MRSandbox weights them;
            None for one.
        imperfections (bool): The scanner's, as in ``scan``.
        mm (bool): Macromolecules in the voxels.
        lipids (bool): Lipids from the skull.

    Returns:
        Tuple[NIfTI_MRS_Plus, Truth, Basis]: The slice, with the acquisition in its
        ProcessingApplied; the drawn concentrations in mM, once, by metabolite; and the
        basis set.
    """
    from mrsandbox.phantom import Unsuppressed

    virtual = _virtual(sequence, brain, scanner, timings, imperfections, mm, lipids)
    lattice = virtual.mrsi(center_mm=tuple(centre), matrix=tuple(matrix), voxel_mm=voxel,
                           averages=averages)
    item = lattice.acquire(seed=seed, npoints=points, dwell_s=1.0 / bandwidth)
    item.add_hdr_field("WaterSuppressed", not isinstance(virtual.water, Unsuppressed))
    drawn = item.hdr_ext["SimulationGroundTruth"]
    truth = [{name: float(c) for name, c in drawn["params"]["conc"].items()}]
    provenance.append(item, provenance.entry(
        "MRSandbox", version(), "MRSI.acquire",
        {"sequence": type(virtual.sequence).__name__, "brain": brain, "scanner": scanner,
         "centre": list(centre), "matrix": list(matrix), "voxel": voxel, "seed": seed,
         "points": points, "bandwidth": bandwidth, "averages": averages,
         "imperfections": imperfections, "mm": mm, "lipids": lipids}))
    played = _basis(virtual, virtual.voxel(tuple(centre), voxel), seed, points, bandwidth,
                    {"brain": brain, "scanner": scanner, "centre": list(centre),
                     "voxel": voxel, "imperfections": imperfections, **timings})
    return NIfTI_MRS_Plus([item]), truth, played


#*************#
#   diagram   #
#*************#
def diagram(sequence: Any, scanner: str = "generic_3t", readout_ms: float = 12.0) -> Any:
    """
    A pulse sequence diagram of a sequence, as MRSandbox's console draws one: the RF lane
    with shaped pulses' envelopes and ideal pulses as bars, each with its flip angle,
    coherence filters marked, the gradient lanes as the scanner would play them, and the
    readout.

    Args:
        sequence: A sequence of ``sequence`` or ``read_pulseq``.
        scanner (str): The scanner preset whose gradient limits shape the waveforms.
        readout_ms (float): The readout's length on the diagram.

    Returns:
        matplotlib.figure.Figure: Made without pyplot.
    """
    from matplotlib.figure import Figure
    from mrsandbox.sequences.events import CoherenceFilter, Delay, Gradient, ShapedPulse
    from mrsandbox.sequences.timeline import event_spans, gradient_waveforms

    sandbox = _mrsandbox()
    spans = [(1000.0 * a, 1000.0 * b) for a, b in event_spans(sequence)]
    end = spans[-1][1] if spans else 0.0
    ink, rf_colour = "#8a9bb3", "#5274f0"
    figure = Figure(figsize=(3.2, 1.7), layout="constrained")
    figure.set_facecolor("none")
    lanes = figure.subplots(4, 1, sharex=True, height_ratios=(2.2, 1, 1, 1))
    rf, grads = lanes[0], lanes[1:]
    for event, (start, stop) in zip(sequence.events, spans):
        if isinstance(event, ShapedPulse):
            envelope = np.abs(np.asarray(event.envelope, dtype=complex))
            height = 1.0 if event.flip_deg is None else float(event.flip_deg) / 180.0
            times = np.linspace(start, stop, envelope.size)
            rf.fill_between(times, 0, height * envelope / (envelope.max() or 1.0),
                            color=rf_colour, alpha=0.9, lw=0)
            if event.flip_deg is not None:
                rf.text(0.5 * (start + stop), height + 0.08, f"{float(event.flip_deg):.0f}°",
                        ha="center", va="bottom", fontsize=6, color=ink)
        elif isinstance(event, CoherenceFilter):
            rf.plot([start, start], [0, 0.35], color=ink, lw=0.8, ls=(0, (1, 1.5)))
            rf.text(start, 0.4, "p", ha="center", va="bottom", fontsize=6, color=ink)
        elif hasattr(event, "flip_deg") and not isinstance(event, (Delay, Gradient)):
            height = float(event.flip_deg) / 180.0
            rf.plot([start, start], [0, height], color=rf_colour, lw=2.0, solid_capstyle="butt")
            rf.text(start, height + 0.08, f"{float(event.flip_deg):.0f}°", ha="center",
                    va="bottom", fontsize=6, color=ink)
    rf.axvspan(end, end + readout_ms, color=ink, alpha=0.18, lw=0)
    rf.text(end + readout_ms / 2, 0.55, "ADC", ha="center", va="center", fontsize=6, color=ink)
    rf.set_ylim(0, 1.35)
    times, waves = gradient_waveforms(sequence, sandbox.Scanner.preset(scanner), 5e-5,
                                      (end + readout_ms) / 1000.0)
    for axis, lane in zip("xyz", grads):
        wave = waves.get(axis, np.zeros(times.size))
        lane.fill_between(1000.0 * times, 0, wave, color=ink, alpha=0.6, lw=0)
        lane.axhline(0, color=ink, lw=0.4, alpha=0.5)
        top = float(np.abs(wave).max()) or 1.0
        lane.set_ylim(-1.15 * top, 1.15 * top)
    for lane, label in zip(lanes, ("RF", "Gx", "Gy", "Gz")):
        lane.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=6, color=ink,
                        labelpad=4)
        lane.set_facecolor("none")
        lane.set_yticks([])
        for side in ("top", "right", "left"):
            lane.spines[side].set_visible(False)
        lane.spines["bottom"].set_color(ink)
        lane.tick_params(axis="x", colors=ink, labelsize=6, length=2)
    for lane in lanes[:-1]:
        lane.spines["bottom"].set_visible(False)
        lane.tick_params(axis="x", length=0)
    lanes[-1].set_xlabel("ms", fontsize=6, color=ink, labelpad=1)
    lanes[-1].set_xlim(-0.02 * (end + readout_ms), 1.02 * (end + readout_ms))
    return figure


#**********#
#   scan   #
#**********#
def scan(sequence: Any = "press", brain: str = "icbm152", scanner: str = "generic_3t",
         position: Sequence[float] = (0.0, -45.0, 35.0), size: float = 20.0, n: int = 1,
         seed: int = 0, points: int = 2048, bandwidth: float = 4000.0, averages: int = 1,
         imperfections: bool = True, mm: bool = True, lipids: bool = True,
         **timings: float) -> Tuple[NIfTI_MRS_Plus, Truth, Basis]:
    """
    Spectra acquired by MRSandbox's virtual scanner: a voxel of its virtual brain, every
    quantity drawn from its priors per seed, through the sequence, as NIfTI-MRS with the
    whole drawn truth in the header (SimulationGroundTruth); and the basis set of the very
    sequence and pulses it played, rendered by the same engine at the first seed's draw.

    Args:
        sequence: A sequence (``sequence``, ``read_pulseq``), or a kind of SEQUENCES by
            name, with ``timings`` for it.
        brain (str): A brain preset of MRSandbox, e.g. 'icbm152' (fetched once and cached).
        scanner (str): A scanner preset, e.g. 'generic_3t', 'siemens_prisma_3t'.
        position (Sequence[float]): The voxel's centre in mm, in the brain's frame.
        size (float): The voxel's edge in mm.
        n (int): Spectra: seeds ``seed`` to ``seed + n - 1``, each its own draw.
        seed (int): The first seed.
        points (int): Points of the FID.
        bandwidth (float): Spectral width in Hz.
        averages (int): Transients averaged into each spectrum.
        imperfections (bool): The scanner's: frequency and phase error, eddy currents, a
            spurious echo, water from outside the voxel; False gives an ideal scanner.
        mm (bool): Macromolecules in the voxel.
        lipids (bool): Lipids from the skull.

    Returns:
        Tuple[NIfTI_MRS_Plus, Truth, Basis]: The spectra, each with the acquisition in its
        ProcessingApplied; the drawn concentrations in mM per spectrum, by metabolite; and
        the basis set.
    """
    from mrsandbox.phantom import Unsuppressed

    virtual = _virtual(sequence, brain, scanner, timings, imperfections, mm, lipids)
    sequence = virtual.sequence
    voxel = virtual.voxel(tuple(position), size)
    items, truth = [], []
    for k in range(n):
        item = voxel.acquire(seed=seed + k, npoints=points, dwell_s=1.0 / bandwidth,
                             averages=averages)
        # what the scan did to the water, as NIfTI-MRS states it
        item.add_hdr_field("WaterSuppressed", not isinstance(virtual.water, Unsuppressed))
        drawn = item.hdr_ext["SimulationGroundTruth"]
        truth.append({name: float(c) for name, c in drawn["params"]["conc"].items()})
        provenance.append(item, provenance.entry(
            "MRSandbox", version(), "Voxel.acquire",
            {"sequence": type(sequence).__name__, "brain": brain, "scanner": scanner,
             "position": list(position), "size": size, "seed": seed + k, "points": points,
             "bandwidth": bandwidth, "averages": averages, "imperfections": imperfections,
             "mm": mm, "lipids": lipids}))
        items.append(item)
    played = _basis(virtual, voxel, seed, points, bandwidth,
                    {"brain": brain, "scanner": scanner, "position": list(position),
                     "size": size, "imperfections": imperfections, **timings})
    return NIfTI_MRS_Plus(items), truth, played
