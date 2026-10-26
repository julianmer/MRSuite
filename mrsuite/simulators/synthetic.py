####################################################################################################
#                                           synthetic.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: Spectra synthesised by the toolboxes from a basis set or a brain model - FSL-MRS's      #
#          synthetic module, spant's sim_brain_1h - as NIfTI-MRS+ with the concentrations that     #
#          made them as truth, each spectrum's recipe in its processing history.                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite.core import provenance
from mrsuite.core.basis import as_file

Truth = List[Dict[str, float]]


#*************#
#   fsl mrs   #
#*************#
def fsl_mrs(basis: Any, n: int = 1, concentrations: Optional[Dict[str, float]] = None,
            broadening: Tuple[float, float] = (9.0, 0.0), shift: float = 0.0, phi0: float = 0.0,
            phi1: float = 0.0, noise: float = 0.1, baseline: Optional[Sequence[float]] = None,
            seed: int = 0) -> Tuple[NIfTI_MRS_Plus, Truth]:
    """
    Spectra synthesised by FSL-MRS from a basis set (``fsl_mrs.utils.synthetic.
    syntheticFromBasisFile``): its Voigt signal model on the basis functions at the given
    concentrations, broadened, shifted, phased, with a baseline and white noise.

    Args:
        basis: A Basis (``mrsuite.io.basis``) or a basis file or folder FSL-MRS reads.
        n (int): Spectra to make; the noise differs, the rest is the same.
        concentrations (Optional[Dict[str, float]]): Per metabolite; FSL-MRS's standard
            concentrations for the metabolites left out (a random 1 to 5 for one it does
            not know, as it says).
        broadening (Tuple[float, float]): Lorentzian and Gaussian broadening in Hz.
        shift (float): Frequency shift in ppm.
        phi0, phi1 (float): Zero- and first-order phase, in radians and seconds, as
            FSL-MRS takes them.
        noise (float): The noise variance, as FSL-MRS's one-coil covariance.
        baseline (Optional[Sequence[float]]): Baseline polynomial coefficients, FSL-MRS's.
        seed (int): Seeds NumPy, which FSL-MRS draws its noise from, so the same seed
            gives the same spectra.

    Returns:
        Tuple[NIfTI_MRS_Plus, Truth]: The spectra, each with how it was made in its
        ProcessingApplied, and the concentrations per spectrum, by metabolite.
    """
    import fsl_mrs
    from fsl_mrs.utils.mrs_io import read_basis
    from fsl_mrs.utils.synthetic import syntheticFromBasisFile

    path = str(as_file(basis))
    read = read_basis(path)
    np.random.seed(seed)
    items, truth = [], []
    for _ in range(n):
        fids, mrs, conc = syntheticFromBasisFile(
            path, concentrations=concentrations, broadening=tuple(broadening), shifting=shift,
            phi0=phi0, phi1=phi1, noisecovariance=[[noise]], baseline=baseline,
            bandwidth=read.original_bw, points=read.original_points)
        item = gen_nifti_mrs(np.asarray(fids).reshape(1, 1, 1, -1), float(mrs.dwellTime),
                             float(mrs.centralFrequency) / 1e6)
        provenance.append(item, provenance.entry(
            "FSL-MRS", fsl_mrs.__version__, "syntheticFromBasisFile",
            {"basis": provenance.describe(path), "concentrations": concentrations,
             "broadening": list(broadening), "shift": shift, "phi0": phi0, "phi1": phi1,
             "noise": noise, "baseline": None if baseline is None else list(baseline),
             "seed": seed}))
        items.append(item)
        truth.append({name: float(c) for name, c in zip(mrs.names, conc)})
    return NIfTI_MRS_Plus(items), truth


#***********#
#   spant   #
#***********#
def spant(n: int = 1, type: str = "normal_v2", pul_seq: str = "seq_slaser_ideal",
          frequency: float = 123.2, bandwidth: float = 2000.0, points: int = 1024,
          centre: float = 4.65, seed: int = 0, **timings: Any) -> Tuple[NIfTI_MRS_Plus, Truth]:
    """
    Spectra simulated by spant's ``sim_brain_1h``: a brain-like spectrum of its basis at
    its typical amplitudes, through an ideal pulse sequence of its library.

    Args:
        n (int): Spectra to make, each from its own seed on.
        type (str): spant's amplitude set, e.g. 'normal_v2'.
        pul_seq (str): spant's sequence, e.g. 'seq_press_ideal', 'seq_slaser_ideal',
            'seq_steam_ideal'; its timings by name in seconds (TE1, TE2, TE3, TE, TM).
        frequency (float): Spectrometer frequency in MHz.
        bandwidth (float): Spectral width in Hz.
        points (int): Points.
        centre (float): The ppm at the centre.
        seed (int): spant's noise draws from R's generator, seeded so.

    Returns:
        Tuple[NIfTI_MRS_Plus, Truth]: The spectra, and spant's amplitudes per spectrum
        where it reports them (``full_output``), else empty.
    """
    from mrsuite import r
    from mrsuite.bridges.spant import _read_back
    from mrsuite.octave import scratch

    folder = scratch("mrsuite_spant_sim_")
    try:
        items, truth = [], []
        for k in range(n):
            output = os.path.join(folder, f"sim{k}.nii.gz")
            done = r.run({"task": "simulate", "func": "sim_brain_1h", "pul_seq": pul_seq,
                          "acq_paras": {"ft": frequency * 1e6, "fs": bandwidth, "N": points,
                                        "ref": centre},
                          "args": [], "kwargs": {"type": type, "full_output": True, **timings},
                          "seed": seed + k, "output": output}, folder)
            # read as the bridge reads what spant writes, on a header of the asked grid
            like = gen_nifti_mrs(np.zeros((1, 1, 1, points), dtype=complex), 1.0 / bandwidth,
                                 frequency)
            item = _read_back(output, like)
            item.add_hdr_field("SpecFreqChemShift", centre)
            provenance.append(item, provenance.entry(
                "spant", r.info()["spant"], "sim_brain_1h",
                {"type": type, "pul_seq": pul_seq, "frequency": frequency,
                 "bandwidth": bandwidth, "points": points, "centre": centre,
                 "seed": seed + k, **timings}))
            items.append(item)
            truth.append({k_: float(v) for k_, v in (done.get("amps") or {}).items()})
        return NIfTI_MRS_Plus(items), truth
    finally:
        shutil.rmtree(folder, ignore_errors=True)
