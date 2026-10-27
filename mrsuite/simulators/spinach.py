####################################################################################################
#                                            spinach.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: Spinach (Hogben, Krzystyniak, Charnock, Hore and Kuprov), the spin dynamics library,    #
#          run as published: 1H spin systems, their chemical shifts and J-couplings, through one   #
#          of its pulse sequences in the full Liouville space, as a basis set every fit takes.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import shutil
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from mrsuite.core.basis import CENTRE, Basis
from mrsuite.octave import scratch
from mrsuite.octave.toolbox import Toolbox

toolbox = Toolbox("spinach", "Spinach", shims="spinach")


#**************#
#   simulate   #
#**************#
def simulate(systems: Dict[str, Dict[str, Any]], sequence: str = "acquire", field: float = 3.0,
             points: int = 2048, bandwidth: float = 2000.0, linewidth: float = 2.0,
             centre: float = CENTRE, approximation: str = "none",
             parameters: Optional[Dict[str, Any]] = None) -> Basis:
    """
    A basis set simulated by Spinach: each spin system built by its ``create`` and
    ``basis``, in spherical-tensor Liouville space (``bas.formalism = 'sphten-liouv'``),
    and played through one of its pulse sequences by ``liquid``, from the transverse
    magnetisation an ideal excitation leaves (``L+``) and detected as ``L+``, as Spinach's
    own 1D examples do (examples/fundamentals/roof_effect.m).

    Args:
        systems (Dict[str, Dict[str, Any]]): Name -> spin system: 'shifts', the protons'
            chemical shifts in ppm, and 'couplings', J in Hz by pair of 0-based proton
            indices, e.g. ``{'Lac': {'shifts': [4.10, 1.31, 1.31, 1.31], 'couplings':
            {(0, 1): 6.9, (0, 2): 6.9, (0, 3): 6.9}}}``.
        sequence (str): A Spinach pulse sequence (its experiments/ folder) taking the
            parameters of ``acquire``: sweep, points, the initial and detected states.
        field (float): Field strength in T.
        points (int): Points of the FID.
        bandwidth (float): Spectral width in Hz.
        linewidth (float): Lorentzian linewidth in Hz, applied to the FIDs.
        centre (float): The ppm at the centre of the spectrum, where Spinach's offset puts
            the rotating frame.
        approximation (str): Spinach's basis set. Only 'none', the full basis, runs under
            Octave; the restricted ones ('IK-2' and others) need MATLAB.
        parameters (Optional[Dict[str, Any]]): Further fields of Spinach's ``parameters``,
            as a sequence other than ``acquire`` asks for them.

    Returns:
        Basis: One FID per spin system, at ``centre``.

    Raises:
        ValueError: For a restricted basis under Octave.
    """
    if approximation != "none" and not toolbox.matlab:
        raise ValueError(f"Spinach's restricted basis approximation={approximation!r} needs "
                         f"sparse broadcasting Octave lacks; under Octave only 'none' runs")
    octave = toolbox.octave
    folder = scratch("spinach_")
    names, fids = list(systems), []
    try:
        for name in names:
            shifts, couplings = _system(systems[name])
            octave.push("mrsuite_shifts", np.asarray(shifts, dtype=float))
            octave.push("mrsuite_couplings", np.asarray(couplings, dtype=float).reshape(-1, 3))
            if parameters:
                octave.push("mrsuite_extra", dict(parameters))
            octave.eval(f"""
                clear sys inter bas p;
                n = numel(mrsuite_shifts);
                sys.magnet = {float(field)}; sys.isotopes = repmat({{'1H'}}, 1, n);
                sys.output = 'hush'; sys.scratch = '{folder}';
                inter.zeeman.scalar = num2cell(mrsuite_shifts(:)');
                inter.coupling.scalar = cell(n, n);
                for k = 1:size(mrsuite_couplings, 1)
                    inter.coupling.scalar{{mrsuite_couplings(k, 1), mrsuite_couplings(k, 2)}} = ...
                        mrsuite_couplings(k, 3);
                end
                bas.formalism = 'sphten-liouv'; bas.approximation = {{'{approximation}'}};
                spin_system = create(sys, inter); spin_system = basis(spin_system, bas);
                mrsuite_mhz = spin('1H') * sys.magnet / (2 * pi) / 1e6;
                p = {'mrsuite_extra' if parameters else 'struct()'};
                p.spins = {{'1H'}}; p.decouple = {{}};
                p.rho0 = state(spin_system, 'L+', '1H');
                p.coil = coil_state(spin_system, 'L+', '1H');
                p.offset = {float(centre)} * mrsuite_mhz; p.sweep = {float(bandwidth)};
                p.npoints = {int(points)};
                mrsuite_fid = liquid(spin_system, @{sequence}, p, 'nmr');
            """)
            fids.append(np.asarray(octave.pull("mrsuite_fid"), dtype=complex).ravel())
        frequency = float(octave.pull("mrsuite_mhz"))
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    # Spinach's 'nmr' FID is already MRSuite's way round: a peak at +f Hz sits f / frequency
    # ppm above the centre (checked on a singlet, see tests/simulators/test_spinach.py)
    t = np.arange(points) / bandwidth
    fids = np.array(fids) * np.exp(-np.pi * linewidth * t)
    return Basis(names, fids, float(bandwidth), frequency, centre=centre, seq=sequence,
                 source={"simulated": {"program": "Spinach", "version": toolbox.version(),
                                       "function": f"liquid, {sequence}",
                                       "runtime": toolbox.runtime(),
                                       "arguments": {"systems": {
                                           name: {"shifts": list(map(float, s["shifts"])),
                                                  "couplings": [[i, j, float(J)] for (i, j), J
                                                                in s.get("couplings", {}).items()]}
                                           for name, s in systems.items()}, "field": field,
                                                     "points": points, "bandwidth": bandwidth,
                                                     "linewidth": linewidth, "centre": centre,
                                                     "approximation": approximation,
                                                     "parameters": parameters or {}}}})


#************#
#   system   #
#************#
def _system(system: Dict[str, Any]) -> Tuple[Sequence[float], list]:
    """A spin system's shifts, and its couplings as rows of 1-based (i, j, J), as Octave
    indexes Spinach's coupling matrix."""
    couplings = [(i + 1, j + 1, float(J)) for (i, j), J in system.get("couplings", {}).items()]
    return system["shifts"], couplings
