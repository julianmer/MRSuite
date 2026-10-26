####################################################################################################
#                                            pulses.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: RF pulses by FID-A's rfPulseTools, with defaults: a sinc and a Gaussian, as FID-A's    #
#          pulse structs (waveform of phase, amplitude and time per point; type; bandwidth).      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import Any


#**********#
#   sinc   #
#**********#
def sinc(lobes: int = 4, points: int = 256, kind: str = "exc") -> Any:
    """
    A sinc RF pulse by FID-A (``rf_sinc``).

    Args:
        lobes (int): Lobes of the sinc.
        points (int): Points of the waveform.
        kind (str): 'exc', 'ref' or 'inv': excitation, refocusing or inversion.

    Returns:
        FID-A's pulse struct: waveform (phase, amplitude, time per point), type, tw1 (time
        times bandwidth at half height), tbw and f0.
    """
    from mrsuite.bridges.fid_a import bridge
    return bridge.rf_sinc(int(lobes), int(points), kind)


#***********#
#   gauss   #
#***********#
def gauss(duration: float = 2.0, bandwidth: float = 2000.0, points: int = 256, kind: str = "exc",
          offset: float = 0.0) -> Any:
    """
    A Gaussian RF pulse by FID-A (``rf_gauss``).

    Args:
        duration (float): Pulse duration in ms.
        bandwidth (float): Bandwidth in Hz.
        points (int): Points of the waveform.
        kind (str): 'exc', 'ref' or 'inv'.
        offset (float): Frequency offset in Hz.

    Returns:
        FID-A's pulse struct, as ``sinc`` gives it.
    """
    from mrsuite.bridges.fid_a import bridge
    return bridge.rf_gauss(float(duration), float(bandwidth), int(points), kind, float(offset))
