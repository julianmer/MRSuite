####################################################################################################
#                                              mlp.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: The quantification MLP of the out-of-distribution robustness paper, supervised and      #
#          self-supervised alike: the two differ in training only. The network is the original,    #
#          from externals/ood-robust-mrs; this wraps it as a Model, spectra in, FitResult out.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from pathlib import Path
from typing import Dict, List

import torch

from mrsuite.core.provenance import git_version
from mrsuite.core.result import FitResult
from mrsuite.models.base import load_file, spectra
from mrsuite.models.pytorch import TorchModel


#**************************************************************************************************#
#                                            Class MLP                                             #
#**************************************************************************************************#
#                                                                                                  #
# A multi-layer perceptron that reads concentrations and signal parameters off a spectrum.         #
#                                                                                                  #
#**************************************************************************************************#
class MLP(TorchModel):
    """A multi-layer perceptron that reads concentrations and signal parameters off a spectrum.

    The spectrum is cut to the points ``first`` to ``last``, scaled to unit norm and
    read by the network; the concentrations and baseline are scaled back by the same
    norm, so they come out in the data's units, as amplitudes of the basis spectra.
    This is FrameworkNN.forward of the original, with its default normalisation.

    Args:
        metabolites (List[str]): The basis spectra, in the network's output order.
        first (int): First point of the spectral window.
        last (int): The point after it.
        acquisition (Dict[str, float]): What it was trained on; see Model.
        width (int): Width of the first hidden layer; each next one halves it.
        depth (int): Number of hidden layers.

    Attributes:
        repo (Path): The original repository, which the network is loaded from.
        signal_parameters (List[str]): What the output holds after the concentrations:
            two linewidths, shift, zero- and first-order phase, and a baseline.
    """

    repo = Path(__file__).resolve().parents[2] / "externals" / "ood-robust-mrs"
    signal_parameters = (["gamma", "sigma", "eps", "phi0", "phi1"]
                         + [f"baseline{i}" for i in range(6)])

    #**********#
    #   init   #
    #**********#
    def __init__(self, metabolites: List[str], first: int, last: int,
                 acquisition: Dict[str, float], width: int = 512, depth: int = 3) -> None:
        super().__init__()
        self.metabolites, self.first, self.last = list(metabolites), first, last
        self.acquisition = dict(acquisition)
        # 'relu' as the published models ran: their configs' 'activation' was never read
        self.net = load_file(self.repo / "models" / "nnModels.py").MLP(
            (2, last - first), len(metabolites), width=width, depth=depth, act="relu")

    #*************#
    #   prepare   #
    #*************#
    def prepare(self, items) -> torch.Tensor:
        return torch.from_numpy(spectra(items))

    #*************#
    #   forward   #
    #*************#
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Spectra (batch, 2, points) to parameters (batch, metabolites + 11)."""
        xs = x[:, :, self.first:self.last]
        norm = torch.sqrt(torch.sum(xs[:, 0, :] ** 2 + xs[:, 1, :] ** 2, dim=1, keepdim=True))
        norm = torch.clamp(norm, min=1e-8)
        theta = self.net((xs / norm.unsqueeze(-1)).float())
        theta[:, :len(self.metabolites)] *= norm
        theta[:, -6:] *= norm
        return theta

    #***************#
    #   interpret   #
    #***************#
    def interpret(self, out: torch.Tensor, items) -> List[FitResult]:
        results = []
        for row in out.cpu().double().numpy():
            raw = dict(zip(self.metabolites, row[:len(self.metabolites)].tolist()))
            concentrations = {"raw": raw}
            creatine = raw.get("Cr", 0) + raw.get("PCr", 0)
            if creatine:
                concentrations["internal"] = {m: v / creatine for m, v in raw.items()}
            parameters = dict(zip(self.signal_parameters, row[len(self.metabolites):].tolist()))
            results.append(FitResult(method=f"mrsuite:{type(self).__name__}",
                                     metabolites=list(self.metabolites),
                                     concentrations=concentrations, raw=parameters))
        return results

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, str]:
        return {**super().runtime(), "ood-robust-mrs": git_version(str(self.repo))}
