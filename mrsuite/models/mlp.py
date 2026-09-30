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
from typing import Any, Dict, List, Optional

import numpy as np
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
class MLP(TorchModel, paper_url="https://arxiv.org/abs/2511.23135"):
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
    objectives = ("concentrations", "spectra")
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

    #***********#
    #   learn   #
    #***********#
    def learn(self, source: Any, steps: int, objective: Optional[str] = None,
              **kwargs) -> List[float]:
        """See Model.learn. 'concentrations' is supervised; 'spectra' is the original's
        self-supervised objective, which needs only the source's spectra and basis."""
        objective = objective or self.objectives[0]
        if objective == "spectra":
            self._voigt = self._signal_model(source)
        return super().learn(source, steps, objective=objective, **kwargs)

    #************#
    #   target   #
    #************#
    def target(self, truth, data, objective: str) -> np.ndarray:
        """What the output is held to: the concentrations of the model's metabolites, or
        for 'spectra' the spectra themselves, which the output must rebuild."""
        if objective == "spectra":
            return spectra(data).astype(np.float32)
        missing = [m for m in self.metabolites if m not in truth.concentrations]
        if missing:
            raise KeyError(f"MLP learns concentrations; the source gives none for {missing}")
        return np.stack([np.asarray(truth.concentrations[m]) for m in self.metabolites],
                        axis=1).astype(np.float32)

    #**********#
    #   loss   #
    #**********#
    def loss(self, target, out, objective: str):
        """The original's losses: 'mae_conc', the mean absolute error on non-negative
        concentrations with macromolecules left out, and for 'spectra' 'mse_specs', the
        squared error between the spectrum and the signal model's spectrum for the
        estimated parameters, over the model's window."""
        from keras import ops
        if objective == "spectra":
            if out.device.type == "meta":       # Keras' build pass carries no data, and
                return ops.sum(out) * 0.0       # would strand the signal model on 'meta'
            spec = self._voigt.forward(out)                     # complex (batch, points)
            spec = torch.stack((spec.real, spec.imag), dim=1).to(target.dtype)
            error = ops.square(spec[:, :, self.first:self.last]
                               - target[:, :, self.first:self.last])
            return ops.mean(ops.sum(error, axis=1))     # real + imaginary, as the original
        kept = ops.convert_to_tensor([0.0 if "mm" in m.lower() or "mac" in m.lower() else 1.0
                                      for m in self.metabolites])
        estimate = ops.maximum(out[:, :len(self.metabolites)], 0)
        return ops.mean(ops.abs(ops.maximum(target, 0) - estimate) * kept)

    #******************#
    #   signal model   #
    #******************#
    def _signal_model(self, source: Any) -> Any:
        """The original VoigtModel over the source's basis, which 'spectra' fits.

        The basis comes as the source gives it - NIfTI-MRS' orientation, no conjugation
        heuristics - with the time and frequency axes built as the original's basis
        object builds them.
        """
        if not source.basis:
            raise ValueError("MLP learns 'spectra' by fitting the source's basis; this "
                             "source gives none")
        missing = [m for m in self.metabolites if m not in source.basis]
        if missing:
            raise KeyError(f"MLP fits its own metabolites; the source's basis lacks "
                           f"{missing}")
        basis = np.stack([np.asarray(source.basis[m], complex) for m in self.metabolites],
                         axis=1)
        points, bandwidth = int(self.acquisition["points"]), self.acquisition["bandwidth"]
        if len(basis) != points:
            raise ValueError(f"the source's basis has {len(basis)} points, the model's "
                             f"spectra {points}")
        dwell = 1 / bandwidth
        t = np.arange(dwell, dwell * (points + 1), dwell)[:points]
        f = np.arange(-bandwidth / 2, bandwidth / 2, bandwidth / points)[:points]
        return load_file(self.repo / "simulation" / "sigModels.py").VoigtModel(
            basis=basis, first=self.first, last=self.last, t=t, f=f)

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, str]:
        return {**super().runtime(), "ood-robust-mrs": git_version(str(self.repo))}
