####################################################################################################
#                                             wand.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: WAND, the wavelet-based decomposition of a spectrum into its metabolites, baseline      #
#          and artifacts. The U-Net and the wavelet transform are the originals, from              #
#          externals/wand-for-mrs; this wraps them as a Model.                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite.core.provenance import git_version
from mrsuite.models.base import load_file, spectra
from mrsuite.models.pytorch import TorchModel


#**************************************************************************************************#
#                                            Class WAND                                            #
#**************************************************************************************************#
#                                                                                                  #
# Decomposes a spectrum into its metabolites, baseline and artifacts in the wavelet domain.        #
#                                                                                                  #
#**************************************************************************************************#
class WAND(TorchModel):
    """Decomposes a spectrum into its metabolites, baseline and artifacts in the wavelet domain.

    The spectrum's window from ``first`` to ``last`` is taken to scalograms of its real
    and imaginary parts, the U-Net predicts a mask per component, and each masked
    scalogram is transformed back. This is FrameworkWAND.prepareData and .forward of
    the original, with its fixed settings: Morlet wavelet (mu 14), 32 voices, softmax
    masks, and the artifacts as what the other masks leave.

    Each spectrum comes back as a NIfTI-MRS with the components along a fifth dimension,
    named in its header - the metabolites, then 'baseline', then 'artifacts' - each
    extended to the full length by repeating its edge values, as the original's scripts do.

    Args:
        metabolites (List[str]): The basis spectra, in the network's output order.
        first (int): First point of the spectral window.
        last (int): The point after it.
        acquisition (Dict[str, float]): What it was trained on; see Model.

    Attributes:
        repo (Path): The original repository, which the network and transform are loaded from.
    """

    repo = Path(__file__).resolve().parents[2] / "externals" / "wand-for-mrs"

    #**********#
    #   init   #
    #**********#
    def __init__(self, metabolites: List[str], first: int, last: int,
                 acquisition: Dict[str, float]) -> None:
        super().__init__()
        self.metabolites, self.first, self.last = list(metabolites), first, last
        self.acquisition = dict(acquisition)
        self.components = self.metabolites + ["baseline", "artifacts"]
        self.cwt = load_file(self.repo / "utils" / "wavelets.py").CWT(
            last - first, wavelet=("morlet", {"mu": 14}), scales="log-piecewise", nv=32,
            padtype=None)
        self.net = load_file(self.repo / "frameworks" / "nnModels.py").UNet(
            n_channels=2, n_classes=2 * (len(metabolites) + 1))

    #*************#
    #   prepare   #
    #*************#
    def prepare(self, items) -> torch.Tensor:
        return torch.from_numpy(spectra(items))

    #*************#
    #   forward   #
    #*************#
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Spectra (batch, 2, points) to components (batch, 2, window, components)."""
        x = x - x[:, :, self.first:self.last].mean(dim=-1, keepdim=True)
        cwtmatr = self.cwt(x[:, :, self.first:self.last])
        masks = self.net(cwtmatr.abs().float())
        masks = masks.view(masks.shape[0], len(self.metabolites) + 1, 2, *masks.shape[-2:])
        artifacts = torch.stack((1 - masks[:, :, 0].sum(dim=1, keepdim=True),
                                 1 - masks[:, :, 1].sum(dim=1, keepdim=True)), dim=2)
        masks = torch.softmax(torch.cat((masks, artifacts), dim=1), dim=1)
        specs = self.cwt.inverse(masks * cwtmatr.unsqueeze(1)).float()
        return specs.permute(0, 2, 3, 1)

    #***************#
    #   interpret   #
    #***************#
    def interpret(self, out: torch.Tensor, items) -> list:
        n = int(self.acquisition.get("points", items[0].shape[3]))
        full = torch.nn.functional.pad(out, (0, 0, self.first, n - self.last), mode="replicate")
        full = full.cpu().double().numpy()
        fids = np.fft.ifft(full[:, 0] + 1j * full[:, 1], axis=1)     # (batch, points, components)
        results = []
        for fid, item in zip(fids, items):
            result = gen_nifti_mrs(fid.reshape(1, 1, 1, *fid.shape), item.dwelltime,
                                   item.spectrometer_frequency[0], nucleus=item.nucleus[0],
                                   dim_tags=["DIM_USER_0", None, None])
            result.set_dim_tag("DIM_USER_0", "DIM_USER_0", info="WAND components", header={
                "Component": {"Value": self.components,
                              "Description": "The metabolites, then baseline and artifacts"}})
            results.append(result)
        return results

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, str]:
        return {**super().runtime(), "wand-for-mrs": git_version(str(self.repo))}
