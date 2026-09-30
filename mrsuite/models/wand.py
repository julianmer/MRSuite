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
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite.core.provenance import git_version
from mrsuite.models.base import load_file, spectra
from mrsuite.models.pytorch import TorchModel

# one at a time through WAND.predict's swap of the running averages; module-level, as a
# lock on the model would stop it pickling for worker processes
_SWAP = threading.Lock()


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

    As a processing step it hands back an ordinary spectrum of the input's shape instead:

        wand.step(remove=["artifacts"])             # the data without them
        wand.step(remove=["baseline", "artifacts"])
        wand.step(keep=["NAA", "Cr", "PCr"])        # only these; "metabolites" for all

    ``remove`` subtracts the components from the data, so outside the network's window
    the data is untouched; ``keep`` adds the components up, zero outside the window.

    It learns the components from any source that gives each metabolite's signal and the
    baseline: the artifacts are what the data holds beyond them.

    Its batch normalisation uses the statistics of the spectra decomposed together, as
    the paper's own evaluation ran it, which quantifies the 2016 challenge better than the
    stored running averages do; a spectrum's decomposition then depends on the others in
    its batch (up to ``batch_size``). ``batch_statistics=False`` uses the averages instead.

    Args:
        metabolites (List[str]): The basis spectra, in the network's output order.
        first (int): First point of the spectral window.
        last (int): The point after it.
        acquisition (Dict[str, float]): What it was trained on; see Model.
        batch_statistics (bool): Normalise with each batch's own statistics.

    Attributes:
        repo (Path): The original repository, which the network and transform are loaded from.
    """

    repo = Path(__file__).resolve().parents[2] / "externals" / "wand-for-mrs"
    objectives = ("components",)

    #**********#
    #   init   #
    #**********#
    def __init__(self, metabolites: List[str], first: int, last: int,
                 acquisition: Dict[str, float], batch_statistics: bool = True) -> None:
        super().__init__()
        self.metabolites, self.first, self.last = list(metabolites), first, last
        self.acquisition = dict(acquisition)
        self.batch_statistics = batch_statistics
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
    #   predict   #
    #*************#
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """In inference mode; with ``batch_statistics`` the stored running averages are set
        aside for the call, so batch normalisation takes the batch's own statistics and the
        weights stay exactly as they were."""
        if not self.batch_statistics:
            return super().predict(x)
        with _SWAP:                     # a second thread would keep the Nones for good
            norms = [m for m in self.modules()
                     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
            kept = [(m.running_mean, m.running_var) for m in norms]
            for m in norms:
                m.running_mean = m.running_var = None
            try:
                return super().predict(x)
            finally:
                for m, (mean, var) in zip(norms, kept):
                    m.running_mean, m.running_var = mean, var

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
    def interpret(self, out: torch.Tensor, items, keep: Optional[List[str]] = None,
                  remove: Optional[List[str]] = None) -> list:
        if keep is not None or remove is not None:
            return self._select(out, items, keep, remove)
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

    #************#
    #   select   #
    #************#
    def _select(self, out: torch.Tensor, items, keep: Optional[List[str]],
                remove: Optional[List[str]]) -> list:
        """Each spectrum with components removed, or only the kept ones, as the input was."""
        if keep is not None and remove is not None:
            raise ValueError("WAND keeps or removes components, not both")
        named = [n for given in (keep if keep is not None else remove)
                 for n in (self.metabolites if given == "metabolites" else [given])]
        unknown = sorted(set(named) - set(self.components))
        if unknown:
            raise ValueError(f"WAND has no component {unknown}; it has {self.components} "
                             f"and 'metabolites'")

        chosen = [i for i, c in enumerate(self.components) if c in named]
        window = out[..., chosen].sum(-1).cpu().double().numpy()      # (batch, 2, window)
        results = []
        for spec, item in zip(window, items):
            full = np.zeros(item.shape[3], complex)
            full[self.first:self.last] = spec[0] + 1j * spec[1]
            fid = np.fft.ifft(full)
            if remove is not None:
                fid = np.asarray(item[:]).reshape(-1) - fid
            results.append(gen_nifti_mrs(fid.reshape(item.shape), item.dwelltime,
                                         item.spectrometer_frequency[0],
                                         nucleus=item.nucleus[0], dim_tags=item.dim_tags))
        return results

    #************#
    #   target   #
    #************#
    def target(self, truth, data, objective: str) -> np.ndarray:
        """The components over the window, centred as forward centres its input.

        Each metabolite and the baseline come from the truth; the artifacts are the rest
        of the data - noise, water, lipids, whatever the source added - which is how the
        original defines them.
        """
        missing = [c for c in self.components[:-1] if c not in truth.components]
        if missing:
            raise KeyError(f"WAND learns each component; the source gives no {missing}")
        parts = [np.asarray(truth.components[c]) for c in self.components[:-1]]
        parts.append(data.numpy().reshape(len(data), -1) - sum(parts))
        specs = np.fft.fft(np.stack(parts, axis=-1), axis=1)[:, self.first:self.last]
        specs = specs - specs.mean(axis=1, keepdims=True)
        return np.stack([specs.real, specs.imag], axis=1).astype(np.float32)

    #**********#
    #   loss   #
    #**********#
    def loss(self, target, out, objective: str):
        """The mean squared error over every component, as the original's 'mse'."""
        from keras import ops
        return ops.mean(ops.square(out - target))

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, str]:
        return {**super().runtime(), "wand-for-mrs": git_version(str(self.repo))}
