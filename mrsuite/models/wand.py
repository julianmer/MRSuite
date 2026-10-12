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
from typing import Dict, List, Optional

import numpy as np
import torch

from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite import externals
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
class WAND(TorchModel, paper_url="https://doi.org/10.1002/nbm.70038"):
    """Decomposes a spectrum into its metabolites, baseline and artifacts in the wavelet domain.

    The spectrum's window from ``first`` to ``last`` is taken to scalograms of its real
    and imaginary parts, the U-Net predicts a mask per component, and each masked
    scalogram is transformed back. This is FrameworkWAND.prepareData and .forward of
    the original, with its fixed settings: Morlet wavelet (mu 14), 32 voices, softmax
    masks, and the artifacts as what the other masks leave.

    Each spectrum comes back as a NIfTI-MRS with the components along a fifth dimension,
    named in its header - the metabolites, any nuisance, then 'baseline', then
    'artifacts' - each extended to the full length by repeating its edge values, as the
    original's scripts do.

    As a processing step it hands back an ordinary spectrum of the input's shape instead:

        wand.step(remove=["artifacts"])             # the data without them
        wand.step(remove=["baseline", "artifacts"])
        wand.step(keep=["NAA", "Cr", "PCr"])        # only these; "metabolites" for all

    ``remove`` subtracts the components from the data, so outside the network's window
    the data is untouched; ``keep`` adds the components up, zero outside the window.

    It learns the components from any source that gives each metabolite's signal, the
    baseline and each nuisance it names: the artifacts are what the data holds beyond them.
    A nuisance is a signal to take out on its own, such as water and lipids, which
    'metabolites' in ``keep`` does not include.

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
        nuisance (List[str]): Further components, between the metabolites and the baseline,
            e.g. ['nuisance'] for the MRSI challenge's water and lipids. Empty by default.
        baseline (bool): Learn the baseline as a component of its own; without, it is part
            of the artifacts, as for training data that gives no baseline apart.
        zero_fill (int): Zero-fill each FID to this many times its points before the
            transform, ``first`` and ``last`` counting on that grid; the components come
            back on the data's own points. 1 by default: as the original, which on short
            spectra (the MRSI challenge's 384 points) gives back ~1.7 times its input.
        weights (Optional[Dict[str, float]]): Weights of components in the loss, by name,
            1 for the others: e.g. a nuisance far larger than the metabolites weighed down
            to their size.

    Attributes:
        repo (Path): The original repository, which the network and transform are loaded from.
    """

    external = "wand-for-mrs"
    repo = externals.path(external)
    objectives = ("components",)

    #**********#
    #   init   #
    #**********#
    def __init__(self, metabolites: List[str], first: int, last: int,
                 acquisition: Dict[str, float], batch_statistics: bool = True,
                 nuisance: List[str] = (), baseline: bool = True, zero_fill: int = 1,
                 weights: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        self.metabolites, self.first, self.last = list(metabolites), first, last
        self.acquisition = dict(acquisition)
        self.batch_statistics = batch_statistics
        self.nuisance, self.baseline, self.zero_fill = list(nuisance), baseline, zero_fill
        self.components = (self.metabolites + self.nuisance + ["baseline"] * baseline
                           + ["artifacts"])
        self.weights = dict(weights or {})
        unknown = sorted(set(self.weights) - set(self.components))
        if unknown:
            raise ValueError(f"WAND has no component {unknown} to weigh; it has {self.components}")
        self.repo = externals.ensure(self.external)
        self.cwt = load_file(self.repo / "utils" / "wavelets.py").CWT(
            last - first, wavelet=("morlet", {"mu": 14}), scales="log-piecewise", nv=32,
            padtype=None)
        self.net = load_file(self.repo / "frameworks" / "nnModels.py").UNet(
            n_channels=2, n_classes=2 * (len(self.components) - 1))

    #*************#
    #   prepare   #
    #*************#
    def prepare(self, items) -> torch.Tensor:
        return torch.from_numpy(spectra(items, self.zero_fill))

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
        masks = masks.view(masks.shape[0], len(self.components) - 1, 2, *masks.shape[-2:])
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
        full = torch.nn.functional.pad(out, (0, 0, self.first, n * self.zero_fill - self.last),
                                       mode="replicate")
        full = full.cpu().double().numpy()
        fids = np.fft.ifft(full[:, 0] + 1j * full[:, 1], axis=1)[:, :n]  # (batch, points, comps)
        results = []
        for fid, item in zip(fids, items):
            result = gen_nifti_mrs(fid.reshape(1, 1, 1, *fid.shape), item.dwelltime,
                                   item.spectrometer_frequency[0], nucleus=item.nucleus[0],
                                   dim_tags=["DIM_USER_0", None, None])
            result.set_dim_tag("DIM_USER_0", "DIM_USER_0", info="WAND components", header={
                "Component": {"Value": self.components,
                              "Description": "The metabolites, any nuisance, then baseline "
                                             "and artifacts"}})
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
            full = np.zeros(item.shape[3] * self.zero_fill, complex)
            full[self.first:self.last] = spec[0] + 1j * spec[1]
            fid = np.fft.ifft(full)[:item.shape[3]]
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
        stacked = np.stack(parts, axis=-1)
        specs = np.fft.fft(stacked, n=stacked.shape[1] * self.zero_fill,
                           axis=1)[:, self.first:self.last]
        specs = specs - specs.mean(axis=1, keepdims=True)
        return np.stack([specs.real, specs.imag], axis=1).astype(np.float32)

    #**********#
    #   loss   #
    #**********#
    def loss(self, target, out, objective: str):
        """The mean squared error over every component, as the original's 'mse', each
        component's errors times its weight where ``weights`` gives one."""
        from keras import ops
        error = ops.square(out - target)
        if self.weights:
            error = error * ops.convert_to_tensor(
                [self.weights.get(c, 1.0) for c in self.components], dtype=error.dtype)
        return ops.mean(error)

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, str]:
        return {**super().runtime(), "wand-for-mrs": git_version(str(self.repo))}
