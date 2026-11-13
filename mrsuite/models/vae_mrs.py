####################################################################################################
#                                            vae_mrs.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-12                                                                              #
#                                                                                                  #
# Purpose: VAE-MRS, the variational autoencoder that synthesises MEGA-PRESS transients (van de     #
#          Sande et al. 2026), as a pipeline step. The network and its sampling are the            #
#          original's own, loaded from its repository; what is here is the way in from NIfTI-MRS   #
#          and back.                                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite import externals
from mrsuite.core.method import keep_header
from mrsuite.core.provenance import portable_file
from mrsuite.models.base import classpath, load_file
from mrsuite.models.pytorch import TorchModel

# the settings of the original's config.yaml its network is built from
SETTINGS = ("num_points", "latent_dim", "data_type", "use_conditioning", "condition_dim",
            "film_enabled_layers")

# the edit conditions the network was trained with, as the original orders them
CONDITIONS = ("OFF", "ON")

METHODS = ("reconstruction", "random_sampling", "interpolation", "random_interpolation")


#**************************************************************************************************#
#                                           Class VAEMRS                                           #
#**************************************************************************************************#
#                                                                                                  #
# Synthesises MEGA-PRESS transients from the measured ones of a scan.                              #
#                                                                                                  #
#**************************************************************************************************#
class VAEMRS(TorchModel, paper_url="https://arxiv.org/abs/2603.00736"):
    """Synthesises MEGA-PRESS transients from the measured ones of a scan.

    Each transient's spectrum is encoded to a latent vector under its edit condition; new
    vectors are drawn around or between those of the transients given, and decoded to new
    transients. This is the test step of the original's Framework, with its network
    (ConvNeXtVAE) and its sampling functions as published:

        vae = VAEMRS.published()
        more = vae(scan, method="random_sampling", samples=9)

    A scan is one NIfTI-MRS with its transients along DIM_DYN and the two edit conditions
    along DIM_EDIT, the first OFF and the second ON, preprocessed as the training data was:
    aligned, eddy-current corrected, creatine at 3.027 ppm, phased. It comes back the same
    way, with the synthetic transients along DIM_DYN and on the scan's own scale (the
    network works on spectra scaled to a maximum of one; the original's export leaves them
    so).

    The methods are the original's:

        reconstruction        every transient decoded from its posterior mean
        random_sampling       ``samples`` vectors per transient, with noise around its own
        interpolation         ``samples`` vectors per transient, on the line through them
        random_interpolation  the interpolated ones with that noise added

    Args:
        config (Dict[str, Any]): The settings of the original's config.yaml its network
            reads: num_points, latent_dim, data_type, use_conditioning, condition_dim and
            film_enabled_layers.
        noise_std (List[float]): The noise per latent dimension, as the original
            calibrated it.
        acquisition (Dict[str, float]): What it was trained on; see Model.

    Attributes:
        repo (Path): The original repository, which the network is loaded from.
    """

    external = "vae-mrs"
    repo = externals.path(external)

    #**********#
    #   init   #
    #**********#
    def __init__(self, config: Dict[str, Any], noise_std: List[float],
                 acquisition: Dict[str, float]) -> None:
        super().__init__()
        self.config, self.noise_std = dict(config), list(noise_std)
        self.acquisition = dict(acquisition)
        self.repo = externals.ensure(self.external)
        self.net = load_file(self.repo / "modules" / "models.py").ConvNeXtVAE(self.config)

    #***************#
    #   published   #
    #***************#
    @classmethod
    def published(cls) -> "VAEMRS":
        """The paper's model: the checkpoint the original repository ships, with the
        settings of its config.yaml, which its README states the checkpoint was trained
        with. The weights are loaded strictly and the recipe says where they came from."""
        import yaml

        repo = externals.ensure(cls.external)
        with open(repo / "configs" / "config.yaml") as fh:
            settings = yaml.safe_load(fh)
        # acquired as its export writes the scans' headers (analysis.save_spectra_nifti):
        # a dwell time of 0.5 ms at 127.74 MHz
        model = cls(config={key: settings[key] for key in SETTINGS},
                    noise_std=settings["noise_std"],
                    acquisition={"points": settings["num_points"], "bandwidth": 2000.0,
                                 "frequency": 127.74})
        checkpoint = repo / "checkpoints" / "best-model.ckpt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        model.net.load_state_dict({k[len("model."):]: v for k, v in state.items()
                                   if k.startswith("model.")})
        model.recipe = {"training": {
            "model": classpath(cls), "checkpoint": portable_file(checkpoint),
            "converted with": {cls.external: externals.version(cls.external)}}}
        return model

    #***********#
    #   check   #
    #***********#
    def check(self, item) -> None:
        """Refuse a scan without transients and two edit conditions, or acquired unlike
        the training data."""
        tags = list(item.dim_tags)
        if "DIM_DYN" not in tags or "DIM_EDIT" not in tags \
                or item.shape[4 + tags.index("DIM_EDIT")] != len(CONDITIONS):
            raise ValueError(f"VAEMRS takes transients along DIM_DYN with OFF and ON along "
                             f"DIM_EDIT, not shape {item.shape} with {tags}")
        self._compare({"points": item.shape[3], "bandwidth": 1 / item.dwelltime,
                       "frequency": item.spectrometer_frequency[0]})

    #*********#
    #   run   #
    #*********#
    def run(self, data, method: str = "random_sampling", samples: int = 3,
            noise_std: Optional[Sequence[float]] = None, kind: str = "linear",
            originals: bool = False, seed: Optional[int] = None) -> list:
        """
        Each scan's synthetic transients.

        Args:
            data: The scans.
            method (str): One of METHODS.
            samples (int): Latent vectors drawn per transient given.
            noise_std (Optional[Sequence[float]]): The noise per latent dimension, or one
                value for all. Defaults to the model's own.
            kind (str): The interpolation between the transients' vectors, as
                scipy.interpolate.interp1d names it.
            originals (bool): Put the transients given in front of the synthetic ones.
            seed (Optional[int]): Seeds torch first, for the same draw again.

        Returns:
            list: Per scan, a NIfTI-MRS of its synthetic transients.
        """
        if method not in METHODS:
            raise ValueError(f"VAEMRS has no method '{method}'; it has {list(METHODS)}")
        items = data.list()
        for item in items:
            self.check(item)
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.as_tensor(self.noise_std if noise_std is None else noise_std,
                                dtype=torch.float32)
        return [self._scan(item, method, samples, noise, kind, originals) for item in items]

    #**********#
    #   scan   #
    #**********#
    def _scan(self, item, method: str, samples: int, noise: torch.Tensor, kind: str,
              originals: bool):
        """One scan through the network, condition by condition."""
        tags = list(item.dim_tags)
        # the samples as stored, which the original reads; indexing the object conjugates them
        fids = np.moveaxis(np.asarray(item.image[:]).reshape(item.shape[3:]),
                           (1 + tags.index("DIM_DYN"), 1 + tags.index("DIM_EDIT")), (1, 2))
        fids = fids.reshape(fids.shape[:3])                    # (points, transients, edit)
        specs = np.fft.fftshift(np.fft.fft(fids, axis=0, norm="ortho"), axes=0)
        scale = np.abs(specs).max() or 1.0
        specs = specs / scale

        device = next(self.parameters()).device
        mode = self.training
        self.eval()
        try:
            with torch.no_grad():
                out = [self._draw(torch.from_numpy(
                           np.stack([specs[:, :, k].T.real, specs[:, :, k].T.imag], axis=-1)
                           .astype(np.float32)).to(device), condition, method, samples,
                           noise.to(device), kind, originals)
                       for k, condition in enumerate(CONDITIONS)]
        finally:
            self.train(mode)

        out = torch.stack(out, dim=-1).cpu().double().numpy()  # (transients, points, 2, edit)
        specs = (out[:, :, 0] + 1j * out[:, :, 1]).transpose(1, 0, 2) * scale
        fids = np.fft.ifft(np.fft.ifftshift(specs, axes=0), axis=0, norm="ortho")
        result = gen_nifti_mrs(fids.reshape(1, 1, 1, *fids.shape), item.dwelltime,
                               item.spectrometer_frequency[0], nucleus=item.nucleus[0],
                               dim_tags=["DIM_DYN", "DIM_EDIT", None], no_conj=True)
        keep_header(item, result)
        return result

    #**********#
    #   draw   #
    #**********#
    def _draw(self, x: torch.Tensor, condition: str, method: str, samples: int,
              noise: torch.Tensor, kind: str, originals: bool) -> torch.Tensor:
        """The synthetic spectra of one condition, (transients, points, 2), by the
        original's own encode, sampling and decode."""
        mu, logvar = self.net.encode(x, [condition] * len(x))
        z = mu if method == "reconstruction" else self.net.reparameterize(mu, logvar)
        if method == "random_sampling":
            z = self.net.random_sampling(z, samples, noise_std=noise)
        elif method == "interpolation":
            z = self.net.interpolate(z, samples, interpolation_kind=kind)
        elif method == "random_interpolation":
            z = self.net.interpolate_with_random_noise(z, samples, noise_std=noise,
                                                       interpolation_kind=kind)
        drawn = self.net.decode(z, [condition] * len(z))
        return torch.cat([x, drawn], dim=0) if originals else drawn

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, str]:
        return {**super().runtime(), "vae-mrs": externals.version(self.external)}
