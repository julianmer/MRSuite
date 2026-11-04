####################################################################################################
#                                              mrsi.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-04                                                                              #
#                                                                                                  #
# Purpose: Spectroscopic imaging with the same blocks: one slice of the 2024 ISMRM MRSI challenge  #
#          (Sub1, z = 16), a 4-coil array synthesised onto it and its k-space undersampled twofold #
#          by Augmentrum, coils combined and phases corrected by FSL-MRS, the voxels in the brain  #
#          mask taken apart and fitted by LCModel one by one and by FSL-MRS's whole-volume         #
#          fsl_mrsi, each put back on the grid as maps over the challenge's true maps.             #
#                                                                                                  #
#          Needs LCModel (fetched on first use) and FSL-MRS; the challenge data (Zenodo) is        #
#          fetched on first use. --n takes the first n voxels of the mask for a quick look.        #
#                                                                                                  #
#          python examples/mrsi.py --n 50 --out out/mrsi                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from mrsuite import io, mrsi
from mrsuite.bridges import augmentrum, fsl_mrs, lcmodel
from mrsuite.core.pipeline import Pipeline
from mrsuite.datasets import mrsi_challenge


#**********#
#   main   #
#**********#
def main(n: Optional[int], out: Path, workers: int) -> None:
    """The slice through augmentation, processing and two fits; the maps into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    spectra, truth, mask = mrsi_challenge.volume("Sub1", signal="all-nuisance", z=16)
    basis = io.basis(mrsi_challenge.basis(str(out / "mrsi_challenge_2024.BASIS")))

    augmented = Pipeline([augmentrum.step.coil_sampling(mode="synthesize", n_coils=4, seed=0),
                          augmentrum.step.undersampling(acceleration_factor=2.0, us_seed=0)]
                         ).run(spectra)
    processed = Pipeline([fsl_mrs.step.coilcombine(),
                          fsl_mrs.step.mrsi_phase_corr()]).run(augmented)

    # voxel by voxel: the mask's voxels as single spectra, LCModel on each
    voxels, index = mrsi.voxels(processed, mask)
    if n is not None:                                            # the first n, for both fits
        voxels, index = voxels[:n], index[:n]
        mask = np.zeros_like(mask)
        mask[tuple(index.T)] = True
    by_voxel = Pipeline([lcmodel.step.fit(basis=basis)]).run(voxels)
    # the whole volume at once: FSL-MRS's own fsl_mrsi
    whole, whole_index = fsl_mrs.fit_mrsi(processed, basis, mask, workers=workers)

    for name, fits, where in (("lcmodel", by_voxel, index), ("fsl_mrsi", whole, whole_index)):
        figure = mrsi.maps(fits, where, truth)
        figure.savefig(out / f"maps_{name}.png", dpi=150)
    print(f"{len(by_voxel)} voxels by LCModel, {len(whole)} by fsl_mrsi; maps in {out}")


#**********#
#   args   #
#**********#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="voxels to fit (default: all)")
    parser.add_argument("--out", type=Path, default=Path("out/mrsi"))
    parser.add_argument("--workers", type=int, default=4, help="processes for fsl_mrsi")
    args = parser.parse_args()
    main(args.n, args.out, args.workers)
