####################################################################################################
#                                     digital_phantom_driver.py                                    #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: One voxel of MRS-Digital-Phantom, in a process of its own with its checkout as the      #
#          working folder, as its code reads ./data and imports its own top-level packages.        #
#          Usage: python digital_phantom_driver.py job.json; writes the job's result .npz.         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import os
import sys

import numpy as np


#**********#
#   main   #
#**********#
def main(path: str) -> None:
    """Build the phantom the job names, simulate its voxel, and save what MRSuite reads."""
    with open(path) as fh:
        job = json.load(fh)
    sys.path.insert(0, os.getcwd())
    import torch
    from simulation.mrs_phantom import DigitalPhantom

    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    phantom = DigitalPhantom(skeleton=job["skeleton"], path2metabs=job["path2metabs"],
                             path2basis=job["path2basis"], simulation_params=job["params"],
                             gui=None, metabs=[])

    # a cube of edge `size` mm at `centre`, the brain's centre of mass unless given
    labels = phantom.get_label_data().numpy()
    centre = job["centre"] or np.argwhere(np.isin(labels, [1, 2, 3])).mean(0).round().astype(int)
    half = [int(round(job["size"] / 2 / s)) for s in phantom.spacing]
    coords = tuple(int(v) for c, h in zip(centre, half) for v in (c - h, c + h))
    spec, _, _, _, _ = phantom.simulate_data(coords)

    basis = phantom.basis_set
    truth = phantom.signal_model.final_voxel_concentrations
    np.savez(job["result"], spectrum=np.asarray(spec), dwell=basis.dwelltime, frequency=basis.cf,
             te=basis.TE, coords=np.array(coords), names=np.array(list(truth)),
             concentrations=np.array([float(v) for v in truth.values()]),
             fractions=json.dumps({k: float(v) for k, v in
                                   phantom.signal_model.voxel_tissue_fractions.items()}))


if __name__ == "__main__":
    main(sys.argv[1])
