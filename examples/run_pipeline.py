####################################################################################################
#                                         run_pipeline.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-07-25                                                                              #
#                                                                                                  #
# Purpose: Testing the pipline functionality. The script loads data, initializes the pipeline      #
#          with desired modules, and runs the pipeline on the data.                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import os
from typing import Tuple

# third-party
from fsl_mrs.utils import mrs_io

# own
from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core.pipeline import Pipeline, results
from mrsuite import fid_a, fsl_mrs
from mrsuite.datasets import challenge2016


#***********************************************#
#   load data from OpenNeuro dataset ds005371   #
#***********************************************#
def load_openneuro_data(config = dict) -> Tuple[NIfTI_MRS_Plus, NIfTI_MRS_Plus, None]:
    """
    Load MRS data from the OpenNeuro dataset ds005371.

    Args:
        config (dict): Configuration dictionary containing paths and selection criteria.

    Returns:
        data (NIfTI_MRS_Plus): Loaded MRS data.
        water (NIfTI_MRS_Plus): Loaded water reference data.
        None: Placeholder for additional data if needed.
    """
    path2data = config['path2data']
    selection = config['selection']

    files = [sub for sub in os.listdir(path2data) if 'sub' in sub]
    data_list = [
        mrs_io.read_FID(os.path.join(path2data, sub, ses, 'mrs', file))
        for sub in files
        for ses in os.listdir(os.path.join(path2data, sub))
        for file in os.listdir(os.path.join(path2data, sub, ses, 'mrs'))
        if file.endswith('.nii.gz') and selection in file and 'svs' in file
    ]
    data = NIfTI_MRS_Plus(data_list[:config['test_size']])

    water_list = [
        mrs_io.read_FID(os.path.join(path2data, sub, ses, 'mrs', file))
        for sub in files
        for ses in os.listdir(os.path.join(path2data, sub))
        for file in os.listdir(os.path.join(path2data, sub, ses, 'mrs'))
        if file.endswith('.nii.gz') and selection in file and 'svs' in file
    ]
    water = NIfTI_MRS_Plus(water_list[:config['test_size']])

    return data, water, None



#***************************************************#
#   parse command line arguments and build config   #
#***************************************************#
def parse_args() -> dict:
    """
    Parse command line arguments and build a configuration dictionary.

    Returns:
        config (dict): Configuration dictionary with dataset, paths, and other parameters.
    """
    parser = argparse.ArgumentParser(description="Run MRS pipeline on selected dataset.")
    parser.add_argument('--dataset', choices=['ismrm', 'openneuro'], required=True,
                        help="Which dataset to use: 'ismrm' or 'openneuro'")
    parser.add_argument('--test_size', type=int, default=3,
                        help="Number of test samples to load")
    parser.add_argument('--selection', type=str, default='herculespress',
                        choices=['herculespress', 'herculesslaser', 'hermespress', 'hermesslaser'],
                        help="Substring to select OpenNeuro MRS files "
                             "(only used if dataset is 'openneuro')")

    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    config = {
        'dataset': args.dataset,
        'test_size': args.test_size,
        'selection': args.selection,
        'project_root': project_root,
    }

    if args.dataset == 'ismrm':
        config.update({'path2basis': str(challenge2016.BASIS)})
    elif args.dataset == 'openneuro':
        config.update({
            'path2data': os.path.join(project_root, 'data/openneuro_ds005371'),
            'path2water': None  # not used explicitly
        })

    return config


#**********#
#   main   #
#**********#
def main():
    """
    Main function to run the MRS pipeline on the selected dataset.

    Parses command line arguments, loads the dataset, initializes the pipeline,
    and runs the processing pipeline.
    """
    config = parse_args()

    print(f"Running pipeline on dataset: {config['dataset']}")
    if config['dataset'] == 'ismrm':
        data, water, _ = challenge2016.load(config['test_size'])
    elif config['dataset'] == 'openneuro':
        data, water, _ = load_openneuro_data(config)
    else:
        raise ValueError("Unsupported dataset. Choose 'ismrm' or 'openneuro'.")

    # A reference is just data: it goes through a pipeline of its own, and whatever
    # needs it later closes over the processed one.
    if water is not None:
        water = Pipeline([fsl_mrs.step.coilcombine()]).run(water)

    # Any callable is a step. `bridge.step.f(args)` binds the arguments and takes the
    # data later; your own function goes in as itself. FID-A and FSL-MRS interleave.
    data = Pipeline([
        fsl_mrs.step.coilcombine(),
        # fsl_mrs.step.align(dim='DIM_DYN'),
        # fid_a.step.op_rmbadaverages(4, 't'),
        # fsl_mrs.step.average(dim='DIM_DYN'),
        # fsl_mrs.step.ecc(water),
    ]).run(data)

    print("Processing complete.")
    print(data.dim_tags)
    print(data.shape)
    # anything a step returned that was not spectra: metrics, indices
    print(sorted(results(data)), sorted(results(data, index=0)))

    # analysis comes after, and returns its results rather than hiding them in the data
    # results = fsl_mrs.fit(data, mrs_io.read_basis(config['path2basis']))


#***************************#
#   run from command line   #
#***************************#
if __name__ == "__main__":

    # This script can be run directly from the command line.
    # It will parse command line arguments to determine the dataset and test size.
    # Example usage(s):
    # python -m examples.run_pipeline --dataset ismrm --test_size 10
    # python -m examples.run_pipeline --dataset openneuro --test_size 10 --selection herculespress

    main()