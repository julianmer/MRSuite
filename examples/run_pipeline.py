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
import re
from typing import Tuple

# third-party
from fsl_mrs.utils import mrs_io

# own
from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core.pipeline import Pipeline
from mrsuite.transforms.processing import fid_a, fsl_mrs


#*************************************************#
#   load fitting challenge data from ISMRM 2016   #
#*************************************************#
def load_ISMRM2016_challenge_data(config: dict) -> Tuple[NIfTI_MRS_Plus, NIfTI_MRS_Plus, list]:
    """
    Load MRS data from the ISMRM 2016 fitting challenge.

    Args:
        config (dict): Configuration dictionary containing paths and selection criteria.

    Returns:
        data (NIfTI_MRS_Plus): Loaded MRS data.
        water (NIfTI_MRS_Plus): Loaded water reference data.
        concs (dict): Set of ground truth concentrations for metabolites.
    """
    import pandas as pd

    def load_ISMRM2016_challenge_concs(path2conc: str) -> dict:
        truth = {'Ace': 0.0}  # initialize, Ace is only partially present

        file = pd.read_excel(path2conc, header=17)
        for i, met in enumerate(file['Metabolites']):
            if not isinstance(met, str): break
            truth[met] = file['concentration'][i]

        truth['Mac'] = truth.pop('MMBL')  # rename key MMBL to Mac
        return dict(sorted(truth.items()))

    def listing(path, suffix=None):
        """Sorted, suffix-filtered listing of a data directory.

        The challenge repository keeps water-suppressed and unsuppressed spectra in one
        folder (`dataset1_WS.txt` alongside `dataset1_nWS.txt`), so the two sets have to
        be separated by suffix. Sorting is by the dataset number rather than by name:
        spectra, water references and ground truth are paired positionally, and both
        `os.listdir` order and plain lexicographic order would misalign them
        (`dataset10` sorts before `dataset2`).
        """
        names = [f for f in os.listdir(path) if not f.startswith('.')]
        if suffix:
            names = [f for f in names if f.endswith(suffix)]
        return sorted(names, key=lambda f: (int(re.search(r'\d+', f).group())
                                            if re.search(r'\d+', f) else 0, f))

    files = listing(config['path2concs'])[:config['test_size']]
    concs = [load_ISMRM2016_challenge_concs(os.path.join(config['path2concs'], f)) for f in files]

    files = listing(config['path2data'], config.get('data_suffix'))[:config['test_size']]
    data_list = [mrs_io.read_FID(os.path.join(config['path2data'], f)) for f in files]
    data = NIfTI_MRS_Plus(data_list)

    if config['path2water']:
        files = listing(config['path2water'], config.get('water_suffix'))[:config['test_size']]
        water_list = [mrs_io.read_FID(os.path.join(config['path2water'], f)) for f in files]
        water = NIfTI_MRS_Plus(water_list)
    else:
        water = None

    return data, water, concs


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
        # spectra and basis come from the challenge submodule; only the ground truth
        # is kept in-repo, as that repository does not carry it
        challenge = os.path.join(project_root, 'data', '2016_fitting_challenge')
        config.update({
            'path2basis': os.path.join(challenge, 'basisset_LCModel', 'press3T_30ms.BASIS'),
            'path2concs': os.path.join(project_root, 'data', '2016_fitting_challenge_gts'),
            'path2data': os.path.join(challenge, 'datasets_JMRUI'),
            'path2water': os.path.join(challenge, 'datasets_JMRUI'),
            'data_suffix': '_WS.txt',
            'water_suffix': '_nWS.txt',
        })
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
        data, water, _ = load_ISMRM2016_challenge_data(config)
    elif config['dataset'] == 'openneuro':
        data, water, _ = load_openneuro_data(config)
    else:
        raise ValueError("Unsupported dataset. Choose 'ismrm' or 'openneuro'.")

    pipeline = Pipeline(modules=[
        fsl_mrs.FSL_CoilCombine(verbose=True),
        # fsl_mrs.FSL_Align(verbose=True),
        # fsl_mrs.FSL_Average(verbose=True),
        fid_a.FIDA_AlignAverages_fd(verbose=True),
        # fid_a.FIDA_EddyCorrect(),

    ])

    processed_data, processed_water = pipeline.run(data, water)

    print("Processing complete.")
    print(processed_data.dim_tags)
    print(processed_data.shape)
    print(processed_water.shape)
    print(processed_data[0].header)
    print(processed_data[0].hdr_ext)


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