####################################################################################################
#                                         run_pipeline.py                                          #
####################################################################################################
#                                                                                                  #
# Authors:  J. P. Merkofer (j.p.merkofer@tue.nl)                                                   #
#                                                                                                  #
# Created:  20/07/25                                                                               #
#                                                                                                  #
# Purpose: Testing the pipline functionality. The script loads data, initializes the pipeline      #
#          with desired modules, and runs the pipeline on the data.                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os

from fsl_mrs.utils import mrs_io

# own
from core.nifti_mrs_plus import NIfTI_MRS_Plus
from core.transform import Transform
from core.pipeline import Pipeline
from utils.auxiliary import load_ISMRM2016_challenge_concs



#**********#
#   main   #
#**********#
if __name__ == "__main__":
    print("Running pipeline test...")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # # uncomment the following lines to set up the configuration for the ISMRM 2016 fitting challenge data.
    # config = {
    #     'path2basis': os.path.join(project_root, 'data', 'ISMRM2016_fitting_challenge', 'press3T_30ms.BASIS'),
    #     'path2concs': os.path.join(project_root, 'data', 'ISMRM2016_fitting_challenge', 'ground_truth'),
    #     'path2data': os.path.join(project_root, 'data', 'ISMRM2016_fitting_challenge', 'datasets_JMRUI_WS'),
    #     'path2water': os.path.join(project_root, 'data', 'ISMRM2016_fitting_challenge', 'datasets_JMRUI_nWS'),
    #
    #     'test_size': 10,  # number of test files to load
    # }
    #
    # # load ground truth
    # files = os.listdir(config['path2concs'])[:config['test_size']]
    # concs = [load_ISMRM2016_challenge_concs(os.path.join(config['path2concs'], file))
    #          for file in files]   # load excel
    #
    # # load data
    # files = os.listdir(config['path2data'])[:config['test_size']]
    # data_list = [mrs_io.read_FID(os.path.join(config['path2data'], file)) for file in files]
    #
    # # create NIfTI_MRS_Plus object
    # data = NIfTI_MRS_Plus.from_nifti_list(data_list)
    #
    # # water reference (optional)
    # if 'path2water' in config:
    #     files = os.listdir(config['path2water'])[:config['test_size']]
    #     water_list = [mrs_io.read_FID(os.path.join(config['path2water'], file)) for file in files]
    #     water = NIfTI_MRS_Plus.from_nifti_list(water_list)
    # else:
    #     water = None

    # uncomment the following lines to set up the configuration for the OpenNeuro data.
    config = {
        'path2data': os.path.join(project_root, 'data', 'openneuro_ds005371'),
        'selection': 'herculespress',  # select data based on a sequence
        'test_size': 3,  # number of test files to load
    }

    # load data
    files = [sub for sub in os.listdir(config['path2data']) if 'sub' in sub]
    data_list = [
        mrs_io.read_FID(os.path.join(config['path2data'], sub, ses, 'mrs', file))
        for sub in files
        for ses in os.listdir(os.path.join(config['path2data'], sub))
        for file in os.listdir(os.path.join(config['path2data'], sub, ses, 'mrs'))
        if file.endswith('.nii.gz') and config['selection'] in file and 'svs' in file
    ]
    data = NIfTI_MRS_Plus.from_nifti_list(data_list[:config['test_size']])

    # load water reference (optional)
    water_list = [
        mrs_io.read_FID(os.path.join(config['path2data'], sub, ses, 'mrs', file))
        for sub in files
        for ses in os.listdir(os.path.join(config['path2data'], sub))
        for file in os.listdir(os.path.join(config['path2data'], sub, ses, 'mrs'))
        if file.endswith('.nii.gz') and config['selection'] in file and 'svs' in file
    ]
    water = NIfTI_MRS_Plus.from_nifti_list(water_list[:config['test_size']])

    from processing.fsl_mrs import FSL_CoilCombine

    # create pipeline
    pipeline = Pipeline(modules=[
        FSL_CoilCombine()
    ])

    # run pipeline
    processed_data, processed_water = pipeline.run(data, water)

    print(processed_data.dim_tags)
    print(processed_data[0])