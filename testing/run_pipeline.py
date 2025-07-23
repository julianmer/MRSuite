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
from utils.auxiliary import load_ISMRM2016_challenge_concentrations



#**********#
#   main   #
#**********#
if __name__ == "__main__":
    print("Running pipeline test...")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    config = {
        'path2basis': os.path.join(project_root, 'data', 'ISMRM2016_fitting_challenge', 'press3T_30ms.BASIS'),
        'path2concs': os.path.join(project_root, 'data', 'ISMRM2016_fitting_challenge', 'ground_truth'),
        'path2data': os.path.join(project_root, 'data', 'ISMRM2016_fitting_challenge', 'datasets_JMRUI_WS'),
        'path2water': os.path.join(project_root, 'data', 'ISMRM2016_fitting_challenge', 'water_reference'),

        'test_size': 10,  # number of test files to load
    }

    # load ground truth
    files = os.listdir(config['path2concs'])[:config['test_size']]
    concs = [load_ISMRM2016_challenge_concentrations(os.path.join(config['path2concs'], file))
             for file in files]   # load excel

    # load data
    files = os.listdir(config['path2data'])[:config['test_size']]
    data = [mrs_io.read_FID(os.path.join(config['path2data'], file)) for file in files]

    # water reference (optional)
    if 'path2water' in config:
        files = os.listdir(config['path2water'])[:config['test_size']]
        water = [mrs_io.read_FID(config['path2water'] + file) for file in files]
    else:
        water = None

    # create NIfTI_MRS_Plus object
    nifti_mrs_plus = NIfTI_MRS_Plus.from_nifti_list(data)   # TODO: separate object for water or not?

    print(nifti_mrs_plus.shape())
    print(nifti_mrs_plus.to_tensor().shape)
    print(nifti_mrs_plus.get_dim_tags())

    # create pipeline
    pipeline = Pipeline(modules=[
        Transform(),  # replace with actual transformation modules
        Transform()   # add more transformations as needed
    ])

    # run pipeline
    processed_data = pipeline.run(nifti_mrs_plus)