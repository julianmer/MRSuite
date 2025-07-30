####################################################################################################
#                                           pipeline.py                                            #
####################################################################################################
#                                                                                                  #
# Authors:  J. P. Merkofer (j.p.merkofer@tue.nl)                                                   #
#                                                                                                  #
# Created:  20/07/25                                                                               #
#                                                                                                  #
# Purpose: Manages the execution of a sequence of transformations on NIfTI_MRS+ objects,           #
#          allowing for batch processing and analysis of MRS data in a streamlined manner.         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import List, Optional

# own
from core.nifti_mrs_plus import NIfTI_MRS_Plus
from core.transform import Transform


#**************************************************************************************************#
#                                          Class Pipeline                                          #
#**************************************************************************************************#
#                                                                                                  #
# Accepts a list of modules (that inherit from Interface), runs them in order on input data, and   #
# handles chaining of transformations.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class Pipeline:

    def __init__(self, modules: Optional[List[Transform]] = None) -> None:
        """
        Initialize the Pipeline with a list of modules.

        Parameters:
            modules (list): List of modules to be executed in the pipeline.
        """
        self.modules = modules if modules is not None else []

    def run(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus] = None) \
            -> (NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]):
        """
        Run the pipeline on the provided data.

        Parameters:
            data: Input data to be processed by the pipeline.
            water: Optional water reference data, if required by the modules.

        Returns:
            Processed data after passing through all modules.
        """
        for module in self.modules:
            if hasattr(module, 'forward'):
                data, water = module.forward(data, water=water)
            else:
                raise AttributeError(f"Module {module.__class__.__name__} does not have a "
                                     f"'forward' method.")
        return data, water