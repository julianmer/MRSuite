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



#**************************************************************************************************#
#                                          Class Pipeline                                          #
#**************************************************************************************************#
#                                                                                                  #
# Accepts a list of modules (that inherit from Interface), runs them in order on input data, and   #
# handles chaining of transformations.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class Pipeline():

    #****************************#
    #   general initialization   #
    #****************************#
    def __init__(self, modules=None):
        """
        Initialize the Pipeline with a list of modules.

        Parameters:
            modules (list): List of modules to be executed in the pipeline.
        """
        self.modules = modules if modules is not None else []


    #**********************#
    #   run the pipeline   #
    #**********************#
    def run(self, data):
        """
        Run the pipeline on the provided data.

        Parameters:
            data: Input data to be processed by the pipeline.

        Returns:
            Processed data after passing through all modules.
        """
        for module in self.modules:
            if hasattr(module, 'forward'):
                data = module.forward(data)
            else:
                raise AttributeError(f"Module {module.__class__.__name__} does not have a "
                                     f"'forward' method.")
        return data