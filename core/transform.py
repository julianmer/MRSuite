####################################################################################################
#                                           transform.py                                           #
####################################################################################################
#                                                                                                  #
# Authors:  J. P. Merkofer (j.p.merkofer@tue.nl)                                                   #
#                                                                                                  #
# Created:  20/07/25                                                                               #
#                                                                                                  #
# Purpose: Defines the interface for transformations applied to NIfTI_MRS_Plus objects.            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#



#**************************************************************************************************#
#                                         Class Transform                                          #
#**************************************************************************************************#
#                                                                                                  #
#  This class serves as a base for all transformations applied to NIfTI_MRS_Plus objects.          #
#  It provides a structure for defining transformations, including initialization and a forward    #
#  method for processing.                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class Transform():

    #****************************#
    #   general initialization   #
    #****************************#
    def __init__(self, *args, **kwargs):
        """
        Initialize the Transform with optional arguments.

        Parameters:
            *args: Positional arguments for the transformation.
            **kwargs: Keyword arguments for the transformation.
        """
        pass


    #**********************#
    #   forward method    #
    #**********************#
    def forward(self, data):
        """
        Apply the transformation to the provided data.

        Parameters:
            data: Input data to be transformed.

        Returns:
            Transformed data.
        """
        raise NotImplementedError("The forward method must be implemented in subclasses.")