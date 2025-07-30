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
from typing import Optional

# own
from core.nifti_mrs_plus import NIfTI_MRS_Plus


#**************************************************************************************************#
#                                         Class Transform                                          #
#**************************************************************************************************#
#                                                                                                  #
#  This class serves as a base for all transformations applied to NIfTI_MRS_Plus objects.          #
#  It provides a structure for defining transformations, including initialization and a forward    #
#  method for processing.                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class Transform:

    def __init__(self,  *args: object, **kwargs: object) -> None:
        """
        Initialize the Transform with optional arguments.

        Parameters:
            *args: Positional arguments for the transformation.
            **kwargs: Keyword arguments for the transformation.
        """
        pass

    def __call__(self, data: NIfTI_MRS_Plus,
                 water: Optional[NIfTI_MRS_Plus] = None) -> NIfTI_MRS_Plus:
        """ Make the Transform class callable. """
        return self.forward(data, water=water)

    def forward(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus] = None) \
            -> (NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]):
        """
        Apply the transformation to the provided data.

        Parameters:
            data: Input data to be transformed.
            water: Optional water reference data, if required by the transformation.

        Returns:
            Transformed data (and optionally water reference data).
        """
        raise NotImplementedError("The forward method must be implemented in subclasses.")