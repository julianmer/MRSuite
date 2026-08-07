####################################################################################################
#                                           transform.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-07-25                                                                              #
#                                                                                                  #
# Purpose: Defines the interface for transformations applied to NIfTI_MRS_Plus objects.            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import Optional, Tuple

# own
from nifti_mrs_plus import NIfTI_MRS_Plus


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
    """
    Base class for transformations applied to NIfTI_MRS_Plus objects.

    This class defines the interface for transformations, including initialization and a forward
    method for processing the data. It is intended to be subclassed for specific transformations.
    The forward method must be implemented in subclasses to define the specific transformation
    logic. It can also handle optional water reference data if required by the transformation.

    Attributes:
        verbose (bool): If True, print additional information during processing.

    Raises:
        NotImplementedError: If the forward method is not implemented in a subclass.
        TypeError: If the input data is not of type NIfTI_MRS_Plus
    """

    def __init__(self,  verbose: bool = False):
        """
        Initialize the Transform with optional arguments.

        Args:
            verbose (bool): If True, print additional information during processing.
        """
        self.verbose = verbose
        self._vprint = self._vprint if verbose else lambda msg: None

    def __call__(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus] = None) \
            -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """
        Make the Transform class callable.
        """
        if not isinstance(data, NIfTI_MRS_Plus):
            raise TypeError("Input data must be of type NIfTI_MRS_Plus.")
        return self.forward(data, water=water)

    def forward(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus] = None) \
            -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """
        Apply the transformation to the provided data.

        Args:
            data: Input data to be transformed.
            water: Optional water reference data, if required by the transformation.

        Returns:
            Transformed data (and optionally water reference data).
        """
        raise NotImplementedError("The forward method must be implemented in subclasses.")

    def _vprint(self, msg: str):
        """
        Print a message if verbose mode is enabled.
        """
        print(f"[{self.__class__.__name__}] {msg}")