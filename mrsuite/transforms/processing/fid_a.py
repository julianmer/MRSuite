####################################################################################################
#                                            fid_a.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-08-02                                                                              #
#                                                                                                  #
# Purpose: Defines the interface and transformation methods for FID-A processing and analysis.     #
#          Integrates Octave to run MATLAB-based FID-A functions...                                #
#                                                                                                  #
# Mention: Simpson, R., Devenyi, G.A., Jezzard, P., Hennessy, T.J. and Near, J. (2017),            #
#          Advanced processing and simulation of MRS data using the FID appliance (FID-A) —        #
#          An open source, MATLAB-based toolkit. Magn. Reson. Med., 77: 23-33.                     #
#          doi: https://doi.org/10.1002/mrm.26091                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os

# third-party
from oct2py import Oct2Py

# own
from mrsuite.core.nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core.transform import Transform


#**************************************************************************************************#
#                                       Class FIDA_Transform                                       #
#**************************************************************************************************#
#                                                                                                  #
# This class extends the Transform class to provide FID-A specific transformations for MRS data.   #
# It integrates with Octave to run MATLAB-based FID-A functions directly from Python.              #
#                                                                                                  #
#**************************************************************************************************#
class FIDA_Transform(Transform):
    """
    Base class for FID-A transformations.

    This class initializes the Octave environment and adds all FID-A paths recursively.
    It is intended to be subclassed for specific FID-A transformations.

    Attributes:
        verbose (bool): If True, print additional information during processing.
        octave (Oct2Py): Instance of Oct2Py to run MATLAB code.
        _vprint (function): Function to print messages if verbose is True.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the FIDATransform.

        Args:
            verbose (bool): If True, print additional information during processing.
        """
        super().__init__(verbose=verbose)
        self.octave = Oct2Py()
        if not verbose:
            self.octave.eval("warning('off', 'all');")

        # add all FID-A folders recursively
        fidA_root = 'dependencies/fidA'
        for root, dirs, files in os.walk(fidA_root):
            self.octave.addpath(root)

        self._vprint(f"Added all FID-A paths from {fidA_root}")


#**************************************************************************************************#
#                                      Class FIDA_EddyCorrect                                      #
#**************************************************************************************************#
#                                                                                                  #
# This class implements the FID-A Eddy Current Correction transformation, from op_ecc.m.           #
#                                                                                                  #
#**************************************************************************************************#
class FIDA_EddyCorrect(FIDA_Transform):
    """
    FID-A dddy current correction transformation.

    This class applies eddy current correction to MRS data using FID-A's op_ecc function.
    It requires a water reference for the correction process.

    Attributes:
        verbose (bool): If True, print additional information during processing.
        octave (Oct2Py): Instance of Oct2Py to run MATLAB code.
        _vprint (function): Function to print messages if verbose is True.

    Raises:
        ValueError: If no water reference is provided.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the FIDA_EddyCorrect transformation.

        Args
            verbose (bool): If True, print additional information during processing.
        """
        super().__init__(verbose=verbose)

    def forward(self, data: NIfTI_MRS_Plus, water: NIfTI_MRS_Plus = None):
        """
        Apply Eddy Current Correction to the provided MRS data using FID-A's op_ecc.

        Args:
            data (NIfTI_MRS_Plus): Input MRS data to be corrected.
            water (NIfTI_MRS_Plus): Water reference data.

        Returns:
            list: List of FID-A structs (one per dynamic).
        """
        if water is None:
            raise ValueError("FIDA_EddyCorrect requires a water reference.")

        # TODO: Implement the actual Eddy Current Correction logic
        pass
