####################################################################################################
#                                            fsl_mrs.py                                            #
####################################################################################################
#                                                                                                  #
# Authors:  J. P. Merkofer (j.p.merkofer@tue.nl)                                                   #
#                                                                                                  #
# Created:  20/07/25                                                                               #
#                                                                                                  #
# Purpose: Defines the interface and transformation methods for FSL-MRS processing and analysis.   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

from fsl_mrs.utils.preproc import nifti_mrs_proc
from fsl_mrs.utils.preproc.combine import estimate_noise_cov, CovarianceEstimationError
from fsl_mrs.utils.mrs_io import read_FID

from nifti_mrs.nifti_mrs import NIFTI_MRS

# ownn
from core.nifti_mrs_plus import NIfTI_MRS_Plus
from core.transform import Transform



#**************************************************************************************************#
#                                       Class FSL_Transform                                        #
#**************************************************************************************************#
#                                                                                                  #
#  This class extends the Transform class to provide specific transformations for FSL-MRS          #
#  processing on the NIfTI_MRS_Plus objects. It implements the forward method to apply the         #
#  transformation logic.                                                                           #
#                                                                                                  #
#**************************************************************************************************#
class FSL_Transform(Transform):

    #****************************#
    #   general initialization   #
    #****************************#
    def __init__(self, verbose=False, conjugate=False, apply_to_water=True):
        """
        Initialize the FSL_Transform with optional parameters.

        Parameters:
            verbose (bool): If True, print verbose messages during processing.
            conjugate (bool): If True, apply conjugation to the data.
            apply_to_water (bool): If True, apply the transformation to water reference as well.
        """
        super().__init__()
        self.verbose = verbose
        self.conjugate = conjugate
        self.apply_to_water = apply_to_water


    #****************************************#
    #   forward method on single NIfTI_MRS   #
    #****************************************#
    def forward_single(self, nifti_obj, idx):
        """
        Apply the transformation to a single NIfTI_MRS object.

        Parameters:
            nifti_obj (NIFTI_MRS): The NIFTI_MRS object to be transformed.
            idx (int): Index of the item in the list for logging purposes.

        Returns:
            NIFTI_MRS: The transformed NIFTI_MRS object.
        """
        raise NotImplementedError("Subclasses must implement forward_single.")


    #*****************************************#
    #   process a list of NIfTI_MRS objects   #
    #*****************************************#
    def process_list(self, items, name="data"):
        """
        Process a list of NIfTI_MRS objects, applying conjugation if specified.

        Parameters:
            items (list): List of NIFTI_MRS objects to be processed.
            name (str): Name of the items for logging purposes.

        Returns:
            list: List of transformed NIFTI_MRS objects.
        """
        items = self._maybe_listify(items, name)
        items = self._maybe_conjugate(items, name)
        return [self.forward_single(x, idx=i) for i, x in enumerate(items)]


    #********************#
    #   forward method   #
    #********************#
    def forward(self, data, water=None):
        """
        Apply the transformation to the provided data and optional water reference.

        Parameters:
            data (list or NIFTI_MRS_Plus): Input data to be transformed.
            water (list or NIFTI_MRS_Plus, optional): Optional water reference data.

        Returns:
            tuple: Transformed data and water reference (if applicable).
        """
        data_out = self.process_list(data, name="data")
        water_out = self.process_list(water, name="water") if (self.apply_to_water and water) \
            else water
        data_out = NIfTI_MRS_Plus.from_nifti_list(data_out)
        if water_out is not None: water_out = NIfTI_MRS_Plus.from_nifti_list(water_out)
        return data_out, water_out


    #*********************#
    #   utility methods   #
    #*********************#
    def _vprint(self, msg):
        """
        Print a message if verbose mode is enabled.

        Parameters:
            msg (str): The message to be printed.
        """
        if self.verbose:
            print(f"[FSL_Transform]: {msg}")

    def _maybe_listify(self, obj, name="data"):
        """
        Ensure the input is a list of NIFTI_MRS objects or has a .list() method.
        If the input is None, return an empty list.

        Parameters:
            obj (list or NIFTI_MRS_Plus): The input object to be checked.
            name (str): Name of the object for error messages.

        Returns:
            list: A list of NIFTI_MRS objects.

        Raises:
            TypeError: If the input is not a list or does not have a .list() method.
        """
        if obj is None:
            return []
        if hasattr(obj, "list"):  # e.g., NIfTI_MRS_Plus or custom wrapper
            obj = obj.list()
        if not isinstance(obj, list):
            raise TypeError(f"{name} must be a list or have a .list() method.")
        if not all(isinstance(i, NIFTI_MRS) for i in obj):
            raise TypeError(f"All elements in {name} must be instances of NIFTI_MRS.")
        return obj

    def _maybe_conjugate(self, items, name="data"):
        """
        Apply conjugation to the items if specified.

        Parameters:
            items (list): List of NIFTI_MRS objects to be conjugated.
            name (str): Name of the items for logging purposes.

        Returns:
            list: List of conjugated NIFTI_MRS objects if conjugation is enabled,
                  otherwise the original list.
        """
        if self.conjugate:
            self._vprint(f"Applying conjugation to {name}")
            return [nifti_mrs_proc.conjugate(x) for x in items]
        return items



#**************************************************************************************************#
#                                       Class FSL_CoilCombine                                      #
#**************************************************************************************************#
#                                                                                                  #
#  This class extends the FSL_Transform to provide specific coil combination methods for FSL-MRS.  #
#                                                                                                  #
#**************************************************************************************************#
class FSL_CoilCombine(FSL_Transform):

    #****************************#
    #   general initialization   #
    #****************************#
    def __init__(self, verbose=False, conjugate=False, apply_to_water=True, use_reference=True):
        """
        Initialize the FSL_CoilCombine with optional parameters.

        Parameters:
            verbose (bool): If True, print verbose messages during processing.
            conjugate (bool): If True, apply conjugation to the data.
            apply_to_water (bool): If True, apply the transformation to water reference as well.
            use_reference (bool): If True, use a water reference for coil combination.
        """
        super().__init__(verbose=verbose, conjugate=conjugate, apply_to_water=apply_to_water)
        self.use_reference = use_reference
        self._reference = None


    #****************************************#
    #   forward method on single NIfTI_MRS   #
    #****************************************#
    def forward_single(self, nifti_obj, idx):
        """
        Apply coil combination to a single NIfTI_MRS object.

        Parameters:
            nifti_obj (NIFTI_MRS): The NIFTI_MRS object to be transformed.
            idx (int): Index of the item in the list for logging purposes.

        Returns:
            NIFTI_MRS: The transformed NIFTI_MRS object after coil combination.
        """
        if 'DIM_COIL' not in nifti_obj.dim_tags:
            self._vprint(f"[item {idx}] No DIM_COIL found — skipping coil combine.")
            return nifti_obj

        self._vprint(f"[item {idx}] Applying coil combination" + (" with reference"
                                                                  if self._reference[idx] else ""))
        return nifti_mrs_proc.coilcombine(nifti_obj, reference=self._reference[idx] if self.use_reference else None)


    #********************#
    #   forward method   #
    #********************#
    def forward(self, data, water=None):
        """
        Apply coil combination to the provided data and optional water reference.
        This method processes the data and water reference, applying coil combination
        and optionally using a water reference for improved results.

        Parameters:
            data (list or NIFTI_MRS_Plus): Input data to be transformed.
            water (list or NIFTI_MRS_Plus, optional): Optional water reference data.

        Returns:
            tuple: Transformed data and water reference (if applicable).
        """
        water_list = self._maybe_listify(water, name="water") if self.apply_to_water else []
        if self.use_reference and water_list:
            self._prepare_reference(water_list)

        data_out = self.process_list(data, name="data")
        water_out = self.process_list(water, name="water") if self.apply_to_water and water \
            else water
        data_out = NIfTI_MRS_Plus.from_nifti_list(data_out)
        if water_out is not None: water_out = NIfTI_MRS_Plus.from_nifti_list(water_out)
        return data_out, water_out


    #*********************#
    #   utility methods   #
    #*********************#
    def _maybe_average(self, nifti_obj):
        """
        Average over the dynamic or edit dimensions in the NIfTI_MRS object if they exist.

        Parameters:
            nifti_obj (NIFTI_MRS): The NIFTI_MRS object to be averaged.

        Returns:
            NIFTI_MRS: The averaged NIFTI_MRS object.
        """
        for dim in ('DIM_DYN', 'DIM_EDIT'):
            if dim in nifti_obj.dim_tags:
                self._vprint(f"Averaging over {dim} in reference")
                nifti_obj = nifti_mrs_proc.average(nifti_obj, dim)
        return nifti_obj

    def _prepare_reference(self, water_list):
        """
        Prepare the water reference for coil combination by averaging over dynamic or edit dimensions.
        This method processes the water_list to create a reference for coil combination.

        Parameters:
            water_list (list): List of NIFTI_MRS objects representing water references.
        """
        self._reference = [self._maybe_average(water_list[i]) for i in range(len(water_list))]