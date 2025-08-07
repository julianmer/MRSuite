####################################################################################################
#                                            fsl_mrs.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-07-25                                                                              #
#                                                                                                  #
# Purpose: Defines the interface and transformation methods for FSL-MRS processing and analysis.   #
#                                                                                                  #
# Mention: Clarke WT, Stagg CJ, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis package.    #
#          Magnetic Resonance in Medicine 2021;85:2950–2964.                                       #
#          doi: https://doi.org/10.1002/mrm.28630.                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import List, Optional, Tuple

# third-party
from fsl_mrs.utils.preproc import nifti_mrs_proc

from nifti_mrs.nifti_mrs import NIFTI_MRS

# own
from mrsuite.core.nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core.transform import Transform


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
    """
    Base class for FSL-MRS transformations.

    This class provides a structure for wrapping FSL-MRS transformations to apply to
    NIfTI_MRS_Plus objects.

    Attributes:
        verbose (bool): If True, print additional information during processing.
        conjugate (bool): If True, apply conjugation to the data.
        apply_to_water (bool): If True, apply the transformation to water reference as well.

    Raises:
        NotImplementedError: If the forward_single method is not implemented in a subclass.
        TypeError: If the input type is not NIfTI_MRS_Plus or does not contain NIFTI_MRS objects.
    """

    def __init__(self, verbose: bool = False, conjugate: bool = False,
                 apply_to_water: bool = True, **kwargs: object) -> None:
        """
        Initialize the FSL_Transform with optional parameters.

        Args:
            verbose (bool): If True, print verbose messages during processing.
            conjugate (bool): If True, apply conjugation to the data.
            apply_to_water (bool): If True, apply the transformation to water reference as well.
        """
        super().__init__(verbose=verbose)
        self.conjugate = conjugate
        self.apply_to_water = apply_to_water
        self.kwargs = kwargs

    def forward_single(self, nifti_obj: NIFTI_MRS, idx: int) -> NIFTI_MRS:
        """
        Apply the transformation to a single NIfTI_MRS object.

        Args:
            nifti_obj (NIFTI_MRS): The NIFTI_MRS object to be transformed.
            idx (int): Index of the item in the list for logging purposes.

        Returns:
            NIFTI_MRS: The transformed NIFTI_MRS object.
        """
        raise NotImplementedError("Subclasses must implement forward_single.")

    def process_list(self, items: List[NIFTI_MRS], name: str = "data") -> List[NIFTI_MRS]:
        """
        Process a list of NIfTI_MRS objects, applying conjugation if specified.

        Args:
            items (list): List of NIFTI_MRS objects to be processed.
            name (str): Name of the items for logging purposes.

        Returns:
            list: List of transformed NIFTI_MRS objects.
        """
        items = self._maybe_listify(items, name)
        items = self._maybe_conjugate(items, name)
        return [self.forward_single(x, idx=i) for i, x in enumerate(items)]

    def forward(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus] = None) \
            -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """
        Apply the transformation to the provided data and optional water reference.

        Args:
            data (list or NIFTI_MRS_Plus): Input data to be transformed.
            water (list or NIFTI_MRS_Plus, optional): Optional water reference data.

        Returns:
            tuple: Transformed data and water reference (if applicable).
        """
        data_out = self.process_list(data, name="data")
        data_out = NIfTI_MRS_Plus.from_nifti_list(data_out)
        if self.apply_to_water:
            water_out = self.process_list(water, name="water") if water is not None else None
            water_out = NIfTI_MRS_Plus.from_nifti_list(water_out) if water_out is not None else None
        return data_out, water_out

    def _maybe_listify(self, obj: NIfTI_MRS_Plus, name: str = "data") -> List[NIFTI_MRS]:
        """
        Ensure the input is a list of NIFTI_MRS objects or has a .list() method.
        If the input is None, return an empty list.

        Args:
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

    def _maybe_conjugate(self, items: List[NIFTI_MRS], name: str = "data") -> List[NIFTI_MRS]:
        """
        Apply conjugation to the items if specified.

        Args:
            items (list): List of NIFTI_MRS objects to be conjugated.
            name (str): Name of the items for logging purposes.

        Returns:
            list: List of conjugated NIFTI_MRS objects if conjugation is enabled,
                  otherwise the original list.
        """
        if self.conjugate:
            self._vprint(f"Applying conjugation to {name}")
            return [nifti_mrs_proc.conjugate(data=x, **self.kwargs) for x in items]
        return items


#**************************************************************************************************#
#                                       Class FSL_CoilCombine                                      #
#**************************************************************************************************#
#                                                                                                  #
#  This class extends the FSL_Transform to provide specific coil combination methods for FSL-MRS.  #
#                                                                                                  #
#**************************************************************************************************#
class FSL_CoilCombine(FSL_Transform):
    """
    Class for performing coil combination on NIfTI_MRS objects using FSL-MRS methods.

    This class inherits from FSL_Transform and overrides the forward function in order to prepare
    the reference for coil combination and apply the transformation to the data.

    Attributes:
        verbose (bool): If True, print verbose messages during processing.
        conjugate (bool): If True, apply conjugation to the data.
        apply_to_water (bool): If True, apply the transformation to water reference as well.
        use_reference (bool): If True, use a water reference for coil combination.
        _reference (Optional[List[NIFTI_MRS]]): List of water references for coil
            combination, prepared by averaging over dynamic or edit dimensions.

    Raises:
        TypeError: If the input type is not NIfTI_MRS_Plus or does not contain NIFTI_MRS objects.
    """

    def __init__(self, verbose: bool = False, conjugate: bool = False,
                 apply_to_water: bool = True, use_reference: bool = True, **kwargs: object) -> None:
        """
        Initialize the FSL_CoilCombine with optional parameters.

        Args:
            verbose (bool): If True, print verbose messages during processing.
            conjugate (bool): If True, apply conjugation to the data.
            apply_to_water (bool): If True, apply the transformation to water reference as well.
            use_reference (bool): If True, use a water reference for coil combination.
        """
        super().__init__(verbose=verbose, conjugate=conjugate, apply_to_water=apply_to_water,
                         **kwargs)
        self.use_reference = use_reference
        self._reference = None

    def forward_single(self, nifti_obj: NIFTI_MRS, idx: int) -> NIFTI_MRS:
        """
        Apply coil combination to a single NIfTI_MRS object.

        Args:
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
        return nifti_mrs_proc.coilcombine(
            data=nifti_obj,
            reference=self._reference[idx] if self.use_reference else None,
            **self.kwargs
        )

    def forward(self, data: NIfTI_MRS_Plus, water: Optional[NIfTI_MRS_Plus] = None) \
            -> Tuple[NIfTI_MRS_Plus, Optional[NIfTI_MRS_Plus]]:
        """
        Apply coil combination to the provided data and optional water reference.

        Args:
            data (list or NIFTI_MRS_Plus): Input data to be transformed.
            water (list or NIFTI_MRS_Plus, optional): Optional water reference data.

        Returns:
            tuple: Transformed data and water reference (if applicable).
        """
        water_list = self._maybe_listify(water, name="water") if self.apply_to_water else []
        if self.use_reference and water_list:
            self._prepare_reference(water_list)

        data_out = self.process_list(data, name="data")
        data_out = NIfTI_MRS_Plus.from_nifti_list(data_out)
        if self.apply_to_water:
            water_out = self.process_list(water, name="water") if water is not None else None
            water_out = NIfTI_MRS_Plus.from_nifti_list(water_out) if water_out is not None else None
        return data_out, water_out

    def _maybe_average(self, nifti_obj: NIFTI_MRS, idx: int) -> NIFTI_MRS:
        """
        Average over the dynamic or edit dimensions in the NIfTI_MRS object if they exist.

        Args:
            nifti_obj (NIFTI_MRS): The NIFTI_MRS object to be averaged.
            idx (int): Index of the item in the list for logging purposes.

        Returns:
            NIFTI_MRS: The averaged NIFTI_MRS object.
        """
        for dim in ('DIM_DYN', 'DIM_EDIT'):
            if dim in nifti_obj.dim_tags:
                self._vprint(f"[Item {idx}] Averaging over {dim} in reference")
                nifti_obj = nifti_mrs_proc.average(data=nifti_obj, dim=dim, **self.kwargs)
        return nifti_obj

    def _prepare_reference(self, water_list: List[NIFTI_MRS]) -> None:
        """
        Prepare the reference for coil combination by averaging over dynamic or edit dimensions.

        Args:
            water_list (list): List of NIFTI_MRS objects representing water references.

        Returns:
            None: The method modifies the internal state of the class to store the
                  averaged references.
        """
        self._reference = [self._maybe_average(water_list[i], i) for i in range(len(water_list))]


#**************************************************************************************************#
#                                         Class FSL_Average                                        #
#**************************************************************************************************#
#                                                                                                  #
#  This class extends the FSL_Transform to provide averaging method for FSL-MRS data.              #
#                                                                                                  #
#**************************************************************************************************#
class FSL_Average(FSL_Transform):
    """
    Class for averaging NIfTI_MRS objects over a specified dimension.

    This class inherits from FSL_Transform and implements the forward_single method to apply
    averaging to the NIfTI_MRS objects. It can average over a specified dimension.

    Attributes:
        verbose (bool): If True, print verbose messages during processing.
        conjugate (bool): If True, apply conjugation to the data.
        apply_to_water (bool): If True, apply the transformation to water reference as well.
        dim (str): Dimension to average over, default is 'DIM_DYN'.

    Raises:
        TypeError: If the input type is not NIfTI_MRS_Plus or does not contain NIFTI_MRS objects.
    """

    def __init__(self, verbose: bool = False, conjugate: bool = False,
                 apply_to_water: bool = True, dim: str = 'DIM_DYN', **kwargs: object) -> None:
        """
        Initialize the FSL_Average with optional parameters.

        Args:
            verbose (bool): If True, print verbose messages during processing.
            conjugate (bool): If True, apply conjugation to the data.
            apply_to_water (bool): If True, apply the transformation to water reference as well.
            dim (str): Dimension to average over, default is 'DIM_DYN'.
        """
        super().__init__(verbose=verbose, conjugate=conjugate, apply_to_water=apply_to_water,
                         **kwargs)
        self.dim = dim

    def forward_single(self, nifti_obj: NIFTI_MRS, idx: int) -> NIFTI_MRS:
        """
        Apply averaging to a single NIfTI_MRS object.

        Args:
            nifti_obj (NIFTI_MRS): The NIFTI_MRS object to be transformed.
            idx (int): Index of the item in the list for logging purposes.

        Returns:
            NIFTI_MRS: The averaged NIFTI_MRS object.
        """
        self._vprint(f"[item {idx}] Averaging over all dimensions")
        return nifti_mrs_proc.average(data=nifti_obj, dim=self.dim, **self.kwargs)


#**************************************************************************************************#
#                                         Class FSL_Align                                          #
#**************************************************************************************************#
#                                                                                                  #
#  This class extends the FSL_Transform to align phase and frequency where there are multiple      #
#  transients. Can be run across a dimension (specified by a tag), or all spectra stored in        #
#  higher dimensions.                                                                              #
#                                                                                                  #
#**************************************************************************************************#
class FSL_Align(FSL_Transform):
    """
    Class for aligning phase and frequency of NIfTI_MRS objects.

    This class inherits from FSL_Transform and implements the forward_single method to apply
    phase and frequency alignment to the NIfTI_MRS objects. It can align over a
    specified dimension or all spectra stored in higher dimensions.

    Attributes:
        verbose (bool): If True, print verbose messages during processing.
        conjugate (bool): If True, apply conjugation to the data.
        apply_to_water (bool): If True, apply the transformation to water reference as well.
        dim (str): Dimension to align over or 'all', default is 'DIM_DYN'.

    Raises:
        TypeError: If the input type is not NIfTI_MRS_Plus or does not contain NIFTI_MRS objects.
        ValueError: If an invalid dimension is specified for alignment.
    """

    def __init__(self, verbose: bool = False, conjugate: bool = False,
                 apply_to_water: bool = True, dim: str = 'DIM_DYN', **kwargs: object) -> None:
        """
        Initialize the FSL_Align with optional parameters.

        Args:
            verbose (bool): If True, print verbose messages during processing.
            conjugate (bool): If True, apply conjugation to the data.
            apply_to_water (bool): If True, apply the transformation to water reference as well.
            dim (str): Dimension to align over or 'all', default is 'DIM_DYN'.
        """
        super().__init__(verbose=verbose, conjugate=conjugate, apply_to_water=apply_to_water,
                         **kwargs)
        self.dim = dim


    def forward_single(self, nifti_obj: NIFTI_MRS, idx: int) -> NIFTI_MRS:
        """
        Align phase and frequency for a single NIfTI_MRS object.

        Args:
            nifti_obj (NIFTI_MRS): The NIFTI_MRS object to be transformed.
            idx (int): Index of the item in the list for logging purposes.

        Returns:
            NIFTI_MRS: The aligned NIFTI_MRS object.
        """
        self._vprint(f"[item {idx}] Aligning phase and frequency")
        return nifti_mrs_proc.align(data=nifti_obj, dim=self.dim, **self.kwargs)