####################################################################################################
#                                        nifti_mrs_plus.py                                         #
####################################################################################################
#                                                                                                  #
# Authors:  J. P. Merkofer (j.p.merkofer@tue.nl)                                                   #
#                                                                                                  #
# Created:  20/07/25                                                                               #
#                                                                                                  #
# Purpose: Handles batch aware NIfTI_MRS data, extending the NIfTI_MRS class to support a          #
#          batch/subjects dimension.                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from copy import deepcopy
import numpy as np

from nifti_mrs import nifti_mrs
from nifti_mrs.utils import modify_hdr_ext



#**************************************************************************************************#
#                                       Class NIfTI_MRS_Plus                                       #
#**************************************************************************************************#
#                                                                                                  #
# This class extends the NIfTI_MRS class to include an additional batch/subjects dimension.        #
# It is designed to handle NIfTI files with multiple subjects, allowing to optimize batch          #
# processing of MRS data.                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class NIfTI_MRS_Plus():

    #****************************#
    #   general initialization   #
    #****************************#
    def __init__(self, *args, **kwargs):
        """
        Initialize the NIfTI_MRS_Plus object with data and header.

        Parameters:
            *args: Positional arguments for the NIfTI_MRS_Plus object.
            **kwargs: Keyword arguments for the NIfTI_MRS_Plus object.
        """
        self.nifti_list = args[0] if args else []

        self._batched_data = None
        self._updated_hdr_ext = None


    #*************************#
    #   list initialization   #
    #*************************#
    @classmethod
    def from_nifti_list(cls, nifti_list):
        """
        Create an instance of NIfTI_MRS_Plus from a list of NIfTI_MRS objects.

        Parameters:
            nifti_list (list): List of NIfTI_MRS objects to be combined.

        Returns:
            NIfTI_MRS_Plus: An instance containing the combined data.
        """
        # validate all elements are NIfTI_MRS instances
        if not all(isinstance(n, nifti_mrs.NIFTI_MRS) for n in nifti_list):
            raise TypeError("All elements must be instances of NIFTI_MRS.")
        return cls(nifti_list)


    #*************************#
    #   get shape of tensor   #
    #*************************#
    @property
    def shape(self):
        """
        Return shape of the batched tensor.
        """
        return (len(self.nifti_list),) + self.nifti_list[0].shape


    #********************#
    #   dimension tags   #
    #********************#
    @property
    def dim_tags(self):
        """
        Return dimension tags from the first subject.
        """
        return self.nifti_list[0].dim_tags


    #***********************#
    #   tensor conversion   #
    #***********************#
    def numpy(self):
        """
        Return batched tensor of shape [B, ...] where B = number of subjects.
        """
        if self._batched_data is None:
            self._batched_data = np.stack([n[:].conj() for n in self.nifti_list], axis=0)
        return self._batched_data


    #*********************#
    #   list conversion   #
    #*********************#
    def list(self):
        """
        Return list of NIFTI_MRS objects, applying updated header if needed.
        """
        if self._updated_hdr_ext is not None:
            for nifti in self.nifti_list:
                nifti.hdr_ext = deepcopy(self._updated_hdr_ext)
        return self.nifti_list


    #**********************#
    #   header utilities   #
    #**********************#
    def sync_headers(self, source_idx=0):
        """
        Manually sync headers from one subject to all others.
        """
        ref_hdr = deepcopy(self.nifti_list[source_idx].hdr_ext)
        for nifti in self.nifti_list:
            nifti.hdr_ext = deepcopy(ref_hdr)
        self._updated_hdr_ext = ref_hdr
