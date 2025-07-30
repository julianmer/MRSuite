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
import numpy as np

from copy import deepcopy

from nifti_mrs import nifti_mrs
from nifti_mrs.utils import modify_hdr_ext

from typing import List, Iterator, Optional, Union



#**************************************************************************************************#
#                                       Class NIfTI_MRS_Plus                                       #
#**************************************************************************************************#
#                                                                                                  #
# This class extends the NIfTI_MRS class to include an additional batch/subjects dimension.        #
# It is designed to handle NIfTI files with multiple subjects, allowing to optimize batch          #
# processing of MRS data.                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class NIfTI_MRS_Plus:

    def __init__(self, nifti_list: List[nifti_mrs.NIFTI_MRS],
                 *args: object, **kwargs: object) -> None:
        """ Initialize the NIfTI_MRS_Plus object with data and header.

        Parameters:
            nifti_listList (list): List of NIFTI_MRS objects to be combined.
            *args: Positional arguments for the NIfTI_MRS_Plus object.
            **kwargs: Keyword arguments for the NIfTI_MRS_Plus object.
        """
        self.nifti_list: List[nifti_mrs.NIFTI_MRS] = nifti_list

        self._batched_data: Optional[np.ndarray] = None
        self._updated_hdr_ext: Optional[object] = None

    def __call__(self, *args: object, **kwargs: object) -> "NIfTI_MRS_Plus":
        """ Make the NIfTI_MRS_Plus class callable. """
        return self.from_nifti_list(*args, **kwargs)

    def __getitem__(self, index: int) -> nifti_mrs.NIFTI_MRS:
        """ Allow indexing into the nifti_list directly from the NIfTI_MRS_Plus object. """
        return self.nifti_list[index]

    def __len__(self) -> int:
        """ Return the number of NIFTI_MRS elements in the batch. """
        return len(self.nifti_list)

    @classmethod
    def from_nifti_list(cls, nifti_list: List[nifti_mrs.NIFTI_MRS]) -> "NIfTI_MRS_Plus":
        """ Create an instance of NIfTI_MRS_Plus from a list of NIfTI_MRS objects.

        Parameters:
            nifti_list (list): List of NIfTI_MRS objects to be combined.

        Returns:
            NIfTI_MRS_Plus: An instance containing the combined data.
        """
        # validate all elements are NIfTI_MRS instances
        if not all(isinstance(n, nifti_mrs.NIFTI_MRS) for n in nifti_list):
            raise TypeError("All elements must be instances of NIFTI_MRS.")
        return cls(nifti_list)

    def sync_headers(self, source_idx: int = 0) -> None:
        """Sync headers from one subject to all others.

        Updates all headers to match the header of the subject at 'source_idx'.

        Parameters:
            source_idx (int): Index of the reference subject.
        """
        # TODO: check usage of this method...
        ref_hdr = deepcopy(self.nifti_list[source_idx].hdr_ext)
        for nifti in self.nifti_list:
            nifti.hdr_ext = deepcopy(ref_hdr)
        self._updated_hdr_ext = ref_hdr

    def numpy(self) -> np.ndarray:
        """ Return batched tensor of shape [B, ...] where B = number of subjects. """
        if self._batched_data is None:
            self._batched_data = np.stack([n[:].conj() for n in self.nifti_list], axis=0)
        return self._batched_data

    def list(self) -> List[nifti_mrs.NIFTI_MRS]:
        """ Return list of NIFTI_MRS objects, applying updated header if needed. """
        if self._updated_hdr_ext is not None:
            for nifti in self.nifti_list:
                nifti.hdr_ext = deepcopy(self._updated_hdr_ext)
        return self.nifti_list

    @property
    def shape(self) -> tuple[int, ...]:
        """ Return shape of the batched tensor. """
        return (len(self.nifti_list),) + self.nifti_list[0].shape

    @property
    def dim_tags(self) -> object:
        """ Return dimension tags from the first subject. """
        return self.nifti_list[0].dim_tags