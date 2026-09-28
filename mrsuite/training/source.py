####################################################################################################
#                                            source.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: What a model learns from: batches of spectra and what each really holds. A simulator,   #
#          a benchmark's training split or your own data is a source by giving batches in this     #
#          one shape, so any model learns from any of them.                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nifti_mrs.create_nmrs import gen_nifti_mrs

from nifti_mrs_plus import NIfTI_MRS_Plus


#**************************************************************************************************#
#                                           Class Truth                                            #
#**************************************************************************************************#
#                                                                                                  #
# What a batch of spectra really holds, by name, one value or signal per spectrum.                 #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Truth:
    """What a batch of spectra really holds, by name, one value or signal per spectrum.

    By name rather than by position, so a model and a source that list their basis in
    different orders still line up, and a name one lacks fails loudly.

    Attributes:
        concentrations (Dict[str, np.ndarray]): Metabolite -> (n,) amplitudes of the
            source's basis functions, which is what a model quantifies.
        components (Dict[str, np.ndarray]): Name -> complex FIDs shaped as the data,
            (n, points, *dims), in the data's own convention: each basis function's
            signal, 'baseline', 'water', 'noise', whatever the source can tell apart.
            Together with anything unnamed they add up to the data.
        parameters (Dict[str, np.ndarray]): Name -> (n,) values of the source's own
            signal model, e.g. linewidths and phases, in its own terms.
    """

    concentrations: Dict[str, np.ndarray] = field(default_factory=dict)
    components: Dict[str, np.ndarray] = field(default_factory=dict)
    parameters: Dict[str, np.ndarray] = field(default_factory=dict)


#**************************************************************************************************#
#                                           Class Source                                           #
#**************************************************************************************************#
#                                                                                                  #
# Batches of spectra with their truth, to learn from.                                              #
#                                                                                                  #
#**************************************************************************************************#
class Source:
    """Batches of spectra with their truth, to learn from.

    A subclass states the acquisition its spectra share and implements ``draw``; the
    batch comes back as NIfTI-MRS+ holding one array, so a model prepares it exactly as
    it prepares real data, from the array rather than from each spectrum. Spectra have
    any shape NIfTI-MRS allows: several coils, say, are a fifth dimension with its tag.

    Attributes:
        acquisition (Dict[str, float]): 'points', 'bandwidth' (Hz), 'frequency' (MHz)
            and 'centre' (ppm), as a model states what it was trained on.
        dim_tags (List[Optional[str]]): Tags of the dimensions after the points, as
            NIfTI-MRS names them (e.g. ['DIM_COIL']); empty for single spectra.
    """

    acquisition: Dict[str, float] = {}
    dim_tags: List[Optional[str]] = []

    #**********#
    #   draw   #
    #**********#
    def draw(self, n: int) -> Tuple[np.ndarray, Truth]:
        """The next ``n`` spectra as (n, points, *dims) complex FIDs, and their truth."""
        raise NotImplementedError

    #***********#
    #   batch   #
    #***********#
    def batch(self, n: int) -> Tuple[NIfTI_MRS_Plus, Truth]:
        """
        The next ``n`` spectra as NIfTI-MRS+, and their truth.

        Args:
            n (int): Spectra in the batch.

        Returns:
            Tuple[NIfTI_MRS_Plus, Truth]: The spectra, held as one array, and the truth.
        """
        fids, truth = self.draw(n)
        fids = fids.reshape(n, 1, 1, 1, *fids.shape[1:])
        item = self._template(fids.shape[1:])
        return NIfTI_MRS_Plus([item.copy() for _ in range(n)]).set_data(fids), truth

    #**************#
    #   template   #
    #**************#
    def _template(self, shape: Tuple[int, ...]) -> Any:
        """One spectrum's header for this shape, built once. Each batch gets copies of its
        own: NIfTI-MRS+ writes its array back into its objects, so batches cannot share."""
        cache = self.__dict__.setdefault("_templates", {})
        if shape not in cache:
            a = self.acquisition
            tags = (list(self.dim_tags) + [None] * 3)[:3]
            item = gen_nifti_mrs(np.zeros(shape, complex), 1 / a["bandwidth"],
                                 a["frequency"], dim_tags=tags)
            if "centre" in a:
                item.add_hdr_field("SpecFreqChemShift", a["centre"])
            cache[shape] = item
        return cache[shape]
