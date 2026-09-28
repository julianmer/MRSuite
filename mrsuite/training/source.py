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
from typing import Dict, Tuple

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
        components (Dict[str, np.ndarray]): Name -> (n, points) complex FIDs in the data's
            own convention, e.g. each metabolite's signal and 'baseline'. Together with
            the noise and anything unnamed they add up to the data.
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
    it prepares real data, at no cost per spectrum.

    Attributes:
        acquisition (Dict[str, float]): 'points', 'bandwidth' (Hz), 'frequency' (MHz)
            and 'centre' (ppm), as a model states what it was trained on.
    """

    acquisition: Dict[str, float] = {}

    #**********#
    #   draw   #
    #**********#
    def draw(self, n: int) -> Tuple[np.ndarray, Truth]:
        """The next ``n`` spectra as (n, points) complex FIDs, and their truth."""
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
        return self._template(n).set_data(fids.reshape(n, 1, 1, 1, -1)), truth

    #**************#
    #   template   #
    #**************#
    def _template(self, n: int) -> NIfTI_MRS_Plus:
        """Headers for a batch of ``n``, built once: only the samples change per batch."""
        cache = self.__dict__.setdefault("_templates", {})
        if n not in cache:
            a = self.acquisition
            item = gen_nifti_mrs(np.zeros((1, 1, 1, int(a["points"])), complex),
                                 1 / a["bandwidth"], a["frequency"])
            if "centre" in a:
                item.add_hdr_field("SpecFreqChemShift", a["centre"])
            cache[n] = NIfTI_MRS_Plus([item.copy() for _ in range(n)])
        return cache[n]
