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

from typing import List, Optional, Tuple

# third-party
import numpy as np

from nifti_mrs.nifti_mrs import NIFTI_MRS
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
    """

    def __init__(self, verbose: bool = False, apply_to_water: bool = True) -> None:
        """
        Initialize the FIDATransform.

        Args:
            verbose (bool): If True, print additional information during processing.
            apply_to_water (bool): If True, apply the transformation to water reference data as well.
        """
        super().__init__(verbose=verbose)
        self.apply_to_water = apply_to_water

        # initialize Octave environment
        self.octave = Oct2Py()
        if not verbose:
            self.octave.eval("warning('off', 'all');")

        # add all FID-A (sub)folders recursively
        self.octave.eval("addpath(genpath('./externals/fid-a'))")

    def forward_single(self, data: dict, water: Optional[dict] = None, idx: int = 0) \
            -> Tuple[dict, Optional[dict]]:
        """
        Apply the transformation to a single FID-A object.

        Args:
            data (dict): Input MRS data in FID-A struct format.
            water (Optional[dict]): Optional water reference data in FID-A struct format.
            idx (int): Index of the item in the list for logging purposes.

        Returns:
            Tuple(dict, Optional[dict]): Transformed data and water reference (if applicable).
        """
        raise NotImplementedError("Subclasses must implement forward_single.")

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
        data_fida = [self.nifti2fida(d) for d in data.list()]
        if water is not None:
            water_fida = [self.nifti2fida(w) for w in water.list()]
        else:
            water_fida = None

        # apply the transformation to each item in the list
        out_data, out_water = [], []
        for idx, (d, w) in enumerate(zip(data_fida, water_fida or [None] * len(data_fida))):
            td, tw = self.forward_single(d, w, idx)
            out_data.append(td)
            if tw is not None:
                out_water.append(tw)

        # convert back to NIfTI_MRS_Plus
        # TODO: account for dim pos etc.
        # TODO: adjust headers etc.
        data._batched_data = out_data
        if water is not None:
            water._batched_data = out_water
        return data, water

    def nifti2fida(self, data: NIFTI_MRS) -> dict:
        """
        Convert NIfTI_MRS data to FID-A struct format.

        Adapted from Georg Oeltzschner's io_loadspec_niimrs.m.

        Args:
            data (NIfTI_MRS): Input MRS data to be converted.

        Returns:
            dict: FID-A struct representation of the input data.
        """
        hdr = data.header   # extract the header and header extensions
        hdr_ext = data.hdr_ext  # extract the header extension
        fids = data[:]   # extract the time-domain data

        # extract spectrometer frequency and dwell time
        f0 = hdr_ext['SpectrometerFrequency']
        dt = hdr['pixdim'][4]
        sw = 1 / dt

        # specify dimensions
        dims = {
            'x': 1,
            'y': 2,
            'z': 3,
            't': 4,
            'coils': 0,
            'averages': 0,
            'subSpecs': 0,
            'extras': 0,
        }

        # extract dimensions from the header
        mapping = {
            'DIM_COIL': 'coils',
            'DIM_DYN': 'averages',
            'DIM_EDIT': 'subSpecs',
            'DIM_ISIS': 'subSpecs',
            # everything else falls into 'extras'
        }

        for dim_number in (5, 6, 7):
            key = f'dim_{dim_number}'
            if key in hdr_ext:
                dim_value = hdr_ext[key]
                if dim_value in mapping:
                    dims[mapping[dim_value]] = dim_number
                elif dim_value.startswith(('DIM_INDIRECT', 'DIM_PHASE_CYCLE',
                                           'DIM_MEAS', 'DIM_USER')):
                    dims['extras'] = dim_number
                else:
                    raise ValueError(f'Unknown dimension value specified in {key}: {dim_value}')

        # parse the NIfTI hdr.dim field
        all_dims = hdr['dim'][1:]  # all dimensions (including singletons)
        n_pts = all_dims[dims['t'] - 1]

        # find the number of averages
        if dims['subSpecs'] != 0:
            if dims['averages'] != 0:
                averages = all_dims[dims['averages'] - 1] * all_dims[dims['subSpecs'] - 1]
                raw_averages = averages
            else:
                averages = all_dims[dims['subSpecs'] - 1]
                raw_averages = 1
        else:
            if dims['averages'] != 0:
                averages = all_dims[dims['averages'] - 1]
                raw_averages = averages
            else:
                averages = 1
                raw_averages = 1

        # find the number of subspecs
        if dims['subSpecs'] != 0:
            subspecs = all_dims[dims['subSpecs'] - 1]
            raw_subspecs = subspecs
        else:
            subspecs = 1
            raw_subspecs = subspecs

        # order the data and dimensions
        if np.prod(all_dims[:3]) == 1:
            dims['x'] = 0
            dims['y'] = 0
            dims['z'] = 0
            fids = np.squeeze(fids)

            # subtract 3 from nonzero dims (since x, y, z collapsed)
            sqzDims = []
            for key, val in dims.items():
                if val != 0:
                    dims[key] = val - 3  # still MATLAB-style
                    sqzDims.append(key)

            # helper to permute with MATLAB-style dims
            def permute_fids(order):
                # subtract 1 for numpy indexing
                np_order = [d - 1 for d in order if d > 0]
                return np.transpose(fids, np_order)

            # reorder depending on how many dims survived
            if len(sqzDims) == 5:
                fids = permute_fids([dims['t'], dims['coils'], dims['averages'], dims['subSpecs'], dims['extras']])
                dims.update(t=1, coils=2, averages=3, subSpecs=4, extras=5)

            elif len(sqzDims) == 4:
                if dims['extras'] == 0:
                    fids = permute_fids([dims['t'], dims['coils'], dims['averages'], dims['subSpecs']])
                    dims.update(t=1, coils=2, averages=3, subSpecs=4, extras=0)
                elif dims['subSpecs'] == 0:
                    fids = permute_fids([dims['t'], dims['coils'], dims['averages'], dims['extras']])
                    dims.update(t=1, coils=2, averages=3, subSpecs=0, extras=4)
                elif dims['averages'] == 0:
                    fids = permute_fids([dims['t'], dims['coils'], dims['subSpecs'], dims['extras']])
                    dims.update(t=1, coils=2, averages=0, subSpecs=3, extras=4)
                elif dims['coils'] == 0:
                    fids = permute_fids([dims['t'], dims['averages'], dims['subSpecs'], dims['extras']])
                    dims.update(t=1, coils=0, averages=2, subSpecs=3, extras=4)

            elif len(sqzDims) == 3:
                if dims['extras'] == 0 and dims['subSpecs'] == 0:
                    fids = permute_fids([dims['t'], dims['coils'], dims['averages']])
                    dims.update(t=1, coils=2, averages=3, subSpecs=0, extras=0)
                elif dims['extras'] == 0 and dims['averages'] == 0:
                    fids = permute_fids([dims['t'], dims['coils'], dims['subSpecs']])
                    dims.update(t=1, coils=2, averages=0, subSpecs=3, extras=0)
                elif dims['extras'] == 0 and dims['coils'] == 0:
                    fids = permute_fids([dims['t'], dims['averages'], dims['subSpecs']])
                    dims.update(t=1, coils=0, averages=2, subSpecs=3, extras=0)

            elif len(sqzDims) == 2:
                if dims['extras'] == 0 and dims['subSpecs'] == 0 and dims['averages'] == 0:
                    fids = permute_fids([dims['t'], dims['coils']])
                    dims.update(t=1, coils=2, averages=0, subSpecs=0, extras=0)
                elif dims['extras'] == 0 and dims['subSpecs'] == 0 and dims['coils'] == 0:
                    fids = permute_fids([dims['t'], dims['averages']])
                    dims.update(t=1, coils=0, averages=2, subSpecs=0, extras=0)
                elif dims['extras'] == 0 and dims['averages'] == 0 and dims['coils'] == 0:
                    fids = permute_fids([dims['t'], dims['subSpecs']])
                    dims.update(t=1, coils=0, averages=0, subSpecs=2, extras=0)

            elif len(sqzDims) == 1:
                dims.update(t=1, coils=0, averages=0, subSpecs=0, extras=0)

            # size of final data
            sz = fids.shape

            # conjugate
            fids = np.conj(fids)

            # FFT along time dimension (convert dims['t'] → numpy index)
            t_axis = dims['t'] - 1
            specs = np.fft.fftshift(np.fft.ifft(fids, axis=t_axis), axes=t_axis)

        else:
            raise ValueError("Data dimensions do not match expected single-voxel format."
                             "FID-A is primarily tailored towards single-voxel data. Maybe see "
                             "the csi_mod branch of the FID-A GitHub repository to modify this "
                             "code...")

        # calculate B0 from spectrometer frequency depending on nucleus
        nucleus = hdr_ext['ResonantNucleus'] if 'ResonantNucleus' in hdr_ext else '1H'
        gamma = {
            '1H': 42.577,
            '2H': 6.536,
            '3HE': -32.434,
            '7LI': 16.546,
            '13C': 10.708,
            '19F': 40.052,
            '23NA': 11.262,
            '31P': 17.235,
            '129XE': -11.777
        }
        Bo = [f0[i] / gamma.get(n, 42.577) for i, n in enumerate(nucleus)]

        # calculate t and ppm arrays using the calculated parameters
        f = [(-sw / 2) + (sw / (2 * n_pts)) + i * sw / n_pts for i in range(n_pts)]
        ppm = [(-f[i] / (Bo[0] * 42.577)) + 4.65 for i in range(n_pts)]  # convert to ppm
        t = [i * dt for i in range(n_pts)]

        # fill in additional FID-A format variables
        fida_struct = {
            'fids': fids,
            'specs': specs,
            'sz': sz,
            'dims': dims,
            'Bo': Bo,
            'averages': averages,
            'rawAverages': raw_averages,
            'subspecs': subspecs,
            'rawSubspecs': raw_subspecs,
            'te': hdr_ext['EchoTime'] if 'EchoTime' in hdr_ext else 0,
            'tr': hdr_ext['RepetitionTime'] if 'RepetitionTime' in hdr_ext else 0,
            't': t,
            'ppm': ppm,
            'spectralwidth': sw,
            'dwelltime': dt,
            'txfrq': [f * 1e6 for f in f0],  # convert to Hz
            'date': '',
            'nii_mrs': {
                'hdr': hdr,
                'hdr_ext': hdr_ext
            },
            'seq': hdr_ext['SequenceName'] if 'SequenceName' in hdr_ext else '',

            'flags': {
                'writtentostruct': 1,
                'gotparams': 1,
                'leftshifted': 0,
                'filtered': 0,
                'zeropadded': 0,
                'freqcorrected': 0,
                'phasecorrected': 0,
                'averaged': 1 if dims['averages'] == 0 else 0,
                'addedrcvrs': 1 if dims['coils'] == 0 else 0,
                'subtracted': 0,
                'writtentotext': 0,
                'downsampled': 0,
                'isFourSteps': 0 if dims['subSpecs'] == 0 else (
                    1 if sz[dims['subSpecs'] - 1] == 4 else 0)
            }
        }
        return fida_struct


#**************************************************************************************************#
#                                    Class FIDA_AlignAverages_fd                                   #
#**************************************************************************************************#
#                                                                                                  #
# This class implements the FID-A Align Averages transformation, from op_alignAverages_fd.m.       #
#                                                                                                  #
#**************************************************************************************************#
class FIDA_AlignAverages_fd(FIDA_Transform):
    """
    FID-A Align Averages transformation.

    Performs time-domain spectral registration using a limited range of frequencies to correct
    frequency and phase drifts.

    Attributes:
        verbose (bool): If True, print additional information during processing.
        octave (Oct2Py): Instance of Oct2Py to run MATLAB code.
    """

    def __init__(self, verbose: bool = False, minppm: float = 0.2, maxppm: float = 4.0,
                 tmax: float = 4.0, med: str = 'n') -> None:
        """
        Initialize the FIDA_AlignAverages_fd transformation.

        Args:
            verbose (bool): If True, print additional information during processing.
            minppm (float): Minimum frequency range in ppm for alignment.
            maxppm (float): Maximum frequency range in ppm for alignment.
            tmax (float): Maximum time in seconds to use for alignment.
            med (str): Method for alignment ('n', 'y', 'a', or 'r').
                        - 'n': Align to a single average with the lowest 'unlikeness'.
                        - 'y': Align to the median of the averages.
                        - 'a': Align to the average of the averages.
                        - 'r': Align to an externally provided reference spectrum.
        """
        super().__init__(verbose=verbose)
        self.minppm = minppm
        self.maxppm = maxppm
        self.tmax = tmax
        self.med = med

    def forward_single(self, data: dict, water: Optional[dict] = None, idx: int = 0) \
            -> Tuple[dict, Optional[dict]]:
        """
        Apply Align Averages to the provided MRS data using FID-A's op_alignAverages_fd.m.

        Args:
            data (dict): Input MRS data in FID-A struct format.
            water (Optional[dict]): Optional water reference data in FID-A struct format.
            idx (int): Index of the item in the list for logging purposes.

        Returns:
            Tuple[dict, Optional[dict]]: Transformed data and water reference (if applicable).
        """
        if self.med == 'r' and water is None:
            raise ValueError("FIDA_AlignAverages_fd requires a water reference for 'r' mode.")

        data, _, _ = self.octave.feval("op_alignAverages_fd", data, self.minppm, self.maxppm,
                                       self.tmax, self.med, water)

        if self.apply_to_water:
            water, _, _ = self.octave.feval("op_alignAverages_fd", water, self.minppm, self.maxppm,
                                            self.tmax, self.med, water)
        return data, water


#**************************************************************************************************#
#                                      Class FIDA_EddyCorrect                                      #
#**************************************************************************************************#
#                                                                                                  #
# This class implements the FID-A Eddy Current Correction transformation, from op_ecc_klose.m.     #
#                                                                                                  #
# TODO: Check!                                                                                     #
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

    def forward_single(self, data: dict, water: Optional[dict] = None, idx: int = 0) \
            -> List[dict]:
        """
        Apply Eddy Current Correction to the provided MRS data using FID-A's op_ecc_klose.m.

        Args:
            data (dict): Input MRS data in FID-A struct format.
            water (Optional[dict]): Optional water reference data in FID-A struct format.
            idx (int): Index of the item in the list for logging purposes.

        Returns:
            Tuple[dict, Optional[dict]]: Transformed data and water reference (if applicable).
        """
        if water is None:
            raise ValueError("FIDA_EddyCorrect requires a water reference.")

        return self.octave.feval("op_ecc_klose", data, water)