####################################################################################################
#                                             fid_a.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Bridge onto FID-A's Octave function surface, plus the NIfTI-MRS <-> FID-A               #
#          reachable, since oct2py resolves functions dynamically; what the bridge adds            #
#          is the conversion at the boundary, batching over NIfTI-MRS+, and provenance.            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs.nifti_mrs import NIFTI_MRS
from oct2py import Oct2Py

from mrsuite.bridges.base import Bridge
from mrsuite.core.provenance import git_version
from mrsuite.octave import resolve_runtime, session

# Absolute, so the bridge works from any working directory.
FIDA_ROOT = Path(__file__).resolve().parents[2] / "externals" / "fid-a"


#**************************************************************************************************#
#                                         Class FIDACodec                                          #
#**************************************************************************************************#
#                                                                                                  #
# Converts NIfTI-MRS to FID-A structs and back again.                                              #
#                                                                                                  #
#**************************************************************************************************#
class FIDACodec:
    """Convert NIfTI-MRS to FID-A structs and back."""

    #****************#
    #   nifti2fida   #
    #****************#
    def nifti2fida(self, data: NIFTI_MRS, centre_ppm: Optional[float] = None) -> dict:
        """
        Convert NIfTI_MRS data to FID-A struct format.

        Adapted from Georg Oeltzschner's io_loadspec_niimrs.m.

        Args:
            data (NIfTI_MRS): Input MRS data to be converted.
            centre_ppm (Optional[float]): ppm at the centre of the spectrum. Defaults to
                what the data states as SpecFreqChemShift, which NIfTI-MRS takes as 4.65
                for 1H when unstated, rather than to a constant: FID-A's own loaders
                disagree (4.68 in io_loadspec_niimrs, 4.7 in io_loadspec_twix), and the
                data knows better than either.

        Returns:
            dict: FID-A struct representation of the input data.
        """
        if centre_ppm is None:
            centre_ppm = data.SpecFreqChemShift
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

            # np.squeeze drops every singleton axis, not only x, y and z. A non-spatial
            # dimension of length one — a single coil, a single average — goes with them,
            # so simply subtracting 3 would leave dims pointing at axes the array no
            # longer has and the permutation below would raise. Renumber against the axes
            # that actually survive and zero the ones that do not; that also matches the
            # FID-A convention, where an absent dimension is 0 rather than a size-1 axis.
            survivors = {old: new + 1 for new, old in
                         enumerate(i for i in range(fids.ndim) if fids.shape[i] > 1)}
            fids = np.squeeze(fids)

            sqzDims = []
            for key, val in dims.items():
                if val != 0:
                    dims[key] = survivors.get(val - 1, 0)  # MATLAB-style; 0 means absent
                    if dims[key] != 0:
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

            # Octave has no 1-D arrays: a numpy (N,) arrives as a 1xN row, while FID-A
            # indexes the time axis as dimension 1 and reads such data as a single
            # time point across N columns. Make the column orientation explicit so
            # every FID-A routine sees the axis it expects.
            if fids.ndim == 1:
                fids = fids.reshape(-1, 1)
                specs = specs.reshape(-1, 1)
                sz = fids.shape

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
        ppm = [(-f[i] / (Bo[0] * 42.577)) + centre_ppm for i in range(n_pts)]
        t = [i * dt for i in range(n_pts)]

        # fill in additional FID-A format variables
        fida_struct = {
            'fids': fids,
            'specs': specs,
            # ndarray, not the shape tuple: oct2py maps a Python tuple to an Octave
            # cell array, and FID-A does arithmetic on sz (in.sz(1)*zpFactor), which
            # fails on a cell with "binary operator '*' not implemented".
            'sz': np.asarray(sz, dtype=float),
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
            # the ppm axis alone is not enough: op_zeropad and Osprey's fit rebuild the
            # axis from centerFreq, and would silently shift the spectrum without it
            'centerFreq': centre_ppm,
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

    # dimension tags map onto FID-A's fixed slots; kept beside nifti2fida's copy so the
    # forward and reverse conversions cannot drift apart
    _TAG_TO_FIDA = {
        'DIM_COIL': 'coils',
        'DIM_DYN': 'averages',
        'DIM_EDIT': 'subSpecs',
        'DIM_ISIS': 'subSpecs',
    }

    #****************#
    #   fida2nifti   #
    #****************#
    def fida2nifti(self, fida: dict) -> NIFTI_MRS:
        """
        Convert a FID-A struct back to NIfTI-MRS.

        The inverse of :meth:`nifti2fida`. Two things make it reconstructible: the
        forward pass stashes the source header under ``nii_mrs``, and FID-A keeps the
        dimension bookkeeping in ``dims`` up to date as operations add or remove axes.

        A FID-A operation may legitimately change the shape - averaging collapses the
        averages axis, coil combination collapses coils - so the surviving axes are read
        back out of ``dims`` rather than assumed. Tags for axes that no longer exist are
        dropped, and the remaining ones keep their original relative order so dim_5
        stays ahead of dim_6.

        Args:
            fida (dict): MRS data in FID-A struct format, as produced by nifti2fida and
                possibly modified by FID-A operations.

        Returns:
            NIFTI_MRS: The data in NIfTI-MRS form.

        Raises:
            KeyError: If the struct carries no ``nii_mrs`` provenance to rebuild from.
        """
        if 'nii_mrs' not in fida:
            raise KeyError("FID-A struct has no 'nii_mrs' header provenance; it did not "
                           "come from nifti2fida and cannot be converted back.")

        # hdr_ext is an Hdr_Ext object on the way in, but a mapping once the struct has
        # made a round trip through Octave. The dict check has to come first: oct2py's
        # Struct is a dict subclass whose attribute access *fabricates* an entry, so
        # hasattr(raw_ext, 'to_dict') is true for it and returns a Struct, not a method.
        raw_ext = fida['nii_mrs']['hdr_ext']
        if isinstance(raw_ext, dict):
            hdr_ext = dict(raw_ext)
        elif callable(getattr(raw_ext, 'to_dict', None)):
            hdr_ext = dict(raw_ext.to_dict())
        else:
            hdr_ext = dict(raw_ext)

        def as_int(v):
            return int(np.asarray(v).ravel()[0])

        dims = {k: as_int(v) for k, v in dict(fida['dims']).items()}

        # nifti2fida conjugates on the way in; undo that here
        fids = np.conj(np.asarray(fida['fids']))

        # after the forward permute the axes sit in ascending `dims` order
        order = [name for _, name in
                 sorted((pos, name) for name, pos in dims.items()
                        if name in ('t', 'coils', 'averages', 'subSpecs', 'extras')
                        and pos > 0)]
        fids = fids.reshape([fids.shape[i] if i < fids.ndim else 1
                             for i in range(len(order))])

        # which original tag fed each FID-A slot, in the order the header declared them
        surviving, tags, singleton = [], [], []
        for n in (5, 6, 7):
            tag = hdr_ext.get(f'dim_{n}')
            if tag is None:
                continue
            slot = self._TAG_TO_FIDA.get(tag, 'extras')
            if dims.get(slot, 0) > 0 and slot in order and slot not in surviving:
                surviving.append(slot)
                tags.append(tag)
            elif dims.get(slot, 0) == 0:
                # FID-A marks an absent dimension with 0 and squeezes length-1 axes, so
                # a single coil or a single average disappears from its bookkeeping
                # entirely. The axis carried no data, but dropping the tag would still
                # change the header, so it is restored as a singleton. A dimension that
                # an operation genuinely collapsed (averaging 32 -> 1) is not restored,
                # because its slot is only cleared once FID-A has consumed it.
                tags.append(tag)
                singleton.append(len(tags) - 1)

        target = ['t'] + surviving
        fids = np.transpose(fids, [order.index(name) for name in target])
        fids = fids.reshape((1, 1, 1) + fids.shape)      # restore x, y, z
        for pos in singleton:                            # re-insert squeezed length-1 axes
            fids = np.expand_dims(fids, axis=4 + pos)

        for n in (5, 6, 7):                              # rewrite the tag list
            hdr_ext.pop(f'dim_{n}', None)
        for i, tag in enumerate(tags):
            hdr_ext[f'dim_{5 + i}'] = tag

        f0 = np.asarray(fida['txfrq']).ravel()[0] / 1e6  # back to MHz
        # a list from Python, an array from an Octave cell (Osprey hands it back as one)
        nucleus = str(np.asarray(hdr_ext.get('ResonantNucleus', ['1H'])).ravel()[0])

        return gen_nifti_mrs(
            fids,
            float(np.asarray(fida['dwelltime']).ravel()[0]),
            float(f0),
            nucleus=nucleus,
            dim_tags=list(tags) + [None] * (3 - len(tags)),
            # no_conj is left at its default. Its docstring says True "stops
            # conjugation", but measured against nifti-mrs the flag behaves the other
            # way round: gen_nifti_mrs(1+2j, no_conj=True) returns 1-2j, while
            # no_conj=False returns the data untouched, which is what is wanted here
            # since the conjugation nifti2fida applied has already been undone above.
            no_conj=False)


#**************************************************************************************************#
#                                         Class FIDABridge                                         #
#**************************************************************************************************#
#                                                                                                  #
# Dispatches onto FID-A, converting NIfTI-MRS at the boundary.                                     #
#                                                                                                  #
#**************************************************************************************************#
class FIDABridge(Bridge):
    """Dispatch onto FID-A, converting NIfTI-MRS at the boundary.

    Attributes:
        nout (int): Default number of outputs to request from Octave.
    """

    name = "fid_a"
    program = "FID-A"

    #**********#
    #   init   #
    #**********#
    def __init__(self, fida_root: Optional[str] = None, verbose: bool = False,
                 runtime: Optional[str] = None):
        super().__init__()
        self._root = str(fida_root or FIDA_ROOT)
        self._verbose = verbose
        self._runtime = runtime
        self._octave: Optional[Oct2Py] = None
        self._octave_version: Optional[str] = None
        self._codec = None
        self.nout = 1

    #*************#
    #   version   #
    #*************#
    def version(self) -> Optional[str]:
        """The toolbox checkout's exact state, e.g. 'V1.2-357-g1eaa207'."""
        return git_version(self._root)

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, Any]:
        """Which interpreter ran the call, where, and at which version."""
        which = resolve_runtime(self._runtime)
        if self._octave_version is None:
            name = "MATLAB" if which == "matlab" else "Octave"
            self._octave_version = f"{name} {str(self.octave.eval('version()', nout=1)).strip()}"
        return {"runtime": which, "interpreter": self._octave_version}

    #************#
    #   octave   #
    #************#
    @property
    def octave(self) -> Oct2Py:
        """Octave session, started on first use.

        Deferred so that importing MRSuite does not start Octave; only code that
        actually calls FID-A pays for it.
        """
        if self._octave is None:
            self._octave = session(self._root, runtime=self._runtime, verbose=self._verbose)
        return self._octave

    #***********#
    #   codec   #
    #***********#
    @property
    def codec(self) -> FIDACodec:
        """Converter for NIfTI-MRS <-> FID-A structs.

        Standalone, so converting costs nothing beyond numpy — starting Octave is
        deferred to the first call that actually needs it.
        """
        if self._codec is None:
            self._codec = FIDACodec()
        return self._codec

    #*************#
    #   resolve   #
    #*************#
    def _resolve(self, func: str) -> Callable:
        if not self.exists(func):
            raise AttributeError(
                f"FID-A has no function '{func}' on the Octave path ({self._root}).")

        def target(*args, **kwargs):
            nout = kwargs.pop("nout", self.nout)
            out = self.octave.feval(func, *args, nout=nout, **kwargs)
            # several outputs: oct2py hands back a list, MATLAB's runtime a tuple
            return tuple(out) if nout > 1 else out

        return target

    #****************#
    #   to toolbox   #
    #****************#
    def _skip_reason(self, func, args, kwargs):
        """Decline operations that are meaningless on degenerate input.

        FID-A does not guard alignment against single-average data:
        op_alignAverages_fd indexes in.sz(in.dims.averages), so an absent averages
        dimension becomes in.sz(0) and Octave raises on the zero subscript. There is
        nothing to align either way, so decline instead.
        """
        if "alignAverages" not in func:
            return None
        data = args[0] if args else None
        tags = list(getattr(data, "dim_tags", []) or [])
        if "DIM_DYN" not in tags:
            return "no DIM_DYN: nothing to align"
        n = data.shape[data.dim_position("DIM_DYN")]
        if n < 2:
            return f"DIM_DYN has length {n}: nothing to align"
        return None

    def _to_toolbox(self, obj: NIFTI_MRS):
        return self.codec.nifti2fida(obj)

    #******************#
    #   from toolbox   #
    #******************#
    def _from_toolbox(self, obj: Any):
        # a FID-A struct carries the provenance stashed by nifti2fida; anything else
        # (metrics, index vectors) is a plain result and passes straight through
        if isinstance(obj, dict) and "nii_mrs" in obj and "fids" in obj:
            return self.codec.fida2nifti(obj)
        return obj

    #************#
    #   exists   #
    #************#
    def exists(self, func: str) -> bool:
        """Whether Octave can resolve ``func`` as a file or built-in."""
        try:
            return int(self.octave.eval(f"exist('{func}')", nout=1)) != 0
        except Exception:                      # noqa: BLE001 - treat as absent
            return False

    #***************#
    #   available   #
    #***************#
    def available(self) -> List[str]:
        """Every ``.m`` FID-A ships, from the filesystem rather than Octave.

        Cheaper and more complete than interrogating the session, and it works before
        Octave has been started.
        """
        found = []
        for dirpath, _, files in os.walk(self._root):
            if os.sep + "." in dirpath:
                continue
            found += [f[:-2] for f in files if f.endswith(".m")]
        return sorted(set(found))


bridge = FIDABridge()


#*************#
#   getattr   #
#*************#
def __getattr__(name: str) -> Any:
    """Module-level dispatch, so `from mrsuite.bridges.fid_a import op_averaging` works."""
    return getattr(bridge, name)


#*********#
#   dir   #
#*********#
def __dir__() -> List[str]:
    return sorted(set(globals()) | set(bridge.available()))
