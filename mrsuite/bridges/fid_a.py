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
import inspect
import keyword
import os
import re
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs.nifti_mrs import NIFTI_MRS
from oct2py import Oct2Py

from mrsuite import externals
from mrsuite.bridges.base import Bridge, defaults
from mrsuite.core.provenance import git_version
from mrsuite.octave import resolve_runtime, session

# Absolute, so the bridge works from any working directory.
FIDA_ROOT = externals.path("fid-a")


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
    external = "fid-a"

    #**********#
    #   init   #
    #**********#
    def __init__(self, fida_root: Optional[str] = None, verbose: bool = False,
                 runtime: Optional[str] = None):
        super().__init__()
        self._fida_root = fida_root
        self._verbose = verbose
        self._runtime = runtime
        self._octave: Optional[Oct2Py] = None
        self._octave_version: Optional[str] = None
        self._codec = None
        self.nout = 1

    #**********#
    #   root   #
    #**********#
    @property
    def _root(self) -> str:
        """The toolbox's folder, fetched on first use when MRSuite is installed."""
        return str(self._fida_root or externals.ensure(self.external))

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

    #***********#
    #   about   #
    #***********#
    def about(self, func: str) -> Tuple[Optional[inspect.Signature], str]:
        """What a function's .m file says of it: its arguments, from the function line, and
        the comment block beside it. Read from the file, so Octave need not start."""
        path = _files(self._root).get(func)
        if path is None:                                 # an Octave built-in
            return None, f"Octave's {func}, dispatched by MRSuite."
        return _m_file(path, self.nout)

    #*************#
    #   resolve   #
    #*************#
    def _resolve(self, func: str) -> Callable:
        root = self._root               # fetched here, so a failed fetch says so, once
        if not self.exists(func):
            raise AttributeError(
                f"FID-A has no function '{func}' on the Octave path ({root}).")

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

        An operation over coils, averages or sub-spectra has nothing to do on data that
        no longer has them, and FID-A (or Osprey's copy) fails there rather than say so:
        op_alignAverages_fd indexes in.sz(in.dims.averages), so an absent averages
        dimension becomes in.sz(0) and Octave raises on the zero subscript;
        op_combinesubspecs sums over dimension 0; Osprey's op_addrcvrs and op_averaging
        abort where FID-A's return the data. So decline instead.
        """
        over, doing = next((v for k, v in OVER.items() if func.startswith(k)), ((), None))
        data = args[0] if args else None
        tags = list(getattr(data, "dim_tags", []) or [])
        found = next((t for t in over if t in tags), None)
        if not over:
            return None
        if found is None:
            return f"no {' or '.join(over)}: nothing to {doing}"
        n = data.shape[data.dim_position(found)]
        if n < 2:
            return f"{found} has length {n}: nothing to {doing}"
        return None

    #**************#
    #   declined   #
    #**************#
    def _declined(self, func, data, kwargs):
        """The data, and nothing for each further output asked for (``nout``)."""
        nout = kwargs.get("nout", self.nout)
        return data if nout == 1 else (data, *[None] * (nout - 1))

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
        return sorted(_files(self._root))

    #***********#
    #   steps   #
    #***********#
    def steps(self) -> List[str]:
        """The op_ functions, FID-A's operations on a spectrum's struct."""
        return [name for name in self.available() if name.startswith("op_")]

    #*************#
    #   folders   #
    #*************#
    def folders(self) -> Dict[str, str]:
        """The toolbox's top folder each function lives in, e.g. 'processingTools'; '' at
        the root."""
        root = os.path.abspath(self._root)
        return {name: os.path.relpath(path, root).split(os.sep)[0].replace(".m", "")
                if os.path.dirname(os.path.abspath(path)) != root else ""
                for name, path in _files(root).items()}


bridge = FIDABridge()


# the operations over one dimension of the data (as the codec maps FID-A's), and what they do
SUBSPECS = ("DIM_EDIT", "DIM_ISIS")
OVER = {"op_addrcvrs": (("DIM_COIL",), "combine"), "op_averaging": (("DIM_DYN",), "average"),
        "op_rmbadaverages": (("DIM_DYN",), "remove"), "op_alignAverages": (("DIM_DYN",), "align"),
        "op_combinesubspecs": (SUBSPECS, "combine"), "op_alignMPSubspecs": (SUBSPECS, "align")}


#***********#
#   files   #
#***********#
@lru_cache(maxsize=None)
def _files(root: str) -> Dict[str, str]:
    """Every .m file under a toolbox by function name, the first found where two share one."""
    found = {}
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(files):
            if f.endswith(".m"):
                found.setdefault(f[:-2], os.path.join(dirpath, f))
    return found


#*******************#
#   function line   #
#*******************#
FUNCTION = re.compile(r"^\s*function\b\s*(?:\[?([\w\s,~.]*?)\]?\s*=\s*)?(\w+)\s*(?:\(([^)]*)\))?")


#************#
#   m file   #
#************#
def _m_file(path: str, nout: int) -> Tuple[Optional[inspect.Signature], str]:
    """A .m file's arguments, positional and the data first as Octave takes them, with
    ``nout`` for how many outputs to ask for, from the function line named after the file
    (a script's own functions and a class's methods are not it); and its help, the file's
    first comment block, as Octave's help reads it."""
    with open(path, errors="replace") as fh:
        lines = fh.read().splitlines()
    stem = os.path.splitext(os.path.basename(path))[0]
    found = None
    for i, text in enumerate(lines):
        if not re.match(r"\s*function\b", text):
            continue
        declared, k = _code(text), i                     # joined over its ... continuations
        while declared.endswith("...") and k + 1 < len(lines):
            k += 1
            declared = declared[:-3] + " " + _code(lines[k])
        match = FUNCTION.match(declared)
        if match and match.group(2) == stem:
            found = match
            break
    block = "\n".join(_comments(lines))
    if found is None:                                    # a script, or a class
        return None, block or f"{os.path.basename(path)}, without help."

    names = [n.strip() for n in (found.group(3) or "").split(",") if n.strip()]
    given = _optional(lines[i + 1:], names)
    parameters = []
    try:
        for k, name in enumerate(names):
            if name == "varargin":
                parameters.append(inspect.Parameter("varargin",
                                                    inspect.Parameter.VAR_POSITIONAL))
                continue
            if name == "~":                              # an argument the function ignores
                name = f"unused{k}"
            elif not name.isidentifier() or keyword.iskeyword(name):    # e.g. FID-A's 'in'
                given = {(name + "_" if n == name else n): v for n, v in given.items()}
                name += "_"
            parameters.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY))
        parameters.append(inspect.Parameter("nout", inspect.Parameter.KEYWORD_ONLY,
                                            default=nout))
        signature = defaults(inspect.Signature(parameters), given)
    except ValueError:                       # a declaration not read: told, the call still works
        signature = None
    return signature, block or f"{os.path.basename(path)}, without help."


#**************************************************************************************************#
#                                          Class Computed                                          #
#**************************************************************************************************#
#                                                                                                  #
# A default a function works out for itself, so a call leaves the argument out.                    #
#                                                                                                  #
#**************************************************************************************************#
class Computed:
    """A default a function works out for itself, so a call leaves the argument out."""

    def __repr__(self) -> str:
        return "set by the toolbox"


#**************#
#   optional   #
#**************#
NARGIN = re.compile(r"\bnargin\s*(<=|<|==)\s*(\d+)")
EXIST = re.compile(r"~\s*exist\(\s*'(\w+)'|\bexist\(\s*'(\w+)'\s*(?:,\s*'var'\s*)?\)\s*==\s*0")


def _optional(body: List[str], names: List[str]) -> Dict[str, Any]:
    """
    The arguments a function's body gives a value when they are not passed, the way
    FID-A and Osprey do it: the first assignment to an argument, under ``if nargin<k``
    (or ``<=``) that fires whenever the argument is missing (k at least its place), or
    under ``if ~exist('x','var')``. A plain value (number, text, logical, row of numbers)
    right under the guard is its default; one worked out, or chosen in a branch below the
    guard (a switch, an if on the data), is Computed; one asked for interactively
    (``input(``) stays required, since nobody is there to answer.

    Args:
        body: The lines after the function line.
        names: The arguments, in order.
    """
    out: Dict[str, Any] = {}
    decided, depth, guards = set(), 0, []               # guards: (depth, line) still open
    for text in map(_code, body):
        if re.match(r"function\b", text):              # the file's next function
            break
        if re.match(r"(if|for|parfor|while|switch|try)\b", text):
            if re.search(r"\bend\s*;?$", text):          # all on one line: opens nothing
                continue
            depth += 1
            if NARGIN.search(text) or EXIST.search(text):
                guards.append((depth, text))
            continue
        if re.match(r"(else|elseif|end)\b", text):     # a guard holds only up to these
            guards = [g for g in guards if g[0] < depth]
            depth -= text.startswith("end")
            continue
        assigned = re.match(r"(\w+)\s*=(?!=)\s*(.+?);?$", text)
        name = assigned and assigned.group(1)
        if name not in names or name in decided:
            continue
        decided.add(name)
        at, guard = guards[-1] if guards else (0, "")
        place = names.index(name) + 1
        # the most arguments it fires for (k-1 for <, k for <=) covers every call without it
        fires = any(int(k) - (how == "<") >= place - 1
                    for how, k in NARGIN.findall(guard) if how != "==") \
            or name in {a or b for a, b in EXIST.findall(guard)}
        if fires and "input(" not in assigned.group(2):
            out[name] = _matlab(assigned.group(2)) if at == depth else Computed()
    return out


#************#
#   matlab   #
#************#
NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _matlab(text: str) -> Any:
    """A MATLAB value written plainly, as Python, else Computed."""
    text = text.strip()
    if text in ("true", "false"):
        return text == "true"
    if re.fullmatch(NUMBER, text):
        return int(text) if re.fullmatch(r"[-+]?\d+", text) else float(text)
    if re.fullmatch(r"'(?:[^']|'')*'", text):
        return text[1:-1].replace("''", "'")
    if re.fullmatch(r'"[^"]*"', text):
        return text[1:-1]
    row = re.fullmatch(rf"\[\s*({NUMBER}(?:\s*[,\s]\s*{NUMBER})*)\s*\]", text)
    if row:
        return [_matlab(v) for v in re.split(r"[,\s]+", row.group(1).strip())]  # not a cell
    return Computed()


#**********#
#   code   #
#**********#
def _code(line: str) -> str:
    """A line without its comment."""
    return line.split("%", 1)[0].strip()


#**************#
#   comments   #
#**************#
def _comments(lines: List[str]) -> List[str]:
    """A file's first comment block, without its comment signs: before the function line,
    as FID-A writes it, or after it, as Osprey does."""
    block, started = [], False
    for text in lines:
        stripped = text.strip()
        if stripped.startswith("%"):
            block.append(stripped.lstrip("%").rstrip())
            started = True
        elif started:
            break
    return block


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
