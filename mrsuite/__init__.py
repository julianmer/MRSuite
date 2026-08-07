"""MRSuite — MRS processing, analysis, AI model integration, and benchmarking.

Nothing is re-exported here on purpose. Importing the transform subpackages pulls in
Octave (via oct2py) and FSL-MRS, which is slow and fails on machines that only need
part of the toolbox, so submodules are imported explicitly by the caller.
"""

__version__ = "0.1.0a0"
