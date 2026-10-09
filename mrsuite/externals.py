####################################################################################################
#                                           externals.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: Where the other people's code MRSuite runs lies: BasiC, FID-A, Osprey, MRS-Sim,         #
#          MRSpecLAB and the repositories of the published models. None of it ships in the wheel;  #
#          it carries its own licences and is run as published. A source checkout has it as the    #
#          externals/ submodules. An installed MRSuite clones each on first use, at the commit     #
#          its submodule pins, into a folder of the user's ($MRSUITE_HOME, else ~/.mrsuite).       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# name -> (git url, commit), as the checkout's submodules pin them (a test holds them equal)
REGISTRY = {
    "basic": ("https://github.com/julianmer/BasiC.git",
              "07aec21485214e47ad947174bf6106e605bd049e"),
    "fid-a": ("https://github.com/CIC-methods/FID-A.git",
              "1eaa2075625745beb7632f608dc2947b73295309"),
    "mrs-sim": ("https://github.com/JohnLaMaster/MRS-Sim.git",
                "dffb5ed7dd8ada1070cb931a63e3fa8b6c1de875"),
    "mrspeclab": ("https://github.com/MRSEPFL/MRSpecLAB.git",
                  "6e7166496bdea9e42f3a00aa360cdd71f0a18b58"),
    "ood-robust-mrs": ("https://github.com/julianmer/OoD-Robust-MRS-Quantification.git",
                       "451cf711d3ae4a1f94d531f3a2fbb4d7ceabeabb"),
    "osprey": ("https://github.com/schorschinho/osprey.git",
               "98b2e354f633211e5ff79ab6d263edde674a2e58"),
    "wand-for-mrs": ("https://github.com/julianmer/WAND-for-MRS.git",
                     "2642659ccf8f5ca6376b44a351dd9420a846b1f0"),
}

CHECKOUT = Path(__file__).resolve().parents[1]


#**********#
#   home   #
#**********#
def home() -> Path:
    """MRSuite's own folder of the user's: $MRSUITE_HOME, else ~/.mrsuite."""
    return Path(os.environ.get("MRSUITE_HOME", Path.home() / ".mrsuite"))


#**********#
#   root   #
#**********#
def root() -> Path:
    """The folder the externals lie in: the checkout's externals/, else the user's."""
    if (CHECKOUT / ".gitmodules").is_file():
        return CHECKOUT / "externals"
    return home() / "externals"


#**********#
#   path   #
#**********#
def path(name: str) -> Path:
    """Where an external lies, or will once fetched: the checkout's submodule, else a folder
    per pinned commit, so an upgraded MRSuite runs what it pins, not what an older one fetched."""
    if (CHECKOUT / ".gitmodules").is_file():
        return root() / name
    return root() / name / REGISTRY[name][1][:12]


#************#
#   ensure   #
#************#
def ensure(name: str) -> Path:
    """
    An external's folder, cloned at its pinned commit first if it is not there yet.

    Args:
        name (str): A key of REGISTRY.

    Returns:
        Path: The external's folder.

    Raises:
        RuntimeError: In a checkout whose submodule is not initialised, or if the clone fails.
    """
    url, commit = REGISTRY[name]
    dest = path(name)
    if dest.is_dir() and any(dest.iterdir()):
        return dest
    if (CHECKOUT / ".gitmodules").is_file():
        raise RuntimeError(f"{dest} is empty: git -C {CHECKOUT} submodule update --init "
                           f"externals/{name}")

    # cloned beside its place, each call into its own folder, and moved in whole, so a failed
    # or concurrent clone never leaves a half-filled folder that would pass for the external;
    # large files in Git LFS are example data, not code, and are left on the server
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = Path(tempfile.mkdtemp(prefix=".part", dir=dest.parent))
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    try:
        for command in (["git", "clone", "--quiet", url, str(part)],
                        ["git", "-C", str(part), "checkout", "--quiet", commit]):
            run = subprocess.run(command, env=env, capture_output=True, text=True)
            if run.returncode != 0:
                raise RuntimeError(f"fetching {name} at {commit[:7]} from {url} failed:\n"
                                   f"{run.stderr.strip()}")
        try:
            part.rename(dest)
        except OSError:
            if not (dest.is_dir() and any(dest.iterdir())):   # not another process' clone
                raise
    finally:
        shutil.rmtree(part, ignore_errors=True)
    return dest
