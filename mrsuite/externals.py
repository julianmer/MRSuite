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
#          The datasets it downloads, and what it builds and scratches, lie by the same rule.      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import contextlib
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterator, List, Optional

# name -> (git url, commit), as the checkout's submodules pin them (a test holds them equal)
REGISTRY = {
    "agnostic": ("https://github.com/agudmundson/agnostic.git",
                 "40294d7c0cd25a16186795a1636ff2a1e4f94ed1"),
    "basic": ("https://github.com/julianmer/BasiC.git",
              "e53037b3f0f59eaa812c02ae365e156b3d16e9ab"),
    "fid-a": ("https://github.com/CIC-methods/FID-A.git",
              "1eaa2075625745beb7632f608dc2947b73295309"),
    "gannet": ("https://github.com/markmikkelsen/Gannet.git",
               "f437bb77224d6fd5a02489670dee461a54e30e34"),
    "mrs-digital-phantom": ("https://github.com/dennisvds/MRS-Digital-Phantom.git",
                            "acf2dcb2015a2d27560295157774fdde8ef7d9a2"),
    "mrs-sim": ("https://github.com/JohnLaMaster/MRS-Sim.git",
                "dffb5ed7dd8ada1070cb931a63e3fa8b6c1de875"),
    "mrs4brain": ("https://github.com/MRS4Brain/MRS4Brain-toolbox.git",
                  "364e12e39b79b0a93bc22f89929d717ab24352d3"),
    "mrspeclab": ("https://github.com/MRSEPFL/MRSpecLAB.git",
                  "6e7166496bdea9e42f3a00aa360cdd71f0a18b58"),
    "ood-robust-mrs": ("https://github.com/julianmer/OoD-Robust-MRS-Quantification.git",
                       "451cf711d3ae4a1f94d531f3a2fbb4d7ceabeabb"),
    "osprey": ("https://github.com/schorschinho/osprey.git",
               "98b2e354f633211e5ff79ab6d263edde674a2e58"),
    "pygamma": ("https://github.com/pygamma-mrs/gamma.git",
                "4707348a9b26c9bcf99aa22d04c2e8ccbe5bf05c"),
    "simntg": ("https://github.com/mrshub/SimnTG.git",
               "98444f6db495a3e17d3d179e8f36ee1fe0c7b345"),
    "spinach": ("https://github.com/IlyaKuprov/Spinach.git",
                "4138eb5fc2414a6b0f510d1e34103189599d1a58"),
    "spm12": ("https://github.com/spm/spm12.git",
              "03ac9473cad402407b6472228377c0167fdc54b8"),
    "wand-for-mrs": ("https://github.com/julianmer/WAND-for-MRS.git",
                     "2642659ccf8f5ca6376b44a351dd9420a846b1f0"),
}

CHECKOUT = Path(__file__).resolve().parents[1]

_FETCHING: List[str] = []       # what is downloading now, said in the GUI's status
_UMASK = os.umask(0o022)        # read once, so what is fetched is as readable as any file made
os.umask(_UMASK)


#**********#
#   home   #
#**********#
def home() -> Path:
    """MRSuite's own folder of the user's: $MRSUITE_HOME, else ~/.mrsuite."""
    return Path(os.environ.get("MRSUITE_HOME", Path.home() / ".mrsuite")).expanduser().absolute()


#**************#
#   checkout   #
#**************#
def checkout() -> bool:
    """Whether MRSuite runs from a source checkout, which has the submodules, rather than
    as installed."""
    return (CHECKOUT / ".gitmodules").is_file()


#***********#
#   local   #
#***********#
def local(name: str) -> Path:
    """A folder of MRSuite's own, for what it fetches, builds or scratches: the checkout's,
    else the user's, so an installed MRSuite never writes into its own package."""
    return (CHECKOUT if checkout() else home()) / name


#**********#
#   root   #
#**********#
def root() -> Path:
    """The folder the externals lie in: the checkout's externals/, else the user's."""
    return local("externals")


#**********#
#   path   #
#**********#
def path(name: str) -> Path:
    """Where an external lies, or will once fetched: the checkout's submodule, else a folder
    per pinned commit, so an upgraded MRSuite runs what it pins, not what an older one fetched."""
    if checkout():
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
    if checkout():
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
        move_in(part, dest)
    finally:
        shutil.rmtree(part, ignore_errors=True)
    return dest


#*************#
#   move in   #
#*************#
def move_in(part: Path, dest: Path) -> None:
    """A folder made whole beside its place, moved into it; where another process moved its
    own in first, that one stands and ``part`` is left to the caller to remove."""
    part.chmod(0o777 & ~_UMASK)                  # a temporary folder is its owner's alone
    try:
        part.rename(dest)
    except OSError:
        if not (dest.is_dir() and any(dest.iterdir())):
            raise


#**************#
#   fetching   #
#**************#
@contextlib.contextmanager
def fetching(what: str, where: Path, say: bool = True) -> Iterator[None]:
    """
    Say what is downloading while it does: once on stderr, and in now() for the GUI.

    Args:
        what (str): What, as a person reads it, its size included.
        where (Path): The folder it goes into.
        say (bool): Whether to say it on stderr too; not where the fetch says it already.
    """
    message = f"Downloading {what} into {where}…"
    if say:
        print(message, file=sys.stderr, flush=True)
    _FETCHING.append(message)
    try:
        yield
    finally:
        _FETCHING.remove(message)


#*********#
#   now   #
#*********#
def now() -> Optional[str]:
    """What is downloading now, as fetching says it, or None."""
    return _FETCHING[-1] if _FETCHING else None


#**************#
#   download   #
#**************#
def download(url: str, dest: Path, size: Optional[int] = None,
             md5: Optional[str] = None) -> Path:
    """
    One file, written beside its place and moved in once it is whole and checked, so an
    interrupted or corrupted download never passes for the file.

    Args:
        url (str): Where it is served.
        dest (Path): Where it goes.
        size (Optional[int]): Its published size in bytes, checked.
        md5 (Optional[str]): Its published MD5, checked.

    Returns:
        Path: dest.

    Raises:
        RuntimeError: If what arrived is not the size or checksum published.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5()
    with tempfile.NamedTemporaryFile(prefix=".part", dir=dest.parent, delete=False) as part:
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                while chunk := response.read(1 << 20):
                    part.write(chunk)
                    digest.update(chunk)
            part.close()
            got = os.path.getsize(part.name)
            if size is not None and got != size:
                raise RuntimeError(f"{url} gave {got} bytes, not the {size} published")
            if md5 and digest.hexdigest() != md5:
                raise RuntimeError(f"{url} does not match its published MD5 {md5}")
            os.chmod(part.name, 0o666 & ~_UMASK)
            os.replace(part.name, dest)
        finally:
            Path(part.name).unlink(missing_ok=True)
    return dest
