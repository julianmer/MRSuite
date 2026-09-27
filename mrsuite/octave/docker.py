####################################################################################################
#                                            docker.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Octave in a container, for machines without one installed. The container is a           #
#          drop-in for the local interpreter rather than a separate execution model: the           #
#          host tree is bind-mounted at its own paths, so oct2py's live session, the .mat          #
#          files it exchanges and every toolbox path work exactly as they do locally.              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

IMAGE = "mrsuite-octave"
CONTAINER = os.environ.get("MRSUITE_OCTAVE_CONTAINER", "mrsuite-octave")
DOCKERFILE = HERE / "dockerfile"      # it sits with the shims it serves

# oct2py exchanges data as .mat files whose absolute paths it embeds in the commands it
# sends, so the container has to see them at those same paths. Keeping the scratch inside
# the repo means the single repo mount covers it.
TEMP_DIR = REPO / ".octave-tmp"


#*************#
#   command   #
#*************#
def _docker(*args: str, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    proc = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed:\n{proc.stderr[-2000:]}")
    return proc


#***************#
#   available   #
#***************#
def available() -> bool:
    """Whether a Docker daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    return _docker("info", check=False, timeout=30).returncode == 0


#***********#
#   image   #
#***********#
def build(force: bool = False, image: str = IMAGE, dockerfile: Path = DOCKERFILE) -> str:
    """Build an image if it is not already present: Octave's, unless told otherwise.

    Args:
        force: Rebuild even when the image exists.
        image: The tag to build.
        dockerfile: Its dockerfile; the folder it sits in is the build context.

    Returns:
        The image tag.
    """
    have = _docker("image", "inspect", image, check=False, timeout=60).returncode == 0
    if have and not force:
        return image

    # named explicitly: docker build looks for a capitalised Dockerfile by default
    _docker("build", "-t", image, "-f", str(dockerfile), str(Path(dockerfile).parent),
            timeout=3600)
    return image


#***********#
#   state   #
#***********#
def _state() -> Optional[str]:
    """The container's state, or None if it does not exist."""
    proc = _docker("inspect", "-f", "{{.State.Status}}", CONTAINER, check=False, timeout=60)
    return proc.stdout.strip() if proc.returncode == 0 else None


#************#
#   mounts   #
#************#
def _mounts() -> List[str]:
    """Host paths the container must see, each at its own absolute path.

    The repo covers the toolboxes, the shims, the data and oct2py's scratch. oct2py also
    adds its own package directory to the Octave path, so that has to be visible too.
    """
    import oct2py

    paths = [str(REPO), str(Path(oct2py.__file__).resolve().parent)]
    return [arg for p in paths for arg in ("-v", f"{p}:{p}")]


#****************#
#   is current   #
#****************#
def _is_current() -> bool:
    """Whether the existing container runs the current image and sees every path this
    environment needs."""
    proc = _docker("inspect", "-f",
                   "{{.Image}} {{range .Mounts}}{{.Source}}:{{.Destination}} {{end}}",
                   CONTAINER, check=False, timeout=60)
    have = proc.stdout.split()
    return bool(have) and have[0] == image_id() and set(_mounts()[1::2]) <= set(have[1:])


#**************#
#   image id   #
#**************#
def image_id() -> str:
    """The current image's id, or '' when it has not been built."""
    proc = _docker("image", "inspect", "-f", "{{.Id}}", IMAGE, check=False, timeout=60)
    return proc.stdout.strip()


#***********#
#   start   #
#***********#
def start() -> str:
    """Start the Octave container, building the image first if needed.

    Idempotent: an already-running container is left alone. Returns the container name.
    """
    state = _state()
    if state is not None and not _is_current():
        # made from another environment, whose oct2py is elsewhere, or from an image
        # since rebuilt: either way it is not the runtime this session asks for
        stop(remove=True)
        state = None
    if state == "running":
        return CONTAINER
    if state is not None:
        _docker("start", CONTAINER, timeout=120)
        return CONTAINER

    build()
    TEMP_DIR.mkdir(exist_ok=True)
    # run as the host user, so the .mat files octave writes stay readable outside the
    # container; a bare uid has no passwd entry, hence the explicit HOME
    _docker("run", "-d", "--name", CONTAINER,
            "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
            *_mounts(), "-w", str(REPO), IMAGE, timeout=300)
    return CONTAINER


#**********#
#   stop   #
#**********#
def stop(remove: bool = False) -> None:
    """Stop the container, optionally removing it."""
    if _state() is None:
        return
    _docker("stop", CONTAINER, timeout=120)
    if remove:
        _docker("rm", CONTAINER, timeout=120)


#****************#
#   executable   #
#****************#
def executable() -> str:
    """Write the launcher oct2py should start instead of a local octave-cli.

    oct2py will not take a command with arguments: it resolves OCTAVE_EXECUTABLE with
    shutil.which() and rejects a name without "octave" in it, so the docker invocation
    has to be behind a path. Generated rather than shipped, since it is a build artefact
    of the chosen container, not source.
    """
    TEMP_DIR.mkdir(exist_ok=True)
    path = TEMP_DIR / "octave-docker"
    path.write_text(f'#!/bin/sh\nexec docker exec -i "{CONTAINER}" octave-cli "$@"\n')
    path.chmod(0o755)
    return str(path)
