####################################################################################################
#                                        test_runtimes.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Local Octave and the container must be interchangeable. Every case here runs on         #
#          each runtime that the machine has, and the last one runs the same call on both          #
#          and compares: a container that merely works is not enough if it disagrees.              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import shutil

import numpy as np
import pytest

from mrsuite.bridges.fid_a import FIDABridge
from mrsuite.octave import docker, resolve_runtime, session

HAVE_LOCAL = shutil.which("octave-cli") is not None
HAVE_DOCKER = docker.available()

RUNTIMES = ([("local", HAVE_LOCAL)] + [("docker", HAVE_DOCKER)])
runtimes = pytest.mark.parametrize(
    "runtime",
    [pytest.param(n, marks=pytest.mark.skipif(not ok, reason=f"{n} runtime unavailable"))
     for n, ok in RUNTIMES])


#*************#
#   fixture   #
#*************#
@pytest.fixture(scope="module")
def bridges():
    """One FID-A bridge per available runtime, kept alive across the module.

    Session-scoped by hand rather than per test: starting Octave, let alone a container,
    is far more expensive than any assertion here.
    """
    made = {}
    for name, ok in RUNTIMES:
        if ok:
            made[name] = FIDABridge(runtime=name)
    yield made
    for b in made.values():
        if b._octave is not None:
            b._octave.exit()


#*************#
#   resolve   #
#*************#
def test_resolve_rejects_an_unknown_runtime():
    with pytest.raises(ValueError, match="unknown runtime"):
        resolve_runtime("podman")


#**********#
#   auto   #
#**********#
@pytest.mark.skipif(not (HAVE_LOCAL or HAVE_DOCKER), reason="no Octave at all")
def test_auto_prefers_local_when_present():
    assert resolve_runtime("auto") == ("local" if HAVE_LOCAL else "docker")


#*************#
#   session   #
#*************#
@runtimes
def test_session_runs_octave(runtime):
    oc = session(str(docker.REPO), runtime=runtime)
    try:
        assert int(oc.eval("1 + 1", nout=1)) == 2
    finally:
        oc.exit()


#***********#
#   shims   #
#***********#
@runtimes
def test_sessions_keep_their_exchange_files_apart(runtime):
    """oct2py passes every call through writer.mat and reader.mat in its temp folder, so
    sessions sharing one, as parallel workers would, overwrite each other's data."""
    a = session(str(docker.REPO), runtime=runtime)
    b = session(str(docker.REPO), runtime=runtime)
    try:
        assert a.temp_dir != b.temp_dir
        assert int(a.eval("1 + 1", nout=1)) == int(b.eval("1 + 1", nout=1)) == 2
    finally:
        a.exit()
        b.exit()


@runtimes
def test_shims_are_on_the_path(runtime, bridges):
    """The shims are added last so they sit at the front, on either runtime.

    hilbert is the sharp case: the container ships the forge `signal` package, so this
    also pins that the shim still wins and the two runtimes take the same code path.
    """
    oc = bridges[runtime].octave

    assert int(oc.eval("exist('hilbert')", nout=1)) != 0
    assert "mrsuite" in str(oc.eval("which('hilbert')", nout=1))


#***********#
#   fid-a   #
#***********#
@runtimes
def test_fid_a_dispatches(runtime, bridges, spectrum):
    out = bridges[runtime].op_addphase(spectrum, 30.0)

    arr = np.asarray(out[:])
    assert arr.size > 0 and np.all(np.isfinite(arr))


#***********#
#   agree   #
#***********#
@pytest.mark.skipif(not (HAVE_LOCAL and HAVE_DOCKER), reason="needs both runtimes")
def test_the_two_runtimes_agree(bridges, spectrum):
    """The point of the whole exercise: a result must not depend on where Octave ran."""
    local = np.asarray(bridges["local"].op_addphase(spectrum, 30.0)[:])
    container = np.asarray(bridges["docker"].op_addphase(spectrum, 30.0)[:])

    assert local.shape == container.shape
    worst = np.max(np.abs(local - container))
    assert worst < 1e-10, f"runtimes disagree by {worst:.3e}"


#*****************#
#   spm12 build   #
#*****************#
def test_spm12_is_built_again_for_a_runtime_it_was_not_built_for(tmp_path, monkeypatch):
    """SPM's MEX lie in its checkout, one build at a time: what it was built for is kept,
    and another runtime cleans it with SPM's own target first."""
    import mrsuite.octave as octave

    (tmp_path / "src").mkdir()
    builds = []

    def make(cmd, **kwargs):
        builds.append(cmd[-1])
        (tmp_path / "spm_bsplinc.mex").write_text("built")
        return type("Done", (), {"returncode": 0, "stderr": ""})()

    tag = {"value": "linux-x86_64-octave9"}
    monkeypatch.setattr(octave, "resolve_runtime", lambda runtime=None: "local")
    monkeypatch.setattr(octave, "platform_tag", lambda which: tag["value"])
    monkeypatch.setattr(octave.subprocess, "run", make)

    octave.build_spm12(str(tmp_path))
    octave.build_spm12(str(tmp_path))                     # built for this one already
    tag["value"] = "linux-x86_64-octave10"
    octave.build_spm12(str(tmp_path))

    assert len(builds) == 2
    assert all(b.startswith("make PLATFORM=octave distclean && ") for b in builds)
    assert (tmp_path / "src" / ".mrsuite-platform").read_text() == "linux-x86_64-octave10\n"
