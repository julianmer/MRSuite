####################################################################################################
#                                        test_externals.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: Other people's code, where an installed MRSuite finds it. It runs what a checkout       #
#          runs, the commits the submodules pin, fetched once on first use and never on import.    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import shutil
import subprocess
import sys
import threading

import pytest

from mrsuite import externals

REAL = externals.root()      # the checkout's own externals/, before a test pretends otherwise


#*********#
#   git   #
#*********#
def git(folder, *args) -> str:
    return subprocess.run(["git", "-C", str(folder), "-c", "user.name=t", "-c", "user.email=t",
                           "-c", "commit.gpgsign=false", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


#**********#
#   repo   #
#**********#
def repo(folder) -> list:
    """A repository of two commits, standing in for an upstream; both commits, oldest first."""
    folder.mkdir()
    git(folder, "init", "--quiet")
    commits = []
    for name in ("first", "second"):
        (folder / name).write_text(name)
        git(folder, "add", ".")
        git(folder, "commit", "--quiet", "-m", name)
        commits.append(git(folder, "rev-parse", "HEAD"))
    return commits


#***************#
#   installed   #
#***************#
@pytest.fixture
def installed(tmp_path, monkeypatch):
    """MRSuite as pip installs it: no checkout around it, its home in tmp_path."""
    monkeypatch.setattr(externals, "CHECKOUT", tmp_path / "site-packages")
    monkeypatch.setenv("MRSUITE_HOME", str(tmp_path / "home"))
    return tmp_path / "home" / "externals"


#**********#
#   pins   #
#**********#
@pytest.mark.skipif(not (externals.CHECKOUT / ".gitmodules").is_file(), reason="not a checkout")
def test_an_install_runs_the_commits_the_submodules_pin():
    """Read from the index, so a staged submodule bump has to bring the registry along."""
    lines = git(externals.CHECKOUT, "config", "-f", ".gitmodules", "--get-regexp",
                r"^submodule\.externals/.*\.path$").splitlines()
    names = sorted(line.split()[1].removeprefix("externals/") for line in lines)

    assert names == sorted(externals.REGISTRY)
    for name in names:
        url = git(externals.CHECKOUT, "config", "-f", ".gitmodules",
                  f"submodule.externals/{name}.url")
        commit = git(externals.CHECKOUT, "ls-files", "-s", f"externals/{name}").split()[1]
        assert externals.REGISTRY[name] == (url, commit), name


#***********#
#   fetch   #
#***********#
def test_an_install_fetches_an_external_at_its_pin_once(installed, tmp_path, monkeypatch):
    first, _ = repo(tmp_path / "upstream")
    monkeypatch.setitem(externals.REGISTRY, "tool", (str(tmp_path / "upstream"), first))

    folder = externals.ensure("tool")
    shutil.rmtree(tmp_path / "upstream")              # a second clone would now fail

    assert folder.is_relative_to(installed)
    assert git(folder, "rev-parse", "HEAD") == first
    assert not (folder / "second").exists()
    assert externals.ensure("tool") == folder
    assert os.listdir(folder.parent) == [folder.name], "nothing left of the clone beside it"


#*************#
#   upgrade   #
#*************#
def test_an_upgraded_install_runs_its_own_pin(installed, tmp_path, monkeypatch):
    """What an older MRSuite fetched is never taken for what this one pins."""
    first, second = repo(tmp_path / "upstream")
    monkeypatch.setitem(externals.REGISTRY, "tool", (str(tmp_path / "upstream"), first))
    old = externals.ensure("tool")
    monkeypatch.setitem(externals.REGISTRY, "tool", (str(tmp_path / "upstream"), second))

    assert git(externals.ensure("tool"), "rev-parse", "HEAD") == second
    assert git(old, "rev-parse", "HEAD") == first


#*************#
#   threads   #
#*************#
def test_threads_fetching_at_once_all_get_the_external(installed, tmp_path, monkeypatch):
    first, _ = repo(tmp_path / "upstream")
    monkeypatch.setitem(externals.REGISTRY, "tool", (str(tmp_path / "upstream"), first))
    got, errors = [], []

    def fetch():
        try:
            got.append(externals.ensure("tool"))
        except Exception as err:                          # noqa: BLE001 - the test's answer
            errors.append(err)

    threads = [threading.Thread(target=fetch) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(set(got)) == 1 and git(got[0], "rev-parse", "HEAD") == first
    assert os.listdir(got[0].parent) == [got[0].name]


#************#
#   failed   #
#************#
def test_a_failed_fetch_leaves_nothing_that_would_pass_for_the_external(installed, tmp_path,
                                                                         monkeypatch):
    repo(tmp_path / "upstream")
    monkeypatch.setitem(externals.REGISTRY, "tool", (str(tmp_path / "upstream"), "0" * 40))

    with pytest.raises(RuntimeError, match="fetching tool at 0000000"):
        externals.ensure("tool")
    assert os.listdir(installed / "tool") == []


#***************#
#   submodule   #
#***************#
def test_a_checkout_says_how_to_get_a_submodule_it_lacks(tmp_path, monkeypatch):
    (tmp_path / ".gitmodules").write_text("")
    (tmp_path / "externals" / "osprey").mkdir(parents=True)
    monkeypatch.setattr(externals, "CHECKOUT", tmp_path)

    with pytest.raises(RuntimeError, match=f"git -C {tmp_path} submodule update --init "
                                           f"externals/osprey"):
        externals.ensure("osprey")


#************#
#   import   #
#************#
def test_nothing_is_fetched_on_import(tmp_path):
    """Importing MRSuite must not clone every toolbox; using one fetches it."""
    script = (f"from pathlib import Path\n"
              f"import mrsuite.externals as e\n"
              f"e.CHECKOUT = Path({str(tmp_path / 'site-packages')!r})\n"
              f"import mrsuite, mrsuite.models, mrsuite.simulators.mrs_sim\n"
              f"for name in sorted(mrsuite._BRIDGES - {{'fsl_mrs'}}):\n"
              f"    getattr(mrsuite, name)\n")
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env={**os.environ, "MRSUITE_HOME": str(tmp_path / "home")})

    assert run.returncode == 0, run.stderr
    assert not (tmp_path / "home").exists()


#***************#
#   one fetch   #
#***************#
def test_a_failed_fetch_is_tried_once_and_said(installed, monkeypatch):
    from mrsuite.bridges.fid_a import FIDABridge
    calls = []

    def fail(name):
        calls.append(name)
        raise RuntimeError(f"fetching {name} failed")
    monkeypatch.setattr(externals, "ensure", fail)

    with pytest.raises(RuntimeError, match="fetching fid-a failed"):
        FIDABridge().op_addphase
    assert calls == ["fid-a"]


#************#
#   models   #
#************#
def test_a_model_runs_from_where_its_repository_was_fetched(installed, monkeypatch):
    """Not from where MRSuite thought, on import, it would lie."""
    from mrsuite.models import MLP
    monkeypatch.setattr(externals, "ensure", lambda name: REAL / name)

    model = MLP(["NAA", "Cr"], 0, 64, {"points": 128, "bandwidth": 4000.0})

    assert model.repo == REAL / "ood-robust-mrs"


#*************#
#   convert   #
#*************#
def test_converting_fetches_the_original_first(installed, monkeypatch):
    from mrsuite.models import MLP
    from mrsuite.models import convert

    def fetch(name):
        raise RuntimeError(f"fetch {name}")
    monkeypatch.setattr(externals, "ensure", fetch)

    with pytest.raises(RuntimeError, match="fetch ood-robust-mrs"):
        convert.run_reference(MLP, {"task": "config"}, python=sys.executable)


#************#
#   docker   #
#************#
def test_the_octave_container_sees_what_an_install_fetched(installed):
    from mrsuite.octave import docker

    assert f"{installed}:{installed}" in docker._mounts()
    assert installed.is_dir(), "made here, else Docker makes it as root"
