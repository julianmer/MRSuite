####################################################################################################
#                                        test_externals.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: Other people's code, where MRSuite finds it: the commits it pins, each fetched once on  #
#          first use as GitHub's archive of that commit, and never on import.                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import io
import os
import shutil
import subprocess
import sys
import tarfile
import threading

import pytest

from mrsuite import externals

OOD = externals.path("ood-robust-mrs")   # the checkout's own, before a test pretends otherwise
FIRST, SECOND = "1" * 40, "2" * 40


#**************#
#   upstream   #
#**************#
def upstream(folder) -> str:
    """GitHub, standing in: an archive per commit, FIRST holding 'first' and SECOND 'first' and
    'second', each in the one folder GitHub's archives hold; its URL."""
    for commit, names in ((FIRST, ["first"]), (SECOND, ["first", "second"])):
        (folder / "archive").mkdir(parents=True, exist_ok=True)
        with tarfile.open(folder / "archive" / f"{commit}.tar.gz", "w:gz") as tar:
            for name in names:
                info = tarfile.TarInfo(f"tool-{commit}/{name}")
                info.size = len(name)
                tar.addfile(info, io.BytesIO(name.encode()))
    return folder.as_uri()


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
def test_every_external_is_a_commit_github_serves_an_archive_of():
    for name, (url, commit) in externals.REGISTRY.items():
        assert url.startswith("https://github.com/") and len(commit) == 40, name
        assert externals.archive(name) == (f"{url.removesuffix('.git')}/archive/{commit}.tar.gz")
        assert externals.version(name) == commit[:7]


#***********#
#   fetch   #
#***********#
def test_an_install_fetches_an_external_at_its_pin_once(installed, tmp_path, monkeypatch):
    monkeypatch.setitem(externals.REGISTRY, "tool", (upstream(tmp_path / "upstream"), FIRST))

    folder = externals.ensure("tool")
    shutil.rmtree(tmp_path / "upstream")              # a second fetch would now fail

    assert folder.is_relative_to(installed)
    assert sorted(os.listdir(folder)) == ["first"]
    assert externals.ensure("tool") == folder
    assert os.listdir(folder.parent) == [folder.name], "nothing left of the fetch beside it"


#**************#
#   checkout   #
#**************#
def test_a_checkout_fetches_into_its_own_externals(tmp_path, monkeypatch):
    (tmp_path / "checkout").mkdir()
    (tmp_path / "checkout" / ".gitmodules").write_text("")
    monkeypatch.setattr(externals, "CHECKOUT", tmp_path / "checkout")
    monkeypatch.setitem(externals.REGISTRY, "tool", (upstream(tmp_path / "upstream"), FIRST))

    assert externals.ensure("tool") == tmp_path / "checkout" / "externals" / "tool" / FIRST[:12]


#*************#
#   as made   #
#*************#
def test_scripts_stay_executable_and_links_stay_links(installed, tmp_path, monkeypatch):
    """What a toolbox runs as a program, or reads through a link, works as it did upstream."""
    (tmp_path / "upstream" / "archive").mkdir(parents=True)
    with tarfile.open(tmp_path / "upstream" / "archive" / f"{FIRST}.tar.gz", "w:gz") as tar:
        script = tarfile.TarInfo(f"tool-{FIRST}/run.sh")
        script.size, script.mode = 4, 0o755
        tar.addfile(script, io.BytesIO(b"true"))
        link = tarfile.TarInfo(f"tool-{FIRST}/docs/run.sh")
        link.type, link.linkname = tarfile.SYMTYPE, "../run.sh"
        tar.addfile(link)
    monkeypatch.setitem(externals.REGISTRY, "tool", ((tmp_path / "upstream").as_uri(), FIRST))

    folder = externals.ensure("tool")

    assert os.access(folder / "run.sh", os.X_OK)
    assert (folder / "docs" / "run.sh").is_symlink()
    assert (folder / "docs" / "run.sh").read_text() == "true"


#*************#
#   upgrade   #
#*************#
def test_an_upgraded_install_runs_its_own_pin(installed, tmp_path, monkeypatch):
    """What an older MRSuite fetched is never taken for what this one pins."""
    url = upstream(tmp_path / "upstream")
    monkeypatch.setitem(externals.REGISTRY, "tool", (url, FIRST))
    old = externals.ensure("tool")
    monkeypatch.setitem(externals.REGISTRY, "tool", (url, SECOND))

    assert sorted(os.listdir(externals.ensure("tool"))) == ["first", "second"]
    assert sorted(os.listdir(old)) == ["first"]


#*************#
#   threads   #
#*************#
def test_threads_fetching_at_once_all_get_the_external(installed, tmp_path, monkeypatch):
    monkeypatch.setitem(externals.REGISTRY, "tool", (upstream(tmp_path / "upstream"), FIRST))
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
    assert len(set(got)) == 1 and os.listdir(got[0]) == ["first"]
    assert os.listdir(got[0].parent) == [got[0].name]


#************#
#   failed   #
#************#
def test_a_failed_fetch_leaves_nothing_that_would_pass_for_the_external(installed, tmp_path,
                                                                         monkeypatch):
    monkeypatch.setitem(externals.REGISTRY, "tool", (upstream(tmp_path / "upstream"), "0" * 40))

    with pytest.raises(RuntimeError, match="fetching tool at 0000000"):
        externals.ensure("tool")
    assert os.listdir(installed / "tool") == []


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
    monkeypatch.setattr(externals, "ensure", lambda name: OOD)

    model = MLP(["NAA", "Cr"], 0, 64, {"points": 128, "bandwidth": 4000.0})

    assert model.repo == OOD


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

    home = installed.parent                     # the toolboxes, data, scratch and MEX
    assert f"{home}:{home}" in docker.mounts()
    assert home.is_dir(), "made here, else Docker makes it as root"


#***********#
#   local   #
#***********#
def test_what_an_install_fetches_builds_and_scratches_is_its_own(installed):
    home = installed.parent
    assert externals.local("data") == home / "data" and externals.root() == installed
    assert not externals.local("data").is_relative_to(externals.CHECKOUT)


#**************#
#   download   #
#**************#
def test_a_download_is_kept_only_whole_and_as_published(tmp_path):
    import hashlib

    served = tmp_path / "served.bin"
    served.write_bytes(b"x" * 3000)
    url, md5 = served.as_uri(), hashlib.md5(served.read_bytes()).hexdigest()
    dest = tmp_path / "got" / "file.bin"

    with pytest.raises(RuntimeError, match="2999 published"):
        externals.download(url, dest, size=2999)
    with pytest.raises(RuntimeError, match="MD5"):
        externals.download(url, dest, md5="0" * 32)
    assert not list(dest.parent.iterdir()), "nothing kept, no part left"

    assert externals.download(url, dest, 3000, md5).read_bytes() == served.read_bytes()


#**************#
#   fetching   #
#**************#
def test_a_download_is_said_while_it_lasts_and_then_not(tmp_path, capsys):
    with externals.fetching("a dataset (6 MB)", tmp_path):
        assert externals.now() == f"Downloading a dataset (6 MB) into {tmp_path}…"
        with externals.fetching("another", tmp_path, say=False):
            assert externals.now().startswith("Downloading another")
    assert externals.now() is None
    assert capsys.readouterr().err == f"Downloading a dataset (6 MB) into {tmp_path}…\n"


#*************#
#   move in   #
#*************#
def test_a_folder_moves_in_whole_and_another_process_first_stands(tmp_path):
    first, second, dest = tmp_path / "first", tmp_path / "second", tmp_path / "dest"
    for part in (first, second):
        part.mkdir(mode=0o700)
        (part / "made").write_text(part.name)

    externals.move_in(first, dest)
    externals.move_in(second, dest)                       # too late: no error, first stands

    assert (dest / "made").read_text() == "first" and second.is_dir()
    assert dest.stat().st_mode & 0o055, "readable by others, as any folder made here"
    with pytest.raises(OSError):
        externals.move_in(tmp_path / "never-made", tmp_path / "nowhere")
