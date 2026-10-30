####################################################################################################
#                                         test_launcher.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-30                                                                              #
#                                                                                                  #
# Purpose: The GUI as an app, as each system is given it: what is made starts this Python's GUI    #
#          with the PATH it was installed with and MRSuite's icon, and uninstalling removes it.    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import plistlib
import shlex
import subprocess
import sys

from mrsuite.gui import launcher


#***********#
#   macos   #
#***********#
def test_macos_gets_an_app_bundle_with_the_icon(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(launcher, "APPLICATIONS", tmp_path / "Applications")
    (tmp_path / "Applications").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", "/opt/octave/bin:/usr/bin")

    app, = launcher.install()

    assert app == tmp_path / "Applications" / "MRSuite.app"
    with open(app / "Contents" / "Info.plist", "rb") as fh:
        info = plistlib.load(fh)
    assert (info["CFBundleExecutable"], info["CFBundlePackageType"]) == ("MRSuite", "APPL")
    assert (app / "Contents" / "Resources" / info["CFBundleIconFile"]).is_file()
    script = app / "Contents" / "MacOS" / "MRSuite"
    assert os.access(script, os.X_OK)
    assert subprocess.run(["/bin/sh", "-n", str(script)]).returncode == 0
    assert "export PATH=/opt/octave/bin:/usr/bin\n" in script.read_text()
    assert f"exec {shlex.join([sys.executable, '-m', 'mrsuite.gui'])}\n" in script.read_text()

    assert launcher.install() == [app], "made anew"
    assert launcher.uninstall() == [app] and not app.exists()
    assert launcher.uninstall() == []


#*****************#
#   own folder   #
#*****************#
def test_macos_falls_back_to_the_users_own_applications(tmp_path, monkeypatch):
    """Where the Applications folder every user sees is not this user's to write to; once it
    is, the app moves there, so one app is there."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(launcher, "APPLICATIONS", tmp_path / "Applications")
    (tmp_path / "Applications").mkdir(mode=0o555)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    try:
        own, = launcher.install()
    finally:
        (tmp_path / "Applications").chmod(0o755)

    assert own == tmp_path / "home" / "Applications" / "MRSuite.app" and own.is_dir()
    system, = launcher.install()
    assert system == tmp_path / "Applications" / "MRSuite.app" and not own.exists()
    assert launcher.uninstall() == [system]


#***********#
#   linux   #
#***********#
def test_linux_gets_a_menu_entry_with_the_icon(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/home/me/my env/bin:/usr/bin")

    entry, = launcher.install()

    assert entry == tmp_path / "applications" / "mrsuite.desktop"
    fields = dict(line.split("=", 1) for line in entry.read_text().splitlines()[1:])
    assert (fields["Type"], fields["Name"], fields["Terminal"]) == \
        ("Application", "MRSuite", "false")
    assert shlex.split(fields["Exec"]) == ["env", "PATH=/home/me/my env/bin:/usr/bin",
                                           sys.executable, "-m", "mrsuite.gui"]
    assert os.path.isfile(fields["Icon"])

    assert launcher.uninstall() == [entry] and not entry.exists()


#*************#
#   windows   #
#*************#
def test_windows_gets_shortcuts_made_by_its_own_shell(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    calls = []

    def powershell(args, env, **kwargs):
        calls.append((args, env))
        made = "C:\\Start\\MRSuite.lnk\nC:\\Desk\\MRSuite.lnk\n"
        return subprocess.CompletedProcess(args, 0, made, "")
    monkeypatch.setattr(launcher.subprocess, "run", powershell)

    made = launcher.install()

    assert [str(p) for p in made] == ["C:\\Start\\MRSuite.lnk", "C:\\Desk\\MRSuite.lnk"]
    (args, env), = calls
    assert args[0] == "powershell" and "CreateShortcut" in args[-1]
    assert env["MRSUITE_PYTHON"] == sys.executable and os.path.isfile(env["MRSUITE_ICON"])
    assert "MRSUITE_REMOVE" not in env

    launcher.uninstall()
    assert calls[-1][1]["MRSUITE_REMOVE"] == "1"
