####################################################################################################
#                                          test_graph.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: The GUI's graph without the GUI: its blocks, its wires, what it runs and keeps, and     #
#          that the script it writes, run outside it, makes exactly what it made.                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import builtins
import json
import threading

import numpy as np
import pytest

from mrsuite import externals
from mrsuite.datasets import challenge2016
from mrsuite.gui import blocks, catalog
from mrsuite.gui.graph import Graph, brief, export, exports, thumb, traces

needs_challenge = pytest.mark.skipif(not challenge2016.BASIS.exists(),
                                     reason="challenge submodule not checked out")


#***********#
#   graph   #
#***********#
@pytest.fixture
def graph():
    """Two challenge spectra, broadened, then fitted with their water reference, and their
    SNR on a branch of its own."""
    graph = Graph()
    data = graph.add("ismrm2016")
    graph.set(data, "n", 2)
    broaden = graph.add("apodization")
    graph.set(broaden, "amount", [5.0])
    fit = graph.add("lcm-fit")
    graph.set(fit, "basis", str(challenge2016.BASIS))
    graph.set(fit, catalog.MORE, {"te": 0.030, "tr": 2.0})
    snr = graph.add("snr")
    graph.connect(data, "spectra", broaden, "data")
    graph.connect(broaden, "out", fit, "data")
    graph.show(fit, "water")
    graph.connect(data, "water", fit, "water")
    graph.connect(data, "spectra", snr, "data")
    return graph


#************#
#   blocks   #
#************#
@pytest.mark.parametrize("bridge", ["fsl_mrs", "fid_a", "osprey", "spant", "mrspeclab"])
def test_every_block_names_a_function_its_toolbox_has(bridge):
    try:
        have = set(catalog.functions(bridge, everything=True))
    except Exception as err:                           # noqa: BLE001 - a toolbox not here
        pytest.skip(f"{bridge} not available: {err}")
    named = {f for b in blocks.BLOCKS.values() for where, f in b.options if where == bridge}

    assert named and named <= have, named - have


@pytest.mark.parametrize("bridge", ["fsl_mrs", "fid_a", "osprey", "spant", "mrspeclab"])
def test_every_block_but_a_fit_offers_steps_of_its_toolbox(bridge):
    """Takes spectra first, as a step does (a fit's own wrapping takes them too)."""
    try:
        steps = set(catalog.module(bridge).steps())
    except Exception as err:                           # noqa: BLE001 - a toolbox not here
        pytest.skip(f"{bridge} not available: {err}")
    named = {f for b in blocks.BLOCKS.values() if b.group != "fit"
             for where, f in b.options if where == bridge}

    assert named and named <= steps, named - steps


def test_a_plain_function_block_is_found():
    signature, _ = catalog.about("mrsuite.datasets.challenge2016", "snr")

    assert list(signature.parameters) == ["data"]


#************#
#   script   #
#************#
@needs_challenge
def test_the_script_makes_what_the_graph_made(graph):
    graph.run()
    namespace = {}
    exec(graph.script(), namespace)

    made, again = graph.output("n3"), namespace["lcm_fit"]
    assert [r.conc("molarity") for r in again] == [r.conc("molarity") for r in made]
    assert np.array_equal(namespace["snr"], graph.output("n4"))
    assert "ismrm2016_spectra, ismrm2016_water, ismrm2016_truth = challenge2016.load(n=2)" \
        in graph.script()


@needs_challenge
def test_a_script_up_to_a_block_has_only_what_it_needs(graph):
    script = graph.script("n4")

    assert "snr = challenge2016.snr(ismrm2016_spectra)" in script
    assert "fsl_mrs" not in script


#************#
#   reruns   #
#************#
@needs_challenge
def test_a_change_reruns_only_the_blocks_from_it(graph):
    graph.run()
    data, broadened, snr = (graph.nodes[k].made for k in ("n1", "n2", "n4"))
    graph.set("n3", catalog.MORE, {"te": 0.030, "tr": 3.0})

    assert graph.nodes["n3"].made is None, "what the changed block made is dropped"
    graph.run()
    assert (graph.nodes["n1"].made, graph.nodes["n2"].made, graph.nodes["n4"].made) == \
        (data, broadened, snr), "the blocks before it and beside it did not run again"
    assert graph.nodes["n3"].made is not None


@needs_challenge
def test_running_up_to_a_block_runs_only_what_it_needs(graph):
    graph.run("n4")

    assert [k for k, n in graph.nodes.items() if n.made is not None] == ["n1", "n4"]


@needs_challenge
def test_an_edit_while_running_drops_what_the_run_made_after_it(graph):
    graph.run(progress=lambda k: k == "n3" and graph.set("n2", "amount", [3.0]))

    assert graph.nodes["n2"].made is None and graph.nodes["n3"].made is None
    graph.run()
    assert graph.nodes["n3"].made is not None


@needs_challenge
@pytest.mark.parametrize("edited, kept", [("n4", True), ("n2", False)])
def test_what_a_block_makes_while_another_is_edited_is_kept(graph, monkeypatch, edited, kept):
    """An edit from the page, in its own thread, while the broadening runs: of a block
    beside it (the SNR) the broadening's result stays; of the broadening itself, it goes."""
    def edit():
        page = threading.Thread(target=graph.set, args=(edited, catalog.MORE, {"x": 1}))
        page.start()
        page.join()

    code = graph.code
    monkeypatch.setattr(builtins, "page_edits", edit, raising=False)
    monkeypatch.setattr(graph, "code", lambda k: (lambda line, imports, names: (
        line + "\npage_edits()" if k == "n2" else line, imports, names))(*code(k)))

    assert graph.run() is False, "an edit stops the run"
    assert (graph.nodes["n2"].made is not None) == kept
    assert graph.nodes["n1"].made is not None


def test_a_file_read_changed_drops_what_the_blocks_after_it_made(tmp_path):
    graph = Graph()
    read, snr = graph.add("files"), graph.add("snr")
    graph.set(read, "files", [str(tmp_path / "a.nii.gz")])
    graph.connect(read, "spectra", snr, "data")
    graph.nodes[snr].made = {"snr": 1.0}

    graph.set(read, "files", [str(tmp_path / "a.dat")])            # read otherwise now

    assert graph.nodes[snr].made is None


#***********#
#   basis   #
#***********#
@needs_challenge
@pytest.mark.skipif(not (externals.path("basic") / "basis_converter").exists(),
                    reason="BasiC submodule not checked out")
def test_a_basis_set_block_gives_a_fit_its_basis(tmp_path, monkeypatch):
    """The challenge's own basis, read by BasiC and written again, fits as the original,
    and the fit says which basis it was."""
    monkeypatch.setenv("MRSUITE_HOME", str(tmp_path))
    graph = Graph()
    data, basis, fit, given = (graph.add(k) for k in ("ismrm2016", "basis", "lcm-fit",
                                                         "lcm-fit"))
    graph.set(data, "n", 1)
    graph.set(basis, "files", [str(challenge2016.BASIS)])
    graph.set(given, "basis", str(challenge2016.BASIS))
    for target in (fit, given):
        graph.connect(data, "spectra", target, "data")
    graph.connect(basis, "basis", fit, "basis")

    assert graph.run()
    namespace = {}
    exec(graph.script(), namespace)

    assert f"basis = io.basis([{str(challenge2016.BASIS)!r}])\n" in graph.script()
    made = graph.output(basis)
    assert made.names == namespace["basis"].names and len(made.names) == 21
    assert made.checksum() == namespace["basis"].checksum()
    conc = [r.conc("raw") for r in graph.output(fit)]
    assert conc == [r.conc("raw") for r in graph.output(given)]
    assert conc == [r.conc("raw") for r in namespace["lcm_fit"]]
    said = json.loads(graph.output(fit)[0].provenance[-1]["Details"])
    assert said["kwargs"]["basis"]["basis"]["sha256"] == made.checksum()


#************#
#   folder   #
#************#
def test_a_folder_block_loads_a_study_and_shows_the_roles_it_found(study):
    graph = Graph()
    folder, fit = graph.add("folder"), graph.add("lcm-fit")
    graph.set(folder, "folder", str(study))
    assert graph.ports(folder)[1] == [], "its outputs are known once it ran"

    with pytest.warns(UserWarning, match="do not line up"):
        assert graph.run(folder)

    assert [p.name for p in graph.ports(folder)[1]] == ["svs", "mrsref", "mm"]
    assert len(graph.output(folder, "mrsref").list()) == 3
    graph.connect(folder, "svs", fit, "data")
    graph.ask(fit)
    graph.show(fit, "water")                        # a port once put on the block
    graph.connect(folder, "mrsref", fit, "water")
    assert f"collect.load({str(study)!r})" in graph.script()
    assert f"{graph.nodes[folder].name}['mrsref']" in graph.script()


def test_a_blocks_own_list_of_files_loads_and_is_written_out_in_full(study):
    """What the window collected, from anywhere, as (path, role, group) triples."""
    graph = Graph()
    folder = graph.add("folder")
    chosen = [(str(study / "extra/press_metab_PFL.nii.gz"), "svs", "one"),
              (str(study / "sub-02/mrs/sub-02_acq-press_mrsref.nii.gz"), "mrsref", "one")]
    graph.set(folder, "entries", chosen)

    assert graph.run(folder)
    assert [p.name for p in graph.ports(folder)[1]] == ["svs", "mrsref"]
    assert all(repr(c) in graph.script() for c in chosen)
    assert Graph.from_dict(graph.to_dict()).nodes[folder].values["entries"] == chosen


def test_what_the_window_put_right_goes_into_the_script(study):
    graph = Graph()
    folder = graph.add("folder")
    graph.set(folder, "folder", str(study))
    graph.set(folder, "roles", {"extra/press_mm_PFL.nii.gz": ""})      # left out

    assert graph.run(folder)
    assert [p.name for p in graph.ports(folder)[1]] == ["svs", "mrsref"]
    assert "roles={'extra/press_mm_PFL.nii.gz': ''}" in graph.script()


#**************#
#   basis add   #
#**************#
@needs_challenge
@pytest.mark.skipif(not (externals.path("basic") / "basis_converter").exists(),
                    reason="BasiC submodule not checked out")
def test_a_spectrum_wired_into_a_basis_set_becomes_one_of_its_functions():
    graph = Graph()
    data, basis, add = graph.add("ismrm2016"), graph.add("basis"), graph.add("basis-add")
    graph.set(data, "n", 1)
    graph.set(basis, "files", [str(challenge2016.BASIS)])
    graph.connect(basis, "basis", add, "basis")
    graph.connect(data, "water", add, "spectrum")
    graph.set(add, "name", "W")

    assert graph.run()
    assert graph.output(add).names[-1] == "W" and len(graph.output(add).names) == 22
    assert "io.add_to_basis(basis, ismrm2016_water, name='W')" in graph.script()
    fit = graph.add("lcm-fit")
    graph.connect(add, "basis", fit, "basis")                   # a basis port takes it


#***********#
#   files   #
#***********#
def test_spectra_cannot_be_wired_into_the_files_a_block_reads():
    graph = Graph()
    data, basis = graph.add("ismrm2016"), graph.add("basis")

    with pytest.raises(ValueError, match="cannot go into"):
        graph.connect(data, "spectra", basis, "files")


#*************#
#   convert   #
#*************#
@needs_challenge
@pytest.mark.skipif(not (externals.path("basic") / "basis_converter").exists(),
                    reason="BasiC submodule not checked out")
def test_a_converted_basis_set_is_the_one_wired_in(tmp_path, monkeypatch):
    monkeypatch.setenv("MRSUITE_HOME", str(tmp_path))
    graph = Graph()
    basis, convert = graph.add("basis"), graph.add("basis-convert")
    graph.set(basis, "files", [str(challenge2016.BASIS)])
    graph.set(convert, "folder", str(tmp_path / "run"))
    graph.set(convert, "tool", "jmrui")
    graph.connect(basis, "basis", convert, "basis")

    assert graph.run()
    written = graph.output(convert)
    exec(graph.script().replace(str(tmp_path / "run"), str(tmp_path / "script")), {})

    assert sorted(p.stem for p in written) == sorted(graph.output(basis).names)
    assert [p.read_text().replace("/run/", "/script/") for p in written] == \
        [(tmp_path / "script" / p.name).read_text() for p in written]   # each names itself


#***********#
#   kinds   #
#***********#
def test_a_basis_set_and_converted_files_go_only_where_they_fit():
    """A basis is no spectra, and files written for another tool are no basis either."""
    graph = Graph()
    basis, convert, broaden, fit, other = (graph.add(k) for k in (
        "basis", "basis-convert", "apodization", "lcm-fit", "basis"))
    for source, output in ((basis, "basis"), (convert, "files")):
        with pytest.raises(ValueError, match="cannot go into spectra"):
            graph.connect(source, output, broaden, "data")
    with pytest.raises(ValueError, match="files cannot go into basis"):
        graph.connect(convert, "files", fit, "basis")
    with pytest.raises(ValueError, match="spectra cannot go into basis"):
        graph.connect(graph.add("ismrm2016"), "spectra", convert, "basis")
    graph.connect(basis, "basis", fit, "basis")
    graph.connect(basis, "basis", convert, "basis")
    graph.connect(convert, "files", other, "files")

    assert [e.input for e in graph.edges] == ["basis", "basis", "files"]


#***********#
#   thumb   #
#***********#
def test_a_file_made_shows_its_name():
    from pathlib import Path

    assert thumb(Path("/x/basis.BASIS")) == {"kind": "text", "text": "basis.BASIS"}


#***************#
#   basis set   #
#***************#
def test_a_basis_set_is_drawn_by_metabolite_and_written_out_as_a_basis_file(tmp_path,
                                                                           monkeypatch):
    """Two peaks, at 2.01 and 3.03 ppm on a set centred at 3.0, as Osprey's are."""
    from mrsuite.core.basis import Basis

    monkeypatch.setenv("MRSUITE_HOME", str(tmp_path))
    t = np.arange(2048) / 4000.0
    fids = np.array([np.exp(2j * np.pi * (ppm - 3.0) * 123.2 * t - t * 20)
                     for ppm in (2.01, 3.03)])
    basis = Basis(["NAA", "Cr"], fids, 4000.0, 123.2, centre=3.0)
    if not (externals.path("basic") / "basis_converter").exists():
        pytest.skip("BasiC submodule not checked out")

    ppm, ys, labels = traces(basis)
    line = thumb(basis)
    written, = export(basis, tmp_path / "out", "set", "basis")

    assert labels == ["NAA", "Cr"]
    assert np.allclose([ppm[np.argmax(y)] for y in ys], [2.01, 3.03],   # the nearest point
                       atol=(ppm[1] - ppm[0]) / 2)
    assert line["kind"] == "line" and max(line["y"]) == 1
    assert exports(basis) == ["basis"] and written == tmp_path / "out" / "set.BASIS"
    assert written.read_bytes() == basis.lcmodel().read_bytes()


#************#
#   errors   #
#************#
@needs_challenge
def test_a_failure_is_kept_on_its_block(graph):
    graph.set("n3", catalog.MORE, None)                # no echo time for water scaling

    with pytest.raises(ValueError, match="echo and repetition time"):
        graph.run()
    assert "echo and repetition time" in graph.nodes["n3"].error
    graph.set("n3", catalog.MORE, {"te": 0.030, "tr": 2.0})
    assert graph.nodes["n3"].error is None


def test_an_input_left_unwired_is_said():
    graph = Graph()
    graph.add("averaging")

    with pytest.raises(ValueError, match="wire something into data"):
        graph.run()


#**************#
#   in order   #
#**************#
def test_arguments_octave_takes_in_order_are_set_in_order():
    """One left out before one set is its default where FID-A writes it plainly (nsd=3);
    where FID-A works it out (M), it has to be set."""
    graph = Graph()
    data = graph.add("ismrm2016")
    plain, worked = graph.add("fid_a:op_rmbadaverages"), graph.add("fid_a:op_removeWater")
    for step in (plain, worked):
        graph.connect(data, "spectra", step, "data")
    graph.set(plain, "domain", "f")
    graph.set(worked, "plot_bool", 0)

    assert "fid_a.step.op_rmbadaverages(3, 'f')" in graph.code(plain)[0]
    with pytest.raises(ValueError, match="set M too"):
        graph.code(worked)
    graph.set(worked, "M", 100)
    assert "fid_a.step.op_removeWater([4.4, 5], 30, 100, 0)" in graph.code(worked)[0]


#***********#
#   wires   #
#***********#
def test_wires_keep_to_their_kinds_and_make_no_loops():
    graph = Graph()
    data, fit, again = graph.add("ismrm2016"), graph.add("lcm-fit"), graph.add("averaging")

    with pytest.raises(ValueError, match="truth cannot go into spectra"):
        graph.connect(data, "truth", fit, "data")
    graph.connect(data, "spectra", again, "data")
    graph.connect(data, "water", again, "data")
    assert [e.output for e in graph.edges] == ["water"], "an input takes one wire"
    with pytest.raises(ValueError, match="loop"):
        graph.connect(again, "out", again, "data")


def test_an_input_taking_many_gets_them_by_name():
    graph = Graph()
    data, one, two, score = (graph.add(k) for k in
                             ("ismrm2016", "lcm-fit", "lcm-fit", "benchmark"))
    for fit in (one, two):
        graph.connect(data, "spectra", fit, "data")
        graph.connect(fit, "out", score, "methods")
    graph.connect(data, "truth", score, "truth")
    graph.set(score, "references", "optimal")
    graph.set(score, "output_dir", "runs")

    assert graph.code(score)[0] == (
        "benchmark = scoring.score({'lcm_fit': lcm_fit, 'lcm_fit_2': lcm_fit_2}, "
        "ismrm2016_truth, references=['optimal'], output_dir='runs', "
        "metrics={'mae': scoring.mae, 'bias': scoring.bias})")


def test_another_toolbox_starts_its_form_and_keeps_the_wires_it_takes():
    graph = Graph()
    data, fit = graph.add("ismrm2016"), graph.add("lcm-fit")
    graph.connect(data, "spectra", fit, "data")
    graph.show(fit, "water")
    water = graph.connect(data, "water", fit, "water")
    graph.set(fit, "basis", "x")

    graph.choose(fit, 1)                               # LCModel: it takes water too
    assert graph.nodes[fit].values == {} and water in graph.edges
    graph.choose(fit, 2)                               # Osprey: no water
    assert [e.input for e in graph.edges] == ["data"] and graph.nodes[fit].shown == []


def test_a_step_starts_as_its_toolboxs_own_pipeline_runs_it():
    """FID-A's run_pressproc_auto.m phases on creatine, 2.9-3.1 ppm."""
    graph = Graph()
    data, phase = graph.add("ismrm2016"), graph.add("phasing")
    graph.connect(data, "spectra", phase, "data")

    assert graph.nodes[phase].values == {}             # FSL-MRS: its own defaults
    graph.choose(phase, 1)
    assert "fid_a.step.op_autophase(2.9, 3.1)" in graph.code(phase)[0]
    assert graph.nodes[graph.add("fid_a:op_autophase")].values == {"ppmmin": 2.9, "ppmmax": 3.1}
    assert graph.nodes[graph.add("fid_a:op_ppmref")].values == \
        {"ppmmin": 2.9, "ppmmax": 3.1, "ppmrefval": 3.027}


def test_an_argument_is_on_the_block_when_needed_or_put_there():
    graph = Graph()
    data, fit = graph.add("ismrm2016"), graph.add("lcm-fit")
    graph.connect(data, "spectra", fit, "data")
    graph.set(fit, "basis", "set.BASIS")
    names = lambda: [p.name for p in graph.ports(fit)[0]]

    assert names() == ["data", "basis"], "basis has no default; water, ppmlim, ... do"
    with pytest.raises(ValueError, match="no such port"):
        graph.connect(data, "water", fit, "water")
    graph.show(fit, "ppmlim")
    graph.show(fit, "water")
    assert names() == ["data", "basis", "water", "ppmlim"], "in the function's order"
    graph.connect(data, "water", fit, "water")
    graph.connect(graph.add("basis"), "basis", fit, "basis")   # a wire takes the form's place
    line = graph.code(fit)[0]
    assert "water=ismrm2016_water" in line and "basis=basis" in line

    graph.nodes[fit].made = {"fit": 1}               # as if it had run
    graph.show(fit, "water", False)                    # off the block, its wire with it
    assert names() == ["data", "basis", "ppmlim"] and graph.nodes[fit].made is None
    assert "water" not in graph.code(fit)[0]


def test_removing_a_block_takes_its_wires():
    graph = Graph()
    data, step = graph.add("ismrm2016"), graph.add("averaging")
    graph.connect(data, "spectra", step, "data")

    graph.remove(data)

    assert list(graph.nodes) == [step] and graph.edges == []


#***********#
#   names   #
#***********#
def test_a_block_is_named_as_python_allows():
    graph = Graph()

    assert [graph.nodes[graph.add(k)].name for k in
            ("averaging", "averaging", "fsl_mrs:fit", "simulation")] == \
        ["averaging", "averaging_2", "fit", "simulation"]
    assert graph.nodes[graph.add("fsl_mrs:average")].name == "average"


#**********#
#   save   #
#**********#
def test_a_saved_graph_comes_back_the_same(tmp_path):
    graph = Graph()
    data, step = graph.add("ismrm2016", (10, 20)), graph.add("zero-filling")
    graph.connect(data, "spectra", step, "data")
    graph.set(step, "npoints", (4096, 1.5))
    graph.save(tmp_path / "g.json")

    again = Graph.load(tmp_path / "g.json")

    assert again.script() == graph.script()
    assert again.nodes[step].values == {"npoints": (4096, 1.5)}, "a tuple stays a tuple"
    assert again.nodes[data].position == (10, 20)
    assert again.nodes[again.add("averaging")].id == "n3"
    with pytest.raises(ValueError, match="not a saved graph"):
        Graph.from_dict(json.loads('{"format": "x"}'))


def test_a_saved_graph_keeps_its_toolboxes_wires_and_texts(tmp_path):
    graph = Graph()
    data, fit, read = graph.add("ismrm2016"), graph.add("lcm-fit"), graph.add("files")
    graph.choose(fit, 1)                                           # LCModel
    graph.set(fit, "basis", "a 'quoted' path.basis")
    graph.show(fit, "water")
    graph.connect(data, "water", fit, "water")
    graph.nodes[read].found = ["metab", "wref"]
    graph.save(tmp_path / "g.json")

    again = Graph.load(tmp_path / "g.json")

    node = again.nodes[fit]
    assert (node.option, node.values) == (1, {"basis": "a 'quoted' path.basis"})
    assert again.edges == graph.edges and again.nodes[read].found == ["metab", "wref"]
    assert node.shown == ["water"]


@pytest.mark.parametrize("change", [
    lambda n, e: n[1].update(name="import os; os.system('x'); snr"),   # code for a name
    lambda n, e: n[1].update(block="fsl_mrs:average(); import os"),   # code for a block
    lambda n, e: n[1].update(values={"ok; import os": "1"}),           # code for an argument
    lambda n, e: n[1].update(values={"x": "2"} and {"x": 2}),          # not Python text
    lambda n, e: e.append(["n1", "truth", "n2", "data"]),              # a wire that won't fit
    lambda n, e: n[1].update(name=n[0]["name"]),                      # two blocks, one name
])
def test_a_saved_graph_that_is_not_one_mrsuite_wrote_is_refused(change):
    graph = Graph()
    graph.connect(graph.add("ismrm2016"), "spectra", graph.add("snr"), "data")
    saved = graph.to_dict()
    change(saved["nodes"], saved["edges"])

    with pytest.raises(ValueError, match="not a graph MRSuite can read"):
        Graph.from_dict(saved)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), ..., {"a": float("-inf")}])
def test_a_value_python_text_cannot_give_back_is_refused(value):
    graph = Graph()

    with pytest.raises(ValueError, match="cannot be written"):
        graph.set(graph.add("zero-filling"), "npoints", value)


def test_a_further_argument_named_as_a_python_keyword_is_spelt_out():
    graph = Graph()
    data, step = graph.add("ismrm2016"), graph.add("fsl_mrs:average")
    graph.connect(data, "spectra", step, "data")
    graph.set(step, catalog.MORE, {"lambda": 1, "ok": 2})

    line = graph.code(step)[0]
    assert "ok=2" in line and "**{'lambda': 1}" in line
    compile(line, "<line>", "exec")


#*************#
#   outputs   #
#*************#
@needs_challenge
def test_files_read_show_what_they_hold(tmp_path):
    data, _, _ = challenge2016.load(1)
    data.list()[0].save(str(tmp_path / "scan.nii.gz"))
    graph = Graph()
    read = graph.add("files")
    graph.set(read, "files", [str(tmp_path / "scan.nii.gz")])

    assert [p.name for p in graph.ports(read)[1]] == ["spectra"]
    graph.run()
    assert len(graph.output(read)) == 1
    assert graph.code(read)[0] == f"files = io.read({str(tmp_path / 'scan.nii.gz')!r})"


#************#
#   export   #
#************#
@needs_challenge
def test_what_a_block_made_is_written_out(graph, tmp_path):
    graph.run()

    assert len(export(graph.output("n2"), tmp_path, "broadened", "spectra")) == 2
    assert export(graph.output("n2"), tmp_path, "broadened", "figure")[0].exists()
    assert export(graph.output("n3"), tmp_path, "fit", "results")[0].exists()
    with pytest.raises(ValueError, match="cannot be written out as results"):
        export(graph.output("n2"), tmp_path, "broadened", "results")


#************#
#   thumbs   #
#************#
@needs_challenge
def test_thumbnails_show_a_spectrum_and_a_fit(graph):
    graph.run()
    line, bars = thumb(graph.output("n1")), thumb(graph.output("n3"))

    assert line["kind"] == "line" and 0 <= min(line["y"]) and max(line["y"]) == 1
    assert bars["kind"] == "bars" and bars["labels"] and max(bars["y"]) == 1


#************#
#   traces   #
#************#
def test_a_spectrum_is_drawn_on_its_ppm_axis():
    if not challenge2016.BASIS.exists():
        pytest.skip("challenge submodule not checked out")
    data, _, _ = challenge2016.load(1)
    ppm, ys, _ = traces(data.list()[0])
    window = (ppm > 1.8) & (ppm < 2.2)

    assert ppm[window][np.argmax(ys[0][window])] == pytest.approx(2.01, abs=0.02)   # NAA


def test_a_spectrum_is_drawn_about_its_nucleus_own_centre():
    from nifti_mrs.create_nmrs import gen_nifti_mrs

    t = np.arange(1024) * 2e-4
    spectrum = gen_nifti_mrs(np.exp(-t / 0.05 + 0j).reshape(1, 1, 1, -1), 2e-4, 49.9, nucleus="31P")
    ppm, ys, _ = traces(spectrum)

    assert ppm[np.argmax(ys[0])] == pytest.approx(0.0, abs=0.1)


def test_coils_and_averages_are_drawn_apart_one_dimension_at_a_time_and_stacked():
    """Spread by coil, each trace is that coil's mean over its averages; by average, coil 0's
    averages; stacked, each sits a share of the largest amplitude above the one before."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs

    t = np.arange(512) * 2.5e-4
    base = np.exp(2j * np.pi * (2.0 - 4.65) * 123.2 * t - t / 0.05)
    fids = np.stack([np.stack([(c + 1) * (a + 1) * base for a in range(3)], -1)   # coil, avg
                     for c in range(2)], -1).reshape(1, 1, 1, -1, 3, 2)
    fids = np.moveaxis(fids, 4, 5)                                          # (coil, average)
    spectrum = gen_nifti_mrs(fids, 2.5e-4, 123.2, dim_tags=["DIM_COIL", "DIM_DYN", None])
    one = lambda fid: np.fft.fftshift(np.fft.fft(fid)).real

    ppm, ys, labels = traces(spectrum)
    assert labels == ["coil 0", "coil 1"]
    assert np.allclose(ys[1], one(2 * 2 * base)), "coil 1, averaged over (1+2+3)/3 = 2"

    _, ys, labels = traces(spectrum, spread="DIM_DYN")
    assert labels == ["dyn 0", "dyn 1", "dyn 2"]
    assert np.allclose(ys[2], one(3 * base)), "average 2 of coil 0"

    _, flat, _ = traces(spectrum, spread="DIM_DYN")
    _, stacked, _ = traces(spectrum, spread="DIM_DYN", stack=0.5)
    step = 0.5 * max(np.abs(y).max() for y in flat)
    assert np.allclose(stacked[2] - flat[2], 2 * step) and np.allclose(stacked[0], flat[0])


def test_a_thumbnail_shows_from_half_a_ppm():
    """A large peak at 0.35 ppm stays out of the picture; one at 2 ppm sits where 0.5-4.2
    ppm, falling, puts it."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    from nifti_mrs_plus import NIfTI_MRS_Plus

    t, sf = np.arange(2048) * 2.5e-4, 123.2
    fid = sum(a * np.exp(2j * np.pi * (p - 4.65) * sf * t - t / 0.05) for a, p in
              [(10, 0.35), (1, 2.0)])
    spectrum = gen_nifti_mrs(fid.reshape(1, 1, 1, -1), 2.5e-4, sf)
    ppm, ys, _ = traces(spectrum)
    assert ppm[np.argmax(ys[0])] == pytest.approx(0.35, abs=0.02)

    y = thumb(NIfTI_MRS_Plus([spectrum]))["y"]

    assert np.argmax(y) / len(y) == pytest.approx((4.2 - 2.0) / (4.2 - 0.5), abs=0.02)


#****************#
#   call lines   #
#****************#
@pytest.mark.parametrize("key", [k for k, b in blocks.BLOCKS.items() if b.call and k != "files"])
def test_a_call_blocks_line_fits_its_function(key):
    """Every field and input given, the line binds to the function it calls."""
    import ast
    import inspect

    graph = Graph()
    b = blocks.BLOCKS[key]
    space = {}
    exec("\n".join(b.imports), space)
    function = eval(b.call, space)
    id = graph.add(key)
    for f in b.fields:
        graph.set(id, f.name, f.choices[0] if f.choices else 1)
    for p in b.inputs:
        key, output = {blocks.RESULTS: ("lcm-fit", "out"), blocks.BASIS: ("basis", "basis"),
                       blocks.SPECTRA: ("ismrm2016", "spectra")}.get(p.kind, ("ismrm2016", "truth"))
        graph.connect(graph.add(key), output, id, p.name)

    call = ast.parse(graph.code(id)[0]).body[0].value
    inspect.signature(function).bind(*call.args, **{k.arg: k.value for k in call.keywords})


#***********#
#   kinds   #
#***********#
def test_what_flows_meets_only_what_takes_it():
    graph = Graph()
    simulated, score = graph.add("simulation"), graph.add("benchmark")
    model, averaging = graph.add("model"), graph.add("averaging")

    with pytest.raises(ValueError, match="cannot go into"):        # MRS-Sim's own truth
        graph.connect(simulated, "truth", score, "truth")
    graph.connect(model, "out", averaging, "data")                  # e.g. WAND's spectra


#****************#
#   invalidate   #
#****************#
def test_a_change_drops_what_every_block_after_it_made():
    graph = Graph()
    data, step, snr, other = (graph.add(k) for k in ("ismrm2016", "averaging", "snr",
                                                       "ismrm2016"))
    graph.connect(data, "spectra", step, "data")
    graph.connect(step, "out", snr, "data")
    for k in (data, step, snr, other):
        graph.nodes[k].made = {}

    graph.set(data, "n", 1)
    assert [graph.nodes[k].made for k in (data, step, snr)] == [None] * 3
    assert graph.nodes[other].made == {}
    with pytest.raises(ValueError, match="loop"):
        graph.connect(snr, "out", step, "data")
    graph.nodes[snr].made = {}
    graph.disconnect(graph.edges[-1])
    assert graph.nodes[snr].made is None, "a wire gone drops what it fed"
    graph.nodes[snr].made = {}
    graph.connect(other, "spectra", snr, "data")
    assert graph.nodes[snr].made is None, "a wire new drops it too"


#************#
#   traces   #
#************#
def test_traces_keep_each_voxel_apart():
    from nifti_mrs.create_nmrs import gen_nifti_mrs

    fids = np.zeros((2, 1, 1, 64), complex)
    fids[1, 0, 0] = np.exp(2j * np.pi * 0.1 * np.arange(64))       # one line, voxel 1 only
    spectrum = gen_nifti_mrs(fids, 1 / 2000, 123.2)

    _, ys, _ = traces(spectrum)

    assert len(ys) == 2 and np.abs(ys[0]).max() == 0 and np.abs(ys[1]).max() > 0


#***********#
#   brief   #
#***********#
def test_an_error_is_said_in_one_line():
    told = RuntimeError("spec2nii failed:\nTraceback (most recent call last):\n  File \"x\", "
                        "line 1\nValueError: no such dimension")

    assert brief(told) == "RuntimeError: spec2nii failed: ValueError: no such dimension"
    octave = RuntimeError("Octave evaluation error:\nerror: Receivers combined!\nerror: called")
    assert brief(octave) == "RuntimeError: Octave evaluation error: error: Receivers combined!"
    assert brief(ValueError("x" * 500)).endswith("…") and len(brief(ValueError("x" * 500))) == 240


#*************#
#   asking   #
#*************#
def test_drawing_never_waits_on_a_toolbox_and_one_that_fails_is_asked_once(monkeypatch):
    """Asking a toolbox can take seconds (R, MRSpecLAB), and one not installed fails each
    time: the canvas draws without asking, and a failure is kept."""
    asked = []

    def module(bridge):
        asked.append(bridge)
        raise RuntimeError(f"{bridge} is not installed here")

    monkeypatch.setattr(catalog, "_ABOUT", {})
    monkeypatch.setattr(catalog, "module", module)
    graph = Graph()
    step = graph.add("averaging")

    assert [p.name for p in graph.ports(step)[0]] == ["data"] and asked == []
    graph.ask(step)
    graph.ask(step)
    graph.ports(step)
    assert asked == ["fsl_mrs"], "asked once, its failure kept"


#***************#
#   on blocks   #
#***************#
def test_a_block_shows_each_argument_left_unwired_and_no_more_than_once():
    pytest.importorskip("nicegui")
    from mrsuite.gui.canvas import view

    graph = Graph()
    data, fit = graph.add("ismrm2016"), graph.add("lcm-fit")
    graph.connect(data, "spectra", fit, "data")
    graph.set(fit, "ppmlim", (1, 3))
    shown = lambda: {p["name"]: p["value"] for p in view(graph, None, {})[0][1]["data"]["inputs"]}

    assert "ppmlim" not in shown(), "set in the form, but not put on the block"
    graph.show(fit, "ppmlim")
    graph.show(fit, "water")
    assert shown()["ppmlim"] == "(1, 3)" and shown()["water"] == "None"
    graph.connect(data, "water", fit, "ppmlim")
    assert shown()["ppmlim"] == "", "a wire gives it, not the form"
    graph.set(fit, catalog.MORE, {"ppmlim": (0, 4)})
    with pytest.raises(ValueError, match="ppmlim is set in its own field or by wire"):
        graph.code(fit)


def test_how_many_outputs_octave_gives_is_not_a_blocks_to_set():
    names = [f.name for f in catalog.fields(catalog.about("fid_a", "op_getSNR")[0])]

    assert "nout" not in names and names[0] == "NAAppmmin"
