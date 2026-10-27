####################################################################################################
#                                         test_agnostic.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: AGNOSTIC's spectra read the right way round, on their own grids, with each metabolite's #
#          amplitude found where AGNOSTIC put it; on a shard already downloaded.                   #
#                                                                                                  #
####################################################################################################
import numpy as np
import pytest

from mrsuite.datasets import agnostic

SHARD = 21
needs_shard = pytest.mark.skipif(
    not any((agnostic.DATA / f"data{SHARD}.{s}").is_file() for s in ("tar", "tgz")),
    reason=f"AGNOSTIC's data{SHARD} is not downloaded")


@needs_shard
def test_agnostic_spectra_come_on_their_own_grids_the_right_way_round():
    data, truth = agnostic.load(SHARD, n=3, components=("Metab",))

    assert len(data.list()) == 3 and len(truth) == 3
    for item, amounts in zip(data.list(), truth):
        fid = np.asarray(item[:]).ravel()
        ppm = agnostic.CENTRE + np.fft.fftshift(np.fft.fftfreq(fid.size, item.dwelltime)) / \
            item.spectrometer_frequency[0]
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid)))
        band = lambda centre: spectrum[np.abs(ppm - centre) < 0.1].max()
        # its aromatic spins put some signal near the mirror too, so twice, not tenfold
        assert band(2.01) > 2 * band(2 * agnostic.CENTRE - 2.01), "NAA, not its mirror"
        assert 1.4 * agnostic.GAMMA <= item.spectrometer_frequency[0] <= 3.1 * agnostic.GAMMA
        assert 0.010 <= item.hdr_ext["EchoTime"] <= 0.080
        assert amounts["NAA"] > 0 and {"Cre", "PCr", "Glu", "MM092"} <= set(amounts)


@needs_shard
def test_every_spin_of_a_metabolite_carries_its_one_amplitude():
    # the order MRSuite rebuilds from metab_dict must be AGNOSTIC's: shifted by one, a
    # metabolite's spins would read neighbours' amplitudes, which differ
    import tarfile
    path = next(agnostic.DATA / f"data{SHARD}.{s}" for s in ("tar", "tgz")
                if (agnostic.DATA / f"data{SHARD}.{s}").is_file())
    with tarfile.open(path) as archive:
        member = next(m for m in archive if m.name.endswith(".npz"))
        amplitudes = np.load(archive.extractfile(member))["Amplitude"]
    starts = sorted(agnostic.metabolites().values())
    assert amplitudes.shape[1] == 182
    for start, end in zip(starts, starts[1:] + [amplitudes.shape[1]]):
        assert np.all(amplitudes[:, start:end] == amplitudes[:, [start]]), start
