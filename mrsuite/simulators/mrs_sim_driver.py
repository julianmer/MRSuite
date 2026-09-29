####################################################################################################
#                                        mrs_sim_driver.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Runs MRS-Sim for MRSuite. Not imported by MRSuite: it runs in a process of its own,     #
#          with the MRS-Sim checkout as its working directory, and is driven by                    #
#          mrsuite.simulators.mrs_sim. It draws the parameters from the config's ranges, runs      #
#          the physics model, and returns the data with its truth, each component taken through    #
#          every step the data is.                                                                 #
#                                                                                                  #
#          Usage: python mrs_sim_driver.py job.json                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import os
import sys

sys.dont_write_bytecode = True           # MRS-Sim keeps its __pycache__ under version control
sys.path.insert(0, os.getcwd())          # the checkout, where src and its basis sets live

import numpy as np
import torch

from src.aux import inv_Fourier_Transform, sample_baselines, sample_resWater
from src.mainFcns import prepare


#************#
#   sample   #
#************#
def sample(cfg, pm, ind, columns, names, n):
    """Every parameter drawn uniformly within the config's ranges, as MRS-Sim scales them.

    The reference metabolites are fixed at their ratios, since MRS-Sim states every
    amplitude relative to them; one Gaussian width is shared by the metabolites and one
    by the macromolecules and lipids, as MRS-Sim's line shape model has it; each other
    basis function is left out with the config's dropout probability; and what the
    config switches off is zero, so the truth does not report it.
    """
    params = pm.quantify_params(torch.rand(n, pm.min_ranges.shape[1]))

    reference = cfg.wrt_metab.split(",")
    ratios = getattr(cfg, "wrt_metab_ratio", None) or [1 / len(reference)] * len(reference)
    for name, ratio in zip(reference, ratios):
        params[:, columns[name.lower()]] = ratio

    background = [i for i, m in enumerate(names) if "mm" in m.lower() or "lip" in m.lower()]
    metabolites = [i for i in range(len(names)) if i not in background]
    widths = ind["g"]
    for group in (metabolites, background):
        if group:
            params[:, [widths[i] for i in group]] = params[:, [widths[group[0]]]]
    if cfg.lineshape == "lorentzian":
        params[:, list(widths)] = 0
    if cfg.lineshape == "gaussian":
        params[:, list(ind["d"])] = 0

    for i, name in enumerate(names):
        if name not in reference:
            params[torch.rand(n) < cfg.drop_prob, columns[name.lower()]] = 0

    off = {"b0": ["b0", "b0_dir"], "eddy": ["ecc"], "fshift_g": ["f_shift"],
           "fshift_i": ["f_shifts"], "noise": ["snr"], "phi0": ["phi0"], "phi1": ["phi1"]}
    for flag, keys in off.items():
        if not getattr(cfg, flag):
            for key in keys:
                params[:, ind[key]] = 0
    return params


#**********#
#   main   #
#**********#
def main(job):
    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"] % 2 ** 32)       # one of MRS-Sim's samplers draws from NumPy
    path = os.path.join(job["folder"], "config.json")
    with open(path, "w") as fh:
        json.dump({**job["config"], "totalEntries": job["n"]}, fh)
    cfg, water_cfg, baseline_cfg, pm, l, ind, p, n = prepare(path)
    # every basis function the model sums, in its order, from the model itself: prepare
    # adds 'mac' and 'lip' group keys to ind, which clobber a basis function of that name
    shown = {m.lower(): m for m in job["config"]["metabolites"]}
    names = [shown.get(m.lower(), m) for m in pm._metab[:len(ind["metabolites"])]]
    columns = {m.lower(): i for m, i in zip(names, ind["metabolites"])}

    params = sample(cfg, pm, ind, columns, names, n)
    original = params.clone()               # forward scales some in place

    kept = {}
    summing = pm.line_summing

    def line_summing(fid, params, mm):
        kept["lines"] = fid.clone()         # each basis function, just before the sum
        return summing(fid=fid, params=params, mm=mm)

    pm.line_summing = line_summing
    data, fit, baseline, water, _, quantities, _ = pm.forward(
        params=params, diff_edit=None, b0=cfg.b0, gen=False, eddy=cfg.eddy, fids=True,
        phi0=cfg.phi0, phi1=cfg.phi1, noise=cfg.noise, apodize=cfg.apodize,
        offsets=bool(baseline_cfg or water_cfg), fshift_g=cfg.fshift_g, fshift_i=cfg.fshift_i,
        resample=False, baselines=sample_baselines(n, **baseline_cfg) if baseline_cfg else False,
        coil_phi0=False, coil_sens=False, magnitude=False, multicoil=1, zero_fill=False,
        broadening=cfg.broadening, coil_fshift=False,
        residual_water=sample_resWater(n, **water_cfg) if water_cfg else False,
        drop_prob=cfg.drop_prob, presim={"baseline": None, "reswater": None})

    def complex_(x):
        x = torch.as_tensor(np.asarray(x))
        return (x[..., 0, :] + 1j * x[..., 1, :]).numpy()

    def after_summing(fid):
        """The steps forward applies to the sum, in its order, with the same parameters."""
        if cfg.phi0:
            fid = pm.zero_order_phase(fid=fid, phi0=original[:, ind["phi0"]])
        if cfg.phi1:
            fid = pm.first_order_phase(fid=fid, phi1=original[:, ind["phi1"]])
        if cfg.fshift_g:
            fid = pm.frequency_shift(fid=fid, param=original[:, ind["f_shift"]].clone())
        if cfg.eddy:
            fid = pm.first_order_eddy_currents(fid=fid, params=original[:, ind["ecc"]])
        if cfg.apodize:
            fid = pm.apodization(fid=fid, hz=cfg.apodize)
        return fid

    lines = kept["lines"]
    summed, clean_fit = complex_(lines.sum(1)), complex_(fit)[:, -1]
    scale = (np.sum(summed * clean_fit.conj(), -1) /
             np.sum(np.abs(clean_fit) ** 2, -1)).real          # forward's normaliser, recovered

    # the second copy: the first carries the batch's first signal for every spectrum, as
    # forward indexes a (batch, channel, time) tensor with [..., 0, :, :]
    noisy = complex_(data)[:, -1]
    components = {m: c for m, c in zip(names, complex_(after_summing(lines)).transpose(1, 0, 2)
                                       / scale[:, None])}
    for name, offset in (("baseline", baseline), ("water", water)):
        if offset is not None:              # added as spectra, so returned as spectra
            fid = inv_Fourier_Transform(torch.as_tensor(np.asarray(offset)))
            components[name] = complex_(after_summing(fid))[:, 0]
    components["noise"] = noisy - sum(components.values())

    # the parameters by MRS-Sim's own index names; per-line ones by their basis function
    labels = [f"p{i}" for i in range(original.shape[1])]
    for key, where in ind.items():
        if isinstance(where, int):
            labels[where] = key
        elif key in ("d", "g", "f_shifts") and len(where) == len(names):
            for m, i in zip(names, where):
                labels[i] = f"{key}:{m}"
        elif key in ("b0_dir", "ecc", "coil_snr", "coil_sens", "coil_fshift", "coil_phi0"):
            for j, i in enumerate(where):
                labels[i] = f"{key}{j}"
    for name, i in columns.items():
        labels[i] = name                    # ind's 'mac'/'lip' keys may hide one

    np.savez(job["result"], data=noisy, names=np.array(list(components)),
             labels=np.array(labels),
             components=np.stack(list(components.values()), 1) if job["components"] else [],
             metabolites=np.array(names),
             concentrations=original[:, list(ind["metabolites"])].numpy(),
             parameters=original.numpy(), snr=original[:, ind["snr"]].numpy(),
             range_min=pm.min_ranges.numpy().ravel(), range_max=pm.max_ranges.numpy().ravel(),
             dwell=float(np.asarray(pm.header["dwelltime"]).ravel()[0]),
             frequency=float(np.asarray(pm.header["carrier_frequency"]).ravel()[0]),
             centre=float(np.asarray(pm.header["centerFreq"]).ravel()[0]))


if __name__ == "__main__":
    with open(sys.argv[1]) as fh:
        main(json.load(fh))
