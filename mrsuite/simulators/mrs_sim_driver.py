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
#          mrsuite.simulators.mrs_sim. It draws the parameters with MRS-Sim's own sampler, runs    #
#          the physics model, and returns the data with its truth, each component taken through    #
#          every step the data is, and MRS-Sim's own provenance and MRSsynMRS table.               #
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

from dataclasses import asdict

import numpy as np
import torch

from src.aux import inv_Fourier_Transform, sample_baselines, sample_resWater
from src.mainFcns import prepare
from src.mrssynmrs import export_mrssynmrs_table
from src.provenance import collect_provenance
from src.sampling import UniformRangeSampler


#************#
#   sample   #
#************#
def sample(cfg, pm, ind, columns, names, n, sampler):
    """Every parameter drawn by MRS-Sim's own sampler, then the dataset's own choices.

    The reference metabolites are fixed at their ratios, since MRS-Sim states every
    amplitude relative to them; one Gaussian width is shared by the metabolites and one
    by the macromolecules and lipids, as MRS-Sim's line shape model has it; each other
    basis function is left out with the config's dropout probability; and what the
    config switches off is zero, so the truth does not report it. MRS-Sim leaves these
    to the caller, as per-dataset choices.
    """
    params = sampler.sample(n).tensor

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
    sampler = UniformRangeSampler(pm, seed=job["seed"])
    # the basis functions by position: prepare adds 'mac' and 'lip' group keys to ind,
    # which hide a basis function of that name, so no column is looked up by its name
    names = sampler.registry.metabolite_names
    amplitudes = list(ind["metabolites"])
    columns = {m.lower(): i for m, i in zip(names, amplitudes)}
    others = [i for i in range(len(sampler.registry.labels())) if i not in set(amplitudes)]

    params = sample(cfg, pm, ind, columns, names, n, sampler)
    original = params.clone()               # forward scales some in place

    kept = {}
    summing = pm.line_summing

    def line_summing(fid, params, mm):
        kept["lines"] = fid.clone()         # each basis function, just before the sum
        return summing(fid=fid, params=params, mm=mm)

    pm.line_summing = line_summing
    result = pm.forward(
        params=params, diff_edit=None, b0=cfg.b0, gen=False, eddy=cfg.eddy, fids=True,
        phi0=cfg.phi0, phi1=cfg.phi1, noise=cfg.noise, apodize=cfg.apodize,
        offsets=bool(baseline_cfg or water_cfg), fshift_g=cfg.fshift_g, fshift_i=cfg.fshift_i,
        resample=False, baselines=sample_baselines(n, **baseline_cfg) if baseline_cfg else False,
        coil_phi0=False, coil_sens=False, magnitude=False, multicoil=1, zero_fill=False,
        broadening=cfg.broadening, coil_fshift=False,
        residual_water=sample_resWater(n, **water_cfg) if water_cfg else False,
        drop_prob=cfg.drop_prob, presim={"baseline": None, "reswater": None},
        return_components=True)

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
    summed, clean_fit = complex_(lines.sum(1)), complex_(result.nuisance_free)
    scale = (np.sum(summed * clean_fit.conj(), -1) /
             np.sum(np.abs(clean_fit) ** 2, -1)).real          # forward's normaliser, recovered

    # with noise off MRS-Sim gives no noisy spectrum, and the data is the noise-free one
    noisy = complex_(result.noise_free_total if result.noisy is None else result.noisy)
    components = {m: c for m, c in zip(names, complex_(after_summing(lines)).transpose(1, 0, 2)
                                       / scale[:, None])}
    for name, offset in (("baseline", result.baseline), ("water", result.residual_water)):
        if offset is not None:              # added as spectra, so returned as spectra
            fid = inv_Fourier_Transform(torch.as_tensor(np.asarray(offset)))
            components[name] = complex_(after_summing(fid))[:, 0]
    # what the data holds beyond them: MRS-Sim's own noise field is the noise as drawn,
    # before the global phase, shift and eddy currents reach it
    components["noise"] = noisy - sum(components.values())

    enabled = {key: bool(getattr(cfg, key)) for key in
               ("b0", "eddy", "phi0", "phi1", "fshift_g", "fshift_i", "noise", "apodize")}
    enabled.update(baseline=bool(baseline_cfg), residual_water=bool(water_cfg))
    record = collect_provenance(pm, config=cfg, enabled_components=enabled, sampler=sampler)
    table = export_mrssynmrs_table(pm, record, config=cfg, sampler=sampler)
    labels = sampler.registry.labels()
    low, high = pm.min_ranges.numpy().ravel(), pm.max_ranges.numpy().ravel()

    np.savez(job["result"], data=noisy, names=np.array(list(components)),
             components=np.stack(list(components.values()), 1) if job["components"] else [],
             metabolites=np.array(names),
             # each basis function's amplitude in the data: MRS-Sim samples ratios and
             # normalises every spectrum after, so the ratios are divided by that normaliser
             concentrations=original[:, amplitudes].numpy() / scale[:, None],
             concentration_ranges=np.stack([low[amplitudes], high[amplitudes]], 1),
             labels=np.array([labels[i] for i in others]),
             parameters=original[:, others].numpy(),
             parameter_ranges=np.stack([low[others], high[others]], 1),
             snr=original[:, ind["snr"]].numpy(),
             mrssynmrs=json.dumps(table, default=str),
             provenance=json.dumps(asdict(record), default=str),
             dwell=float(np.asarray(pm.header["dwelltime"]).ravel()[0]),
             frequency=float(np.asarray(pm.header["carrier_frequency"]).ravel()[0]),
             centre=float(np.asarray(pm.header["centerFreq"]).ravel()[0]))


if __name__ == "__main__":
    with open(sys.argv[1]) as fh:
        main(json.load(fh))
