import os
import numpy as np
import torch
from glob import glob
from ase.io import read
from ase.optimize import BFGS
from nff.io.ase_calcs import NeuralFF, EnsembleNFF
from nff.io.ase import AtomsBatch
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from mcmc.calculators import EnsembleNFFSurface

DEVICE = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}" if torch.cuda.is_available() else "cpu"

nnids = [str(i) for i in range(10)]
base_name = 'gen9'
base_ml_dir = f"/scratch/alevoj1/YingjieS/DQX/MLIP/{base_name}"
model_dirs = [os.path.join(base_ml_dir, str(x), "best_model") for x in nnids]
models = [NeuralFF.from_file(modeldir, device=DEVICE).model for modeldir in model_dirs]
calc = EnsembleNFF(models, device=DEVICE)

energy_dict = {
    "Mg": -1092.297413,
    "Ca": -1029.314503,
    "O": -460.868328,
    "Si": -168.304750,
    "H": -16.471143
}

traj_files = sorted(glob("init-*.traj"))

for filename in traj_files:
    print(f"📂 Processing: {filename}")

    atoms = read(filename)
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    cell = atoms.get_cell()

    bulk = AtomsBatch(
        positions=positions,
        numbers=numbers,
        cell=cell,
        pbc=True,
        cutoff=5.0,
        props={'energy': 0, 'energy_grad': []},
        calculator=calc,
        directed=True,
        device=DEVICE
    )
    bulk.update_nbr_list()
    bulk.set_calculator(calc)

    traj_name = filename.replace("init", "opt")
    opt = BFGS(bulk, trajectory=traj_name)
    opt.run(fmax=0.05, steps=100)

    energy_std = calc.results.get("energy_std", None)
    if energy_std is not None:
        energy_std = energy_std / len(bulk) * 1000  # meV/atom

    forces_std = calc.results.get("forces_std", None)
    force_std_mean = np.mean(np.linalg.norm(forces_std, axis=1)) if forces_std is not None else 0.0

    symbols = bulk.get_chemical_symbols()
    num_O = symbols.count("O")
    num_Mg = symbols.count("Mg")
    num_Ca = symbols.count("Ca")
    num_H = symbols.count("H")
    num_Si = symbols.count("Si")

    corrected_energy = float(
        bulk.get_potential_energy()
        + (num_O * energy_dict["O"] +
           num_Mg * energy_dict["Mg"] +
           num_Ca * energy_dict["Ca"] +
           num_H * energy_dict["H"] +
           num_Si * energy_dict["Si"])
    )

    print(f"🧪 Formula: {bulk.get_chemical_formula()}")
    print(f"⚡ Raw energy = {corrected_energy:.6f} eV")
    print(f"📉 Energy s.t.d. = {float(np.mean(energy_std)):.6f} meV/atom")
    print(f"🎯 Mean force s.t.d. = {force_std_mean:.6f} eV/Å")
    print("✅ Done.\n")

