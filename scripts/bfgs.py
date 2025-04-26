import sys
import os
import numpy as np
import torch
from nff.io.ase_calcs import NeuralFF, EnsembleNFF
from nff.io.ase import AtomsBatch
from ase.optimize import BFGS
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from mcmc.calculators import EnsembleNFFSurface
from ase.io import read

DEVICE = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}" if torch.cuda.is_available() else "cpu"

nnids = [str(i) for i in range(10)]

base_name = os.path.basename(os.getcwd())
base_ml_dir = f"/scratch/alevoj1/YingjieS/DQX/MLIP/{base_name}"
model_dirs = [os.path.join(base_ml_dir, str(x), "best_model") for x in nnids]

models = [NeuralFF.from_file(modeldir, device=DEVICE).model for modeldir in model_dirs]

calc = EnsembleNFF(models, device=DEVICE)

energy_dict = {"Mg": -1092.297413, "Ca": -1029.314503, "O": -460.868328, "Si": -168.304750, "H": -16.471143}

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
start = task_id * 20
end = start + 20

for i in range(start, end):
    filename = f"/scratch/alevoj1/YingjieS/DQX/mattergen/{base_name}/generated/gen_{i}.cif"
    if not os.path.exists(filename):
        print(f"{filename} not found. Skipping.")
        continue

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

    opt = BFGS(bulk, trajectory=f'/scratch/alevoj1/YingjieS/DQX/mattergen/{base_name}/opt/opt_{i}.traj')
    opt.run(fmax=0.05, steps=200)

    energy_std = calc.results.get("energy_std", None)/len(bulk)*1000
    forces_std = calc.results.get("forces_std", None)
    force_std_mean = np.mean(np.linalg.norm(forces_std, axis=1))

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

    print(f"gen_{i}.cif: {bulk.get_chemical_formula()}")
    print(f"Raw energy = {corrected_energy:.6f} eV")
    print(f"Energy s.t.d. = {float(np.mean(energy_std)):.6f} meV/atom")
    print(f"Mean force s.t.d. = {force_std_mean:.6f} eV/Å")
    print("\n")

    with open("ensemble_results.txt", "a") as f:
        f.write(f"gen_{i}.cif: {bulk.get_chemical_formula()}\n")
        f.write(f"Raw energy = {corrected_energy:.6f} eV\n")
        f.write(f"Energy s.t.d. = {float(np.mean(energy_std)):.6f} meV/atom\n")
        f.write(f"Mean force s.t.d. = {force_std_mean:.6f} eV/Å\n")
        f.write("\n")
