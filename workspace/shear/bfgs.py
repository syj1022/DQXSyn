import os
import torch
import numpy as np
from ase.io import read
from ase.optimize import BFGS
from nff.io.ase_calcs import NeuralFF, EnsembleNFF
from nff.io.ase import AtomsBatch
from nff.utils.cuda import cuda_devices_sorted_by_free_mem

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
num_batches = 10

DEVICE = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}" if torch.cuda.is_available() else "cpu"

nnids = [str(i) for i in range(10)]
base_name = 'gen15'
base_ml_dir = f"/scratch/alevoj1/YingjieS/DQX/MLIP/{base_name}"
model_dirs = [os.path.join(base_ml_dir, str(x), "best_model") for x in nnids]

models = [NeuralFF.from_file(modeldir, device=DEVICE).model for modeldir in model_dirs]

calc = EnsembleNFF(models, device=DEVICE)

energy_dict = {"Mg": -1092.297413, "Ca": -1029.314503, "O": -460.868328, "Si": -168.304750, "H": -16.471143}

folders = sorted([f for f in os.listdir('.') if os.path.isdir(f)])
batch_size = len(folders) // num_batches
start_idx = task_id * batch_size
end_idx = (task_id + 1) * batch_size if task_id < num_batches - 1 else len(folders)

for folder in folders[start_idx:end_idx]:
    folder_path = os.path.join('.', folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.cif'):
            file_path = os.path.join(folder_path, filename)

            atoms = read(file_path)

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

            opt = BFGS(bulk)
            opt.run(fmax=0.05, steps=200)

            energy_std = calc.results.get("energy_std", None) / len(bulk) * 1000
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

            print(f"{filename}: {bulk.get_chemical_formula()}")
            print(f"Raw energy = {corrected_energy:.6f} eV")
            print(f"Energy s.t.d. = {float(np.mean(energy_std)):.6f} meV/atom")
            print(f"Mean force s.t.d. = {force_std_mean:.6f} eV/Å")
            print("\n")

            with open(f"{folder}/ensemble_results.txt", "a") as f:
                f.write(f"{filename}: {bulk.get_chemical_formula()}\n")
                f.write(f"Raw energy = {corrected_energy:.6f} eV\n")
                f.write(f"Energy s.t.d. = {float(np.mean(energy_std)):.6f} meV/atom\n")
                f.write(f"Mean force s.t.d. = {force_std_mean:.6f} eV/Å\n")
                f.write("\n")

