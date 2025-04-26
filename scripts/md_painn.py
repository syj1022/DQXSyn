import os
import numpy as np
from nff.io.ase import AtomsBatch
from nff.io.ase_calcs import NeuralFF, EnsembleNFF
from mcmc import MCMC
import torch
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from mcmc.calculators import EnsembleNFFSurface
from ase.io import read
from ase import units
from ase.md.langevin import Langevin

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_SHM_DISABLE"] = "1"
DEVICE = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}" if torch.cuda.is_available() else "cpu"

nnids = [str(i) for i in range(10)]

base_name = 'bulk'
base_ml_dir = f"/scratch/alevoj1/YingjieS/DQX/MLIP/{base_name}"
model_dirs = [os.path.join(base_ml_dir, str(x), "best_model") for x in nnids]

models = []
for modeldir in model_dirs:
    m = NeuralFF.from_file(modeldir, device=DEVICE).model
    models.append(m)

nff_surf_calc = EnsembleNFFSurface(models, device=DEVICE)

ase_atoms = read("init.traj")
positions = ase_atoms.get_positions()
numbers = ase_atoms.get_atomic_numbers()
cell = ase_atoms.cell.array

atoms = AtomsBatch(
    positions=positions,
    numbers=numbers,
    cell=cell,
    pbc=True,
    cutoff=5.0,
    props={'energy': 0, 'energy_grad': []},
    calculator=nff_surf_calc,
    directed=True,
    device=DEVICE
)
_ = atoms.update_nbr_list()

dyn = Langevin(atoms, 0.5*units.fs, temperature_K=1000, friction=0.01)

def write_frame():
    dyn.atoms.write('final.xyz', append=True)

dyn.attach(write_frame, interval=1)
dyn.run(2000)
print("MD finished!")


