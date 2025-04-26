import numpy as np
import pandas as pd
from ase.io import read
from ase.io.trajectory import Trajectory
import os
import subprocess
import glob

name = input('Enter name: ')
csv_file = f"{name}.csv"

for i in range(100):
    folder = f"{i:02d}"

    opt_path = os.path.join("dft", folder, "rlx.traj")

    try:
        traj = Trajectory(opt_path)
        cell = read(opt_path).get_cell()
        formula = read(opt_path).get_chemical_formula()

        n_samples = int(0.7 * len(traj))  ## only sample a part of full data
        sampled_indices = np.random.choice(len(traj), size=n_samples, replace=False)

        for idx in sampled_indices:
            a = traj[idx]
            energy = a.get_potential_energy()
            forces = a.get_forces()
            n = a.get_atomic_numbers()
            xyz = a.get_positions()
            nxyz = np.column_stack([n, xyz])

            df = pd.DataFrame({
                'name': [name],
                'formula': [formula],
                'lattice': [cell.tolist()],
                'nxyz': [nxyz.tolist()],
                'force': [forces.tolist()],
                'energy': [energy]
            })

            df.to_csv(csv_file, mode='a', index=False, header=False)

        print(f"✓ Added data from {folder}")

    except Exception as e:
        print(f"Error processing {folder}: {e}")

