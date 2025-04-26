import os
from glob import glob
from ase.io import read
import yaml

with open("phonopy_disp.yaml", "r") as yf:
    phonopy_data = yaml.safe_load(yf)

displacements = phonopy_data.get("displacements", [])

traj_files = sorted(glob("opt-*.traj"))
output_file = "FORCE_SETS"

with open(output_file, "w") as f:
    atoms = read(traj_files[0], index=-1)
    total_atoms = len(atoms)

    f.write(f"{total_atoms}\n{len(traj_files)}\n\n")

    for i, filename in enumerate(traj_files):
        atoms = read(filename, index=-1)
        forces = atoms.get_forces()

        if i < len(displacements):
            atom_index = displacements[i]["atom"]
            f.write(f"{atom_index}\n")

            disp = displacements[i]["displacement"]
            f.write("  {0:.10f}  {1:.10f}  {2:.10f}\n".format(*disp))
        else:
            f.write(f"  0\n")
            f.write("  {0:.10f}  {1:.10f}  {2:.10f}\n".format(0.0, 0.0, 0.0))

        for force in forces:
            fx, fy, fz = [x / 25.711043 for x in force]  # Convert eV/Å to Ry/Bohr
            f.write(f"  {fx: .10f}  {fy: .10f}  {fz: .10f}\n")
        f.write("\n")

