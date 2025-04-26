import os
import shutil
import subprocess

input_file = "ensemble_results.txt"
cif_folder = "generated"
relax_script = "/scratch/alevoj1/YingjieS/DQX/scripts/relax_bulk.py"

structures = []

with open(input_file, "r") as infile:
    lines = infile.readlines()
    for i in range(0, len(lines), 5):
        cif_line = lines[i].strip()
        energy_std_line = lines[i+2].strip()
        force_std_line = lines[i+3].strip()

        cif_name = cif_line.split(":")[0].strip()
        energy_std = float(energy_std_line.split("=")[-1].strip().split()[0])
        force_std = float(force_std_line.split("=")[-1].strip().split()[0])

        if energy_std > 40 or force_std > 1:
            structures.append((cif_name, force_std))

structures.sort(key=lambda x: x[1], reverse=True)

for idx, (cif_name, force_std) in enumerate(structures):
    folder_name = f"dft/{idx:02d}"
    os.makedirs(folder_name, exist_ok=True)

    cif_file = os.path.join(cif_folder, cif_name)
    dst_cif = os.path.join(folder_name, os.path.basename(cif_name))
    shutil.copy(cif_file, dst_cif)

    subprocess.run([
        "python", relax_script,
        "--input", dst_cif
    ], cwd=folder_name)

