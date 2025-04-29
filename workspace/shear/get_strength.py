import os
import numpy as np
import re
from ase.io import read
from scipy.interpolate import UnivariateSpline

for folder in os.listdir():
    full_path = os.path.join(os.getcwd(), folder)
    if not os.path.isdir(full_path):
        continue

    results_path = os.path.join(full_path, "ensemble_results.txt")
    if not os.path.exists(results_path):
        print(f"Skipping {folder} — no ensemble_results.txt found.")
        continue

    energies = []
    filenames = []

    with open(results_path, "r") as f:
        prev_line = ""
        current_filename = ""
        energy = None
        energy_std = None
        force_std = None

        for line in f:
            line = line.strip()

            if line.startswith("strain_"):
                current_filename = line.split(':')[0].strip()

            elif line.startswith("Raw energy"):
                if prev_line.startswith("gen"):
                    continue
                energy = float(line.split('=')[1].split()[0])

            elif line.startswith("Energy s.t.d."):
                energy_std = float(line.split('=')[1].split()[0])

            elif line.startswith("Mean force s.t.d."):
                force_std = float(line.split('=')[1].split()[0])

                if energy is not None and energy_std is not None and force_std is not None:
                    if energy_std < 60 and force_std < 1.5:
                        filenames.append(os.path.join(full_path, current_filename))
                        energies.append(energy)

                energy = None
                energy_std = None
                force_std = None

            prev_line = line

    if len(energies) < 4:
        print(f"Skipping {folder} — not enough valid data points (found {len(energies)})")
        continue

    def shear_index(filename):
        return int(re.search(r"strain_(\d+)", filename).group(1))

    sorted_pairs = sorted(zip(filenames, energies), key=lambda x: shear_index(x[0]))
    filenames, energies = zip(*sorted_pairs)
    energies = np.array(energies)

    shear_strains = np.linspace(0, 0.1, len(energies))

    atoms = read(filenames[0])
    volume = atoms.get_volume()

    spline = UnivariateSpline(shear_strains, energies, s=0)
    strain_fine = np.linspace(shear_strains[0], shear_strains[-1], 1000)
    stress = spline.derivative()(strain_fine) / volume

    shear_strength = np.max(stress)
    strain_at_max = strain_fine[np.argmax(stress)]

    with open(os.path.join(full_path, "shear_strength.txt"), "w") as out:
        out.write(f"Shear strength: {shear_strength:.6f} eV/Å³ ({shear_strength * 160.2:.2f} GPa)\n")
        out.write(f"At shear strain: {strain_at_max:.6f}\n")

    print(f"{folder}: Shear strength = {shear_strength:.6f} eV/Å³")

