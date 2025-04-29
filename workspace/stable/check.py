import glob
import re

energy_pattern = re.compile(r"Energy s\.t\.d\.\s*=\s*([-\d\.]+)\s*meV/atom", re.IGNORECASE)
force_pattern = re.compile(r"Mean force s\.t\.d\.\s*=\s*([-\d\.]+)\s*eV/Å", re.IGNORECASE)

files = glob.glob("out.*")

for file in files:
    with open(file, 'r') as f:
        content = f.read()

    energy_matches = [float(m) for m in energy_pattern.findall(content)]
    force_matches = [float(m) for m in force_pattern.findall(content)]

    if energy_matches and force_matches:
        max_energy_std = max(energy_matches)
        max_force_std = max(force_matches)
        print(f"📄 {file} -> Max Energy s.t.d.: {max_energy_std:.6f} meV/atom, Max Force s.t.d.: {max_force_std:.6f} eV/Å")

