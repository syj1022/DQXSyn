import glob
import numpy as np
from ase import Atoms
from ase.io import write

input_files = sorted(glob.glob("supercell-*.in"))

for infile in input_files:
    num = infile.split("-")[1].split(".")[0]
    outfile = f"init-{num}.traj"

    with open(infile, "r") as f:
        lines = f.readlines()

    cell = []
    positions = []
    symbols = []

    reading_cell = False
    reading_positions = False

    for line in lines:
        line = line.strip()

        if line.startswith("CELL_PARAMETERS"):
            reading_cell = True
            reading_positions = False
            continue
        elif line.startswith("ATOMIC_POSITIONS"):
            reading_positions = True
            reading_cell = False
            continue
        elif any(line.startswith(s) for s in ["ATOMIC_SPECIES", "&", "K_POINTS", "SYSTEM", "ELECTRONS"]):
            reading_cell = False
            reading_positions = False
            continue
        elif line == "" or line.startswith("!"):
            continue

        if reading_cell:
            try:
                cell.append([float(x) for x in line.split()])
            except ValueError:
                reading_cell = False
                continue

        elif reading_positions:
            parts = line.split()
            if len(parts) >= 4:
                symbols.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])

    cell = np.array(cell) * 0.529177
    atoms = Atoms(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
    write(outfile, atoms)
    print(f"✅ Successfully wrote: {outfile}")

