from ase import io
import numpy as np
import os

def apply_shear_strain(atoms, shear_strain):
    cell = atoms.get_cell()
    cell[0, 1] += shear_strain * cell[0, 0]
    atoms.set_cell(cell, scale_atoms=True)

shear_strains = np.linspace(0, 0.1, 11)

# Get list of subdirectories in the current directory
subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]

for subdir in subdirs:
    # Look for a .cif file in the subdir
    cif_files = [f for f in os.listdir(subdir) if f.endswith('.cif')]
    if not cif_files:
        print(f"No .cif file found in {subdir}, skipping.")
        continue

    material_path = os.path.join(subdir, cif_files[0])
    material = io.read(material_path)

    output_dir = os.path.join(subdir, 'shear_strained')
    os.makedirs(output_dir, exist_ok=True)

    for i, shear_strain in enumerate(shear_strains):
        atoms_copy = material.copy()
        apply_shear_strain(atoms_copy, shear_strain)
        output_file = os.path.join(output_dir, f'strain_{i}.cif')
        atoms_copy.write(output_file)

    print(f"Generated shear-strained structures in: {output_dir}")

