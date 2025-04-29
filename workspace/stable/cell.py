import os
from ase import io

base_dir = '.'  # Adjust the path

for folder_num in range(200):
    folder_name = f"{folder_num:02d}"  # Folder names: 00, 01, ..., 199
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.cif'):
                structure = io.read(os.path.join(folder_path, file_name))
                print(f"Cell for {folder_name}/{file_name}: \n{structure.get_cell()}")

