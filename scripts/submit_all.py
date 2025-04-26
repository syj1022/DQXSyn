import os
import subprocess

base_path = os.getcwd()

for i in range(100):
    folder = f"{i:02d}"
    folder_path = os.path.join(base_path+'/dft', folder)

    if os.path.isdir(folder_path):
        print(f"Submitting job in {folder_path}")
        subprocess.run(["sbatch", "/home/yingjies/scripts/chestnut_qe7.sub"], cwd=folder_path)
    else:
        print(f"{folder_path} does not exist. Skipping.")

