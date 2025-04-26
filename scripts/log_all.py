import subprocess
import os
import glob
from ase.io import read

os.chdir("dft")
for i in range(100):
    os.chdir(f"{i:02d}")
    cif_file = glob.glob("*.cif")[0]
    a = read(cif_file)
    a.write("init.traj")
    subprocess.run(["python", "/home/yingjies/scripts/log2traj_forces.py", "pw.out", "rlx.traj"])
    os.chdir("../")
os.chdir("../")
