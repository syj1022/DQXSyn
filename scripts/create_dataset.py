import sys
import os
import re
import numpy as np
from ast import literal_eval
import torch
import nff.data as d
import pandas as pd
from collections import defaultdict
from nff.data.dataset import Dataset, split_train_validation_test, stratified_split
import pickle

data = ["gen15"]

def data_to_tensor(data):
    converted_matrices = []
    for data_string in data:
        cleaned_data_string = re.sub(r'(?<![\[\],])\s+', ', ', data_string)
        matrix = literal_eval(cleaned_data_string)
        converted_matrices.append(matrix)
    tensor_list = [torch.tensor(matrix) for matrix in converted_matrices]
    return tensor_list

def calculate_energy_correction(formula):
    pattern = r"([A-Z][a-z]*)(\d*)"
    counts = defaultdict(int)
    for element, count in re.findall(pattern, formula):
        counts[element] += int(count) if count else 1
    energy_dict = {"Mg": -1092.297413, "Ca": -1029.314503, "O": -460.868328, "Si": -168.304750, "H": -16.471143}
    total_energy_correction = sum(count * energy_dict[element] * 23.060542 for element, count in counts.items())
    return total_energy_correction

for da in data:
    data = pd.read_csv(f"/scratch/alevoj1/YingjieS/DQX/data/{da}/data.csv")

    formula, lattice, nxyz, force, energy = data['formula'], data['lattice'].values, data['nxyz'].values, data['force'].values, data['energy'].values

    formula_data = formula.values
    lattice_data, nxyz_data, force_data = map(data_to_tensor, [lattice, nxyz, force])
    energy_data = energy.squeeze() * 23.060542  # Convert to kcal/mol

    energy_data -= [calculate_energy_correction(f) for f in formula_data]

    props = {
        'formula': formula_data.tolist(),
        'lattice': lattice_data,
        'nxyz': nxyz_data,
        'energy': energy_data.tolist(),
        'energy_grad': [-f for f in force_data]
    }

    dataset = d.Dataset(props.copy(), units='kcal/mol')
    dataset.generate_neighbor_list(cutoff=5)

    dataset.save(f"/scratch/alevoj1/YingjieS/DQX/data/{da}/full_dataset.pth.tar")

    train_dset, val_dset, test_dset = split_train_validation_test(dataset, stratified=True, targ_name='formula', val_size=0.2, test_size=0.1, seed=11)
    final_dsets = [dataset.copy(), train_dset.copy(), val_dset.copy(), test_dset.copy()]
    dset_labels = ['full', 'train', 'val', 'test']

    for dset, label in zip(final_dsets, dset_labels):
        dset.save(f"/scratch/alevoj1/YingjieS/DQX/data/{da}/{label}_dataset.pth.tar")

