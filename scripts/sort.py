import re
import numpy as np
from collections import defaultdict

def parse_formula(formula):
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    counts = defaultdict(int)
    for (element, count) in elements:
        counts[element] += int(count) if count else 1
    return counts

molar_ratios = {
    'CaO': 1,
    'MgO': 1,
    'SiO2': 1,
    'O2': 1,
}

T = 1000
k_B = 8.617333262e-2   ## in meV unit
kT = k_B * T

mu_O2 = kT * np.log(molar_ratios['O2'])
mu_O = 0.5 * mu_O2

chemical_potentials = {
    'Ca': kT * np.log(molar_ratios['CaO'])-mu_O,
    'Mg': kT * np.log(molar_ratios['MgO'])-mu_O,
    'Si':kT * np.log(molar_ratios['SiO2']) -2 * mu_O,
    'O': mu_O,
}

print(chemical_potentials)

with open('ensemble_results.txt', 'r') as f:
    content = f.read()

blocks = content.strip().split('\n\n')

data = []
for block in blocks:
    filename = re.search(r'^(.*?):', block).group(1)
    formula = re.search(r':\s*(.*)', block.splitlines()[0]).group(1)
    raw_energy = float(re.search(r'Raw energy = ([\-\d\.Ee]+)', block).group(1))
    formation_energy = float(re.search(r'Formation Energy = ([\-\d\.Ee]+)', block).group(1))
    energy_std = float(re.search(r'Energy s.t.d. = ([\d\.Ee]+)', block).group(1))
    force_std = float(re.search(r'Mean force s.t.d. = ([\d\.Ee]+)', block).group(1))

    composition = parse_formula(formula)

    correction = sum(chemical_potentials.get(elem, 0.0) * count for elem, count in composition.items())
    n_atoms = sum(composition.values())
    corrected_formation_energy = formation_energy - correction / n_atoms  # correction is already in meV

    entry = {
        'filename': filename,
        'formula': formula,
        'formation_energy': corrected_formation_energy,
        'energy_std': energy_std,
        'force_std': force_std,
    }
    data.append(entry)

filtered_data = [d for d in data if d['energy_std'] <= 40 and d['force_std'] <= 1]
sorted_data = sorted(filtered_data, key=lambda x: x['formation_energy'])

formation_energies = np.array([d['formation_energy'] for d in sorted_data])
boltzmann_factors = np.exp(-formation_energies / kT)
partition_function = np.sum(boltzmann_factors)
probabilities = boltzmann_factors / partition_function

for i, d in enumerate(sorted_data):
    d['boltzmann_prob'] = probabilities[i]

for d in sorted_data:
    print(f"{d['filename']}: {d['formula']} P = {d['boltzmann_prob']:.6f}, Formation Energy = {d['formation_energy']:.3f} meV/atom, Energy s.t.d. = {d['energy_std']} meV/atom, Mean force s.t.d. = {d['force_std']} eV/Å")

