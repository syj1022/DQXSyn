import os
import re
import numpy as np
from collections import defaultdict
import streamlit as st
from ase.io import read

print("Current working directory:", os.getcwd())

def parse_formula(formula):
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    counts = defaultdict(int)
    for (element, count) in elements:
        counts[element] += int(count) if count else 1
    return counts

def get_G_corr(filename, T):
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip().startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        temp = float(parts[0])
                        if temp == T:
                            return float(parts[1])
                    except ValueError:
                        continue
    except FileNotFoundError:
        pass
    return 0.0

def get_O_G_corr(T):
    from ase.thermochemistry import IdealGasThermo
    thermo_h2 = IdealGasThermo(
        vib_energies=[0.0049848, 0.012958, 0.0161448, 0.5484644],
        potentialenergy=-32.9423,
        atoms=read('workspace/H2/opt.traj'),
        geometry='linear', symmetrynumber=2, spin=0)

    thermo_h2o = IdealGasThermo(
        vib_energies=[0.02377, 0.02554, 0.20178, 0.46406, 0.47837],
        potentialenergy=-496.2706,
        atoms=read('workspace/H2O/opt.traj'),
        geometry='nonlinear', symmetrynumber=2, spin=0)

    return thermo_h2o.get_gibbs_energy(T, 101325, verbose=False) - thermo_h2.get_gibbs_energy(T, 101325, verbose=False) + 2.46 + 460.8683
  
def load_sorted_data(T, molar_ratios):
    k_B = 8.617333262e-2  # meV/K
    kT = k_B * T

    mu_O2 = kT * np.log(molar_ratios.get('O2', 1))
    mu_O = 0.5 * mu_O2

    chemical_potentials = {
        'Ca': kT * np.log(molar_ratios.get('CaO',1)) - mu_O,
        'Mg': kT * np.log(molar_ratios.get('MgO',1)) - mu_O,
        'Si': kT * np.log(molar_ratios.get('SiO2',1)) - 2 * mu_O,
        'O': mu_O,
    }

    G_corr = {
        'Ca': get_G_corr('workspace/ref/cao_mapping.txt', T),
        'Mg': get_G_corr('workspace/ref/mgo_mapping.txt', T),
        'Si': get_G_corr('workspace/ref/sio2_mapping.txt', T),
        'O': get_O_G_corr(T),
    }

    with open('workspace/ensemble_results.txt', 'r') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')

    data = []
    for block in blocks:
        filename_match = re.search(r'^(.*?):', block)
        formula_match = re.search(r':\s*(.*)', block.splitlines()[0])
        formation_energy_match = re.search(r'Formation Energy = ([\-\d\.Ee]+)', block)
        energy_std_match = re.search(r'Energy s.t.d. = ([\d\.Ee]+)', block)
        force_std_match = re.search(r'Mean force s.t.d. = ([\d\.Ee]+)', block)

        if not (filename_match and formula_match and formation_energy_match and energy_std_match and force_std_match):
            continue

        filename = filename_match.group(1)
        formula = formula_match.group(1)
        formation_energy = float(formation_energy_match.group(1))
        energy_std = float(energy_std_match.group(1))
        force_std = float(force_std_match.group(1))

        composition = parse_formula(formula)

        dir_path = None
        for dir_name in os.listdir('workspace/stable'):
            candidate_path = os.path.join('workspace/stable', dir_name)
            if os.path.isdir(candidate_path):
                if filename in os.listdir(candidate_path):
                    dir_path = candidate_path
                    break

        if dir_path is None:
            continue

        index = os.path.basename(dir_path)
        G_correction = get_G_corr(f'workspace/stable/{index}/mapping.txt', T)

        correction = sum((chemical_potentials.get(elem, 0.0)+G_corr.get(elem, 0.0)) * count for elem, count in composition.items())
        n_atoms = sum(composition.values())
        corrected_formation_energy = formation_energy - correction / n_atoms
        gibbs_formation_energy = corrected_formation_energy + G_correction

        data.append({
            'filename': filename,
            'formula': formula,
            'formation_energy': corrected_formation_energy,
            'gibbs_formation_energy': gibbs_formation_energy,
            'energy_std': energy_std,
            'force_std': force_std,
        })

    filtered_data = [d for d in data if d['energy_std'] <= 40 and d['force_std'] <= 1]
    sorted_data = sorted(filtered_data, key=lambda x: x['gibbs_formation_energy'])

    gibbs_formation_energies = np.array([d['gibbs_formation_energy'] for d in sorted_data])
    boltzmann_factors = np.exp(-gibbs_formation_energies / kT)
    partition_function = np.sum(boltzmann_factors)
    probabilities = boltzmann_factors / partition_function

    for i, d in enumerate(sorted_data):
        d['boltzmann_prob'] = probabilities[i]

    return sorted_data

# ========== STREAMLIT UI ==========

st.title("🔬 Mg-Ca-Si-O")

T = st.slider("Temperature (K)", 300, 2000, 1000, step=10)
col1, col2 = st.columns(2)
with col1:
    CaO = st.slider("CaO molar ratio", 0.01, 10.0, 0.1)
    MgO = st.slider("MgO molar ratio", 0.01, 10.0, 0.1)
with col2:
    SiO2 = st.slider("SiO₂ molar ratio", 0.01, 10.0, 0.1)
    O2 = st.slider("O₂ molar ratio", 0.01, 50.0, 0.1)

sorted_data = load_sorted_data(T, {
    'CaO': CaO, 'MgO': MgO, 'SiO2': SiO2, 'O2': O2
})

top_n = 30
top_structures = sorted_data[:top_n]

df = pd.DataFrame(top_structures)

df_sorted = df.sort_values('boltzmann_prob', ascending=False)

st.bar_chart(df_sorted.set_index('formula')['boltzmann_prob'])
