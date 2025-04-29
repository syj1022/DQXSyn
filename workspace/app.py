import os
import re
import tempfile
import altair as alt
import numpy as np
import pandas as pd
from collections import defaultdict
import streamlit as st
from ase.io import read

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

def load_shear_strength_for_structure(directory):
    shear_strength_file = os.path.join(directory, 'shear_strength.txt')
    if os.path.exists(shear_strength_file):
        try:
            with open(shear_strength_file, 'r') as f:
                strength = f.readline().strip()
                return float(strength)
        except ValueError:
            return None
    return None
    
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

        shear_strength = load_shear_strength_for_structure(dir_path)
        
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
            'shear_strength': shear_strength,
        })

    filtered_data = [d for d in data if d['energy_std'] <= 40 and d['force_std'] <= 1]
    sorted_data = sorted(filtered_data, key=lambda x: x['gibbs_formation_energy'])

    gibbs_formation_energies = np.array([d['gibbs_formation_energy'] for d in sorted_data])
    boltzmann_factors = np.exp(-gibbs_formation_energies / kT)
    partition_function = np.sum(boltzmann_factors)
    probabilities = boltzmann_factors / partition_function

    for i, d in enumerate(sorted_data):
        d['boltzmann_prob'] = probabilities[i]

    strengths = [
        d['shear_strength'] * d['boltzmann_prob']
        for d in sorted_data if d['shear_strength'] is not None
    ]
    weighted_avg_strength = sum(strengths) if strengths else None

    if weighted_avg_strength is not None:
        print(f"Boltzmann-weighted Avg. Shear Strength: {weighted_avg_strength:.2f} GPa")
    else:
        print("Shear strength data missing for most structures.")
        
    return sorted_data

# ========== STREAMLIT UI ==========

st.title("🔬 Mg-Ca-Si-O Phase Distribution")

T = st.slider("Temperature (K)", 300, 2000, 1000, step=10)
O2 = st.slider("O₂ partial pressure (atm)", 0.01, 50.0, 1.0, step=0.1)

col1, col2, col3 = st.columns(3)
with col1:
    MgO = st.slider("MgO molar ratio", 0.01, 10.0, 1.0, step=0.1)
with col2:
    CaO = st.slider("CaO molar ratio", 0.01, 10.0, 1.0, step=0.1)
with col3:
    SiO2 = st.slider("SiO₂ molar ratio", 0.01, 10.0, 1.0, step=0.1)

sorted_data = load_sorted_data(T, {
    'CaO': CaO, 'MgO': MgO, 'SiO2': SiO2, 'O2': O2
})

df = pd.DataFrame(sorted_data)

if 'boltzmann_prob' in df.columns:
    df_top = df.head(30).copy()
    
    df_top['index'] = range(len(df_top))
    
    base = alt.Chart(df_top).encode(
        x=alt.X('index:O', 
                axis=alt.Axis(title='Bulk Structure', labels=False, ticks=False)),
        tooltip=['formula', 'boltzmann_prob:Q', 'formation_energy:Q', 'gibbs_formation_energy:Q']
    )
    
    bars = base.mark_bar().encode(
        y=alt.Y('boltzmann_prob:Q', title='Composition'),
        color=alt.Color('boltzmann_prob:Q', legend=None, scale=alt.Scale(scheme='blues')))
    
    text = base.mark_text(
        align='center',
        baseline='bottom',
        dy=5,
        angle=270,
        fontSize=10
    ).encode(
        y='boltzmann_prob:Q',
        text='formula:N'
    )
    
    chart = (bars + text).properties(
        width=800,
        height=500
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_view(
        strokeWidth=0
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    st.subheader("Summary Table")
    st.dataframe(
        df_top[['formula', 'boltzmann_prob', 'formation_energy', 'gibbs_formation_energy']].rename(columns={
            'boltzmann_prob': 'Composition',
            'formation_energy': 'Formation Energy (meV/atom)',
            'gibbs_formation_energy': 'Formation Free Energy (meV/atom)'
        }).style.format({
            'Composition': '{:.2%}',
            'Formation Energy (meV/atom)': '{:.3f}',
            'Formation Free Energy (meV/atom)': '{:.3f}'
        }),
        height=400,
        use_container_width=True
    )


# [Previous imports remain the same]

if 'boltzmann_prob' in df.columns:
    df_top = df.head(30).copy()
    df_top['index'] = range(len(df_top))

    def make_structure_path(filename):
        if isinstance(filename, str) and filename.strip():  # Ensure filename is valid
            return os.path.join('workspace', 'generated', filename)
        else:
            return None

    df_top['structure_path'] = df_top['filename'].apply(make_structure_path)

    st.subheader("Visualized Structures")

    for idx in range(len(df_top)):
        formula = df_top.at[idx, 'formula'] if 'formula' in df_top.columns else 'Unknown'
        prob = df_top.at[idx, 'boltzmann_prob'] if 'boltzmann_prob' in df_top.columns else 0
        structure_path = df_top.at[idx, 'structure_path']

        with st.expander(f"{formula} - Composition: {prob:.2%}"):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.write(f"**Formation Energy:** {df_top.at[idx, 'formation_energy']:.3f} meV/atom")
                st.write(f"**Formation Free Energy:** {df_top.at[idx, 'gibbs_formation_energy']:.3f} meV/atom")
                st.write(f"**Probability:** {prob:.2%}")
                st.write(f"**Filename:** {df_top.at[idx, 'filename']}")

            with col2:
                if structure_path and os.path.exists(structure_path):
                    try:
                        # Read the structure
                        atoms = read(structure_path)

                        st.write(f"**File type:** {os.path.splitext(structure_path)[1]}")
                        st.write(f"**Atoms:** {len(atoms)}")

                        # Display structure in 3D viewer if possible
                        try:
                            import py3Dmol
                            with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as tmp:
                                atoms.write(tmp.name, format='cif')
                                view = py3Dmol.view()
                                view.addModel(open(tmp.name).read(), 'cif')
                                view.setStyle({
                                    'sphere': {'colorscheme': 'Jmol', 'scale': 0.3},
                                    'stick': {'colorscheme': 'Jmol', 'radius': 0.2}
                                })
                                view.zoomTo()
                                st.components.v1.html(view._make_html(), height=400)
                        except ImportError:
                            st.warning("3D viewer not available. Install it with: pip install py3Dmol")
                            st.code(atoms)
                    except Exception as e:
                        st.error(f"Error reading structure: {str(e)}")
                        st.write(f"Attempted path: {structure_path}")
                else:
                    st.warning(f"Structure file not found at: {structure_path}")
                    if structure_path:
                        dir_path = os.path.dirname(structure_path)
                        if os.path.exists(dir_path):
                            st.write("Directory contents:")
                            st.code(os.listdir(dir_path))

