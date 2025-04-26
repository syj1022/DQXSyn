import re

def modify_pw_input(input_file):
    with open(input_file, 'r') as f:
        content = f.read()

    content = content.replace('d0', '')

    content = re.sub(r'([A-Z][a-z]?)1', r'\1', content)

    cell_params_match = re.search(r'(CELL_PARAMETERS.*?\n(?:.*\n){3})', content)
    if cell_params_match:
        cell_params = cell_params_match.group(1)

        content = content.replace(cell_params, '')

        k_points_index = content.find('K_POINTS automatic')
        if k_points_index != -1:

            cell_params = cell_params.replace('CELL_PARAMETERS', 'CELL_PARAMETERS angstrom')
            content = content[:k_points_index] + cell_params + content[k_points_index:]

    with open(input_file, 'w') as f:
        f.write(content)

modify_pw_input('pw.in')
