import glob
import os

def extract_mapping(file):
    mapping = {}
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    first_col = float(parts[0])
                    if 300 <= first_col <= 2000:
                        second_col = float(parts[1])
                        mapping[first_col] = second_col * 0.0103641
                except ValueError:
                    continue
    return mapping

files = glob.glob("out.*_*")

for file in files:
    mapping = extract_mapping(file)
    if mapping:
        base = os.path.basename(file)
        if "_" in base:
            suffix = base.split("_", 1)[1]  # split at first "_" and take the second part
        else:
            suffix = base  # fallback

        output_filename = f"{suffix}_mapping.txt"

        with open(output_filename, 'w') as out_f:
            out_f.write("# First_column    Converted_second_column\n")
            for key in sorted(mapping.keys()):
                out_f.write(f"{key:.1f}    {mapping[key]:.6f}\n")
        print(f"✅ Saved mapping to {output_filename}")

