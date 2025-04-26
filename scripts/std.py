gen = input("Enter gen: ")

force_std_total = 0.0
count = 0

with open(f"{gen}/ensemble_results.txt", "r") as f:
    for line in f:
        if "Mean force s.t.d." in line:
            value_str = line.split("Mean force s.t.d. =")[1].split("eV")[0].strip()
            value = float(value_str)
            if value <= 10 and value > 0:
                force_std_total += value
                count += 1

if count > 0:
    average_force_std = force_std_total / count
    print(f"Average Mean force s.t.d. = {average_force_std:.6f} eV/atom")

energy_std_total = 0.0
count = 0

with open(f"{gen}/ensemble_results.txt", "r") as f:
    for line in f:
        if "Energy s.t.d." in line:
            value_str = line.split("Energy s.t.d. =")[1].split("meV")[0].strip()
            value = float(value_str)
            if value <= 1000:
                energy_std_total += value
                count += 1

if count > 0:
    average_energy_std = energy_std_total / count
    print(f"Average Energy s.t.d. = {average_energy_std:.6f} meV/atom")
