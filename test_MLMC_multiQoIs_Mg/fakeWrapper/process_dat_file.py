import numpy as np

# read dat file
data = np.loadtxt("MultilevelEstimators-multiQoIs.dat", delimiter=",")
nb_of_rows = data.shape[0]
nb_of_qoi = data.shape[1] - 1

# get max level
max_level = int(np.amax(data, axis=0)[0])
print(f"max_level: {max_level}")

# gather all samples
samples = []
dsamples = []
for level in range(max_level):
    samples.append([])
    dsamples.append([])
    for row in range(nb_of_rows):
        if data[row, 0] == level and (row == nb_of_rows - 1 or data[row + 1, 0] == max(0, level - 1)):
            samples[level].append(row)
            if level > 0:
                dsamples[level].append(row + 1)
    print(f"Found {len(samples[level])} samples on level {level}")

# print statistics
for qoi in range(nb_of_qoi):
    print(f"Statistics for qoi {qoi}:")
    print(*[f"{s:<12s}" for s in ("level", "E_l", "dE_l", "V_l", "dV_l")])
    for level in range(max_level):
        x = data[samples[level], qoi + 1]
        dx = data[dsamples[level], qoi + 1]
        print(f"{level:<13d}", end="")
        print(f"{np.mean(x):<13.5f}", end="")
        print(f"{np.abs(np.mean(x - dx)):<13.5f}" if level > 0 else "-".ljust(13), end="")
        print(f"{np.var(x):<13.5f}", end="")
        print(f"{np.abs(np.var(x - dx)):<13.5f}" if level > 0 else "-".ljust(13))
