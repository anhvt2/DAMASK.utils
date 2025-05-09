
""" This script:
    (1) reads a material.config from DAMASK, 
    (2) extracts crystallographic orientations (Euler angles)
    (3) dumps to a .npy file
    Note: mostly written by ChatGPT
"""

import numpy as np
import re

filename = "material.config"

# Flags and storage
in_texture_section = False
grain_data = []

with open(filename, 'r') as file:
    lines = file.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    # Check for section start
    if re.match(r"<texture>", line):
        in_texture_section = True
        i += 1
        continue
    elif re.match(r"<\w+>", line) and not line.startswith("<texture>"):
        # Exit texture section if another section starts
        in_texture_section = False
    if in_texture_section:
        # Check for grain header
        grain_match = re.match(r'\[grain(\d+)\]', line)
        if grain_match:
            grain_id = int(grain_match.group(1))
            if i + 1 < len(lines):
                data_line = lines[i + 1]
                angle_match = re.search(
                    r'phi1\s+([0-9.]+)\s+Phi\s+([0-9.]+)\s+phi2\s+([0-9.]+)', data_line)
                if angle_match:
                    phi1 = float(angle_match.group(1))
                    Phi = float(angle_match.group(2))
                    phi2 = float(angle_match.group(3))
                    grain_data.append((grain_id, phi1, Phi, phi2))
            i += 2
            continue
    i += 1

# # Output results
# verbose = True
# if verbose:
# for grain in grain_data:
#     print(f"Grain {grain[0]}: phi1={grain[1]}, Phi={grain[2]}, phi2={grain[3]}")

np.save('texture.npy', np.array(grain_data))
