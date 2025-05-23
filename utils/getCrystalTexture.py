#!/usr/bin/env python3

""" This script:
    (1) reads a material.config from DAMASK, 
    (2) extracts crystallographic orientations (Euler angles) from <texture>
    (3) extracts microstructure mapping from <microstructure>
    (4) dumps both to .npy files
"""

import numpy as np
import re
import pandas as pd

filename = "material.config"

# Flags and storage
in_texture_section = False
in_microstructure_section = False
texture_data = []
microstructure_data = []

with open(filename, 'r') as file:
    lines = file.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    # --- Detect section changes ---
    if re.match(r"<texture>", line):
        in_texture_section = True
        in_microstructure_section = False
        i += 1
        continue
    elif re.match(r"<microstructure>", line):
        in_texture_section = False
        in_microstructure_section = True
        i += 1
        continue
    elif re.match(r"<\w+>", line):
        in_texture_section = False
        in_microstructure_section = False

    # --- Parse <texture> section ---
    if in_texture_section:
        grain_match = re.match(r'\[grain(\d+)\]', line)
        if grain_match:
            grain_id = int(grain_match.group(1))
            if i + 1 < len(lines):
                data_line = lines[i + 1]
                angle_match = re.search(r'phi1\s+([0-9.]+)\s+Phi\s+([0-9.]+)\s+phi2\s+([0-9.]+)', data_line)
                if angle_match:
                    phi1 = float(angle_match.group(1))
                    Phi = float(angle_match.group(2))
                    phi2 = float(angle_match.group(3))
                    texture_data.append((grain_id, phi1, Phi, phi2))
            i += 2
            continue

    # --- Parse <microstructure> section ---
    elif in_microstructure_section:
        grain_match = re.match(r'\[grain(\d+)\]', line)
        if grain_match and i + 2 < len(lines):
            grain_id = int(grain_match.group(1))
            crystallite_line = lines[i + 1].strip()
            constituent_line = lines[i + 2].strip()

            # Extract crystallite number
            crystallite_match = re.search(r'crystallite\s+(\d+)', crystallite_line)
            crystallite = int(crystallite_match.group(1)) if crystallite_match else None

            # Extract phase, texture, fraction
            constituent_match = re.search(r'phase\s+(\d+)\s+texture\s+(\d+)\s+fraction\s+([0-9.]+)', constituent_line)
            if constituent_match:
                phase = int(constituent_match.group(1))
                texture = int(constituent_match.group(2))
                fraction = float(constituent_match.group(3))
                microstructure_data.append((grain_id, crystallite, phase, texture, fraction))

            i += 3
            continue

    i += 1

# # Save results
# np.save('texture.npy', np.array(texture_data))
# np.save('microstructure.npy', np.array(microstructure_data))

# Convert to DataFrames
texture_df = pd.DataFrame(texture_data, columns=["grain_id", "phi1", "Phi", "phi2"])
microstructure_df = pd.DataFrame(microstructure_data, columns=["grain_id", "crystallite", "phase", "texture", "fraction"])

# Merge on grain_id
df = pd.merge(texture_df, microstructure_df, on="grain_id", how="outer")

# Sort by grain_id (optional)
df.sort_values(by="grain_id", inplace=True)

# Save to CSV or pickle
df.to_csv("material_config.csv", index=False)

