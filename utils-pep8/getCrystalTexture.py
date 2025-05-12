#!/usr/bin/env python3

"""This script:
(1) reads a material.config from DAMASK,
(2) extracts crystallographic orientations (Euler angles)
(3) dumps to a .npy file
Note: mostly written by ChatGPT
"""

import re

import numpy as np

FILENAME = "material.config"

# Flags and storage
GRAIN_DATA = []

with open(FILENAME, 'r') as file:
    lines = file.readlines()

I = 0
while I < len(lines):
    line = lines[I].strip()
    # Check for section start
    if re.match(r"<texture>", line):
        in_texture_section = True
        I += 1
        continue
    if re.match(r"<\w+>", line) and not line.startswith("<texture>"):
        # Exit texture section if another section starts
        in_texture_section = False
    if in_texture_section:
        # Check for grain header
        grain_match = re.match(r'\[grain(\d+)\]', line)
        if grain_match:
            grain_id = int(grain_match.group(1))
            if I + 1 < len(lines):
                data_line = lines[I + 1]
                angle_match = re.search(
                    r'phi1\s+([0-9.]+)\s+Phi\s+([0-9.]+)\s+phi2\s+([0-9.]+)', data_line
                )
                if angle_match:
                    phi1 = float(angle_match.group(1))
                    Phi = float(angle_match.group(2))
                    phi2 = float(angle_match.group(3))
                    GRAIN_DATA.append((grain_id, phi1, Phi, phi2))
            I += 2
            continue
    I += 1
# if verbose:

np.save('texture.npy', np.array(GRAIN_DATA))
