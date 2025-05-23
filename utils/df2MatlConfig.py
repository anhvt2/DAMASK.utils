import pandas as pd
import numpy as np

# Save formatted material.config file from merged DataFrame
output_file = "material_reconstructed.config"

with open(output_file, 'w') as f:
    # Write <texture> section
    f.write("<texture>\n")
    for _, row in df.iterrows():
        f.write(f"[grain{int(row['grain_id'])}]\n")
        f.write(f"(gauss) phi1 {row['phi1']:.3f}   Phi {row['Phi']:.3f}    phi2 {row['phi2']:.3f}   scatter 0.0   fraction 1.0\n")

    f.write("\n")

    # Write <microstructure> section
    f.write("<microstructure>\n")
    for _, row in df.iterrows():
        f.write(f"[grain{int(row['grain_id'])}]\n")
        f.write(f"crystallite {int(row['crystallite'])}\n")
        f.write(f"(constituent)   phase {int(row['phase'])} texture {int(row['texture'])} fraction {row['fraction']:.1f}\n")
    f.write("\n")

