import pandas as pd

def write_material_config(df, filename='material.config'):
    """
    Write a material.config file from a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing columns:
        ['grain_id', 'phi1', 'Phi', 'phi2', 'crystallite', 'phase', 'texture', 'fraction']

    filename : str
        The output filename (default: 'material.config')
    """
    with open(filename, 'w') as f:
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
            f.write(f"(constituent)   phase {int(row['phase'])} texture {int(row['texture'])} fraction {row['fraction']:.6f}\n")
        f.write("\n")
