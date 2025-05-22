# fast_colormap_setup.py
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

df = pd.read_csv("fast-table-float-1024.csv")
colors = df[['RGB_r', 'RGB_g', 'RGB_b']].values
fast_cmap = ListedColormap(colors, name='fast')
mpl.colormaps.register(name='fast', cmap=fast_cmap)
