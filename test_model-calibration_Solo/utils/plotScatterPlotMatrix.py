
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")

inpData = np.loadtxt('postproc.input.dat', delimiter=',')
print(inpData.shape)

label_list = []
for i in range(inpData.shape[1]):
	label_list += ['x%s' % str(i+1)]

print(label_list)

# df = pd.DataFrame(inpData, columns = [label_list], index=range(inpData.shape[0]))
# sns.pairplot(df[label_list])
df = pd.DataFrame(inpData)
# sns.pairplot(df)

# https://seaborn.pydata.org/examples/pair_grid_with_kde.html
g = sns.PairGrid(df, diag_sharey=False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot)

plt.show()

