
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")

inpData = np.loadtxt('postproc.input.dat', delimiter=',', dtype=float)
print(inpData.shape)

label_list = []
for i in range(inpData.shape[1]):
	label_list += ['x%s' % str(i+1)]

print(label_list)

df = pd.DataFrame(inpData) # , columns = [label_list]) # , index=range(inpData.shape[0]))
# sns.pairplot(df[label_list])
# df = pd.DataFrame(inpData)
# sns.pairplot(df)

# rename column
for i in range(inpData.shape[1]):
	df = df.rename(columns={i: 'x%d' % i})

# https://seaborn.pydata.org/examples/pair_grid_with_kde.html
g = sns.PairGrid(df, diag_sharey=False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.kdeplot, fill=True, clip=((0,1), (1,0)))
g.map_diag(sns.kdeplot, clip=(0,1))


# xlabels,ylabels = [],[]

# for ax in g.axes[-1,:]:
#     xlabel = ax.xaxis.get_label_text()
#     xlabels.append(xlabel)

# for ax in g.axes[:,0]:
#     ylabel = ax.yaxis.get_label_text()
#     ylabels.append(ylabel)

# for i in range(len(xlabels)):
#     for j in range(len(ylabels)):
#         g.axes[j,i].xaxis.set_label_text(xlabels[i])
#         g.axes[j,i].yaxis.set_label_text(ylabels[j])

plt.show()

