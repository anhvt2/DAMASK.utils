
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

o = np.loadtxt('output.dat', delimiter=',')
o[:,1] /= 1e6
df = pd.DataFrame(o, columns = [r"$\varepsilon_Y$",r"$\sigma_Y$"])

g = sns.jointplot(data=df, x=r"$\varepsilon_Y$", y=r"$\sigma_Y$", kind="kde", xlim=(0.00175, 0.00225), ylim=(2,18)) #, marker='o')
# g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
g.plot_marginals(sns.rugplot, color="r", height=0.0, clip_on=False)

# g = sns.pairplot(df)



# g.set_xlabel(r"$\varepsilon_Y$", fontsize=24)
# g.set_ylabel(r"$\sigma_Y$", fontsize=24)
g.set_axis_labels(xlabel=r"$\varepsilon_Y$", ylabel=r"$\sigma_Y$", fontsize=24)
# plt.xticks([0.002, 0.003, 0.004])
import matplotlib.ticker as ticker
g.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(0.00025))
g.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(2))
# g.set_title('Correlation plot: W', fontsize=24)
plt.show()

