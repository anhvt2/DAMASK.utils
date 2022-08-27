
import numpy as np
import seaborn as sns
import pandas as pd

d = np.loadtxt('Ta_Euler_CLC.txt', skiprows=6)
df = pd.DataFrame(d, columns = ['phi1','theta','phi2'])

# g = sns.PairGrid(df)
# g.map_upper(sns.histplot)
# g.map_lower(sns.kdeplot, fill=True)
# g.map_diag(sns.histplot, kde=True)

sns.pairplot(df)
plt.show()
