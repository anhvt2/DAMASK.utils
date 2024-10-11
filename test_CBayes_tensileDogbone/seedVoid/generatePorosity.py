
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(8)

porosity = np.random.uniform(low=0, high=10, size=500)

np.savetxt('porosity.txt', porosity, fmt='%.16f')
df = pd.DataFrame(porosity)

# plt.figure()
df.plot(kind = "hist", density = True, bins = 50)
plt.show()
# df.plot(kind = "kde")
