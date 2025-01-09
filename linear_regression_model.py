import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import DataHandler

DH = DataHandler('Bodegraven.csv',' TR2','50-10')
data = DH.return_data()
data['bias'] = 1
Y = np.array(data['number_tap_changes'])
del data['number_tap_changes']
X = np.array(data[['bias','time_units']])
beta = np.linalg.inv(X.T @ X) @ X.T @ Y
print('coefficients',beta)
print('MAE:', np.mean(np.abs(Y - X @ beta)))
plt.plot(data.index,Y)
plt.plot(data.index,X @ beta)
plt.show()

plt.plot(data.index,Y - X @ beta)
plt.show()