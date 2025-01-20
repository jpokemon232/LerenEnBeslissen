import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import DataHandler

train_percentage = 0.9

DH = DataHandler('Bodegraven.csv',' TR2','50-10')
data = DH.return_data()
data['bias'] = 1
Y = np.array(data['number_tap_changes'])
del data['number_tap_changes']
X = np.array(data[['bias','time_units']])

print('shape X', X.shape)
print('shape Y', Y.shape)

X_train, Y_train, X_test, Y_test = X[:int(len(X)*train_percentage)],Y[:int(len(X)*train_percentage)],X[int(len(X)*train_percentage):],Y[int(len(X)*train_percentage):]

beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
print('coefficients',beta)

print('Train MAE:', np.mean(np.abs(Y_train - X_train @ beta)))
print('Train MSE:', np.mean((Y_train - X_train @ beta)**2))
print('Test MAE:', np.mean(np.abs(Y_test - X_test @ beta)))
print('Test MSE:', np.mean((Y_test - X_test @ beta)**2))

plt.plot(data.index,Y,label='True')
plt.plot(data.index,X @ beta,label='Predict')
plt.legend()
plt.show()

plt.plot(data.index,Y - X @ beta,label='residue')
plt.legend()
plt.show()