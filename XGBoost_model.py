import xgboost as xgb
from util import DataHandler
import numpy as np
import matplotlib.pyplot as plt
DH = DataHandler('Dordrecht.csv','TR2','50-13')
data = DH.return_data()


data['test'] = 0
num = 1
new_data = []
for index in range(len(data)):
    num += 1
    if data['jumps'].iloc[index] == 1:
        new_data.append(num)
        num = 1


X = []
Y = []
for index in range(len(new_data)-32):
    X.append(new_data[index:index+32])
    Y.append(new_data[index+32])

X = np.array(X)
Y = np.array(Y)
model = xgb.XGBRegressor()
model.fit(X,Y)
print(model.score(X,Y))
Y_pred = model.predict(X)

index = 0
output_pred = np.array([0]*len(data))
for p in range(len(Y_pred)):
    output_pred[index:] += 1
    index += int(Y_pred[p])

# zit nog een fout in, de True en Predicted plots hebben een lagg van 32, dus de predicted lijn moet 32 naar links worden geschoven
plt.plot(data.index,data['number_tap_changes'],label='True')
plt.plot(data.index,output_pred,label='predict')
plt.legend()
plt.show()