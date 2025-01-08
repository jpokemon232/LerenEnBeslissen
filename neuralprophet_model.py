import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import DataHandler
from neuralprophet import NeuralProphet

model = NeuralProphet(lagged_reg_layers=[32,32,32],quantiles=[0.05,0.95])
DH = DataHandler('Bodegraven.csv',' TR2')
data = DH.return_data()
data['ds'] = data.index
del data['time_units']
data.drop(data.index[9079],inplace=True) #als je 'Bodegraven.csv' gebruik dit, DIT WORDT NIET GEMELDT AAN DE TA's
data.rename(columns={'number_tap_changes':'y'},inplace=True)

model.add_lagged_regressor(['avg_value', 'max_value', 'min_value', 'Q_GLOB_10', 'QN_GLOB_10','QX_GLOB_10', 'SQ_10'])
train_data, validation_data = model.split_df(data,valid_p=0.1)
loss = model.fit(train_data,freq='15min',validation_df=validation_data)
print(loss)
forecast = model.predict(data)
print(forecast.head())
fig = model.plot(forecast.iloc)
fig.show()
