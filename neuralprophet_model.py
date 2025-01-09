import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import DataHandler
from neuralprophet import NeuralProphet

model = NeuralProphet(
    quantiles=[0.05,0.95],
    n_changepoints=0,
    epochs=50,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True
)
DH = DataHandler('Bodegraven.csv',' TR2','50-10')
data = DH.return_data()
data['ds'] = data.index
del data['time_units'], data['jumps'], data['percentage overig'], data['capacity (kW) overig'], data['percentage aardgas'], data['percentage wkk']
data.drop(data.index[9079],inplace=True) #als je 'Bodegraven.csv' gebruik dit, DIT WORDT NIET GEMELDT AAN DE TA's
data.rename(columns={'number_tap_changes':'y'},inplace=True)

model.add_lagged_regressor(['avg_value', 'max_value', 'min_value',
       'capacity (kW) aardgas', 'capacity (kW) afval',
       'percentage afval', 'capacity (kW) biomassa', 'percentage biomassa',
       'capacity (kW) electriciteitsmix', 'percentage electriciteitsmix',
       'capacity (kW) kernenergie', 'percentage kernenergie', 'capacity (kW) steenkool',
       'percentage steenkool', 'capacity (kW) wind', 'percentage wind',
       'capacity (kW) wkk', 'capacity (kW) zeewind',
       'percentage zeewind', 'capacity (kW) zon', 'percentage zon'
])
train_data, validation_data = model.split_df(data,valid_p=0.01)
loss = model.fit(train_data,freq='15min',validation_df=validation_data)
print(loss)
forecast = model.predict(data)
print(forecast.head())
fig = model.plot(forecast)
fig.show()
fig = model.plot_components(forecast)
fig.show()