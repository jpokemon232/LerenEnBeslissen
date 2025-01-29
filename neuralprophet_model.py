import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import DataHandler
from neuralprophet import NeuralProphet




L_data = pd.read_csv('H:\\project\\ND-LerenEnBeslissen\\stedin-dataset\\Bodegraven.csv')
print(np.unique(L_data['locatie']))
print(np.unique(L_data['spanning(kV)']))
print(np.unique(L_data['transformator']))
losses = []
train_percentage = 0.7
for loc in np.unique(L_data['locatie']):
    for span in np.unique(L_data['spanning(kV)']):
        for trans in np.unique(L_data['transformator']):
            print(loc,trans,span)
            try:

                model = NeuralProphet(epochs=20)
                DH = DataHandler(loc + '.csv',trans,span)
                data = DH.return_data()
                data['ds'] = data.index
                del data['time_units'], data['jumps'], data['percentage overig'], data['capacity (kW) overig'], data['percentage aardgas'], data['percentage wkk']
                data.drop(data.index[9079],inplace=True) #als je 'Bodegraven.csv' gebruik dit, DIT WORDT NIET GEMELDT AAN DE TA's
                data.rename(columns={'number_tap_changes':'y'},inplace=True)
                data = data[['ds','y']]
                #model.add_lagged_regressor(['avg_value', 'max_value', 'min_value',
                #    'capacity (kW) aardgas', 'capacity (kW) afval',
                #    'percentage afval', 'capacity (kW) biomassa', 'percentage biomassa',
                #    'capacity (kW) electriciteitsmix', 'percentage electriciteitsmix',
                #    'capacity (kW) kernenergie', 'percentage kernenergie', 'capacity (kW) steenkool',
                #    'percentage steenkool', 'capacity (kW) wind', 'percentage wind',
                #    'capacity (kW) wkk', 'capacity (kW) zeewind',
                #    'percentage zeewind', 'capacity (kW) zon', 'percentage zon'
                #])
                train_data, validation_data = model.split_df(data,valid_p=1-train_percentage)
                loss = model.fit(train_data,freq='15min',validation_df=validation_data)
                print(loss)
                losses.append(loss)
                
                forecast = model.predict(data)
                print('done',loc,trans,span)
                del model
            except:
                pass
for loss in losses:
    print(loss)
fig = model.plot(forecast)
fig.show()
fig = model.plot_components(forecast)
fig.show()
