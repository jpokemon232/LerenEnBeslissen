import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import DataHandler
from prophet import Prophet
import time

train_percentage = 0.7

DH = DataHandler('Dordrecht.csv','TR1','150-50')
data = DH.return_data()
params = {'growth': 'linear', 'n_changepoints': 1, 'yearly_seasonality': False, 'weekly_seasonality': True, 'daily_seasonality': True, 'seasonality_mode': 'additive', 'seasonality_prior_scale': 40.63629950345158, 'holidays_prior_scale': 24.512734115358192, 'changepoint_prior_scale': 0.4120677638045942, 'mcmc_samples': 0}
m = Prophet(**params)
data['hist_timestamp'] = data.index
data = data.copy()
data.rename(columns={'number_tap_changes':'y','hist_timestamp':'ds'},inplace=True)
#print('shape data', data.shape)
train_data, test_data = data.iloc[:int(len(data)*train_percentage)],data.iloc[int(len(data)*train_percentage):]
#print('shape train', train_data.shape)
#print('shape test', test_data.shape)
m.fit(train_data)
#calculating the training losses
prediction = m.predict(train_data)
#through an relu
prediction['yhat'] = prediction['yhat'] * (prediction['yhat'] > 0)
prediction['yhat_lower'] = prediction['yhat_lower'] * (prediction['yhat_lower'] > 0)
#print(prediction.head())
print('train MAE:', np.mean(np.abs(np.array(train_data['y']) - np.array(prediction['yhat']))))
#calculating the test losses
prediction = m.predict(test_data)
#through an relu
prediction['yhat'] = prediction['yhat'] * (prediction['yhat'] > 0)
prediction['yhat_lower'] = prediction['yhat_lower'] * (prediction['yhat_lower'] > 0)
print('test MAE:', np.mean(np.abs(np.array(test_data['y']) - np.array(prediction['yhat']))))


# predict for the entire dataset
prediction = m.predict(data)
plt.plot(data['ds'],data['y'])
plt.plot(prediction['ds'],prediction['yhat'])
plt.plot(prediction['ds'],prediction['yhat_lower'])
plt.plot(prediction['ds'],prediction['yhat_upper'])
plt.show()

plt.plot(prediction['ds'],np.array(data['y']) - np.array(prediction['yhat']))
plt.plot(prediction['ds'],np.array(data['y']) - np.array(prediction['yhat_lower']))
plt.plot(prediction['ds'],np.array(data['y']) - np.array(prediction['yhat_upper']))
plt.show()

future = m.make_future_dataframe(periods=30)
future_prediction = m.predict(future)
plt.plot(data['ds'],data['y'])
plt.plot(future_prediction['ds'],future_prediction['yhat'])
plt.plot(future_prediction['ds'],future_prediction['yhat_lower'])
plt.plot(future_prediction['ds'],future_prediction['yhat_upper'])
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import DataHandler
from prophet import Prophet
import time
train_percentage = 0.7
L_data = pd.read_csv('H:\\project\\ND-LerenEnBeslissen\\stedin-dataset\\Dordrecht.csv')
print(np.unique(L_data['locatie']))
print(np.unique(L_data['spanning(kV)']))
print(np.unique(L_data['transformator']))

train_percentage = 0.7
for loc in np.unique(L_data['locatie']):
    for span in np.unique(L_data['spanning(kV)']):
        for trans in np.unique(L_data['transformator']):
            try:
                    DH = DataHandler('Dordrecht.csv','TR1','150-50')
                    data = DH.return_data()

                    params = {'growth': 'linear', 'n_changepoints': 1, 'yearly_seasonality': False, 'weekly_seasonality': True, 'daily_seasonality': True, 'seasonality_mode': 'additive', 'seasonality_prior_scale': 40.63629950345158, 'holidays_prior_scale': 24.512734115358192, 'changepoint_prior_scale': 0.4120677638045942, 'mcmc_samples': 0}
                    m = Prophet(**params)
                    data['hist_timestamp'] = data.index
                    data = data.copy()
                    data.rename(columns={'number_tap_changes':'y','hist_timestamp':'ds'},inplace=True)

                    #print('shape data', data.shape)

                    train_data, test_data = data.iloc[:int(len(data)*train_percentage)],data.iloc[int(len(data)*train_percentage):]

                    #print('shape train', train_data.shape)
                    #print('shape test', test_data.shape)

                    m.fit(train_data)
                    #calculating the training losses
                    prediction = m.predict(train_data)

                    #through an relu
                    prediction['yhat'] = prediction['yhat'] * (prediction['yhat'] > 0)
                    prediction['yhat_lower'] = prediction['yhat_lower'] * (prediction['yhat_lower'] > 0)

                    #print(prediction.head())
                    print('train MAE:', np.mean(np.abs(np.array(train_data['y']) - np.array(prediction['yhat']))))

                    #calculating the test losses
                    prediction = m.predict(test_data)

                    #through an relu
                    prediction['yhat'] = prediction['yhat'] * (prediction['yhat'] > 0)
                    prediction['yhat_lower'] = prediction['yhat_lower'] * (prediction['yhat_lower'] > 0)
                    print('test MAE:', np.mean(np.abs(np.array(test_data['y']) - np.array(prediction['yhat']))))
                    print(loc,trans,span)
            except:
                pass
'''
# predict for the entire dataset
prediction = m.predict(data)
plt.plot(data['ds'],data['y'])
plt.plot(prediction['ds'],prediction['yhat'])
plt.plot(prediction['ds'],prediction['yhat_lower'])
plt.plot(prediction['ds'],prediction['yhat_upper'])
plt.show()

plt.plot(prediction['ds'],np.array(data['y']) - np.array(prediction['yhat']))
plt.plot(prediction['ds'],np.array(data['y']) - np.array(prediction['yhat_lower']))
plt.plot(prediction['ds'],np.array(data['y']) - np.array(prediction['yhat_upper']))
plt.show()

future = m.make_future_dataframe(periods=30)
future_prediction = m.predict(future)
plt.plot(data['ds'],data['y'])
plt.plot(future_prediction['ds'],future_prediction['yhat'])
plt.plot(future_prediction['ds'],future_prediction['yhat_lower'])
plt.plot(future_prediction['ds'],future_prediction['yhat_upper'])
plt.show()
'''