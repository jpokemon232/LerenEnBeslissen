import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import DataHandler
from prophet import Prophet

train_percentage = 0.7

DH = DataHandler('Dordrecht.csv','TR2','50-13')
data = DH.return_data()

params = {'growth': 'linear', 'n_changepoints': 0, 'yearly_seasonality': True, 'weekly_seasonality': True, 'daily_seasonality': False, 'seasonality_mode': 'additive', 'seasonality_prior_scale': 20.994646323919405, 'holidays_prior_scale': 71.27161286840024, 'changepoint_prior_scale': 0.9085752455110839, 'mcmc_samples': 16}
m = Prophet(**params)
data['hist_timestamp'] = data.index
data = data.copy()
data.rename(columns={'number_tap_changes':'y','hist_timestamp':'ds'},inplace=True)

print('shape data', data.shape)

train_data, test_data = data.iloc[:int(len(data)*train_percentage)],data.iloc[int(len(data)*train_percentage):]

print('shape train', train_data.shape)
print('shape test', test_data.shape)

m.fit(train_data)
#calculating the training losses
prediction = m.predict(train_data)

#through an relu
prediction['yhat'] = prediction['yhat'] * (prediction['yhat'] > 0)
prediction['yhat_lower'] = prediction['yhat_lower'] * (prediction['yhat_lower'] > 0)

print(prediction.head())
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