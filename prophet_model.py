import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import DataHandler
from prophet import Prophet

DH = DataHandler('Dordrecht.csv','TR2','50-13')
data = DH.return_data()

m = Prophet(
    n_changepoints=0,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True
)
data['hist_timestamp'] = data.index
prophet_data = data.copy()
prophet_data.rename(columns={'number_tap_changes':'y','hist_timestamp':'ds'},inplace=True)

#prophet_data = prophet_data.iloc[:-4*24*30]
#print(prophet_data)

m.fit(prophet_data)  # df is a pandas.DataFrame with 'y' and 'ds' columns
prediction = m.predict(prophet_data)

prediction['yhat'] = prediction['yhat'] * (prediction['yhat'] > 0)
prediction['yhat_lower'] = prediction['yhat_lower'] * (prediction['yhat_lower'] > 0)

print(prediction.head())
print('MAE:', np.mean(np.abs(np.array(prophet_data['y']) - np.array(prediction['yhat']))))

plt.plot(data['hist_timestamp'],data['number_tap_changes'])
plt.plot(prediction['ds'],prediction['yhat'])
plt.plot(prediction['ds'],prediction['yhat_lower'])
plt.plot(prediction['ds'],prediction['yhat_upper'])
plt.show()

plt.plot(prediction['ds'],np.array(prophet_data['y']) - np.array(prediction['yhat']))
plt.plot(prediction['ds'],np.array(prophet_data['y']) - np.array(prediction['yhat_lower']))
plt.plot(prediction['ds'],np.array(prophet_data['y']) - np.array(prediction['yhat_upper']))
plt.show()

future = m.make_future_dataframe(periods=30)
future_prediction = m.predict(future)
plt.plot(data['hist_timestamp'],data['number_tap_changes'])
plt.plot(future_prediction['ds'],future_prediction['yhat'])
plt.plot(future_prediction['ds'],future_prediction['yhat_lower'])
plt.plot(future_prediction['ds'],future_prediction['yhat_upper'])
plt.show()