import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('data/Bodegraven.csv')
data = data[data['transformator'] == ' TR2']

data['hist_timestamp'] = pd.to_datetime(data['hist_timestamp'])
data.index = np.arange(len(data))
data['jumps'] = np.abs(np.sign(data['avg_value'].diff()).diff())
data.loc[data['jumps'] == 2,'jumps'] = 1
data.loc[np.sign(data['avg_value'].diff()) == 0,'jumps'] = 0
if data['avg_value'].iloc[0] != data['avg_value'].iloc[1]:
    data.loc[1,'jumps'] = 1
else:
    data.loc[1,'jumps'] = 0
data.loc[0,'jumps'] = 0
data['number_tap_changes'] = np.cumsum(data['jumps'])
from prophet import Prophet
m = Prophet()
prophet_data = data.copy(deep=True)
prophet_data.rename(columns={'number_tap_changes':'y','hist_timestamp':'ds'},inplace=True)
prophet_data['test'] = prophet_data['y'].iloc[1:]
prophet_data.loc[0,'test'] = 0 

prophet_data = prophet_data.iloc[:-60*(4*12)]
print('data',data)
print('prophet_data',prophet_data)

def remove_timezone(dt):
   
    # HERE `dt` is a python datetime 
    # object that used .replace() method
    return dt.replace(tzinfo=None)
 
# APPLY THE ABOVE FUNCTION TO
# REMOVE THE TIMEZONE INFORMATION
# FROM EACH RECORD OF TIMESTAMP COLUMN IN DATAFRAME
prophet_data['ds'] = prophet_data['ds'].apply(remove_timezone)

m.fit(prophet_data)  # df is a pandas.DataFrame with 'y' and 'ds' columns
prediction = m.predict(prophet_data)
future = m.make_future_dataframe(periods=30)
future_prediction = m.predict(future)
plt.plot(data['hist_timestamp'],data['number_tap_changes'],label="True")
plt.plot(future_prediction['ds'],future_prediction['yhat'],label="Pred")
plt.plot(future_prediction['ds'],future_prediction['yhat_lower'],label="Pred Low")
plt.plot(future_prediction['ds'],future_prediction['yhat_upper'],label="Pred High")
plt.legend()
plt.show()