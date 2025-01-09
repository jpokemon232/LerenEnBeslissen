import os
import pandas as pd

file_path = os.path.dirname(os.path.abspath(__file__))
files = os.listdir(file_path)
dataset = []
remove_columns = ["validfrom (UTC)","point","type","granularity","timezone","activity","classification","volume (kWh)","emission (kg CO2)","emissionfactor (kg CO2/kWh)"]
for file in files:
    if file.endswith('.csv') and file != 'dataset.csv':
        print(file)
        data = pd.read_csv(file_path + "\\" + file)
        for colum in remove_columns:
            del data[colum]
        data.set_index('validto (UTC)',inplace=True)
        data = data.rename(columns={'capacity (kW)': 'capacity (kW) ' + file.split('-')[0], 'percentage': 'percentage ' + file.split('-')[0]})
        print(data)
        dataset.append(data)

dataset = pd.concat(dataset,axis=1)
dataset.to_csv(file_path + "\\dataset.csv")