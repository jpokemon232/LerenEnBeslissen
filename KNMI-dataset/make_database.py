import re
import os

file_path = os.path.dirname(os.path.abspath(__file__))
files = os.listdir(file_path)
for file in files:
    if not(file.endswith('.csv') or file.endswith('.py')):
        content = open(file_path + "\\" + file,'r').read()
        content = '\n'.join(content.split('\n')[14:])
        content = content[2:]
        content = content.replace(" locatie a ", "")
        content = re.sub('(?<=\S) (?=\S)', '_', content)
        content = re.sub('(?<=[0-9])_(?=[0-9])', '#', content)
        for _ in range(20):
            content = content.replace("  ", " ")
        content = content.replace(" ", ",")
        content = re.sub(',\n', '\n', content)
        content = content.replace("#", " ")
        open(file_path + "\\" + file + '.csv','w').write(content)
        
import pandas as pd

files = os.listdir(file_path)
dataset = []
for file in files:
    if file.endswith('.csv') and file != 'dataset.csv':
        print(file)
        data = pd.read_csv(file_path + "\\" + file)
        print(data)
        dataset.append(data)

dataset = pd.concat(dataset)
dataset = dataset.set_index('DTG')
dataset = dataset.sort_index()
dataset.to_csv(file_path + "\\dataset.csv")
        
import folium
import numpy as np

coordinates = np.array(dataset.loc[dataset.index == '2024-03-01 00:10:00', ['LATITUDE','LONGITUDE']])

# Create a map centered around a central location
map = folium.Map(location=[40.7128, -74.0060], zoom_start=2)

# Add markers for each coordinate
for coord in coordinates:
    folium.Marker(location=coord).add_to(map)

# Save the map to an HTML file to view in a browser
map.save(file_path + "\\station_map.html")