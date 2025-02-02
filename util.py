import pandas as pd
import numpy as np
import os

class DataHandler():
    def __init__(self, file_name, transformator_name, spanning):
        self.stedin_coords = {
            'Bodegraven.csv': [52.082312846233485, 4.745974681868533],
            'Dordrecht.csv': [51.81272771092374, 4.691237389608873],
            'Middelharnis.csv': [51.75313067046199, 4.16449084069382]
        }
        self.get_data(file_name, transformator_name, spanning)
        self.get_objective()
        self.concat_data()
    
    def get_data(self, file_name, transformator_name, spanning):
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.stedin_data = pd.read_csv(file_path + '\\stedin-dataset\\' + file_name)
        self.stedin_data = self.stedin_data[self.stedin_data['transformator'] == transformator_name]
        self.stedin_data = self.stedin_data[self.stedin_data['spanning(kV)'] == spanning]
        self.stedin_data['hist_timestamp'] = pd.to_datetime(self.stedin_data['hist_timestamp'])

        def remove_timezone(dt):
            return dt.replace(tzinfo=None)
        self.stedin_data['hist_timestamp'] = self.stedin_data['hist_timestamp'].apply(remove_timezone)
        self.stedin_data = self.stedin_data.set_index('hist_timestamp')

        del self.stedin_data['locatie'], self.stedin_data['spanning(kV)'], self.stedin_data['transformator']
        self.stedin_data['time_units'] = np.arange(len(self.stedin_data))
      
        self.NED_data = pd.read_csv(file_path + '\\NED-dataset\\dataset.csv')
        self.NED_data['validto (UTC)'] = pd.to_datetime(self.NED_data['validto (UTC)'])
        self.NED_data.set_index('validto (UTC)',inplace=True)
        self.NED_data = self.NED_data.resample('15min').interpolate()
        mask = (self.NED_data.index >= self.stedin_data.index[0]) & (self.NED_data.index <= self.stedin_data.index[-1])
        self.NED_data = self.NED_data.loc[mask]
        '''
        # get the entire database
        self.KNMI_data = pd.read_csv(file_path + '\\KNMI-dataset\\dataset.csv')
        del self.KNMI_data['NAME'],self.KNMI_data['ALTITUDE']
        KNMI_coords = np.array(self.KNMI_data.loc[self.KNMI_data['DTG'] == self.KNMI_data['DTG'].iloc[0], ['LOCATION','LATITUDE','LONGITUDE']])
        distances = (stedin_coords[0] - KNMI_coords[:,1])**2 + (stedin_coords[1] - KNMI_coords[:,2])**2
        # calculate the best KNMI location in reference to where the transformer is, and only use that data
        best_KNMI_location = KNMI_coords[np.argmin(distances)][0]
        self.KNMI_data = self.KNMI_data.loc[self.KNMI_data['LOCATION'] == best_KNMI_location]
        self.KNMI_data['DTG'] = pd.to_datetime(self.KNMI_data['DTG'])
        self.KNMI_data = self.KNMI_data.set_index('DTG')
        # get the correct time resolution
        interpolate_part = self.KNMI_data[['Q_GLOB_10','QN_GLOB_10','QX_GLOB_10','SQ_10']].resample('15min').interpolate()
        ffill_part = self.KNMI_data[['LOCATION','LATITUDE','LONGITUDE']].resample('15min').ffill()
        self.KNMI_data = ffill_part.join(interpolate_part)
        # only use data on the same timeframe as the stedin data
        mask = (self.KNMI_data.index >= self.stedin_data.index[0]) & (self.KNMI_data.index <= self.stedin_data.index[-1])
        self.KNMI_data = self.KNMI_data.loc[mask]
        del self.KNMI_data['LOCATION'], self.KNMI_data['LATITUDE'], self.KNMI_data['LONGITUDE']
        '''

    def get_objective(self):
        self.stedin_data['jumps'] = np.abs(np.sign(self.stedin_data['avg_value'].diff()).diff())
        self.stedin_data.loc[self.stedin_data['jumps'] == 2,'jumps'] = 1
        self.stedin_data.loc[np.sign(self.stedin_data['avg_value'].diff()) == 0,'jumps'] = 0
        if self.stedin_data['avg_value'].iloc[0] != self.stedin_data['avg_value'].iloc[1]:
            self.stedin_data.at[self.stedin_data.index[1], 'jumps'] = 1
        else:
            self.stedin_data.at[self.stedin_data.index[1], 'jumps'] = 0
        self.stedin_data.at[self.stedin_data.index[0], 'jumps'] = 0
        self.stedin_data['number_tap_changes'] = np.cumsum(self.stedin_data['jumps'])
        
    def concat_data(self):
        self.data = self.stedin_data

    def return_data(self):
        return self.data
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    DH = DataHandler('Bodegraven.csv',' TR2', '50-10')
    data = DH.return_data()