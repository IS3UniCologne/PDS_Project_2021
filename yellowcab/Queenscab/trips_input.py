import pandas as pd
from .geo import *
import numpy as np

# Modules extract trips data belong to Queens borough and return additional information:
# get_trips(filename) returns trips data of a given file (e.g. 01.parquet)
# get_time(column) adds additional columns of
    # Month
    # Weekday
    # Hour
    # Weekend (binary)
# get_duration returns trip duration
# get_position(column) adds additional columns of position(Longitude and Latitude),


class trips:
    def __init__(self, filename):
        self.f = filename

    # Location ID of Queens
    def get_locationID(self):
        file = pd.read_csv(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.csv', sep=',')
        return file[file.Borough == 'Queens']

    # Return full trips data of Queens borough of a given file
    def get_trips(self):
        z = self.get_locationID()
        queens = z.LocationID.unique()
        raw = pd.read_parquet(f'C:/Users/kyral/Documents/MIB 2019/BI Capstone Project/trip_data/{self.f}',
                                  engine='pyarrow')
        df = raw[(raw['PULocationID'].isin(queens))&(raw['DOLocationID'].isin(queens))]
        return df

    # Add new time columns of weekday, month and hour
    def get_time(self, column):
        df = self.get_trips()
        df['month'] = df[str(column)].dt.month
        df['hour'] = df[str(column)].dt.hour

        # Return day of week
        df['weekday'] = df[str(column)].dt.weekday
        df["weekday"] = df["weekday"].replace(list(range(0,7)),"Mon Tue Wed Thu Fri Sat Sun".split())

        # Return if a date is a weekend
        df['weekend'] = df[str(column)].dt.weekday
        df['weekend'] = df['weekend'].replace(list(range(0,6)),0).replace(6,1)
        return df

    # Calculate trip duration
    def get_duration(self):
        df = self.get_trips()
        df['duration'] = df['tpep_dropoff_datetime']-df['tpep_pickup_datetime']
        return df

    # Return longitude and latitude of a location ID
    def get_position(self,column):
        df = self.get_trips()
        test = self.get_locationID()

        # Create a dictionary of centroids
        d = []
        for i in test.LocationID:
            data = geo(i)
            d.append(data.get_centroid())
        f = dict(zip(test.LocationID,d))

        # Apply dictionary to column
        df['dummy'] = df[column].map(f)
        df[['longitude', 'latitude']] = pd.DataFrame(df['dummy'].tolist(), index=df.index)
        df = df.drop(['dummy'],axis=1)
        df['longitude'] = df['longitude'].round().astype(int)
        df['latitude'] = df['latitude'].round().astype(int)
        return df





