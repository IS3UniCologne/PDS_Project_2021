import pandas as pd
from .geo import *

# Module gets data trips and  additional information include:
    # get_borough_locationID(borough) returns a list of all location ID in that borough, default "borough = 'Queens'"
    # get_trips() returns a given fraction data of trips data, default fraction=0.05
    # get_time(column) expands the given column into additional columns of Month, Weekday, Hour, Weekend (binary). Default column is pickup time
    # get_duration() returns trip duration
    # get_position(column) adds additional columns of Longitude and Latitude. Default column is pick up location
#-------------------------------------------------------------------------------------------

class trips:
    def __init__(self, fraction=0.1):
        pass

    # Location ID of a borough
    def get_borough_locationID(self,borough='Queens'):
        file = pd.read_csv(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.csv', sep=',')
        return file[file.Borough == borough]['LocationID'].unique()

    # Return a given fraction data of full trips data
    def get_trips(self, fraction=0.05):
        f = '01 02 03 04 05 06 07 08 09 10 11 12'.split( )
        result = []
        for i in f:
            raw = pd.read_parquet(f'C:/Users/kyral/Documents/MIB 2019/BI Capstone Project/trip_data/{i}.parquet',

                                  engine='pyarrow')
            df = raw.head(round(fraction*int(raw.shape[0])))
            result.append(df)
        return pd.concat(result)

    # Add new time columns of weekday, month and hour
    def get_time(self, column='tpep_pickup_datetime'):
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
    def get_position(self,column='PULocationID'):
        df = self.get_trips()
        f = geo().get_centroid()

        #Apply dictionary to column
        df['dummy'] = df[column].map(f)
        df.dropna(inplace=True)
        df[['longitude', 'latitude']] = pd.DataFrame(df['dummy'].tolist(), index=df.index)
        df = df.drop(['dummy'],axis=1)
        df['longitude'] = df['longitude'].round().astype(int)
        df['latitude'] = df['latitude'].round().astype(int)
        return df






