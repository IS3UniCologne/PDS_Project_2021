import pandas as pd
from .trips_input import *
from .geo import *
import datetime

# Modules returns detailed information of trips for a give trips dataframe
    # get_duration() returns trip duration
    # get_time(column) expands the given column into additional columns of Month, Weekday, Hour, Weekend (binary). Default column is pickup time
    # get_aggregate(column) calculate aggregate statistics description for trips duration, grouped by given column. Default column is month
    # get_position(column) adds additional columns of Longitude and Latitude. Default column is pick up location

# and visualize important findings such as:
    # the number of started trips per region in a map for the month with the most trips
    # the  distribution  of  trip  lengths of each month
    # heatmap of ....
#----------------------------------------------------------------------------------------------------
class trips_info:
    def __init__(self, dataframe):
        self.df = dataframe

    def get_duration(self):
        df = self.df
        df['duration'] = df['tpep_dropoff_datetime']-df['tpep_pickup_datetime']
        return df['duration']

        # Add new time columns of weekday, month and hour
    def get_time(self, column='tpep_pickup_datetime'):
        df = self.df
        df['month'] = df[str(column)].dt.month
        df['hour'] = df[str(column)].dt.hour

        # Return day of week
        df['weekday'] = df[str(column)].dt.weekday
        df["weekday"] = df["weekday"].replace(list(range(0, 7)), "Mon Tue Wed Thu Fri Sat Sun".split())

        # Return if a date is a weekend
        df['weekend'] = df[str(column)].dt.weekday
        df['weekend'] = df['weekend'].replace(list(range(0, 6)), 0).replace(6, 1)
        return df

        # Calculate trip duration
    def get_duration(self):
        df = self.df
        df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
        return df

    # Return aggregate values by time
    def get_aggregate(self, column='month'):
        df = self.get_time(column='tpep_pickup_datetime')
        df['duration'] = abs(self.get_duration()['duration'])
        d = df.groupby([column])
        return d['duration'].describe()

        # Return longitude and latitude of a location ID
    def get_position(self, column='PULocationID'):
        df = self.df
        f = geo().get_centroid()

        # Apply dictionary to column
        df['dummy'] = df[column].map(f)
        df.dropna(inplace=True)
        df[['longitude', 'latitude']] = pd.DataFrame(df['dummy'].tolist(), index=df.index)
        df = df.drop(['dummy'], axis=1)
        df['longitude'] = df['longitude'].round().astype(int)
        df['latitude'] = df['latitude'].round().astype(int)
        return df

    # def start_trips(self):
    #     df = self.get_time()
    #     d = df.groupby(['month'])
