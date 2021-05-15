import pandas as pd
from .trips_input import *
import datetime

# Modules calculate aggregate statistics for trips duration at Queens borough
# and visualize important findings such as:
    # the number of started trips per region in a map for the month with the most trips
    # the  distribution  of  trip  lengths of each month
    # heatmap of ....
# get_aggregate(column) returns aggregate statistics, grouped by specified column
#

class time:
    def __init__(self, dataframe):
        self.df = dataframe

    def get_time(self):
        df = self.df
        df['month'] = df['tpep_pickup_datetime'].dt.month
        df['hour'] = df['tpep_pickup_datetime'].dt.hour

        # Return day of week
        df['weekday'] = df['tpep_pickup_datetime'].dt.weekday
        df["weekday"] = df["weekday"].replace(list(range(0, 7)), "Mon Tue Wed Thu Fri Sat Sun".split())

        # Return if a date is a weekend
        df['weekend'] = df['tpep_pickup_datetime'].dt.weekday
        df['weekend'] = df['weekend'].replace(list(range(0, 6)), 0).replace(6, 1)
        return df

    def get_duration(self):
        df = self.df
        df['duration'] = df['tpep_dropoff_datetime']-df['tpep_pickup_datetime']
        return df['duration']

    # Return aggregate values by time
    def get_aggregate(self, column='month'):
        df = self.get_time()
        df['duration'] = abs(self.get_duration())

        d = df.groupby([column])
        return d['duration'].describe()


    # def start_trips(self):
    #     df = self.get_time()
    #     d = df.groupby(['month'])
