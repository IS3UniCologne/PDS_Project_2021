import pandas as pd
from .geo import *

# Module gets data trips and  additional information include:
    # get_borough_locationID(borough) returns a list of all location ID in that borough, default "borough = 'Queens'"
    # get_trips() returns a given fraction data of trips data, default fraction=0.05

#-------------------------------------------------------------------------------------------

class trips:
    def __init__(self, fraction=0.1):
        pass

    # Location ID of a borough
    def get_borough_locationID(self,borough='Queens'):
        file = pd.read_csv(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.csv', sep=',')
        return file[file.Borough == borough.capitalize()]['LocationID'].unique()

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








