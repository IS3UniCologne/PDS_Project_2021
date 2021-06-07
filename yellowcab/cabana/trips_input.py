import pandas as pd
from .geo import *
from sys import getsizeof

# Module gets data trips and  additional information include:
    # get_borough_locationID(borough) returns a list of all location ID in that borough, default "borough = 'Queens'"
    # get_trips() returns a given fraction data of random trips data, default: fraction=0.05, optimize = True
#-------------------------------------------------------------------------------------------

class trips_input:
    def __init__(self):
        pass

    # Get list of borough
    def borough_list(self):
        file = pd.read_csv(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.csv', sep=',')
        return file.Borough.unique()

    # Location ID of a borough
    def get_borough_locationID(self,borough='Queens'):
        file = pd.read_csv(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.csv', sep=',')
        return file[file.Borough == borough]['LocationID'].unique()


    # Return a given fraction data of full trips data
    def get_trips(self, fraction=0.5, optimize=True):
        f = '01 02 03 04 05 06 07 08 09 10 11 12'.split( )
        result = []
        for i in f:
            raw = pd.read_parquet(f'C:/Users/kyral/Documents/MIB 2019/BI Capstone Project/trip_data/{i}.parquet',

                                  engine='pyarrow')
            df = raw.sample(frac=fraction,random_state=0)
            result.append(df)
        d = pd.concat(result)

        if optimize == True:
        # Optimize dtype to reduce file size
            d.passenger_count = d.passenger_count.astype('uint8')
            d.RatecodeID = d.RatecodeID.astype('uint8')
            d.payment_type = d.payment_type.astype('uint8')
            d.PULocationID = d.PULocationID.astype('uint16'  )
            d.DOLocationID = d.DOLocationID.astype('uint16')
            monetary = 'fare_amount tip_amount total_amount tolls_amount extra mta_tax'.split( )
            d[monetary]= abs(d[monetary].apply(lambda x: (x*100).astype('int32')))
            d.improvement_surcharge = d.improvement_surcharge.astype('float32')
            d.congestion_surcharge = d.congestion_surcharge.astype('float32')
            d.trip_distance = abs(d.trip_distance.astype('float32'))
        elif optimize ==False:
            pass
        return d

    def get_queens(self):
        qb = self.get_borough_locationID(borough='Queens')
        f = '01 02 03 04 05 06 07 08 09 10 11 12'.split()
        result = []
        for i in f:
            raw = pd.read_parquet(f'C:/Users/kyral/Documents/MIB 2019/BI Capstone Project/trip_data/{i}.parquet', engine='pyarrow')
            df = raw[raw.PULocationID.isin(qb)==True]
            result.append(df)
        d = pd.concat(result)
        return d