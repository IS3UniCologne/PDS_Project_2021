from cabana import *
import numpy as np
import matplotlib.pyplot as plt
import time
import geopandas as gpd
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sys import getsizeof
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

def main():
        # Test geo() module,
    # data = geo()
    # print(data.get_centroid()) # giving a dict of approx. 260 keys
    # print(data.map_locationID(location=10))
    # print(data.get_map())
    # print(data.df())
#-------------------------------------------------------------
        #Test trip_inputs() module
    # x = trips_input()
    # b = x.borough_list()
    # l = x.get_borough_locationID(borough="Bronx")
    # print(b,l)

    # t = x.get_trips(fraction=1)
    # print(t.info())
    # print(t.head())

    #     # Histogram of passenger_count
    # counts, bins = np.histogram(t.iloc[:,2], bins=10, range=(0, 10))
    # print(counts, bins)
    # plt.ylim([0,6000000])
    # plt.xlim([0,10])
    # plt.hist(bins[:-1], bins, weights=counts, color='mediumblue', ec='darkblue')
    # plt.show()
#-------------------------------------------------------------
       #Test trips_info() module
    # y = trips_info(t)
    # df = y.get_time(column='tpep_dropoff_datetime')
    # print(df.info())
    # print(df.head())

    # d = y.get_duration()
    # print(d.head())
    # print(d.info())
    #
    # df = y.get_position()
    # print(df.info())

    # a,b,c = y.outlier(df['longitude'])
    # d,e,f = y.outlier(df['latitude'])
    # start_location = df[(df['longitude']>a)&(df['longitude']<b)&(df['latitude']>d)&(df['latitude']<e)]
    # sl = start_location.loc[:,['longitude','latitude']]
    # print(sl.describe())
    # print(f'Number of outlier for longitude is {c}')
    # print(f'Number of outlier for latitude is {f}')

    # fig, axs = plt.subplots(ncols=2)
    # sns.distplot(sl['longitude'],color='b', ax=axs[0])
    # sns.distplot(sl['latitude'], color='r', ax=axs[1])
    # plt.show()

    # m = y.get_aggregate('month')
    # h = y.get_aggregate('hour')
    # d = y.get_aggregate('weekday')
    # print(m,h,d)

    # d = y.map_best_month()
    # print(d)
#--------------------------------------------------------------
    # Plot trip_distance distribution
    # start_time = time.time()
    # g,h,k = y.outlier(t['trip_distance'])
    # d = t[(t['trip_distance']<h)&(t['trip_distance']>g)]
    # sns.distplot(d['trip_distance'])
    # plt.show()
    # end_time = time.time()
    # print('Run time', end_time-start_time)
#-----------------------------------------------------------------------------
    # # Create full NYC file
    # x = trips_input()
    # d = x.get_trips(fraction=1,optimize=True)
    # start_time = time.time()
    # cols = 'tpep_dropoff_datetime tpep_pickup_datetime PULocationID DOLocationID'.split( )
    # t = d[cols]
    # y = trips_info(t)
    # df = y.get_position()
    # df2 = y.get_time()
    # df3 = y.get_duration()
    # nyc = pd.concat((df,df2),axis=1)
    # nyc['duration'] = df3.duration
    # nyc = nyc.drop(cols,axis=1)
    # # cols_to_use = nyc.columns.difference(t.columns)
    # full_nyc = pd.concat((nyc,d),axis=1)
    # full_nyc = full_nyc.drop(['tpep_dropoff_datetime', 'tpep_pickup_datetime'], axis=1)
    # # tp = trips_info(time_position)
    # # ready = tp.structure()
    # end_time = time.time()
    # print('Run time ',end_time-start_time)
    # print(full_nyc.info())
    # print(full_nyc.head())
    # full_nyc.to_csv(r'C:/Users/kyral/Documents/MIB 2019/BI Capstone Project/trip_data/nyc.csv')
# --------------------------------------------------------------
    # # Test Queens
    x = trips_input()
    df = x.get_queens()
    print(df.info())
    print(df.head())
#--------------------------------------------------------------------------------------------------------
    # #Build model
    # def pre_process(d):
    #     d.passenger_count = d.passenger_count.astype('uint8')
    #     d.RatecodeID = d.RatecodeID.astype('uint8')
    #     d.payment_type = d.payment_type.astype('uint8')
    #     d.PULocationID = d.PULocationID.astype('uint16')
    #     d.DOLocationID = d.DOLocationID.astype('uint16')
    #     monetary = 'fare_amount tip_amount total_amount tolls_amount extra mta_tax'.split()
    #     d[monetary] = abs(d[monetary].apply(lambda x: (x * 100).astype('int32')))
    #     d.improvement_surcharge = d.improvement_surcharge.astype('float32')
    #     d.congestion_surcharge = d.congestion_surcharge.astype('float32')
    #     d.trip_distance = abs(d.trip_distance.astype('float32'))
    #     return d

    #Chunked full file
    # start_time = time.time()
    # data = pd.read_csv(r'C:/Users/kyral/Documents/MIB 2019/BI Capstone Project/trip_data/full.csv',chunksize=100000,sep=',')

    # models = []
    # for chunk in data:
    #     chunk = pre_process(chunk)
    #     model = LinearRegression()
    #     model.fit(chunk.drop(['passenger_count','tpep_dropoff_datetime','tpep_pickup_datetime'],axis=1), chunk['passenger_count'])
    #     models.append(model)
    #     model.predict(chunk.drop(['passenger_count','tpep_dropoff_datetime','tpep_pickup_datetime'],axis=1))
    # end_time = time.time()
    # print('Process time:', end_time - start_time)



if __name__ == '__main__':
    main()
