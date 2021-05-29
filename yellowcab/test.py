from cabana import *
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

def main():
        # Test geo() module,
    data = geo()
    # print(data.get_centroid()) # giving a dict of approx. 260 keys
    # print(data.map_locationID(location=10))
    # print(data.get_map())
#-------------------------------------------------------------
        # Test trip_inputs() module
    x = trips_input()
    # l = x.get_borough_locationID(borough="Bronx")
    # print(l)

    t = x.get_trips(fraction=0.5)
    print(t.info())
    print(t.head())

       #  Histogram of passenger_count
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

    # df = y.get_position()
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
    # g,h,k = outlier(df['trip_distance'])
    # d = df[(df['trip_distance']<y)&(df['trip_distance']>x)]
    # sns.distplot(d['trip_distance'])
    # plt.show()

if __name__ == '__main__':
    main()
