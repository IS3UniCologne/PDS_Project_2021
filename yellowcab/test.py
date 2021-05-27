from cabana import *
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns


def outlier(series):
    descriptive = series.describe().transpose()
    iqr = descriptive.loc['75%'] - descriptive.loc['25%']
    low_boundary = descriptive.loc['25%'] - 1.5 * iqr
    up_boundary = descriptive.loc['75%'] + 1.5 * iqr
    outliers = sum(0 if x >= low_boundary and x <= up_boundary else 1 for x in series)
    return low_boundary,up_boundary,outliers

def main():
        # Test geo() module,
    # data = geo()
    # print(data.get_centroid()) - giving a dict of approx. 260 keys
    # print(data.map_locationID(location=10))
    # print(data.get_map())
#-------------------------------------------------------------
        # Test trip_inputs() module
    # x = trips()
    # l = x.get_borough_locationID(borough="Bronx")
    # print(l)

    # t = x.get_trips(fraction=0.3)
    # print(t.info())
    # print(t.head())
    # print(t.iloc[:,2].unique())

       # # Histogram of passenger_count
    # counts, bins = np.histogram(t.iloc[:,2], bins=10, range=(0, 10))
    # print(counts, bins)
    # plt.ylim([0,6000000])
    # plt.xlim([0,10])
    # plt.hist(bins[:-1], bins, weights=counts, color='mediumblue', ec='darkblue')
    # plt.show()

#-------------------------------------------------------------
        # Test trips_info() module
    # y = trips_info(t)
    # df = y.get_time(column='tpep_dropoff_datetime')
    # print(df.info())
    # print(df.head())

    # d = y.get_duration()
    # print(d.head())
    # print(d.info())

    # df = y.get_position()
    # print(df.info())
    # print(df.head())

    # m = y.get_aggregate('month')
    # h = y.get_aggregate('hour')
    # d = y.get_aggregate('weekday')
    # print(m,h,d)
#-------------------------------------------------------------
        # Test map_trips_number

    # x = map_trips_number()
    # r = x.get_map()
    # print(r.head())
    # print(r.info())
    # print(r)
#--------------------------------------------------------------
    # Test full file
    # start_time = time.time()
    result = []
    for i in range(1,10):
        raw = pd.read_parquet(f'C:/Users/kyral/Documents/MIB 2019/BI Capstone Project/trip_data/0{i}.parquet', engine='pyarrow')
        # df = raw[raw['PULocationID'].isin(queens)]
        result.append(raw)
    for j in range(10,13):
        raw = pd.read_parquet(f'C:/Users/kyral/Documents/MIB 2019/BI Capstone Project/trip_data/{j}.parquet', engine='pyarrow')
        # df = raw[(raw['PULocationID'].isin(queens))|(raw['DOLocationID'].isin(queens))]
        result.append(raw)
    df =pd.concat(result)
    # print(df.info())
    # print(df.describe())

    # x,y,z = outlier(df['trip_distance'])
    # d = df[(df['trip_distance']<y)&(df['trip_distance']>x)]

    # sns.distplot(d['trip_distance'])
    # plt.show()

    t = trips_info(df)
    test = t.get_position(df)
    x, y, z = outlier(test['longitude'])
    print(test.info())
    # end_time = time.time()
    # print(f'Full file histogram time is: {end_time-start_time}')


if __name__ == '__main__':
    main()
