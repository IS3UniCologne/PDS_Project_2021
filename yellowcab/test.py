from cabana import *
import numpy as np
import matplotlib.pyplot as plt


def main():
        # Test geo() module,
    # data = geo()
    # print(data.get_centroid()) - giving a dict of approx. 260 keys
    # print(data.map_locationID(location=10))
    # print(data.get_map())
#-------------------------------------------------------------
        # Test trip_inputs() module
    x = trips()
    # l = x.get_borough_locationID(borough="Bronx")
    # print(l)

    t = x.get_trips(fraction=0.3)
    # print(t.info())
    # print(t.head())
    # print(t.iloc[:,2].unique())
    counts, bins = np.histogram(t.iloc[:,2], bins=10, range=(0, 10))
    print(counts, bins)
    plt.ylim([0,6000000])
    plt.xlim([0,10])
    plt.hist(bins[:-1], bins, weights=counts, color='mediumblue', ec='darkblue')
    plt.show()

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
    # print(r)



if __name__ == '__main__':
    main()
