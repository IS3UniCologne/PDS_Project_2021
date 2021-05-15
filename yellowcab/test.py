from cabana import *
import pandas as pd
import datetime


def main():
    # Test geo() module, giving a dict of approx. 260 keys
    # data = geo()
    # print(data.get_centroid())
#-------------------------------------------------------------
    # Test trip_inputs() module
    x = trips()
    # l = x.get_borough_locationID(borough="Bronx")
    # print(l)

    t = x.get_trips(fraction=0.05)
    # print(t.info())
    # print(t.head())
#-------------------------------------------------------------
    # Test trips_info() module
    y = trips_info(t)
    # df = y.get_time(column='tpep_dropoff_datetime')
    # print(df.info())
    # print(df.head())
    #
    # d = y.get_duration()
    # print(d.head())
    # print(d.info())
    #
    df = y.get_position()
    print(df.info())
    print(df.head())

    # m = y.get_aggregate('month')
    # h = y.get_aggregate('hour')
    # d = y.get_aggregate('weekday')
    # print(m,h,d)


if __name__ == '__main__':
    main()
