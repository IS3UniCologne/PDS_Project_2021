from cabana import *
import pandas as pd
import datetime


def main():
    # data = geo()
    # print(data.get_centroid())
#-------------------------------------------------------------
    x = trips()
    # l = x.get_borough_locationID(borough='Bronx')
    # print(l)

    t = x.get_trips(fraction=0.05)
    # print(t.info())

    # df = x.get_time(column='tpep_pickup_datetime')
    # print(df.info())
    # print(df.head())
    #
    # d = x.get_duration()
    # print(d.head())
    # print(d.info())
    #
    # df = x.get_position()
    # print(df.info())
    # print(df.head())

#-------------------------------------------------------------
    y = time(t)
    m = y.get_aggregate('month')
    h = y.get_aggregate('hour')
    d = y.get_aggregate('weekday')
    print(m,h,d)


    # c = m.loc[:,['tpep_pickup_datetime','tpep_dropoff_datetime','duration']]
    # neg = c[c.duration < pd.Timedelta(days=0)]
    # pos = c[c.duration > pd.Timedelta(days=1)]
    # neg = 0
    # print(neg.head(10))
    # print(neg.info())
    # print(pos.head(10))
    # print(pos.info())


if __name__ == '__main__':
    main()
