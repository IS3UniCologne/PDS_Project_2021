from Queenscab import *
import time

def main():
    # data = geo(56)
    # print(data.get_centroid())

    x = trips('01.parquet')

    # # df = x.get_time('tpep_pickup_datetime')
    # # print(df.info())
    # # print(df.head())
    #
    # # print(x.get_duration().head())

    df = x.get_position('PULocationID')
    print(df.info())
    print(df.head(10))










if __name__ == '__main__':
    main()
