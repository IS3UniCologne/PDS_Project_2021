import pandas as pd

# Location ID of Queens: Open taxi_zones file and return only Queens borough
def zone():
    file = pd.read_csv('input/taxi_zones.csv',sep=',')
    return file[file.Borough == 'Queens']

# Filter Queens trips: Open trip files and return data relates to Queens borough, then merge seperate files into 1 dataframe.
def data():
    z = zone()
    queens = z.LocationID.unique()
    result = []
    for i in range(1,10):
        raw = pd.read_parquet(f'C:/Users/kyral/Documents/MIB 2019/BI Capstone Project/trip_data/0{i}.parquet', engine='pyarrow')
        df = raw[raw['PULocationID'].isin(queens)]
        result.append(df)
    for j in range(10,13):
        raw = pd.read_parquet(f'C:/Users/kyral/Documents/MIB 2019/BI Capstone Project/trip_data/{j}.parquet', engine='pyarrow')
        df = raw[(raw['PULocationID'].isin(queens))|(raw['DOLocationID'].isin(queens))]
        result.append(df)
    return pd.concat(result)

def main():
    df = zone()
    #df.to_csv("C:/Users/kyral/Documents/GitHub/PDS_Yellowcab_UoC/data/output/locationID.csv")

    d = data()
    print(d.info())
    print(d.head())
    #d.to_csv("C:/Users/kyral/Documents/GitHub/PDS_Yellowcab_UoC/data/output/trips.csv")

if __name__ == '__main__':
    main()
