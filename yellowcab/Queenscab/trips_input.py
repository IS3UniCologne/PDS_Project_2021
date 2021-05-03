import pandas as pd

# Modules extract trips data belong to Queens borough and return additional information:
# get_trips(filename) returns trips data of a given file (e.g. 01.parquet)
# get_time(column) add additional columns of
    # Month
    # Weekday
    # Hour
    # Weekend(binary)
# get_duration return trip duration
# get_position(column) add additional colomns of
    # Start Position(Longitude and Latitude),
    #End Position(see above).

class trips:
    def __init__(self, filename):
        self.f = filename

    # Location ID of Queens: Open taxi_zones file and return only Queens borough
    def get_locationID(self):
        file = pd.read_csv(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.csv', sep=',')
        return file[file.Borough == 'Queens']

    # Return trips data of Queens borough of a given file
    def get_trips(self):
        z = self.get_locationID()
        queens = z.LocationID.unique()
        raw = pd.read_parquet(f'C:/Users/kyral/Documents/MIB 2019/BI Capstone Project/trip_data/{self.f}',
                                  engine='pyarrow')
        df = raw[raw['PULocationID'].isin(queens)]
        return df

    # Parse datetime into Mo
    def get_time(self, column):
        df = self.get_trips()
        df['month'] = df[str(column)].dt.month
        df['hour'] = df[str(column)].dt.hour

        # Return day of week
        df['weekday'] = df[str(column)].dt.weekday
        df["weekday"] = df["weekday"].replace(list(range(0,7)),"Mon Tue Wed Thu Fri Sat Sun".split())

        # Return if a date is a weekend
        df['weekend'] = df[str(column)].dt.weekday
        df['weekend'] = df['weekend'].replace(list(range(0,6)),0).replace(6,1)
        return df




