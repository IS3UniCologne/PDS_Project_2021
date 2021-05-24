from .geo import *
from .trips_input import *
from .trips_info import *
import matplotlib.pyplot as plt
from descartes import PolygonPatch
# import geopandas as gpd

#--------------------------------------------------------------------------
# Map the number of start trip in regions of month with the most trips
#--------------------------------------------------------------------------
class map_trips_number:
    def __init__(self):
        self.g = world = gpd.read_file(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.geojson')

    def get_map(self):
        # Get month with the most trips
        x = trips()
        t = x.get_trips(fraction=0.05)
        y = trips_info(t)
        m = y.get_aggregate('month')
        month =  m[m['count'] == m['count'].max()].index[0].item()

        # Return borough from PULocationID of the chosen month
        file = pd.read_csv(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.csv', sep=',')
        boro = dict(zip(file.LocationID.values,file.Borough.values))

        mon = t[t.month == month]
        mon['borough'] = mon['PULocationID'].map(boro)
        mon = mon[mon.borough != "Unknown"]
        gr = mon.groupby(['borough']).count()

        self.g.plot()
        plt.show

