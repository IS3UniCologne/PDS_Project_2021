import geojson
import numpy as np
from shapely.geometry import shape
from shapely.geometry import Point

# Modules exploring geo data and area central point.
# First we find all the geo data of Queens borough
# and return central point of the specific location

class geo:
    # Get geo data of Queens borough
    def __init__(self, location):
        self.location = location

        with open(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.geojson') as f:
            self.d = geojson.load(f)

        self.data = [i for i in self.d['features'] if 'Queens' in i['properties'].values()]
        self.l = np.array([i['properties']['LocationID'] for i in self.data])

    # Find central point of a specific location
    def get_centroid(self):
        if self.location in self.l:
            index = np.where(self.l == self.location)
            i = index[0].item()
            t = shape(self.data[i]['geometry']).centroid
            point = Point(t)
            return point.x,point.y
        else:
            pass





