import geojson
import numpy as np
from shapely.geometry import shape
from shapely.geometry import Point

# Module returns a dictionary of location ID as keys and central points as values

class geo:
    # Get geo data of a specific location
    def __init__(self):
        pass

        with open(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.geojson') as f:
            self.d = geojson.load(f)

        # self.data = [i for i in self.d['features'] if 'Queens' in i['properties'].values()]
        self.l = np.array([i['properties']['LocationID'] for i in self.d['features']])

    # Find central point of a specific location
    def get_centroid(self):
        # index = np.where(self.l == self.location)
        # i = index[0].item()
        center = []
        for i in range(len(self.l)):
            t = shape(self.d['features'][i]['geometry']).centroid
            point = Point(t)
            center.append([point.x,point.y])
        return dict(zip(self.l,center))
        # return len(self.l)







