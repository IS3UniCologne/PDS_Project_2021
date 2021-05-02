import geojson
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

    # Find central point of each location
    def get_centroid(self):
        result = []
        for i in self.data:
            if self.location in i['properties'].values():
                    t = shape(i['geometry']).centroid
                    point = Point(t)
                    result.append((point.x, point.y))

        if len(result) > 0:
            print(result)
        print(f"Location ID {self.location} is not in Queens borough")



