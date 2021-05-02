import geojson
from shapely.geometry import shape
from shapely.geometry import Point

# Modules geo exploring geo data and area central point
class geo:
    # First we choose borough area
    # then find all the geo data of that borough
    def __init__(self, borough, location):
        self.b = str(borough)
        self.l = int(location)

        with open(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.geojson') as f:
            self.d = geojson.load(f)

        self.data = [i for i in self.d['features'] if self.b in i['properties'].values()]

    # Find central point of each location
    def get_centroid(self):
        # self.t = [i['geometry'] for i in self.data if self.location in i['properties'].values()]
        # print(len(self.t))
        for i in self.data:
            if self.l in i['properties'].values():
                t = shape(i['geometry']).centroid
                point = Point(t)
                return [point.x, point.y]



