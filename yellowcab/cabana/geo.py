import geojson
import numpy as np
from shapely.geometry import shape
from shapely.geometry import Point
import shapely.ops as so
from descartes import PolygonPatch
import matplotlib.pyplot as plt


# Module using geo data to return information
    # get_centroid() returns a dictionary of location ID as keys and central points as values
    # get_area(location) illustrates area and its central point
    # get_map() prints NYC's map
#-------------------------------------------------------------------------------

class geo:
    # Get geo data of a specific location
    def __init__(self):

        with open(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.geojson') as f:
            self.d = geojson.load(f)

        self.l = np.array([i['properties']['LocationID'] for i in self.d['features']])

    # Find central point of a specific location
    def get_centroid(self):
        center = []
        for i in range(len(self.l)):
            t = shape(self.d['features'][i]['geometry']).centroid
            point = Point(t)
            center.append([point.x,point.y])
        return dict(zip(self.l,center))
        # return len(self.l)

    # Map one areas based on location ID
    def get_area(self,location = 22):
        index = np.where(self.l == location)
        i = index[0].item()
        poly = self.d['features'][i]['geometry']
        point = self.get_centroid()[location]

        BLUE = '#6699cc'
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(point[0],point[1], 'ro')
        ax.add_patch(PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2))
        ax.axis('scaled')
        plt.show()

    # Map NYC areas
    def get_map(self):
        poly = [self.d['features'][i]['geometry'] for i in range(len(self.l))]
        BLUE = '#6699cc'
        fig = plt.figure()
        ax = fig.gca()
        for p in poly:
            ax.add_patch(PolygonPatch(p, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2))
        ax.axis('scaled')
        plt.show()







