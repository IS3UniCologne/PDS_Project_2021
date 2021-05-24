import geojson
import numpy as np
from shapely.geometry import shape
from shapely.geometry import Point
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from .trips_input import *

#-------------------------------------------------------------------------------
# Module using geo data to return information
    # get_centroid() returns a dictionary of location ID as keys and central points as values
    # get_area(location) illustrates area and its central point, default location = 1
    # get_map() prints NYC's map
#-------------------------------------------------------------------------------

class geo:
    # Get geo data of a specific location
    def __init__(self):

        with open(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.geojson') as f:
            self.d = geojson.load(f)

        self.l = np.array([i['properties']['LocationID'] for i in self.d['features']])
        self.b = np.unique(np.array([i['properties']['borough'] for i in self.d['features']]))


    # Find central point of a specific location
    def get_centroid(self):
        center = []
        for i in range(len(self.l)):
            t = shape(self.d['features'][i]['geometry']).centroid
            point = Point(t)
            center.append([point.x,point.y])
        return dict(zip(self.l,center))
        # return len(self.l)

    # Map a location ID with its central point
    def map_locationID(self,location = 1):
        try:
            index = np.where(self.l == location)
            i = index[0].item()
            borough = self.d['features'][i]['properties']['borough']
            poly = self.d['features'][i]['geometry']
            point = self.get_centroid()[location]

            BLUE = '#6699cc'
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(point[0],point[1], 'ro')
            ax.add_patch(PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2))
            ax.axis('scaled')
            plt.title(f'Map of location {location}, borough {borough}')
            plt.show()
        except ValueError:
            pass

    # def map_borough(self,borough='Queens'):
    #     x = trips()
    #     v = [x.get_borough_locationID(i) for i in self.b]
    #     # d = dict(zip(self.b,v))
    #     poly = []
    #     for i in self.b:
    #         for j in v:
    #             index = np.where(self.l == j)
    #             ind = index[0].item()
    #             t = self.d['features'][ind]['geometry']['coordinates']
    #             poly.append(t)
    #
    #     BLUE = '#6699cc'
    #     fig = plt.figure()
    #     ax = fig.gca()
    #     for p in poly:
    #         ax.add_patch(PolygonPatch(p, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2))
    #         ax.axis('scaled')
    #     plt.title(f'The number of of trips start from a region {v[0]}')
    #     plt.show()


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









