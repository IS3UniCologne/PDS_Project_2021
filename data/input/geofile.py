import geojson
from shapely.geometry import shape
import matplotlib.pyplot as plt
from descartes import PolygonPatch

def map():
    # Open geojson file
    with open('input/taxi_zones.geojson') as f:
        gj = geojson.load(f)

    c = [i['geometry']['coordinates'] for i in gj['features'] if "Queens" in i['properties'].values()]
    return c

def main():
    t = map()
    # Return central point
    central = shape(t).centroid
    print(central)

    # # Plot shape of areas
    # BLUE = '#6699cc'
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.add_patch(PolygonPatch(t, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2))
    # ax.plot(1031085.718603112,164018.7544031972,'ro')
    # ax.axis('scaled')
    # plt.show()

if __name__ == '__main__':
    main()