import geojson
from shapely.geometry import shape

def map():
    # Open geojson file
    with open('input/taxi_zones.geojson') as f:
        gj = geojson.load(f)

    # Extract geo-information of Queens borough
    for i in gj['features']:
        if 'Queens' in i['properties'].values():
            return i['geometry']

    # c = [i['geometry']['coordinates'] for i in gj['features'] if "Queens" in i['properties'].values()]

def main():
    x = map()
    # Return central point
    print(shape(x).centroid)

if __name__ == '__main__':
    main()