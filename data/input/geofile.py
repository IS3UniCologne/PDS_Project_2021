import geojson

def map():
    with open('input/taxi_zones.geojson') as f:
        gj = geojson.load(f)
    features = gj['features']
    return features

if __name__ == '__main__':
    main()