from explore import geo

def main():
    data = geo('Queens', 2)
    data.get_centroid()

if __name__ == '__main__':
    main()
