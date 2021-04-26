import pandas as pd

def data():
    file = pd.read_csv('input/taxi_zones.csv',sep=',',index_col=0)
    return file[file.Borough == 'Queens']

def main():
    df = data()
    print(df.info())
    df.to_csv("C:/Users/kyral/Documents/GitHub/PDS_Yellowcab_UoC/data/output/queens.csv")

if __name__ == '__main__':
    main()
