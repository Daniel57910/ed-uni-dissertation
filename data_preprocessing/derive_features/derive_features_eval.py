import pandas as pd
import dask.dataframe as dd
import pprint
WRITE_PATH = 'datasets/calculated_features/files_used_2'

def main():
    df = dd.read_parquet(f'{WRITE_PATH}/*.parquet')
    df = df.compute()
   
    for col in df.columns:
        print(col) 
    



if __name__ == "__main__":
    main()