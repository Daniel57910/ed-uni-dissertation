import pandas as pd
import numpy as np
import logging

TEST_U_ID = 2373355
SAMPLE_COLS = [
    'user_id',
    'date_time',
    'row_count',
    'label_30',
    'session_30',
]
def outer_apply_inflection_points(inflections):
    def inner_find_session(row):
        inflection_for_user = inflections[row['user_id']]
        index_for_row = row['row_count']
        return 1 + np.searchsorted(inflection_for_user, index_for_row, side='left')

    return inner_find_session

def get_inflection_points_5(subset):
    return subset[subset['label_5'] == False].index.values

def get_inflection_points_30(subset):
    return subset[subset['label_30'] == False].index.values

import pdb
class SessionCalculate:
    logger = logging.getLogger(__name__)
    def __init__(self, df, write_path, use_gpu) -> None:
        self.df = df
        self.write_path = write_path
        self.use_gpu = use_gpu
    
    def calculate_inflections(self):
       
        initial_df = self.df.copy() 
        self.logger.info('Calculating subsequent date time')
        self.df['next_date_time'] = self.df.groupby('user_id')['date_time'].shift(-1)
        self.df = self.df.drop_duplicates(subset=['user_id', 'date_time'], keep='last').reset_index()
        print(self.df[self.df['user_id'] == 10])
        if self.use_gpu:
            self.logger.info('Bringing to CPU for second calculation')
            self.df = self.df.to_pandas()
           
            
        self.df['diff_seconds'] = (self.df['next_date_time'] - self.df['date_time']).apply(lambda x: x.total_seconds())
        self.logger.info('Diff seconds calculated')
        if self.use_gpu:
            import cudf
            self.logger.info('Bringing back to GPU for final calculations')
            self.df = cudf.from_pandas(self.df)

        self.df['diff_minutes'] = (self.df['diff_seconds'] / 60)
        self.df['label_5'] = (self.df['diff_minutes'] < 5)
        self.df['label_30'] = self.df['diff_minutes'] < 30
        
        self.logger.info(f'Labels calculated: removing rows with diff seconds > 0')
        self.logger.info(self.df[self.df['user_id'] == 10].head(10))
        
        self.df = self.df.drop(columns=['next_date_time', 'diff_seconds'])
        self.logger.info(f'Number of rows following drop: {self.df.shape[0]}')
        self.logger.info(f'Sorting rows by date time and applying row count')
        self.df = self.df.sort_values(['date_time']).reset_index()
        self.df['row_count'] = self.df.index.values
        self.logger.info(f'Sorted rows and applied row count on updated index')  
        self.logger.info('Calculating inflection points')
        
        inflections_5, inflections_30 = self.df.groupby('user_id').apply(get_inflection_points_5), self.df.groupby('user_id').apply(get_inflection_points_30)

        inner_inflection_5, inner_inflection_30 = outer_apply_inflection_points(inflections_5), outer_apply_inflection_points(inflections_30)
     
     
        self.logger.info('Applying inflection points to session 30')
        self.df['session_30'] = self.df.apply(inner_inflection_30, axis=1)
       
        self.logger.info('Applying inflection points to session 5') 
        self.df['session_5'] = self.df.apply(inner_inflection_5, axis=1)
        
        self.logger.info('Inflections calculated')
   
    
    def write_inflections_parquet(self):
        self.logger.info(f'Writing inflections to {self.write_path}.csv')
        # write_path = self.write_path + '.parquet.gzip'
        self.df.to_csv(self.write_path + '.csv')