import pandas as pd
import numpy as np
import logging

TEST_U_ID = [2373355, 10, 4301]
SAMPLE_COLS = [
    'user_id',
    'date_time',
    'row_count',
    'label_30',
    'session_30',
]

from pprint import pformat
class SessionCalculate:
    logger = logging.getLogger(__name__)
    def __init__(self, df, write_path, use_gpu, test_env) -> None:
        self.df = df
        self.write_path = write_path
        self.use_gpu = use_gpu
        self.test_env = test_env
    
    def calculate_inflections(self):
       
        self.logger.info('Calculating subsequent date time')
        self.df['next_date_time'] = self.df.groupby('user_id')['date_time'].shift(-1)
        self.df = self.df.drop_duplicates(subset=['user_id', 'date_time'], keep='last').reset_index()
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
        self.df['session_5'] = (self.df['diff_minutes'] < 5)
        self.df['session_30'] = self.df['diff_minutes'] < 30
        
        self.logger.info(f'Labels calculated: removing rows with diff seconds > 0')
        self.logger.info(self.df[self.df['user_id'] == 10].head(10))
       
        

        self.df = self.df.drop(columns=['next_date_time', 'diff_seconds'])
        self.logger.info(f'Number of rows following drop: {self.df.shape[0]}')
        self.logger.info(f'Sorting rows by date time and applying row count')
        self.df = self.df.sort_values(['date_time']).reset_index()
        self.df['row_count'] = self.df.index.values
        self.logger.info(f'Sorted rows and applied row count on updated index')  
        self.logger.info('Calculating inflection points')
        self.df['user_id'] = self.df['user_id'].astype('int32')
       
        inflections_5_merge = self.df[self.df['session_5'] == False].sort_values(by=['date_time'])
        inflections_30_merge = self.df[self.df['session_30'] == False].sort_values(by=['date_time']) 
       
        self.logger.info('Calculating session 5 inflections') 
        inflections_5_merge['session_5'] = inflections_5_merge.groupby('user_id').cumcount() + 1
        self.logger.info('Calculating session 30 inflections')
        inflections_30_merge['session_30'] = inflections_30_merge.groupby('user_id').cumcount() + 1
        
        inflections_5_merge = inflections_5_merge[['user_id', 'date_time', 'row_count', 'session_5']].sort_values(by=['row_count', 'user_id'])
        inflections_30_merge = inflections_30_merge[['user_id', 'date_time', 'row_count', 'session_30']].sort_values(by=['row_count', 'user_id'])
        inflections_5_merge = inflections_5_merge.drop(columns=['date_time'])
        
        inflections_30_merge = inflections_30_merge.rename(columns={'date_time': 'session_end_time'})

        self.df = self.df.drop(columns=['session_5', 'session_30', 'diff_minutes'])
        
        self.df = pd.merge_asof(self.df.sort_values(by=['row_count', 'user_id']), inflections_5_merge, on='row_count', by='user_id', direction='forward')
        self.df = pd.merge_asof(self.df.sort_values(by=['row_count', 'user_id']), inflections_30_merge, on='row_count', by='user_id', direction='forward')
        if self.use_gpu:
            import cudf
            self.logger.info('Bringing back to GPU for labelling')
            self.df = self.df.to_pandas()
            self.df['session_terminates_30_minutes'] = (self.df['session_end_time'] - self.df['date_time']).apply(lambda x: x.total_seconds() / 60) < 30
            self.df = cudf.from_pandas(self.df)
        else:
            self.df['session_terminates_30_minutes'] = (self.df['session_end_time'] - self.df['date_time']).apply(lambda x: x.total_seconds() / 60) < 30
 
        self.logger.info('Inflections calculated')
  
        session_end_30_minutes = self.df[self.df['session_terminates_30_minutes'] == False].shape[0]
        self.logger.info(f'Percent sessions end in 30 {session_end_30_minutes / self.df.shape[0]}')
        self.logger.info(f'Columns for df')
        self.logger.info(pformat(self.df.columns))
    def write_inflections_parquet(self):
    
        self.df = self.df.drop(columns=['index', 'level_0'])
       
        if not self.test_env:
            self.df = self.df.drop(columns=['diff_minutes', 'row_count'])
        
        if self.use_gpu:
            import dask_cudf as ddf
            self.logger.info('Bringing back to dask GPU for final calculations')
            self.df = ddf.from_cudf(self.df, npartitions=30)
        else:
            import dask.dataframe as ddf
            self.logger.info('Bringing back to dask CPU for final calculations')
            self.df = ddf.from_pandas(self.df, npartitions=30)
        
        self.logger.info(f'Writing inflections to {self.write_path}')    
        # write_path = self.write_path + '.parquet.gzip'
        self.df.to_parquet(self.write_path)