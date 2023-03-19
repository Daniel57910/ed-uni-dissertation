import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 400)
pd.options.mode.chained_assignment = None  # default='warn'
DEBUG_COLS = [
    'user_id',
    'date_time',
    'session_30',
    'session_5',
    'label_5',
    'label_30',
    'diff_minutes'
]
    
def test_session_assignment():

    df = pd.read_csv('datasets/frequency_encoded_2.csv')
    df = df[DEBUG_COLS]
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(['date_time'])

    less_5 = df[df['label_5'] == True]
    less_30 = df[df['label_30'] == True]
    
    assert less_5.shape[0] == df[df['diff_minutes'] < 5].shape[0]
    assert less_30.shape[0] == df[df['diff_minutes'] < 30].shape[0]
        
def test_session_boundary():
    calculated_df = pd.read_csv('datasets/frequency_encoded_2.csv')
    calculated_df = calculated_df[DEBUG_COLS]
    calculated_df['date_time'] = pd.to_datetime(calculated_df['date_time'])
    calculated_df = calculated_df.sort_values(['date_time'])
    
    for user_stats, df_subset in calculated_df.groupby(['user_id', 'session_30']):
        df_subset_in_range = df_subset.iloc[:df_subset.shape[0] - 2]
        inflection = df_subset.iloc[-1]
        if df_subset_in_range.shape[0] > 2:
            df_subset_in_range = df_subset_in_range.iloc[0: df_subset_in_range.shape[0] - 2]
            assert df_subset_in_range[df_subset_in_range['label_30'] == False].shape[0] == 0
            assert inflection['label_30'] == False
        
        if inflection.session_30 > 1:
            user_id, session_30 = user_stats
            previous_inflections = calculated_df[
                (calculated_df['user_id'] == user_id) & (calculated_df['session_30'] == session_30 - 1)
            ]
            
            max_row = previous_inflections.iloc[-1]
            if max_row.shape[0] > 0:
                assert max_row['label_30'] == False
        

