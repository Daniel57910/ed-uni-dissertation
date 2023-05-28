import os

LABEL = [
    "continue_work_session_30_minutes"
]

METADATA = [
    "user_id",
    "session_30_raw",
    "cum_platform_event_raw",
    "cum_platform_time_raw",
    "cum_session_time_raw",
    "global_events_user",
    "global_session_time",
    
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second"
]

OUT_FEATURE_COLUMNS = [
    "country_count", 
    "date_hour_sin", 
    "date_hour_cos",
    "date_minute_sin",
    "date_minute_cos",
    
    "session_30_count",
    "session_5_count",
    "cum_session_event_count",
    "delta_last_event",
    "cum_session_time",
    
    "expanding_click_average",
    "cum_platform_time",
    "cum_platform_events",
    "cum_projects",
    "average_event_time",
    
    "rolling_session_time",
    "rolling_session_events",
    "rolling_session_gap",
    "previous_session_time",
    "previous_session_events",
]


GROUPBY_COLS = ['user_id']

LOAD_COLS = LABEL + METADATA + OUT_FEATURE_COLUMNS

S3_BUCKET = 'dissertation-data-dmiller'
BASE_CHECK_PATH = 'lstm_experiments/checkpoints/data_v1/n_files_30/ordinal'


LSTM_CHECKPOINTS = {
    'seq_10': os.path.join(BASE_CHECK_PATH, 'sequence_length_10', 'data_partition_None', '2023_04_28_20_16/clickstream-epoch=63-loss_valid=0.59.ckpt'),
    'seq_20': os.path.join(BASE_CHECK_PATH, 'sequence_length_20', 'data_partition_None', '2023_04_29_11_43/clickstream-epoch=85-loss_valid=0.59.ckpt'),
}