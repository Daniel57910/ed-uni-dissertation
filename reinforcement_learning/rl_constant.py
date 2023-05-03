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

PREDICTION_COLS = [
    'seq_10',
]


GROUPBY_COLS = ['user_id']

RL_STAT_COLS = [
    'session_size',
    'sim_size',
    'session_minutes',
    'sim_minutes',
    'reward',
    'session_30_raw',
    'cum_platform_time_raw',
    'global_session_time',
]
