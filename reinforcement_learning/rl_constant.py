FEATURE_COLUMNS = [
    
    "user_count",
    "project_count", 
    "country_count", 
    "date_hour_sin", 
    "date_hour_cos",
    "date_minute_sin",
    "date_minute_cos",
    
    "session_30_count",
    "session_5_count",
    "cum_session_event",
    "cum_session_time",
    "expanding_click_average",
   
    "cum_platform_time",
    "cum_platform_event",
    "cum_projects",
    "average_event_time",
    "delta_last_event",
    
    "rolling_session_time",
    "rolling_session_events",
    "rolling_session_gap",
    "previous_session_time",
    "previous_session_events",
]



METADATA = [
    "user_id",
    "session_30_count_raw",
    "cum_platform_event_raw",
    "cum_platform_time_raw",
    "cum_session_time_raw",
    "cum_session_event_raw",
    "date_time"
]

RL_STAT_COLS = [
    'session_size',
    'session_minutes',
    'size_cutoff',
    'time_cutoff',
    'reward'
]

PREDICTION_COLS = [
    "label",
    "pred"
]

LOAD_COLS = list(set(FEATURE_COLUMNS + METADATA + RL_STAT_COLS + PREDICTION_COLS))