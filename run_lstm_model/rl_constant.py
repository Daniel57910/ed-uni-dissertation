LABEL = [
    "session_terminates_30_minutes"
]

METADATA = [
    "user_id",
    "session_30_raw",
    
    "cum_session_event_raw",
    "cum_session_time_raw",
    
    "cum_platform_event_raw",
    "global_events_user",
    "global_session_time_minutes",
]

DATE_TIME = [
    "date_time",
]

DATE_COLS = [
    'year',
    'month',
    'day',
    'hour',
    'minute',
]
OUT_FEATURE_COLUMNS = [
    "country_count",
    "timestamp_raw",
    "date_hour_sin",
    
    "date_hour_cos",
    "session_5_count",
    "session_30_count",
    
    "cum_session_event_count",
    "delta_last_event",
    "cum_session_time_minutes",
    
    "expanding_click_average",
    "cum_platform_time_minutes",
    "cum_platform_events",
    
    "cum_projects",
    "average_event_time",
    "rolling_session_time",
    
    "rolling_session_events",
    "rolling_session_gap",
    "session_event_count",
]

PREDICTION_COLS = [
    'prediction',
]

METADATA_STAT_COLUMNS = [
    'session_size',
    'sim_size',
    'session_minutes',
    'ended',
    'incentive_index',
    'reward',
    'n_episodes',
    'time_in_session',
]



TORCH_LOAD_COLS = LABEL + METADATA + DATE_TIME + OUT_FEATURE_COLUMNS + ['prediction']