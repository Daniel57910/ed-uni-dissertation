
FEATURE_COLS = [
    "pred_ordinal_10",
    
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


META_COLS = [
    "user_id",
    "label",
    "session_30",
    "cum_session_event_raw",
    "cum_session_time_raw",
    "glob_platform_event",
    "glob_platform_time",
    
]

RL_STAT_COLUMNS = [
    'user_id',
    'session_30',
    'session_size',
    'sim_size',
    'session_minutes',

]
