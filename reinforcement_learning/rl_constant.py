FEATURE_COLS = [

    "country_count",
    "date_hour_cos",
    "date_hour_sin",
    "date_minute_cos",
    "date_minute_sin",
    
    "session_30_count",
    "session_5_count",
    "cum_session_event",
    "convolved_delta_event",
    "cum_session_time",
    
    "expanding_click_average",
    "cum_platform_time",
    "cum_platform_event",
    "cum_projects",
    "average_event_time",
    
    "rolling_session_time",
    "rolling_session_events",
    "rolling_session_gap",
    "previous_session_time",
    "previous_session_events",
]

METADATA_COLS = [
    "user_id",
    "date_time",
    "session_30_count_raw",
    "cum_platform_event_raw",
    "cum_platform_time_raw",
    "cum_session_event_raw",
    "global_events_user",
    "global_session_time",
    "reward",
    "session_minutes",
    "session_size",
    "sim_minutes",
    "sim_size",
    "reward"
]

RL_STAT_COLS = [
    'session_size',
    'session_minutes',
    'sim_size',
    'sim_minutes',
    'global_session_time',

]

PREDICTION_COLS = [
    "seq_40",
    "label"
]