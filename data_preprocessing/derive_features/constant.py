

INITIAL_LOAD_COLUMNS = [
    'user_id',
    'project_id',
    'date_time',
    'session_5_count',
    'session_30_count',
    'session_terminates_30_minutes',
    'country',
]

LABEL = [
    "session_terminates_30_minutes"
]


METADATA = [
    "user_id",
    "session_30_raw",
    "events_in_session_raw",
    "global_events_user",
    "global_session_time_minutes",
]

DATE_TIME = [
    "date_time"
    "year",
    "month",
    "day",
    "hour",
    "minute"
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



GROUPBY_COLS = ['user_id']

TORCH_LOAD_COLS = LABEL + METADATA + DATE_TIME + OUT_FEATURE_COLUMNS