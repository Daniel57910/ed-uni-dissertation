SCALED_COLS =[
    'timestamp',
    'time_diff_seconds',
    '30_minute_session_count',
    '5_minute_session_count',
    'task_within_session_count',
    'user_count',
    'project_count',
    'country_count',
]

GENERATED_COLS = [
    'cum_events',
    'cum_projects',
    'cum_time',
    'cum_time_within_session',
    'av_time_across_clicks',
    'av_time_across_clicks_session',
    'rolling_average_tasks_within_session',
    'rolling_av_time_within_session',
    'rolling_time_between_sessions',
]

ENCODED_COLS = [
    'user_id',
    'project_id',
    'country'
]


GROUPBY_COLS = ['user_id']

TIMESTAMP_INDEX = 1

INITIAL_LOAD_COLUMNS = [
    'label_30',
    'date_time',
    'user_id',
    'project_id',
    'diff_minutes',
    'session_5',
    'session_30',
    'country',
]

TIMESTAMP_INDEX = 1

COUNTRY_ENCODING = {
    'Finland': 1,
    'United States': 2,
    'China': 3,
    'Singapore': 4,
}

PARTITION_LIST = [
    {
        'name': '125k',
        'size': 125000,
        'indexes': None
    },
    {
        'name': '125m',
        'size': 1250000,
        'indexes': None
    },
    {
        'name': '5m',
        'size': 5000000,
    },
    {
        'name': '10m',
        'size': 10000000,
    },
    {
        'name': '20m',
        'size': 20000000,
    },
    {
        'name': 'full',
        'size': None,
    }
]


