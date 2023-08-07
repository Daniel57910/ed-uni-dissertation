
cols = ['index', 'project_id', 'date_time', 'session_5_count', 'session_30_count', 'session_terminates_30_minutes', 'country', 'user_id', 'country_count', 'timestamp_raw', 'date_hour', 'date_hour_sin', 'date_hour_cos', 'cum_session_event_count', 'delta_last_event', 'cum_session_time_minutes', 'expanding_click_average', 'cum_platform_time_minutes', 'cum_platform_events', 'cum_projects', 'average_event_time', 'rolling_session_time', 'rolling_session_events', 'rolling_session_gap', 'session_event_count', 'session_time_minutes', 'rolling_session_gap', 'session_30_raw', 'global_session_time_minutes', 'global_events_user', 'cum_session_event_raw', 'cum_platform_event_raw', 'session_30_count_raw', 'year', 'month', 'day', 'hour', 'minute']


for col in sorted(cols):
    print(col)