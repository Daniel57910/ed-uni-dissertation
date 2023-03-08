import argparse
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dask import dataframe as dd
from dask.distributed import Client
from dask.distributed import LocalCluster


def time_based_csv():

    cluster = LocalCluster()
    cluster.scale(5)
    client = Client(cluster)
    df = dd.read_csv('../datasets/data_preprocessing/time_split/time_split_*.csv', assume_missing=True)

    agg_over_year = df.groupby(['country', 'year']).agg({'subjects_ids': 'count'}).compute().reset_index()
    agg_over_year = agg_over_year.sort_values(by=['country', 'year'])

    agg_over_month = df.groupby(['country', 'month']).agg({'subjects_ids': 'count'}).compute().reset_index()
    agg_over_month = agg_over_month.sort_values(by=['country', 'month'])

    agg_over_day = df.groupby(['country', 'day']).agg({'subjects_ids': 'count'}).compute().reset_index()
    agg_over_day = agg_over_day.sort_values(by=['country', 'day'])

    agg_over_hour = df.groupby(['country', 'hour']).agg({'subjects_ids': 'count'}).compute().reset_index()
    agg_over_hour = agg_over_hour.sort_values(by=['country', 'hour'])

    agg_over_weekened = df.groupby(['country', 'is_weekend']).agg({'subjects_ids': 'count'}).compute().reset_index()
    agg_over_weekened = agg_over_weekened.sort_values(by=['country', 'is_weekend'])

    agg_over_year.to_csv('../datasets/summary_stats/time_based_stats/activity_by_year.csv', index=False)
    agg_over_month.to_csv('../datasets/summary_stats/time_based_stats/activity_by_month.csv', index=False)
    agg_over_day.to_csv('../datasets/summary_stats/time_based_stats/activity_by_day.csv', index=False)
    agg_over_hour.to_csv('../datasets/summary_stats/time_based_stats/activity_by_hour.csv', index=False)
    agg_over_weekened.to_csv('../datasets/summary_stats/time_based_stats/activity_by_weekend.csv', index=False)


def over_year():
    over_year = pd.read_csv('../datasets/summary_stats/time_based_stats/activity_by_year.csv')
    over_year_plot = sns.barplot(data=over_year, x='year', y='subjects_ids', hue='country').set_title('Engagement by Year Per Country')
    plt.show()

def over_month():
    over_month = pd.read_csv('../datasets/summary_stats/time_based_stats/activity_by_month.csv')
    over_month_plot = sns.barplot(data=over_month, x='month', y='subjects_ids', hue='country').set_title('Engagement by Month Per Country')
    plt.show()

def over_day():
    over_day = pd.read_csv('../datasets/summary_stats/time_based_stats/activity_by_day.csv')
    over_day_plot = sns.barplot(data=over_day, x='day', y='subjects_ids', hue='country').set_title('Engagement by Day Per Country')
    plt.show()

def over_weekend():
    over_weekend = pd.read_csv('../datasets/summary_stats/time_based_stats/activity_by_weekend.csv')
    over_weekend_plot = sns.barplot(data=over_weekend, x='is_weekend', y='subjects_ids', hue='country').set_title('Engagement by Weekend Per Country')
    plt.show()

def over_hour():
    over_hour = pd.read_csv('../datasets/summary_stats/time_based_stats/activity_by_hour.csv')
    over_hour_plot = sns.barplot(data=over_hour, x='hour', y='subjects_ids', hue='country').set_title('Engagement by Hour Per Country')
    plt.show()

def over_weekend():
    over_weekend = pd.read_csv('../datasets/summary_stats/time_based_stats/activity_by_weekend.csv')
    over_weekend_plot = sns.barplot(data=over_weekend, x='is_weekend', y='subjects_ids', hue='country').set_title('Engagement by Weekend Per Country')
    plt.show()
