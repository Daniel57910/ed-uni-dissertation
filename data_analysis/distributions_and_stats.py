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

GLOBAL_UNIQUE_USERS = 109986
GLOBAL_UNIQUE_PROJECTS = 874
GLOBAL_TOTAL_SUBJECTS = 39488608
CORE_PER_COUNTRY_PATH = "../datasets/summary_stats/per_country"

def subject_bin(subjects_per_user, axis=None):
    bin_range_low = np.arange(0, 100, 10)
    bin_range_mid = np.arange(100, 200, 25)
    bin_range_high = np.arange(200, 1000, 100)
    bin_range = np.concatenate((bin_range_low, bin_range_mid, bin_range_high))
    sub_1000 = subjects_per_user[subjects_per_user.subjects_ids < 1000]

    if axis:
        sns.histplot(sub_1000.subjects_ids, kde=True, ax=axis)
    else:
        sns.histplot(sub_1000.subjects_ids, kde=True)
        plt.show()

def subjects_per_project_bin(subjects_per_project, axis=None):
    bin_range_low = np.arange(0, 100, 10)
    bin_range_mid = np.arange(100, 200, 25)
    bin_range_high = np.arange(200, 1000, 100)

    subjects_per_project = subjects_per_project.sort_values(by='subjects_ids', ascending=False)
    pop_subjects = subjects_per_project[subjects_per_project.subjects_ids > 47595]
    if axis:
        sns.histplot(pop_subjects.subjects_ids, kde=True, ax=axis)
    else:
        sns.histplot(pop_subjects.subjects_ids, kde=True)
        plt.show()


def workflow_bin(workflows_per_user, axis=None):

    bin_range_low = np.arange(0, 50, 5)
    bin_range_mid = np.arange(50, 200, 10)
    bin_range_high = np.arange(200, 1000, 100)

    bin_range = np.concatenate((bin_range_low, bin_range_mid, bin_range_high))
    histogram = np.histogram(workflows_per_user.workflow_id, bins=bin_range)

    df = pd.DataFrame({'workflow_bin': histogram[1][1:], 'workflow_count': histogram[0]})
    if axis:
        sns.barplot(data=df, x='workflow_bin', y='workflow_count', ax=axis)
    else:
        sns.barplot(data=df, x='workflow_bin', y='workflow_count')
        plt.show()


def project_multiple_countries(df_matrix):
    fig, axes = plt.subplots(4, 1)

    for country_name, ax in zip(df_matrix, axes):
        df = df_matrix[country_name]
        df = df[df.project_id < 500]
        sns.histplot(df.project_id, kde=True, ax=ax).set(title=f'Projects Per user {country_name}')

    plt.tight_layout()
    plt.show()


def subject_multiple_countries(df_matrix):
    fig, axes = plt.subplots(4, 1)

    for country_name, ax in zip(df_matrix, axes):
        df = df_matrix[country_name]
        df = df[df.subjects_ids < 1000]
        sns.histplot(df.subjects_ids, kde=True, ax=ax).set(title=f'Subjects Per user {country_name}')

    plt.tight_layout()
    plt.show()

def subject_per_project_countries(df_matrix):
    fig, axes = plt.subplots(4, 1)

    for country_name, ax in zip(df_matrix, axes):
        df = df_matrix[country_name]
        df = df[df.subjects_ids > 47595]
        sns.histplot(df.subjects_ids, kde=True, ax=ax).set(title=f'Subjects Per Project {country_name}')

    plt.tight_layout()
    plt.show()

def workflow_multiple_countries(df_matrix):

    fig, axes = plt.subplots(4, 1)

    for country_name, ax in zip(df_matrix, axes):
        df = df_matrix[country_name]
        df = df[df.workflow_id < 1000]
        sns.histplot(df.workflow_id, kde=True, ax=ax).set(title=f'Workflows Per user {country_name}')

    plt.tight_layout()
    plt.show()

def project_bin(projects_per_user, axis=None):

    bin_range_low = np.arange(0, 100, 10)
    bin_range_mid = np.arange(100, 200, 25)
    bin_range_high = np.arange(200, 550, 50)

    sub_250_projects = projects_per_user[projects_per_user.project_id < 500]
    print(sub_250_projects.count() / GLOBAL_UNIQUE_USERS)

    bin_range = np.concatenate((bin_range_low, bin_range_mid, bin_range_high))
    histogram = np.histogram(projects_per_user.project_id, bins=bin_range)
    df = pd.DataFrame({'project_bin': histogram[1][1:], 'project_count': histogram[0]})
    if axis:
        sns.histplot(sub_250_projects.project_id, kde=True, ax=axis)
    else:
        sns.histplot(sub_250_projects.project_id, kde=True)
        plt.show()


def country_matrix(project_path):
    project_files = os.listdir(project_path)
    project_matrix = {}

    for path in project_files:
        country_name = path.split("_")[0]
        print(f'Getting distributions for {country_name}')
        project_matrix[country_name] = pd.read_csv(os.path.join(project_path, path))

    return project_matrix

def generate_per_country_distributions(df=None):

        if df is not None:
            df = df[['user_id', 'project_id', 'workflow_id', 'subjects_ids', 'country']]

            unique_countries = df.country.unique().compute()
            for country in unique_countries:
                sample_df = df[df.country == country]
                print(f'Getting distributions for {country}')
                projects_per_user = sample_df.groupby('user_id').agg({'project_id': 'count'}).compute()
                workflows_per_user = sample_df.groupby('user_id').agg({'workflow_id': 'count'}).compute()
                subjects_per_user = sample_df.groupby('user_id').agg({'subjects_ids': 'count'}).compute()
                subjects_per_project = sample_df.groupby('project_id').agg({'subjects_ids': 'count'}).compute()

                projects_per_user.to_csv(f'../datasets/summary_stats/{country}_projects_per_user.csv', index=False)
                workflows_per_user.to_csv(f'../datasets/summary_stats/{country}_workflows_per_user.csv', index=False)
                subjects_per_user.to_csv(f'../datasets/summary_stats/{country}_subjects_per_user.csv', index=False)
                subjects_per_project.to_csv(f'../datasets/summary_stats/{country}_subjects_per_project.csv', index=False)

        else:
            projects_per_user = os.path.join(CORE_PER_COUNTRY_PATH, 'projects_per_user_country')
            workflow_per_user = os.path.join(CORE_PER_COUNTRY_PATH, 'workflow_per_user_country')
            subjects_per_user = os.path.join(CORE_PER_COUNTRY_PATH, 'subject_per_user_country')
            subject_per_project_country = os.path.join(CORE_PER_COUNTRY_PATH, 'subject_per_project_country')

            # projects_matrix = country_matrix(projects_per_user)
            # workflow_matrix = country_matrix(workflow_per_user)
            # subjects_matrix = country_matrix(subjects_per_user)
            subject_per_project_matrix = country_matrix(subject_per_project_country)

            # project_multiple_countries(projects_matrix)
            # workflow_multiple_countries(workflow_matrix)
            # subject_multiple_countries(subjects_matrix)
            subject_per_project_countries(subject_per_project_matrix)



def generate_distributions(df=None):
    print('Generating distributions...')

    if df is None:
        subjects_per_user = pd.read_csv('../datasets/summary_stats/subjects_per_user.csv')
        workflows_per_user = pd.read_csv('../datasets/summary_stats/workflows_per_user.csv')
        projects_per_user = pd.read_csv('../datasets/summary_stats/projects_per_user.csv')
        subjects_per_project = pd.read_csv('../datasets/summary_stats/subjects_per_project.csv')

    else:
        subjects_per_user = df[['user_id', 'subjects_ids']].groupby('user_id').subjects_ids.count().reset_index().compute()
        workflows_per_user = df[['user_id', 'workflow_id']].groupby('user_id').workflow_id.count().reset_index().compute()
        projects_per_user = df[['user_id', 'project_id']].groupby('user_id').project_id.count().reset_index().compute()
        subjects_per_project = df[['project_id', 'subjects_ids']].groupby('project_id').subjects_ids.count().reset_index().compute()

        subjects_per_project.to_csv('../datasets/summary_stats/subjects_per_project.csv', index=False)
        subjects_per_user.to_csv('../datasets/summary_stats/subjects_per_user.csv', index=False)
        workflows_per_user.to_csv('../datasets/summary_stats/workflows_per_user.csv', index=False)
        projects_per_user.to_csv('../datasets/summary_stats/projects_per_user.csv', index=False)

    subject_bin(subjects_per_user)
    workflow_bin(workflows_per_user)
    project_bin(projects_per_user)
    subjects_per_project_bin(subjects_per_project)


def display_summary_stats(df):

    project_per_country = df.groupby('country').project_id.count().compute()
    unique_project_per_country = df[['country', 'project_id']].drop_duplicates().groupby('country').count().compute()

    workflow_per_country = df.groupby('country').workflow_id.count().compute()
    unique_workflow_per_country = df[['country', 'workflow_id']].drop_duplicates().groupby('country').count().compute()

    subject_per_country = df.groupby('country').subjects_ids.count().compute()
    unique_subject_per_country = df[['country', 'subjects_ids']].drop_duplicates().groupby('country').count().compute()

    print(f'Project per country: {project_per_country}')
    print(f'Unique project per country: {unique_project_per_country}')


    print(f'Workflow per country: {workflow_per_country}')
    print(f'Unique workflow per country: {unique_workflow_per_country}')

    print(f'Subject per country: {subject_per_country}')
    print(f'Unique subject per country: {unique_subject_per_country}')

    print(f'Min max date')
    print(df['date_time'].min().compute(), df['date_time'].max().compute())

    print(f'Global unique statistics')
    print(f'Unique users: {df.user_id.nunique().compute()}')
    print(f'Unique projects: {df.project_id.nunique().compute()}')
    print(f'Unique workflows: {df.workflow_id.nunique().compute()}')
    print(f'Unique subjects: {df.subjects_ids.nunique().compute()}')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../datasets/labelled/clickstream_data_sorted/')
    parser.add_argument('--dask-client', '-da', default=False, action='store_true')
    parser.add_argument('--summary-stats', '-s', default=False, action='store_true')
    parser.add_argument('--distributions', '-d', default=False, action='store_true')
    parser.add_argument('--per-country-dist', '-p', default=True, action='store_true')

    args = parser.parse_args()

    if args.dask_client:
        cluster = LocalCluster()
        cluster.scale(10)
        client = Client(cluster)
        df = dd.read_csv(args.path + '*.csv')

    if args.summary_stats:
        display_summary_stats(df)

    if args.distributions:
        if args.dask_client:
            generate_distributions(df)
        else:
            generate_distributions()

    if args.per_country_dist:
        if args.dask_client:
            generate_per_country_distributions(df)
        else:
            generate_per_country_distributions()


if __name__ == "__main__":

    main()
