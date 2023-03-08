import pandas as pd
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

def _get_metrics_from_tensorboard(event_acc, scalar):
    train_metrics, val_metrics = event_acc.Scalars(f'{scalar}/train'), event_acc.Scalars(f'{scalar}/valid')
    train_df, val_df = pd.DataFrame(train_metrics), pd.DataFrame(val_metrics)

    train_df.drop(columns=['wall_time'], inplace=True)
    val_df.drop(columns=['wall_time'], inplace=True)

    train_df.rename(columns={'value': f'train_{scalar}'}, inplace=True)
    val_df.rename(columns={'value': f'val_{scalar}'}, inplace=True)
    results = pd.merge(train_df, val_df, on='step', how='outer')
    results = results.dropna()
    results = results.drop(columns=['step'])
    return results

def graph_and_metrics(loss, acc, prec, rec, experiment_name):
    result_keys = {
        'BCE Loss': loss,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    fig.suptitle(f'Loss, Accuracy, Precision, Recall for {experiment_name}', fontsize=16)

    results_matrix = pd.concat([loss, acc, prec, rec], axis=1)
    for i, (key, value) in enumerate(result_keys.items()):
        fig_ = sns.lineplot(value, ax=axes[i])
        fig_.set(xlabel='Epoch', ylabel=key, title=key)
    return results_matrix

def top_metrics(results):
    metric_container = []
    for experiment_name, df in results.items():
        metric_container.append(_best_metrics(df, experiment_name))
    
    return pd.DataFrame(metric_container)

def tensorboard_results(log_dir_list, experiment_name):

    loss_list, acc_list, prec_list, rec_list = [], [], [], []

    for log_dir in log_dir_list:
        print(f'Processing {log_dir}...')
        events = EventAccumulator(log_dir)
        events.Reload()
        
        loss_list.append(_get_metrics_from_tensorboard(events, 'loss_e'))
        acc_list.append(_get_metrics_from_tensorboard(events, 'acc'))
        prec_list.append(_get_metrics_from_tensorboard(events, 'prec'))
        rec_list.append(_get_metrics_from_tensorboard(events, 'rec'))


    for df_list in [loss_list, acc_list, prec_list, rec_list]:
        for indx, df_index in enumerate(df_list):
            if indx != len(df_list) - 1:
                cutoff = df_list[indx + 1].shape[0]
                df_index = df_index.head(100 - cutoff)
                print(df_index.shape)

    loss, acc, prec, rec = (
        pd.concat(loss_list).reset_index().drop(columns=['index']),
        pd.concat(acc_list).reset_index().drop(columns=['index']),
        pd.concat(prec_list).reset_index().drop(columns=['index']),
        pd.concat(rec_list).reset_index().drop(columns=['index'])
    )

    if not os.path.exists('lstm_results/all_files/validation'):
        os.makedirs('lstm_results/all_files/validation')

    exp = pd.concat([loss, acc, prec, rec], axis=1)
    exp.to_csv(f'lstm_results/all_files/validation/{experiment_name}_all.csv')
    
 

    # return graph_and_metrics(loss, acc, prec, rec, experiment_name)

def _best_metrics(df, name):

    # last row from columns
    return {
        'Experiment': name,
        'BCE Loss': df['val_loss_e'].iloc[-1].round(4),
        'Accuracy': df['val_acc'].iloc[-1].round(4),
        'Precision': df['val_prec'].iloc[-1].round(4),
        'Recall': df['val_rec'].iloc[-1].round(4),
    }


GRAPHS = {
    'seq_10': [
        's3://dissertation-data-dmiller/lstm_experiments/results/61/ordinal/sequence_length_10/None/2023_02_07_14_49/version_0'
    ],
    'seq_20': [
        's3://dissertation-data-dmiller/lstm_experiments/results/61/ordinal/sequence_length_20/None/2023_02_08_08_27/version_0'
    ],
    'seq_30': [
        's3://dissertation-data-dmiller/lstm_experiments/results/61/ordinal/sequence_length_30/None/2023_02_07_21_44/version_0'
    ],
    'seq_40': [
        's3://dissertation-data-dmiller/lstm_experiments/results/61/ordinal/sequence_length_40/None/2023_02_08_08_15/version_0',
        's3://dissertation-data-dmiller/lstm_experiments/results/61/ordinal/sequence_length_40/None/2023_02_08_21_22/version_0'
    ],
    'seq_10_heuristic': [
       's3://dissertation-data-dmiller/lstm_experiments/results/61/ordinal/sequence_length_10/None/2023_02_09_11_08/version_0',
    ],
    'seq_20_heuristic': [
        's3://dissertation-data-dmiller/lstm_experiments/results/61/ordinal/sequence_length_20/None/2023_02_09_19_22/version_0'
    ],
    'seq_30_heuristic': [
        's3://dissertation-data-dmiller/lstm_experiments/results/61/ordinal/sequence_length_30/None/2023_02_10_09_39/version_0',
        's3://dissertation-data-dmiller/lstm_experiments/results/61/ordinal/sequence_length_30/None/2023_02_10_16_33/version_0'
    ],
    'seq_40_heuristic': [
        's3://dissertation-data-dmiller/lstm_experiments/results/61/ordinal/sequence_length_40/None/2023_02_10_17_48/version_0',
        's3://dissertation-data-dmiller/lstm_experiments/results/61/ordinal/sequence_length_40/None/2023_02_12_08_49/version_0'
    ]

}

LOSS_GRAPHS = [
    'seq_10', 'seq_20', 'seq_30', 'seq_40'
]

def plot_loss_graphs(graph_name, ax):
    graph, heuristic =(
        pd.read_csv(f'lstm_results/all_files/validation/{graph_name}_all.csv', usecols=['train_loss_e', 'val_loss_e']),
        pd.read_csv(f'lstm_results/all_files/validation/{graph_name}_heuristic_all.csv', usecols=['train_loss_e', 'val_loss_e'])
    )

    graph.columns = ['train_loss', 'val_loss']
    heuristic.columns = ['train_loss_heuristic', 'val_loss_heuristic']
    graph['exp_name'] = graph_name
    heuristic['exp_name'] = graph_name + '_heuristic'
    graph['epoch'] = graph.index
    heuristic['epoch'] = heuristic.index

    # graph

    df = pd.merge(graph, heuristic, on=['epoch'], how='outer')
    print(df.columns)

    fig_ = sns.lineplot(df[[col for col in df.columns if 'loss' in col]], ax=ax)
    fig_.set(xlabel='Epoch', ylabel='BCE Loss', title=f'Training and Validation Losses for {graph_name} and {graph_name}_heuristic')




def loss_plot_container():
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    fig.suptitle('Training and Validation Losses For Sequence Lengths 10, 20, 30, 40', fontsize=16)

    for graph_name, ax in zip(LOSS_GRAPHS, axs):
        plot_loss_graphs(graph_name, ax)

    plt.show()


def main():

    if not os.path.exists('lstm_results/all_files/validation'):
        for exp_name, log_files in GRAPHS.items():
            print(f'Processing {exp_name}...')
            tensorboard_results(log_files, exp_name)
    
    loss_plot_container()


    # loss_plot_container(

if __name__ == '__main__':
    main()