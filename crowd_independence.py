import argparse

from pathlib import Path

from crowdkit.aggregation import MajorityVote, Wawa, DawidSkene
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin
from scipy import stats
from sklearn import metrics

plt.style.use('tableau-colorblind10')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5



CROWD_METHODS = {'DawidSkene': DawidSkene(n_iter=10), 'MajorityVote': MajorityVote(), 'Wawa': Wawa()}

def worker_pivot(df):
    return df.pivot(columns='worker', index=['task', 'Match File', 'Row No', 'Ground Truth'], values='label').reset_index()

def independence_test(df, col1, col2):
    res = stats.chi2_contingency(pd.crosstab(df[col1], df[col2]))
    return res[1]

def print_conditional_independences(df):
    crowd_members = df['worker'].unique()
    df = worker_pivot(df)
    dfs = {True: df[df['Ground Truth'] == 1], False: df[df['Ground Truth'] == 0]}
    test_stats = []
    for i in range(len(crowd_members)):
        m1 = crowd_members[i]
        for j in range(i+1, len(crowd_members)):
            
            m2 = crowd_members[j]
            test_stat = 1
            test_stat = min(independence_test(dfs[True], m1, m2), independence_test(dfs[False], m1, m2))
            test_stats.append((test_stat, m1, m2))
    for stat in sorted(test_stats):
        print(stat)

def partial_correlations(df):
    df = worker_pivot(df).drop(columns=['Row No'])
    partial = partial_corr(df, 'Ground Truth')[0]
    fig = plot_correlations(partial)
    return fig
    

def partial_corr(df, covariate_col):
    df_pivot = worker_pivot(df).drop(columns=['Row No'])
    df_dropped = df_pivot.drop(columns=[covariate_col])
    corr_matrix = df_dropped.corr() # To bootstrap a DataFrame with the right rows/columns
    pval_matrix = corr_matrix.copy()
    cols = corr_matrix.columns
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                pval_matrix.loc[col1, col2] = 0
                continue
            partial = df_pivot.partial_corr(x=col1, y=col2, covar=covariate_col)
            corr_matrix.loc[col1, col2] = partial.loc['pearson', 'r']
            pval_matrix.loc[col1, col2] = partial.loc['pearson', 'p-val']
    return (corr_matrix, pval_matrix)

def plot_correlations(df):
    data = df.to_numpy()
    fig, ax = plt.subplots(layout='constrained')

    size = len(df.columns)

    im = ax.imshow(data, cmap='coolwarm', vmin=-1, vmax=1)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Partial correlations")

    ax.set_xticks(np.arange(size), labels=df.columns)
    ax.set_yticks(np.arange(size), labels=df.columns)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(size + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return fig

def plot_lines(df, dataset):
    df['# Removed'] = range(len(df))

    fig, ax = plt.subplots(layout='constrained')

    for col in df.columns[:-1]:
        ax.plot('# Removed', col, data=df, marker='o')
    ax.legend(loc='lower left', fancybox=True, shadow=True, ncol=2)
    ax.set_ylabel('F1')
    ax.set_xlabel("# of workers removed")
    ax.set_title(dataset)
    return fig

def iterative_crowd(df, workers, methods):
    '''
    methods should be a Dict of str to a function that takes in a df and returns a series of predictions indexed by task
    '''
    df = df.copy()
    df_pivot = worker_pivot(df)
    crowd = ordered_crowd(df_pivot, workers)

    out = {method: [] for method in methods.keys()}
    
    for idx, worker in enumerate(crowd):
        for method, func in methods.items():
            predictions = func(df, worker)
            predictions.name = method
            predictions = df_pivot[['task', 'Ground Truth']].merge(predictions, on='task')
            accuracy = sum(predictions['Ground Truth'] == predictions[method]) / len(predictions)
            f1 = metrics.f1_score(predictions['Ground Truth'], predictions[method])
            out[method].append(f1 * 100)
            #out[method].append(accuracy * 100)
        df = df[df['worker'] != worker]
        df_pivot = worker_pivot(df)
    return pd.DataFrame(out)

def method_majority_vote(df, curr_worker):
    return MajorityVote().fit_predict(df)

def method_dawid_skene(df, curr_worker):
    return DawidSkene(n_iter=10).fit_predict(df)

def method_best_remaining(df, curr_worker):
    return df.loc[df['worker'] == curr_worker, ['task', 'label']].set_index('task')['label']

def method_optimal_f1(df, curr_worker):
    workers = list(df['worker'].unique())
    df_pivot = worker_pivot(df)
    grouped = df_pivot.groupby(workers).mean().reset_index()
    levels = grouped.loc[grouped['Ground Truth'] <= 0.5, 'Ground Truth'].unique()
    levels = sorted(levels, reverse=True) + [0.0]
    best_f1 = -1.0
    best_output = None
    for level in levels:
        grouped['agg_label'] = (grouped['Ground Truth'] > level)
        output = df_pivot.merge(grouped[workers + ['agg_label']], on=workers).set_index('task')['agg_label']
        predictions = df_pivot[['task', 'Ground Truth']].merge(output, on='task')
        f1 = metrics.f1_score(predictions['Ground Truth'], predictions['agg_label'])
        if f1 > best_f1:
            best_f1 = f1
            best_output = output
    return best_output
    
def method_optimal_accuracy(df, curr_worker):
    workers = list(df['worker'].unique())
    df_pivot = worker_pivot(df)
    grouped = df_pivot.groupby(workers).mean().reset_index()
    grouped['agg_label'] = 0
    grouped.loc[grouped['Ground Truth'] > 0.5, 'agg_label'] = 1
    return df_pivot.merge(grouped[workers + ['agg_label']], on=workers).set_index('task')['agg_label']

def ordered_crowd(df_pivot, workers):
    f1s = []
    for col in workers:
        f1s.append((metrics.f1_score(df_pivot['Ground Truth'], df_pivot[col]), col))
    return [col for _, col in sorted(f1s, reverse=True)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--temp", default='0.0')
    parser.add_argument("--datasets", nargs='+', default=['wdc_unseen', 'cameras', 'computers', 'shoes', 'watches'])
    args = parser.parse_args()
    exp_dir = Path(args.experiment_dir)
    temp_str = args.temp.replace('.', '_')

    df = pd.read_csv(exp_dir / 'full.csv')
    match_files = [f'er_results/{dataset}.csv' for dataset in args.datasets]
    df = df[df['Match File'].isin(match_files)]
    df['task'] = df['Match File'] + ': ' + df['Row No'].astype(str)
    df = df.rename(columns={"Story Name": "worker", "Story Answer": "label"})
    workers = df['worker'].unique()
    df.loc[df['label'] == -1, 'label'] = 0
    # Boolean to int
    df['Ground Truth'] = df['Ground Truth'] * 1
    corrs = partial_corr(df, 'Ground Truth')[0]
    fig = plot_correlations(corrs)

    fig.savefig(f'corrs/{exp_dir.name}.pdf')
    corrs.to_csv(f'corrs/{exp_dir.name}.csv')

    print_conditional_independences(df)
    
    for dataset in df['Match File'].unique():
        dataset_name = dataset.split('/')[-1].split('.')[0]
        results = iterative_crowd(df[df['Match File'] == dataset], workers, methods={
            'MajorityVote': method_majority_vote,
            'DawidSkene': method_dawid_skene,
            'Best Remaining': method_best_remaining,
            'Optimal Crowd Method': method_optimal_accuracy,
        })
        fig = plot_lines(results, dataset_name)
        fig.savefig(f'corrs/{exp_dir.name}-{dataset_name}.pdf')
