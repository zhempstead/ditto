import math

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd

CROWD_METHOD = 'DawidSkene'

plt.style.use('tableau-colorblind10')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

SUBPLOT_LABEL_SIZE = 16


def plot_grouped(ax, df, groups, datasets, min=50, group_labels=None, add_legend=False):
    colors, patterns, legend = color_pattern_legend(groups.keys())

    x = np.arange(len(datasets))
    bars_per_dataset = len(groups)
    width = 1 / (bars_per_dataset + 1)
    for idx, (label, row_filter) in enumerate(groups.items()):
        df_group = df[row_filter]
        f1s = [df_group.loc[df_group['dataset'] == d, 'F1'] for d in datasets]
        y = [f1.mean() for f1 in f1s]
        for midx, f1 in enumerate(f1s):
            if len(f1) == 0:
                ax.text(x[midx] + idx*width - width/3, min + 0.5, 'x', color='red')
        print(f"{label}: {sum(y) / len(y)}")
        rects = ax.bar(x + idx*width, y, width,
            color=colors[label], hatch=patterns[label], edgecolor='black')
        if group_labels:
            include_idxs = group_labels[idx]
            labels = [None] * len(y)
            for label_idx in range(len(labels)):
                if label_idx not in include_idxs:
                    labels[label_idx] = ' '

        else:
            labels = None
        ax.bar_label(rects, labels=labels, padding=3, rotation='vertical')
    print("-----")
    # Add some text for labels, title and custom x-axis tick labels, etc.
    dataset_labels = datasets2labels(datasets)
    ax.set_ylabel('F1 score')
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xticks(x + (bars_per_dataset - 1) * width / 2, dataset_labels)
    #ax.yaxis.set_minor_locator(plt.FixedLocator(np.arange(50, 100, 5)))
    #ax.yaxis.grid(True, which='both')
    if add_legend:
        ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.45, -0.15),
              fancybox=True, shadow=True, ncol=2)
    ax.set_ylim(min, 120)

COLOR_LEGEND = {
    'Basic': 0,
    'Basic + Few-shot': 1,
    'Basic + Uniform Few-shot': 5,
    'Crowd': 2,
    'ChatGPT Crowd': 2,
    'Crowd + Few-shot': 3,
    'ChatGPT Crowd + Few-shot': 3,
    'Finetuned Ada': 4,
    #'Finetuned Ada (full)': 4,
    #'Finetuned Ada (large)': 4,
    #'Finetuned Ada (small -> xlarge)': 4,
    'Ditto': 5,
    #'Ditto (full)': 5,
    #'Ditto (xlarge)': 5,
    #'Ditto (small -> xlarge)': 5,
}

PATTERN_LEGEND = {
    'Multi-rep': '//',
    'CoT': 'xx',
    'cameras': '\\\\'
}

def color_pattern_legend(group_labels):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']

    colors = {}
    patterns = {}
    color_legend = []
    pattern_legend = []
    existing_legend = set()

    for label in group_labels:
        color_legend_label = None
        color = None
        pattern_legend_label = None
        pattern = None
        for ll, c in COLOR_LEGEND.items():
            if label.startswith(ll):
                color_legend_label = ll
                color = c
        if color is None:
            import pdb; pdb.set_trace()
        for ll, p in PATTERN_LEGEND.items():
            if ll in label:
                pattern_legend_label = ll
                pattern = p
        if pattern_legend_label == 'cameras':
            pattern_legend_label = "'cameras' training set"
        if color_legend_label not in existing_legend:
            color_legend.append(Patch(facecolor=color_cycle[color], label=color_legend_label))
            existing_legend.add(color_legend_label)
        if pattern is not None and pattern_legend_label not in existing_legend:
            pattern_legend.append(Patch(
                facecolor='w',
                edgecolor='black',
                hatch=pattern,
                label=pattern_legend_label))
            existing_legend.add(pattern_legend_label)
        colors[label] = color
        patterns[label] = pattern

    colors = {label: color_cycle[color] for label, color in colors.items()}
    return colors, patterns, color_legend + pattern_legend


def plot_worker_methods(ax, df, worker_groups, crowd_groups, datasets, add_legend=False):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    df = df[df['dataset'].isin(datasets)]

    num_groups = 2
    num_bars = len(worker_groups) + len(crowd_groups)

    width = 1 / (num_bars + num_groups - 1)

    idx = 0
    for worker, row_filter in worker_groups.items():
        f1 = df.loc[row_filter, 'F1'].mean()
        f1_min = df.loc[row_filter, 'F1'].min()
        f1_max = df.loc[row_filter, 'F1'].max()
        rects = ax.bar([idx*width], [f1], width, label=None, yerr=[[f1 - f1_min], [f1_max - f1]], color=colors[0], edgecolor='black', capsize=5)
        ax.bar_label(rects, padding=3, rotation='vertical')
        idx += 1
    idx += 1

    for m_idx, (method, row_filter) in enumerate(crowd_groups.items()):
        f1 = df.loc[row_filter, 'F1'].mean()
        f1_min = df.loc[row_filter, 'F1'].min()
        f1_max = df.loc[row_filter, 'F1'].max()
        capsize = 5
        if f1_max == f1_min:
            capsize = 0
        rects = ax.bar([idx*width], [f1], width, label=method, yerr=[[f1 - f1_min], [f1_max - f1]], color=colors[1 + m_idx], edgecolor='black', capsize=capsize)
        ax.bar_label(rects, padding=3, rotation='vertical')
        idx += 1
    idx += 1

    labels = ['Prompt templates', 'Crowd truth inf. methods']
    ax.set_xticks([
        (len(worker_groups) / 2) * width,
        (len(worker_groups) + 1 + len(crowd_groups) / 2) * width,
    ], labels)
    ax.set_ylabel('F1 score')
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_ylim(0, 130)
    if add_legend:
        ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=2)


def datasets2labels(datasets):
    labels = []
    for dataset in datasets:
        if 'seen' in dataset:
            label = f'Products\n({dataset})'
            if dataset == 'unseen' and 'seen' not in datasets and 'half-seen' not in datasets:
                label = 'Products'
        else:
            label = f'LSPM\n{dataset}'
        labels.append(label)
    return labels

if __name__ == '__main__':
    df = pd.read_csv('../gpt_di/entity_resolution/full.csv')

    df.loc[df['dataset'] == 'wdc_seen', 'dataset'] = 'seen'
    df.loc[df['dataset'] == 'wdc_half', 'dataset'] = 'half-seen'
    df.loc[df['dataset'] == 'wdc_unseen', 'dataset'] = 'unseen'

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained', sharey=True, figsize=[19.2, 4.8], gridspec_kw={'width_ratios': [2, 3, 1]})

    groups1 = {
        'Basic': (df['experiment'] == 'temp0-shots0') & (df['method'] == 'baseline'),
        'Basic + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == 'baseline'),
        'Crowd': (df['experiment'] == 'temp0-shots0') & (df['method'] == CROWD_METHOD),
        'Crowd + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == CROWD_METHOD),
    }
    fig_a, ax = plt.subplots(layout='constrained')
    plot_grouped(ax, df, groups1, ['unseen', 'cameras', 'computers', 'shoes', 'watches'], min=30, add_legend=True)
    fig_a.savefig('plots/1.1a.pdf')

    plot_grouped(ax1, df, groups1, ['unseen', 'cameras', 'computers', 'shoes', 'watches'], min=30)

    groups2 = {
        'Basic': (df['experiment'] == 'temp0-shots0') & (df['method'] == 'baseline'),
        'Basic + Multi-rep': (df['experiment'] == 'temp2-shots0') & (df['method'] == 'baseline'),
        'Basic + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == 'baseline'),
        'Basic + Few-shot + Multi-rep': (df['experiment'] == 'temp2-shots2') & (df['method'] == 'baseline'),
        'Crowd': (df['experiment'] == 'temp0-shots0') & (df['method'] == CROWD_METHOD),
        'Crowd + Multi-rep': (df['experiment'] == 'temp2-shots0') & (df['method'] == CROWD_METHOD),
        'Crowd + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == CROWD_METHOD),
        'Crowd + Few-shot + Multi-rep': (df['experiment'] == 'temp2-shots2') & (df['method'] == CROWD_METHOD),
    }
    fig_b, ax = plt.subplots(layout='constrained')
    plot_grouped(ax, df, groups2, ['cameras', 'computers', 'shoes', 'watches'], add_legend=True)
    fig_b.savefig('plots/1.1b.pdf')
    plot_grouped(ax2, df, groups2, ['cameras', 'computers', 'shoes', 'watches'], min=30)

    groups3 = {
        'Basic': (df['experiment'] == 'temp0-shots0') & (df['method'] == 'baseline'),
        'Basic + CoT': (df['experiment'] == 'temp0-cot0') & (df['method'] == 'baseline'),
        'Basic + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == 'baseline'),
        'Basic + Uniform Few-shot + CoT': (df['experiment'] == 'temp0-cot2') & (df['method'] == 'baseline'),
    }
    fig_c, ax = plt.subplots(layout='constrained')
    plot_grouped(ax, df, groups3, ['unseen'], min=30, add_legend=True)
    fig_c.savefig('plots/1.1c.pdf')
    plot_grouped(ax3, df, groups3, ['unseen'], min=30)
    _, _, legend = color_pattern_legend({**groups1, **groups2, **groups3})
    ax2.set_ylabel(None)
    ax3.set_ylabel(None)
    ax1.set_title('(a)', y=-0.35, size=SUBPLOT_LABEL_SIZE)
    ax2.set_title('(b)', y=-0.35, size=SUBPLOT_LABEL_SIZE)
    ax3.set_title('(c)', y=-0.35, size=SUBPLOT_LABEL_SIZE)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, 0.045), fancybox=True, shadow=True, ncol=len(legend))
    fig.savefig('plots/1.1.pdf')

    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained', sharey=True, figsize=[12.8, 4.8], gridspec_kw={'width_ratios': [5, 4]})

    worker_groups = {
        worker: (df['experiment'] == 'temp0-shots2') & (df['method'] == worker)
        for worker in ['layperson', 'veryplain', 'baseline', 'security', 'plain', 'journalist', 'customer', 'detective']
    }
    crowd_groups = {
        method: (df['experiment'] == 'temp0-shots2') & (df['method'] == method)
        for method in ['MajorityVote', 'DawidSkene', 'GLAD', 'EBCC', 'BWA', 'GoldStandard']
    }
    fig_a, ax = plt.subplots(layout='constrained')
    plot_worker_methods(ax, df, worker_groups, crowd_groups, ['unseen', 'cameras', 'computers', 'shoes', 'watches'], add_legend=True)
    fig_a.savefig('plots/1.2a.pdf')
    plot_worker_methods(ax1, df, worker_groups, crowd_groups, ['unseen', 'cameras', 'computers', 'shoes', 'watches'])

    worker_groups = {
        'Basic + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == 'baseline'),
        'Basic': (df['experiment'] == 'temp0-shots0') & (df['method'] == 'baseline'),
        'Basic + Uniform Few-shot + CoT': (df['experiment'] == 'temp0-cot2') & (df['method'] == 'baseline'),
        'Basic + CoT': (df['experiment'] == 'temp0-cot0') & (df['method'] == 'baseline'),
    }
    crowd_groups = {
        method: (df['experiment'] == 'cot_crowd') & (df['method'] == method)
        for method in ['MajorityVote', 'DawidSkene', 'GLAD', 'EBCC', 'BWA', 'GoldStandard']
    }
    fig_b, ax = plt.subplots(layout='constrained')
    plot_worker_methods(ax, df, worker_groups, crowd_groups, ['unseen'], add_legend=True)
    fig_b.savefig('plots/1.2b.pdf')
    plot_worker_methods(ax2, df, worker_groups, crowd_groups, ['unseen'])
    h, l = ax1.get_legend_handles_labels(legend_handler_map=None)
    ax2.set_ylabel(None)
    ax1.set_title('(a)', y=-0.28, size=SUBPLOT_LABEL_SIZE)
    ax2.set_title('(b)', y=-0.28, size=SUBPLOT_LABEL_SIZE)
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.06), fancybox=True, shadow=True, ncol=len(l))
    fig.savefig('plots/1.3.pdf')
    
    groups = {
        'Basic': (df['experiment'] == 'temp0-shots0') & (df['method'] == 'baseline'),
        'Basic + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == 'baseline'),
        'Basic + Few-shot (cameras)': (df['experiment'] == 'temp0-shots2c') & (df['method'] == 'baseline'),
        'Crowd': (df['experiment'] == 'temp0-shots0') & (df['method'] == CROWD_METHOD),
        'Crowd + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == CROWD_METHOD),
        'Crowd + Few-shot (cameras)': (df['experiment'] == 'temp0-shots2c') & (df['method'] == CROWD_METHOD),
    }
    fig, ax = plt.subplots(layout='constrained')
    plot_grouped(ax, df, groups, ['computers', 'shoes', 'watches'], add_legend=True)
    fig.subplots_adjust(right=0.95, top=0.95, bottom=0.33)
    fig.savefig('plots/1.4.pdf')

    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, layout='constrained', sharey=True, figsize=[12.8, 9.6])
    fig, (ax1, ax4) = plt.subplots(1, 2, layout='constrained', sharey=True, figsize=[12.8, 5.4])


    ditto = { f'Ditto ({size}) (cameras)': (df['model'] == 'ditto') & \
        (df['size'] == size) & \
        (df['train_set'] == 'cameras')
        for size in ['xlarge']
    }
    ada = { f'Finetuned Ada ({size}) (cameras)': (df['model'] == 'ada') & \
        (df['size'] == size) & \
        (df['train_set'] == 'cameras')
        for size in ['large'] }
    chatgpt = {
        'ChatGPT Crowd': (df['experiment'] == 'temp0-shots0') & (df['method'] == CROWD_METHOD),
        'ChatGPT Crowd + Few-shot (cameras)': (df['experiment'] == 'temp0-shots2c') & (df['method'] == CROWD_METHOD),
    }
    groups1 = {**ditto, **ada, **chatgpt}
    fig_a, ax = plt.subplots(layout='constrained')
    plot_grouped(ax, df, groups1, ['computers', 'shoes', 'watches'], min=40, add_legend=True)
    fig_a.savefig('plots/2a.pdf')
    plot_grouped(ax1, df, groups1, ['computers', 'shoes', 'watches'], min=40)

    ditto = { f'Ditto (small -> xlarge) ({size})': (df['model'] == 'ditto') & \
        (df['size'] == size) & \
        df['train_set'].isna() 
        for size in ['small', 'medium', 'large', 'xlarge']
    }
    ada = { f'Finetuned Ada (small -> xlarge) ({size})': (df['model'] == 'ada') & \
        (df['size'] == size) & \
        df['train_set'].isna() 
        for size in ['small', 'medium', 'large', 'xlarge'] }
    chatgpt = {
        'ChatGPT Crowd': (df['experiment'] == 'temp0-shots0') & (df['method'] == CROWD_METHOD),
        'ChatGPT Crowd + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == 'baseline'),
    }
    group_labels = [
        [],
        [],
        [0, 4],
        [1,2,3],
        [0],
        [1],
        [],
        [2,3,4],
        [],
        [0,1,2,3,4],
    ]
    groups2 = {**ditto, **ada, **chatgpt}
    #plot_grouped(ax2, df, groups2, ['unseen', 'cameras', 'computers', 'shoes', 'watches'], min=40, group_labels=group_labels) 


    groups3 = {
        'Ditto (full)': (df['model'] == 'ditto') & \
            df['train_set'].isna() & \
            ((df['size'] == 'xlarge') | ((df['size'] == 'large') & df['dataset'].str.contains('seen'))),
        'Finetuned Ada (full)': (df['model'] == 'ada') & \
            df['train_set'].isna() & \
            ((df['size'] == 'xlarge') | ((df['size'] == 'large') & df['dataset'].str.contains('seen'))),
        'ChatGPT Crowd + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == 'baseline'),
    }
    #plot_grouped(ax3, df, groups3, ['unseen', 'cameras', 'computers', 'shoes', 'watches'], min=40)


    groups4 = {
        'Ditto (full)': (df['model'] == 'ditto') & (df['size'] == 'large'),
        'Finetuned Ada (full)': (df['model'] == 'ada') & (df['size'] == 'large'),
        'ChatGPT Crowd': (df['experiment'] == 'temp0-shots0') & (df['method'] == CROWD_METHOD),
        'ChatGPT Crowd + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == CROWD_METHOD),
    }
    fig_b, ax = plt.subplots(layout='constrained')
    plot_grouped(ax, df, groups4, ['seen', 'half-seen', 'unseen'], min=40, add_legend=True)
    fig_b.savefig('plots/2b.pdf')
    plot_grouped(ax4, df, groups4, ['seen', 'half-seen', 'unseen'], min=40)

    #_, _, legend = color_pattern_legend({**groups1, **groups2, **groups3, **groups4})
    _, _, legend = color_pattern_legend({**groups1, **groups4})
    #ax2.set_ylabel(None)
    ax4.set_ylabel(None)
    ax1.set_title('(a)', y=-0.45, size=SUBPLOT_LABEL_SIZE)
    #ax2.set_title('(b)', y=-0.3, size=SUBPLOT_LABEL_SIZE)
    #ax3.set_title('(c)', y=-0.3, size=SUBPLOT_LABEL_SIZE)
    ax4.set_title('(b)', y=-0.45, size=SUBPLOT_LABEL_SIZE)
    #ax4.set_title('(d)', y=-0.3, size=SUBPLOT_LABEL_SIZE)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, 0.07), fancybox=True, shadow=True, ncol=3)
    fig.subplots_adjust(top=0.95, bottom=0.3, left=0.1, right=0.95, wspace=0.05)
    fig.savefig('plots/2.pdf')

    fixed_results = df[df['experiment'].str.startswith('fixed_shots') & (df['method'] == CROWD_METHOD)]
    print(f"Fixed results mean: {fixed_results['F1'].mean()}")
    print(f"Population standard deviation: {fixed_results['F1'].std()}")

    df_table = df[~df['crowd'] & df['experiment'].isin(['temp0-shots0', 'temp0-shots2']) & df['dataset'].isin(['unseen', 'cameras', 'computers', 'shoes', 'watches'])]
    df_table = df_table[['dataset', 'experiment', 'method', 'F1']]
    df_table.loc[df_table['dataset'] == 'unseen', 'dataset'] = 'Products'
    df_table.loc[df_table['experiment'] == 'temp0-shots0', 'experiment'] = '0-shot'
    df_table.loc[df_table['experiment'] == 'temp0-shots2', 'experiment'] = '2-shot'
    df_table['Rank'] = df_table.groupby(['dataset', 'experiment'])['F1'].rank(method="dense", ascending=False).astype(int)
    df_table = df_table.loc[df_table['Rank'] <= 3, ['dataset', 'Rank', 'experiment', 'method']]
    df_table = df_table.sort_values(['dataset', 'Rank', 'experiment'])
    print(df_table.pivot(columns='Rank', index=['dataset', 'experiment'], values='method'))


    workers = ['layperson', 'veryplain', 'baseline', 'security', 'plain', 'journalist', 'customer', 'detective']
    '''
    group1 = {
        f'{worker}: 0-shot': (df['experiment'] == 'temp0-shots0') & (df['method'] == worker) for worker in workers
    }
    group2 = {
        f'{worker}: 2-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == worker) for worker in workers
    }
    group3 = {
        'ChatGPT Crowd': (df['experiment'] == 'temp0-shots0') & (df['method'] == CROWD_METHOD),
        'ChatGPT Crowd + Few-shot': (df['experiment'] == 'temp0-shots2') & (df['method'] == CROWD_METHOD),
    }
    plot_grouped(ax, df, {**group1, **group3}, ['seen', 'half-seen', 'unseen'], min=0)
    fig.savefig('plots/explore.pdf')

    plot_grouped(ax, df, {**group2, **group3}, ['seen', 'half-seen', 'unseen'], min=0)
    fig.savefig('plots/explore2.pdf')
    '''
