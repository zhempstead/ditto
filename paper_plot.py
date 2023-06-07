from matplotlib import pyplot
import numpy as np
import pandas as pd

CROWD_METHOD = 'DawidSkene'

def plot_grouped(df, title):
    datasets = df['Dataset'].unique()
    fig, ax = pyplot.subplots(layout='constrained')

    x = np.arange(len(datasets))
    rows_per_dataset = len(df) / len(datasets)
    width = 1 / (rows_per_dataset + 1)
    for idx, (_, rows) in enumerate(df.groupby('Order')):
        label = rows.iloc[0]['Label']
        print(idx, label)
        print(rows)
        rects = ax.bar(x + idx*width, rows['F1']*100, width, label=label)
        ax.bar_label(rects, padding=3, rotation='vertical') 
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1 score')
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_title(title)
    ax.set_xticks(x + rows_per_dataset*width/2, datasets)
    ax.yaxis.set_minor_locator(pyplot.FixedLocator(np.arange(50, 100, 5)))
    ax.yaxis.grid(True, which='both')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)
    ax.set_ylim(50, 100)

    return fig

def add_labels(df):
    label_orders = df.apply(lambda row: get_label(row), axis=1)
    df['Label'] = [lo[0] for lo in label_orders]
    df['Order'] = [lo[1] for lo in label_orders]
    return df

def get_label(row):
    order = 0
    if row['Dataset Size'] == 'medium':
        order += 0.1
    elif row['Dataset Size'] == 'large':
        order += 0.2
    elif row['Dataset Size'] == 'xlarge':
        order += 0.3
    if row['Method'] == 'cameras-trained':
        order += 0.5

    if row['Experiment'] == 'crowd-temp0-shots0-results':
        if row['Method'] == CROWD_METHOD:
            return (f'ChatGPT Crowd', 4)
        elif row['Method'] == 'baseline':
            return (f'ChatGPT', 2)
    elif row['Experiment'] == 'crowd-temp2-shots0-results':
        if row['Method'] == CROWD_METHOD:
            return (f'ChatGPT Crowd + Reps', 5)
        elif row['Method'] == 'baseline':
            return (f'ChatGPT + Reps', 3)
    elif row['Experiment'] == 'crowd-temp0-shots2-results':
        if row['Method'] == CROWD_METHOD:
            return (f'ChatGPT Crowd + 2-Shot', 8)
        elif row['Method'] == 'baseline':
            return (f'ChatGPT + 2-Shot', 6)
    elif row['Experiment'] == 'crowd-temp2-shots2-results':
        if row['Method'] == CROWD_METHOD:
            return (f'ChatGPT Crowd + Reps + 2-Shot', 9)
        elif row['Method'] == 'baseline':
            return (f'ChatGPT + Reps + 2-Shot', 7)
    elif row['Experiment'].startswith('ditto'):
        return (f'Ditto ({row["Dataset Size"]})', order)
    elif row['Experiment'].startswith('finetune'):
        return (f'Finetuned Ada ({row["Dataset Size"]})', 1 + order)
    return (None, 10)

        
if __name__ == '__main__':
    df = pd.read_csv('../gpt_di/entity_resolution/full.csv')
    df = add_labels(df)
    df = df.sort_values(['Dataset', 'Order', 'Method'])

    df = df[df['Dataset'].isin(['cameras', 'computers', 'shoes', 'watches'])]

    df_1 = df[df['Order'].isin([2, 3, 4, 5, 6, 7, 8, 9])]
    fig = plot_grouped(df_1, 'Plot 1')
    fig.savefig('plot_1.png')

    df_2 = df[(df['Dataset'] != 'cameras') & (df['Order'].isin([0.5, 0.6, 0.7, 0.8, 1.5, 1.6, 1.7, 1.8, 4]))]
    fig = plot_grouped(df_2, 'Plot 2')
    fig.savefig('plot_2.png')

    df_3 = df[df['Order'].isin([0.3, 1.2, 6, 8])]
    fig = plot_grouped(df_3, 'Plot 3')
    fig.savefig('plot_3.png')

    # Display info about variation in shot helpfulness
    df_fixed = df[df['Experiment'].str.contains('fixed_shots')]
    baseline_zeroshot = df[(df['Dataset'] == 'computers') & (df['Experiment'] == 'crowd-temp0-shots0-results')]
    for method in ['baseline', 'MajorityVote', 'Wawa', CROWD_METHOD]:
        bzs = baseline_zeroshot[baseline_zeroshot['Method'] == method].iloc[0]['F1']
        grouped = df_fixed[df_fixed['Method'] == method]['F1']
        print(f"Method {method}:")
        print(f"- zero-shot F1: {bzs}")
        print(f"- minimum few-shot F1: {grouped.min()}")
        print(f"- maximum few-shot F1: {grouped.max()}")
        print(f"- mean few-shot F1: {grouped.mean()}")
        print(f"- median few-shot F1: {grouped.median()}")


