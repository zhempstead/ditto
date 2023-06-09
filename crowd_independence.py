import argparse

from pathlib import Path

from crowdkit.aggregation import MajorityVote, Wawa, DawidSkene
import pandas as pd
from scipy import stats
from sklearn import metrics

CROWD_METHODS = {'DawidSkene': DawidSkene(n_iter=10), 'MajorityVote': MajorityVote(), 'Wawa': Wawa()}

def story_pivot(df):
    return df.pivot(columns='Story Name', index=['task', 'Match File', 'Row No', 'Ground Truth'], values='Story Answer').reset_index()

def independence_test(df, col1, col2):
    res = stats.chi2_contingency(pd.crosstab(df[col1], df[col2]))
    return res[1]

def print_conditional_independences(df):
    crowd_members = df['Story Name'].unique()
    df = story_pivot(df)
    dfs = {True: df[df['Ground Truth'] == 1], False: df[df['Ground Truth'] == 0]}
    for i in range(len(crowd_members)):
        m1 = crowd_members[i]
        for j in range(i+1, len(crowd_members)):
            m2 = crowd_members[j]
            for truth in True, False:
                df_test = dfs[truth]
                print(f"{m1} {m2} {truth}")
                print(independence_test(df_test, m1, m2))

def print_correlations(df):
    df = story_pivot(df)
    dfs = {True: df[df['Ground Truth'] == 1], False: df[df['Ground Truth'] == 0]}
    print("True:")
    print(dfs[True].corr())
    print("False:")
    print(dfs[False].corr())

def print_overall_f1s(df_pivot, crowd_members):
    f1s = []
    for col in crowd_members:
        f1s.append((metrics.f1_score(df_pivot['Ground Truth'], df_pivot[col]), col))
    for metric, col in sorted(f1s, reverse=True):
        print(col, metric)
    return [(col, metric) for metric, col in sorted(f1s, reverse=True)]


def print_dataset_f1s(df_pivot, crowd_members):
    for dataset in df_pivot['Match File'].unique():
        print()
        print(dataset)
        print('---')
        print_overall_f1s(df_pivot[df_pivot['Match File'] == dataset], crowd_members)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root")
    parser.add_argument("--experiment", default="crowd-temp0-shots2-results")
    parser.add_argument("--temp", default='0.0')
    args = parser.parse_args()
    exp_dir = Path(args.results_root) / 'entity_resolution' / args.experiment
    temp_str = args.temp.replace('.', '_')

    df = pd.read_csv(exp_dir / 'full.csv')
    df['task'] = df['Match File'] + ': ' + df['Row No'].astype(str)
    prompts = df['Story Name'].unique()
    df.loc[df['Story Answer'] == -1, 'Story Answer'] = 0
    # Boolean to int
    df['Ground Truth'] = df['Ground Truth'] * 1
    print_conditional_independences(df)
    print_correlations(df)

    df_pivot = story_pivot(df)
    #for method in CROWD_METHODS.keys():
    #    crowd_df = pd.read_csv(exp_dir / f'{method}_results-temperature{temp_str}.csv')
    #    crowd_df = crowd_df.rename(columns={'Vote': method})
    #    crowd_df = crowd_df[['Match File', 'Row No', method]]
    #    df_pivot = df_pivot.merge(crowd_df, on=['Match File', 'Row No'])
    #print_overall_f1s(df_pivot, ['baseline', 'layperson'])
    #print_dataset_f1s(df_pivot, ['baseline', 'layperson'])
    f1_ordered = print_overall_f1s(df_pivot, prompts)
    dfc = df.copy()
    dfc = dfc.rename(columns={"Story Name": "worker", "Story Answer": "label"})
    #f1_ordered.reverse()
    for col, f1 in f1_ordered:
        print()
        print(f"{col}: {f1}")
        print("---")
        for method, mfunc in CROWD_METHODS.items():
            mseries = mfunc.fit_predict(dfc)
            joined = df_pivot.merge(mseries, on='task')
            f1 = metrics.f1_score(joined['Ground Truth'], joined['agg_label'])
            print(f"{method}: {f1}")
        dfc = dfc[dfc['worker'] != col]
            

    #print_dataset_f1s(df_pivot, prompts)
    import pdb; pdb.set_trace()
