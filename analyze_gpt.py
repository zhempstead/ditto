import numpy as np
import pandas as pd
import sklearn
import yaml
import argparse
from pathlib import Path
from crowdkit.aggregation import MajorityVote, Wawa, DawidSkene, GLAD
from ebcc import ebcc_vb
from bwa import bwa

def main(args):
    with open(args.config) as cf:
        conf = yaml.safe_load(cf)

    exp_dir = Path(conf['experiment_dir'])

    print("Gathering baselines...")
    baseline_f1s = gather_baselines(exp_dir / 'baselines', conf['datasets'].keys())

    print("Gathering experiments...")
    df, truth = gather_experiments(exp_dir, conf['experiments'].keys(), conf['datasets'].keys())

    print("Worker F1s...")
    worker_f1s = calculate_worker_f1s(df, truth)

    print("Basic crowds...")
    basic_crowds = {exp: {exp: []} for exp, default in conf['experiments'].items() if default}
    basic_crowd_f1s = calculate_crowd_f1s(df, truth, basic_crowds, conf['default_methods'], conf['default_workers'])

    print("Custom crowds...")
    custom_crowd_f1s = calculate_crowd_f1s(df, truth, conf['crowds'], conf['default_methods'], conf['default_workers'])

    crowd_f1s = pd.concat([basic_crowd_f1s.reset_index(), custom_crowd_f1s.reset_index()])
    worker_f1s = worker_f1s.reset_index()
    crowd_f1s = crowd_f1s.rename(columns={'crowd': 'experiment'})
    worker_f1s = worker_f1s.rename(columns={'worker': 'method'})
    crowd_f1s['crowd'] = True
    worker_f1s['crowd'] = False
    crowd_f1s['model'] = 'ChatGPT'
    worker_f1s['model'] = 'ChatGPT'

    f1s = pd.concat([baseline_f1s, worker_f1s, crowd_f1s])
    f1s.to_csv(exp_dir / 'full.csv', index=False)
    


def gather_experiments(experiment_dir, experiments, datasets):
    dfs = []
    exp_dirs = [experiment_dir / exp for exp in experiments]
    for exp in experiments:
        df = pd.read_csv(experiment_dir / exp / 'full.csv', usecols=['Match File', 'Row No', 'Rep No', 'Story Name', 'Story Answer', 'Ground Truth'])
        df['experiment'] = exp
        dfs.append(df)
    df = pd.concat(dfs)
    df['dataset'] = df['Match File'].str.split('/').str[-1].str.split('.').str[0]
    df['worker'] = df['Story Name']
    df['task'] = df['Row No']
    df['rep'] = df['Rep No']
    df['label'] = (df['Story Answer'] == 1) * 1
    df['truth'] = df['Ground Truth']
    df = df[df['dataset'].isin(datasets)]
    truth = df[['dataset', 'task', 'truth']].drop_duplicates()
    df = df.set_index(['experiment', 'dataset', 'worker', 'task', 'rep'])
    df = df[['label']]
    truth = truth.set_index(['dataset', 'task'])
    return df, truth

def calculate_worker_f1s(df, truth):
    df = df.groupby(level=['experiment', 'dataset', 'worker', 'task']).mean() > 0.5

    out = df.reset_index()[['experiment', 'dataset', 'worker']].drop_duplicates()
    out['F1'] = -1.0
    out = out.set_index(['experiment', 'dataset', 'worker'])

    df = df.join(truth)

    for (experiment, dataset, worker), df_group in df.groupby(level=['experiment', 'dataset', 'worker']):
        f1 = sklearn.metrics.f1_score(df_group['truth'], df_group['label']) * 100
        out.loc[(experiment, dataset, worker)] = f1

    if out['F1'].min() < 0.0:
        raise ValueError("Missing value in 'out'")
    return out

def calculate_crowd_f1s(df, truth, crowds, methods, default_workers):
    df = df.reorder_levels(['experiment', 'worker', 'dataset', 'task', 'rep'])
    df['exp_worker'] = df.index.get_level_values('experiment') + '_' + df.index.get_level_values('worker')

    out = {'crowd': [], 'dataset': [], 'method': [], 'F1': []}

    for crowd, experiments in crowds.items():
        print(f'  crowd {crowd}...')
        df_crowd = df.loc[experiments.keys()]
        exp_workers = []
        for exp, workers in experiments.items():
            if not workers:
                workers = default_workers
            exp_workers += [f'{exp}_{worker}' for worker in workers]
        df_crowd = df_crowd[df_crowd['exp_worker'].isin(exp_workers)]
        df_crowd = df_crowd.reset_index()[['dataset', 'task', 'exp_worker', 'label']]
        df_crowd = df_crowd.rename(columns={'exp_worker': 'worker'})
        for dataset, df_crowd_dataset in df_crowd.groupby('dataset'):
            print(f'    dataset {dataset}...')
            truth_dataset = truth.loc[dataset]
            for method in methods:
                args = [df_crowd_dataset]
                if method == 'GoldStandard':
                    args.append(truth_dataset)
                prediction = CROWD_METHODS[method]().fit_predict(*args)
                prediction = truth_dataset.join(prediction)
                f1 = sklearn.metrics.f1_score(prediction['truth'], prediction['agg_label']) * 100
                out['crowd'].append(crowd)
                out['dataset'].append(dataset)
                out['method'].append(method)
                out['F1'].append(f1)

    out = pd.DataFrame(out)
    out = out.set_index(['crowd', 'dataset', 'method'])
    return out

def gather_baselines(baseline_dir, datasets):
    out = {'model': [], 'dataset': [], 'size': [], 'train_set': [], 'F1': []}
    for dataset in datasets:
        df = pd.read_csv(baseline_dir / f'{dataset}.csv')
        for col in df.columns:
            if not col.startswith('pred_'):
                continue
            if 'ditto' in col:
                model = 'ditto'
            else:
                model = 'ada'
            dataset_size = None
            for size in ['small', 'medium', 'large', 'xlarge']:
                if size in col:
                    dataset_size = size
            train_set = None
            if dataset != 'cameras' and 'cameras' in col:
                train_set = 'cameras'
            
            f1 = sklearn.metrics.f1_score(df['match'], df[col]) * 100
            out['model'].append(model)
            out['dataset'].append(dataset)
            out['size'].append(dataset_size)
            out['train_set'].append(train_set)
            out['F1'].append(f1)
        return pd.DataFrame(out)


class EBCC:
    def fit_predict(self, df):
        df['worker_codes'], _ = pd.factorize(df['worker'])
        elbos = []
        results = []
        for _ in range(40):
            seed = np.random.randint(1e8)
            prediction, elbo = ebcc_vb(df[['task', 'worker_codes', 'label']].values, num_groups=10, seed=seed, empirical_prior=True)
            elbos.append(elbo)
            results.append((prediction, seed, elbo))
        predictions, seed, elbo = results[np.argmax(elbos)]
        out = pd.Series(predictions.argmax(axis=1), name='agg_label')
        out.index.name = 'task'
        return out

class BWA:
    def fit_predict(self, df):
        df['worker_codes'], _ = pd.factorize(df['worker'])
        predictions = bwa(df[['task', 'worker_codes', 'label']].values)
        out = pd.Series(predictions.argmax(axis=1), name='agg_label')
        out.index.name = 'task'
        return out

class GoldStandard:
    def fit_predict(self, df, truth):
        workers = list(df['worker'].unique())
        df = df.groupby(['worker', 'task']).mean().reset_index()
        df_pivot = df.pivot(columns='worker', index='task', values='label').join(truth)
        grouped = df_pivot.groupby(workers).mean().reset_index()
        grouped['agg_label'] = 0
        grouped.loc[grouped['truth'] > 0.5, 'agg_label'] = 1
        predictions = df_pivot.reset_index().merge(grouped[workers + ['agg_label']], on=workers)
        return predictions.set_index('task')['agg_label']

CROWD_METHODS = {'DawidSkene': DawidSkene, 'MajorityVote': MajorityVote, 'Wawa': Wawa, 'GLAD': GLAD, 'EBCC': EBCC, 'BWA': BWA, 'GoldStandard': GoldStandard}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./experiments_conf.yaml")
    args = parser.parse_args()
    main(args)
