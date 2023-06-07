from pathlib import Path

import pandas as pd

DATA_DIR = Path("../gpt_di/entity_resolution")

BASELINE_DIR = DATA_DIR / 'baselines'

SIZES = ['small', 'medium', 'large', 'xlarge']

dfs = []

for statfile in DATA_DIR.rglob("stats.csv"):
    df = pd.read_csv(statfile)
    parent = statfile.parent.relative_to(DATA_DIR)
    df['Experiment'] = str(parent)
    df['Model'] = 'ChatGPT'
    df['Dataset Size'] = None
    dfs.append(df)

full_df = pd.concat(dfs)

for baseline in BASELINE_DIR.glob("*.csv"):
    df = pd.read_csv(baseline)
    dataset = baseline.stem
    for col in df.columns:
        if not col.startswith('pred_'):
            continue
        if 'ditto' in col:
            model = 'ditto'
        else:
            model = 'ada'
        dataset_size = None
        for size in SIZES:
            if size in col:
                dataset_size = size
        if dataset != 'cameras' and 'cameras' in col:
            method = 'cameras-trained'
        else:
            method = 'normal'
        
        tps = len(df[df[col] & df['match']])
        fps = len(df[df[col] & ~df['match']])
        fns = len(df[~df[col] & df['match']])

        precision = tps / (tps + fps)
        recall = tps / (tps + fns)
        f1 = 2 * (precision * recall) / (precision + recall)

        row = {
            'Dataset': baseline.stem, 'Method': method, 'Temp': 0.0, 'Crowd': False,
            'F1': f1, 'Precision': precision, 'Recall': recall,
            'Experiment': col[5:], 'Model': model, 'Dataset Size': dataset_size,
        }
        full_df = full_df.append(row, ignore_index=True)

full_df.to_csv(DATA_DIR / 'full.csv', index=False)

