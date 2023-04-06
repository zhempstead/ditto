from pathlib import Path
import sys

import pandas as pd

def handle_start_line(line):
    tokens = line.split(' ')
    try:
        task_pos = tokens.index('--task') + 1
        runid_pos = tokens.index('--run_id') + 1
    except ValueError:
        print(line)
        print(tokens)
        raise
    task = tokens[task_pos]
    runid = int(tokens[runid_pos])
    return task, runid

def handle_epoch_line(line):
    tokens = line.strip().split(' ')
    epoch = int(tokens[1][:-1])
    best_f1, val = tokens[-1].split('=')
    assert best_f1 == 'best_f1'
    return epoch, float(val)

txtdir = Path(sys.argv[1])

results = {'dataset': [], 'index': [], 'epochs': [], 'fscore': []}

curr_dataset = None
curr_index = None
curr_epoch = 0
curr_fscore = None

def add_result():
    print(f"  Results for {curr_dataset} {curr_index}")
    results['dataset'].append(curr_dataset)
    results['index'].append(curr_index)
    results['epochs'].append(curr_epoch)
    results['fscore'].append(curr_fscore)


for txtfile in txtdir.glob('*.txt'):
    print(f"Processing '{txtfile}'...")
    for line in open(txtfile):
        if line == '\n':
            continue
        elif line.startswith("CUDA_VISIBLE_DEVICES="):
            if curr_dataset is not None:
                add_result()
            curr_dataset, curr_index = handle_start_line(line)
            curr_epoch = 0
            curr_fscore = None
        elif line.startswith("epoch "):
            curr_epoch, curr_fscore = handle_epoch_line(line)
    if curr_epoch > 1:
        add_result()
    curr_dataset = None

df = pd.DataFrame(results)
df.to_csv('ditto_results_full.csv', index=False)
df = df[['dataset', 'fscore']]
df.groupby('dataset').mean().reset_index().to_csv('ditto_results_summary.csv', index=False)
