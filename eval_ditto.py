import os
import argparse
import json
import sys
import torch
from torch.utils import data
import numpy as np
from pathlib import Path
import random

import pandas as pd

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import evaluate, DittoModel

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-task", type=str, default="Structured/Beer")
    parser.add_argument("--test-task", type=str, default="Structured/Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--unseen", type=str, default=None)

    hp = parser.parse_args()

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    train_task = hp.train_task
    test_task = hp.test_task

    test_df_name = None
    for dataset in ["cameras", "computers", "shoes", "watches", "Amazon-Google"]:
        if dataset in test_task:
            test_df_name = dataset
    if hp.unseen is not None:
        test_df_name = f'{test_df_name}_{hp.unseen}'

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    train_config = configs[train_task]
    test_config = configs[test_task]

    validset = train_config['validset']
    testset = test_config['testset']
    if hp.unseen is not None:
        testset = testset.format(unseen=hp.unseen)
        train_config['testset'] = train_config['testset'].format(unseen=hp.unseen)

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        summarizer = Summarizer(train_config, lm=hp.lm)
        validset = summarizer.transform_file(validset, max_len=hp.max_len)
        testset = summarizer.transform_file(testset, max_len=hp.max_len)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(train_config, hp.dk)
        else:
            injector = GeneralDKInjector(train_config, hp.dk)

        validset = injector.transform_file(validset)
        testset = injector.transform_file(testset)

    # load train/dev/test sets
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_dataset = DittoDataset(testset, lm=hp.lm)

    valid_iter = data.DataLoader(dataset=valid_dataset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=valid_dataset.pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=test_dataset.pad)
    
    # Load model
    ckpt_path = os.path.join(hp.logdir, train_task, 'model.pt')
    model = DittoModel(device='cpu', lm=hp.lm, alpha_aug=hp.alpha_aug)
    model.load_state_dict(torch.load(ckpt_path)['model'])

    # train and evaluate the model
    dev_f1, th = evaluate(model, valid_iter)
    test_f1, ys, preds = evaluate(model, test_iter, threshold=th, return_preds=True)

    test_df_name = f'er_validation/{test_df_name}.csv'
    df = pd.read_csv(test_df_name)
    #assert all([bool(y) for y in ys] == df['match'])
    df[f'pred_ditto_{train_task}'] = [bool(pred) for pred in preds]
    df.to_csv(test_df_name, index=False)
