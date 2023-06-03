import os
import subprocess
import sys
import time

datasets = """Dirty/DBLP-ACM
Dirty/DBLP-GoogleScholar
Dirty/iTunes-Amazon
Dirty/Walmart-Amazon
Structured/Amazon-Google
Structured/Beer
Structured/DBLP-ACM
Structured/DBLP-GoogleScholar
Structured/Fodors-Zagats
Structured/iTunes-Amazon
Structured/Walmart-Amazon
Textual/Abt-Buy
Textual/Company""".split('\n')

special_datasets = {
    'Structured/Beer': (32, 40),
    'Structured/iTunes-Amazon': (32, 40),
    'Structured/Fodors-Zagats': (32, 40),
    'Dirty/iTunes-Amazon': (32, 40),
    'Textual/Company': (32, 3)
}

ops = """swap
swap
append_col
del
swap
drop_col
swap
swap
append_col
drop_col
drop_col
swap
del""".split('\n')


lms = ['roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'roberta',
       'roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'bert']

# lms = ['xlnet', 'roberta', 'roberta', 'roberta', 'xlnet', 'bert',
#        'bert', 'xlnet', 'roberta', 'bert', 'roberta', 'roberta', 'bert']

# lms = """distilbert
# bert
# xlnet
# roberta""".split('\n')


for dataset, op, lm in zip(datasets, ops, lms):
    if dataset != "Structured/Amazon-Google":
        continue
    if dataset in special_datasets:
        batch_size, epochs = special_datasets[dataset]
    else:
        batch_size, epochs = 32, 15

    for da in [True]:
        for dk in [True]:
            for run_id in range(5):
                cmd = """CUDA_VISIBLE_DEVICES=0 python train_ditto.py \
              --task %s \
              --logdir results_ditto/ \
              --finetuning \
              --save_model \
              --batch_size %d \
              --lr 3e-5 \
              --fp16 \
              --lm %s \
              --n_epochs %d \
              --run_id %d""" % (dataset, batch_size, lm, epochs, run_id)
                if 'Company' in dataset:
                    cmd += ' --summarize'
                if da:
                    cmd += ' --da %s' % op
                if dk:
                    cmd += ' --dk general'
                print(cmd)
                #cmd = f"echo {dataset}"
                s = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                while s.poll() is None:
                    l = s.stdout.readline() # This blocks until it receives a newline.
                    print(l.decode('ascii'))
                    # When the subprocess terminates there might be unconsumed output 
                    # that still needs to be processed.
                print(s.stdout.read().decode('ascii'))
                print(s.poll())
                if s.poll() != 0:
                    raise ValueError("Failed")

