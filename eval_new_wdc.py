datasets = ["new_wdc"]
attrs = ['title', 'title_description', 'title_description_brand', 'title_description_brand_specTableContent']
sizes = ["small", "medium", "large"]

import os
import subprocess
import sys
import time

gpu_id = 0

for d in datasets:
    for size in sizes:
        test_task = '_'.join(['new_wdc', size])
        train_task = test_task
        for dk in [True]:
            for da in [True]:
                for unseen in ['seen', 'half', 'unseen']:
                    #for run_id in range(5):
                    for run_id in range(1):
                        cmd = """CUDA_VISIBLE_DEVICES=%d python eval_ditto.py \
                          --test-task %s \
                          --train-task %s \
                          --logdir results_new_wdc/ \
                          --fp16 \
                          --finetuning \
                          --save_model \
                          --batch_size 64 \
                          --lr 3e-5 \
                          --n_epochs 10 \
                          --run_id %d \
                          --unseen %s""" % (gpu_id, test_task, train_task, run_id, unseen)
                        if da:
                            cmd += ' --da del'
                        if dk:
                            cmd += ' --dk product'
                        cmd += ' --summarize'
                        print(cmd)
                        #os.system(cmd)
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
