from pathlib import Path
import re
import sys

import pandas as pd

LANGUAGE_TAG = r'"@[a-z][a-z](-[a-zA-Z]*)?'
COLVAL = r'COL ([a-zA-Z]+) VAL +'

indir = Path(sys.argv[1])
for infile in list(indir.rglob('*.txt.*')) + list(indir.rglob('*.txt')):
    if 'jsonl' in infile.name:
        continue
    newname = infile.name.replace('.txt', '') + '.csv'
    outfile = infile.parent / newname
    print(f"Processing {infile} to {outfile}...")
    lefts = []
    rights = []
    ys = []
    prompts = []
    completions = []
    with open(infile) as f:
        for line in f.readlines():
            line = line.strip()
            line = re.sub(LANGUAGE_TAG, '', line)
            line = line.replace('"', '')
            line = re.sub(COLVAL, r'\n\1: ', line)
            left, right, y = line.split('\t')
            left = left.strip()
            right = right.strip()
            prompt = f"{left}\n\n###\n\n{right}\n\n###\n\nSame product?"
            y = bool(int(y))
            completion = " yes" if y else " no"
            lefts.append(left)
            rights.append(right)
            prompts.append(prompt)
            ys.append(y)
            completions.append(completion)
    df = pd.DataFrame({'left': lefts, 'right': rights, 'match': ys, 'prompt': prompts, 'completion': completions})
    df.to_csv(outfile, index=False)
