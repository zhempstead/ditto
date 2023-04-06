from pathlib import Path
import re
import sys

import pandas as pd

LANGUAGE_TAG = r'"@[a-z][a-z](-[a-zA-Z]*)?'
COLVAL = r'COL ([a-zA-Z]+) VAL +'

indir = Path(sys.argv[1])
for infile in indir.rglob('test.txt'):
    outfile = infile.parent / 'test.csv'
    print(f"Processing {infile} to {outfile}...")
    lefts = []
    rights = []
    ys = []
    with open(infile) as f:
        for line in f.readlines():
            line = line.strip()
            line = re.sub(LANGUAGE_TAG, '', line)
            line = line.replace('"', '')
            line = re.sub(COLVAL, r'\n\1: ', line)
            left, right, y = line.split('\t')
            left = left.strip()
            right = right.strip()
            y = bool(int(y))
            lefts.append(left)
            rights.append(right)
            ys.append(y)
    df = pd.DataFrame({'left': lefts, 'right': rights, 'match': ys})
    df.to_csv(outfile, index=False)
