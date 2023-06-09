import argparse

import pandas as pd

ATTRIBUTES = ['brand', 'title', 'description', 'price', 'priceCurrency']

def ditto_row(row):
    out = []
    for example in ['right', 'left']:
        out_example = []
        for attribute in ATTRIBUTES:
            col = f'{attribute}_{example}'
            if row[col] is None:
                continue
            out_example.append(f"COL {attribute} VAL {row[col]}")
        out.append(' '.join(out_example))
    out.append(str(row['label']))
    return '\t'.join(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("out_ditto")
    args = parser.parse_args()

    df = pd.read_json(args.infile, compression='gzip', lines=True)
    series = df.apply(ditto_row, axis=1).to_list()
    with open(args.out_ditto, 'w') as fout:
        fout.write('\n'.join(series))
