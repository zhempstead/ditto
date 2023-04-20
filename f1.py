import pandas as pd
from pathlib import Path

for infile in list(Path('er_results').rglob('*.csv')):
    df = pd.read_csv(infile)
    print(infile)
    cols = [col for col in df.columns if col.startswith('pred')]
    for col in cols:
        if df[col].dtype == bool:
            counts = df.groupby(['match', col]).count()['left']
            tp = counts[(True, True)]
            fp = counts[(False, True)]
            fn = counts[(True, False)]
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = tp*2 / (tp*2 + fp + fn)
            print(f'- {col}: {f1:.3f}')
            print(f'  - precision: {precision:.3f}')
            print(f'  - recall: {recall:.3f}')
        else:
            total = df[col].max()
            print(f'- {col} (int):')
            for i in range(total):
                print(f'  - yes if >{i}:')
                df['tmp'] = df[col] > i
                counts = df.groupby(['match', 'tmp']).count()['left']
                tp = counts[(True, True)]
                fp = counts[(False, True)]
                fn = counts[(True, False)]
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = tp*2 / (tp*2 + fp + fn)
                print(f'    - f1: {f1:.3f}')
                print(f'    - precision: {precision:.3f}')
                print(f'    - recall: {recall:.3f}')


    print()
    #print(df.groupby(['match'] + cols).count()['left'])
    print()

