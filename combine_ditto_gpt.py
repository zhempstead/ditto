import pandas as pd

for product in ['all', 'cameras', 'computers', 'shoes', 'watches']:
    gpt = pd.read_csv(f'gpt_{product}_2.csv')
    ditto = pd.read_csv(f'data/wdc/{product}/test_pred.csv')
    for col in ditto:
        if col not in gpt:
            continue
        assert(all(gpt[col] == ditto[col]))
    df = ditto.merge(gpt)
    df = df.rename(columns={'chatgpt_pred': 'pred_chatgpt_0', 'ditto_pred': 'pred_ditto_xl'})
    df.to_csv(f'er_results/{product}.csv', index=False)
