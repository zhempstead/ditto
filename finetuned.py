from dotenv import load_dotenv
import pandas as pd
import openai
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

FINETUNED_MODELS = {
    ('all', 'small', 'ada'): "ada:ft-university-of-chicago-2023-04-11-06-11-59",
    ('all', 'small', 'curie'): "curie:ft-university-of-chicago-2023-04-11-05-38-00",
    ('all', 'large', 'ada'): "ada:ft-university-of-chicago-2023-04-11-05-48-56",
    ('cameras', 'small', 'ada'): "ada:ft-university-of-chicago-2023-04-07-21-15-23",
    ('cameras', 'small', 'curie'): "curie:ft-university-of-chicago-2023-04-07-21-17-27",
    ('cameras', 'large', 'ada'): "ada:ft-university-of-chicago-2023-04-10-22-46-30",
    ('cameras', 'large', 'curie'): "curie:ft-university-of-chicago-2023-04-10-22-46-10",
    ('shoes', 'small', 'ada'): "ada:ft-university-of-chicago-2023-04-10-23-10-45",
    ('shoes', 'small', 'curie'): "curie:ft-university-of-chicago-2023-04-10-23-14-33",
    ('shoes', 'large', 'ada'): "ada:ft-university-of-chicago-2023-04-07-22-21-08",
    ('shoes', 'large', 'curie'): "curie:ft-university-of-chicago-2023-04-10-22-03-04",
    ('shoes', 'xlarge', 'ada'): "ada:ft-university-of-chicago-2023-04-10-22-10-02",
    ('Amazon-Google', None, 'ada'): "ada:ft-university-of-chicago-2023-04-11-17-00-45",
    ('Amazon-Google', None, 'curie'): "curie:ft-university-of-chicago-2023-04-11-16-54-37",
}

def add_results(df, colname, model):
    results = []
    for idx, row in df.iterrows():
        prompt = f"{row['left']}\n\n###\n\n{row['right']}\n\n###\n\nSame product?"
        resp = yes_or_no(prompt, model)
        results.append(resp)
    df[colname] = results
    return df

def yes_or_no(chat, model):
    resp = get_response(chat, model)
    if "yes" in resp.lower():
        return True
        chatgpt_pred.append(True)
    elif "no" in resp.lower():
        return False
    print(resp)
    raise ValueError(resp)


@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_response(prompt, model):
    '''
    send the prompt and data to chatgpt.
    If we do not get the answer in timeout seconds,
    resend the request.
    ...I have a feeling this will not work...
    '''
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0,
        max_tokens=5,
    )
    return response["choices"][0]["text"]

if __name__ == '__main__':
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    #dataset = 'cameras'
    #data_size = 'large'
    #model = 'curie'

    for (dataset, data_size, model) in FINETUNED_MODELS.keys():
        if dataset != "Amazon-Google":
            continue
        print(dataset, data_size, model)
        if data_size is None:
            colname = f'pred_finetune_{model}'
        else:
            colname = f'pred_finetune_{model}_{data_size}'
        df = pd.read_csv(f'er_results/{dataset}.csv')
        df = add_results(df, colname, FINETUNED_MODELS[(dataset, data_size, model)])
        df.to_csv(f'er_results/{dataset}.csv', index=False)