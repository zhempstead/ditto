import argparse
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

def query_df(df, context, col, shots=0, df_shots=None, temperature=0, trials=1):
    if not col in df.columns:
        df[col] = None
    preamble = construct_preamble(context)

    numrows = len(df)
    count = 0
    for idx, row in df.iterrows():
        count += 1
        print(f"{count}/{numrows}")
        if trials == 1:
            resp = query_row(row, preamble, shots, df_shots, temperature)
            print(resp)
            df.loc[idx, col] = resp
        else:
            resps = []
            for trial in range(trials):
                print(f"  Trial {trial+1}/{trials}")
                resp = query_row(row, preamble, shots, df_shots, temperature)
                
                resps.append(query_row(row, preamble, shots, df_shots, temperature))
            print(f"  Result {sum(resps)}")
            df.loc[idx, col] = sum(resps)
    return df

def query_row(row, preamble, shots=0, df_shots=None, temperature=0):
    prompt = construct_prompt(row)
    chat = preamble.copy()
    if shots > 0:
        pos = df_shots[df_shots['match']].sample(n=shots)
        neg = df_shots[~df_shots['match']].sample(n=shots)
        df_ex = pd.concat([pos, neg])
        df_ex = df_ex.sample(frac=1)
        for i in range(shots*2):
            prompt_ex = construct_prompt(df_ex.iloc[i])
            chat.append({"role": "user", "content": prompt_ex})
            resp_ex = "Yes." if df_ex.iloc[i]['match'] else "No."
            chat.append({"role": "assistant", "content": resp_ex})
    chat.append({"role": "user", "content": prompt})
    return yes_or_no(chat, temperature)

def yes_or_no(chat, temperature=0):
    for max_tokens in [5, 10, 20, 50]:
        resp = get_response(chat, max_tokens=max_tokens, temperature=temperature)
        if "yes" in resp.lower():
            return True
        elif "no" in resp.lower():
            return False
    print(resp)
    chat.append({"role": "assistant", "content": resp})
    chat.append({"role": "user", "content": "Please answer just yes or no. If you are uncertain, make your best guess. Do the entries refer to the same product?"})
    resp = get_response(chat, max_tokens=20)
    if "yes" in resp.lower():
        return True
    elif "no" in resp.lower():
        return False
    print("Coundn't parse response:")
    print(resp)
    return False

def construct_preamble(context):
    preamble = [{"role": "system", "content": "You are a helpful assistant."}]
    preamble.append({"role": "user", "content": f"We are trying to integrate product data from two different databases. The goal is to look at two product entries, one from each database, and determine whether the two entries refer to the same product or not. Since the databases are different, there will still be some differences between entries that refer to the same product. {context}\n\nDo you understand?"})
    preamble.append({"role": "assistant", "content": "Yes, I understand."})
    return preamble


def construct_prompt(row):
    left = row['left'].split('\t')
    right = row['right'].split('\t')
    left = '\n'.join([f"- {l}" for l in left])
    right = '\n'.join([f"- {r}" for r in right])
    return f"""Here is an entry from the first database:
{left}

Here is an entry from the second database:
{right}

As best as you can tell, do these entries refer to the same product? (Answer Yes or No)"""

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_response(chat, max_tokens=5, temperature=0):
    '''
    send the prompt and data to chatgpt.
    If we do not get the answer in timeout seconds,
    resend the request.
    ...I have a feeling this will not work...
    '''
    #print("Sending: {}".format(chat))
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    #print(f'{response["usage"]["prompt_tokens"]} prompt tokens used.')
    #print(response["choices"][0]["message"]["content"])
    return response["choices"][0]["message"]["content"]

if __name__ == '__main__':
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    parser = argparse.ArgumentParser()
    parser.add_argument("testfile", type=str)
    parser.add_argument("--trainfile", type=str)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--product", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()

    if args.shots > 0 and args.trainfile is None:
        raise ValueError("Must specify training set for few-shot learning")
    if args.trials > 1 and args.temperature == 0.0:
        raise ValueError("Must increase temperature if doing multiple trials")
    if args.product is not None:
        context = f"The products in these databases are {args.product}."
    else:
        context = None

    df_test = pd.read_csv(args.testfile)
    df_train = None
    if args.trainfile is not None:
        df_train = pd.read_csv(args.trainfile)

    col = f"pred_chatgpt_{args.shots}"
    if args.temperature > 0.0:
        col = f"{col}_temp{args.temperature}"
    df_test = query_df(df_test, context, col, args.shots, df_train, args.temperature, args.trials)
    df_test.to_csv(args.testfile, index=False)
