import pandas as pd
import openai
import signal
import argparse
from dotenv import load_dotenv
import os
from pathlib import Path
from googletrans import Translator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff
from prompt_generator import TEMPLATES

STORIES = ['baseline', 'plain', 'veryplain', 'customer', 'journalist', 'security', 'layperson', 'detective']
DATASETS = ['cameras', 'computers', 'shoes', 'watches']
ALL_DATASETS = DATASETS + ['Amazon-Google', 'wdc_seen', 'wdc_half', 'wdc_unseen', 'wdc']

'''
Purpose: Run multiple prompts with multiple temperatures,
and figure out which configuration is best for entity resolution. We also have functions
to analyze the results and figure out which configuration will give
the best performance.
    
'''

class MyTimeoutException(Exception):
    pass

#register a handler for the timeout
def handler(signum, frame):
    print("Waited long enough!")
    raise MyTimeoutException("STOP")

def get_candidate(df_row):
    leftrow = df_row['left']
    rightrow = df_row['right']
    return (leftrow, rightrow)

def build_chat(prompt_tmp, row1, row2, examples=[], cot=False):
    fullprompt = prompt_tmp.get_prompt(row1, row2)
    
    if cot:
        followup = '\n\nAnswer: Let\'s think step-by-step.'
        system = "You are a helpful assistant who thinks step-by-step and then gives final yes or no answers."
    else:
        followup = 'Begin your answer with YES or NO.'
        system = "You are a helpful assistant who can only answer YES or NO and then explain your reasoning."

    if prompt_tmp.lang != 'english':
        translator = Translator()
        followup = translator.translate(followup, src='english', dest=fullprompt.lang).text
    chat = [{"role": "system", "content": system}]
    chat += prompt_tmp.get_chat(row1, row2, examples, followup)
    return chat

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def raw_response(chat, temp_val, lang='english', timeout=30, max_tokens=30):
    print("Sending: {}".format(chat))
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=chat,
        temperature=temp_val,
        max_tokens=max_tokens,
    )
    chat_response = response["choices"][0]["message"]["content"]
    
    if lang != 'english':
        translator = Translator()
        chat_response = translator.translate(chat_response, src=prompt_tmp.lang, dest='english').text
    
    return chat_response

def response_suffwtemp(prompt_tmp, row1, row2, temp_val, timeout=30):
    chat = build_chat(prompt_tmp, row1, row2)
    return raw_response(chat, temp_val, lang=prompt_tmp.lang, timeout=timeout)


def parse_enresponse(response, cot=False):
    if cot:
        if 'yes' in response.lower()[-6:]:
            return 1
        elif 'no' in response.lower()[-6:]:
            return 0
        else:
            return -1
    if response.lower().startswith('yes'):
        return 1
    elif response.lower().startswith('no'):
        return 0
    else:
        return -1

def storysuff(match_file, story_name, samp_range : list, samp_type, rows, match_prefix, num_reps=10, shots=0, shot_df=None, uniform_shot_offset=None, outdir='matchwsuff', cot=False, shot_fullrandom=False):
    '''
    Query chatGPT with the given story on the given rows of the match file at different temperatures,
    repeating each prompt a specified number of times at each temperature.

    Parameters
    ----------
    match_file : str
        Name of the file where candidates are stored.
    story_name : str
        Story name to use.
    samp_range : list
        List of temperature values to use.
    samp_type : str
        Type of sampling (we could use nucleus sampling too, but we are opting for temperature)
    rows : list
        list of rows to use from the match file, for sampling purposes.
    match_prefix : str
        The prefix describing the match file to be used on output files of this function.
    num_reps : int, optional
        Number of times each prompt should be repeated. The default is 10.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    story_tmp = TEMPLATES[story_name]
    
    df = pd.read_csv(match_file)
    outdir = Path(outdir)

    if shot_fullrandom:
        shot_yes = shot_df[shot_df['match']].sample(n=len(df)*shots, replace=True).reset_index(drop=True)
        shot_no = shot_df[~shot_df['match']].sample(n=len(df)*shots, replace=True).reset_index(drop=True)
    else:
        shot_yes = shot_df[shot_df['match']].sample(n=len(df)*shots, replace=True, random_state=100).reset_index(drop=True)
        shot_no = shot_df[~shot_df['match']].sample(n=len(df)*shots, replace=True, random_state=100).reset_index(drop=True)
    if story_name == 'customer':
        if cot:
            raise ValueError("CoT not compatible with 'customer' worker")
        shot_yes, shot_no = shot_no, shot_yes

    if cot:
        max_tokens = 300
    else:
        max_tokens = 30
    
    for r_ind in rows:
        dfrow = df.loc[r_ind]
        row_gt = dfrow['match']
        match = get_candidate(dfrow)
        for i in range(num_reps):
            for sval in samp_range:
                outname = outdir / ('-'.join([match_prefix, story_name, str(r_ind), f'rep{i}', f'{samp_type}{str(sval).replace(".", "_")}', f'{shots}shot']) + '.csv')
                if outname.exists():
                    print(f"{outname} exists...")
                    continue
                if samp_type == 'temperature':
                    examples = []
                    for j in range(shots):
                        if uniform_shot_offset is not None:
                            shot_idx = uniform_shot_offset * shots + j
                        else:
                            shot_idx = r_ind * shots + j
                        if cot:
                            examples.append((shot_yes.loc[shot_idx, 'left'], shot_yes.loc[shot_idx, 'right'], shot_yes.loc[shot_idx, 'cot']))
                            examples.append((shot_no.loc[shot_idx, 'left'], shot_no.loc[shot_idx, 'right'], shot_no.loc[shot_idx, 'cot']))
                        else:
                            examples.append((shot_yes.loc[shot_idx, 'left'], shot_yes.loc[shot_idx, 'right'], 'YES.'))
                            examples.append((shot_no.loc[shot_idx, 'left'], shot_no.loc[shot_idx, 'right'], 'NO.'))
                    print(outdir.name, i, match_prefix, story_name, r_ind)
                    chat = build_chat(story_tmp, match[0], match[1], examples, cot=cot)
                    story_response = raw_response(chat, sval, max_tokens=max_tokens)
                    story_answer = parse_enresponse(story_response, cot=cot)
                    if story_answer == -1:
                        chat.append({'role': 'assistant', 'content': story_response})
                        if cot:
                            chat.append({'role': 'user', 'content': "What is your final answer? Please answer in a single word (YES or NO). If you are uncertain, make your best guess."})
                            chat.append({'role': 'user', 'content': "Please answer in a single word (YES or NO). If you are uncertain, make your best guess."})
                        story_response2 = raw_response(chat, sval, max_tokens=5)
                        story_answer = parse_enresponse(story_response2, cot=False)
                else:
                    raise Exception("Sampling Type not supported: {}".format(samp_type))
                
                outdct = {}
                outdct['Match File'] = [match_file]
                outdct['Row No'] = [r_ind]
                outdct['Rep No'] = [i]
                outdct['Sampling Type'] = [samp_type]
                outdct['Sampling Param'] = [sval]
                outdct['Shots'] = [shots]
                outdct['Story Name'] = [story_name]
                outdct['Story Response'] = [story_response]
                outdct['Story Answer'] = [story_answer]
                outdct['Ground Truth'] = [row_gt]
                outdf = pd.DataFrame(outdct)
                outdf.to_csv(outname)


def query(args):
    load_dotenv()
    openai.api_key = os.getenv(f"OPENAI_API_KEY{args.key}")
    uniform_shot_offset = None
    if args.uniform_shots:
        uniform_shot_offset = args.uniform_shot_offset
    if args.cot:
        if args.shots_cot:
            print("--shots-cot is incompatible with --cot")
    for num_reps in range(1, args.reps + 1):
        print(f"Rep {num_reps}:")
        for d in args.datasets:
            print(f"Dataset {d}:")
            maindf = pd.read_csv(f'{args.source}/{d}.csv')
            if args.shot_dataset is None:
                traindf = pd.read_csv(f'er_train/{d}.csv')
            else:
                traindf = pd.read_csv(f'er_train/{args.shot_dataset}.csv')
            if args.cot or args.shots_cot:
                traindf = traindf[~traindf['cot'].isna()]
            match_prefix = d
            match_outfolder = f'{d}results'
            ditto_dct = maindf['match'].to_dict()
            rep_row = ditto_dct.keys()
            for s in args.stories:
                print(f"Story {s}")
                storysuff(f'{args.source}/{d}.csv', s, args.temps, 'temperature', rep_row, match_prefix, num_reps=num_reps, shots=args.shots, shot_df=traindf, uniform_shot_offset=uniform_shot_offset, outdir=args.rawdir, cot=args.cot, shot_fullrandom=args.shot_fullrandom)

EXAMPLES = [
    (
        "- title: instant immersion spanish deluxe 2.0\n- manf/modelno: topics entertaiment\n- price: 49.99",
        "- title: instant immers spanish dlux 2\n- manf/modelno: NULL\n- price: 36.11",
    ),
    (
        "- title: adventure workshop 4th-6th grade 7th edition\n- manf/modelno: encore software\n- price: 19.99",
        "- title: encore inc adventure workshop 4th-6th grade 8th edition\n- manf/modelno: NULL\n- price: 17.1",
    ),
    (
        "- title: sharp printing calculator\n- manf/modelno: sharp el1192bl\n- price: 57.63",
        "- title: new-sharp shr-el1192bl two-color printing calculator 12-digit lcd black red\n- manf/modelno: NULL\n-price: 56.0",
    ),
]

def examples(args):
    load_dotenv()
    openai.api_key = os.getenv(f"OPENAI_API_KEY{args.key}")
    story_tmp = TEMPLATES[args.story]
    for idx, (left, right) in enumerate(EXAMPLES):
        print(f"Example {idx+1}:")
        print("---")
        chat = build_chat(story_tmp, left, right, [])
        response = raw_response(chat, 0.0, max_tokens=1000)
        for entry in chat:
            print(f"<{entry['role']}>\n{entry['content']}")
            print()
        print("<response>")
        print(response)
        print()
        print()

def combine(args):
    os.makedirs(args.outdir, exist_ok=True)
    small_dfs = []
    big_dfs = []
    counter = 0
    print("Fixing '^M' characters...")
    os.system(f"bash ./fix_newline.sh {args.rawdir}")
    for f in Path(args.rawdir).glob(f'*.csv'):
        df = pd.read_csv(f)
        small_dfs.append(df)
        if len(df) > 1:
            import pdb; pdb.set_trace()
        counter += 1
        if counter % 1000 == 0:
            print(f'{counter}...')
            big_dfs.append(pd.concat(small_dfs))
            small_dfs = []
    print("Concatting...")
    df = pd.concat(big_dfs + small_dfs)
    # Fix customer query
    customer_0 = (df['Story Name'] == 'customer') & (df['Story Answer'] == 0)
    customer_1 = (df['Story Name'] == 'customer') & (df['Story Answer'] == 1)
    df.loc[customer_0, 'Story Answer'] = 1
    df.loc[customer_1, 'Story Answer'] = 0
    print("Writing...")
    df.to_csv(f'{args.outdir}/full.csv', index=False)

def retry(args):
    load_dotenv()
    openai.api_key = os.getenv(f"OPENAI_API_KEY{args.key}")
    result_file = f"{args.outdir}/full.csv"
    mf2df = {}
    full_df = pd.read_csv(result_file)
    to_fix = full_df[full_df['Story Answer'] == -1.0]
    total = len(to_fix)
    fixed = 0
    progress = 0
    for idx, row in to_fix.iterrows():
        mf = row['Match File']
        if mf not in mf2df:
            mf2df[mf] = pd.read_csv(mf)
        df = mf2df[mf]
        source_row = mf2df[mf].loc[row['Row No']]
        story_tmp = TEMPLATES[row['Story Name']]
        chat = build_chat(story_tmp, source_row['left'], source_row['right'])
        response = row['Story Response']
        if type(response) == str:
            chat.append({'role': 'assistant', 'content': row['Story Response']})
        chat.append({'role': 'user', 'content': "Please answer just YES or NO. If you are uncertain, make your best guess."})
        resp = raw_response(chat, row['Sampling Param'])
        answer = parse_enresponse(resp)
        progress += 1
        if answer != -1:
            fixed += 1
        print(f"Fixed {fixed}/{progress}; total {total}")
        full_df.loc[idx, 'Story Answer'] = answer
        if progress % 100 == 0:
            full_df.to_csv(result_file, index=False)
    full_df.to_csv(result_file, index=False) 
        
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_query = subparsers.add_parser('query')
    parser_query.add_argument("--source", default='er_results', help="Directory of source datasets")
    parser_query.add_argument("--stories", nargs='+', default=STORIES, choices=list(TEMPLATES.keys()))
    parser_query.add_argument("--datasets", nargs='+', default=DATASETS, choices=ALL_DATASETS)
    parser_query.add_argument("--reps", type=int, default=1)
    parser_query.add_argument("--temps", type=float, nargs='+', default=[0.0])
    parser_query.add_argument("--key", type=int, required=True, help='OpenAI API key index')
    parser_query.add_argument("--shots", type=int, default=0)
    parser_query.add_argument("--shot-dataset", default=None, choices=ALL_DATASETS)
    parser_query.add_argument("--shots-cot", action='store_true', help='Only use few-shot examples that have a CoT description associated (as a baseline for cot)')
    parser_query.add_argument("--cot", action='store_true', help='Chain-of-thought prompting')
    parser_query.add_argument("--shot-fullrandom", action='store_true', help='Fully random shots, different for each worker')
    parser_query.add_argument("--uniform-shots", action='store_true')
    parser_query.add_argument("--uniform-shot-offset", type=int, default=0)
    parser_query.add_argument("--rawdir", required=True)
    parser_query.set_defaults(func=query)

    parser_query = subparsers.add_parser('examples')
    parser_query.add_argument("--story", default='baseline', choices=list(TEMPLATES.keys()))
    parser_query.add_argument("--key", type=int, required=True)
    parser_query.set_defaults(func=examples)

    parser_combine = subparsers.add_parser('combine')
    parser_combine.add_argument("--rawdir", required=True)
    parser_combine.add_argument("--outdir", required=True)
    parser_combine.set_defaults(func=combine)

    parser_retry = subparsers.add_parser('retry')
    parser_retry.add_argument("--key", type=int, required=True)
    parser_retry.add_argument("--rawdir", required=True)
    parser_retry.set_defaults(func=retry)

    args = parser.parse_args()
    args.func(args)
