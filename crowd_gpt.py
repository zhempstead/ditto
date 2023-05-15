import pandas as pd
import pickle
import openai
import signal
import argparse
from dotenv import load_dotenv
import os
from pathlib import Path
import re
from crowdkit.aggregation import MajorityVote, Wawa, DawidSkene
from googletrans import Translator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff
from prompt_generator import TEMPLATES


STORIES = ['baseline', 'plain', 'veryplain', 'customer', 'journalist', 'security', 'layperson', 'detective']
DATASETS = ['cameras', 'computers', 'shoes', 'watches', 'Amazon-Google']

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
    leftrow = leftrow.replace('\n', '\t')
    leftrow = ' "' + leftrow + '".'
    rightrow = rightrow.replace('\n', '\t')
    rightrow = ' "' + rightrow + '".'
    return (leftrow, rightrow)

def build_chat(prompt_tmp, row1, row2, examples=[]):
    fullprompt = prompt_tmp.get_prompt(row1, row2)
    
    followup = 'Begin your answer with YES or NO.'
    if prompt_tmp.lang != 'english':
        translator = Translator()
        followup = translator.translate(followup, src='english', dest=fullprompt.lang).text
    chat = [{"role": "system", "content": "You are a helpful assistant who can only answer YES or NO and then explain your reasoning."}]
    chat += prompt_tmp.get_chat(row1, row2, examples, followup)
    return chat

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def raw_response(chat, temp_val, lang='english', timeout=30):
    print("Sending: {}".format(chat))
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=temp_val,
        max_tokens=30
    )
    chat_response = response["choices"][0]["message"]["content"]
    
    if lang != 'english':
        translator = Translator()
        chat_response = translator.translate(chat_response, src=prompt_tmp.lang, dest='english').text
    
    return chat_response

def response_suffwtemp(prompt_tmp, row1, row2, temp_val, timeout=30):
    chat = build_chat(prompt_tmp, row1, row2)
    return raw_response(chat, temp_val, lang=prompt_tmp.lang, timeout=timeout)


def parse_enresponse(response):
    if response.lower().startswith('yes'):
        return 1
    elif response.lower().startswith('no'):
        return 0
    else:
        return -1

def storysuff(match_file, story_name, samp_range : list, samp_type, rows, match_prefix, num_reps=10, shots=0, shot_df=None, outdir='matchwsuff'):
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
    
    #story_fname = 'all_prompts/' + story_name + 'prompt_english.pkl'
    #with open(story_fname, 'rb') as fh:
    #    story_tmp = pickle.load(fh)
    story_tmp = TEMPLATES[story_name]
    
    df = pd.read_csv(match_file)
    outdir = Path(outdir)

    shot_yes = shot_df[shot_df['match']].sample(n=len(df)*shots, replace=True, random_state=100).reset_index(drop=True)
    shot_no = shot_df[~shot_df['match']].sample(n=len(df)*shots, replace=True, random_state=100).reset_index(drop=True)

    
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
                    shot_idx = r_ind * shots
                    for i in range(shots):
                        examples.append((shot_yes.loc[shot_idx, 'left'], shot_yes.loc[shot_idx, 'right'], 'YES.'))
                        examples.append((shot_no.loc[shot_idx, 'left'], shot_no.loc[shot_idx, 'right'], 'NO.'))
                        shot_idx += 1
                    chat = build_chat(story_tmp, match[0], match[1], examples)
                    story_response = raw_response(chat, sval)
                    story_answer = parse_enresponse(story_response)
                    if story_answer == -1:
                        chat.append({'role': 'assistant', 'content': story_response})
                        chat.append({'role': 'user', 'content': "Please answer in a single word (YES or NO). If you are uncertain, make your best guess."})
                        story_response = raw_response(chat, sval)
                        story_answer = parse_enresponse(story_response)
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
    

def crowd_gather(raw_df, temp, outdir):
    '''
    Run different crowd methods on all chatgpt responses.

    Parameters
    ----------
    fullfname : TYPE
        DESCRIPTION.
    temp : TYPE
        DESCRIPTION.
    ditto_dct : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    df = raw_df[raw_df['Sampling Param'] == temp]
    out_schema = ['worker', 'task', 'label']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    raw_labels = df['Story Answer'].tolist()
    new_labels = [max(x, 0) for x in raw_labels] #change -1s to 0s.
    
    out_dct['worker'] = df['Story Name'].tolist()
    out_dct['label'] = new_labels
    matchfiles = df['Match File'].tolist()
    rownos = df['Row No'].tolist()
    tasklst = []
    pairslst = []
    
    for i in range(len(matchfiles)):
        new_el = matchfiles[i] + ':' + str(rownos[i])
        tasklst.append(new_el)
        pairslst.append((matchfiles[i], rownos[i]))
    
    pairsset = set(pairslst)
    
    out_dct['task'] = tasklst
    
    out_df = pd.DataFrame(out_dct)
    
    agg_mv = MajorityVote().fit_predict(out_df)
    agg_wawa = Wawa().fit_predict(out_df)
    agg_ds = DawidSkene(n_iter=10).fit_predict(out_df)
    
    mv_dct = agg_mv.to_dict()
    wawa_dct = agg_wawa.to_dict()
    ds_dct = agg_ds.to_dict()
    
    #res_schema = ['Match File', 'Row No', 'Vote', 'Ditto Answer', 'Ground Truth']
    res_schema = ['Match File', 'Row No', 'Vote', 'Ground Truth']
    mv_res = {}
    wawa_res = {}
    ds_res = {}
    
    for rs in res_schema:
        mv_res[rs] = []
        wawa_res[rs] = []
        ds_res[rs] = []
    
    for pair in pairsset:
        mv_res['Match File'].append(pair[0])
        mv_res['Row No'].append(pair[1])
        wawa_res['Match File'].append(pair[0])
        wawa_res['Row No'].append(pair[1])
        ds_res['Match File'].append(pair[0])
        ds_res['Row No'].append(pair[1])
        
        task_ind = pair[0] + ':' + str(pair[1])
        mv_res['Vote'].append(mv_dct[task_ind])
        wawa_res['Vote'].append(wawa_dct[task_ind])
        ds_res['Vote'].append(ds_dct[task_ind])
        
        pair_df = df[(df['Match File'] == pair[0]) & (df['Row No'] == pair[1])]
        pair_gt = pair_df['Ground Truth'].unique().tolist()[0]
        mv_res['Ground Truth'].append(pair_gt)
        wawa_res['Ground Truth'].append(pair_gt)
        ds_res['Ground Truth'].append(pair_gt)
        
        #mv_res['Ditto Answer'].append(ditto_dct[pair[1]])
        #wawa_res['Ditto Answer'].append(ditto_dct[pair[1]])
        #ds_res['Ditto Answer'].append(ditto_dct[pair[1]])
    
    mv_df = pd.DataFrame(mv_res)
    wawa_df = pd.DataFrame(wawa_res)
    ds_df = pd.DataFrame(ds_res)
    
    mv_df.to_csv(outdir + '/MajorityVote_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    wawa_df.to_csv(outdir + '/Wawa_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    ds_df.to_csv(outdir + '/DawidSkene_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)


def perprompt_majorities(df, temp, outdir):
    '''
    Compute the majority response across iterations for the same prompt.

    Parameters
    ----------
    fullfname : TYPE
        DESCRIPTION.
    temp : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    tmpdf = df[df['Sampling Param'] == temp]
    out_schema = ['Match File', 'Row No', 'Prompt', 'Yes Votes', 'No Votes', 'Majority']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    for prompt in tmpdf['Story Name'].unique().tolist():
        promptdf = tmpdf[tmpdf['Story Name'] == prompt]
        for match_file in promptdf['Match File'].unique().tolist():
            for rowno in promptdf['Row No'].unique().tolist():
                cand_df = promptdf[(promptdf['Match File'] == match_file) & (promptdf['Row No'] == rowno)]
                vote_dct = cand_df['Story Answer'].value_counts().to_dict()
                if 1 in vote_dct:
                    yesvotes = vote_dct[1]
                else:
                    yesvotes = 0
                
                if 0 in vote_dct:
                    novotes = vote_dct[0]
                else:
                    novotes = 0
                othervotes = sum([vote_dct[k] for k in vote_dct if k != 0 and k != 1])
                out_dct['Match File'].append(match_file)
                out_dct['Row No'].append(rowno)
                out_dct['Prompt'].append(prompt)
                out_dct['Yes Votes'].append(yesvotes)
                out_dct['No Votes'].append(novotes + othervotes)
                if yesvotes > (novotes + othervotes):
                    out_dct['Majority'].append(True)
                else:
                    out_dct['Majority'].append(False)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(outdir + '/per_promptresults_tmperature' + str(temp).replace('.', '_') + '.csv', index=False)


def get_stats(method_names, temperatures, story_names, outdir):
    '''
    Generate a csv to compare the precision, recall, f1 of crowd methods, individual prompt responses, etc. to baselines

    Parameters
    ----------
    method_names : TYPE
        DESCRIPTION.
    temps : TYPE
        DESCRIPTION.
    story_names : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    method2df = {}
    story2df = {}
    for mn in method_names:
        for temp in temperatures:
            strtemp = str(temp).replace('.', '_')
            method2df[(mn, temp)] = pd.read_csv(f'{outdir}/{mn}_results-temperature{strtemp}.csv')

    for temp in temperatures:
        strtemp = str(temp).replace('.', '_')
        story_df = pd.read_csv(f'{outdir}/per_promptresults_tmperature{strtemp}.csv')
        for sn in story_names:
            story2df[(sn, temp)] = story_df[story_df['Prompt'] == sn]
    truth_df = method2df[(method_names[0], temperatures[0])][['Match File', 'Row No', 'Ground Truth']].copy()

    out_dict = {'Dataset': [], 'Method': [], 'Temp': [], 'Crowd': [], 'F1': [], 'Precision': [], 'Recall': []}

    for mn in method_names:
        for temp in temperatures:
            full_df = method2df[(mn, temp)]
            full_df['Vote'] = (full_df['Vote'] == 1)

            for dataset, df in full_df.groupby('Match File'):
                our_tps = df[(df['Vote'] == df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
                our_tns = df[(df['Vote'] == df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
                our_fps = df[(df['Vote'] != df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
                our_fns = df[(df['Vote'] != df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            
                our_precision = our_tps / (our_tps + our_fps)
                our_recall = our_tps / (our_tps + our_fns)
                our_f1 = 2 * (our_precision * our_recall) / (our_precision + our_recall)

                out_dict['Dataset'].append(dataset.split('/')[1].split('.')[0])
                out_dict['Method'].append(mn)
                out_dict['Temp'].append(temp)
                out_dict['Crowd'].append(True)
                out_dict['F1'].append(our_f1)
                out_dict['Precision'].append(our_precision)
                out_dict['Recall'].append(our_recall)
            
    for sn in story_names:
        for temp in temperatures:
            full_df = story2df[(sn, temp)]
            full_df['Vote'] = full_df['Majority']
            full_df = full_df.merge(truth_df, on=['Match File', 'Row No'])

            for dataset, df in full_df.groupby('Match File'):
                our_tps = df[(df['Vote'] == df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
                our_tns = df[(df['Vote'] == df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
                our_fps = df[(df['Vote'] != df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
                our_fns = df[(df['Vote'] != df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            
                our_precision = our_tps / (our_tps + our_fps)
                our_recall = our_tps / (our_tps + our_fns)
                our_f1 = 2 * (our_precision * our_recall) / (our_precision + our_recall)

                out_dict['Dataset'].append(dataset.split('/')[1].split('.')[0])
                out_dict['Method'].append(sn)
                out_dict['Temp'].append(temp)
                out_dict['Crowd'].append(False)
                out_dict['F1'].append(our_f1)
                out_dict['Precision'].append(our_precision)
                out_dict['Recall'].append(our_recall)

    pd.DataFrame(out_dict).to_csv(f'{outdir}/stats.csv', index=False)


def query(args):
    load_dotenv()
    openai.api_key = os.getenv(f"OPENAI_API_KEY{args.key}")
    for num_reps in range(1, args.reps + 1):
        print(f"Rep {num_reps}:")
        for d in args.datasets:
            print(f"Dataset {d}:")
            maindf = pd.read_csv(f'er_results/{d}.csv')
            traindf = pd.read_csv(f'er_train/{d}.csv')
            match_prefix = d
            match_outfolder = f'{d}results'
            ditto_dct = maindf['match'].to_dict()
            rep_row = ditto_dct.keys()
            for s in args.stories:
                print(f"Story {s}")
                storysuff(f'er_results/{d}.csv', s, args.temps, 'temperature', rep_row, match_prefix, num_reps=num_reps, shots=args.shots, shot_df=traindf, outdir=args.rawdir)


def combine(args):
    os.makedirs(args.outdir, exist_ok=True)
    #regex = re.compile(r'(?P<dataset>[A-Za-z]+(-[A-Za-z]+)?)-(?P<story>[a-z]+)-(?P<row>[0-9]+)-rep(?P<rep>[0-9]+)-temperature(?P<temp>[0-2])_0-(?P<shot>[0-9])shot.csv')
    small_dfs = []
    big_dfs = []
    counter = 0
    #print("Fixing '^M' characters...")
    #os.system(f"""cd {args.rawdir} && echo ./* | xargs sed -i 's/#/ /g'""")
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
    df = pd.concat(big_dfs)
    print("Writing...")
    df.to_csv(f'{args.outdir}/full.csv', index=False)

def analyze(args):
    result_file = f"{args.outdir}/full.csv"
    df = pd.read_csv(result_file)
    df = df[~df['Sampling Param'].isna()]
    df['Rep No'] = df['Rep No'].astype(float).astype(int)
    df['Row No'] = df['Row No'].astype(float).astype(int)
    if args.reps is not None:
        df = df[df['Rep No'] < args.reps]
    for temp in args.temps:
        print(f"Temp {temp}:")
        crowd_gather(df, temp, args.outdir)
        perprompt_majorities(df, temp, args.outdir)
    get_stats(["MajorityVote", "Wawa", "DawidSkene"], args.temps, args.stories, args.outdir)

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
    #this is a representative row, 
    #in that the first is a true positive, next is false positive, next is true negative, next is false negative
    # rep_row = [23, 25, 46, 0, 29, 31]
    # rep_row = [2, 12, 30, 0, 1, 3, 46, 201, 302, 44, 136, 207]
    # ditto_dct = {2 : True, 12 : True, 30 : True, 0 : False, 1 : False, 3 : False, 46 : True, 201 : True, 302 : True, 44 : False, 136 : False, 207 : False}
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_query = subparsers.add_parser('query')
    parser_query.add_argument("--stories", nargs='+', default=STORIES, choices=STORIES)
    parser_query.add_argument("--datasets", nargs='+', default=DATASETS, choices=DATASETS)
    parser_query.add_argument("--reps", type=int, default=10)
    parser_query.add_argument("--temps", type=float, nargs='+', default=[2.0])
    parser_query.add_argument("--key", type=int, required=True)
    parser_query.add_argument("--shots", type=int, default=0)
    parser_query.add_argument("--rawdir", required=True)
    parser_query.set_defaults(func=query)

    parser_combine = subparsers.add_parser('combine')
    parser_combine.add_argument("--rawdir", required=True)
    parser_combine.add_argument("--outdir", required=True)
    parser_combine.set_defaults(func=combine)

    parser_analyze = subparsers.add_parser('analyze')
    parser_analyze.add_argument("--reps", type=int, default=10)
    parser_analyze.add_argument("--temps", type=float, nargs='+', default=[2.0])
    parser_analyze.add_argument("--stories", nargs='+', default=STORIES, choices=STORIES)
    parser_analyze.add_argument("--outdir", required=True)
    parser_analyze.set_defaults(func=analyze)

    parser_retry = subparsers.add_parser('retry')
    parser_retry.add_argument("--key", type=int, required=True)
    parser_retry.add_argument("--rawdir", required=True)
    parser_retry.set_defaults(func=retry)

    args = parser.parse_args()
    args.func(args)
