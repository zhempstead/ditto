import pandas as pd
import pickle
import openai
import signal
import os
from crowdkit.aggregation import MajorityVote, Wawa, DawidSkene
from googletrans import Translator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

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

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def response_wtemp(prompt_tmp, row1, row2, temp_val, timeout=30):
    fullprompt = prompt_tmp.get_prompt(row1, row2)
    
    followup = 'Can you summarize your answer as \'YES\' or \'NO\'?'
    if prompt_tmp.lang != 'english':
        translator = Translator()
        followup = translator.translate(followup, src='english', dest=fullprompt.lang).text
    
    chat = [{"role": "system", "content": "You are a helpful assistant."}]
    fullmsg = {"role": "user", "content": fullprompt }
    chat.append(fullmsg)
    print("Sending: {}".format(chat))
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=temp_val
    )
    chat_explanation = response["choices"][0]["message"]["content"]
    
    chat.append({"role": "assistant", "content": chat_explanation})
    chat.append({"role" : "user", "content": followup})
    print("Now Sending: {}".format(chat))
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=temp_val
    )
    chat_response = response["choices"][0]["message"]["content"]
    
    if prompt_tmp.lang != 'english':
        translator = Translator()
        chat_explanation = translator.translate(chat_explanation, src=prompt_tmp.lang, dest='english').text
        chat_response = translator.translate(chat_response, src=prompt_tmp.lang, dest='english').text
    
    return chat_explanation, chat_response

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def response_suffwtemp(prompt_tmp, row1, row2, temp_val, timeout=30):
    fullprompt = prompt_tmp.get_prompt(row1, row2)
    
    followup = 'Begin your answer with YES or NO.'
    if prompt_tmp.lang != 'english':
        translator = Translator()
        followup = translator.translate(followup, src='english', dest=fullprompt.lang).text
    
    fullprompt = fullprompt + ' ' + followup
    
    chat = [{"role": "system", "content": "You are a helpful assistant who can only answer YES or NO and then explain your reasoning."}]
    fullmsg = {"role": "user", "content": fullprompt }
    chat.append(fullmsg)
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
    
    if prompt_tmp.lang != 'english':
        translator = Translator()
        chat_response = translator.translate(chat_response, src=prompt_tmp.lang, dest='english').text
    
    return chat_response

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def response_wtopp(prompt_tmp, row1, row2, topp_val, timeout=30):
    fullprompt = prompt_tmp.get_prompt(row1, row2)
    
    followup = 'Can you summarize your answer as \'YES\' or \'NO\'?'
    if prompt_tmp.lang != 'english':
        translator = Translator()
        followup = translator.translate(followup, src='english', dest=fullprompt.lang).text
    
    chat = [{"role": "system", "content": "You are a helpful assistant."}]
    fullmsg = {"role": "user", "content": fullprompt }
    chat.append(fullmsg)
    print("Sending: {}".format(chat))
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        top_p=topp_val
    )
    chat_explanation = response["choices"][0]["message"]["content"]
    
    chat.append({"role": "assistant", "content": chat_explanation})
    chat.append({"role" : "user", "content": followup})
    print("Now Sending: {}".format(chat))
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        top_p=topp_val
    )
    chat_response = response["choices"][0]["message"]["content"]
    
    if prompt_tmp.lang != 'english':
        translator = Translator()
        chat_explanation = translator.translate(chat_explanation, src=prompt_tmp.lang, dest='english').text
        chat_response = translator.translate(chat_response, src=prompt_tmp.lang, dest='english').text
        
    
    return chat_explanation, chat_response

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def test_tempwresp(timeout=30):
    prompt = "Suppose you're a detective who has been hired to detect illegal copies of products so that law enforcement can fine those selling these copies. Suppose you are given the following information about a product that a business may be trying to sell illegally: " 
    prompt += "\"title: microsoft visual studio test agent 2005 cd 1 processor license manufacturer: microsoft software price: 5099.0\"."
    prompt += " You confront the storefront owner potentially selling the illegal copy. You find the following information: "
    prompt += "\"title: individual software professor teaches microsoft office 2007 manufacturer: price: 29.99\". "
    prompt += "Would you charge the storefront owner for selling an illegal copy of the product?"
    prompt += " Begin your answer with YES or NO."
    full_msg = {"role" : "user", "content" : prompt}
    
    chat = [{"role": "system", "content": "You are a helpful assistant who is only capable of answering 'yes' or 'no' and then explaining your reasoning."}]
    chat.append(full_msg)
    print("Sending: {}".format(chat))
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=2.0
    )
    chat_rsp = response["choices"][0]["message"]["content"]
    print(chat_rsp)
    
def parse_enresponse(response):
    if response.lower().startswith('yes'):
        return 1
    elif response.lower().startswith('no'):
        return 0
    else:
        return -1

#it's possible that changing languages will not
#give you a YES or NO back.
def parse_nonenresponse(response):
    if 'yes' in response.lower():
        return 1
    elif 'no' in response.lower():
        return 0
    else:
        return -1

def plain_vs_story(match_file, story_name, samp_range : list, samp_type, rows, num_reps=5):
    story_fname = 'all_prompts/' + story_name + 'prompt_english.pkl'
    plain_fname = 'all_prompts/plainprompt_english.pkl'
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    with open(plain_fname, 'rb') as fh:
        plain_tmp = pickle.load(fh)
    
    df = pd.read_csv(match_file)
    
    for r_ind in rows:
        dfrow = df.loc[r_ind]
        row_gt = dfrow['match']
        match = get_candidate(dfrow)
        for i in range(num_reps):
            for sval in samp_range:
                outname = 'match-' + str(r_ind) + '-' + 'rep' + str(i) + '-' + samp_type + str(sval).replace('.', '_') + '.csv'
                if samp_type == 'temperature':
                    plain_explanation, plain_response = response_wtemp(plain_tmp, match[0], match[1], sval)
                    story_explanation, story_response = response_wtemp(story_tmp, match[0], match[1], sval)
                elif samp_type == 'nucleus':
                    plain_explanation, plain_response = response_wtopp(plain_tmp, match[0], match[1], sval)
                    story_explanation, story_response = response_wtopp(story_tmp, match[0], match[1], sval)
                else:
                    raise Exception("Sampling Type not supported: {}".format(samp_type))
                
                plain_answer = parse_enresponse(plain_response)
                story_answer = parse_enresponse(story_response)
                outdct = {}
                outdct['Match File'] = [match_file]
                outdct['Row No'] = [r_ind]
                outdct['Rep No'] = [i]
                outdct['Sampling Type'] = [samp_type]
                outdct['Sampling Param'] = [sval]
                outdct['Story Name'] = [story_name]
                outdct['Plain Explanation'] = [plain_explanation]
                outdct['Plain Response'] = [plain_response]
                outdct['Plain Answer'] = [plain_answer]
                outdct['Story Explanation'] = [story_explanation]
                outdct['Story Response'] = [story_response]
                outdct['Story Answer'] = [story_answer]
                outdct['Ground Truth'] = [row_gt]
                outdf = pd.DataFrame(outdct)
                outdf.to_csv(outname)

def plain_vs_storysuff(match_file, story_name, samp_range : list, samp_type, rows, num_reps=5):
    story_fname = 'all_prompts/' + story_name + 'prompt_english.pkl'
    plain_fname = 'all_prompts/veryplainprompt_english.pkl'
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    with open(plain_fname, 'rb') as fh:
        plain_tmp = pickle.load(fh)
    
    df = pd.read_csv(match_file)
    
    for r_ind in rows:
        dfrow = df.loc[r_ind]
        row_gt = dfrow['match']
        match = get_candidate(dfrow)
        for i in range(num_reps):
            for sval in samp_range:
                outname = 'matchwsuff-' + str(r_ind) + '-' + 'rep' + str(i) + '-' + samp_type + str(sval).replace('.', '_') + '.csv'
                if samp_type == 'temperature':
                    plain_response = response_suffwtemp(plain_tmp, match[0], match[1], sval)
                    story_response = response_suffwtemp(story_tmp, match[0], match[1], sval)
                else:
                    raise Exception("Sampling Type not supported: {}".format(samp_type))
                
                plain_answer = parse_enresponse(plain_response)
                story_answer = parse_enresponse(story_response)
                outdct = {}
                outdct['Match File'] = [match_file]
                outdct['Row No'] = [r_ind]
                outdct['Rep No'] = [i]
                outdct['Sampling Type'] = [samp_type]
                outdct['Sampling Param'] = [sval]
                outdct['Story Name'] = [story_name]
                outdct['Plain Response'] = [plain_response]
                outdct['Plain Answer'] = [plain_answer]
                outdct['Story Response'] = [story_response]
                outdct['Story Answer'] = [story_answer]
                outdct['Ground Truth'] = [row_gt]
                outdf = pd.DataFrame(outdct)
                outdf.to_csv(outname)

def storysuff(match_file, story_name, samp_range : list, samp_type, rows, match_prefix, num_reps=10):
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
    
    story_fname = 'all_prompts/' + story_name + 'prompt_english.pkl'
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    
    df = pd.read_csv(match_file)
    
    for r_ind in rows:
        dfrow = df.loc[r_ind]
        row_gt = dfrow['match']
        match = get_candidate(dfrow)
        for i in range(num_reps):
            for sval in samp_range:
                outname = 'matchwsuff' + match_prefix + story_name + '-' + str(r_ind) + '-' + 'rep' + str(i) + '-' + samp_type + str(sval).replace('.', '_') + '.csv'
                if samp_type == 'temperature':
                    story_response = response_suffwtemp(story_tmp, match[0], match[1], sval)
                else:
                    raise Exception("Sampling Type not supported: {}".format(samp_type))
                
                story_answer = parse_enresponse(story_response)
                outdct = {}
                outdct['Match File'] = [match_file]
                outdct['Row No'] = [r_ind]
                outdct['Rep No'] = [i]
                outdct['Sampling Type'] = [samp_type]
                outdct['Sampling Param'] = [sval]
                outdct['Story Name'] = [story_name]
                outdct['Story Response'] = [story_response]
                outdct['Story Answer'] = [story_answer]
                outdct['Ground Truth'] = [row_gt]
                outdf = pd.DataFrame(outdct)
                outdf.to_csv(outname)
    

def plain_vs_lang(match_file, lang_name, samp_range : list, samp_type, rows, num_reps=5):
    raise Exception("Not implemented")
    
def combine_results(match_file, story_name, folder):
    out_schema = ['Match File', 'Row No', 'Rep No', 'Sampling Type',
                  'Sampling Param', 'Story Name', 'Plain Prompt', 'Plain Response', 'Story Prompt', 'Story Response', 'Ground Truth']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    story_fname = 'all_prompts/' + story_name + 'prompt_english.pkl'
    plain_fname = 'all_prompts/plainprompt_english.pkl'
    matchdf = pd.read_csv(match_file)
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    with open(plain_fname, 'rb') as fh:
        plain_tmp = pickle.load(fh)
    for partf in os.listdir(folder):
        f = os.path.join(folder, partf)
        if f.endswith('.csv'):
            df = pd.read_csv(f)
            for row in df.to_dict(orient='records'):
                r_ind = row['Row No']
                dfrow = matchdf.loc[r_ind]
                match = get_candidate(dfrow)
                plain_prompt = plain_tmp.get_prompt(match[0], match[1])
                story_prompt = story_tmp.get_prompt(match[0], match[1])
                out_dct['Plain Prompt'].append(plain_prompt)
                out_dct['Story Prompt'].append(story_prompt)
                
                for c in df.columns:
                    out_dct[c].append(row[c])
    
    outdf = pd.DataFrame(out_dct)
    outdf.to_csv(folder + '_full.csv')

def combine_storyresults(match_file, story_name, folder):
    out_schema = ['Match File', 'Row No', 'Rep No', 'Sampling Type',
                  'Sampling Param', 'Story Name', 'Story Prompt', 'Story Response', 'Story Answer', 'Ground Truth']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    story_fname = 'all_prompts/' + story_name + 'prompt_english.pkl'
    matchdf = pd.read_csv(match_file)
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    for partf in os.listdir(folder):
        f = os.path.join(folder, partf)
        if f.endswith('.csv'):
            df = pd.read_csv(f)
            for row in df.to_dict(orient='records'):
                r_ind = row['Row No']
                dfrow = matchdf.loc[r_ind]
                match = get_candidate(dfrow)
                story_prompt = story_tmp.get_prompt(match[0], match[1])
                out_dct['Story Prompt'].append(story_prompt)
                
                for c in df.columns:
                    if c in out_dct:
                        out_dct[c].append(row[c])
    
    outdf = pd.DataFrame(out_dct)
    outdf.to_csv(folder + '_full.csv', index=False)

def extract_storyprompt(story_name, row0, row1):
    story_fname = 'all_prompts/' + story_name + 'prompt_english.pkl'
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    story_prompt = story_tmp.get_prompt(row0, row1)
    return story_prompt
    

def combine_storiesresults(match_file, folder):
    '''
    Combine all the csvs in 'folder' into one csv.
    The name of this csv will be 'folder_full.csv'

    Parameters
    ----------
    match_file : str
        Name of file containing candidates.
    folder : str
        Folder containing csvs to be combined.

    Returns
    -------
    None.

    '''
    
    out_schema = ['Match File', 'Row No', 'Rep No', 'Sampling Type',
                  'Sampling Param', 'Story Name', 'Story Prompt', 'Story Response', 'Story Answer', 'Ground Truth']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    matchdf = pd.read_csv(match_file)
    for partf in os.listdir(folder):
        f = os.path.join(folder, partf)
        if f.endswith('.csv'):
            df = pd.read_csv(f)
            for row in df.to_dict(orient='records'):
                r_ind = row['Row No']
                dfrow = matchdf.loc[r_ind]
                match = get_candidate(dfrow)
                story_prompt = extract_storyprompt(row['Story Name'], match[0], match[1])
                out_dct['Story Prompt'].append(story_prompt)
                
                for c in df.columns:
                    if c in out_dct:
                        out_dct[c].append(row[c])
    
    outdf = pd.DataFrame(out_dct)
    outdf.to_csv(folder + '_full.csv', index=False)

def count_garbage():
    garb_cnt = 0
    for f in os.listdir():
        if f.startswith('matchwsuff-') and f.endswith('.csv'):
            df = pd.read_csv(f)
            if -1 in df['Story Answer'].tolist():
                garb_cnt += 1
    
    print("Garbage Count: {}".format(garb_cnt))

def analyze_singlerep(fname, temp_level, rep_no):
    df = pd.read_csv(fname)
    stats_dct = {}
    
    for row_no in df['Row No'].unique():
        df_row = df[(df['Row No'] == row_no) & (df['Sampling Param'] == temp_level) & (df['Rep No'] == rep_no)]
        #there should only be one row.
        ans_lst = df_row['Story Answer'].to_list()
        gt_lst = df_row['Ground Truth'].to_list()
        if True in gt_lst and 1 in ans_lst:
            stats_dct[row_no] = 'tp'
        elif True in gt_lst and (0 in ans_lst or -1 in ans_lst):
            stats_dct[row_no] = 'fn'
        elif False in gt_lst and (0 in ans_lst or -1 in ans_lst):
            stats_dct[row_no] = 'tn'
        elif False in gt_lst and 1 in ans_lst:
            stats_dct[row_no] = 'fp'
    
    print("Stats: {}".format(stats_dct))

def crowd_gather(fullfname, temp, ditto_dct):
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
    raw_df = pd.read_csv(fullfname)
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
    
    res_schema = ['Match File', 'Row No', 'Vote', 'Ditto Answer', 'Ground Truth']
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
        
        mv_res['Ditto Answer'].append(ditto_dct[pair[1]])
        wawa_res['Ditto Answer'].append(ditto_dct[pair[1]])
        ds_res['Ditto Answer'].append(ditto_dct[pair[1]])
    
    mv_df = pd.DataFrame(mv_res)
    wawa_df = pd.DataFrame(wawa_res)
    ds_df = pd.DataFrame(ds_res)
    
    mv_df.to_csv('MajorityVote_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    wawa_df.to_csv('Wawa_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    ds_df.to_csv('DawidSkene_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)

def crowd_analysis(fullfname, temp, ditto_dct):
    raw_df = pd.read_csv(fullfname)
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
    mv_inst = MajorityVote()
    agg_mv = mv_inst.fit_predict(out_df)
    wawa_inst = Wawa()
    agg_wawa = wawa_inst.fit_predict(out_df)
    ds_inst = DawidSkene(n_iter=10)
    agg_ds = ds_inst.fit_predict(out_df)
    
    print(wawa_inst.skills_)
    wawa_probs = wawa_inst.fit_predict_proba(out_df)
    print(wawa_probs)
    
    mv_dct = agg_mv.to_dict()
    wawa_dct = agg_wawa.to_dict()
    ds_dct = agg_ds.to_dict()
    
    
    res_schema = ['Match File', 'Row No', 'Vote', 'Ditto Answer', 'Ground Truth']
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
        
        mv_res['Ditto Answer'].append(ditto_dct[pair[1]])
        wawa_res['Ditto Answer'].append(ditto_dct[pair[1]])
        ds_res['Ditto Answer'].append(ditto_dct[pair[1]])
    
    mv_df = pd.DataFrame(mv_res)
    wawa_df = pd.DataFrame(wawa_res)
    ds_df = pd.DataFrame(ds_res)
    
    mv_df.to_csv('MajorityVote_analysis' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    wawa_df.to_csv('Wawa_analysis' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    ds_df.to_csv('DawidSkene_analysis' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)

def perprompt_majorities(fullfname, temp):
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
    df = pd.read_csv(fullfname)
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
    out_df.to_csv('per_promptresults_tmperature' + str(temp).replace('.', '_') + '.csv', index=False)

def get_stats(method_names, temps, story_names):
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
    stat_schema = ['Method Name', 'Temperature', 'Ditto Precision', 'Ditto Recall',
                   'Ditto f1', 'Crowd Precision', 'Crowd Recall', 'Crowd f1']
    
    for sn in story_names:
        sprec = sn + ' Precision'
        srec = sn + ' Recall'
        sf = sn + ' f1'
        stat_schema += [sprec, srec, sf]
    
    stats_dct = {}
    for o in stat_schema:
        stats_dct[o] = []
    
    for mn in method_names:
        for tmp in temps:
            vote_file = mn + '_results-temperature' + str(tmp).replace('.', '_') + '.csv'
            df = pd.read_csv(vote_file)
            df['Vote_bool'] = (df['Vote'] == 1)
            
            ditto_tps = df[(df['Ditto Answer'] == df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            ditto_tns = df[(df['Ditto Answer'] == df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
            ditto_fps = df[(df['Ditto Answer'] != df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
            ditto_fns = df[(df['Ditto Answer'] != df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            
            our_tps = df[(df['Vote_bool'] == df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            our_tns = df[(df['Vote_bool'] == df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
            our_fps = df[(df['Vote_bool'] != df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
            our_fns = df[(df['Vote_bool'] != df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            
            ditto_precision = ditto_tps / (ditto_tps + ditto_fps)
            ditto_recall = ditto_tps / (ditto_tps + ditto_fns)
            ditto_f1 = 2 * (ditto_precision * ditto_recall) / (ditto_precision + ditto_recall)
            
            our_precision = our_tps / (our_tps + our_fps)
            our_recall = our_tps / (our_tps + our_fns)
            our_f1 = 2 * (our_precision * our_recall) / (our_precision + our_recall)
            
            stats_dct['Method Name'].append(mn)
            stats_dct['Temperature'].append(tmp)
            stats_dct['Ditto Precision'].append(ditto_precision)
            stats_dct['Ditto Recall'].append(ditto_recall)
            stats_dct['Ditto f1'].append(ditto_f1)
            stats_dct['Crowd Precision'].append(our_precision)
            stats_dct['Crowd Recall'].append(our_recall)
            stats_dct['Crowd f1'].append(our_f1)
            
            story_df = pd.read_csv('per_promptresults_tmperature' + str(tmp).replace('.', '_') + '.csv')
            for sn in story_names:
                sndf = story_df[story_df['Prompt'] == sn]
                ans_dct = {}
                for row in sndf.to_dict(orient='records'):
                    rowno = row['Row No']
                    snvote = row['Majority']
                    ans_dct[rowno] = snvote
                
                sn_tps = 0
                sn_tns = 0
                sn_fps = 0
                sn_fns = 0
                for rowno in ans_dct:
                    gt = df[df['Row No'] == rowno]['Ground Truth'].tolist()[0]
                    if gt == ans_dct[rowno] and gt == True:
                        sn_tps += 1
                    elif gt == ans_dct[rowno] and gt == False:
                        sn_tns += 1
                    elif gt != ans_dct[rowno] and gt == True:
                        sn_fns += 1
                    elif gt != ans_dct[rowno] and gt == False:
                        sn_fps += 1
                
                sn_precision = sn_tps / (sn_tps + sn_fps)
                sn_recall = sn_tps / (sn_tps + sn_fns)
                sn_f1 = 2 * (sn_precision * sn_recall) / (sn_precision + sn_recall)
                stats_dct[sn + ' Precision'].append(sn_precision)
                stats_dct[sn + ' Recall'].append(sn_recall)
                stats_dct[sn + ' f1'].append(sn_f1)
                
    
    stats_df = pd.DataFrame(stats_dct)
    stats_df.to_csv('dittovsmultiprompt_stats.csv', index=False)

def get_singlerepstats(fullfname):
    '''
    

    Parameters
    ----------
    fullfname : str
        The name of the full dataframe containing all chatGPT responses across different temperatures,
        prompt templates, entity resolution candidates, and prompt repetitions.

    Returns
    -------
    None. This writes the stats to disk.

    '''
    raise Exception("Not implemented")

def crowd_bytemp(method_name, tmp1, tmp2):
    '''
    For each crowd method, determine exactly which points change, causing
    the crowd method to improve as temperature increases. Note that
    the method name is only there for annotating purposes--we would have noticed
    significant shifts between temperatures for that method, which is why
    we want to compare responses between the two temperatures.

    Returns
    -------
    None.

    '''
    st_tmp1 = str(tmp1).replace('.', '_')
    st_tmp2 = str(tmp2).replace('.', '_')
    mfile1 = method_name + '_results-temperature' + st_tmp1 + '.csv'
    mfile2 = method_name + '_results-temperature' + st_tmp2 + '.csv'
    pfile1 = 'per_promptresults_tmperature' + st_tmp1 + '.csv'
    pfile2 = 'per_promptresults_tmperature' + st_tmp2 + '.csv'
    
    df1 = pd.read_csv(mfile1)
    df2 = pd.read_csv(mfile2)
    joindf = pd.merge(df1, df2, on='Row No')
    diffs = joindf[joindf['Vote_x'] != joindf['Vote_y']]
    diffrownos = diffs['Row No'].unique().tolist()
    
    
    pdf1 = pd.read_csv(pfile1)
    pdf2 = pd.read_csv(pfile2)
    pdiff1 = pdf1[pdf1['Row No'].isin(diffrownos)]
    pdiff2 = pdf2[pdf2['Row No'].isin(diffrownos)]
    sidebyside = pd.merge(pdiff1, pdiff2, on=['Row No', 'Prompt'])
    
    sidebyside.to_csv(method_name + '-temp' + st_tmp1 + '-to-temp' + st_tmp2 + '-comparison.csv')
        

def analyze_rsps_bytemp(fullfname, max_votes):
    '''
    Analyze responses by temperature to understand how/whether prompts affect
    response variability.

    Parameters
    ----------
    fullfname : str
        The name of the full dataframe containing all chatGPT responses across different temperatures,
        prompt templates, entity resolution candidates, and prompt repetitions.
    max_votes : int
        The maximum number of times any prompt could vote. This is equivalent to the
        number of times each prompt was repeated.

    Returns
    -------
    None.This writes our analysis to disk.

    '''
    out_schema = ['No Temp 0', 'No Temp 1', 'No Temp 2', 'Temp1 - Temp0', 'Temp0 - Temp1', 'Temp2 - Temp1', 'Temp1 - Temp2']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    tmp0df = pd.read_csv('per_promptresults_tmperature0_0.csv')
    tmp1df = pd.read_csv('per_promptresults_tmperature1_0.csv')
    tmp2df = pd.read_csv('per_promptresults_tmperature2_0.csv')
    
    tmp0same = tmp0df[(tmp0df['Yes Votes'] == max_votes) | (tmp0df['No Votes'] == max_votes)]
    tmp1same = tmp1df[(tmp1df['Yes Votes'] == max_votes) | (tmp1df['No Votes'] == max_votes)]
    tmp2same = tmp0df[(tmp2df['Yes Votes'] == max_votes) | (tmp2df['No Votes'] == max_votes)]
    
    out_dct['No Temp 0'].append(tmp0same.shape[0])
    out_dct['No Temp 1'].append(tmp1same.shape[0])
    out_dct['No Temp 2'].append(tmp2same.shape[0])
    
    #we only want to see whether the same prompts appear, so we should drop everything else
    tmp0samewma = tmp0same.drop(['Majority', 'Yes Votes', 'No Votes'], axis=1)
    tmp1samewma = tmp1same.drop(['Majority', 'Yes Votes', 'No Votes'], axis=1)
    tmp2samewma = tmp2same.drop(['Majority', 'Yes Votes', 'No Votes'], axis=1)
    
    #difference between temperature 0 and 1
    one_not0 = pd.merge(tmp1samewma, tmp0samewma, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    zero_not1 = pd.merge(tmp0samewma, tmp1samewma, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    two_not1 = pd.merge(tmp2samewma, tmp1samewma, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    one_not2 = pd.merge(tmp1samewma, tmp2samewma, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    #NOTE: we can always print the above if we want to see the actual prompt and row number counts.
    #for now, we will just add sizes
    out_dct['Temp1 - Temp0'].append(one_not0.shape[0])
    out_dct['Temp0 - Temp1'].append(zero_not1.shape[0])
    out_dct['Temp2 - Temp1'].append(two_not1.shape[0])
    out_dct['Temp1 - Temp2'].append(one_not2.shape[0])
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv('rsp_variation_stats.csv', index=False)

def results_folder(outfolder, match_prefix):
    '''
    Move the individual response csvs into one folder, called outfolder

    Parameters
    ----------
    outfolder : str
        Folder in which to place all csvs.
    match_prefix : str
        Name of the base file from which candidates were chosen (e.g., shoes.csv). This is part of the csv file name prefix

    Returns
    -------
    None.

    '''
    os.mkdir(outfolder)
    for f in os.listdir():
        if f.startswith('matchwsuff' + match_prefix) and f.endswith('.csv'):
            os.replace(f, os.path.join(outfolder, f))
        
    
    

if __name__=='__main__':
    #this is a representative row, 
    #in that the first is a true positive, next is false positive, next is true negative, next is false negative
    # rep_row = [23, 25, 46, 0, 29, 31]
    # rep_row = [2, 12, 30, 0, 1, 3, 46, 201, 302, 44, 136, 207]
    # ditto_dct = {2 : True, 12 : True, 30 : True, 0 : False, 1 : False, 3 : False, 46 : True, 201 : True, 302 : True, 44 : False, 136 : False, 207 : False}
    maindf = pd.read_csv('../ditto_erdata/shoes.csv')
    match_prefix = 'shoes'
    match_outfolder = 'shoesresults'
    ditto_dct = maindf['match'].to_dict()
    rep_row = ditto_dct.keys()
    
    # plain_vs_storysuff('../ditto_erdata/Amazon-Google.csv', 'detective', [0.0, 1.0, 2.0], 'temperature', rep_row)
    stories = ['veryplain', 'customer', 'journalist', 'security', 'layperson', 'detective']
    for s in stories:
        storysuff('../ditto_erdata/shoes.csv', s, [0.0, 1.0, 2.0], 'temperature', rep_row, match_prefix)
    # plain_vs_lang('chinese (traditional)', [0.0, 0.5, 0.9, 1.4], 'temperature', rep_row)
    # combine_results('detective', 'storytemp_sepclear')
    # for i in range(10):
    #     test_tempwresp(timeout=60)
    # count_garbage()
    # combine_storyresults('detective', 'nogarb')
    results_folder('../ditto_erdata/shoes.csv', match_outfolder, match_prefix)
    combine_storiesresults('../ditto_erdata/shoes.csv', match_outfolder)
    # analyze_singlerep('nogarb_full.csv', 2.0, 0)
    fullresult_file = match_outfolder + '_full.csv'
    crowd_gather(fullresult_file, 0.0, ditto_dct)
    crowd_gather(fullresult_file, 1.0, ditto_dct)
    crowd_gather(fullresult_file, 2.0, ditto_dct)
    # crowd_analysis('shoesresults_full.csv', 2.0, ditto_dct)
    perprompt_majorities(fullresult_file, 0.0)
    perprompt_majorities(fullresult_file, 1.0)
    perprompt_majorities(fullresult_file, 2.0)
    # get_stats('DawidSkene_results-temperature2_0.csv')
    # get_stats('MajorityVote_results-temperature2_0.csv')
    # get_stats('Wawa_results-temperature2_0.csv')
    
    method_names = ['MajorityVote', 'Wawa', 'DawidSkene']
    temps = [0.0, 1.0, 2.0]
    get_stats(method_names, temps, stories)
    # analyze_rsps_bytemp('shoesresults_full.csv', 5)
    # crowd_bytemp('Wawa', 1.0, 2.0)
    # crowd_bytemp('DawidSkene', 1.0, 2.0)
    
    
