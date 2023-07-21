import pandas as pd
import pickle
'''
Purpose: define classes to make mass prompt generation easier.
'''

class IntegrationPrompt:
    def __init__(self, preamble, c1sentence, c2sentence, question):
        self.preamble = preamble
        self.c1sentence = c1sentence
        self.c2sentence = c2sentence
        self.question = question
        self.lang = 'english'
    
    def get_prompt(self, candidate1, candidate2, include_preamble=True):
        full_st = self.c1sentence + '\n' + candidate1 + '\n\n' + self.c2sentence + '\n' + candidate2 + '\n\n' + self.question
        if include_preamble:
            full_st = self.preamble + '\n\n' + full_st
        return full_st

    def get_chat(self, candidate1, candidate2, examples, followup):
        chat = []
        include_preamble = True
        for example in examples:
            message = self.get_prompt(example[0], example[1], include_preamble=include_preamble) + ' ' + followup
            chat.append({"role": "user", "content": message})
            include_preamble = False
            chat.append({"role": "assistant", "content": example[2]})
        message = self.get_prompt(candidate1, candidate2, include_preamble=include_preamble) + ' ' + followup
        chat.append({"role": "user", "content": message})
        return chat

    
    def get_prompt_nospaces(self, candidate1, candidate2):
        full_st = self.preamble + self.c1sentence + candidate1 + self.c2sentence + candidate2 + self.question
        return full_st

def generate_detectivetemp():
    preamble = 'Suppose you\'re a detective who has been hired to detect illegal copies of products so that law enforcement can fine those selling these copies.'
    c1sentence = 'Suppose you are given the following information about a product that a business may be trying to sell illegally:'
    c2sentence = 'You confront the storefront owner potentially selling the illegal copy. You find the following information:'
    # question = 'Would you charge the storefront owner for selling an illegal copy of the product?'
    question = 'Based on the available information, is it likely that the storefront owner is selling an illegal copy of the product?'
    detprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return detprompt

def generate_detectiveprompt(er_row1, er_row2):
    preamble = 'Suppose you\'re a detective who has been hired to detect illegal copies of products so that law enforcement can fine those selling these copies.'
    c1sentence = 'Suppose you are given the following information about a product that a business may be trying to sell illegally:'
    c2sentence = 'You confront the storefront owner potentially selling the illegal copy. You find the following information:'
    # question = 'Would you charge the storefront owner for selling an illegal copy of the product?'
    question = 'Based on the available information, is it likely that the storefront owner is selling an illegal copy of the product?'
    detprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return detprompt.get_prompt(er_row1, er_row2)

def generate_laypersontemp():
    preamble = 'I\'m a father with a kid for whom I’m trying to find a gift.'
    c1sentence = 'I\'ve been wandering around the store all day looking for the following product for my kid:'
    c2sentence = 'I found a product that seems like what I want:'
    question = 'But I can\'t tell. Is this the product I\'m looking for?'
    layprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return layprompt

def generate_layperson(er_row1, er_row2):
    preamble = 'I\'m a father with a kid for whom I’m trying to find a gift.'
    c1sentence = 'I\'ve been wandering around the store all day looking for the following product for my kid:'
    c2sentence = 'I found a product that seems like what I want:'
    question = 'But I can\'t tell. Is this the product I\'m looking for?'
    layprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return layprompt.get_prompt(er_row1, er_row2)

def generate_customertemp():
    preamble = 'Suppose you are an employee who works at the customer support division of a large company.'
    preamble += ' You receive a complaint from a customer demanding a refund because the product label information did not match the actual product they received.'
    c1sentence = 'This is the product label information in the customer\'s claim:'
    c2sentence = 'Upon inspection of the product the customer purchased, you find the following information:'
    question = 'Do you owe the customer a refund?'
    csprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return csprompt

def generate_customer(er_row1, er_row2):
    preamble = 'Suppose you are an employee who works at the customer support division of a large company.'
    preamble += ' You receive a complaint from a customer demanding a refund because the product label information did not match the actual product they received.'
    c1sentence = 'This is the product label information in the customer\'s claim:'
    c2sentence = 'Upon inspection of the product the customer purchased, you find the following information:'
    question = 'Do you owe the customer a refund?'
    csprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return csprompt.get_prompt(er_row1, er_row2)

def generate_securitytemp():
    preamble = 'I am a computer security expert for a large e-commerce company.'
    preamble += ' It\'s very common for our users to spoof pages of existing products to scam unsuspecting customers. It\'s killing our business!'
    c1sentence = 'I found two product pages with the following product information:'
    c2sentence = 'and'
    question = 'Are these pages talking about the same product? If so, one of them is spoofed...'
    secprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return secprompt

def generate_security(er_row1, er_row2):
    preamble = 'I am a computer security expert for a large e-commerce company.'
    preamble += ' It\'s very common for our users to spoof pages of existing products to scam unsuspecting customers. It\'s killing our business!'
    c1sentence = 'I found two product pages with the following product information:'
    c2sentence = 'and'
    question = 'Are these pages talking about the same product? If so, one of them is spoofed...'
    secprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return secprompt.get_prompt(er_row1, er_row2)

def generate_journalisttemp():
    preamble = 'I’m a journalist doing a piece on the dwindling importance of product descriptions in distinguishing two products without seeing them in-person.'
    c1sentence = 'As an example, I wanted to have the following product descriptions that might be different, but represent the same product:'
    c2sentence = 'and'
    question = 'Is this an effective example of product descriptions that might look different, but describe the same product?'
    jprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return jprompt

def generate_journalist(er_row1, er_row2):
    preamble = 'I’m a journalist doing a piece on the dwindling importance of product descriptions in distinguishing two products without seeing them in-person.'
    c1sentence = 'As an example, I wanted to have the following product descriptions that might be different, but represent the same product:'
    c2sentence = 'and'
    question = 'Is this an effective example of product descriptions that might look different, but describe the same product?'
    jprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return jprompt.get_prompt(er_row1, er_row2)

def generate_baselinetemp():
    preamble = 'We are trying to integrate product data from two different databases. The goal is to look at two product entries, one from each database, and determine whether the two entries refer to the same product or not. Since the databases are different, there will still be some differences between entries that refer to the same product.'
    c1sentence = 'Here is an entry from the first database:'
    c2sentence = 'Here is an entry from the second database:'
    question = 'As best as you can tell, do these entries refer to the same product?'
    baselineprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return baselineprompt

def generate_baseline_negative_temp():
    preamble = 'We are trying to integrate product data from two different databases. The goal is to look at two product entries, one from each database, and determine whether the two entries refer to the same product or not. Since the databases are different, there will still be some differences between entries that refer to the same product.'
    c1sentence = 'Here is an entry from the first database:'
    c2sentence = 'Here is an entry from the second database:'
    question = 'False positives are worse than false negatives - only say YES if you are pretty confident. As best as you can tell, do these entries refer to the same product?'
    baselineprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return baselineprompt

def generate_baseline_positive_temp():
    preamble = 'We are trying to integrate product data from two different databases. The goal is to look at two product entries, one from each database, and determine whether the two entries refer to the same product or not. Since the databases are different, there will still be some differences between entries that refer to the same product.'
    c1sentence = 'Here is an entry from the first database:'
    c2sentence = 'Here is an entry from the second database:'
    question = 'False negatives are worse than false positives - only say NO if you are pretty confident. As best as you can tell, do these entries refer to the same product?'
    baselineprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return baselineprompt

def generate_baseline_front_temp():
    preamble = 'We are trying to integrate product data from two different databases. The goal is to look at two product entries, one from each database, and determine whether the two entries refer to the same product or not. Since the databases are different, there will still be some differences between entries that refer to the same product.'
    c1sentence = 'Here is an entry from the first database:'
    c2sentence = 'Here is an entry from the second database:'
    question = 'Pay more attention to the first half of each entry - it is more informative. As best as you can tell, do these entries refer to the same product?'
    baselineprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return baselineprompt

def generate_baseline_back_temp():
    preamble = 'We are trying to integrate product data from two different databases. The goal is to look at two product entries, one from each database, and determine whether the two entries refer to the same product or not. Since the databases are different, there will still be some differences between entries that refer to the same product.'
    c1sentence = 'Here is an entry from the first database:'
    c2sentence = 'Here is an entry from the second database:'
    question = 'Pay more attention to the second half of each entry - it is more informative. As best as you can tell, do these entries refer to the same product?'
    baselineprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return baselineprompt

def generate_plaintemp():
    preamble = 'Consider the following pair of csv file rows:'
    c1sentence = ''
    c2sentence = 'and'
    question = 'Can you confirm if this pair of rows is a match after performing entity resolution on two tables?'
    plainprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return plainprompt

def generate_veryplaintemp():
    preamble = ''
    c1sentence = ''
    c2sentence = '\n###\n'
    question = 'Same product?'
    veryplainprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return veryplainprompt

def generate_veryplain(er_row1, er_row2):
    preamble = ''
    c1sentence = ''
    c2sentence = '\n###\n'
    question = 'Same product?'
    veryplainprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return veryplainprompt.get_prompt(er_row1, er_row2)

TEMPLATES = {
    'detective': generate_detectivetemp(),
    'customer': generate_customertemp(),
    'layperson': generate_laypersontemp(),
    'security': generate_securitytemp(),
    'journalist': generate_journalisttemp(),
    'baseline': generate_baselinetemp(),
    'baselinenegative': generate_baseline_negative_temp(),
    'baselinepositive': generate_baseline_positive_temp(),
    'baselinefront': generate_baseline_front_temp(),
    'baselineback': generate_baseline_back_temp(),
    'plain': generate_plaintemp(),
    'veryplain': generate_veryplaintemp(),
}

if __name__=='__main__':
    '''
    df = pd.read_csv('../ditto_erdata/Amazon-Google.csv')
    leftrow = df.loc[0]['left']
    rightrow = df.loc[0]['right']
    leftrow = leftrow.replace('\n', '\t')
    leftrow = ' "' + leftrow + '".'
    rightrow = rightrow.replace('\n', '\t')
    rightrow = ' "' + rightrow + '".'
    '''
    
    # print(generate_detectiveprompt(leftrow, rightrow))
    # print(generate_layperson(leftrow, rightrow))
    # print(generate_customer(leftrow, rightrow))
    # print(generate_security(leftrow, rightrow))
    # print(generate_journalist(leftrow, rightrow))
    # print(generate_veryplain(leftrow, rightrow))
    det_temp = generate_detectivetemp()
    lay_temp = generate_laypersontemp()
    cust_temp = generate_customertemp()
    sec_temp = generate_securitytemp()
    j_temp = generate_journalisttemp()
    baseline_temp = generate_baselinetemp()
    plain_temp = generate_plaintemp()
    veryplain_temp = generate_veryplaintemp()
    all_temps = [det_temp, cust_temp, lay_temp, sec_temp, j_temp, baseline_temp, plain_temp, veryplain_temp]
    temp_names = ['detective', 'customer', 'layperson', 'security', 'journalist', 'baseline', 'plain', 'veryplain']
    for i,tmp in enumerate(all_temps):
        fname = 'all_prompts/' + temp_names[i] + 'prompt_english.pkl'
        with open(fname, 'wb') as fh:
            pickle.dump(tmp, file=fh)
