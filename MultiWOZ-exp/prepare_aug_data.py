import json
from lib2to3.pgen2 import token
import re, os
s_set=['area', 'child', 'requirement', 'location', 'constraint', 'hour', 'specification', 'rating', 'hotel', 
       'place', 'kid', 'price', 'breakfast', 'venue', 'restaurant', 'range', 'include', 'meal']
ly_set=['moderate', 'expensive', 'cheap']
er_set=['cheap']

from transformers import GPT2Tokenizer
from reader import MultiWozReader
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
reader=MultiWozReader(tokenizer)
def process_data(data):
    new_data={}
    for sent in data:
        new_sent=sent.strip(r"'").strip().lower()
        new_sent=re.sub(r"(\S)([,'\.\?\!])", r"\1 \2", new_sent)
        for w in s_set:
            new_sent=new_sent.replace(w+'s', w+' -s')
        for w in ly_set:
            new_sent=new_sent.replace(w+'ly', w+' -ly')
        for w in er_set:
            new_sent=new_sent.replace(w+'er', w+' -er')
        #new_sent=re.sub(r'- (\S)', r'-\1', new_sent)
        new_data[new_sent]=[]
        for variant in data[sent]:
            new_var=variant.strip(r"'").strip().lower()
            new_var=re.sub(r"(\S)([,'\.\?\!])", r"\1 \2", new_var)
            #new_var=re.sub(r'- (\S)', r'-\1', new_var)
            new_data[new_sent].append(new_var)
    return new_data

if __name__=='__main__':
    data_path='data/multi-woz-2.1-processed/processed_aug_data.json'
    if os.path.exists((data_path)):
        processed_utters=json.load(open(data_path, 'r'))
    else:
        aug_utters=json.load(open('data/multi-woz-2.1-processed/aug_data.json', 'r'))
        processed_utters=process_data(aug_utters)
        json.dump(processed_utters, open(data_path, 'w'), indent=2)
    raw_data=json.load(open('data/multi-woz-2.1-processed/data_for_damd.json', 'r'))
    count=0
    for dial in reader.train:
        dial_id=dial[0]['dial_id']+'.json'
        for idx, turn in enumerate(dial):
            raw_turn=raw_data[dial_id]['log'][idx]
            if raw_turn['user'] not in processed_utters:
                continue
            turn['user_variants']=[]
            for user_var in processed_utters[raw_turn['user']]:
                encoded_user_var=reader.modified_encode('<sos_u> ' + user_var + ' <eos_u>')
                turn['user_variants'].append(encoded_user_var)
            count+=1
    print('Augmented turns:', count)
    encoded_data={
        'train':reader.train,
        'dev':reader.dev,
        'test':reader.test
    }
    print(len(reader.train), len(reader.dev), len(reader.test))
    json.dump(encoded_data, open('data/multi-woz-2.1-processed/encoded_data.json', 'w'), indent=2)

