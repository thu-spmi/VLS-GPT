import os, json, copy, re, zipfile
from collections import OrderedDict
from ontology_cross import all_domains_cross


# 2.0
#data_path = 'data/multi-woz/'
#save_path = 'data/multi-woz-analysis/'
#save_path_exp = 'data/multi-woz-processed/'
#CrossWoz
data_path = 'data/CrossWOZ/'
save_path = 'data/CrossWoz-analysis/'
save_path_exp = 'data/CrossWoz-processed/'
data_file = 'train.json'
domains = all_domains_cross
# all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']
#diatype=['独立多领域', '不独立多领域', '独立多领域+交通', '单领域', '不独立多领域+交通']
def analysis():
    compressed_raw_data = {}#-metadata
    goal_of_dials = {}
    req_slots = {}
    info_slots = {}
    dom_count = {}
    dom_fnlist = {}
    domain_value=[]
    all_domain_specific_slots = set()
    for domain in domains:
        req_slots[domain] = []
        info_slots[domain] = []

    #archive = zipfile.ZipFile(data_path+data_file+'.zip', 'r')
    data = open(data_path+data_file, 'r').read().lower()
    ref_nos = list(set(re.findall(r'\"reference\"\: \"(\w+)\"', data)))
    data = json.loads(data)
    #for fn, dial in data.items():
    #    if dial['type'] not in diatype:
    #        diatype.append(dial['type'])

    for fn, dial in data.items():
        goals = dial['goal']
        
        logs = dial['messages']

        # get compressed_raw_data and goal_of_dials
        compressed_raw_data[fn] = {'goal': {}, 'log': []}
        goal_of_dials[fn] = {}
        #for dom, goal in goals.items(): # get goal of domains that are in demmand
        for goal in goals:
            dom=goal[1]
            state=goal[2:]
            compressed_raw_data[fn]['goal'][dom] = state
            goal_of_dials[fn][dom] = state

        for turn in logs:
            if turn['role']=='usr': # user's turn
                compressed_raw_data[fn]['log'].append({'text': turn['content']})
                for quadra in turn['dialog_act']:
                    if(quadra[0]=='request'):
                        domain=quadra[1]
                        req_s =quadra[2]
                        if req_s not in req_slots[domain]:
                            req_slots[domain]+= [req_s]
                    
            else: # system's turn
                compressed_raw_data[fn]['log'].append({'text': turn['content']})
                for domain in domains:
                    info_ss = turn['sys_state'][domain]
                    for info_s in info_ss:
                        all_domain_specific_slots.add(domain+'-'+info_s)
                        if info_s not in info_slots[domain]:
                            info_slots[domain]+= [info_s]

                    
                    
                """
                meta = turn['metadata']
                turn_dict = {'text': turn['text'], 'metadata': {}}
                for dom, book_semi in meta.items(): # for every domain, sys updates "book" and "semi"
                    book, semi = book_semi['book'], book_semi['semi']
                    record = False
                    for slot, value in book.items(): # record indicates non-empty-book domain
                        if value not in ['', []]:
                            record = True
                    if record:
                        turn_dict['metadata'][dom] = {}
                        turn_dict['metadata'][dom]['book'] = book # add that domain's book
                    record = False
                    for slot, value in semi.items(): # here record indicates non-empty-semi domain
                        if value not in ['', []]:
                            record = True
                            break
                    if record:
                        for s, v in copy.deepcopy(semi).items():
                            if v == 'not mentioned':
                                del semi[s]
                        if not turn_dict['metadata'].get(dom):
                            turn_dict['metadata'][dom] = {}
                        turn_dict['metadata'][dom]['semi'] = semi # add that domain's semi
                compressed_raw_data[fn]['log'].append(turn_dict) # add to log the compressed turn_dict
                """

            # get domain statistics
        dial_type = dial['type'] # determine the dialog's type:5 choices
        dial_domains=[]
        for goal in goals:
            if goal[1] not in dial_domains:
                 dial_domains.append(goal[1])
        #get multi-domain
        dom_str = ''
        for dom in dial_domains: 
            if not dom_count.get(dom+'_'+dial_type): # count each domain type, with single or multi considered
                dom_count[dom+'_'+dial_type] = 1
            else:
                dom_count[dom+'_'+dial_type] += 1
            if not dom_fnlist.get(dom+'_'+dial_type): # keep track the file number of each domain type
                dom_fnlist[dom+'_'+dial_type] = [fn]
            else:
                dom_fnlist[dom+'_'+dial_type].append(fn)
            dom_str += '%s_'%dom
        dom_str = dom_str[:-1] # substract the last char in dom_str
        
        if dial_type!='单领域': # count multi-domains
            if not dom_count.get(dom_str):
                dom_count[dom_str] = 1
            else:
                dom_count[dom_str] += 1
            if not dom_fnlist.get(dom_str):
                dom_fnlist[dom_str] = [fn]
            else:
                dom_fnlist[dom_str].append(fn)
            ######

            # get informable and requestable slots statistics
        

    # result statistics
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_exp):
        os.mkdir(save_path_exp)
    with open(save_path+'req_slots.json', 'w') as sf:
        json.dump(req_slots,sf,indent=2,ensure_ascii=False)#add ensure_ascii=False to show Chinese
    with open(save_path+'info_slots.json', 'w') as sf:
        json.dump(info_slots,sf,indent=2,ensure_ascii=False)
    with open(save_path+'all_domain_specific_info_slots.json', 'w') as sf:
        json.dump(list(all_domain_specific_slots),sf,indent=2,ensure_ascii=False)
        print("slot num:", len(list(all_domain_specific_slots)))
    with open(save_path+'goal_of_each_dials.json', 'w') as sf:
        json.dump(goal_of_dials, sf, indent=2,ensure_ascii=False)
    with open(save_path+'compressed_data.json', 'w') as sf:
        json.dump(compressed_raw_data, sf, indent=2)#ensure_ascii=False
    with open(save_path + 'domain_count.json', 'w') as sf:
        single_count = [d for d in dom_count.items() if 'single' in d[0]]
        multi_count = [d for d in dom_count.items() if 'multi' in d[0]]
        other_count = [d for d in dom_count.items() if 'multi' not in d[0] and 'single' not in d[0]]
        dom_count_od = OrderedDict(single_count+multi_count+other_count)
        json.dump(dom_count_od, sf, indent=2,ensure_ascii=False)
    with open(save_path_exp + 'reference_no.json', 'w') as sf:
        json.dump(ref_nos,sf,indent=2)
    with open(save_path_exp + 'domain_files.json', 'w') as sf:
        json.dump(dom_fnlist, sf, indent=2,ensure_ascii=False)


if __name__ == '__main__':
    analysis()
