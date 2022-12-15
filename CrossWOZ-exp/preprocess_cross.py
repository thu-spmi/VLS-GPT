import json,  os, re, copy, zipfile
import spacy
import ontology_cross as ontology
import utils
from collections import OrderedDict
from tqdm import tqdm
from config_cross import global_config as cfg
from db_ops_cross import CrossWozDB
from clean_dataset import clean_slot_values, clean_text, my_clean_text

  
class DataPreprocessor(object):
    def __init__(self):
        self.db = CrossWozDB(cfg.dbs) # load all processed dbs
        # data_path = 'data/multi-woz/annotated_user_da_with_span_full.json'
        data_path = 'data/CrossWOZ/data.json'
        self.convlab_data = json.loads(open(data_path).read().lower())
        # self.delex_sg_valdict_path = 'data/multi-woz-processed/delex_single_valdict.json'
        self.delex_base_path = 'data/CrossWOZ/ontology.json'
        self.delex_refs_path = 'data/CrossWoz-processed/reference_no.json'
        self.delex_refs = json.loads(open(self.delex_refs_path, 'r').read())
        self.get_name_set()


    def get_name_set(self):
        ontology_temp=json.loads(open(self.delex_base_path).read().lower())
        self.name_set={}
        self.domain_map={'餐馆':'restaurant', '景点':'attraction', '酒店':'hotel'}
        for domain in ['餐馆', '景点', '酒店']:
            self.name_set[domain]=ontology_temp[domain]['名称']

    def delex_by_valdict(self, text):
        text=re.sub(r'(\d.)?\d分', '[value_score]', text)
        for domain in self.name_set:
            for name in self.name_set[domain]:
                if name in text:
                    if '[nearby_' in text:
                        text=text.replace(name, '[nearby_{}]'.format(self.domain_map[domain]))
                    elif '[value_name]' in text:
                        text=text.replace(name, '[value_name]')
                    '''
                    if '周边' in text or '周围' in text or '附近' in text:
                        text=text.replace(name, '[nearby_{}]'.format(self.domain_map[domain]))
                    else:
                        text=text.replace(name, '[value_name]')
                    '''
        text=text.replace('\t', '')
        text=text.replace('\n', '')
        return text
    
    def delex_by_annotation(self, turn):
        resp=turn['content']
        for act in turn['dialog_act']:
            if act[3]!='' and act[3]!='none':
                if act[2]=='电话' and ' ' in act[3]:
                    for phone in act[3].split():
                        resp=resp.replace(phone, '[value_phone]')
                elif '酒店设施-' in act[2]:
                    value=act[2].split('-')[1]
                    resp=resp.replace(value, ontology.reqslot2placeholder['酒店设施'])
                else:
                    resp=resp.replace(act[3], ontology.reqslot2placeholder[act[2]])
        return resp

    def preprocess_main(self, save_path=None, is_test=False):
        """
        """
        data = {}
        count=0
        self.unique_da = {}
        ordered_sysact_dict = {}
        no_user_count=0
        # yyy
        for fn, raw_dial in tqdm(list(self.convlab_data.items())):
            #if fn!='12333':
             #   continue
            count +=1

            compressed_goal = {} # for every dialog, keep track the goal, domains, requests
            compressed_final_goal = {}
            goals= raw_dial['final_goal']
            dial_domains, dial_reqs = [], []
            for goal in goals:
                dom=goal[1]
                if dom in compressed_final_goal:
                    compressed_final_goal[dom].append(goal)
                else:
                     compressed_final_goal[dom]=[]
                     compressed_final_goal[dom].append(goal)
            goals= raw_dial['goal']
            dial_domains, dial_reqs = [], []
            for goal in goals:
                dom=goal[1]
                if dom in compressed_goal:
                    compressed_goal[dom].append(goal)
                else:
                     compressed_goal[dom]=[]
                     compressed_goal[dom].append(goal)
                if dom in ontology.all_domains_cross:
                    dial_domains.append(dom)
            
            for dial_turn in raw_dial['messages']:
                if dial_turn['role']=='usr': # user's turn
                    for quadra in dial_turn['dialog_act']:
                        if(quadra[0]=='request'):
                            dial_reqs.append(quadra[2])    

            dial_reqs = list(set(dial_reqs))

            dial = {'goal': compressed_goal,'final_goal': compressed_final_goal, 'log': []}
            single_turn = {}
            constraint_dict = OrderedDict()
            prev_constraint_dict = {}
            prev_turn_domain = ['general']
            ordered_sysact_dict[fn] = {}

            #for turn_num, dial_turn in enumerate(raw_dial['log']):
            for dial_turn in raw_dial['messages']:
                if  dial_turn['role']=='usr':
                    single_turn['user'] = dial_turn['content']
                    #single_turn['user_delex'] = self.delex_by_valdict(dial_turn['content'])
                else: # system
                    dial_state =dial_turn['sys_state']
                    s_delex=self.delex_by_annotation(dial_turn)
                    s_delex=self.delex_by_valdict(s_delex)
                    #s_delex= self.delex_by_valdict(dial_turn['content'])
                    single_turn['resp'] = s_delex
                    single_turn['nodelx_resp'] = dial_turn['content'].replace('(\t)|(\n)', '')
                    # get belief state, semi=informable/book=requestable, put into constraint_dict
                    for domain in dial_domains:
                        if not constraint_dict.get(domain):
                            constraint_dict[domain] = OrderedDict()
                        info_sv = dial_state[domain]
                        for s,v in info_sv.items():
                            #s,v = clean_slot_values(domain, s,v)
                            if s=='selectedresults':
                                for vtemp in v:
                                    if vtemp != '':
                                        constraint_dict[domain][s] = vtemp
                            else:
                                if v != '':
                                    constraint_dict[domain][s] = v
                    constraints = [] # list in format of [domain] slot value
                    turn_dom_bs = []
                    for domain, info_slots in constraint_dict.items():
                        if info_slots:
                            constraints.append('['+domain+']')
                            for slot, value in info_slots.items():
                                if slot!='selectedresults':
                                    constraints.append(slot)
                                    constraints.extend(value.split())
                            if domain not in prev_constraint_dict:
                                turn_dom_bs.append(domain)
                            elif prev_constraint_dict[domain] != constraint_dict[domain]:
                                turn_dom_bs.append(domain)


                    sys_act_dict = {}
                    turn_dom_da = set()
                    for act in dial_turn['dialog_act']:
                        #d, a = act.split('-') # split domain-act
                        turn_dom_da.add(act[1])
                    turn_dom_da = list(turn_dom_da)#greet can be removed from turn_dom_da
                    #if len(turn_dom_da) != 1 and 'general' in turn_dom_da:
                    #    turn_dom_da.remove('general')
                    #if len(turn_dom_da) != 1 and 'booking' in turn_dom_da:
                    #    turn_dom_da.remove('booking')
                    # get turn domain
                    turn_domain = turn_dom_bs
                    for dom in turn_dom_da:
                        if dom != 'booking' and dom not in turn_domain:
                            turn_domain.append(dom)
                    if not turn_domain:
                        turn_domain = prev_turn_domain
                    if len(turn_domain) == 2 and 'greeting' in turn_domain:
                        turn_domain.remove('greeting')
                    if len(turn_domain) == 2:
                        if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                            turn_domain = turn_domain[::-1]

                    # get system action
                    for dom in turn_domain:
                        sys_act_dict[dom] = {}
                    add_to_last_collect = []
                    booking_act_map = {'inform': 'offerbook', 'book': 'offerbooked'}
                    for act in dial_turn['dialog_act']:
                        if act[0] == 'general':
                            continue
                        d = act[1]
                        a = act[0]
                        if d == 'general' and d not in sys_act_dict:
                            sys_act_dict[d] = {}
                        if d == 'booking':
                            d = turn_domain[0]
                            a = booking_act_map.get(a, a)
                        add_p = []
                        p=act[2]
                        if p == 'none':
                            continue
                        #elif ontology.da_abbr_to_slot_name.get(p):
                        #    p = ontology.da_abbr_to_slot_name[p]
                        if p not in add_p:
                            add_p.append(p)
                        add_to_last = True if a in ['request', 'reqmore', 'bye', 'offerbook'] else False
                        if add_to_last:
                            add_to_last_collect.append((d,a,add_p))
                        else:
                            sys_act_dict[d][a] = add_p
                    for d, a, add_p in add_to_last_collect:
                        sys_act_dict[d][a] = add_p

                    for d in copy.copy(sys_act_dict):
                        acts = sys_act_dict[d]
                        if not acts:
                            del sys_act_dict[d]
                        if 'inform' in acts and 'offerbooked' in acts:
                            for s in sys_act_dict[d]['inform']:
                                sys_act_dict[d]['offerbooked'].append(s)
                            del sys_act_dict[d]['inform']


                    ordered_sysact_dict[fn][len(dial['log'])] = sys_act_dict

                    sys_act = []
                    if dial_turn['dialog_act']:
                        if 'greet' in dial_turn['dialog_act'][0]:
                            sys_act.extend(['[general]', '[greet]'])
                    for d, acts in sys_act_dict.items():
                        sys_act += ['[' + d + ']']
                        for a, slots in acts.items():
                            self.unique_da[d+'-'+a] = 1
                            sys_act += ['[' + a + ']']
                            sys_act += slots


                    # get db pointers
                    #self.db object
                    matnums = self.db.get_match_num(constraint_dict,act)
                    for domain in turn_domain:
                        if domain!='greet':
                            match_dom = domain
                    match = matnums[match_dom]
                    dbvec = self.db.addDBPointer(match_dom, match)
                    #bkvec = self.db.addBookingPointer(dial_turn['dialog_act'])
                    single_turn['pointer'] = ','.join([str(d) for d in dbvec ]) # 4 database pointer for domains, 2 for booking + bkvec
                    single_turn['match'] = str(match)
                    single_turn['constraint'] = ' '.join(constraints)
                    single_turn['sys_act'] = ' '.join(sys_act)
                    single_turn['turn_num'] = len(dial['log'])
                    single_turn['turn_domain'] = ' '.join(['['+d+']' for d in turn_domain])

                    prev_turn_domain = copy.deepcopy(turn_domain)
                    prev_constraint_dict = copy.deepcopy(constraint_dict)
                    if 'user' in single_turn:
                        dial['log'].append(single_turn)
                    else:
                        no_user_count+=1

                    single_turn = {}


            data[fn] = dial
        print('Turns without user:', no_user_count)
        with open('data/CrossWoz-analysis/dialog_acts.json', 'w') as f:
            json.dump(ordered_sysact_dict, f, indent=2,ensure_ascii=False)
        with open('data/CrossWoz-analysis/dialog_act_type.json', 'w') as f:
            json.dump(self.unique_da, f, indent=2,ensure_ascii=False)
        return data


if __name__=='__main__':
    if not os.path.exists('data/CrossWoz-processed'):
        os.mkdir('data/CrossWoz-processed')
    dh = DataPreprocessor()
    data = dh.preprocess_main()
    

    with open('data/CrossWoz-processed/data_for_damd.json', 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

