import numpy as np
import os
import csv
import random
import logging
import json
import spacy
import utils
import ontology_cross as ontology
from copy import deepcopy
from collections import OrderedDict
from db_ops_cross import CrossWozDB
from transformers import BertTokenizer
from config_cross import global_config as cfg
from util.crosswoz.dbquery import list2dict
from dst import parser_bs

class _ReaderBase(object):

    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = None
        self.db = None
        self.set_stats = {}

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            #修改记录：对于turn_len=0情况的处理
            if turn_len==0:
                continue
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5:
                del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == cfg.batch_size:
                # print('batch size: %d, batch num +1'%(len(batch)))
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        # print('last batch size: %d, batch num +1'%(len(batch)))
        if (len(batch) % len(cfg.cuda_device)) != 0:
            batch = batch[:-(len(batch) % len(cfg.cuda_device))]
        #修改记录：直接将剩下的对话作为新batch
        if len(batch)>0:
            all_batches.append(batch)
        '''
        if len(batch) > 0.5 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        '''
        return all_batches

    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = []
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialog = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if idx_in_batch>=len(v_list):
                        print('list out of range',key, v_list)
                        continue
                    value = v_list[idx_in_batch]
                    dial_turn[key] = value
                dialog.append(dial_turn)
            dialogs.append(dialog)
        return dialogs

    def inverse_transpose_turn(self, turn_list):
        """
        eval, one dialog at a time
        """
        dialogs = {}
        turn_num = len(turn_list)
        dial_id = turn_list[0]['dial_id']
        dialogs[dial_id] = []
        for turn_idx in range(turn_num):
            dial_turn = {}
            turn = turn_list[turn_idx]
            for key, value in turn.items():
                if key=='dial_id':
                    continue
                if key == 'pointer' and self.db is not None:
                    turn_domain = turn['turn_domain'][-1]
                    value = self.db.pointerBack(value, turn_domain)
                dial_turn[key] = value
            dialogs[dial_id].append(dial_turn)
        return dialogs

    def get_eval_data(self, set_name='dev'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]

        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_turns = 0
        num_dials = len(dial)
        for d in dial:
            num_turns += len(d)

        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials

        return dial
        
    def get_batches(self, set_name,data=None):
        #修改记录：新增参数data
        """
        compute dataset stats.
        """
        global dia_count
        log_str = ''

        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        if data:
            dial=data
        if cfg.low_resource and set_name == 'train':
            # dial = random.sample(dial, int(len(dial)*0.01))
            dial = random.sample(dial, 100)
            logging.info('Low Resource setting, finetuning size: {}'.format(len(dial)))
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []

        
        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_training_steps = 0
        num_turns = 0
        num_dials = 0

        for k in turn_bucket:
            if set_name != 'test' and k == 1 or k >= 17:
                continue
            batches = self._construct_mini_batch(turn_bucket[k])
            if len(batches)==0:
                continue
            log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n" % (
                k, len(turn_bucket[k]), len(batches), len(batches[-1]))
            # print("turn num:%d, dial num:v%d, batch num: %d, "%(k, len(turn_bucket[k]), len(batches)))
            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches
        log_str += 'total batch num: %d\n' % len(all_batches)
        # print('total batch num: %d'%len(all_batches))
        # print('dialog count: %d'%dia_count)
        # return all_batches

        # log stats
        # logging.info(log_str)
        # cfg.num_training_steps = num_training_steps * cfg.epoch_num
        self.set_stats[set_name]['num_training_steps_per_epoch'] = num_training_steps
        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials
        if data is None:
            if set_name == 'train':
                random.shuffle(all_batches)
        return all_batches
    
    def get_nontranspose_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield batch

    def get_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield self.transpose_batch(batch)

    def save_result(self, write_mode, results, field, write_title=False,result_name=None):
        #修改记录：修改结果文件存储路径
        result_name=result_name if result_name is not None else 'result.csv'
        with open(os.path.join(cfg.eval_load_path,result_name), write_mode) as rf:
            if write_title:
                rf.write(write_title+'\n')
            writer = csv.DictWriter(rf, fieldnames=field)
            writer.writeheader()
            writer.writerows(results)
        return None
    
    def load_result(self,result_path):
        # 读取result1.csv，第一行为格式说明，
        results=[]
        field=[]
        with open(result_path, 'r') as rf:
            reader=csv.reader(rf)
            for n,line in enumerate(reader):
                entry={}
                if line==['DECODED RESULTS:']:
                    continue
                elif field==[]:
                    field=line
                else:
                    for i,key in enumerate(field):
                        entry[key]=line[i]
                    results.append(entry)
        return results,field


    def save_result_report(self, results):
        # if 'joint_goal' in results[0]:
        #     with open(cfg.result_path[:-4] + '_report_dst.txt', 'w') as rf:
        #         rf.write('joint goal\tslot_acc\tslot_f1\tact_f1\n')
        #         for res in results:
        #             a,b,c,d = res['joint_goal'], res['slot_acc'], res['slot_f1'], res['act_f1']
        #             rf.write('%2.1f\t%2.1f\t%2.1f\t%2.1f\n'%(a,b,c,d))
        # elif 'joint_goal_delex' in results[0]:
        #     with open(cfg.result_path[:-4] + '_report_bsdx.txt', 'w') as rf:
        #         rf.write('joint goal\tslot_acc\tslot_f1\tact_f1\n')
        #         for res in results:
        #             a,b,c,d = res['joint_goal_delex'], res['slot_acc_delex'], res['slot_f1_delex'], res['act_f1']
        #             rf.write('%2.1f\t%2.1f\t%2.1f\t%2.1f\n'%(a,b,c,d))
        ctr_save_path = cfg.result_path[:-4] + '_report_ctr%s.csv' % cfg.seed
        write_title = False if os.path.exists(ctr_save_path) else True
        if cfg.aspn_decode_mode == 'greedy':
            setting = ''
        elif cfg.aspn_decode_mode == 'beam':
            setting = 'width=%s' % str(cfg.beam_width)
            if cfg.beam_diverse_param > 0:
                setting += ', penalty=%s' % str(cfg.beam_diverse_param)
        elif cfg.aspn_decode_mode == 'topk_sampling':
            setting = 'topk=%s' % str(cfg.topk_num)
        elif cfg.aspn_decode_mode == 'nucleur_sampling':
            setting = 'p=%s' % str(cfg.nucleur_p)
        res = {'exp': cfg.eval_load_path, 'true_bspn': cfg.use_true_curr_bspn, 'true_aspn': cfg.use_true_curr_aspn,
               'decode': cfg.aspn_decode_mode, 'param': setting, 'nbest': cfg.nbest, 'selection_sheme': cfg.act_selection_scheme,
               'match': results[0]['match'], 'success': results[0]['success'], 'bleu': results[0]['bleu'], 'act_f1': results[0]['act_f1'],
               'avg_act_num': results[0]['avg_act_num'], 'avg_diverse': results[0]['avg_diverse_score']}
        with open(ctr_save_path, 'a') as rf:
            writer = csv.DictWriter(rf, fieldnames=list(res.keys()))
            if write_title:
                writer.writeheader()
            writer.writerows([res])


class CrossWozReader(_ReaderBase):

    def __init__(self, tokenizer):
        super().__init__()
        self.nlp = spacy.load('zh_core_web_sm')

        self.db = CrossWozDB(cfg.dbs)

        # self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path) # add special tokens later
        self.tokenizer = tokenizer
        #修改记录：删除了if cfg.mode=='train'语句
        self.add_sepcial_tokens()

        self.domain_files = json.loads(open(cfg.domain_file_path, 'r').read().encode())
        self.slot_value_set = json.loads(
            open(cfg.slot_value_set_path, 'r').read())
        if cfg.multi_acts_training:
            self.multi_acts = json.loads(open(cfg.multi_acts_path, 'r').read())

        test_list = [l.strip().lower()#test dialog number
                     for l in open(cfg.test_list, 'r').readlines()]
        dev_list = [l.strip().lower()
                    for l in open(cfg.dev_list, 'r').readlines()]
        self.dev_files, self.test_files = {}, {}
        for fn in test_list:
            self.test_files[fn.replace('.json', '')] = 1
        for fn in dev_list:
            self.dev_files[fn.replace('.json', '')] = 1

        # for domain expanse aka. Cross domain
        self.exp_files = {}
        # if 'all' not in cfg.exp_domains:
        #     for domain in cfg.exp_domains:
        #         fn_list = self.domain_files.get(domain)
        #         if not fn_list:
        #             raise ValueError(
        #                 '[%s] is an invalid experiment setting' % domain)
        #         for fn in fn_list:
        #             self.exp_files[fn.replace('.json', '')] = 1
        all_domains_list = list(self.domain_files.keys())
        if 'all' not in cfg.exp_domains:
            domains = self.get_exp_domains(cfg.exp_domains, all_domains_list)
            logging.info(domains)
            for domain in domains:
                fn_list = self.domain_files.get(domain)
                if not fn_list:
                    raise ValueError(
                        '[%s] is an invalid experiment setting' % domain)
                for fn in fn_list:
                    self.exp_files[fn.replace('.json', '')] = 1
        #

        self._load_data()


        self.multi_acts_record = None
        self.get_special_ids()
    
    def get_special_ids(self):
        self.sos_b_id=self.tokenizer.convert_tokens_to_ids('<sos_b>')
        self.sos_a_id=self.tokenizer.convert_tokens_to_ids('<sos_a>')
        self.sos_r_id=self.tokenizer.convert_tokens_to_ids('<sos_r>')
        self.eos_b_id=self.tokenizer.convert_tokens_to_ids('<eos_b>')
        self.eos_a_id=self.tokenizer.convert_tokens_to_ids('<eos_a>')
        self.eos_r_id=self.tokenizer.convert_tokens_to_ids('<eos_r>')
        self.sos_db_id=self.tokenizer.convert_tokens_to_ids('<sos_db>')
        self.eos_db_id=self.tokenizer.convert_tokens_to_ids('<eos_db>')
        self.sos_u_id=self.tokenizer.convert_tokens_to_ids('<sos_u>')
        self.eos_u_id=self.tokenizer.convert_tokens_to_ids('<eos_u>')

    def get_exp_domains(self, exp_domains, all_domains_list):
        # not used for crosswoz
        if 'hotel' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'hotel']
                domains = [d for d in all_domains_list if 'hotel' not in d and 'multi' not in d]
            else:
                # ['hotel']
                domains = ['hotel_single', 'hotel_multi']
        if 'train' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'train']
                domains = [d for d in all_domains_list if 'train' not in d and 'multi' not in d]
            else:
                # ['train']
                domains = ['train_single', 'train_multi']
        if 'attraction' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'attraction']
                domains = [d for d in all_domains_list if 'attraction' not in d and 'multi' not in d]
            else:
                # ['attraction']
                domains = ['attraction_single', 'attraction_multi']
        if 'restaurant' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'restaurant']
                domains = [d for d in all_domains_list if 'restaurant' not in d and 'multi' not in d]
            else:
                # ['restaurant']
                domains = ['restaurant_single', 'restaurant_multi']
        if 'taxi' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'taxi']
                domains = [d for d in all_domains_list if 'taxi' not in d and 'multi' not in d]
            else:
                # ['taxi']
                domains = ['taxi_single', 'taxi_multi']
        return domains

    def add_sepcial_tokens(self):
        """
            add special tokens to gpt tokenizer
            serves a similar role of Vocab.construt()
            make a dict of special tokens
        """
        special_tokens = []

        for word in ontology.all_domains_cross + ['general']:
            word = '[' + word + ']'
            special_tokens.append(word)

        for word in ontology.all_acts:
            word = '[' + word + ']'
            special_tokens.append(word)
        
        for k, v in ontology.reqslot2placeholder.items():
            special_tokens.append(k)
            if v not in special_tokens:
                special_tokens.append(v)
        
        
        special_tokens.extend(ontology.special_tokens)

        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        logging.info('Added special tokens to gpt tokenizer.')

        cfg.pad_id = self.tokenizer.encode('<pad>')[0]



    def _load_data(self, save_temp=True):
        """
        load processed data and encode, or load already encoded data
        """
        if save_temp: # save encoded data
            if 'all' in cfg.exp_domains:
                #data.json is the encoded file of standard delexicalized data
                #data2.json is obtained by changing '[value_choice]' to corresponding db pointer
                #different data file needs different vocab file
                if cfg.delex_as_damd:
                    encoded_file = os.path.join(cfg.data_path, 'new_db_se_blank_encoded.data.json')
                else:
                    encoded_file = os.path.join(cfg.data_path, 'new_db_se_blank_encoded.data2.json')
                # encoded: no sos, se_encoded: sos and eos
                # db: add db results every turn
            else:
                xdomain_dir = './experiments_Xdomain/data'
                if not os.path.exists(xdomain_dir):
                    os.makedirs(xdomain_dir)
                encoded_file = os.path.join(xdomain_dir, '{}-encoded.data.json'.format('-'.join(cfg.exp_domains))) 

            if os.path.exists(encoded_file):
                logging.info('Reading encoded data from {}'.format(encoded_file))
                self.data = json.loads(
                    open(cfg.data_path+cfg.data_file, 'r', encoding='utf-8').read().lower())
                encoded_data = json.loads(open(encoded_file, 'r', encoding='utf-8').read())
                self.train = encoded_data['train']
                self.dev = encoded_data['dev']
                self.test = encoded_data['test']
            else:
                logging.info('Encoding data now and save the encoded data in {}'.format(encoded_file))
                # not exists, encode data and save
                self.data = json.loads(
                    open(cfg.data_path+cfg.data_file, 'r', encoding='utf-8').read().lower())
                self.train, self.dev, self.test = [], [], []
                for fn, dial in self.data.items():
                    if '.json' in fn:
                        fn = fn.replace('.json', '')
                    if 'all' in cfg.exp_domains or self.exp_files.get(fn):
                        if self.dev_files.get(fn):
                            self.dev.append(self._get_encoded_data(fn, dial))
                        elif self.test_files.get(fn):
                            self.test.append(self._get_encoded_data(fn, dial))
                        else:
                            self.train.append(self._get_encoded_data(fn, dial))
                
                # save encoded data
                encoded_data = {'train': self.train, 'dev': self.dev, 'test': self.test}
                json.dump(encoded_data, open(encoded_file, 'w'), indent=2)
                logging.info('encoded file saved in %s'%encoded_file)

        else: # directly read processed data and encode
            self.data = json.loads(
                open(cfg.data_path+cfg.data_file, 'r', encoding='utf-8').read().lower())
            self.train, self.dev, self.test = [], [], []
            for fn, dial in self.data.items():
                if '.json' in fn:
                    fn = fn.replace('.json', '')
                if 'all' in cfg.exp_domains or self.exp_files.get(fn):
                    if self.dev_files.get(fn):
                        self.dev.append(self._get_encoded_data(fn, dial))
                    elif self.test_files.get(fn):
                        self.test.append(self._get_encoded_data(fn, dial))
                    else:
                        self.train.append(self._get_encoded_data(fn, dial))

        random.shuffle(self.train)
        # random.shuffle(self.dev)
        # random.shuffle(self.test)
        logging.info('train size:{}, dev size:{}, test size:{}'.format(len(self.train), len(self.dev), len(self.test)))

    def _get_encoded_data(self, fn, dial):
        encoded_dial = []
        for idx, t in enumerate(dial['log']):  # tokenize to list of ids
            enc = {}
            enc['dial_id'] = fn


            # gpt use bpe to encode strings, very very slow. ~9min
            # in tokenization_utils.encode I find encode can pad_to_max_length, and reutrn tensor
            enc['user'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize( 
                '<sos_u> ' +
                t['user'] + ' <eos_u>'))
            #if cfg.delex_as_damd:
            enc['bspn'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                '<sos_b> ' +
                t['constraint'] + ' <eos_b>'))
            enc['resp'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                '<sos_r> ' +
                t['resp'] + ' <eos_r>'))
            enc['aspn'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                '<sos_a> ' +
                t['sys_act'] + ' <eos_a>'))


            enc['pointer'] = [int(i) for i in t['pointer'].split(',')]
            enc['turn_domain'] = t['turn_domain'].split()
            enc['turn_num'] = t['turn_num']

            # add db results to enc, at every turn
            db_pointer = self.bspan_to_DBpointer(t['constraint'], t['turn_domain'].split())
            # db_tokens = ['<sos_db>', '<eos_db>', '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']
            enc['db'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
                '<sos_db> ' +
                db_pointer + ' <eos_db>'))

            encoded_dial.append(enc)
        return encoded_dial
    
    def get_pv_batch(self, pv_batch, user=None, resp=None, bspn=None, aspn=None, side='sys'):
        assert side in ['sys', 'user']
        new_pv_batch=[] # pv_batch for next turn
        if side=='sys':
            if pv_batch is None:# first turn
                for u, r, b in zip(user, resp, bspn): 
                    if cfg.input_history:
                        new_pv_batch.append(u+r)
                    elif cfg.input_prev_resp:
                        new_pv_batch.append(b+r)
                    else:
                        new_pv_batch.append(b)
            else:
                for hist, u, r, b in zip(pv_batch,user, resp, bspn):
                    if cfg.input_history:
                        new_pv_batch.append(hist+u+r)
                    elif cfg.input_prev_resp:
                        new_pv_batch.append(b+r)
                    else:
                        new_pv_batch.append(b)
        else:# user's pv batch
            for r, a in zip(resp, aspn):
                new_pv_batch.append(r)
        return new_pv_batch

    def bspan_to_constraint_dict(self, bspan, bspn_mode='bspn'):    
        constraint_dict=list2dict(parser_bs(bspan))
        return constraint_dict

    def bspan_to_DBpointer(self, bspan, turn_domain):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        # print(constraint_dict)
        matnums = self.db.get_match_num(constraint_dict)
        match_dom = ''
        for domain in turn_domain:
            if (domain!='[greet]') and (domain!='[welcome]') and (domain!='[bye]') and (domain!='[thank]') and (domain!='[reqmore]') and (domain!='[general]'):
                match_dom = domain
        match_dom = match_dom[1:-1] if match_dom.startswith('[') else match_dom
        if match_dom=='':
            match_dom='general'
        match = matnums[match_dom]
        # vector = self.db.addDBPointer(match_dom, match)
        vector = self.db.addDBIndicator(match_dom, match)
        return vector
    
    def batch_align(self,contexts,left_len,return_attn=False):
        max_len=max([len(context) for context in contexts])
        max_len=min(1024-left_len,max_len)
        new_contexts=[]
        attentions=[]
        for id, context in enumerate(contexts):
            if len(context)<max_len:
                new_context=(max_len-len(context))*[cfg.pad_id]+context
                attention=(max_len-len(context))*[0]+len(context)*[1]
            else:
                new_context=context[-max_len:]
                attention=len(new_context)*[1]
            new_contexts.append(new_context)
            attentions.append(attention)
        if return_attn:
            return new_contexts, attentions
        return new_contexts

    def convert_eval_batch_turn(self, turn_batch, pv_batch, mode='gen_bspn', bspn_gen=None, db_gen=None, posterior=False):
        eval_batch=[]
        assert mode in ['gen_bspn', 'gen_ar']
        if pv_batch is None:
            if mode=='gen_bspn':
                for u, r in zip(turn_batch['user'], turn_batch['resp']):
                    context=u+r+[self.sos_b_id] if posterior else u+[self.sos_b_id]
                    eval_batch.append(context)
            else:
                for u, b, d, r in zip(turn_batch['user'], bspn_gen, db_gen, turn_batch['resp']):
                    context=u+r+b+d+[self.sos_a_id] if posterior else u+b+d+[self.sos_a_id]
                    eval_batch.append(context)
        else:
            if mode=='gen_bspn':
                for hist, u, r in zip(pv_batch, turn_batch['user'], turn_batch['resp']):
                    context=hist+u+r+[self.sos_b_id] if posterior else hist+u+[self.sos_b_id]
                    eval_batch.append(context)
            else:
                for hist, u, b, d, r in zip(pv_batch, turn_batch['user'], bspn_gen, db_gen, turn_batch['resp']):
                    context=hist+u+r+b+d+[self.sos_a_id] if posterior else hist+u+b+d+[self.sos_a_id]
                    eval_batch.append(context)
        return eval_batch

    def convert_batch_turn(self, 
        turn_batch, 
        pv_batch, 
        first_turn=False, 
        rl_train=False, 
        mode='oracle', 
        side='sys', 
        posterior=False,
        seg_label=False
        ):
        '''
        Args:
        Returns:
        '''
        assert mode in ['oracle', 'gen']
        assert side in ['sys', 'user']
        inputs = {}
        labels = {}
        contexts = []
        label_contexts = []
        if rl_train:
            rl_labels={}
            rl_label_contexts=[]
        if seg_label:
            seg_labels={}
            seg_contexts=[]
        if side=='sys':
            if first_turn:
                if mode=='oracle':
                    batch_zipped = zip(turn_batch['user'], turn_batch['bspn'], 
                        turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
                else:
                    batch_zipped=zip(turn_batch['user'], turn_batch['bspn_gen'], 
                        turn_batch['db_gen'], turn_batch['aspn_gen'], turn_batch['resp_gen'])
                    
                for u, b, db, a, r in batch_zipped:
                    if posterior:
                        context=u+r + b+db+a
                        label_context=len(u+r)*[cfg.pad_id] + b+db+a
                    else:
                        context = u+b+db+a+r
                        label_context=len(u)*[cfg.pad_id]+b+db+a+r
                    contexts.append(context)
                    label_contexts.append(label_context)
                    if rl_train:
                        # 1 for belief state, 2 for system act, 3 for response and 0 for others
                        rl_label_context=len(u)*[0]+len(b)*[1]+len(db)*[0]+len(a)*[2]+len(r)*[3]
                        rl_label_contexts.append(rl_label_context)
                    if seg_label:
                        # 1 for hidden state, 2 for response, 0 for others
                        seg_context=len(u)*[0] + len(b+db+a)*[1] + len(r)*[2]
                        seg_contexts.append(seg_context)
            else:
                if mode=='oracle':
                    batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                        turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
                else:
                    batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn_gen'], 
                        turn_batch['db_gen'], turn_batch['aspn_gen'], turn_batch['resp_gen'])
                for ur, u, b, db, a, r in batch_zipped:
                    if posterior:
                        context = ur + u + r + b + db + a
                        label_context=len(ur+u+r)*[cfg.pad_id] + b+db+a
                    else:
                        context = ur + u + b + db + a + r
                        label_context=len(ur+u)*[cfg.pad_id]+b+db+a+r
                    contexts.append(context)
                    label_contexts.append(label_context)
                    if rl_train:
                        # 1 for belief state, 2 for system act, 3 for response and 0 for others
                        rl_label_context=len(ur+u)*[0]+len(b)*[1]+len(db)*[0]+len(a)*[2]+len(r)*[3]
                        rl_label_contexts.append(rl_label_context)
                    if seg_label:
                        # 1 for hidden state, 2 for response, 0 for others
                        seg_context=len(ur+u)*[0] + len(b+db+a)*[1] + len(r)*[2]
                        seg_contexts.append(seg_context)
        
        elif side=='user':
            if first_turn:
                if mode=='oracle':
                    batch_zipped = zip(turn_batch['goal'], turn_batch['usr_act'], turn_batch['user'])
                else:
                    batch_zipped=zip(turn_batch['goal'], turn_batch['usr_act_gen'], turn_batch['user_gen'])                   
                for g, ua, u in batch_zipped:
                    context = g + ua + u
                    label_context = len(g)*[cfg.pad_id]+ua+u
                    #context = g + [self.sos_r_id, self.eos_r_id] + ua + u
                    #label_context=(len(g)+2)*[cfg.pad_id]+ua+u
                    contexts.append(context)
                    label_contexts.append(label_context)
            else:
                if mode=='oracle':
                    batch_zipped = zip(pv_batch, turn_batch['goal'], turn_batch['usr_act'], turn_batch['user'])
                else:
                    batch_zipped=zip(pv_batch, turn_batch['goal'], turn_batch['usr_act_gen'], turn_batch['user_gen'])                   
                for pv, g, ua, u in batch_zipped:
                    context = pv + g + ua + u
                    label_context=len(pv+g)*[cfg.pad_id]+ua+u
                    contexts.append(context)
                    label_contexts.append(label_context)

        inputs['contexts'] = contexts
        inputs['contexts_np'], inputs['lengths'] = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        labels['contexts']=label_contexts
        labels['contexts_np'], labels['lengths']=utils.padSeqs_gpt(labels['contexts'], cfg.pad_id)
        if seg_label and side=='sys':
            seg_labels['contexts']=seg_contexts
            seg_labels['contexts_np'], seg_labels['lengths']=utils.padSeqs_gpt(seg_labels['contexts'], cfg.pad_id)
            return inputs, labels, seg_labels
        if rl_train and side=='sys':
            rl_labels['contexts']=rl_label_contexts
            rl_labels['contexts_np'], rl_labels['lengths']=utils.padSeqs_gpt(rl_labels['contexts'], cfg.pad_id)
            return inputs, labels, rl_labels
        else:
            return inputs, labels
    
    def split_turn_batch(self, turn_batch, batch_size, other_batch=None):
        batches=[]
        other_batches=[]
        B=len(turn_batch['user'])
        for i in range(0, B, batch_size):
            new_turn_batch={}
            if other_batch:
                other_batches.append(other_batch[i:i+batch_size])
            for key in turn_batch:
                new_turn_batch[key]=turn_batch[key][i:i+batch_size]
            batches.append(new_turn_batch)
        if other_batch:
            return batches, other_batches
        else:
            return batches, None

    def convert_batch_session(self, dial_batch,posterior_train=False, only_resp_label=False,bspn_label=False,bspn_pri=False):
        """
        convert the whole session for training
        concat [U_0, B_0, A_0, R_0, ... , U_n, B_n, A_n, R_n]

        try: [user, bspn, aspn, resp]
        or
        try: [user, bspn, db, aspn, resp]
        """
        #修改记录：增加一个返回值，labels
        #修改记录：根据posterior_train决定返回的序列
        #修改记录：去掉了aspn
        inputs = {}
        labels={}
        bspn_labels={}
        contexts = []
        label_contexts=[]
        bspn_label_contexts=[]
        if not posterior_train:
            if cfg.model_act:
                cell_list = ['user', 'bspn', 'db','aspn', 'resp']
                ignore_list= ['user','bspn','db','aspn'] if only_resp_label else ['user']
            else:
                cell_list = ['user', 'bspn', 'db', 'resp']
                ignore_list=['user','bspn','db'] if only_resp_label else ['user','db']
            
        else:
            if cfg.model_act:
                cell_list=['user','resp','bspn','db','aspn']
                ignore_list=['user','resp']
            else:
                cell_list=['user','resp','bspn']
                ignore_list=['user','resp']
        for idx, dial in enumerate(dial_batch):
            context = []
            label_context=[]
            bspn_label_context=[]
            for turn_num, turn in enumerate(dial):
                for cell in cell_list:
                    if cell=='bspn' and bspn_pri and 'bspn_pri' in turn:
                        cell='bspn_pri'
          
                    if cell=='db':
                        if bspn_pri and 'db_pri' in turn:
                            cell='db_pri'
                        else:
                            db_result=self.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                            turn[cell] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>'))
                    context.extend(turn[cell])
                    if cell in ignore_list:
                        label_context.extend(len(turn[cell])*[cfg.pad_id])#pad_id在计算损失时被自动忽略
                    else:
                        label_context.extend(turn[cell])
                    if bspn_label:
                        bspn_cell_list=['bspn','db','aspn'] if cfg.model_act else ['bspn']
                        if cell in bspn_cell_list:
                            bspn_label_context.extend(turn[cell])
                        else:
                            bspn_label_context.extend(len(turn[cell])*[cfg.pad_id])

            contexts.append(context)
            label_contexts.append(label_context)
            bspn_label_contexts.append(bspn_label_context)


        inputs['contexts'] = contexts
        inputs['contexts_np'], inputs['lengths'] = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        labels['contexts']=label_contexts
        labels['contexts_np'], labels['lengths']=utils.padSeqs_gpt(labels['contexts'], cfg.pad_id)

        if not bspn_label:
            return inputs,labels
        else:
            bspn_labels['contexts']=bspn_label_contexts
            bspn_labels['contexts_np'],bspn_labels['lengths']=utils.padSeqs_gpt(bspn_labels['contexts'], cfg.pad_id)
            return inputs,labels,bspn_labels


    def wrap_result_lm(self, result_dict, eos_syntax=None):
        results = []
        eos_syntax = ontology.eos_tokens if not eos_syntax else eos_syntax
        sos_syntax = ontology.sos_tokens
        # ground truth bs, as, ds.. generate response
        field = ['dial_id', 'turn_num', 'user', 'bspn_gen', 'bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn', 'bspn', 'pointer', 'turn_domain']

        for dial_id, turns in result_dict.items():
            #修改记录：下方的turn_num在源代码中写的是trun_num
            entry = {'dial_id': dial_id, 'turn_num': len(turns)}
            for f in field[2:]:
                entry[f] = '' # ???
            results.append(entry)
            for turn_idx, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key == 'turn_domain':
                        v = ' '.join(v)

                    if key in eos_syntax and v != '':
                        # remove eos tokens
                        v = self.tokenizer.decode(v)
                        v = v.split()
                        # remove eos/sos in span
                        if eos_syntax[key] in v:
                            v.remove(eos_syntax[key])
                        if sos_syntax[key] in v:
                            v.remove(sos_syntax[key])
                        # if key != 'resp_gen':
                        #     # remove eos/sos in span
                        #     if eos_syntax[key] in v:
                        #         v.remove(eos_syntax[key])
                        #     if sos_syntax[key] in v:
                        #         v.remove(sos_syntax[key])
                        # else: # 'resp_gen'
                        #     sos_index = 0
                        #     eos_index = -1
                        #     if sos_syntax[key] in v:
                        #         sos_index = v.index(sos_syntax[key])
                        #     if eos_syntax[key] in v:
                        #         eos_index = v.index(eos_syntax[key])
                        #     else:
                        #         pass # take too long
                        #         # no <eos_r> found, stop at any eos_tokens
                        #         # for i in range(sos_index+1, len(v)):
                        #         #     if v[i] in sos_syntax.values() or v[i] in eos_syntax.values():
                        #         #         eos_index = i
                        #     v = v[sos_index+1: eos_index]


                        # v = self.tokenizer.convert_tokens_to_string(v)
                        v = " ".join(v)
                    else: 
                        pass # v = v
                    entry[key] = v

                results.append(entry)

        return results, field


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    reader = CrossWozReader(tokenizer)
    # for aspan in ["[general] [bye] [welcome] <eos_a>","[train] [inform] trainid destination arrive leave [offerbook] [general] [reqmore] <eos_a>",]:
    #     act = reader.aspan_to_constraint_dict(aspan.split())
    #     print('！！！')
    #     print(act)

    for bspan in ["[景点] 门票 100 - 150 元 游玩时间 1 小 时 - 2 小 时 [餐馆] 推荐菜 干 炸 素 评分 4 分 以 上"]:
        encoded = reader.vocab.sentence_encode(bspan.split())
        print(encoded)
        cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bspn')
        print(cons)
    for bspan in ["[景点] 门票 游玩时间 [酒店] 酒店设施"]:
        encoded = reader.vocab.sentence_encode(bspan.split())
        print(encoded)
        cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bsdx')
        print(cons)

