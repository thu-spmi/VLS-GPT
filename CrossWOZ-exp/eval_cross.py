import math, logging, copy, json
from collections import Counter, OrderedDict
from nltk.util import ngrams

import ontology_cross as ontology
from config_cross import global_config as cfg
from dst import parser_bs
from util.crosswoz.dbquery import Database, list2dict


def inform_str2dict(sent):
    """Convert compacted bs span to a list of triple list
        In: '[domain1] [intent1] value1 value2 [domain2] [intent2] value21' '[酒店] [inform] 酒店设施 - spa 酒店设施 - 早 餐 服 务 免 费 [SEP]'
        Ex:  [[domain1, intent1, [value11,value12]],[domain2, intent2, value21],...] {'景点': [], '餐馆': [], '酒店': ['酒店设施', '-', 'spa', '酒店设施', '-', '早', '餐', '服', '务', '免', '费'], '出租': [], '地铁': []}
    """
    all_domain = ['[景点]', '[餐馆]', '[酒店]', '[出租]', '[地铁]']
    all_intent = ["[inform]", "[request]", "[nooffer]", "[recommend]",
                  "[select]", "[offerbook]", "[offerbooked]", "[nobook]",
                  "[bye]", "[greet]", "[reqmore]", "[welcome]"]
    sent = sent.strip('<sos_a>').strip('<eos_a>').rstrip('[SEP]')  # 去掉可能有的起止标识符
    sent = sent.split()  # 按空格分句
    act = {'景点': [], '餐馆': [], '酒店': [], '出租': [], '地铁': []}
    domain_idx = [idx for idx, token in enumerate(sent) if token in all_domain]  # 提取domain标识的位置
    for i, d_idx in enumerate(domain_idx):  # 对所有domain循环
        next_d_idx = len(sent) if i + 1 == len(domain_idx) else domain_idx[i + 1]  # 下一个domain标识的位置
        domain = sent[d_idx][1:-1]  # 当前domain名称
        sub_span = sent[d_idx + 1:next_d_idx]  # 提取当前domain对应的内容
        sub_s_idx = [idx for idx, token in enumerate(sub_span) if token in all_intent]  # 类似地，提取intent为inform的标识的位置
        for j, s_idx in enumerate(sub_s_idx):  # 类似地，对所有intent循环，构建 [domain, intent, slot] 三元组
            next_s_idx = len(sub_span) if j == len(sub_s_idx) - 1 else sub_s_idx[j + 1]
            intent = sub_span[s_idx]
            if intent == '[inform]':
                for slot in sub_span[s_idx + 1:next_s_idx]:
                    act[domain].append(slot)
    return act


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):
        '''

        :param parallel_corpus: zip(list of str 1,list of str 2)
        :return: bleu4 score
        '''
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]
        empty_num = 0
        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            if hyps == ['']:
                empty_num += 1
                continue

            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]

            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        # print('empty turns:',empty_num)
        return bleu * 100


class CrossWOZEvaluator(object):
    
    def __init__(self, reader):
        self.reader = reader
        self.domains = ontology.all_domains_cross
        # <class 'list'>
        # for multiwoz: ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']
        # for crosswoz: should be ['景点', '餐馆', '酒店', '出租', '地铁'], satisfied

        # for crosswoz:
        # self.domain_files = self.reader.domain_files
        # <dict>
        # for multiwoz: key: domainName_Single/Multi or domain1_domain2, each value is a list of str, each str is a json filename
        # for crosswoz: not using it anymore( it's related to reader.domain_files and domain_count.json, however for crosswoz it's too complicated to change the relevant code)

        self.all_data = self.reader.data
        # each data is a dialog session with goal and log
        # for multiwoz: key: session ID
        self.test_data = self.reader.test
        # <class 'list'> each item is a dialog session with context in the form of idx (instead of str)

        self.bleu_scorer = BLEUScorer()

        self.all_info_slot = []  # each item: 'domain-slot'
        for d, s_list in ontology.informable_slots.items():
            # d: domain (e.g. taxi)
            # s_list: slot list (e.g.<class 'list'>: ['leave', 'destination', 'departure', 'arrive'])
            for s in s_list:
                self.all_info_slot.append(d + '-' + s)

        # only evaluate these slots for dialog success
        # DONE: should change the requestables={...} to fit crosswoz
        # self.requestables = ontology.all_reqslot  # success is related to request instead of inform

    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = turn['dial_id']
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials

    def validation_metric(self, data):
        bleu = self.bleu_metric(data)
        success, match, _, _ = self.context_to_response_eval(data)
        return bleu, success*100, match*100

    def bleu_metric(self, data, eval_dial_list=None):
        gen, truth = [], []
        for row in data:
            if eval_dial_list and row['dial_id'] + '.json' not in eval_dial_list:
                continue
            gen.append(row['resp_gen'])
            truth.append(row['resp'])
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        if gen and truth:
            sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
        else:
            sc = 0.0
        return sc


    def context_to_response_eval(self, data, eval_dial_list=None):

        # 把列表形式的data改为字典，key=dial_id
        dials = self.pack_dial(data)

        # statistics
        match_stats = {'景点': {'TP': 0, 'FP': 0, 'FN': 0, 'P': 0, 'R': 0, 'F1': 0},
                       '餐馆': {'TP': 0, 'FP': 0, 'FN': 0, 'P': 0, 'R': 0, 'F1': 0},
                       '酒店': {'TP': 0, 'FP': 0, 'FN': 0, 'P': 0, 'R': 0, 'F1': 0},
                       '出租': {'TP': 0, 'FP': 0, 'FN': 0, 'P': 0, 'R': 0, 'F1': 0},
                       '地铁': {'TP': 0, 'FP': 0, 'FN': 0, 'P': 0, 'R': 0, 'F1': 0}}
        success_stats = {'景点': {'offer': 0, 'total': 0, 'accuracy': 0},
                         '餐馆': {'offer': 0, 'total': 0, 'accuracy': 0},
                         '酒店': {'offer': 0, 'total': 0, 'accuracy': 0},
                         '出租': {'offer': 0, 'total': 0, 'accuracy': 0},
                         '地铁': {'offer': 0, 'total': 0, 'accuracy': 0}}
        # *****指标说明*****
        # 评测为turn-level
        # [MATCH/INFORM]
        # TP: 【POSIIVE: resp_gen中对于出租领域，出现了cp；对其他领域，出现了[value_name]】 && 【TRUE: bspn_gen与bspn(oracle)查询结果有交集或均为空集】
        # FP: 【POSIIVE: resp_gen中对于出租领域，出现了cp；对其他领域，出现了[value_name]】 && 【FALSE: 与上行相反】
        # TN: 【NEGATIVE: resp_gen中对于出租领域，没有出现cp；对其他领域，没有出现[value_name]】 && 【TRUE: aspn(oracle)有'名称'or'车牌'】
        # -->>DONE: 基于此可以计算P R F1

        # [SUCCESS]
        # TOTAL: 该domain的某个实体属性(名称、车牌除外)的placeholder出现在aspn(oracle)里
        # OFFER: resp_gen中出现了TOTAL里提到的实体属性
        # -->>准确率: ACCURACY = Num(OFFER)/Num(TOTAL)

        # 特殊说明："周边xx"对应[surrounding] [value_name]，视为实体属性。在MATCH评测中忽略它，在SUCCESS评测中考虑。

        # 对每个对话session循环
        for dial_id in dials:
            if eval_dial_list and dial_id + '.json' not in eval_dial_list:  # 如果该session不属于测试集则跳过
                continue
            dial_pred = dials[dial_id]  # 取出当前session的生成数据

            # 对每个turn循环
            for turn_i, turn_pred in enumerate(dial_pred):
                if turn_pred['user']=='':
                    continue

                # *****TEST*****
                # TODO: 观察每个类别的例子看看有没有问题
                # print(turn_pred)
                # print(turn_target)

                # *****INFORM / MATCH RATE*****

                # PREPARING DATA
                resp_pred = turn_pred['resp_gen']  # 提取[value_name]
                aspn_target = turn_pred['aspn']  # 提取 名称 周边xx
                bspn_gen = turn_pred['bspn_gen']
                bspn_gen=bspn_gen.replace('出 发 地', '出发地')
                bspn_gen=bspn_gen.replace('目 的 地', '目的地')
                bspn_gen = parser_bs(bspn_gen)
                bspn_target = turn_pred['bspn']
                bspn_target=bspn_target.replace('出 发 地', '出发地')
                bspn_target=bspn_target.replace('目 的 地', '目的地')
                bspn_target = parser_bs(bspn_target)
                # DONE: deal with [surrounding], 目前的想法如下
                # 出现[surrounding]去掉它及紧随其后的[value_name]再来判断MATCH
                # 例子：
                # "我要住评分4分以上的酒店，请告诉我酒店名称及周边景点"
                # "好的，酒店名称是[value_name]，周边景点有[surrouding] [value_name], [surrounding] [value_name]"
                # -->>"好的，酒店名称是[value_name]，周边景点有, "
                # 这样一来，用户问酒店这个实体，可以通过是否有[value_name]来判断系统是否回复，不会受到[surrounding] [value_name]的干扰

                # 分领域评测
                # GENERAL CASES (出租以外的领域，都有[value_name]) DONE: check 地铁
                # 地铁不是[value_name]而是[value_place]
                db = Database()
                aspn_target_dict = inform_str2dict(aspn_target)
                for domain in ['[景点]', '[餐馆]', '[酒店]', '[地铁]']:
                    if domain in turn_pred['turn_domain'].split():
                        domain = domain[1:-1] if domain.startswith('[') else domain # 去掉domain的括号
                        # calculate all slot in oracle act
                        inform_target = aspn_target_dict.get(domain, [])
                        if len(inform_target)>0:
                            plhd_target = [ontology.reqslot2placeholder[i] for i in inform_target if i in ontology.all_reqslot]
                        if ('[value_name]' in resp_pred) or (domain=='地铁'):
                            # domain为地铁，则只需判断出发地铁站和目的地地铁站是否正确
                            # 2. 判断系统回复的实体是否正确：分domain用生成的bspn_gen查询数据库得到的实体和bspn(oracle)查询到的实体有交集或均为空集
                            # DB Query
                            # DONE:酒店设施识别
                            bspn_gen_dict = list2dict(bspn_gen)  # DONE: list->state dict
                            bspn_target_dict = list2dict(bspn_target)
                            # DONE:目前大小写敏感 ktv KTV识别不出来 把db改为小写放入database_lowercase文件夹用于查询
                            # print(turn_pred)
                            # print(bspn_gen_dict)
                            venue_gen = db.query(bspn_gen_dict, domain)
                            venue_target = db.query(bspn_target_dict, domain)
                            # 对查询到的结果提取名称
                            venue_gen = [v[0] for v in venue_gen]
                            venue_target = [v[0] for v in venue_target] 
                            #if len(set(venue_gen) & set(venue_target)) > 0 or (len(venue_gen) == 0 and len(venue_target) == 0):
                            if set(venue_gen).issubset(set(venue_target)) and len(set(venue_gen))>0: # MATCH
                                # success_stats[domain]['total'] += len(plhd_target)
                                match_stats[domain]['TP'] += 1
                                #**SUCEESS RATE**
                                # **PREPARING DATA**
                                resp_pred = turn_pred['resp_gen'].split()
                                # 分领域评测
                                if len(inform_target) > 0:
                                    plhd_pred = []
                                    for id, item in enumerate(resp_pred):
                                        if '[' in item and ']' in item and item!='[value_name]':
                                            plhd_pred.append(item)
                                    for pt in plhd_target:
                                        if pt!='[value_name]':
                                            success_stats[domain]['total']+=1
                                        if pt in plhd_pred:
                                            success_stats[domain]['offer'] += 1
                                    t=1
                            else:
                                match_stats[domain]['FP'] += 1
                        else:
                            if '名称' in aspn_target:
                                match_stats[domain]['FN'] += 1
                # special case 1 : 出租领域无法进行数据查询
                # 处理方法：
                # UBAR对multiwoz的police和hospital领域的处理方法是出现[value_name]就认为MATCH
                # 此处对于crosswoz，生成的回复出现cp（意为车牌）且此轮真实domain为[出租] 就是MATCH
                if ('[car_number]' in resp_pred) or ('车牌' in aspn_target):# 说明是出租领域
                    inform_target = aspn_target_dict.get('出租', [])
                    if len(inform_target)>0:
                        plhd_target = [ontology.reqslot2placeholder[i] for i in inform_target if i in ontology.all_reqslot]
                        # success_stats['出租']['total'] += len(plhd_target)
                    if '[car_number]' in resp_pred:
                        if '车牌' in aspn_target:
                            match_stats['出租']['TP'] += 1
                            # **SUCEESS RATE**
                            # **PREPARING DATA**
                            resp_pred = turn_pred['resp_gen'].split()
                            # 分领域评测
                            if len(inform_target) > 0:
                                plhd_pred = []
                                for id, item in enumerate(resp_pred):
                                    if '[' in item and ']' in item:
                                        plhd_pred.append(item)
                                for pt in plhd_target:
                                    success_stats['出租']['total']+=1
                                    if pt in plhd_pred:
                                        success_stats['出租']['offer'] += 1
                        else:
                            match_stats['出租']['FP'] += 1
                    else:
                        match_stats['出租']['FN'] += 1

                # *****SUCEESS RATE*****
                # 把周边XX也视为一个【属性】，放在SUCCESS而非MATCH的评测中
                # '名称'不参与评测【不视为实体属性】, '周边xx'参与评测【视为实体属性】
                # 例子：
                # aspn_target: '[welcome] [餐馆] [inform] 名称 电话 周边酒店 地址'
                # aspn_target_dict: {'餐馆':['电话','周边酒店','地址']}，去掉了名称，保留了周边酒店
                # 通过reqslot2placeholder映射为 [value_phone]/ [surrounding] [value_name]/ [value_address]
                # success_stats['餐馆']['total']+=3
                # response中有无: [value_phone]/ [surrounding] [value_name]/ [value_address],
                # 有一个则success_stats['餐馆']['offer']+=1

        # 分domain计算P R F1
        total_num = 0
        avg_precision = 0
        avg_recall = 0
        avg_F1 = 0
        for key in match_stats:
            TP = match_stats[key]['TP']
            FP = match_stats[key]['FP']
            FN = match_stats[key]['FN']
            domain_num = TP + FP + FN
            total_num += domain_num
            precision = 1.0 * TP / (TP + FP) if (TP + FP) else 0.
            recall = 1.0 * TP / (TP + FN) if (TP + FN) else 0.
            F1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.
            match_stats[key]['P'] = precision
            match_stats[key]['R'] = recall
            match_stats[key]['F1'] = F1
            avg_precision += domain_num * precision
            avg_recall += domain_num * recall
            avg_F1 += domain_num * F1
        # 平均的 P R F1
        avg_precision /= total_num
        avg_recall /= total_num
        avg_F1 /= total_num

        # 分domain计算success rate (accuracy)
        offer_num = 0
        total_num = 0
        print(success_stats)
        for key in success_stats:
            domain_stats = success_stats[key]
            domain_stats['accuracy'] = domain_stats['offer'] / domain_stats['total'] if domain_stats['total'] else 0.
            offer_num += domain_stats['offer']
            total_num += domain_stats['total']
        #print('offer_num:{}, total_num:{}'.format(offer_num, total_num))
        #print('domain states:\n', domain_stats)
        avg_success = offer_num / (total_num+1e-10)

        return avg_success, avg_precision, success_stats, len(dials)




if __name__ == '__main__':
    pass
