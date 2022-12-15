import json
from ontology_cross import informable_slots


## for multiwoz:
# GENERAL_TYPO = {
#         # type
#         "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports",
#         "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall",
#         "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture",
#         "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
#         # area
#         "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east",
#         "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre",
#         "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north",
#         "centre of town":"centre", "cb30aq": "none",
#         # price
#         "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate",
#         # day
#         "next friday":"friday", "monda": "monday",
#         # parking
#         "free parking":"free",
#         # internet
#         "free internet":"yes",
#         # star
#         "4 star":"4", "4 stars":"4", "0 star rarting":"none",
#         # others
#         "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
#         '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",
#         }




def parser_bs(sent):
    """Convert compacted bs span to a list of triple list
        Ex:  [[domain, slot, value],[domain, slot, value],...]
    """
    all_domain = ['[景点]', '[餐馆]', '[酒店]', '[出租]', '[地铁]']
    # special cases
    # 电话既是一个单独的slot，又是"酒店设施-免费市内电话"的slot的一部分，为避免弄混先进行转化
    if "电话" in sent:
        if ("免 费 市 内 电话" in sent) or ("国 际 长 途 电话" in sent) or ("免 费 国 内 长 途 电话"):
            sent = sent.replace("电话", "电 话")
    # general
    sent = sent.strip('<sos_b>').strip('<eos_b>').rstrip('[SEP]')  # 去掉可能有的起止标识符
    sent = sent.split()  # 按空格分句
    belief_state = []
    domain_idx = [idx for idx, token in enumerate(sent) if token in all_domain]  # 提取domain标识的位置
    for i, d_idx in enumerate(domain_idx):  # 对所有domain循环
        next_d_idx = len(sent) if i + 1 == len(domain_idx) else domain_idx[i + 1]  # 下一个domain标识的位置
        domain = sent[d_idx]  # 当前domain名称
        slots = informable_slots[domain[1:-1]]
        sub_span = sent[d_idx + 1:next_d_idx]  # 提取当前domain对应的内容
        sub_s_idx = [idx for idx, token in enumerate(sub_span) if token in slots]  # 类似地，提取slot标识的位置
        for j, s_idx in enumerate(sub_s_idx):  # 类似地，对所有slot循环，构建 [domain, slot, value] 三元组，其中value若有多个则用空格隔开
            next_s_idx = len(sub_span) if j == len(sub_s_idx) - 1 else sub_s_idx[j + 1]
            slot = sub_span[s_idx]
            value = ''.join(sub_span[s_idx + 1:next_s_idx])
            bs = [domain, slot, value]
            belief_state.append(bs)
    return belief_state


def ignore_none(pred_belief, target_belief):
    ## for multiwoz:
    # for pred in pred_belief:
    #     if 'catherine s' in pred:
    #         pred.replace('catherine s', 'catherines')

    clean_target_belief = []
    clean_pred_belief = []
    for bs in target_belief:
        if 'not mentioned' in bs or 'none' in bs:
            continue
        clean_target_belief.append(bs)

    for bs in pred_belief:
        if 'not mentioned' in bs or 'none' in bs:
            continue
        clean_pred_belief.append(bs)

    dontcare_slots = []
    for bs in target_belief:
        if 'dontcare' in bs:
            domain = bs.split()[0]
            slot = bs.split()[1]
            dontcare_slots.append('{}_{}'.format(domain, slot))

    target_belief = clean_target_belief
    pred_belief = clean_pred_belief

    return pred_belief, target_belief


def fix_mismatch_jason(slot, value):
    # miss match slot and value
    if slot == "type" and value in ["nigh", "moderate -ly priced", "bed and breakfast",
                                    "centre", "venetian", "intern", "a cheap -er hotel"] or \
            slot == "internet" and value == "4" or \
            slot == "pricerange" and value == "2" or \
            slot == "type" and value in ["gastropub", "la raza", "galleria", "gallery",
                                         "science", "m"] or \
            "area" in slot and value in ["moderate"] or \
            "day" in slot and value == "t":
        value = "none"
    elif slot == "type" and value in ["hotel with free parking and free wifi", "4",
                                      "3 star hotel"]:
        value = "hotel"
    elif slot == "star" and value == "3 star hotel":
        value = "3"
    elif "area" in slot:
        if value == "no":
            value = "north"
        elif value == "we":
            value = "west"
        elif value == "cent":
            value = "centre"
    elif "day" in slot:
        if value == "we":
            value = "wednesday"
        elif value == "no":
            value = "none"
    elif "price" in slot and value == "ch":
        value = "cheap"
    elif "internet" in slot and value == "free":
        value = "yes"

    # some out-of-define classification slot values
    if slot == "area" and value in ["stansted airport", "cambridge", "silver street"] or \
            slot == "area" and value in ["norwich", "ely", "museum", "same area as hotel"]:
        value = "none"
    return slot, value


def default_cleaning(pred_belief, target_belief):
    pred_belief_jason = []
    target_belief_jason = []
    for pred in pred_belief:
        if pred in ['', ' ']:
            continue
        domain = pred.split()[0]
        if 'book' in pred:
            slot = ' '.join(pred.split()[1:3])
            val = ' '.join(pred.split()[3:])
        else:
            slot = pred.split()[1]
            val = ' '.join(pred.split()[2:])

        #if slot in GENERAL_TYPO:
        #    val = GENERAL_TYPO[slot]

        slot, val = fix_mismatch_jason(slot, val)

        pred_belief_jason.append('{} {} {}'.format(domain, slot, val))

    for tgt in target_belief:
        domain = tgt.split()[0]
        if 'book' in tgt:
            slot = ' '.join(tgt.split()[1:3])
            val = ' '.join(tgt.split()[3:])
        else:
            slot = tgt.split()[1]
            val = ' '.join(tgt.split()[2:])

        #if slot in GENERAL_TYPO:
        #    val = GENERAL_TYPO[slot]
        slot, val = fix_mismatch_jason(slot, val)
        target_belief_jason.append('{} {} {}'.format(domain, slot, val))

    turn_pred = pred_belief_jason
    turn_target = target_belief_jason

    return turn_pred, turn_target
