# 2021.7. added by cyc
all_domains_cross = ['景点', '餐馆', '酒店', '出租', '地铁']  # used in reader.py utils.py preprocess2.1.py db_ops.py eval.py
db_domains_cross = ['restaurant', 'hotel', 'attraction', 'taxi', 'metro']  # used in utils.py db_ops.py
domains_cross_switch = {'restaurant': '餐馆', 'hotel': '酒店', 'attraction': '景点', 'taxi': '出租',
                        'metro': '地铁'}  # used in utils.py db_ops.py
all_slots_cross = ["车型","车牌", "酒店设施", "酒店类型", "价格", "名称", "周边景点", "周边餐馆", "地址", "推荐菜", "营业时间",
                      "电话", "评分", "门票", "游玩时间", "周边酒店", "出发地附近地铁站", "目的地附近地铁站", "人均消费"]
# TODO: there's "normlize_slot_names" of multiwoz here -> should check how to change it
normlize_slot_names = {
    "car type": "car",
    "entrance fee": "price",
    "duration": "time",
    "leaveat": 'leave',
    'arriveby': 'arrive',
    'trainid': 'id'
}  # used in "preprocess.py" and "preprocess2.1.py"

# 2021.7.23 added by lzr (add requestable_slots, all_reqslot, informable_slots, all_infoslot contents of crosswoz, arranged in the form of multiwoz requestable_slots)
# 2021.7.26 revised by lzr (remove repeated slots in all_reqslot and all_infslot)
# 2021.8.9 revised by lzr (check value->[value_placeholder])

requestable_slots = {
    "出租": [
        # 出租数据库是模板，不查询，均为占位符
        # dst成功解析出用户的 起点-终点要求即算 match, 在此基础上出现车型or车牌即算 success
        "车型",# 目前不delex 在文中是"# cx"
        "车牌"# 目前不delex 在文中是"# cp"
    ],
    "酒店": [
        # 唯一标识符
        "名称",# [value_name]用于算match
        "地址",# [value_address]
        "电话",# [value_phone]
        # 允许查询的内容：名称、酒店类型、酒店设施、价格、评分、周边景点、周边餐馆、周边酒店
        # 周边信息
        "周边景点",# [surrounding] [value_name] TODO:[surrounding]加入模型
        "周边餐馆",# [surrounding] [value_name]

        "酒店类型",# [value_type]
        "价格",# [value_price]
        "评分",# [value_score]

        # DONE: 考虑如何处理 slot value
        # convlab2的gen_crosswoz_ontology.py里面是把“酒店设施”作为slot的
        # extract_all_ontology.py和extract_all_value.py里面是把“酒店设施-XX”作为slot [√]
        "酒店设施",# crosswoz中有些地方定义"酒店设施-xx"为slot 有些定义"酒店设施"为slot，这里为了处理方便把两者都囊括进来了
    ],
    "景点": [
        # 唯一标识符
        "名称",# [value_name] 用于算match
        "地址",# [value_address]
        "电话",# [value_phone]
        # 周边信息
        "周边餐馆",# [surrounding] [value_name]
        "周边酒店",# [surrounding] [value_name]
        "周边景点",# [surrounding] [value_name]

        # 允许查询数据库：名称、门票、游玩时间、评分、周边景点、周边酒店、周边餐馆
        "游玩时间",# [value_time]
        "门票", # [value_price]
        "评分" # [value_score]
    ],
    "地铁": [
        "出发地附近地铁站", # [value_place]
        "目的地附近地铁站" # [value_place]
    ],
    "餐馆": [
        # 唯一标识符
        "名称",  # [value_name]用于算match
        "地址",  # [value_address]
        "电话",  # [value_phone]

        # 周边信息
        "周边餐馆",  # [surrounding] [value_name]
        "周边酒店",  # [surrounding] [value_name]
        "周边景点",  # [surrounding] [value_name]

        "人均消费", # [value_price]
        "推荐菜", # [value_food] TODO:目前delex有点小问题，形如："推荐菜 有 [value_food], 萝 卜 汤, 小 炒 黄 [value_food], 蟹 黄 [value_food], 烧 [value_food]"
        "营业时间",# [value_time]
        "评分" # [value_score]

    ]
}  # not used
# count: 68

## map req_slots to placeholders, added by lzr on 2021.8.13
reqslot2placeholder={
    "人均消费":'[value_price]',
    "价格":'[value_price]',
    "名称":'[value_name]',
    "周边景点":'[nearby_attraction]',
    "周边酒店":'[nearby_hotel]',
    "周边餐馆":'[nearby_restaurant]',
    "地址":'[value_address]',
    "推荐菜":'[value_food]',
    "游玩时间":'[value_time]',
    "电话":'[value_phone]',
    "出发地附近地铁站": '[value_departure]',
    "目的地附近地铁站":'[value_destination]',
    "出发地": '[value_departure]',
    "目的地":'[value_destination]',
    "营业时间":'[value_time]',
    "评分":'[value_score]',
    "车型":'[car_type]',
    "车牌":'[car_number]',
    "酒店类型":'[value_type]',
    "酒店设施":'[value_facility]',
    "门票":'[value_price]'
}

all_reqslot = [
    "人均消费",
    "价格",
    "出发地附近地铁站",
    "名称",
    "周边景点",
    "周边酒店",
    "周边餐馆",
    "地址",
    "推荐菜",
    "游玩时间",
    "电话",
    "目的地附近地铁站",
    "营业时间",
    "评分",
    "车型",
    "车牌",
    "酒店类型",
    "酒店设施",
    "门票"
]  # not used
# count: 55

informable_slots = {
    "出租": [
        "车牌",
        "出发地",
        "车型",
        "目的地"
    ],
    "酒店": [
        "酒店设施",
        "周边景点",
        "周边餐馆",
        "名称",
        "酒店类型",
        "电话",
        "评分",
        "地址",
        "价格"
    ],
    "景点": [
        "地址",
        "周边餐馆",
        "电话",
        "门票",
        "评分",
        "游玩时间",
        "名称",
        "周边景点",
        "周边酒店"
    ],
    "地铁": [
        "出发地",
        "目的地",
        "出发地附近地铁站",
        "目的地附近地铁站"
    ],
    "餐馆": [
        "评分",
        "人均消费",
        "周边餐馆",
        "周边酒店",
        "地址",
        "电话",
        "名称",
        "推荐菜",
        "营业时间",
        "周边景点"
    ]
}  # used in preprocess.py preprocess2.1.py find_oov_slot.py create_oov_slot.py eval.py dst.py
# count: 72

all_infslot = [
    "人均消费",
    "价格",
    "出发地",
    "出发地附近地铁站",
    "名称",
    "周边景点",
    "周边酒店",
    "周边餐馆",
    "地址",
    "推荐菜",
    "游玩时间",
    "电话",
    "目的地",
    "目的地附近地铁站",
    "营业时间",
    "评分",
    "车型",
    "车牌",
    "酒店类型",
    "酒店设施",
    "门票"
]  # used in dst.py
# count: 57

all_slots = list(set(all_reqslot + all_infslot))  # used in reader.py utils.py dst.py
# count: 57
# inf 比 req 多了 "目的地"、"出发地" 两个slots

facility_list=[
    "接站服务",
    "叫醒服务",
    "接机服务",
    "看护小孩服务",
    "吹风机",
    "无烟房",
    "免费市内电话",
    "收费停车位",
    "租车",
    "24小时热水",
    "国际长途电话",
    "桑拿",
    "暖气",
    "中式餐厅",
    "公共区域提供wifi",
    "所有房间提供wifi",
    "早餐服务",
    "接待外宾",
    "棋牌室",
    "公共区域和部分房间提供wifi",
    "酒店各处提供wifi",
    "免费国内长途电话",
    "会议室",
    "商务中心",
    "西式餐厅",
    "酒吧",
    "宽带上网",
    "行李寄存",
    "温泉",
    "健身房",
    "室外游泳池",
    "洗衣服务",
    "部分房间提供wifi",
    "残疾人设施",
    "早餐服务免费",
    "spa",
    "室内游泳池"
]


get_slot = {}  # used in reader.py
for s in all_slots:
    get_slot[s] = 1





# TODO: the dict below should be adaptive to crosswoz
# mapping slots in dialogue act to original goal slot names
da_abbr_to_slot_name = {
    'addr': "address",
    'fee': "price",
    'post': "postcode",
    'ref': 'reference',
    'ticket': 'price',
    'depart': "departure",
    'dest': "destination",
}

# TODO: change things related to dialog acts ->crosswoz
dialog_acts = {
    'restaurant': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'hotel': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'attraction': ['inform', 'request', 'nooffer', 'recommend', 'select'],
    'train': ['inform', 'request', 'nooffer', 'offerbook', 'offerbooked', 'select'],
    'taxi': ['inform', 'request'],
    'police': ['inform', 'request'],
    'hospital': ['inform', 'request'],
    # 'booking': ['book', 'inform', 'nobook', 'request'],
    'general': ['bye', 'greet', 'reqmore', 'welcome'],
}
all_acts = []
for acts in dialog_acts.values():
    for act in acts:
        if act not in all_acts:
            all_acts.append(act)
# print(all_acts)

dialog_act_params = {
    'inform': all_slots + ['choice', 'open'],
    'request': all_infslot + ['choice', 'price'],
    'nooffer': all_slots + ['choice'],
    'recommend': all_reqslot + ['choice', 'open'],
    'select': all_slots + ['choice'],
    # 'book': ['time', 'people', 'stay', 'reference', 'day', 'name', 'choice'],
    'nobook': ['time', 'people', 'stay', 'reference', 'day', 'name', 'choice'],
    'offerbook': all_slots + ['choice'],
    'offerbooked': all_slots + ['choice'],
    'reqmore': [],
    'welcome': [],
    'bye': [],
    'greet': [],
}

# dialog_acts = ['inform', 'request', 'nooffer', 'recommend', 'select', 'book', 'nobook', 'offerbook', 'offerbooked',
#                         'reqmore', 'welcome', 'bye', 'greet'] # thank
dialog_act_all_slots = all_slots + ['choice', 'open']
# act_span_vocab = ['['+i+']' for i in dialog_act_dom] + ['['+i+']' for i in dialog_acts] + all_slots

# value_token_in_resp = ['address', 'name', 'phone', 'postcode', 'area', 'food', 'pricerange', 'id',
#                                      'department', 'place', 'day', 'count', 'car']
# count: 12


# special slot tokens in belief span
# no need of this, just covert slot to [slot] e.g. pricerange -> [pricerange]
slot_name_to_slot_token = {}

# special slot tokens in responses
# not use at the momoent
slot_name_to_value_token = {
    # 'entrance fee': '[value_price]',
    # 'pricerange': '[value_price]',
    # 'arriveby': '[value_time]',
    # 'leaveat': '[value_time]',
    # 'departure': '[value_place]',
    # 'destination': '[value_place]',
    # 'stay': 'count',
    # 'people': 'count'
}

# todo: 可以向学长确认一下以下内容是否无需修改

# database
db_tokens = ['<sos_db>', '<eos_db>', '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']

special_tokens = ['<pad>', '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>',
                  '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>'] + db_tokens

eos_tokens = {
    'user': '<eos_u>', 'user_delex': '<eos_u>',
    # 为什么要对user进行delex？因为训练序列是先不去词汇化的，再接一个去词汇化的
    'resp': '<eos_r>', 'resp_gen': '<eos_r>', 'pv_resp': '<eos_r>',
    # 为什么没有resp_delex? 因为所有resp都是delex后的 这里的resp相当于resp_delex
    # pv应该是previous 但好像代码里没有用到pv相关的
    'bspn': '<eos_b>', 'bspn_gen': '<eos_b>', 'pv_bspn': '<eos_b>',
    # belief state
    'bsdx': '<eos_b>', 'bsdx_gen': '<eos_b>', 'pv_bsdx': '<eos_b>',
    # belef state delex
    'aspn': '<eos_a>', 'aspn_gen': '<eos_a>', 'pv_aspn': '<eos_a>',
    # dialog act
    'dspn': '<eos_d>', 'dspn_gen': '<eos_d>', 'pv_dspn': '<eos_d>'}
# domain-slot pair

sos_tokens = {
    'user': '<sos_u>', 'user_delex': '<sos_u>',
    'resp': '<sos_r>', 'resp_gen': '<sos_r>', 'pv_resp': '<sos_r>',
    'bspn': '<sos_b>', 'bspn_gen': '<sos_b>', 'pv_bspn': '<sos_b>',
    'bsdx': '<sos_b>', 'bsdx_gen': '<sos_b>', 'pv_bsdx': '<sos_b>',
    'aspn': '<sos_a>', 'aspn_gen': '<sos_a>', 'pv_aspn': '<sos_a>',
    'dspn': '<sos_d>', 'dspn_gen': '<sos_d>', 'pv_dspn': '<sos_d>'}
