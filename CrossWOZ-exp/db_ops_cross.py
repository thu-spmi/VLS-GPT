import json, random, sqlite3
from ontology_cross import all_domains_cross, db_domains_cross,domains_cross_switch
import os
import re
def contains(arr, s):
    return not len(tuple(filter(lambda item: (not (item.find(s) < 0)), arr)))
       
class CrossWozDB(object):
    def __init__(self,db_paths):
        super(CrossWozDB, self).__init__()
        self.data = {}
        self.sql_dbs = {}
        for domain in db_domains_cross:
            with open(db_paths[domain], 'r',encoding= "utf-8") as f:
                self.data[domains_cross_switch[domain]] = json.loads(f.read().lower())#.decode('utf-8')
        """
        self.data = {}
        db_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),'../../../../data/crosswoz/database'))
        with open(os.path.join(db_dir, 'metro_db.json'), 'r', encoding='utf-8') as f:
            self.data['地铁'] = json.load(f)
        with open(os.path.join(db_dir, 'hotel_db.json'), 'r', encoding='utf-8') as f:
            self.data['酒店'] = json.load(f)
        with open(os.path.join(db_dir, 'restaurant_db.json'), 'r', encoding='utf-8') as f:
            self.data['餐馆'] = json.load(f)
        with open(os.path.join(db_dir, 'attraction_db.json'), 'r', encoding='utf-8') as f:
            self.data['景点'] = json.load(f)
        """
        self.schema = {
            '景点': {
                '名称': {'params': None},
                '门票': {'type': 'between', 'params': [None, None]},
                '游玩时间': {'params': None},
                '评分': {'type': 'between', 'params': [None, None]},
                '周边景点': {'type': 'in', 'params': None},
                '周边餐馆': {'type': 'in', 'params': None},
                '周边酒店': {'type': 'in', 'params': None},
            },
            '餐馆': {
                '名称': {'params': None},
                '推荐菜': {'type': 'multiple_in', 'params': None},
                '人均消费': {'type': 'between', 'params': [None, None]},
                '评分': {'type': 'between', 'params': [None, None]},
                '周边景点': {'type': 'in', 'params': None},
                '周边餐馆': {'type': 'in', 'params': None},
                '周边酒店': {'type': 'in', 'params': None}
            },
            '酒店': {
                '名称': {'params': None},
                '酒店类型': {'params': None},
                '酒店设施': {'type': 'multiple_in', 'params': None},
                '价格': {'type': 'between', 'params': [None, None]},
                '评分': {'type': 'between', 'params': [None, None]},
                '周边景点': {'type': 'in', 'params': None},
                '周边餐馆': {'type': 'in', 'params': None},
                '周边酒店': {'type': 'in', 'params': None}
            },
            '地铁': {
                '起点': {'params': None},
                '终点': {'params': None},
            },
            '出租': {
                '起点': {'params': None},
                '终点': {'params': None},
            }
        }
    def query( self,  cur_domain, belief_state, exactly_match=True, return_name=False):
        """
        query database using belief state, return list of entities, same format as database
        :param belief_state: state['belief_state']
        :param cur_domain: maintain by DST, current query domain
        :return: list of entities
        """
        if not cur_domain:
            return []
        cur_query_form = {}
        for slot, value in belief_state.items():
            if not value:
                continue
            if slot == '出发地':
                slot = '起点'
            elif slot == '目的地':
                slot = '终点'
            if slot == '名称':
                # DONE: if name is specified, ignore other constraints
                cur_query_form = {'名称': value}
                break
            elif slot == '评分':
                if re.match('(\d\.\d|\d)', value):
                    if re.match('\d\.\d', value):
                        score = float(re.match('\d\.\d', value)[0])
                    else:
                        score = int(re.match('\d', value)[0])
                    cur_query_form[slot] = [score, None]
                # else:
                #     assert 0, value
            elif slot in ['门票', '人均消费', '价格']:
                low, high = None, None
                if re.match('(\d+)-(\d+)', value):
                    low = int(re.match('(\d+)-(\d+)', value)[1])
                    high = int(re.match('(\d+)-(\d+)', value)[2])
                elif re.match('\d+', value):
                    if '以上' in value:
                        low = int(re.match('\d+', value)[0])
                    elif '以下' in value:
                        high = int(re.match('\d+', value)[0])
                    else:
                        low = high = int(re.match('\d+', value)[0])
                elif slot == '门票':
                    if value == '免费':
                        low = high = 0
                    elif value == '不免费':
                        low = 1
                    else:
                        pass
                        # print(value)
                        # assert 0
                cur_query_form[slot] = [low, high]
            else:
                cur_query_form[slot] = value
        cur_res = self.query_schema(field=cur_domain, args=cur_query_form)
        if cur_domain == '出租':
            res = [cur_res]
        elif cur_domain == '地铁':
            res = []
            for r in cur_res:
                if not res and '起点' in r[0]:
                    res.append(r)
                    break
            for r in cur_res:
                if '终点' in r[0]:
                    res.append(r)
                    break
        else:
            res = cur_res

        return res

    def query_schema(self, field, args):
        if not field in self.schema:
            # if domain not in schema, then return []
            return []
        if not isinstance(args, dict):
            return []
        db = self.data.get(field)
        plan = self.schema[field]
        for key, value in args.items():
            if key!='selectedresults':
                if not key in plan:
                    continue
                value_type = plan[key].get('type')
                if value_type == 'between':
                    if not value[0] is None:
                        plan[key]['params'][0] = float(value[0])
                    if not value[1] is None:
                        plan[key]['params'][1] = float(value[1])
                else:
                    if not isinstance(value, str):
                        continue
                    plan[key]['params'] = value
        if field in ['地铁', '出租']:
            s = plan['起点']['params']
            e = plan['终点']['params']
            if not s or not e:
                return []
            if field == '出租':
                return [
                    '出租 (%s - %s)' % (s, e), {
                        '领域': '出租',
                        '车型': '#CX',
                        '车牌': '#CP'
                    }
                ]
            else:
                def func1(item):
                    if item[0].find(s) >= 0:
                        return ['(起点) %s' % item[0], item[1]]

                def func2(item):
                    if item[0].find(e) >= 0:
                        return ['(终点) %s' % item[0], item[1]]
                    return None

                return list(filter(lambda item: not item is None, list(map(func1, db)))) + list(
                    filter(lambda item: not item is None, list(map(func2, db))))

        def func3(item):
            details = item[1]
            for key, val in args.items():
                if key!='selectedresults':
                    val = details.get(key)
                    absence = val is None
                    if key not in plan:
                        continue
                    options = plan[key]
                    if options.get('type') == 'between':
                        L = options['params'][0]
                        R = options['params'][1]
                        if not L is None:
                            if absence:
                                return False
                        else:
                            L = float('-inf')
                        if not R is None:
                            if absence:
                                return False
                        else:
                            R = float('inf')
                        if L > val or val > R:
                            return False
                    elif options.get('type') == 'in':
                        s = options['params']
                        if not s is None:
                            if absence:
                                return False
                            if contains(val, s):
                                return False
                    elif options.get('type') == 'multiple_in':
                        s = options['params']
                        if not s is None:
                            if absence:
                                return False
                            sarr = list(filter(lambda t: bool(t), s.split(' ')))
                            if len(list(filter(lambda t: contains(val, t), sarr))):
                                return False
                    else:
                        s = options['params']
                        if not s is None:
                            if absence:
                                return False
                            if val.find(s) < 0:
                                return False
            return True

        return list(filter(func3, db))
    


    def oneHotVector(self, domain, num):
        """Return number of available entities for particular domain."""
        vector = [0,0,0,0]
        if num == '':
            return vector
        if domain != 'train':
            if num == 0:
                vector = [1, 0, 0, 0]
            elif num == 1:
                vector = [0, 1, 0, 0]
            elif num <=3:
                vector = [0, 0, 1, 0]
            else:
                vector = [0, 0, 0, 1]
        else:
            if num == 0:
                vector = [1, 0, 0, 0]
            elif num <= 5:
                vector = [0, 1, 0, 0]
            elif num <=10:
                vector = [0, 0, 1, 0]
            else:
                vector = [0, 0, 0, 1]
        return vector


    def addBookingPointer(self, turn_da):
        """Add information about availability of the booking option."""
        # Booking pointer
        # Do not consider booking two things in a single turn.
        vector = [0, 0]
        if turn_da.get('booking-nobook'):
            vector = [1, 0]
        if turn_da.get('booking-book') or turn_da.get('train-offerbooked'):
            vector = [0, 1]
        return vector


    def addDBPointer(self, domain, match_num, return_num=False):
        """Create database pointer for all related domains."""
        # if turn_domains is None:
        #     turn_domains = db_domains
        if domain in all_domains_cross:
            vector = self.oneHotVector(domain, match_num)
        else:
            vector = [0, 0, 0 ,0]
        return vector

    def addDBIndicator(self, domain, match_num, return_num=False):
        """Create database indicator for all related domains."""
        # if turn_domains is None:
        #     turn_domains = db_domains
        if domain in all_domains_cross:
            vector = self.oneHotVector(domain, match_num)
        else:
            vector = [0, 0, 0 ,0]
        
        # '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]'
        if vector == [0,0,0,0]:
            indicator = '[db_nores]'
        else:
            indicator = '[db_%s]' % vector.index(1)
        return indicator

    def get_match_num(self, constraints, act=None,return_entry=False):
        """Create database pointer for all related domains."""
        match = {'greet': '','welcome': '','bye': '','thank': '','reqmore': '','general': ''}
        entry = {}
        # if turn_domains is None:
        #     turn_domains = db_domains           
        for domain in all_domains_cross:
            match[domain] = ''  
            if  constraints.get(domain):
                matched_ents = self.query(domain, constraints[domain])
                match[domain] = len(matched_ents)
                if return_entry :
                    entry[domain] = matched_ents
            if act:
                if act[2]=='周边餐馆' or act[2]=='周边景点' or act[2]=='周边酒店':
                    if act[3]=='无':
                        match[act[1]] = 0
                        entry[act[1]] = act[3]
                    else:
                        match[act[1]] = 1
                        entry[act[1]] = act[3]
        if return_entry:
            return entry
        return match


    def pointerBack(self, vector, domain):
        # multi domain implementation
        # domnum = cfg.domain_num
        if domain.endswith(']'):
            domain = domain[1:-1]
        if domain != 'train':
            nummap = {
                0: '0',
                1: '1',
                2: '2-3',
                3: '>3'
            }
        else:
            nummap = {
                0: '0',
                1: '1-5',
                2: '6-10',
                3: '>10'
            }
        if vector[:4] == [0,0,0,0]:
            report = ''
        else:
            num = vector.index(1)
            report = domain+': '+nummap[num] + '; '

        if vector[-2] == 0 and vector[-1] == 1:
            report += 'booking: ok'
        if vector[-2] == 1 and vector[-1] == 0:
            report += 'booking: unable'

        return report

    def queryJsons(self, domain, constraints, exactly_match=True, return_name=False):
        # need to be more specified
        # query the db   
        if domain == '出租':
            return self.dbs[domain][0][1]
        valid_cons = False
        for v in constraints.values():
            if v not in ["not mentioned", ""]:
                valid_cons = True
        if not valid_cons:
            return []

        match_result = []
        if constraints.get('selectedresults'):
            for db_ent in self.dbs[domain]:
                if db_ent[0]==constraints['selectedresults']:
                    match_result.append(db_ent[1])
        else:
            for db_ent in self.dbs[domain]:
                match = True
                for s, v in constraints.items():
                    v_fix=v.replace(' ', '')
                    ent_v_fix=db_ent[s].repalce(' ', '')
                    #print(v_fix, ent_v_fix)
                    if s not in db_ent:
                        # logging.warning('Searching warning: slot %s not in %s db'%(s, domain))
                        match = False
                        break
                    if ent_v_fix!=v_fix:
                        match = False
                if(match == True):
                   match_result.append(db_ent[1])

        return match_result
        """
        for db_ent in self.dbs[domain]:
            match = True#if slots only have name then match??
            for s, v in constraints.items():
                if s == 'name':
                    continue
                skip_case = {"don't care":1, "do n't care":1, "dont care":1, "not mentioned":1, "dontcare":1, "":1}
                if skip_case.get(v):
                    continue
                if s not in db_ent:
                    # logging.warning('Searching warning: slot %s not in %s db'%(s, domain))
                    match = False
                    break
                # v = 'guesthouse' if v == 'guest house' else v
                # v = 'swimmingpool' if v == 'swimming pool' else v
                v = 'yes' if v == 'free' else v
                if s in ['arrive', 'leave']:
                    try:
                        h,m = v.split(':')   # raise error if time value is not xx:xx format
                        v = int(h)*60+int(m)
                    except:
                        match = False
                        break
                    if db_ent[s]!='?':
                        time = int(db_ent[s].split(':')[0])*60+int(db_ent[s].split(':')[1])
                        #find a train or taxi that can be caught
                        if s == 'arrive' and v>time:
                            match = False
                        if s == 'leave' and v<time:
                            match = False
                else:
                    if exactly_match and v != db_ent[s]:
                        match = False
                        break
                    elif v not in db_ent[s]:
                        match = False
                        break
            if match:
                match_result.append(db_ent)
        if not return_name:
            return match_result
        else:
            if domain == 'train':
                match_result = [e['id'] for e in match_result]
            else:
                match_result = [e['name'] for e in match_result]
            return match_result
        """

    def querySQL(self, domain, constraints):
        if not self.sql_dbs:
            for dom in db_domains_cross:
                db = 'db/{}-dbase.db'.format(dom)
                conn = sqlite3.connect(db)
                c = conn.cursor()
                self.sql_dbs[dom] = c

        sql_query = "select * from {}".format(domain)


        flag = True
        for key, val in constraints.items():
            if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    # val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    # val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

        try:  # "select * from attraction  where name = 'queens college'"
            #print(sql_query)
            return self.sql_dbs[domain].execute(sql_query).fetchall()
        except:
            return []  # TODO test it


if __name__ == '__main__':
    dbPATHs = {
            'attraction': 'database/attraction_db.json',
            'hotel': 'database/hotel_db.json',
            'metro': 'database/metro_db.json',
            'restaurant': 'database/restaurant_db.json',
            'taxi': 'database/taxi_db.json',
        }
    db = CrossWozDB(dbPATHs)
    while True:
        constraints = {}
        inp = input('input belief state in fomat: domain-slot1=value1;slot2=value2...\n')
        domain, cons = inp.split('-')
        for sv in cons.split(';'):
            s, v = sv.split('=')
            constraints[s] = v
        # res = db.querySQL(domain, constraints)
        res = db.queryJsons(domain, constraints, return_name=True)
        report = []
        reidx = {
            'hotel': 8,
            'restaurant': 6,
            'attraction':5,
            'train': 1,
        }
        # for ent in res:
        #     if reidx.get(domain):
        #         report.append(ent[reidx[domain]])
        # for ent in res:
        #     if 'name' in ent:
        #         report.append(ent['name'])
        #     if 'trainid' in ent:
        #         report.append(ent['trainid'])
        print(constraints)
        print(res)
        print('count:', len(res), '\nnames:', report)

