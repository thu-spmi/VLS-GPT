import json
from sklearn.metrics import f1_score, accuracy_score
import sys
import numpy as np
import csv
from dst import ignore_none, default_cleaning, parser_bs
import argparse


def load_result(result_path):
    results = []
    with open(result_path, 'r') as rf:
        reader = csv.reader(rf)
        for n, line in enumerate(reader):
            entry = {}
            if n > 0:
                if n == 1:
                    field = line
                else:
                    for i, key in enumerate(field):
                        entry[key] = line[i]
                    results.append(entry)
    return results, field


def compute_jacc(data, return_db=False, mode='SlotPRF1'):
    '''

    :param data: <list of dict> each dict is as follows:
    e.g. {'dial_id': '8267', 'turn_num': '0', 'user': '您 好 ， 我 老 妈 想 吃 干 炸 素 丸 子 ， 请 帮 我 找 一 家 评分 是 4. 5 分 以 上 的 有 这 道 菜 的 餐 馆 。 提 供 给 我 电话 和 周边酒店 信 息 ， 谢 了 。 ', 'bspn_gen': '[景点] 门票 20 - 50 元 游玩时间 2 小 时 - 3 小 时 评分 4. 5 分 以 上 [餐馆] 推荐菜 干 炸 素 丸 子 评分 4 分 以 上', 'bsdx': '[餐馆] 推荐菜 评分 ', 'resp_gen': '为 您 推 荐 [value_name] ， 评分 [value_score] ， 电话 是 [ value _ phone ] ， 周边酒店 有 [value_name] 、 [value_address] 、 [value_name] 等 。', 'resp': '为 您 推 荐 [value_name] ， 电话 [ value _ phone ] ， 周边酒店 有 [value_name] 、 [value_name] 等 。 ', 'aspn_gen': '', 'aspn': '[餐馆] [ inform ] 电话 ', 'dspn_gen': '', 'dspn': '[餐馆] ', 'bspn': '[餐馆] 推荐菜 干 炸 素 丸 子 评分 3 分 以 上 ', 'pointer': '餐馆: 1;'}
    :param return_db: <boolean>
    :param mode:
    :return:
    '''
    # 1. 'JointGoalAcc' mode -- Joint Accuracy: evaluates whether the predicted dialog state is exactly equal to the ground truth. core equation: set(turn_target) == set(turn_pred)
    # 2. 'SlotPRF1' mode -- Slot Accuracy: evaluates the prediction for non-empty slots only, micro-averaged over all slots
    # P.S.
    # 1. 目前没做[SEP]的cleaning:是否需要做？or去掉？
    # CHECK:  1. 目前是不是没有db和db_gen？没有的话可以去掉？ 2. parser_bs is from dst.py (目前酒店设施还不能正常识别)
    if mode == 'JointGoalAcc':
        num_turns = 0
        joint_acc = 0
        db_acc = 0
        for turn_data in data:
            if turn_data['user'] == '':
                continue
            turn_target = turn_data['bspn']
            turn_pred = turn_data['bspn_gen']
            turn_target = parser_bs(turn_target)
            turn_pred = parser_bs(turn_pred)
            turn_target=[' '.join(item) for item in turn_target]
            turn_pred=[' '.join(item) for item in turn_pred]
            if set(turn_target) == set(turn_pred):
                joint_acc += 1
            num_turns += 1
            if return_db:
                if turn_data['db'] == turn_data['db_gen']:
                    db_acc += 1
        joint_acc /= num_turns
        db_acc /= num_turns
        # print('joint accuracy: {}'.format(joint_acc))
        if return_db:
            return {'precision': joint_acc, 'db_acc': db_acc}
        else:
            return {'precision': joint_acc}
    if mode == 'SlotPRF1':
        # init
        TP, FP, FN = 0, 0, 0
        correct_db = 0
        for turn_data in data:
            if turn_data['user'] == '':
                continue
            if return_db:
                if turn_data['db'] == turn_data['db_gen']:
                    correct_db += 1
            turn_target = turn_data['bspn']
            turn_pred = turn_data['bspn_gen']
            turn_target = parser_bs(turn_target)
            turn_pred = parser_bs(turn_pred)
            for ele in turn_pred:
                if ele in turn_target:
                    TP += 1
                else:
                    FP += 1
            for ele in turn_target:
                if ele not in turn_pred:
                    FN += 1
        precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
        recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
        F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
        print('Slot metrics  ', {'precision': precision, 'recall': recall, 'F1': F1})
        return {'precision': precision, 'recall': recall, 'F1': F1}


def find_case(data1, data2):
    # data1: sup-only results
    # data2: semi-train results
    clean_tokens = ['<|endoftext|>', ]
    for turn_data1, turn_data2 in zip(data1, data2):
        if turn_data1['user'] == '':
            continue
        assert turn_data1['bspn'] == turn_data2['bspn']
        turn_target = turn_data1['bspn']
        turn_pred1 = turn_data1['bspn_gen']
        turn_pred2 = turn_data2['bspn_gen']
        turn_target = parser_bs(turn_target)
        turn_pred1 = parser_bs(turn_pred1)
        turn_pred2 = parser_bs(turn_pred2)
        for turn_pred in [turn_pred1, turn_pred2]:
            for bs in turn_pred:
                if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                    turn_pred.remove(bs)
            new_turn_pred = []
            for bs in turn_pred:
                for tok in clean_tokens:
                    bs = bs.replace(tok, '').strip()
                    new_turn_pred.append(bs)
            turn_pred = new_turn_pred
            # turn_pred, turn_target = ignore_none(turn_pred, turn_target)
            # turn_pred, turn_target = default_cleaning(turn_pred, turn_target)
        if set(turn_target) != set(turn_pred1) and set(turn_target) == set(turn_pred2):
            print(turn_data1['dial_id'])
            print('ground truth: \n u:{} \n b:{} \n r:{}'.format(turn_data1['user'], turn_data1['bspn'],
                                                                 turn_data1['resp']))
            print('sup only:\n b:{} \n r:{}'.format(turn_data1['bspn_gen'], turn_data1['resp_gen']))
            print('semi VL:\n b:{} \n r:{}'.format(turn_data2['bspn_gen'], turn_data2['resp_gen']))


if __name__ == "__main__":
    '''
    for s in ['10','20','30','40','50']:
        path='experiments/all_pre_{}_sd11_lr0.0001_bs2_ga16/best_score_model/result.csv'.format(s)
        results,field=load_result(path)
        joint_acc=compute_jacc(results)
        print('proportion:{}%, joint goal after pretrain:{}'.format(s,joint_acc))
    for s in ['10','20','30','40','50']:
        path='experiments/all_ST_{}_sd11_lr2e-05_bs2_ga16/best_score_model/result.csv'.format(s)
        results,field=load_result(path)
        joint_acc=compute_jacc(results)
        print('proportion:{}%, joint goal after selftrain:{}'.format(s,joint_acc))
    for s in ['10','20','30']:
        path='experiments/all_VL_{}_sd11_lr2e-05_bs2_ga16/best_score_model/result1.csv'.format(s)
        results,field=load_result(path)
        joint_acc=compute_jacc(results)
        print('proportion:{}%, joint goal after VLtrain:{}'.format(s,joint_acc))
    path='/home/liuhong/UBAR/experiments/all_semi_316_sd11_lr2e-05_bs2_ga16/best_score_model/result1.csv'
    results,field=load_result(path)
    joint_acc=compute_jacc(results)
    print('proportion:{}%, joint goal after VLtrain:{}'.format('40',joint_acc))
    '''
    path1 = '/home/liuhong/UBAR/experiments/all_pre_40_sd11_lr0.0001_bs2_ga16/best_score_model/result1.csv'
    path2 = '/home/liuhong/UBAR/experiments/all_VL_40_sd11_lr2e-05_bs2_ga16/best_score_model/result1.csv'
    results1, _ = load_result(path1)
    results2, _ = load_result(path2)
    find_case(results1, results2)
