#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 17:07
#@Author : 韩笑奇
#@File : test_model.py
#@Software: PyCharm
#-*- codeing = utf-8 -*-
#@Time : 2023-02-27 12:28
#@Author : 韩笑奇
#@File : utils.py
#@Software: PyCharm


import random as rd

import numpy as np

import multiprocessing

from config import Config

import metrics as metrics

import heapq

import torch


def sample(config):

    users = rd.sample(config.exist_users,config.batch_size)  # 随机采样

    pos_items,neg_items = [],[]

    def sample_pos_items_for_u(u, num):
        # sample num pos items for u-th user
        pos_items = config.train_items[u]
        n_pos_items = len(pos_items)  # 用户  点击多少个   物品
        pos_batch = []
        while True:
            if len(pos_batch) == num:
                break
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]

            if pos_i_id not in pos_batch:
                pos_batch.append(pos_i_id)
        return pos_batch

    def sample_neg_items_for_u(u, num):
        # sample num neg items for u-th user
        neg_items = []
        while True:
            if len(neg_items) == num:
                break
            neg_id = np.random.randint(low=0, high=config.n_item, size=1)[0]
            if neg_id not in config.train_items[u] and neg_id not in neg_items:
                neg_items.append(neg_id)
        return neg_items

    for u in users:
        pos_items += sample_pos_items_for_u(u, 1)
        neg_items += sample_neg_items_for_u(u, 1)

    return users,pos_items,neg_items

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)   # 前100物品的id

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K)) # 前20个  打分最大的物品中平均几个是购买过的
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))      # 前20个中能命中的比率    就是用户购买的物品 在前二十个物品中的比例
        hit_ratio.append(metrics.hit_at_k(r, K))  # 前二十个最大分的物品中是否存在命中

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x,config,Ks):


    # user u's ratings for user u
    rating = x[0]    # 用户对所有的物品的打分值
    #uid
    u = x[1]     # 打分用户
    #user u's items in the training set
    try:
        training_items = config.user_item_dict[u]    # 获得当前 2748 用户所有的 购买的物品
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = config.user_item_dict[u]

    all_items = set(range(config.item_n))

    test_items = list(all_items - set(training_items))   # 所有 用户未买的物品

    if config.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)
def rating(u_g_embeddings, pos_i_g_embeddings):

    return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

def test(user_list, pos_item_list, neg_item_list,model,config,drop_flag=False):

    model.load_state_dict(torch.load('./model_saved/hxq_dhcf_model.cpkt'))

    model.eval()

    ks = [20,40,60,80,100]

    cores = multiprocessing.cpu_count() // 2


    result = {

        'precision' : np.zeros(len(ks)),

        'recall': np.zeros(len(ks)),

        'ndcg': np.zeros(len(ks)),

        'hit_ratio': np.zeros(len(ks)),

        'auc': 0.
    }
    count = 0

    u_g_embeddings, pos_i_g_embeddings, _ = model(user_list, pos_item_list, neg_item_list,config.layer_size,config.k,config.device)


    rate_batch = rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

    user_batch = user_list


    user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)

    batch_result = []

    n_test_users = len(user_list)

    for tu in user_batch_rating_uid:
        batch_result.append(test_one_user(tu, config, ks))

    count += len(batch_result)

    for re in batch_result:  # re等于5的原因是 Ks = 5
        result['precision'] += re['precision'] / n_test_users
        result['recall'] += re['recall'] / n_test_users
        result['ndcg'] += re['ndcg'] / n_test_users
        result['hit_ratio'] += re['hit_ratio'] / n_test_users
        result['auc'] += re['auc'] / n_test_users

    return result
