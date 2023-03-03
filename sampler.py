#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 15:40
#@Author : 韩笑奇
#@File : sampler.py
#@Software: PyCharm

import numpy as np

from collections import defaultdict

def sample(configs):

    user_arr = np.array(list(configs.user_item_dict.keys()), dtype=np.int32)  # 拿出训练集的所有用户
    user_list = np.random.choice(user_arr, size=configs.train_item_n, replace=True)  # 构造 项目大小的 随机用户 列表

    user_pos_len = defaultdict(int)  # 记录所有用户的id与出现次数
    for u in user_list:
        user_pos_len[u] += 1

    user_pos_sample = dict()
    user_neg_sample = dict()
    for user, pos_len in user_pos_len.items():
        pos_item = configs.user_item_dict[user]
        pos_idx = np.random.choice(pos_item, size=pos_len, replace=True)
        user_pos_sample[user] = list(pos_idx)

        neg_item = np.random.randint(low=0, high=configs.item_n, size=pos_len)
        for i in range(len(neg_item)):
            idx = neg_item[i]
            while idx in pos_item:
                idx = np.random.randint(low=0, high=configs.item_n)
            neg_item[i] = idx
        user_neg_sample[user] = list(neg_item)

    pos_item_list = [user_pos_sample[user].pop() for user in user_list]  # 刚好遍历完所有
    neg_item_list = [user_neg_sample[user].pop() for user in user_list]  # 刚好干完所有的数
    return user_list, pos_item_list, neg_item_list