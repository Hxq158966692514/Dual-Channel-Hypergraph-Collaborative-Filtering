#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 10:04
#@Author : 韩笑奇
#@File : buliddata.py
#@Software: PyCharm

import os

from collections import defaultdict

import scipy.sparse as sp

import numpy as np


def create_matrix(user_n,item_n,user_item_dict):


    adj = sp.dok_matrix((user_n+1,item_n+1),dtype=np.float32)


    for k,v in user_item_dict.items():

        for va in v:

            adj[k,va] = 1


    return adj






def build_data(dataset,configs):



    user_n = 0

    item_n = 0

    train_item_n = 0

    user_item_dict = defaultdict()
    time = 0
    with open(os.path.join(dataset,'train.txt')) as f:

        for line in f.readlines():

            time+=1

            if line == '':
                continue

            x = line.strip().split(' ')

            if len(x)>2:

                userid = int(x[0])

                items = [int(i) for i in x[1:]]

                user_n = max(user_n,userid)

                item_n = max(item_n,max(items))

                user_item_dict[userid] = items

                train_item_n += len(items)
            if time > 100:

                break



    # 构建 二部图矩阵

    adj_matrxi = create_matrix(user_n,item_n,user_item_dict)

    # 进行矩阵的存储

    sp.save_npz('./data_adj/adj_matrix.npz',adj_matrxi.tocsr())


    with open('./config_data/config_detail.txt','w') as f:

        f.write(str(user_n + 1)+'\n')

        f.write(str(item_n + 1)+'\n')

        f.write(str(train_item_n)+'\n')

        f.write(str(dict(user_item_dict))+'\n')

        f.close()




