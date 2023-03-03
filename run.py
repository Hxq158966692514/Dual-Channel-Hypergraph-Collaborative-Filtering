#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 10:02
#@Author : 韩笑奇
#@File : run.py
#@Software: PyCharm

from buliddata import build_data

from utils import bulid_HBUK

import os

# 自封装的    DHCF  代码

from config import Config

from model_dhcf import DHCF

import scipy.sparse as sp

from sampler import sample

import torch

from loss_function import loss_func

import numpy as np

from test_model import test


if __name__ == '__main__':


    configs = Config()



    # 第一步进行数据集的创建

    dataset = './Data/gowalla'

    if not os.path.exists('./data_adj/adj_matrix.npz'):

        build_data(dataset,configs)

    # 进行 k阶超图 相关矩阵的构建


    if not os.path.exists('./data_adj/HI_adj.npz'):

        bulid_HBUK(configs)


    # config数据填充

    with open('./config_data/config_detail.txt','r') as f:

        x = f.readlines()

        configs.user_n = eval(x[0].strip())

        configs.item_n = eval(x[1].strip())

        configs.train_item_n = eval(x[2].strip())

        configs.user_item_dict = eval(x[3].strip())

        f.close()



    # 进行模型的构建


    # 加载HU,与HI

    HU = sp.load_npz('./data_adj/HU_adj.npz')

    HI = sp.load_npz('./data_adj/HI_adj.npz')


    model = DHCF(configs,HU,HI)


    # 开始进行模型的训练

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)


    best_loss = np.inf

    for epoch in range(1):#configs.epochs):

        # 在进行训练之前需要进行采样操作   随机采样     只有这样才能进行 bpr loss 的计算

        user_list, pos_item_list, neg_item_list = sample(configs)

        optimizer.zero_grad()

        user_emb_t,pos_emb_t,neg_emb_t = model(user_list, pos_item_list, neg_item_list,configs.layer_size,configs.k,configs.device)

        # 计算损失函数  使用bprloss


        loss = loss_func(user_list, pos_item_list, neg_item_list,user_emb_t,pos_emb_t,neg_emb_t,model,configs)

        loss.backward()

        optimizer.step()

        if loss < best_loss:

            print(f'{epoch}/{configs.epochs},loss:{loss}')

            torch.save(model.state_dict(),'./model_saved/hxq_dhcf_model.cpkt')

    # 接下来进行模型的测试

    user_list, pos_item_list, neg_item_list = sample(configs)

    ret=test(user_list, pos_item_list, neg_item_list,model,configs)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    rec_loger.append(ret['recall'])
    pre_loger.append(ret['precision'])
    ndcg_loger.append(ret['ndcg'])
    hit_loger.append(ret['hit_ratio'])

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best  precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)


















