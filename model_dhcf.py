#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 12:50
#@Author : 韩笑奇
#@File : model_dhcf.py
#@Software: PyCharm

import torch

import torch.nn as nn

import numpy as np

import scipy.sparse as sp

import pandas as pd


class DHCF(nn.Module):

    def __init__(self,configs,HU,HI):

        super(DHCF,self).__init__()

        self.HU = HU

        self.HI = HI

        self.weight_user_len = self.HU.shape[1]

        self.weight_item_len = self.HI.shape[1]

        # 建立初始化的特征矩阵

        self.emb_user = nn.Embedding(configs.user_n,configs.emb_size)

        self.emb_item = nn.Embedding(configs.item_n,configs.emb_size)

        # 得出可以进行训练的 权重矩阵

        self.user_weight_matrix = self.create_init_matrix(self.weight_user_len)

        self.item_weight_matrix = self.create_init_matrix(self.weight_item_len)

        # 计算结点的度

        self.user_dgree_matrix = self.user_compute_degree(self.HU)

        self.item_dgree_matrix = self.item_compute_degree(self.HI)

        # 计算超边的度

        self.user_hedge_dgree = self.compute_h_degree(self.HU)

        self.item_hedge_dgree = self.compute_h_degree(self.HI)

        # 构造全连接层

        self.embed_size = configs.emb_size

        self.embed_size_end = configs.emb_size_end

        self.fc1 = nn.Linear(self.embed_size,self.embed_size).to(device=configs.device)

        self.fc2 = nn.Linear(self.embed_size,self.embed_size_end).to(device=configs.device)





    def create_init_matrix(self,weight_len):

        dk = sp.dok_matrix((weight_len,weight_len),dtype=np.float32)

        for i in range(weight_len):

            dk[i,i] = 1.


        return dk.tocsr()

    def item_compute_degree(self,df):

        re = df.dot(self.item_weight_matrix)

        re = re.toarray()

        inv = np.sum(re,1)

        return sp.diags(inv)


    def user_compute_degree(self,df):

        re = df.dot(self.user_weight_matrix)

        re = re.toarray()

        inv = np.sum(re,1)

        return sp.diags(inv)

    def compute_h_degree(self,df):

        indexx = df.shape[0]

        columns = df.shape[1]

        df_data = pd.DataFrame(df.toarray(),index=range(indexx),columns=range(columns))

        result = []

        for col in df_data.columns:

            mid = np.sum(list(df_data.loc[:,col]))

            result.append(mid)

        return sp.diags(result)

    def trans_adj(self,D):

        rowsum = np.array(D.sum(1))

        dv_inv = np.power(rowsum,-0.5).flatten()

        dv_inv[np.isinf(dv_inv)] = 0.

        return sp.diags(dv_inv)


    def trans_e_adj(self,D_E):

        rowsum = np.array(D_E.sum(1))

        dv_inv = np.power(rowsum,-1).flatten()

        dv_inv[np.isinf(dv_inv)] = 0.

        return sp.diags(dv_inv)



    def user_normal_adj(self,df):

        H = df

        H_tr = H.T

        W = self.user_weight_matrix

        D = self.user_dgree_matrix

        D_E = self.user_hedge_dgree

        D_tr = self.trans_adj(D)

        D_E_tr = self.trans_e_adj(D_E)


        MID = D_tr.dot(H)

        MID = MID.dot(W)

        MID = MID.dot(D_E_tr)

        MID = MID.dot(H_tr)

        MID = MID.dot(D_tr)

        return MID

    def item_normal_adj(self,df):

        H = df

        H_tr = H.T

        W = self.item_weight_matrix

        D = self.item_dgree_matrix

        D_E = self.item_hedge_dgree

        D_tr = self.trans_adj(D)

        D_E_tr = self.trans_e_adj(D_E)


        MID = D_tr.dot(H)

        MID = MID.dot(W)

        MID = MID.dot(D_E_tr)

        MID = MID.dot(H_tr)

        MID = MID.dot(D_tr)

        return MID

    def forward(self,user_list, pos_item_list, neg_item_list,layer_size,k,device):


        # 进行归一化操作

        user_adj_norm = self.user_normal_adj(self.HU)

        item_adj_norm = self.item_normal_adj(self.HI)

        user_adj_norm = torch.Tensor(user_adj_norm.toarray()).to(device=device,dtype=torch.float32)/k

        user_adj_norm.requires_grad = True

        item_adj_norm = torch.Tensor(item_adj_norm.toarray()).to(device=device,dtype=torch.float32)/k

        item_adj_norm.requires_grad = True



        MU = self.emb_user.weight.to(device=device)

        MI = self.emb_item.weight.to(device=device)

        for layer in range(layer_size):

            MU_mid = torch.matmul(user_adj_norm,MU)

            MI_mid = torch.matmul(item_adj_norm,MI)

            MU_mid = self.fc1(MU_mid)

            MI_mid = self.fc1(MI_mid)

            MU = MU_mid + MU

            MI = MI_mid + MI

            MU = MU.relu()

            MI = MI.relu()


        # 接下来构造超图卷积


        EU = self.fc2(MU)

        EI = self.fc2(MI)

        user_emb_t = EU[user_list,:]

        pos_emb_t = EI[pos_item_list,:]

        neg_emb_t = EI[neg_item_list,:]



        return user_emb_t,pos_emb_t,neg_emb_t








