#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 10:34
#@Author : 韩笑奇
#@File : utils.py
#@Software: PyCharm

import scipy.sparse as sp

from scipy.sparse import vstack

import numpy as np

def bulid_HBUK(configs):

    k = configs.k

    adj_matrix = sp.load_npz('./data_adj/adj_matrix.npz')

    # 对于用户

    A_K_list = []


    A_MID = adj_matrix.T.dot(adj_matrix)


    for i in range(1,k+1):

        A = np.power(A_MID, i)

        A[A>1] = 1

        A_K_list.append(A)

    #H_BUK_list = []

    HU = adj_matrix.dot(A_K_list[0])

    for i in range(1,len(A_K_list)):

        # 为了方便 计算  预测矩阵   再此使用  求和 加求均值

        # for A_k in A_K_list:

            # H_BUK_list.append(adj_matrix.dot(A_K))

        HU = HU + adj_matrix.dot(A_K_list[i])


    # 得出超图关联矩阵 HU

    #HU = vstack([H_BUK_list[0].tocoo(),H_BUK_list[1].tocoo()])

    #for i in range(2,len(H_BUK_list)):

    #    HU = vstack([HU,H_BUK_list[i].tocoo()])


    # 对于物品

    A_UK_list =[]

    A_UK_MID = adj_matrix.dot(adj_matrix.T)

    for i in range(1,k+1):

        A_UK = np.power(A_UK_MID,i)

        A_UK[A_UK>1] = 1

        A_UK_list.append(A_UK)

    # H_BKI_list = []

    HI = adj_matrix.T.dot(A_UK_list[0])

    for i in range(1,len(A_UK_list)):

    # for A_UKs in A_UK_list:

        # H_BKI_list.append(adj_matrix.T.dot(A_UKs))

        HI = HI + adj_matrix.T.dot(A_UK_list[i])


    #HI = vstack([H_BKI_list[0].tocoo(),H_BKI_list[1].tocoo()])

    #for i in range(2,len(H_BKI_list)):

        #HI = vstack([HI,H_BKI_list[i].tocoo()])




    # 保存数据

    sp.save_npz('./data_adj/HU_adj.npz',HU.tocsr())

    sp.save_npz('./data_adj/HI_adj.npz',HI.tocsr())







    print('aaaa')