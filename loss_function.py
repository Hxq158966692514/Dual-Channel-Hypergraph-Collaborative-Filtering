#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 16:32
#@Author : 韩笑奇
#@File : loss_function.py
#@Software: PyCharm

from recbole.model.loss import BPRLoss,EmbLoss

import torch

def loss_func(user_list, pos_item_list, neg_item_list,user_emb_t,pos_emb_t,neg_emb_t,model,configs):

    # 计算贝叶斯个性化排序损失

    pos_scores = torch.mul(user_emb_t,pos_emb_t).sum(dim=1)

    neg_scores = torch.mul(user_emb_t,neg_emb_t).sum(dim=1)

    bpr = BPRLoss()

    bpr_loss = bpr(pos_scores,neg_scores)

    u_ego_embeddings = model.emb_user.weight[user_list].to(device=configs.device)

    pos_ego_embeddings = model.emb_item.weight[pos_item_list].to(device=configs.device)

    neg_ego_embeddings = model.emb_item.weight[neg_item_list].to(device=configs.device)

    reg = EmbLoss()

    reg_loss = reg(u_ego_embeddings,pos_ego_embeddings,neg_ego_embeddings)

    total_loss = bpr_loss + reg_loss * configs.reg_weight

    return total_loss



