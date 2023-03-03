#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 10:49
#@Author : 韩笑奇
#@File : config.py
#@Software: PyCharm


class Config(object):

    def __init__(self):


        self.k = 2

        self.user_n = None

        self.item_n = None

        self.user_item_dict = None

        self.emb_size = 64

        self.layer_size = 3

        self.emb_size_end = 32

        self.train_item_n = None

        self.device = 'cuda'

        self.epochs = 300

        self.reg_weight = 1e-4

        self.test_flag = 'part'



