#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 14:46
#@Author : 韩笑奇
#@File : collection_demo.py
#@Software: PyCharm


from collections import defaultdict


dicts = defaultdict()


ls1 = ['a','b','c','d']

ls2 = [1,2,3,4]


for i in range(len(ls1)):

    dicts[ls1[i]] = ls2[i]

print(dict(dicts))
