#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 9:42
#@Author : 韩笑奇
#@File : 超边test.py
#@Software: PyCharm

import numpy as np


ls1 = [

    [0,1,0],
    [0,0,1],
    [1,1,0]

]



np1 = np.array(ls1)

np2 = np.array(ls1)

re = np.dot(np1,np2.T)

print(np1)

print(np2.T)

print(re)
