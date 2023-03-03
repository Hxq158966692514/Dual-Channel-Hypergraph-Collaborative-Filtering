#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 13:31
#@Author : 韩笑奇
#@File : test_demo2.py
#@Software: PyCharm

import pandas as pd


ls1 = list(str(range(10)))

print(ls1)

ls = [str(i) for i in range(10)]

print(ls)


df = pd.DataFrame([[1,2,3,4],[1,2,3,4]],index=range(2),columns=range(4))

print(list(df.loc[0]))