#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 10:42
#@Author : 韩笑奇
#@File : test_demo.py
#@Software: PyCharm

import scipy.sparse as sp

import numpy as np

from scipy.sparse import vstack


ls =[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]


df = sp.dok_matrix((len(ls),len(ls)),dtype=np.float32)

for i in range(len(ls)):

    for j in range(len(ls)):

        df[i,j] = ls[i][j]

print(df.toarray())

print(df.T.toarray())

df1 = df.tocoo()


df2 = vstack([df1,df1])


print(df2.toarray())



'''df2 = np.power(df,2)

print(df2.toarray())


ls1 = np.array(ls)

ls2 = np.dot(ls1,ls1)

print(ls2)


#re = min(1,df)

#print(re.toarray())


df[df>1]=1

print(df.toarray())


print(df.T.toarray())

#print(np.concatenate((df,df),axis=0))'''




'''df3 = df.tocsr()

df4 = df.todok()


print(df3.dot(df4).toarray())
print(df.dot(df).toarray())


df5 = np.sum(df3.toarray(),1)

df6 = [i for i in df5]

print(df6)

print(df5)

print(sp.diags(df6))'''




df7 = df.tocsr()

df8 = df.tocoo()

df9 = df7.dot(df8)

re = df9.toarray()

re = np.sum(re,1)

print(sp.diags(re))


import pandas as pd

df10 = df.todok()

dfs = pd.DataFrame(df10.toarray())

print(dfs)


x = np.array(df.sum(1))

y = np.power(x,-0.5).flatten()

print(y)


df10 = df.tocoo()

df11 = df.tocsr()

df12 = df10 + df11

print(df12.toarray())




