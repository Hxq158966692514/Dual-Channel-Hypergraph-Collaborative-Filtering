#-*- codeing = utf-8 -*-
#@Time : 2023-03-03 14:17
#@Author : 韩笑奇
#@File : tensor_add_test.py
#@Software: PyCharm



import torch






tf1 = torch.Tensor([[1,2,3,4],[1,2,3,4]]).to(dtype=torch.float32)

tf2 = torch.Tensor([[1,2,3,4],[1,2,3,4]]).to(dtype=torch.float32)

tf3 = tf1 + tf2

print(tf3)
print(tf3/3)