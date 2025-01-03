# import numpy as np
# import torch
# from torchvision.utils import save_image
#
# def min_max_div(input):
#   '''
#   input (B,H,W)
#   '''
#   out_put = torch.zeros_like(input)
#   for i in range(len(input)):
#     min_vals = torch.min(input[i])
#     max_vals = torch.max(input[i])
#
#     # 最小-最大缩放，将x的范围缩放到[0, 1]
#     scaled_x = (input[i] - min_vals) / (max_vals - min_vals)
#     out_put[i] = scaled_x
#   return out_put
#
#
# feat =torch.from_numpy(np.load('./log_cs/x0_wob.npy'))
# print("pass")
# max_feature = feat.max(dim=1)
# max_feature = max_feature.values
# # 去除最大值中大小为 1 的维度，得到 [32, 32, 16] 大小的 tensor
# max_feature = max_feature.squeeze()
# feat_nor = min_max_div(max_feature)
# save_image(feat_nor[0],'./log_cs/x0_wob.png')
# save_image(feat_nor[1],'./log_cs/x1_wob.png')
# save_image(feat_nor[2],'./log_cs/x2_wob.png')
#
# save_image(feat_nor[15],'./log_cs/x5_wob.png')
# print("pass")


# import torchvision
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# import os
# seed = 123
# random.seed(seed)
# np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#
# print(torch.rand(1))
# print(torch.rand((2,3)))


# 导入必要的模块
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# 设置参数α和β
a =9
b =1

# 生成x轴的数据，从0到1，共100个点
x = np.linspace(0, 1, 100)

# 计算Beta分布的概率密度函数值
y = beta.pdf(x, a, b)

# 绘制Beta分布的图形
label_str = r'$\alpha='+str(a)+r'\ \beta='+str(b)
plt.plot(x, y, label=label_str)
plt.title(u'Beta Distribution')
plt.xlabel(u'x')
plt.ylabel(u'PDF')
plt.legend()
plt.show()

val = beta.rvs(a, b)
