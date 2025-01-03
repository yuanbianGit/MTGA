import torch
import math
import random
import torch.nn.functional as F

def randomErasing_patch(input,mask_pro=0.25):
  if random.random()>0.8:
    return input
  else:
    # 随机选取80%的tensor设置为1，得到一个mask
    mask = torch.bernoulli(torch.full((input.size(-2)//8, input.size(-1)//8), (1-mask_pro))).bool()
    mask = mask.type(torch.float).cuda()
    # 将mask扩展到input大小，每个4*4的patch的值为mask的对应一个点的值
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(input.size(-2), input.size(-1)), mode='nearest').squeeze(0).squeeze(0)
    output = input * mask
    return output


def randomErasing_horizontal(input,mask_pro=0.25):
  """
  横向mask
  """
  if random.random()>0.8:
    return input
  else:
    # 生成一个16*8的tensor，值全为1
    mask = torch.ones(input.size(-2)//8, input.size(-1)//8).cuda()
    # 随机选取30%的行tensor设置为0
    row_indices = torch.randperm(input.size(-2)//8)[:int(mask_pro * input.size(-2)//8)] # 随机生成30%的行索引
    mask[row_indices] = 0 # 将对应的行设置为0
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(input.size(-2), input.size(-1)), mode='nearest').squeeze(0).squeeze(0)
    output = input * mask
    return output

def randomErasing_Vertical(input,mask_pro=0.25):
  """
  纵向mask
  """
  if random.random() > 0.8:
    return input
  else:
    # 生成一个16*8的tensor，值全为1
    mask = torch.ones(input.size(-2)//8, input.size(-1)//8).cuda()
    # 随机选取30%的行tensor设置为0
    col_indices = torch.randperm(input.size(-1)//8)[:int(mask_pro * input.size(-1)//8)]  # 随机生成30%的行索引
    mask[:,col_indices] = 0  # 将对应的行设置为0
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(input.size(-2), input.size(-1)), mode='nearest').squeeze(
      0).squeeze(0)
    output = input * mask
    return output

#
# input = torch.ones((2,3,32,16)).cuda()*5
# output = randomErasing_patch(input)
# output = output.detach().cpu().numpy()
# print('yeah')
