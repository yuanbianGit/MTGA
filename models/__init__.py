from __future__ import absolute_import
import torch
import torch.nn as nn

from .DenseNet import *
from .MuDeep import *
from .AlignedReID import *
from .PCB import *
from .HACNN import *
from .IDE import *
from .LSRO import *
from .BOT import *
from .TransReid import TRANSREID,vit
from .mgn import MGN
from .pat import PAT
from .clip.clip_reid import build_clip
from .robust_m import robust_m
from .pfd import PFD
from .HOReid2 import HOReid
__factory = {
  # 1. 
  'hacnn': HACNN,
  'densenet121': DenseNet121,
  'ide': IDE,
  'bot': BOT,
  # 2.
  'aligned': ResNet50,
  'pcb': PCB,
  'mudeep': MuDeep,
  # 3.
  'cam': IDE,
  'hhl': IDE, 
  'lsro': DenseNet121,
  'spgan': IDE,
  'transreid':TRANSREID,
  # 'vit_0.5': vit,
  'vit': vit,
  'mgn':MGN,
  'pat': PAT,
  'clip': build_clip,
  'robust_m':robust_m,
  'pfd': PFD,
  'horeid': HOReid,
}




def get_names():
  return __factory.keys()

def init_model(name, pre_dir, *args, **kwargs):
  if name not in __factory.keys(): 
    raise KeyError("Unknown model: {}".format(name))

  print("Initializing model: {}".format(name))
  net = __factory[name](*args, **kwargs)
  if name=='bot'or name=='transreid' or name=='ide' or  name=='vit' or  name=='vit_norm' or name=='clip':
    net.load_param(pre_dir)
  elif name=='robust_m':
    state = torch.load(pre_dir)
    net.load_state_dict(state['net'])
  elif name=='horeid':
    net.resume_model_from_path(pre_dir, 119)

  else:
    # load pretrained model
    checkpoint = torch.load(pre_dir,encoding='latin1') # for Python 2
    # checkpoint = torch.load(pre_dir, encoding="latin1") # for Python 3
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    change = False
    for k, v in state_dict.items():
      if k[:6] == 'module':
        change = True
        break
    if not change:
      new_state_dict = state_dict
    else:
      from collections import OrderedDict
      new_state_dict = OrderedDict()
      for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    # print(new_state_dict)
    # print(net)
    net.load_state_dict(new_state_dict)
  print("----------------successfully load" +pre_dir )
  # freeze 相当于测试模式，BN和DropOut不同！

  net.eval()
  # 相当于不要梯度传递
  net.volatile = True
  return net