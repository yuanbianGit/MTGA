from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.nn import functional as F
from scipy.stats import beta

__all__ = ['PCB', 'PCB_test']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # logger.info(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

'''训练过程 eccv最后版本'''
"""
class PCB(nn.Module):
  '''
  Based on
  https://github.com/layumi/Person_reID_baseline_pytorch
  '''
  def __init__(self, num_classes):
    super(PCB, self).__init__()

    self.part = 6 # We cut the pool5 to 6 parts
    model_ft = models.resnet50(pretrained=True)
    self.model = model_ft
    self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
    self.dropout = nn.Dropout(p=0.5)
    # remove the final downsample
    self.model.layer4[0].downsample[0].stride = (1,1)
    self.model.layer4[0].conv2.stride = (1,1)
    # define 6 classifiers
    for i in range(self.part):
      name = 'classifier'+str(i)
      setattr(self, name, ClassBlock(2048, num_classes, True, False, 256))

  def forward(self, x, is_training, metaTrain=True,mix_thre=0.6, output_both=False,mix_pro= 0.5, mix_info=None, lamd=None):


    x = self.model.conv1(x)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.maxpool(x)
    
    x = self.model.layer1(x)
    x = self.model.layer2(x)
    x = self.model.layer3(x)
    x = self.model.layer4(x)
    x = self.avgpool(x)
    x = self.dropout(x)
    part = {}
    feature = []
    feature_copy = []
    predict = []
    # get six part feature batchsize*2048*6
    if metaTrain:
        mean_train = 0
        var_train = 0
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            feature.append(part[i])
            predict.append(c(part[i]))

            mean_train += torch.mean(part[i], dim=0)
            var_train += torch.var(part[i], dim=0, unbiased=False)
        out_bn = [mean_train/self.part, var_train/self.part]
        feature_out = torch.cat(feature, dim=1)

    else:

        if mix_pro > mix_thre:
            for i in range(self.part):
              part[i] = torch.squeeze(x[:,:,i])
              name = 'classifier'+str(i)
              c = getattr(self,name)
              mean_train = torch.mean(part[i], dim=0)
              var_train = torch.var(part[i], dim=0, unbiased=False)

              mean_mix = lamd * mean_train + (1 - lamd) * mix_info[0]
              var_mix = lamd * var_train + (1 - lamd) * mix_info[1]
              # mean_mix = mix_info[0]
              # var_mix  =  mix_info[1]
              mean_mix = mean_mix.detach()
              var_mix = var_mix.detach()
              part[i] = ((part[i]-mean_train)/(var_train+1e-5))*var_mix+mean_mix

              feature.append(part[i])
              predict.append(c(part[i]))
            feature_mix = torch.cat(feature, dim=1)
            out_bn = None
            if output_both==1:
                for i in range(self.part):
                    part[i] = torch.squeeze(x[:, :, i])
                    # name = 'classifier' + str(i)
                    # c = getattr(self, name)
                    feature_copy.append(part[i])
                    # predict.append(c(part[i]))
                feature_out_ori = torch.cat(feature_copy, dim=1)
                feature_out = torch.cat([feature_mix,feature_out_ori])
            else:
                feature_out = feature_mix
        else:
            for i in range(self.part):
              part[i] = torch.squeeze(x[:,:,i])
              name = 'classifier'+str(i)
              c = getattr(self,name)
              feature.append(part[i])
              predict.append(c(part[i]))
            feature_out = torch.cat(feature, dim=1)
            out_bn = None
    return [predict, feature_out, out_bn]
"""

class PCB(nn.Module):
  '''
  Based on
  https://github.com/layumi/Person_reID_baseline_pytorch
  '''
  def __init__(self, num_classes):
    super(PCB, self).__init__()

    self.part = 6 # We cut the pool5 to 6 parts
    model_ft = models.resnet50(pretrained=True)
    self.model = model_ft
    self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
    self.dropout = nn.Dropout(p=0.5)
    # remove the final downsample
    self.model.layer4[0].downsample[0].stride = (1,1)
    self.model.layer4[0].conv2.stride = (1,1)
    # define 6 classifiers
    for i in range(self.part):
      name = 'classifier'+str(i)
      setattr(self, name, ClassBlock(2048, num_classes, True, False, 256))

  def forward(self, x, is_training, metaTrain=True,mix_thre=0.6, output_both=False,mix_pro= 0.5, mix_info=None, lamd=None):
    x = self.model.conv1(x)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.maxpool(x)

    x = self.model.layer1(x)
    x = self.model.layer2(x)
    x = self.model.layer3(x)
    x = self.model.layer4(x)
    x = self.avgpool(x)
    x = self.dropout(x)
    part = {}
    feature = []
    feature_copy = []
    predict = []
    # get six part feature batchsize*2048*6
    # for i in range(self.part):
    #     part[i] = torch.squeeze(x[:, :, i])
    #     name = 'classifier' + str(i)
    #     c = getattr(self, name)
    #     feature.append(part[i])
    #     predict.append(c(part[i]))
    # feature_out = torch.cat(feature, dim=1)
    ''' 修改版本'''
    for i in range(self.part):
      part[i] = torch.squeeze(x[:, :, i])
      name = 'classifier' + str(i)
      c = getattr(self, name)
      feature.append(part[i])
      predict.append(c(part[i]))
    feature_out = torch.cat(feature, dim=1)
    # print(predict[1].size())
    stacked = torch.stack(predict)
    predict = torch.mean(stacked, dim=0)

    return [predict, feature_out, None]
    # return [feature_out]

class PCB_test(nn.Module):
  def __init__(self, model):
    super(PCB_test, self).__init__()
    self.part = 6
    self.model = model.model
    self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
    # remove the final downsample
    self.model.layer4[0].downsample[0].stride = (1,1)
    self.model.layer4[0].conv2.stride = (1,1)

  def forward(self, x, is_training):
    x = self.model.conv1(x)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.maxpool(x)

    x = self.model.layer1(x)
    x = self.model.layer2(x)
    x = self.model.layer3(x)
    x = self.model.layer4(x)
    x = self.avgpool(x)
    y = x.view(x.size(0),x.size(1),x.size(2))
    return [y]