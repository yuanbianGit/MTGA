from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

import pdb
import copy
__all__ = ['IDE']
import torch
import random
import math

# 这个是eccv的版本，替换了最后一个bn层
"""
class IDE(nn.Module):
    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=1024, norm=False, dropout=0, num_classes=0):
        super(IDE, self).__init__()

        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        self.base = torchvision.models.resnet50(pretrained=True)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, is_training, metaTrain=True, mix_thre=0.6,mix_pro= 0.5, output_both=False, mix_info=None, lamd=None):

        x = self.base.conv1(x)

        x = self.base.bn1(x)
        x = self.base.relu(x)
        # print(randE)

        x_layer0 = self.base.maxpool(x)
        x_layer1 = self.base.layer1(x_layer0)
        x_layer2 = self.base.layer2(x_layer1)
        x_layer3 = self.base.layer3(x_layer2)
        feat_map = self.base.layer4(x_layer3)
        x = feat_map

        # part_feats = nn.AdaptiveAvgPool2d((6,1))(feat_map).squeeze()
        # part_feats =part_feats.view(part_feats.size(0), -1)

        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            # print(self.feat_bn.track_running_stats)
            # True 那就是利用原来的训练的整体的数据的mean和var去进行
            # print(self.feat_bn.weight.size())
            # # tensor([0.9242, 0.9078, 0.9169,  ..., 0.9259, 0.9219, 0.9118], device='cuda:0')
            # print(self.feat_bn.bias.size())
            # # tensor([0.0773, 0.0747, 0.0630,  ..., 0.0889, 0.0797, 0.0599], device='cuda:0')
            # print(self.feat_bn.running_mean.size())
            # # torch.Size([1024])
            # # torch.Size([1024])
            # # torch.Size([1024])
            # # torch.Size([1024])
            # # tensor([-0.8744, -0.6257, -0.5969,  ..., -1.1609, -0.8203, -0.8314],
            # #        device='cuda:0')
            # print(self.feat_bn.running_var.size())
            # tensor([0.1909, 0.1874, 0.2295,  ..., 0.2413, 0.2021, 0.2206], device='cuda:0')
            '''
            out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, bn_w, self.bias,
            self.training, self.momentum, self.eps)
            
            '''
            if metaTrain:
                mean_train = torch.mean(x, dim=0)
                var_train = torch.var(x, dim=0, unbiased=False)
                x = self.feat_bn(x)
                out_bn = [mean_train, var_train]
            else:
                # bn3
                if mix_pro>mix_thre:
                # if torch.rand(1)>0.6
                    # bn1
                    # mean_train = torch.mean(x, dim=0)
                    # var_train = torch.var(x, dim=0, unbiased=False)
                    # bn2
                    mean_train = self.feat_bn.running_mean
                    var_train = self.feat_bn.running_var

                    mean_mix = lamd*mean_train+(1-lamd)*mix_info[0]
                    var_mix = lamd*var_train+(1-lamd)*mix_info[1]
                    mean_mix = mean_mix.detach()
                    var_mix = var_mix.detach()

                    x_mix = F.batch_norm(
                        x, mean_mix, var_mix, self.feat_bn.weight, self.feat_bn.bias,False)
                    if output_both==1:
                        x_ori = F.batch_norm(
                            x, self.feat_bn.running_mean, self.feat_bn.running_var, self.feat_bn.weight, self.feat_bn.bias, False)
                        x = torch.cat([x_mix, x_ori], dim=0)
                    else:
                        x = x_mix
                    out_bn = None
                else:
                    x = self.feat_bn(x)
                    out_bn = None

        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            logits = self.classifier(x)

        if is_training:
            return [logits, x, out_bn]
        else:
            return [x]


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
"""

# 这是原始版本

class IDE(nn.Module):
    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=1024, norm=False, dropout=0, num_classes=0):
        super(IDE, self).__init__()

        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        self.base = torchvision.models.resnet50(pretrained=True)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, is_training=False, inter_f=False):

        x = self.base.conv1(x)

        x = self.base.bn1(x)
        x = self.base.relu(x)
        # print(randE)

        x_layer0 = self.base.maxpool(x)
        x_layer1 = self.base.layer1(x_layer0)
        x_layer2 = self.base.layer2(x_layer1)
        x_layer3 = self.base.layer3(x_layer2)
        feat_map = self.base.layer4(x_layer3)
        x = feat_map

        # part_feats = nn.AdaptiveAvgPool2d((6,1))(feat_map).squeeze()
        # part_feats =part_feats.view(part_feats.size(0), -1)

        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)

        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            logits = self.classifier(x)

        if is_training:
            # return [logits, x, None]
            return [logits, x]

        else:
            if inter_f:
                # return x_layer1 BIA2
                # return x_layer2 BIA1
                return feat_map
            else:
                return [x]


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))