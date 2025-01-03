import logging
import os
import random
# from threading import local
import torch
import torch.nn as nn

from .vit_pytorch4pat import part_attention_vit_base



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



'''
part attention vit
'''
class PAT(nn.Module):
    def __init__(self, num_classes, pretrain_tag='imagenet'):
        super().__init__()
        model_path_base =  "../pretrainedModel"

        path = 'jx_vit_base_p16_224-80ecf9dd.pth'
        self.model_path = os.path.join(model_path_base, path)
        self.pretrain_choice = 'imagenet'
        self.cos_layer = False
        self.neck ='bnneck'
        self.neck_feat = 'before'
        self.in_planes = 768

        print('using Transformer_type: part token vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = 0

        self.base = part_attention_vit_base\
            (img_size=[256, 128],
            stride_size=[16,16],
            drop_path_rate=0.1,
            drop_rate= 0.0,
            attn_drop_rate=0.0,
            pretrain_tag=pretrain_tag)
        # if self.pretrain_choice == 'imagenet':
        #     self.base.load_param(self.model_path)
        #     print('Loading pretrained ImageNet model......from {}'.format(self.model_path))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, is_training=False):
        layerwise_tokens = self.base(x) # B, N, C
        layerwise_cls_tokens = [t[:, 0] for t in layerwise_tokens] # cls token
        part_feat_list = layerwise_tokens[-1][:, 1: 4] # 3, 768

        layerwise_part_tokens = [[t[:, i] for i in range(1,4)] for t in layerwise_tokens] # 12 3 768
        feat = self.bottleneck(layerwise_cls_tokens[-1])

        if is_training:
            cls_score = self.classifier(feat)
            return [cls_score, layerwise_cls_tokens, layerwise_part_tokens]
        else:
            return [layerwise_cls_tokens[-1]]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading trained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))        
