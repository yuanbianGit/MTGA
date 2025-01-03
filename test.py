from __future__ import absolute_import
from __future__ import print_function, division
import sys
import time
import datetime
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import os.path as osp
import math
from random import sample 
from scipy import io
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import datetime
import models
from models.PCB import PCB_test
# from ReID_attr import get_target_withattr # Need Attribute file
from opts import get_opts
from GD import Generator_2out, MS_Discriminator, Pat_Discriminator, GANLoss, weights_init, Generator
from advloss import DeepSupervision, adv_CrossEntropyLoss, adv_CrossEntropyLabelSmooth, adv_TripletLoss
from util import data_manager
from util.dataset_loader import ImageDataset
from util.utils import fliplr, Logger, save_checkpoint, visualize_ranked_results
from util.eval_metrics import make_results
from util.samplers import RandomIdentitySampler, AttrPool
from util.logger import setup_logger
from collections import OrderedDict

# TODO 要加入cross-test的情况! 和prid!
# Training settings
parser = argparse.ArgumentParser(description='adversarial attack')
parser.add_argument('--root', type=str, default='/data4/by/reid/dataSet', help="root path to data directory")
parser.add_argument('--targetmodel', type=str, default='bot', choices=models.get_names())
parser.add_argument('--targetmodel_dataset', type=str, default='market1501', choices=data_manager.get_names())

parser.add_argument('--test_dataset', type=str, default='market1501', choices=data_manager.get_names())
# parser.add_argument('--dataset', type=str, default='dukemtmcreid', choices=data_manager.get_names())
# parser.add_argument('--dataset', type=str, default='market1501', choices=data_manager.get_names())

# PATH
# parser.add_argument('--G_resume_dir', type=str, default='/data1/by/reid/github/MissRank_old/log_eccv2/wTE_wPE_wGS_wTS3_0.5_10_0.8_new_2/best_G.pth.tar', metavar='path to resume G')
# parser.add_argument('--D_resume_dir', type=str, default='/data1/by/reid/github/MissRank_old/log_eccv2/wTE_wPE_wGS_wTS3_0.5_10_0.8_new_2/best_D.pth.tar', metavar='path to resume D')
parser.add_argument('--G_resume_dir', type=str, default='/data4/by/reid/github/CLIP-ReID-master/ATT_LOG_CVPR_wIDE/base_1_0.3/best_G_V.pth.tar', metavar='path to resume G')
parser.add_argument('--D_resume_dir', type=str, default='/data4/by/reid/github/CLIP-ReID-master/ATT_LOG_CVPR/att_vit0.5_woD_0_0_50(5)-1/best_D_V.pth.tar', metavar='path to resume D')


parser.add_argument('--pre_dir', type=str, default='models', help='path to be attacked model')
parser.add_argument('--attr_dir', type=str, default='', help='path to attribute file')
parser.add_argument('--save_dir', type=str, default='logs', help='path to save model')
parser.add_argument('--ablation', type=str, default='', help='for ablation study')
# varen
parser.add_argument('--mode', type=str, default='test', help='train/test')
parser.add_argument('--D', type=str, default='MSGAN', help='Type of discriminator: PatchGAN or Multi-stage GAN')
parser.add_argument('--normalization', type=str, default='bn', help='bn or in')
parser.add_argument('--loss', type=str, default='xent_htri', choices=['cent', 'xent', 'htri', 'xent_htri'])
parser.add_argument('--ak_type', type=int, default=-1, help='-1 if non-targeted, 1 if attribute attack')
parser.add_argument('--attr_key', type=str, default='upwhite', help='[attribute, value]')
parser.add_argument('--attr_value', type=int, default=2, help='[attribute, value]')
parser.add_argument('--mag_in', type=float, default=16, help='l_inf magnitude of perturbation')
parser.add_argument('--temperature', type=float, default=-1, help="tau in paper")
parser.add_argument('--usegumbel', action='store_true', default=False, help='whether to use gumbel softmax')
parser.add_argument('--use_SSIM', type=int, default=2, help="0: None, 1: SSIM, 2: MS-SSIM ")
# Base
parser.add_argument('--train_batch', default=64, type=int,help="train batch size")
parser.add_argument('--test_batch', default=256, type=int, help="test batch size")
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')

parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num_ker', type=int, default=32, help='generator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--print_freq', type=int, default=20, help="print frequency")
parser.add_argument('--eval_freq', type=int, default=10, help="eval frequency")
# parser.add_argument('--usevis', action='store_true', default=False, help='whether to save vis')
parser.add_argument('--usevis', type=bool, default=True, help='whether to save vis')
parser.add_argument('--lp', default= 8.0 , type=float,metavar='lp', help='lp norm')
parser.add_argument('--test_logDir', type=str, default='./log_test/test', help='[attribute, value]')
parser.add_argument('--vis_dir', type=str, default='./log_test/test', help='[attribute, value]')
parser.add_argument('--G_CHANNEL', default=3, type=int,help="train batch size")


args = parser.parse_args()

Imagenet_mean = [0.485, 0.456, 0.406]
Imagenet_stddev = [0.229, 0.224, 0.225]

is_training = args.mode == 'train'
attr_list = [args.attr_key, args.attr_value]
attr_matrix = None
if args.attr_dir: 
  assert args.dataset in ['dukemtmcreid', 'market1501']
  attr_matrix = io.loadmat(args.attr_dir)
  args.ablation = osp.join('attr', args.attr_key + '=' + str(args.attr_value))
if args.targetmodel=='bot' or args.targetmodel=='transreid' or  args.targetmodel=='vit' or args.targetmodel=='ide' or args.targetmodel=='pat' or(args.targetmodel=='pcb' and args.targetmodel_dataset=='dukemtmcreid')  or args.targetmodel == 'clip':
  pre_dir = osp.join(args.pre_dir, args.targetmodel, args.targetmodel_dataset + '.pth')
elif args.targetmodel=='mgn':
  pre_dir = osp.join(args.pre_dir, args.targetmodel, args.targetmodel_dataset + '.pt')
else:
  pre_dir = osp.join(args.pre_dir, args.targetmodel, args.targetmodel_dataset+'.pth.tar')
save_dir = args.test_logDir
vis_dir = args.vis_dir


def main(opt):
  if not osp.exists(save_dir): os.makedirs(save_dir)
  if not osp.exists(vis_dir): os.makedirs(vis_dir)

  use_gpu = torch.cuda.is_available()
  pin_memory = True if use_gpu else False

  if args.mode == 'train':
    logger = setup_logger("advreid",save_dir,if_train=True)
  else:
    logger = setup_logger("advreid",save_dir,if_train=False)
  logger.info(args)

  if use_gpu:
    logger.info("GPU mode")
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)

  else:
    logger.info("CPU mode")

  ### Setup dataset loader ###

  model_dataset = data_manager.init_img_dataset(root=args.root, name=args.targetmodel_dataset, split_id=opt['split_id'],
                                          cuhk03_labeled=opt['cuhk03_labeled'], cuhk03_classic_split=opt['cuhk03_classic_split'])
  logger.info("Initializing dataset {}".format(args.test_dataset))
  dataset_test = data_manager.init_img_dataset(root=args.root, name=args.test_dataset, split_id=opt['split_id'],
                                                cuhk03_labeled=opt['cuhk03_labeled'],
                                                cuhk03_classic_split=opt['cuhk03_classic_split'])

  # if args.ak_type < 0:  #no-target attack
  #   trainloader = DataLoader(ImageDataset(dataset.train, transform=opt['transform_train']),
  #                            sampler=RandomIdentitySampler(dataset.train, num_instances=opt['num_instances']),
  #                            batch_size=args.train_batch, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=True)
  # elif args.ak_type > 0:  #target attack
  #   trainloader = DataLoader(ImageDataset(dataset.train, transform=opt['transform_train']),
  #                            sampler=AttrPool(dataset.train, args.dataset, attr_matrix, attr_list, sample_num=16),
  #                            batch_size=args.train_batch, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=True)
  queryloader = DataLoader(ImageDataset(dataset_test.query, transform=opt['transform_test']), batch_size=args.test_batch,
                           shuffle=False, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=False)
  galleryloader = DataLoader(ImageDataset(dataset_test.gallery, transform=opt['transform_test']), batch_size=args.test_batch,
                             shuffle=False, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=False)
  
  ### Prepare criterion ###
  # if args.ak_type<0:
  #   clf_criterion = adv_CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu) if args.loss in ['xent', 'xent_htri'] else adv_CrossEntropyLoss(use_gpu=use_gpu)
  # else:
  #   clf_criterion = nn.MultiLabelSoftMarginLoss()
  # metric_criterion = adv_TripletLoss(margin=args.margin, ak_type=args.ak_type)
  # criterionGAN = GANLoss()

  ### Prepare pretrained model
  logger.info("Initializing model traine by {}".format(args.targetmodel_dataset))
  target_net = models.init_model(name=args.targetmodel, pre_dir=pre_dir, num_classes=model_dataset.num_train_pids)
  ### 固定模型参数
  target_net.eval()
  # check_freezen(target_net, need_modified=True, after_modified=False)

  ### Prepare main net ###
  G = Generator(3, args.G_CHANNEL, args.num_ker, norm=args.normalization).apply(weights_init)

  if args.D == 'PatchGAN':
    D = Pat_Discriminator(input_nc=6, norm=args.normalization).apply(weights_init)
  elif args.D == 'MSGAN':
    D = MS_Discriminator(input_nc=6, norm=args.normalization, temperature=args.temperature, use_gumbel=args.usegumbel).apply(weights_init)


  logger.info("Model size: {:.5f}M".format((sum(g.numel() for g in G.parameters())+sum(d.numel() for d in D.parameters()))/1000000.0))

  if use_gpu: 
    # test_target_net = nn.DataParallel(target_net).cuda() if not args.targetmodel == 'pcb' else nn.DataParallel(PCB_test(target_net)).cuda()
    # target_net = nn.DataParallel(target_net).cuda()
    # G = nn.DataParallel(G).cuda()
    # D = nn.DataParallel(D).cuda()
    test_target_net = target_net.cuda() if not args.targetmodel == 'pcb' else PCB_test(target_net).cuda()
    G = G.cuda()
    D = D.cuda()

  if args.mode == 'test':
    epoch = 'test'
    inference(G, D, test_target_net, dataset_test, queryloader, galleryloader, epoch, use_gpu, logger,is_test=True)
    return 0

def inference(G, D, target_net, dataset, queryloader, galleryloader, epoch, use_gpu,logger, is_test=False, ranks=[1, 5, 10, 20]):
  global is_training
  is_training = False
  if args.mode == 'test' and args.G_resume_dir:
    G_resume_dir, D_resume_dir = args.G_resume_dir, args.D_resume_dir
    print(G_resume_dir)
    print(D_resume_dir)

    G_checkpoint, D_checkpoint = torch.load(G_resume_dir), torch.load(D_resume_dir)
    G_state_dict = G_checkpoint['state_dict'] if isinstance(G_checkpoint, dict) and 'state_dict' in G_checkpoint else G_checkpoint
    D_state_dict = D_checkpoint['state_dict'] if isinstance(D_checkpoint, dict) and 'state_dict' in D_checkpoint else D_checkpoint

    G_new_state_dict = OrderedDict()
    for k, v in G_state_dict.items():
      if k[:7]=="module.":
        new_k = k[7:]
        G_new_state_dict[new_k] = v
      else:
        G_new_state_dict[k] = v

    D_new_state_dict = OrderedDict()
    for k, v in D_state_dict.items():
      if k[:7] == "module.":
        new_k = k[7:]
        D_new_state_dict[new_k] = v
      else:
        D_new_state_dict[k] = v
    G.load_state_dict(G_new_state_dict)
    D.load_state_dict(D_new_state_dict)
    logger.info("Sucessfully, loading {} and {}".format(G_resume_dir, D_resume_dir))
    G.eval()
    D.eval()
  with torch.no_grad():
    qf, lqf, new_qf, new_lqf, q_pids, q_camids = extract_and_perturb\
      (queryloader, G, D, target_net, use_gpu, query_or_gallery='query', is_test=is_test, epoch=epoch,logger=logger)
    gf, lgf, g_pids, g_camids = extract_and_perturb\
      (galleryloader, G, D, target_net, use_gpu, query_or_gallery='gallery', is_test=is_test, epoch=epoch,logger=logger)

    if args.ak_type > 0:
      distmat, hits, ignore_list = make_results(new_qf, gf, new_lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type, attr_matrix, args.dataset, attr_list)
      logger.info("Hits rate, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(ranks[0], hits[ranks[0]-1], ranks[1], hits[ranks[1]-1], ranks[2], hits[ranks[2]-1], ranks[3], hits[ranks[3]-1]))
      if not is_test:
        return hits

    else:
      if is_test:
        distmat, cmc, mAP = make_results(qf, gf, lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type)
        new_distmat, new_cmc, new_mAP = make_results(new_qf, gf, new_lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type)
        logger.info("Results ----------")
        logger.info("Before, mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(mAP, ranks[0], cmc[ranks[0]-1], ranks[1], cmc[ranks[1]-1], ranks[2], cmc[ranks[2]-1], ranks[3], cmc[ranks[3]-1]))
        logger.info("After , mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(new_mAP, ranks[0], new_cmc[ranks[0]-1], ranks[1], new_cmc[ranks[1]-1], ranks[2], new_cmc[ranks[2]-1], ranks[3], new_cmc[ranks[3]-1]))
        # if args.usevis:
        #   visualize_ranked_results(distmat, dataset, save_dir=osp.join(vis_dir, 'origin_results'), topk=20)
        # if args.usevis:
        #   visualize_ranked_results(new_distmat, dataset, save_dir=osp.join(vis_dir, 'polluted_results'), topk=20)
      else:
        _, new_cmc, new_mAP = make_results(new_qf, gf, new_lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type)
        logger.info("mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(new_mAP, ranks[0], new_cmc[ranks[0]-1], ranks[1], new_cmc[ranks[1]-1], ranks[2], new_cmc[ranks[2]-1], ranks[3], new_cmc[ranks[3]-1]))
        return new_cmc, new_mAP

def extract_and_perturb(loader, G, D, target_net, use_gpu, query_or_gallery, is_test, epoch,logger):
  f, lf, new_f, new_lf, l_pids, l_camids = [], [], [], [], [], []
  ave_mask, num = 0, 0
  for batch_idx, (imgs, pids, camids, pids_raw) in enumerate(loader):
    if use_gpu: 
      imgs = imgs.cuda()
    ls = extract(imgs, target_net)
    if len(ls) == 1: features = ls[0]
    if len(ls) == 2: 
      features, local_features = ls
      lf.append(local_features.detach().data.cpu())

    f.append(features.detach().data.cpu())
    l_pids.extend(pids)
    l_camids.extend(camids)
    # 如果是query那么加上生成的噪声去生成新的adv_query
    if query_or_gallery == 'query':
      new_imgs, delta, mask = perturb_test(imgs, G, D, train_or_test='test')
      ave_mask += torch.sum(mask.detach()).cpu().numpy()
      num += imgs.size(0)

      ls = extract(new_imgs, target_net)
      if len(ls) == 1: new_features = ls[0]
      if len(ls) == 2: 
        new_features, new_local_features = ls
        new_lf.append(new_local_features.detach().data.cpu())
      new_f.append(new_features.detach().data.cpu())

      ls = [imgs, new_imgs, delta, mask]
      if is_test and args.usevis and batch_idx==0 and args.targetmodel=='ide':
        save_img(ls, pids, camids, epoch, batch_idx)
      # print('已经存储了！')

  f = torch.cat(f, 0)
  if not lf == []: lf = torch.cat(lf, 0)
  l_pids, l_camids = np.asarray(l_pids), np.asarray(l_camids)
  
  logger.info("Extracted features for {} set, obtained {}-by-{} matrix".format(query_or_gallery, f.size(0), f.size(1)))
  if query_or_gallery == 'gallery':
    return [f, lf, l_pids, l_camids]
  elif query_or_gallery == 'query':
    new_f = torch.cat(new_f, 0)
    if not new_lf == []: 
      new_lf = torch.cat(new_lf, 0)
    return [f, lf, new_f, new_lf, l_pids, l_camids]

def extract(imgs, target_net):
  if args.targetmodel in ['pcb', 'lsro','mgn']:
    ls = [target_net(imgs, is_training)[0] + target_net(fliplr(imgs), is_training)[0]]
    # ls = [target_net(imgs, is_training)[0]]
  else: 
    ls = target_net(imgs, is_training)
  for i in range(len(ls)): ls[i] = ls[i].data.cpu()
  return ls

def perturb_test(imgs, G, D, train_or_test='test'):
  n,c,h,w = imgs.size()
  delta = G(imgs)
  delta = L_norm(delta, train_or_test)
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())
  # 加入噪声和没加入噪声的!
  _, mask = D(torch.cat((imgs, new_imgs.detach()), 1))
  # 经过Mask的才是最后的噪声,然后加入生成adv_img
  delta = delta * mask
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())

  for c in range(3):
    new_imgs.data[:, c, :, :] = new_imgs.data[:, c, :, :].clamp((0.0 - Imagenet_mean[c]) / Imagenet_stddev[c],
                                                                (1.0 - Imagenet_mean[c]) / Imagenet_stddev[
                                                                  c])  # do clamping per channel
  if train_or_test == 'train':
    return new_imgs, mask
  elif train_or_test == 'test':
    return new_imgs, delta, mask

def L_norm(delta, mode='train'):
  '''
  大概意思就是clamp使得噪声符合约束！
  '''
  # 当时的transform ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
  delta = torch.clamp(delta,-args.lp/255,args.lp/255)
  if delta.size(1)==1:
    delta = delta.repeat(1,3,1,1)
  for c in range(delta.size(1)):
    delta.data[:,c,:,:] = (delta.data[:,c,:,:]) / Imagenet_stddev[c]
  return delta

def save_img(ls, pids, camids, epoch, batch_idx):
  image, new_image, delta, mask = ls
  # undo normalize image color channels
  delta_tmp = torch.zeros(delta.size())
  for c in range(delta_tmp.size(1)):
    image.data[:,c,:,:] = (image.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]
    new_image.data[:,c,:,:] = (new_image.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]
    delta_tmp.data[:,c,:,:] = (delta.data[:,c,:,:] * Imagenet_stddev[c])
    delta_tmp.data[:, c, :, :] = (delta_tmp.data[:,c,:,:]-delta_tmp.data[:,c,:,:].min())/(delta_tmp.data[:,c,:,:].max()-delta_tmp.data[:,c,:,:].min())

  torchvision.utils.save_image(image.data, osp.join(vis_dir, 'original_epoch{}_batch{}.png'.format(epoch, batch_idx)))
  torchvision.utils.save_image(new_image.data, osp.join(vis_dir, 'polluted_epoch{}_batch{}.png'.format(epoch, batch_idx)))
  torchvision.utils.save_image(delta_tmp.data, osp.join(vis_dir, 'delta_epoch{}_batch{}.png'.format(epoch, batch_idx)))
  torchvision.utils.save_image(mask.data*255, osp.join(vis_dir, 'mask_epoch{}_batch{}.png'.format(epoch, batch_idx)))

def check_freezen(net, need_modified=False, after_modified=None):
  # logger.info(net)
  cc = 0
  for child in net.children():
    for param in child.parameters():
      if need_modified: param.requires_grad = after_modified
      # if param.requires_grad: logger.info('child', cc , 'was active')
      # else: logger.info('child', cc , 'was forzen')
    cc += 1




if __name__ == '__main__':
  opt = get_opts(args.targetmodel)
  main(opt)
