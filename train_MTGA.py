from __future__ import absolute_import
from __future__ import print_function, division
import sys
import time
import datetime
import argparse
import os
import numpy as np
import os.path as osp
import math
from random import sample 
from scipy import io
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import datetime
import models
from models.PCB import PCB_test
from scipy.stats import beta
from itertools import cycle

# from ReID_attr import get_target_withattr # Need Attribute file
from opts import get_opts, Imagenet_mean, Imagenet_stddev
from GD import Generator, MS_Discriminator, Pat_Discriminator, GANLoss, weights_init,ResnetG
from advloss import DeepSupervision, adv_CrossEntropyLoss, adv_CrossEntropyLabelSmooth, adv_TripletLoss,pami_att_TripletLoss
from util import data_manager
from util.dataset_loader import ImageDataset,ImageDataset2
from util.utils import fliplr, Logger, save_checkpoint, visualize_ranked_results
from util.eval_metrics import make_results
from util.samplers import RandomIdentitySampler, AttrPool
from util.logger import setup_logger
from collections import OrderedDict
import random
import learn2learn as l2l
from randomE import randomErasing_patch,randomErasing_horizontal,randomErasing_Vertical

def check_freezen(net, need_modified=False, after_modified=None):
  # logger.info(net)
  cc = 0
  for child in net.children():
    for param in child.parameters():
      if need_modified: param.requires_grad = after_modified
      # if param.requires_grad: logger.info('child', cc , 'was active')
      # else: logger.info('child', cc , 'was forzen')
    cc += 1


# Training settings
parser = argparse.ArgumentParser(description='adversarial attack')
parser.add_argument('--root', type=str, default='../../dataSet', help="root path to data directory")
parser.add_argument('--targetmodel', type=str, default='ide', choices=models.get_names())
parser.add_argument('--dataset', type=str, default='dukemtmcreid', choices=data_manager.get_names())
# PATH
# parser.add_argument('--G_resume_dir', type=str, default='./logs/pcb/dukemtmcreid/best_G.pth.tar', metavar='path to resume G')
parser.add_argument('--pre_dir', type=str, default='models', help='path to be attacked model')
parser.add_argument('--attr_dir', type=str, default='', help='path to attribute file')
parser.add_argument('--save_dir', type=str, default='./log_cs', help='path to save model')
parser.add_argument('--vis_dir', type=str, default='vis', help='path to save visualization result')
parser.add_argument('--ablation', type=str, default='', help='for ablation study')
# var
parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--D', type=str, default='MSGAN', help='Type of discriminator: PatchGAN or Multi-stage GAN')
parser.add_argument('--normalization', type=str, default='bn', help='bn or in')

parser.add_argument('--ak_type', type=int, default=-1, help='-1 if non-targeted, 1 if attribute attack')
parser.add_argument('--attr_key', type=str, default='upwhite', help='[attribute, value]')
parser.add_argument('--attr_value', type=int, default=2, help='[attribute, value]')
parser.add_argument('--mag_in', type=float, default=8, help='l_inf magnitude of perturbation')
parser.add_argument('--temperature', type=float, default=-1, help="tau in paper")
parser.add_argument('--usegumbel', action='store_true', default=False, help='whether to use gumbel softmax')
parser.add_argument('--use_SSIM', type=int, default=0, help="0: None, 1: SSIM, 2: MS-SSIM ")
# Base
parser.add_argument('--train_batch', default=24, type=int,help="train batch size")
parser.add_argument('--test_batch', default=256, type=int, help="test batch size")
parser.add_argument('--epoch', type=int, default=40, help='number of epochs to train for')

parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num_ker', type=int, default=32, help='generator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--maml_lr', type=float, default=0.0001, help='Learning Rate. Default=0.002')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--print_freq', type=int, default=20, help="print frequency")
parser.add_argument('--eval_freq', type=int, default=20, help="eval frequency")
parser.add_argument('--usevis', action='store_true', default=False, help='whether to save vis')
parser.add_argument('--lp', type=float, default=8.0, help='lp max')
parser.add_argument('--xishu', type=float, default=10, help='loss xishu')
parser.add_argument('--log_dir', type=str, default='./log_meta/meta_data')
parser.add_argument('--inter_layer', type=str, default='layer_1', help='[layer_0, layer_1,layer_2,layer_3]')
parser.add_argument('--styl_layer', type=str, default='layer_0', help='[layer_0, layer_1,layer_2,layer_3]')
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--MetaTrainTask_num', type=int, default=5)

parser.add_argument('--multi_domain', type=int, default=1)
parser.add_argument('--multi_model', type=int, default=1)
parser.add_argument('--perturb_erasing', type=int, default=1)
parser.add_argument('--multi_pe', type=int, default=0)
parser.add_argument('--style_mix', type=int, default=1)
parser.add_argument('--lamda', type=float, default=0.5)
parser.add_argument('--Beta_Sum', type=int, default=10)
parser.add_argument('--extra_data', type=int, default=0)
parser.add_argument('--mix_thre', type=float, default=0.6)
parser.add_argument('--out_both', type=int, default=1)

args = parser.parse_args()
is_training = args.mode == 'train'
attr_list = [args.attr_key, args.attr_value]
attr_matrix = None
if args.attr_dir:
  assert args.dataset in ['dukemtmcreid', 'market1501']
  attr_matrix = io.loadmat(args.attr_dir)
  args.ablation = osp.join('attr', args.attr_key + '=' + str(args.attr_value))
if args.targetmodel=='ide':
  pre_dir = osp.join(args.pre_dir, args.targetmodel, args.dataset+'.pth')
  pre_dir_2 = osp.join(args.pre_dir, 'pcb', args.dataset+'.pth')
  pre_dir_3 = osp.join(args.pre_dir, 'vit', args.dataset+'.pth')
  # pre_dir_3 = osp.join(args.pre_dir, 'pfd', args.dataset+'.pth')

else:
  pre_dir = osp.join(args.pre_dir, args.targetmodel, args.dataset+'.pth.tar')
save_dir = args.log_dir
vis_dir = args.log_dir

pdist = torch.nn.PairwiseDistance(p=2)
def main(opt):
  if not osp.exists(save_dir): os.makedirs(save_dir)
  if not osp.exists(vis_dir): os.makedirs(vis_dir)

  use_gpu = torch.cuda.is_available()
  # pin_memory = True if use_gpu else False
  pin_memory = False

  if args.mode == 'train':
    logger = setup_logger("advreid",save_dir, if_train=True)
  else:
    logger = setup_logger("advreid",save_dir, if_train=False)
  logger.info(args)

  if use_gpu:
    logger.info("GPU mode")
    seed = args.seed
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  else:
    logger.info("CPU mode")

  ### Setup dataset loader ###
  logger.info("Initializing dataset {}".format(args.dataset))
  dataset_duke = data_manager.init_img_dataset(root=args.root, name=args.dataset, split_id=opt['split_id'],
                                          cuhk03_labeled=opt['cuhk03_labeled'], cuhk03_classic_split=opt['cuhk03_classic_split'])
  dataset_msmt = data_manager.init_img_dataset(root=args.root, name='viper', split_id=opt['split_id'],
                                          cuhk03_labeled=opt['cuhk03_labeled'],
                                          cuhk03_classic_split=opt['cuhk03_classic_split'])
  dataset_cuhk = data_manager.init_img_dataset(root=args.root, name='cuhk03', split_id=opt['split_id'],
                                               cuhk03_labeled=opt['cuhk03_labeled'],
                                               cuhk03_classic_split=opt['cuhk03_classic_split'])
  trainloader_duke = DataLoader(ImageDataset(dataset_duke.train, transform=opt['transform_train']),
                           sampler=RandomIdentitySampler(dataset_duke.train, num_instances=opt['num_instances']),
                           batch_size=args.train_batch, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=True)
  trainloader_msmt = DataLoader(ImageDataset2(dataset_msmt.train, transform=opt['transform_train']),
                           sampler=RandomIdentitySampler(dataset_msmt.train, num_instances=opt['num_instances']),
                           batch_size=args.train_batch, num_workers=opt['workers'], pin_memory=pin_memory,
                           drop_last=True)
  trainloader_cuhk = DataLoader(ImageDataset2(dataset_cuhk.train, transform=opt['transform_train']),
                                sampler=RandomIdentitySampler(dataset_cuhk.train, num_instances=opt['num_instances']),
                                batch_size=args.train_batch, num_workers=opt['workers'], pin_memory=pin_memory,
                                drop_last=True)

  queryloader = DataLoader(ImageDataset(dataset_duke.query, transform=opt['transform_test']), batch_size=args.test_batch,
                           shuffle=False, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=False)
  galleryloader = DataLoader(ImageDataset(dataset_duke.gallery, transform=opt['transform_test']), batch_size=args.test_batch,
                             shuffle=False, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=False)

  metric_criterion = adv_TripletLoss(margin=args.margin, ak_type=args.ak_type)
  criterionGAN = GANLoss()

  ### Prepare pretrained model
  target_net_ide = models.init_model(name=args.targetmodel, pre_dir=pre_dir, num_classes=dataset_duke.num_train_pids)
  target_net_pcb = models.init_model(name='pcb', pre_dir=pre_dir_2, num_classes=dataset_duke.num_train_pids)
  target_net_vit = models.init_model(name='vit', pre_dir=pre_dir_3, num_classes=dataset_duke.num_train_pids)
  # target_net_vit = models.init_model(name='pfd', pre_dir=pre_dir_3, num_classes=dataset_duke.num_train_pids)


  ### 固定模型参数
  check_freezen(target_net_ide, need_modified=True, after_modified=False)
  check_freezen(target_net_pcb, need_modified=True, after_modified=False)
  check_freezen(target_net_vit, need_modified=True, after_modified=False)

  # target_net.eval()


  G =l2l.algorithms.MAML(Generator(3, 3, args.num_ker, norm=args.normalization,beta=args.beta).apply(weights_init),lr=args.maml_lr,first_order=True)
  if args.D == 'PatchGAN':
    D = Pat_Discriminator(input_nc=6, norm=args.normalization).apply(weights_init)
  elif args.D == 'MSGAN':
    D = MS_Discriminator(input_nc=6, norm=args.normalization, temperature=args.temperature, use_gumbel=args.usegumbel).apply(weights_init)
  check_freezen(G, need_modified=True, after_modified=True)
  check_freezen(D, need_modified=True, after_modified=True)
  logger.info("Model size: {:.5f}M".format((sum(g.numel() for g in G.parameters())+sum(d.numel() for d in D.parameters()))/1000000.0))
  # setup optimizer

  optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
  optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
  if use_gpu:

    test_target_net = target_net_ide.cuda()
    target_net_ide = target_net_ide.cuda()
    target_net_pcb = target_net_pcb.cuda()
    target_net_vit = target_net_vit.cuda()

    G = G.cuda()
    D = D.cuda()


  # Ready
  start_time = time.time()
  train_time = 0
  worst_mAP, worst_rank1, worst_rank5, worst_rank10, worst_epoch = np.inf, np.inf, np.inf, np.inf, 0
  logger.info("==> Start training")
  for epoch in range(1, args.epoch+1):

    start_train_time = time.time()
    train(epoch, G, D, target_net_ide,target_net_pcb,target_net_vit, criterionGAN, optimizer_G, optimizer_D, trainloader_duke, trainloader_msmt,trainloader_cuhk, use_gpu, logger)
    train_time += round(time.time() - start_train_time)
    if epoch % args.eval_freq == 0:
      logger.info("==> Eval at epoch {}".format(epoch))

      cmc, mAP = inference(G, D, test_target_net, dataset_duke, queryloader, galleryloader, epoch, use_gpu,logger, is_test=False)
      is_worst = mAP <= worst_mAP
      save_checkpoint(G.state_dict(), is_worst, 'G', osp.join(save_dir, 'G_ep' + str(epoch) + '.pth.tar'))
      save_checkpoint(D.state_dict(), is_worst, 'D', osp.join(save_dir, 'D_ep' + str(epoch) + '.pth.tar'))
      if is_worst:
        worst_mAP, worst_rank1, worst_epoch = mAP, cmc[0], epoch
        logger.info("==> Worst_epoch is {}, Worst mAP {:.1%}, Worst rank-1 {:.1%}".format(worst_epoch, worst_mAP, worst_rank1))

  elapsed = round(time.time() - start_time)
  elapsed = str(datetime.timedelta(seconds=elapsed))
  train_time = str(datetime.timedelta(seconds=train_time))
  logger.info("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


if args.inter_layer == 'layer_0':
  mask_avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
elif args.inter_layer == 'layer_1':
  mask_avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
elif args.inter_layer == 'layer_2':
  mask_avgpool = nn.AvgPool2d(kernel_size=8, stride=8)
elif args.inter_layer == 'layer_3':
  mask_avgpool = nn.AvgPool2d(kernel_size=16, stride=16)


def min_max_div(input):
  '''
  input (B,H,W)
  '''
  out_put = torch.zeros_like(input)
  for i in range(len(input)):
    min_vals = torch.min(input[i])
    max_vals = torch.max(input[i])

    # 最小-最大缩放，将x的范围缩放到[0, 1]
    scaled_x = (input[i] - min_vals) / (max_vals - min_vals)
    out_put[i] = scaled_x
  return out_put

def train(epoch, G, D, target_net_1,target_net_2,target_net_3, criterionGAN,optimizer_G, optimizer_D, trainloader_1,trainloader_2,trainloader_3,use_gpu,logger):
  G.train()
  D.train()
  is_training=True
  torch.autograd.set_detect_anomaly(True)
  for batch_idx, content in enumerate(zip(trainloader_1,cycle(trainloader_2),trainloader_3)):
    imgs_1, pids, parsing_mask, camids = content[0]
    imgs_2, pids_2, _, _ = content[1]
    imgs_3,pids_3, _, _ = content[2]
    target_models = [target_net_1,target_net_2,target_net_3]
    target_models_NAME = ['ide','pcb','vit']
    iteration_error = 0.0
    iteration_error_D = 0.0
    randomE_list = [randomErasing_patch,randomErasing_Vertical,randomErasing_horizontal]
    if use_gpu:
      imgs_1= imgs_1.cuda()
      imgs_2= imgs_2.cuda()
      imgs_3= imgs_3.cuda()
      imgs_list = [imgs_1, imgs_2, imgs_3]

    #三三组合形成train_task和test_task，27*16组，随机选五组进行训练
    # Todo 这里的任务数量可能还需要调整！
    for _ in range(args.MetaTrainTask_num):
      G_meta_clone = G.clone()
      '''
      parser.add_argument('--multi_domain', type=int, default=1)
      parser.add_argument('--multi_model', type=int, default=1)
      parser.add_argument('--perturb_erasing', type=int, default=1)
      parser.add_argument('--style_mix', type=int, default=1)
      parser.add_argument('--multi_pe', type=int, default=1)
      '''
      # 1到9的组合中，选取两个组合凑成train_task和test_task
      random_D = random.sample(list(range(0, 3)), 2) if args.multi_domain==1 else [0,0]
      random_M = random.sample(list(range(0, 3)), 2) if args.multi_model==1 else [0,0]
      random_E = random.sample(list(range(0, 3)), 2) if args.multi_pe==1 else [0,0]

      train_data = imgs_list[random_D[0]];train_target_model = target_models[random_M[0]];train_earsing=randomE_list[random_E[0]];train_model_name = target_models_NAME[random_M[0]]
      test_data = imgs_list[random_D[1]];test_target_model = target_models[random_M[1]];test_earsing=randomE_list[random_E[1]];test_model_name = target_models_NAME[random_M[1]]
      extra_data = imgs_list[list(set([0,1,2]) - set(random_D))[0]]
      # train task process
      for step in range(1):

        new_imgs, mask = perturb_train(train_data, G_meta_clone, D, train_or_test='train')
        new_imgs = new_imgs.cuda()
        pred_fake_pool, _ = D(torch.cat((train_data, new_imgs.detach()), 1))
        loss_D_fake = criterionGAN(pred_fake_pool, False)
        # Real Detection and Loss
        num = args.train_batch//2
        pred_real, _ = D(torch.cat((train_data[0:num,:,:,:], train_data[num:,:,:,:].detach()), 1))
        loss_D_real = criterionGAN(pred_real, True)
        # GAN loss (Fake Passability Loss)
        pred_fake, _ = D(torch.cat((train_data, new_imgs), 1))
        loss_G_GAN = criterionGAN(pred_fake, True)
        # Re-ID advloss

        ls = train_target_model(train_data, is_training, metaTrain=True)
        if len(ls) == 1: outputs = ls[0]
        if len(ls) == 2: outputs, features = ls
        if len(ls) == 3: outputs, features, bn_info = ls

        ls = train_target_model(new_imgs, is_training)
        if len(ls) == 1: new_outputs = ls[0]
        if len(ls) == 2: new_outputs, new_features = ls
        if len(ls) == 3: new_outputs, new_features, new_bn_info = ls


        if train_model_name == 'pcb':
          reid_loss = - args.xishu/2.6 * torch.mean(pdist(features, new_features))
        else:
          reid_loss = - args.xishu * torch.mean(pdist(features, new_features))
        loss_G_ReID = reid_loss
        # metaTrain_loss_D = (loss_D_fake + loss_D_real)/2
        error_train = loss_G_GAN + loss_G_ReID

        G_meta_clone.adapt(error_train)
        # G_meta_clone.adapt(error_train,allow_nograd=True)

      if args.perturb_erasing==1:
        meta_new_imgs, mask = perturb_train(test_data, G_meta_clone, D, train_or_test='train',randE=test_earsing)
      else:
        meta_new_imgs, mask = perturb_train(test_data, G_meta_clone, D, train_or_test='train')
      # GAN loss (Fake Passability Loss)
      meta_pred_fake, _ = D(torch.cat((test_data, meta_new_imgs), 1))
      meta_loss_G_GAN = criterionGAN(meta_pred_fake, True)

      pred_fake_pool, _ = D(torch.cat((test_data, meta_new_imgs.detach()), 1))
      loss_D_fake = criterionGAN(pred_fake_pool, False)
      # Real Detection and Loss
      num = args.train_batch // 2
      pred_real, _ = D(torch.cat((test_data[0:num, :, :, :], test_data[num:, :, :, :].detach()), 1))
      loss_D_real = criterionGAN(pred_real, True)

      metaTest_loss_D = (loss_D_fake + loss_D_real)/2

      if args.style_mix!=1:
        # Re-ID advloss
        meta_ls = test_target_model(test_data, is_training)
        if len(meta_ls) == 1: meta_outputs = meta_ls[0]
        if len(meta_ls) == 2: meta_outputs, meta_features = meta_ls
        if len(meta_ls) == 3: meta_outputs, meta_features, meta_bn_info = meta_ls

        meta_ls = test_target_model(meta_new_imgs, is_training)
        if len(meta_ls) == 1: meta_new_outputs = meta_ls[0]
        if len(meta_ls) == 2: meta_new_outputs, meta_new_features = meta_ls
        if len(meta_ls) == 3: meta_new_outputs, meta_new_features, meta_new_bn_info = meta_ls
      else:
        # Re-ID advloss
        if args.extra_data ==0:
          bn_info = test_target_model(train_data, is_training)[2]
        else:
          bn_info = test_target_model(extra_data, is_training)[2]
        # print(int(args.Beta_Sum*(args.lamda)))
        # print(int(args.Beta_Sum*(1-(args.lamda))))
        lamd = beta.rvs(math.ceil(args.Beta_Sum*(args.lamda)), math.ceil(args.Beta_Sum*(1-(args.lamda))))
        mix_probablity = torch.rand(1)
        meta_ls = test_target_model(test_data, is_training, metaTrain=False,mix_thre=args.mix_thre,mix_pro=mix_probablity, mix_info=bn_info, lamd=lamd,output_both = args.out_both)

        if len(meta_ls) == 1: meta_outputs = meta_ls[0]
        if len(meta_ls) == 2: meta_outputs, meta_features = meta_ls
        if len(meta_ls) == 3: meta_outputs, meta_features, meta_bn_info = meta_ls

        meta_ls = test_target_model(meta_new_imgs, is_training, metaTrain=False, mix_thre=args.mix_thre, mix_pro=mix_probablity, mix_info=bn_info, lamd=lamd,output_both = args.out_both)
        if len(meta_ls) == 1: meta_new_outputs = meta_ls[0]
        if len(meta_ls) == 2: meta_new_outputs, meta_new_features = meta_ls
        if len(meta_ls) == 3: meta_new_outputs, meta_new_features, meta_new_bn_info = meta_ls

      if test_model_name == 'pcb':
        meta_reid_loss = - args.xishu / 2.6 * torch.mean(pdist(meta_features, meta_new_features))
      else:
        meta_reid_loss = - args.xishu * torch.mean(pdist(meta_features, meta_new_features))

      meta_loss_G_ReID = meta_reid_loss
      iteration_error += meta_loss_G_GAN + meta_loss_G_ReID
      iteration_error_D += metaTest_loss_D
    iteration_error = iteration_error/args.MetaTrainTask_num
    iteration_error_D = iteration_error_D/args.MetaTrainTask_num
    optimizer_G.zero_grad()
    iteration_error.backward()
    optimizer_G.step()

    optimizer_D.zero_grad()
    iteration_error_D.backward()
    optimizer_D.step()

    # if (batch_idx+1) % args.print_freq == 0:
    #   logger.info("===> Epoch[{}]({}/{}) loss_D: {:.4f} loss_G_GAN: {:.4f} loss_G_ReID: {:.4f} loss_G_SSIM: {:.4f}".format(epoch, batch_idx, len(trainloader), loss_D.item(), loss_G_GAN.item(), loss_G_ReID.item(), loss_G_ssim))
    if (batch_idx+1) % args.print_freq == 0:
      logger.info(
        "===> Epoch[{}/{}]({}/{}) loss_D: {:.4f} loss_meta_train: {:.4f} loss_meta_test: {:.4f} ".format(epoch,(args.epoch+1),batch_idx,len(trainloader_1),metaTest_loss_D.item(),loss_G_ReID.item(),meta_loss_G_ReID.item()))

def inference(G, D, target_net, dataset, queryloader, galleryloader, epoch, use_gpu,logger, is_test=False, ranks=[1, 5, 10, 20]):
  global is_training
  is_training = False

  # target_net.eval()
  with torch.no_grad():
    qf, lqf, new_qf, new_lqf, q_pids, q_camids = extract_and_perturb\
      (queryloader, G, D, target_net, use_gpu, query_or_gallery='query', is_test=is_test, epoch=epoch,logger=logger)
    gf, lgf, g_pids, g_camids = extract_and_perturb\
      (galleryloader, G, D, target_net, use_gpu, query_or_gallery='gallery', is_test=is_test, epoch=epoch,logger=logger)

    distmat, cmc, mAP = make_results(qf, gf, lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type)
    new_distmat, new_cmc, new_mAP = make_results(new_qf, gf, new_lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type)
    logger.info("Results ----------")
    logger.info("Before, mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(mAP, ranks[0], cmc[ranks[0]-1], ranks[1], cmc[ranks[1]-1], ranks[2], cmc[ranks[2]-1], ranks[3], cmc[ranks[3]-1]))
    logger.info("After , mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(new_mAP, ranks[0], new_cmc[ranks[0]-1], ranks[1], new_cmc[ranks[1]-1], ranks[2], new_cmc[ranks[2]-1], ranks[3], new_cmc[ranks[3]-1]))
    if args.usevis:
      visualize_ranked_results(distmat, dataset, save_dir=osp.join(vis_dir, 'origin_results'), topk=20)
    if args.usevis:
      visualize_ranked_results(new_distmat, dataset, save_dir=osp.join(vis_dir, 'polluted_results'), topk=20)
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
      G.eval()
      D.eval()
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
      if is_test:
        save_img(ls, pids, camids, epoch, batch_idx)

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
  if args.targetmodel in ['pcb', 'lsro']:
    ls = [target_net(imgs, is_training)[0] + target_net(fliplr(imgs), is_training)[0]]
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

def perturb_train(imgs, G, D, train_or_test='test',cam_ids=None,randE=None):
  n,c,h,w = imgs.size()
  stylize = False
  delta= G(imgs, stylsTrans=stylize, inter_layer=args.styl_layer,cam_info=cam_ids)
  # print(delta.size())
  # NOTE 这里的L_norm能保障灰度图像加上还是灰度图像
  delta = L_norm(delta,train_or_test)
  # print(delta.size()) torch.Size([128, 1, 288, 144])
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())
  # 加入噪声和没加入噪声的!
  _, mask = D(torch.cat((imgs, new_imgs.detach()), 1))
  # mask.size(B,1,256,128)
  # delta.size(B,3,256,128)
  # 经过Mask的才是最后的噪声,然后加入生成adv_img
  if randE!=None:
    mask = randE(mask)
  delta = delta * mask

  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())

  for c in range(3):
    new_imgs.data[:,c,:,:] = new_imgs.data[:,c,:,:].clamp((0.0 - Imagenet_mean[c]) / Imagenet_stddev[c],
                                                          (1.0 - Imagenet_mean[c]) / Imagenet_stddev[c]) # do clamping per channel

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

  for c in range(delta.size(1)):
    delta.data[:,c,:,:] = (delta.data[:,c,:,:]) / Imagenet_stddev[c]
  return delta
# def L_norm(delta, mode='train'):
#   '''
#   大概意思就是将 求噪声的幅值_scaled！
#   '''
#   # 当时的transform ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
#   delta.data += 1
#   delta.data *= 0.5
#   # 从[-1,1]归一化到[0,1]
#   for c in range(3):
#     delta.data[:,c,:,:] = (delta.data[:,c,:,:] - Imagenet_mean[c]) / Imagenet_stddev[c]
#
#   bs = args.train_batch if (mode == 'train') else args.test_batch
#   for i in range(bs):
#     # do per channel l_inf normalization
#     for ci in range(3):
#       try:
#         l_inf_channel = delta[i,ci,:,:].data.abs().max()
#         # l_inf_channel = torch.norm(delta[i,ci,:,:]).data
#         mag_in_scaled_c = args.mag_in/(255.0*Imagenet_stddev[ci])
#         delta[i,ci,:,:].data *= np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu()).float().cuda()
#       except IndexError:
#         break
#   return delta

def save_img(ls, pids, camids, epoch, batch_idx):
  image, new_image, delta, mask = ls
  # undo normalize image color channels
  delta_tmp = torch.zeros(delta.size())
  for c in range(3):
    image.data[:,c,:,:] = (image.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]
    new_image.data[:,c,:,:] = (new_image.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]
    delta_tmp.data[:,c,:,:] = (delta.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]

  if args.usevis:
    torchvision.utils.save_image(image.data, osp.join(vis_dir, 'original_epoch{}_batch{}.png'.format(epoch, batch_idx)))
    torchvision.utils.save_image(new_image.data, osp.join(vis_dir, 'polluted_epoch{}_batch{}.png'.format(epoch, batch_idx)))
    torchvision.utils.save_image(delta_tmp.data, osp.join(vis_dir, 'delta_epoch{}_batch{}.png'.format(epoch, batch_idx)))
    torchvision.utils.save_image(mask.data*255, osp.join(vis_dir, 'mask_epoch{}_batch{}.png'.format(epoch, batch_idx)))


if __name__ == '__main__':
  opt = get_opts(args.targetmodel)
  main(opt)
