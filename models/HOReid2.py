import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from bisect import bisect_right
import os

from models.HOReid_models import Encoder, BNClassifiers
from models.HOReid_models import ScoremapComputer, compute_local_features
from models.HOReid_models import GraphConvNet, generate_adj
from models.HOReid_models import GMNet, PermutationLoss, Verificator, mining_hard_pairs

def accuracy(output, target, topk=[1]):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


class HOReid(nn.Module):

    def __init__(self, num_classes):
        super(HOReid, self).__init__()

        self.pid_num = num_classes
        self.margin = 0.3
        self.branch_num = 14

        # feature learning
        self.encoder = Encoder(class_num=self.pid_num)
        self.bnclassifiers = BNClassifiers(2048, self.pid_num, self.branch_num)
        self.bnclassifiers2 = BNClassifiers(2048, self.pid_num, self.branch_num)  # for gcned features
        self.encoder = nn.DataParallel(self.encoder)
        self.bnclassifiers = nn.DataParallel(self.bnclassifiers)
        self.bnclassifiers2 = nn.DataParallel(self.bnclassifiers2)

        # keypoints model
        self.scoremap_computer = ScoremapComputer(10.0)
        # self.scoremap_computer = nn.DataParallel(self.scoremap_computer)
        self.scoremap_computer = self.scoremap_computer.eval()

        # GCN
        self.linked_edges = \
            [[13, 0], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10],
             [13, 11], [13, 12],  # global
             [0, 1], [0, 2],  # head
             [1, 2], [1, 7], [2, 8], [7, 8], [1, 8], [2, 7],  # body
             [1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12],  # libs
             # [3,4],[5,6],[9,10],[11,12], # semmetric libs links
             ]
        self.adj = generate_adj(self.branch_num, self.linked_edges, self_connect=0.0).cuda()
        self.gcn = GraphConvNet(self.adj, 2048, 2048, 2048, 20.0)

        # graph matching
        self.gmnet = GMNet()

        # verification
        self.verificator = Verificator()

     ## resume model from resume_epoch
    def resume_model_from_path(self, path, resume_epoch):
        self.encoder.load_state_dict(
            torch.load(os.path.join(path, 'encoder_{}.pkl'.format(resume_epoch))))
        self.bnclassifiers.load_state_dict(
            torch.load(os.path.join(path, 'bnclassifiers_{}.pkl'.format(resume_epoch))))
        self.bnclassifiers2.load_state_dict(
            torch.load(os.path.join(path, 'bnclassifiers2_{}.pkl'.format(resume_epoch))))
        self.gcn.load_state_dict(torch.load(
            os.path.join(path, 'gcn_{}.pkl'.format(resume_epoch))))
        self.gmnet.load_state_dict(torch.load(
            os.path.join(path, 'gmnet_{}.pkl'.format(resume_epoch))))
        self.verificator.load_state_dict(
            torch.load(os.path.join(path, 'verificator_{}.pkl'.format(resume_epoch))))


    def forward(self, images, pids,  is_training=False):

        # feature
        feature_maps = self.encoder(images)
        with torch.no_grad():
            score_maps, keypoints_confidence, _ = self.scoremap_computer(images)
        feature_vector_list, keypoints_confidence = compute_local_features(
            feature_maps, score_maps, keypoints_confidence)
        bned_feature_vector_list, cls_score_list = self.bnclassifiers(feature_vector_list)

        # gcn
        gcned_feature_vector_list = self.gcn(feature_vector_list)
        bned_gcned_feature_vector_list, gcned_cls_score_list = self.bnclassifiers2(gcned_feature_vector_list)

        # if is_training:
        #
        #     # mining hard samples
        #     new_bned_gcned_feature_vector_list, bned_gcned_feature_vector_list_p, bned_gcned_feature_vector_list_n = mining_hard_pairs(
        #         bned_gcned_feature_vector_list, pids)
        #
        #     # graph matching
        #     s_p, emb_p, emb_pp = self.gmnet(new_bned_gcned_feature_vector_list, bned_gcned_feature_vector_list_p, None)
        #     s_n, emb_n, emb_nn = self.gmnet(new_bned_gcned_feature_vector_list, bned_gcned_feature_vector_list_n, None)
        #
        #     # verificate
        #     # ver_prob_p = self.verificator(bned_gcned_feature_vector_list, bned_gcned_feature_vector_list_p)
        #     # ver_prob_n = self.verificator(bned_gcned_feature_vector_list, bned_gcned_feature_vector_list_n)
        #     ver_prob_p = self.verificator(emb_p, emb_pp)
        #     ver_prob_n = self.verificator(emb_n, emb_nn)
        #
        #     return (feature_vector_list, gcned_feature_vector_list), \
        #            (cls_score_list, gcned_cls_score_list), \
        #            (ver_prob_p, ver_prob_n), \
        #            (s_p, emb_p, emb_pp),\
        #            (s_n, emb_n, emb_nn), \
        #            keypoints_confidence
        # else:
        bs, keypoints_num = keypoints_confidence.shape
        keypoints_confidence = torch.sqrt(keypoints_confidence).unsqueeze(2).repeat([1, 1, 2048]).view(
            [bs, 2048 * keypoints_num])


        features_stage1 = keypoints_confidence * torch.cat(bned_feature_vector_list, dim=1)
        features_satge2 = torch.cat([i.unsqueeze(1) for i in bned_feature_vector_list], dim=1)
        gcned_features_stage1 = keypoints_confidence * torch.cat(bned_gcned_feature_vector_list, dim=1)
        gcned_features_stage2 = torch.cat([i.unsqueeze(1) for i in bned_gcned_feature_vector_list], dim=1)
        # print(gcned_features_stage1.size())


        return [None, torch.cat([gcned_features_stage1])]


