# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core import bbox2roi

from ..builder import ROTATED_HEADS
from .rotate_standard_roi_head import RotatedStandardRoIHead

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchviz

from mmrotate.core import rbbox2roi, rbbox2result, obb2xyxy
from ..builder import ROTATED_HEADS, build_head
from .rotate_standard_roi_head import RotatedStandardRoIHead

from ..utils.d_classifier import *
from ..utils.angle_utils import cal_loss_angle, cal_simi, AnglePerecption, IncompleteFeatSimulator, AnglePerecption2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

@ROTATED_HEADS.register_module()
class GVRatioRoIHead(RotatedStandardRoIHead):
    """Gliding vertex roi head including one bbox head."""
    
    def __init__(self,
                 warmup=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 version='oc'):

        super(GVRatioRoIHead, self).__init__(bbox_roi_extractor, bbox_head, 
                                    shared_head, train_cfg, test_cfg, 
                                    pretrained, init_cfg, version)

        self.perception = AnglePerecption(4, bbox_head)
        for param in self.perception.parameters():
            param.requires_grad = False       
        self.load_pretrained_weights('/media/datasets/gpu17_models/mmrotate/ours/ours_r50_fpn_3x_roxray_p_le90/20240324_783.pth')
        
        self.simulator = IncompleteFeatSimulator(4, bbox_head['fc_out_channels'])
        self.prototype_features = [torch.zeros((bbox_head['fc_out_channels'])).cuda() for _ in range(bbox_head['num_classes'])]
        self.extractor = nn.Linear(in_features=bbox_head['fc_out_channels'], out_features=bbox_head['fc_out_channels'])
        self.gate = False
        self.iters = 0
        self.warmup = 3000 #3000
        self.warmup_2 = 3400 #3400
        self.level3_before = None
        self.level3_after = None
        self.level3_cls = None
        self.tsne_num = 1
        self.flag = False
        
    def load_pretrained_weights(self, pretrained):
        checkpoint = torch.load(pretrained)
        weight_fc_angle = checkpoint['state_dict']['roi_head.perception.fc_angle.weight']
        bias_fc_angle = checkpoint['state_dict']['roi_head.perception.fc_angle.bias']
        weight_fc_x = checkpoint['state_dict']['roi_head.perception.fc_x.weight']
        bias_fc_x = checkpoint['state_dict']['roi_head.perception.fc_x.bias']
        weight_fc_y = checkpoint['state_dict']['roi_head.perception.fc_y.weight']
        bias_fc_y = checkpoint['state_dict']['roi_head.perception.fc_y.bias']

        with torch.no_grad():
            self.perception.fc_angle.weight.copy_(weight_fc_angle)
            self.perception.fc_angle.bias.copy_(bias_fc_angle)
            self.perception.fc_x.weight.copy_(weight_fc_x)
            self.perception.fc_x.bias.copy_(bias_fc_x)
            self.perception.fc_y.weight.copy_(weight_fc_y)
            self.perception.fc_y.bias.copy_(bias_fc_y)

    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                bbox_results['fix_pred'],
                bbox_results['ratio_pred'],
            )
        return outs
    
    
    def forward_angle(self, x, img_metas, proposal_list, gt_bboxes, gt_labels):
        
        rois = rbbox2roi([bboxes for bboxes in gt_bboxes])
        bbox_results = self._bbox_forward(x, rois)
        
        x_pre, y_pre = self.perception(bbox_results['bbox_feats_flatten'])

        _ , x_pre_max = torch.max(x_pre, dim=-1)
        _ , y_pre_max = torch.max(y_pre, dim=-1)

        return x_pre_max, y_pre_max
    
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      sim_bboxes=None,
                      sim_angles=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            # sampling_results_sim = []
            for i in range(num_imgs):
                gt_hbboxes = obb2xyxy(gt_bboxes[i], self.version)
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_hbboxes, gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_hbboxes,
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])


                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)


        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels, 
                                                    # simulated, angles, 
                                                    sim_bboxes, sim_angles,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing.

        Args:
            x (list[Tensor]): list of multi-level img features.
            rois (list[Tensors]): list of region of interests.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, fix_pred, ratio_pred, bbox_feats_flatten, angle_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            fix_pred=fix_pred,
            ratio_pred=ratio_pred,
            bbox_feats=bbox_feats,
            bbox_feats_flatten=bbox_feats_flatten, angle_pred=angle_pred)
        return bbox_results
    
    def update_prototype_feature(self, feats_ori, scores_ori, x_angle, y_angle, gt_labels, thresh=0.7, epsilon=1e-6):

        
        _, levelx = torch.max(x_angle, dim=-1)
        _, levely = torch.max(y_angle, dim=-1)
        angle_max = torch.max(levelx, levely)
        # satisfied = angle_max < 3
        satisfied = angle_max < 1
        num_classes = scores_ori.shape[1] - 1

        idx = torch.nonzero(satisfied).squeeze(dim=-1)
        if idx.shape[0] == 0:
            prototype_features = []
            for c in range(0, num_classes):
                # tmp_feat = self.prototype_features[c].clone()
                # for l in self.extractor:
                #     tmp_feat = l(tmp_feat)
                prototype_feature = self.prototype_features[c].clone()
                # prototype_feature = self.relu(self.bn(prototype_feature))
                prototype_features.append(prototype_feature)
            prototype_features = torch.stack(prototype_features, dim=0)
            return prototype_features
        feats = torch.index_select(feats_ori, 0, idx)
        labels = torch.index_select(gt_labels, 0, idx)
        scores = torch.index_select(scores_ori, 0, idx)
        
        scores = F.softmax(scores, dim=-1)
        all_class_feat = list()
        for i in range(num_classes):
            idx = torch.nonzero(labels == i).squeeze(dim=-1)
            if idx.shape[0] != 0:
                feats_cls = torch.index_select(feats, 0, idx)
                feat_cls = torch.sum(feats_cls, dim=0)
                avg_feat_cls = feat_cls / idx.shape[0]
            else:
                avg_feat_cls = torch.zeros_like(feats[0,:])
            # cls_prob = scores[:, i].view(scores.size(0), 1)
            # #pdb.set_trace()
            # tmp_class_feat = feats * cls_prob
            # class_feat = torch.sum(tmp_class_feat, dim=0) / (torch.sum(cls_prob) + epsilon)
            all_class_feat.append(avg_feat_cls)
        all_class_feat = torch.stack(all_class_feat, dim = 0)

        # if self.prototype_features == None:
        #     self.prototype_features = all_class_feat.detach().clone()
            # self.prototype_features.requires_grad_()
            
        for c in range(0, num_classes):
            if torch.equal(self.prototype_features[c], torch.zeros_like(self.prototype_features[c])):
                # self.prototype_features[c] = all_class_feat[c].detach().clone()
                self.prototype_features[c] = all_class_feat[c].detach()
                continue
            if torch.equal(all_class_feat[c], torch.zeros_like(all_class_feat[c])):
                continue
            alpha = F.cosine_similarity(self.prototype_features[c], all_class_feat[c], dim=0).item()
            alpha = 0.2
            feature_updated = (1.0 - alpha) * self.prototype_features[c].detach() + alpha * all_class_feat[c]
            # feature_updated = alpha * self.prototype_features[c] + (1 - alpha) * all_class_feat[c]
            self.prototype_features[c] = feature_updated.detach()
        
        # prototype_features = self.relu(self.bn(prototype_features))
        prototype_features = self.get_prototype_features()
        # matrix = torch.zeros((num_classes, num_classes))
        # for a in range(0, num_classes):
        #     for b in range(0, num_classes):
        #         alpha = F.cosine_similarity(self.prototype_features[a], self.prototype_features[b], dim=0)
        #         matrix[a,b] = alpha
        # print(matrix)
        return prototype_features
    
    def get_prototype_features(self):
        prototype_features = []
        for c in range(0, len(self.prototype_features)):
            # tmp_feat = self.prototype_features[c].clone()
            # for l in self.extractor:
            #     tmp_feat = l(tmp_feat)
            # prototype_feature = self.prototype_features[c].clone()
            prototype_feature = self.extractor(self.prototype_features[c].clone())
            # prototype_feature = self.relu(self.bn(prototype_feature))
            prototype_features.append(prototype_feature)
        prototype_features = torch.stack(prototype_features, dim=0)
        return prototype_features
    
    def contrastive_loss(self, prototype_features):
        simi = cal_simi(prototype_features, prototype_features)
        mask = torch.ones_like(simi)
        mask.fill_diagonal_(0)
        bool_mask = mask == 1
        abs_sim_masked = simi[bool_mask]
        loss = torch.mean(abs_sim_masked)
        return loss
    
    def check_full(self):
        if self.prototype_features == None:
            return False
        num_classes = self.prototype_features.shape[0]
        for c in range(num_classes):
            if torch.equal(self.prototype_features[c], torch.zeros_like(self.prototype_features[c])):
                return False
        
        return True
    

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            sim_boxes, sim_angles,
                            img_metas):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        
        if sim_boxes != None:
            sim_rois = rbbox2roi(sim_boxes)
            sim_bbox_results = self._bbox_forward(x, sim_rois)
            
        pos_feats, pos_feats_flatten, pos_scores, pos_angles_pred, pos_gts = get_prepared(
                                                bbox_results['bbox_feats'], 
                                                bbox_results['bbox_feats_flatten'], bbox_results['cls_score'], bbox_results['angle_pred'],
                                                sampling_results, 
                                            )
        
        x_pre, y_pre = self.perception(pos_feats_flatten)
        
        if sim_boxes == None:
            pass
        else:
            x_pre_sim, y_pre_sim = self.perception(sim_bbox_results['bbox_feats_flatten'])

        self.iters += 1
        
        if self.iters > self.warmup:
            prototype_features = self.update_prototype_feature(pos_feats_flatten, pos_scores, x_pre, y_pre, pos_gts, thresh=0.3)

        if self.iters > self.warmup_2:
            x_pre_all, y_pre_all = self.perception(bbox_results['bbox_feats_flatten'])
            feats_sim_all = self.simulator(bbox_results['bbox_feats_flatten'], x_pre_all, y_pre_all)
            x_pre_all_aftersim, y_pre_all_aftersim = self.perception(feats_sim_all)
            angles_aftersim = torch.zeros(x_pre_all_aftersim.shape).cuda()
            loss_angle_aftersim = self.perception.loss_angle_onlysim(x_pre_all_aftersim, y_pre_all_aftersim, angles_aftersim)
            
            
            new_cls_scores = self.bbox_head.forward_cls(feats_sim_all)
            new_reg = self.bbox_head.forward_reg(feats_sim_all)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        bbox_results['fix_pred'],
                                        bbox_results['ratio_pred'], rois,
                                        *bbox_targets)
        
        if self.iters > self.warmup_2:
            feats_sim = self.simulator(pos_feats_flatten, x_pre, y_pre)
            loss_simulate = self.simulator.loss_simulate(feats_sim, pos_gts, prototype_features)
            loss_bbox['loss_simulate'] = loss_simulate 

            new_loss_bbox = self.bbox_head.loss(new_cls_scores,
                                        new_reg, bbox_results['fix_pred'],
                                        bbox_results['ratio_pred'], rois,
                                        *bbox_targets)
            
            loss_bbox['new_loss_cls'] = new_loss_bbox['loss_cls']
            
            full_cls_score = self.bbox_head.forward_cls(prototype_features)
            full_labels = torch.arange(0, prototype_features.shape[0], step=1).cuda()
            label_weights = torch.ones((prototype_features.shape[0]), dtype=torch.float).cuda()
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            loss_cls_ = self.bbox_head.loss_cls(
                    full_cls_score,
                    full_labels,
                    label_weights,
                    avg_factor=avg_factor)

            
            loss_bbox['loss_angle_aftersim'] = loss_angle_aftersim
        

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains \
                the boxes of the corresponding image in a batch, each \
                tensor has the shape (num_boxes, 5) and last dimension \
                5 represent (cx, cy, w, h, a, score). Each Tensor \
                in the second list is the labels with shape (num_boxes, ). \
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        if self.iters > self.warmup_2:
            x_pre_all, y_pre_all = self.perception(bbox_results['bbox_feats_flatten'])
            feats_sim_all = self.simulator(bbox_results['bbox_feats_flatten'], x_pre_all, y_pre_all)
            cls_scores = self.bbox_head.forward_cls(feats_sim_all)
            bbox_results['cls_score'] = (cls_scores + bbox_results['cls_score'])/2

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        fix_pred = bbox_results['fix_pred'],
        ratio_pred = bbox_results['ratio_pred'],
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            fix_pred = fix_pred[0].split(num_proposals_per_img, 0)
            ratio_pred = ratio_pred[0].split(num_proposals_per_img, 0)

        else:
            bbox_pred = (None, ) * len(proposals)
            fix_pred = (None, ) * len(proposals)
            ratio_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                fix_pred[i],
                ratio_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=self.test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels

def get_prepared(feats, feats_flatten, scores, angles_pred, sampling_results, 
                #  simulated, angles
                ):
    batch_size = len(sampling_results)
    feats_list = feats.chunk(batch_size, dim = 0)
    feats_flatten_list = feats_flatten.chunk(batch_size, dim = 0)
    scores_list = scores.chunk(batch_size, dim = 0)
    angles_pred_list = angles_pred.chunk(batch_size, dim = 0)
    gt_idxs = [res.pos_assigned_gt_inds for res in sampling_results]
    pos_cnts = [res.pos_inds.shape[0]  for res in sampling_results]
    pos_gts = [res.pos_gt_labels for res in sampling_results]
    pos_feats_list = []
    pos_feats_flatten_list = []
    pos_scores_list = []
    pos_angles_pred_list = []
    # pos_simulated_list = []
    # pos_angles_list = []
    # for i, (feat, feat_flatten, score, angle_pred, sim_label, angle, gt_idx, pos_cnt) in enumerate(
    #     zip(feats_list, feats_flatten_list, scores_list, angles_pred_list, simulated, angles, gt_idxs, pos_cnts)):
    for i, (feat, feat_flatten, score, angle_pred, gt_idx, pos_cnt) in enumerate(
        zip(feats_list, feats_flatten_list, scores_list, angles_pred_list, gt_idxs, pos_cnts)):
        pos_feats = feat[:pos_cnt, :, :, :]
        pos_feats_flatten = feat_flatten[:pos_cnt, :]
        pos_scores = score[:pos_cnt, :]
        pos_angles_pred = angle_pred[:pos_cnt, :]
        # pos_simulated = torch.index_select(sim_label, 0, gt_idx)
        # pos_angles = torch.index_select(angle, 0, gt_idx)
        pos_feats_list.append(pos_feats)
        pos_feats_flatten_list.append(pos_feats_flatten)
        pos_scores_list.append(pos_scores)
        pos_angles_pred_list.append(pos_angles_pred)
        # pos_simulated_list.append(pos_simulated)
        # pos_angles_list.append(pos_angles)
    pos_feats_all = torch.cat(pos_feats_list, 0)
    pos_feats_flatten_all = torch.cat(pos_feats_flatten_list, 0)
    pos_scores_all = torch.cat(pos_scores_list, 0)
    pos_angles_pred_all = torch.cat(pos_angles_pred_list, 0)
    # pos_simulated_all = torch.cat(pos_simulated_list, 0)
    # pos_angles_all = torch.cat(pos_angles_list, 0)
    pos_gts_all = torch.cat(pos_gts, 0)
    # return pos_feats_all, pos_feats_flatten_all, pos_scores_all, pos_angles_pred_all, pos_simulated_all, pos_angles_all, pos_gts_all
    return pos_feats_all, pos_feats_flatten_all, pos_scores_all, pos_angles_pred_all, pos_gts_all
    # return pos_feats_all, pos_simulated_all, pos_angles_all