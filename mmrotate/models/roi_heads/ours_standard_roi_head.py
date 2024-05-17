# # Copyright (c) OpenMMLab. All rights reserved.
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import torchviz

# from mmrotate.core import rbbox2roi, rbbox2result, obb2xyxy
# from ..builder import ROTATED_HEADS, build_head
# from .rotate_standard_roi_head import RotatedStandardRoIHead

# from ..utils.d_classifier import *
# from ..utils.angle_utils import cal_loss_angle, cal_simi, AnglePerecption, IncompleteFeatSimulator, AnglePerecption2

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import pandas as pd


# @ROTATED_HEADS.register_module()
# class OursStandardRoIHead(RotatedStandardRoIHead):
#     """Ours RCNN roi head including one bbox head."""
#     def __init__(self,
#                  warmup=None,
#                  bbox_roi_extractor=None,
#                  bbox_head=None,
#                  shared_head=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  pretrained=None,
#                  init_cfg=None,
#                  version='oc'):

#         super(OursStandardRoIHead, self).__init__(bbox_roi_extractor, bbox_head, 
#                                     shared_head, train_cfg, test_cfg, 
#                                     pretrained, init_cfg, version)
#         # self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
#         # self.bbox_head_sim = build_head(bbox_head)
#         # self.domain_classifier = DomainClassifier(dim = bbox_head['fc_out_channels'])
#         # self.discriminator = Discriminator(dim = bbox_head['fc_out_channels'])
#         # self.optimizor_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.005, betas=(0.5, 0.999))
#         self.perception = AnglePerecption(4, bbox_head)
#         for param in self.perception.parameters():
#             param.requires_grad = False       
#         self.load_pretrained_weights('/media/datasets/gpu17_models/mmrotate/ours/ours_r50_fpn_3x_roxray_p_le90/20240324_783.pth')
        
#         self.simulator = IncompleteFeatSimulator(4, bbox_head['fc_out_channels'])
#         self.prototype_features = [torch.zeros((bbox_head['fc_out_channels'])).cuda() for _ in range(bbox_head['num_classes'])]
#         # self.prototype_features = torch.zeros((bbox_head['num_classes'], bbox_head['fc_out_channels'])).cuda()
#         # self.prototype_features = torch.zeros((bbox_head['num_classes'], bbox_head['fc_out_channels']), inplace=False).cuda()
#         # self.extractor = [nn.Linear(in_features=bbox_head['fc_out_channels'], out_features=bbox_head['fc_out_channels']).cuda(),
#         #                 nn.Linear(in_features=bbox_head['fc_out_channels'], out_features=bbox_head['fc_out_channels']).cuda(),
#         # ]
#         self.extractor = nn.Linear(in_features=bbox_head['fc_out_channels'], out_features=bbox_head['fc_out_channels'])
#         # self.bn = nn.BatchNorm1d(1024)
#         # self.relu = nn.ReLU(inplace=True)
#         # self.prototype_features = torch.autograd.Variable(torch.zeros((9, 1024)), requires_grad=True).cuda()
#         # self.cat_layer = 
#         # self.prototype_features = torch.zeros(bbox_head['num_classes'] + 1, bbox_head['fc_out_channels']).type(torch.float32).cuda()
#         # self.max_loss_angle = 0.0
#         # self.gate_change_ratio = 8
#         self.gate = False
#         self.iters = 0
#         self.warmup = 3000 #3000
#         self.warmup_2 = 3400 #3400
#         # self.warmup = 0
#         # self.warmup_2 = 50
#         self.level3_before = None
#         self.level3_after = None
#         self.level3_cls = None
#         self.tsne_num = 1
#         self.flag = False
        
#     def load_pretrained_weights(self, pretrained):
#         checkpoint = torch.load(pretrained)
#         weight_fc_angle = checkpoint['state_dict']['roi_head.perception.fc_angle.weight']
#         bias_fc_angle = checkpoint['state_dict']['roi_head.perception.fc_angle.bias']
#         weight_fc_x = checkpoint['state_dict']['roi_head.perception.fc_x.weight']
#         bias_fc_x = checkpoint['state_dict']['roi_head.perception.fc_x.bias']
#         weight_fc_y = checkpoint['state_dict']['roi_head.perception.fc_y.weight']
#         bias_fc_y = checkpoint['state_dict']['roi_head.perception.fc_y.bias']

#         with torch.no_grad():
#             self.perception.fc_angle.weight.copy_(weight_fc_angle)
#             self.perception.fc_angle.bias.copy_(bias_fc_angle)
#             self.perception.fc_x.weight.copy_(weight_fc_x)
#             self.perception.fc_x.bias.copy_(bias_fc_x)
#             self.perception.fc_y.weight.copy_(weight_fc_y)
#             self.perception.fc_y.bias.copy_(bias_fc_y)

#     def forward_dummy(self, x, proposals):
#         """Dummy forward function.

#         Args:
#             x (list[Tensors]): list of multi-level img features.
#             proposals (list[Tensors]): list of region proposals.

#         Returns:
#             list[Tensors]: list of region of interest.
#         """
#         outs = ()
#         rois = rbbox2roi([proposals])
#         if self.with_bbox:
#             bbox_results = self._bbox_forward(x, rois)
#             outs = outs + (bbox_results['cls_score'],
#                            bbox_results['bbox_pred'])
#         return outs
#     def forward_angle(self, x, img_metas, proposal_list, gt_bboxes, gt_labels):
        
#         rois = rbbox2roi([bboxes for bboxes in gt_bboxes])
#         bbox_results = self._bbox_forward(x, rois)
        
#         # loss_domain = make_classify(self.domain_classifier, pos_feats_flatten, pos_simulated)
#         # angle perception
#         # x_pre = self.perception(bbox_results['bbox_feats'])
#         # angle_pre_all = self.perception(bbox_results['bbox_feats_flatten'])
#         x_pre, y_pre = self.perception(bbox_results['bbox_feats_flatten'])
#         # test = x_pre[0,:]
#         # _ , pre_max = torch.max(angle_pre_all, dim=-1)
#         _ , x_pre_max = torch.max(x_pre, dim=-1)
#         _ , y_pre_max = torch.max(y_pre, dim=-1)
#         # _ , y_pre_max  = torch.max(y_pre, dim=-1)
#         # bbox head forward and loss

#         # return pre_max
#         return x_pre_max, y_pre_max

#     # origin
#     def forward_train(self,
#                       x,
#                       img_metas,
#                       proposal_list,
#                       gt_bboxes,
#                       gt_labels,
#                     #   simulated,
#                     #   angles,
#                       sim_bboxes=None,
#                       sim_angles=None,
#                       gt_bboxes_ignore=None,
#                       gt_masks=None):
#         """
#         Args:
#             x (list[Tensor]): list of multi-level img features.
#             img_metas (list[dict]): list of image info dict where each dict
#                 has: 'img_shape', 'scale_factor', 'flip', and may also contain
#                 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                 For details on the values of these keys see
#                 `mmdet/datasets/pipelines/formatting.py:Collect`.
#             proposals (list[Tensors]): list of region proposals.
#             gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                 shape (num_gts, 5) in [cx, cy, w, h, a] format.
#             gt_labels (list[Tensor]): class indices corresponding to each box
#             gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#                 boxes can be ignored when computing the loss.
#             gt_masks (None | Tensor) : true segmentation masks for each box
#                 used if the architecture supports a segmentation task. Always
#                 set to None.

#         Returns:
#             dict[str, Tensor]: a dictionary of loss components
#         """
#         # assign gts and sample proposals
#         if self.with_bbox:
#             num_imgs = len(img_metas)
#             if gt_bboxes_ignore is None:
#                 gt_bboxes_ignore = [None for _ in range(num_imgs)]
#             sampling_results = []
#             # sampling_results_sim = []
#             for i in range(num_imgs):
#                 gt_hbboxes = obb2xyxy(gt_bboxes[i], self.version)
#                 assign_result = self.bbox_assigner.assign(
#                     proposal_list[i], gt_hbboxes, gt_bboxes_ignore[i],
#                     gt_labels[i])
#                 sampling_result = self.bbox_sampler.sample(
#                     assign_result,
#                     proposal_list[i],
#                     gt_hbboxes,
#                     gt_labels[i],
#                     feats=[lvl_feat[i][None] for lvl_feat in x])
#                 # assign_result = self.bbox_assigner.assign(
#                 #     proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
#                 #     gt_labels[i])
#                 # sampling_result = self.bbox_sampler.sample(
#                 #     assign_result,
#                 #     proposal_list[i],
#                 #     gt_bboxes[i],
#                 #     gt_labels[i],
#                 #     feats=[lvl_feat[i][None] for lvl_feat in x])

#                 if gt_bboxes[i].numel() == 0:
#                     sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
#                         (0, gt_bboxes[0].size(-1))).zero_()
#                 else:
#                     sampling_result.pos_gt_bboxes = \
#                         gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

#                 sampling_results.append(sampling_result)


#         losses = dict()
#         # bbox head forward and loss
#         if self.with_bbox:
#             bbox_results = self._bbox_forward_train(x, sampling_results,
#                                                     gt_bboxes, gt_labels, 
#                                                     # simulated, angles, 
#                                                     sim_bboxes, sim_angles,
#                                                     img_metas)
#             losses.update(bbox_results['loss_bbox'])

#         return losses

#     # def forward_train(self,
#     #                   x,
#     #                   img_metas,
#     #                   proposal_list,
#     #                   x_sim,
#     #                   proposal_list_sim,
#     #                   real_gt_bboxes,
#     #                   real_gt_labels,
#     #                   sim_gt_bboxes,
#     #                   sim_gt_labels,
#     #                   gt_bboxes,
#     #                   gt_labels,
#     #                   simulated,
#     #                   angles,
#     #                   gt_bboxes_ignore=None,
#     #                   gt_masks=None):
#     #     """
#     #     Args:
#     #         x (list[Tensor]): list of multi-level img features.
#     #         img_metas (list[dict]): list of image info dict where each dict
#     #             has: 'img_shape', 'scale_factor', 'flip', and may also contain
#     #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#     #             For details on the values of these keys see
#     #             `mmdet/datasets/pipelines/formatting.py:Collect`.
#     #         proposals (list[Tensors]): list of region proposals.
#     #         gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#     #             shape (num_gts, 5) in [cx, cy, w, h, a] format.
#     #         gt_labels (list[Tensor]): class indices corresponding to each box
#     #         gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#     #             boxes can be ignored when computing the loss.
#     #         gt_masks (None | Tensor) : true segmentation masks for each box
#     #             used if the architecture supports a segmentation task. Always
#     #             set to None.

#     #     Returns:
#     #         dict[str, Tensor]: a dictionary of loss components
#     #     """
#     #     # assign gts and sample proposals
#     #     # num_imgs = len(img_metas)
#     #     # for i in range(num_imgs):
#     #     #     reserve = simulated[i] > 0
#     #     #     idx = torch.where(reserve == True)[0]
#     #     #     l = len(idx)
#     #         # if l == 0:
#     #         #     print(img_metas[i])

#     #     if self.with_bbox:
#     #         num_imgs = len(img_metas)
#     #         if gt_bboxes_ignore is None:
#     #             gt_bboxes_ignore = [None for _ in range(num_imgs)]
#     #         # gt_bboxes_ignore_real = sim_gt_bboxes
#     #         # gt_bboxes_ignore_sim = real_gt_bboxes
#     #         sampling_results = []
#     #         sampling_results_sim = []
#     #         # sampling_results_sim = []
#     #         for i in range(num_imgs):
#     #             # real
#     #             assign_result = self.bbox_assigner.assign(
#     #                 proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
#     #                 gt_labels[i])
#     #             sampling_result = self.bbox_sampler.sample(
#     #                 assign_result,
#     #                 proposal_list[i],
#     #                 gt_bboxes[i],
#     #                 gt_labels[i],
#     #                 simulated[i],
#     #                 feats=[lvl_feat[i][None] for lvl_feat in x])

#     #             if gt_bboxes[i].numel() == 0:
#     #                 sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
#     #                     (0, gt_bboxes[0].size(-1))).zero_()
#     #             else:
#     #                 sampling_result.pos_gt_bboxes = \
#     #                     gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]
#     #             sampling_results.append(sampling_result)

#     #             assign_result_sim = self.bbox_assigner.assign(
#     #                 proposal_list_sim[i], gt_bboxes[i], gt_bboxes_ignore[i],
#     #                 gt_labels[i])
#     #             sampling_result_sim = self.bbox_sampler.sample(
#     #                 assign_result_sim,
#     #                 proposal_list_sim[i],
#     #                 gt_bboxes[i],
#     #                 gt_labels[i],
#     #                 1 - simulated[i],
#     #                 feats=[lvl_feat[i][None] for lvl_feat in x_sim])

#     #             if gt_bboxes[i].numel() == 0:
#     #                 sampling_result_sim.pos_gt_bboxes = gt_bboxes[i].new(
#     #                     (0, gt_bboxes[0].size(-1))).zero_()
#     #             else:
#     #                 sampling_result_sim.pos_gt_bboxes = \
#     #                     gt_bboxes[i][sampling_result_sim.pos_assigned_gt_inds, :]

#     #             sampling_results_sim.append(sampling_result_sim)


#     #     losses = dict()
#     #     # bbox head forward and loss
#     #     if self.with_bbox:
#     #         # origin
#     #         # bbox_results = self._bbox_forward_train(x, sampling_results,
#     #         #                                         gt_bboxes, gt_labels, simulated, angles, 
#     #         #                                         img_metas)
#     #         bbox_results = self._bbox_forward_train(x, sampling_results, real_gt_bboxes, real_gt_labels, 
#     #                                                 x_sim, sampling_results_sim, sim_gt_bboxes, sim_gt_labels, gt_bboxes, gt_labels,
#     #                                                 simulated, angles, 
#     #                                                 img_metas)

#     #         losses.update(bbox_results['loss_bbox'])

#     #     return losses

#     # def _bbox_forward_train(self, x, sampling_results, real_gt_bboxes, real_gt_labels, 
#     #                                                 x_sim, sampling_results_sim, sim_gt_bboxes, sim_gt_labels, 
#     #                                                 gt_bboxes, gt_labels, simulated, angles,  
#     #                         img_metas):

#     #     rois = rbbox2roi([res.bboxes for res in sampling_results])
#     #     bbox_results = self._bbox_forward(x, rois)
#     #     rois_sim = rbbox2roi([res.bboxes for res in sampling_results_sim])
#     #     bbox_results_sim = self._bbox_forward(x_sim, rois_sim)
        
#     #     # # domain classify
#     #     pos_feats, pos_feats_flatten, pos_scores, pos_angles_pred, pos_simulated, pos_angles, pos_gts = get_prepared(
#     #     # pos_feats_flatten, pos_simulated, pos_angles = get_prepared(
#     #                                             bbox_results['bbox_feats'], 
#     #                                             bbox_results['bbox_feats_flatten'], bbox_results['cls_score'], bbox_results['angle_pred'],
#     #                                             sampling_results, simulated, angles
#     #                                         )
#     #     pos_feats, pos_feats_flatten_sim, pos_scores, pos_angles_pred, pos_simulated_sim, pos_angles, pos_gts_sim = get_prepared(
#     #     # pos_feats_flatten, pos_simulated, pos_angles = get_prepared(
#     #                                             bbox_results_sim['bbox_feats'], 
#     #                                             bbox_results_sim['bbox_feats_flatten'], bbox_results_sim['cls_score'], bbox_results_sim['angle_pred'],
#     #                                             sampling_results_sim, simulated, angles
#     #                                         )
#     #     a = torch.zeros_like(pos_simulated)
#     #     b = torch.ones_like(pos_simulated_sim)
#     #     pos_simulated_all = torch.cat([a, b], dim=0)
#     #     pos_feats_flatten_all = torch.cat([pos_feats_flatten, pos_feats_flatten_sim], dim=0)
#     #     pos_gts_all = torch.cat([pos_gts, pos_gts_sim], dim=0)

#     #     # unfreeze_params(self.discriminator)
#     #     # self.optimizor_d.zero_grad()
#     #     # loss_d = loss_D(self.discriminator, pos_feats_flatten_all.detach(), pos_simulated_all.detach(), pos_gts_all.detach())
#     #     # loss_d.backward()
#     #     # self.optimizor_d.step()
#     #     # freeze_params(self.discriminator)

#     #     # loss_g = loss_G(self.discriminator, pos_feats_flatten_sim, pos_simulated_sim, pos_gts_sim)
            
#     #     bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
#     #                                               gt_labels, self.train_cfg)
#     #     bbox_targets_sim = self.bbox_head.get_targets(sampling_results_sim, gt_bboxes,
#     #                                               gt_labels, self.train_cfg)

#     #     loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
#     #                                     bbox_results['bbox_pred'], rois,
#     #                                     *bbox_targets)
#     #     loss_bbox_sim = self.bbox_head.loss(bbox_results_sim['cls_score'],
#     #                                     bbox_results_sim['bbox_pred'], rois_sim,
#     #                                     *bbox_targets_sim)
#     #     loss_bbox['loss_cls_sim'] = loss_bbox_sim['loss_cls']
#     #     loss_bbox['loss_bbox_sim'] = loss_bbox_sim['loss_bbox']
#     #     # loss_bbox['loss_g'] = loss_g
#     #     bbox_results.update(loss_bbox=loss_bbox)
#     #     return bbox_results

#     def check_level3(self, x_feat, x_angle, y_angle, x_feat_sim, cls):
#         self.flag = True
#         x_feat_before = x_feat.clone()
#         x_feat_after = x_feat_sim.clone()
#         cls_clone = cls.clone()

#         _, levelx = torch.max(x_angle, dim=-1)
#         _, levely = torch.max(y_angle, dim=-1)
#         angle_max = torch.max(levelx, levely)

#         l3 = angle_max == 2
#         idx_l3 = torch.nonzero(l3).squeeze(dim=-1)

#         if self.level3_before == None:
#             self.level3_before = torch.index_select(x_feat_before, 0, idx_l3)
#         else:
#             temp = torch.index_select(x_feat_before, 0, idx_l3)
#             self.level3_before = torch.cat((self.level3_before, temp), dim=0)
#         if self.level3_after == None:
#             self.level3_after = torch.index_select(x_feat_after, 0, idx_l3)
#         else:
#             temp = torch.index_select(x_feat_after, 0, idx_l3)
#             self.level3_after = torch.cat((self.level3_after, temp), dim=0)
#         if self.level3_cls == None:
#             self.level3_cls = torch.index_select(cls_clone, 0, idx_l3)
#         else:
#             temp = torch.index_select(cls_clone, 0, idx_l3)
#             self.level3_cls = torch.cat((self.level3_cls, temp), dim=0)
#         return 

#     # origin
#     def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, 
#                             # simulated, angles, 
#                             sim_boxes, sim_angles,
#                             img_metas):

#         rois = rbbox2roi([res.bboxes for res in sampling_results])
#         bbox_results = self._bbox_forward(x, rois)
        
#         if sim_boxes != None:
#             sim_rois = rbbox2roi(sim_boxes)
#             sim_bbox_results = self._bbox_forward(x, sim_rois)
        
#         # # domain classify
#         # pos_feats, pos_feats_flatten, pos_scores, pos_angles_pred, pos_simulated, pos_angles, pos_gts = get_prepared(
#         pos_feats, pos_feats_flatten, pos_scores, pos_angles_pred, pos_gts = get_prepared(
#         # pos_feats_flatten, pos_simulated, pos_angles = get_prepared(
#                                                 bbox_results['bbox_feats'], 
#                                                 bbox_results['bbox_feats_flatten'], bbox_results['cls_score'], bbox_results['angle_pred'],
#                                                 sampling_results, 
#                                                 # simulated, angles
#                                             )
#         # loss_domain = make_classify(self.domain_classifier, pos_feats_flatten, pos_simulated)
#         # angle perception
#         # x_pre, y_pre = self.perception(pos_feats)
#         x_pre, y_pre = self.perception(pos_feats_flatten)
#         # angle_pre = self.perception(pos_feats_flatten)
        
#         if sim_boxes == None:
#             # loss_angle = self.perception.loss_angle(x_pre, y_pre, pos_simulated, pos_angles, pos_gts, pos_scores)
#             pass
#         else:
#             ### sim只用作角度分类训练，不参与到整个训练中
#             x_pre_sim, y_pre_sim = self.perception(sim_bbox_results['bbox_feats_flatten'])
#             # loss_angle = self.perception.loss_angle_onlysim(x_pre_sim, y_pre_sim, torch.cat(sim_angles))
#         # self.update_prototype_feature(pos_feats_flatten, pos_scores, angle_pre, pos_gts, thresh=0.6)
#         self.iters += 1
#         # if self.iters > 1:
#         if self.iters > self.warmup:
#             prototype_features = self.update_prototype_feature(pos_feats_flatten, pos_scores, x_pre, y_pre, pos_gts, thresh=0.3)

#         # if self.iters > 1:
#         if self.iters > self.warmup_2:
#             # angle_pre_all = self.perception(bbox_results['bbox_feats_flatten'])
#             # feats_sim_all = self.simulator(bbox_results['bbox_feats_flatten'], angle_pre_all)
            
#     #         feats_sim_all_temp = bbox_results['bbox_feats_flatten']
#     #         for t in range(5):
#     #             x_pre_all, y_pre_all = self.perception(feats_sim_all_temp)
#     #             feats_sim_all_temp = self.simulator(feats_sim_all_temp, x_pre_all, y_pre_all)
#     #             x_pre_all_aftersim, y_pre_all_aftersim = self.perception(feats_sim_all_temp)
#     #             if torch.all(x_pre_all_aftersim[:, 0] == torch.max(x_pre_all_aftersim, dim=1)[0]) and \
#     #    torch.all(y_pre_all_aftersim[:, 0] == torch.max(y_pre_all_aftersim, dim=1)[0]):
#     #                 break
#     #         x_pre_all, y_pre_all = self.perception(feats_sim_all_temp)
#     #         feats_sim_all = feats_sim_all_temp
            
#             x_pre_all, y_pre_all = self.perception(bbox_results['bbox_feats_flatten'])
#             feats_sim_all = self.simulator(bbox_results['bbox_feats_flatten'], x_pre_all, y_pre_all)
#             # self.check_level3(bbox_results['bbox_feats_flatten'], x_pre_all, y_pre_all, feats_sim_all, bbox_results['cls_score'])
#             # self.draw_tsne()
#             # print(feats_sim)
#             # bbox_simi = cal_simi(feats_sim_all, prototype_features)
#             # # print(bbox_simi)
#             # bbox_results['cls_score'] = bbox_simi.softmax(dim=1)
#             # # print(bbox_results['cls_score'])
            
#             ### simulator后的特征需要被角度分类器分成1类
#             x_pre_all_aftersim, y_pre_all_aftersim = self.perception(feats_sim_all)
#             angles_aftersim = torch.zeros(x_pre_all_aftersim.shape).cuda()
#             loss_angle_aftersim = self.perception.loss_angle_onlysim(x_pre_all_aftersim, y_pre_all_aftersim, angles_aftersim)
            
            
#             new_cls_scores = self.bbox_head.forward_cls(feats_sim_all)
#             new_reg = self.bbox_head.forward_reg(feats_sim_all)
#             # bbox_results['cls_score'] = new_cls_scores
            
        
#         bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
#                                                   gt_labels, self.train_cfg)

#         loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
#                                         bbox_results['bbox_pred'], rois,
#                                         *bbox_targets)
        
        
#         if self.iters > self.warmup_2:
#             # feats_sim = self.simulator(pos_feats_flatten, angle_pre)
#             feats_sim = self.simulator(pos_feats_flatten, x_pre, y_pre)
#             loss_simulate = self.simulator.loss_simulate(feats_sim, pos_gts, prototype_features)
#             loss_bbox['loss_simulate'] = loss_simulate 

#             new_loss_bbox = self.bbox_head.loss(new_cls_scores,
#                                         new_reg, rois,
#                                         *bbox_targets)
            
#             loss_bbox['new_loss_cls'] = new_loss_bbox['loss_cls']
#             # loss_bbox['new_loss_bbox'] = new_loss_bbox['loss_bbox']
#             # loss_con = self.contrastive_loss(prototype_features)
#             # loss_bbox['loss_con'] = loss_con
            
#             full_cls_score = self.bbox_head.forward_cls(prototype_features)
#             full_labels = torch.arange(0, prototype_features.shape[0], step=1).cuda()
#             label_weights = torch.ones((prototype_features.shape[0]), dtype=torch.float).cuda()
#             avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
#             loss_cls_ = self.bbox_head.loss_cls(
#                     full_cls_score,
#                     full_labels,
#                     label_weights,
#                     avg_factor=avg_factor)
#             # loss_bbox['loss_full'] = loss_cls_

#             # for c in range(len(self.prototype_features)):
#             #     self.prototype_features[c].retain_grad()
#             # prototype_features.retain_grad()
            
#             loss_bbox['loss_angle_aftersim'] = loss_angle_aftersim
        
#         # loss_bbox['loss_angle'] = loss_angle
        
        
#         bbox_results.update(loss_bbox=loss_bbox)
#         return bbox_results

#     def _bbox_forward(self, x, rois):
#         """Box head forward function used in both training and testing.

#         Args:
#             x (list[Tensor]): list of multi-level img features.
#             rois (list[Tensors]): list of region of interests.

#         Returns:
#             dict[str, Tensor]: a dictionary of bbox_results.
#         """
#         bbox_feats = self.bbox_roi_extractor(
#             x[:self.bbox_roi_extractor.num_inputs], rois)
#         if self.with_shared_head:
#             bbox_feats = self.shared_head(bbox_feats)
#         cls_score, bbox_pred, bbox_feats_flatten, angle_pred = self.bbox_head(bbox_feats)

#         bbox_results = dict(
#             cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, 
#             bbox_feats_flatten=bbox_feats_flatten, angle_pred=angle_pred
#             )
#         return bbox_results

#     def _bbox_forward_sim(self, x_sim, rois):
#         """Box head forward function used in both training and testing.

#         Args:
#             x (list[Tensor]): list of multi-level img features.
#             rois (list[Tensors]): list of region of interests.

#         Returns:
#             dict[str, Tensor]: a dictionary of bbox_results.
#         """
#         bbox_feats = self.bbox_roi_extractor(
#             x_sim[:self.bbox_roi_extractor.num_inputs], rois)
#         if self.with_shared_head:
#             bbox_feats = self.shared_head(bbox_feats)
#         cls_score, bbox_pred, bbox_feats_flatten, angle_pred = self.bbox_head_sim(bbox_feats)

#         bbox_results = dict(
#             cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, 
#             bbox_feats_flatten=bbox_feats_flatten, angle_pred=angle_pred
#             )
#         return bbox_results

#     # def simple_test_bboxes(self,
#     #                        x,
#     #                        img_metas,
#     #                        proposals,
#     #                        rcnn_test_cfg,
#     #                        rescale=False):
#     #     """Test only det bboxes without augmentation.

#     #     Args:
#     #         x (tuple[Tensor]): Feature maps of all scale level.
#     #         img_metas (list[dict]): Image meta info.
#     #         proposals (List[Tensor]): Region proposals.
#     #         rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
#     #         rescale (bool): If True, return boxes in original image space.
#     #             Default: False.

#     #     Returns:
#     #         tuple[list[Tensor], list[Tensor]]: The first list contains \
#     #             the boxes of the corresponding image in a batch, each \
#     #             tensor has the shape (num_boxes, 5) and last dimension \
#     #             5 represent (cx, cy, w, h, a, score). Each Tensor \
#     #             in the second list is the labels with shape (num_boxes, ). \
#     #             The length of both lists should be equal to batch_size.
#     #     """

#     #     rois = rbbox2roi(proposals)
#     #     bbox_results = self._bbox_forward(x, rois)
#     #     img_shapes = tuple(meta['img_shape'] for meta in img_metas)
#     #     scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

#     #     # if self.prototype_features != None:
#     #     #     angles_pre_all = self.perception(bbox_results['bbox_feats_flatten'])
#     #     #     feats_sim = self.simulator(self.prototype_features, angles_pre_all)
#     #     #     bbox_simi = cal_simi(bbox_results['bbox_feats_flatten'], feats_sim)
#     #     #     bbox_results['cls_score'] = bbox_simi.softmax(dim=1)
#     #     # split batch bbox prediction back to each image

#     #     # if self.iters > self.warmup_2:
#     #     # #     angle_pre_all = self.perception(bbox_results['bbox_feats_flatten'])
#     #     # #     feats_sim_all = self.simulator(bbox_results['bbox_feats_flatten'], angle_pre_all)
#     #     #     print('oh yes')
#     #     #     x_pre_all, y_pre_all = self.perception(bbox_results['bbox_feats_flatten'])
#     #     #     feats_sim_all = self.simulator(bbox_results['bbox_feats_flatten'], x_pre_all, y_pre_all)
#     #     #     cls_scores = self.bbox_head.forward_cls(feats_sim_all)
#     #     #     bbox_results['cls_score'] = (cls_scores + bbox_results['cls_score'])/2
            
#     #     cls_score = bbox_results['cls_score']
#     #     bbox_pred = bbox_results['bbox_pred']
#     #     num_proposals_per_img = tuple(len(p) for p in proposals)
#     #     rois = rois.split(num_proposals_per_img, 0)
#     #     cls_score = cls_score.split(num_proposals_per_img, 0)

#     #     # some detector with_reg is False, bbox_pred will be None
#     #     if bbox_pred is not None:
#     #         # the bbox prediction of some detectors like SABL is not Tensor
#     #         if isinstance(bbox_pred, torch.Tensor):
#     #             bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
#     #         else:
#     #             bbox_pred = self.bbox_head.bbox_pred_split(
#     #                 bbox_pred, num_proposals_per_img)
#     #     else:
#     #         bbox_pred = (None, ) * len(proposals)

#     #     # apply bbox post-processing to each image individually
#     #     det_bboxes = []
#     #     det_labels = []
#     #     for i in range(len(proposals)):
#     #         det_bbox, det_label = self.bbox_head.get_bboxes(
#     #             rois[i],
#     #             cls_score[i],
#     #             bbox_pred[i],
#     #             img_shapes[i],
#     #             scale_factors[i],
#     #             rescale=rescale,
#     #             cfg=rcnn_test_cfg)
#     #         det_bboxes.append(det_bbox)
#     #         det_labels.append(det_label)
#     #     return det_bboxes, det_labels

#     def visual(self, feat, perplexity):
#         ts = TSNE(n_components=2, init='pca', random_state=32, perplexity=perplexity)
#         x_ts = ts.fit_transform(feat)
#         x_min, x_max = x_ts.min(0), x_ts.max(0)
#         x_final = (x_ts - x_min) / (x_max - x_min)
#         return x_final

#     def plotlabels(self, S_lowDWeights, Trure_labels, marker_index, t):
#         maker = ['o', '*', 's', '^', 's', 'p', '<', '>', 'D', 'd', 'h', 'H']
#         colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'violet', 'peru', 'olivedrab',
#           'hotpink']
#         True_labels = Trure_labels.reshape((-1, 1))
#         S_data = np.hstack((S_lowDWeights, True_labels))  
#         S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

#         X = S_data.loc[S_data['label'] == t]['x']
#         Y = S_data.loc[S_data['label'] == t]['y']
#         plt.scatter(X, Y, cmap='brg', s=100, marker=maker[marker_index], c=colors[t], edgecolors=colors[t], alpha=0.65)
#         plt.xticks([])  
#         plt.yticks([])  
        
#     def append_output_to_file(self, output_text, file_path):
#         try:
#             with open(file_path, 'a') as file:
#                 file.write(output_text)
#         except Exception as e:
#             print("写入文件时出错:", str(e))

    
#     def draw_tsne(self):
#         self.flag = False
#         file_path = '/home/lkw/log/test.txt'
#         if self.level3_cls == None or self.level3_before == None or self.level3_after == None:
#             print(f'no enough data 1 at {self.tsne_num}\n')
#             output_text = f'no enough data 1 at {self.tsne_num}\n'
#             self.append_output_to_file(output_text, file_path)
#             return
#         num_classes = self.level3_cls.shape[1] - 1
#         p = self.level3_cls.shape[0]
#         if p <= 1:
#             print(f'no enough data level3 at {self.tsne_num}\n')
#             output_text = f'no enough data level3 at {self.tsne_num}\n'
#             self.append_output_to_file(output_text, file_path)
#             return 
#         if p < 10:
#             p = p - 1
#         elif p < 50:
#             p = p - 10
#         else:
#             p = 50
#         prototype_features = []
#         for c in range(0, num_classes):
#             prototype_feature = self.extractor(self.prototype_features[c].clone())
#             prototype_features.append(prototype_feature)
#         prototype_features = torch.stack(prototype_features, dim=0)
#         labels_prototype_features = np.arange(10)
#         level_cls_temp = self.level3_cls[:, :-1].cpu().detach()
#         labels_level3_features = np.argmax(level_cls_temp, axis=1)

#         prototype_features_temp = prototype_features.cpu().detach()
#         level3_before_temp = self.level3_before.cpu().detach()
#         level3_after_temp = self.level3_after.cpu().detach()
#         labels_level3_features_temp = labels_level3_features
        
#         prototype_features_tsne = self.visual(prototype_features_temp, 5)
#         level3_before_tsne = self.visual(level3_before_temp, p)
#         level3_after_tsne = self.visual(level3_after_temp, p)
#         for t in range(10):
#             fig = plt.figure(figsize=(15, 15))
#             self.plotlabels(prototype_features_tsne, labels_prototype_features, 0, t)
#             self.plotlabels(level3_before_tsne, labels_level3_features_temp, 1, t)
#             self.plotlabels(level3_after_tsne, labels_level3_features_temp, 2, t)
#             save_path = f'/new/datasets/tsne/{t}/plot_epoch{self.tsne_num}.png'
#             plt.grid(False)
#             plt.savefig(save_path)
#             plt.clf()

#         self.level3_cls = None
#         self.level3_before = None
#         self.level3_after = None
#         self.tsne_num += 1
#         print(self.tsne_num)

#         # plt.legend()
#         # plt.grid(False)
#         # plt.savefig(save_path)
#         # plt.show()



#     def simple_test_bboxes(self,
#                            x,
#                            img_metas,
#                            proposals,
#                            rcnn_test_cfg,
#                            rescale=False):
#         """Test only det bboxes without augmentation.

#         Args:
#             x (tuple[Tensor]): Feature maps of all scale level.
#             img_metas (list[dict]): Image meta info.
#             proposals (List[Tensor]): Region proposals.
#             rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
#             rescale (bool): If True, return boxes in original image space.
#                 Default: False.

#         Returns:
#             tuple[list[Tensor], list[Tensor]]: The first list contains \
#                 the boxes of the corresponding image in a batch, each \
#                 tensor has the shape (num_boxes, 5) and last dimension \
#                 5 represent (cx, cy, w, h, a, score). Each Tensor \
#                 in the second list is the labels with shape (num_boxes, ). \
#                 The length of both lists should be equal to batch_size.
#         """

#         rois = rbbox2roi(proposals)
#         bbox_results = self._bbox_forward(x, rois)
#         img_shapes = tuple(meta['img_shape'] for meta in img_metas)
#         scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

#         # self.draw_tsne()

#         # if self.prototype_features != None:
#         #     angles_pre_all = self.perception(bbox_results['bbox_feats_flatten'])
#         #     feats_sim = self.simulator(self.prototype_features, angles_pre_all)
#         #     bbox_simi = cal_simi(bbox_results['bbox_feats_flatten'], feats_sim)
#         #     bbox_results['cls_score'] = bbox_simi.softmax(dim=1)
#         # split batch bbox prediction back to each image

#         if self.iters > self.warmup_2:
#         # if True:
#         #     angle_pre_all = self.perception(bbox_results['bbox_feats_flatten'])
#         #     feats_sim_all = self.simulator(bbox_results['bbox_feats_flatten'], angle_pre_all)
#             x_pre_all, y_pre_all = self.perception(bbox_results['bbox_feats_flatten'])
#             feats_sim_all = self.simulator(bbox_results['bbox_feats_flatten'], x_pre_all, y_pre_all)
#             cls_scores = self.bbox_head.forward_cls(feats_sim_all)
#             bbox_results['cls_score'] = (cls_scores + bbox_results['cls_score'])/2
            
#         cls_score = bbox_results['cls_score']
#         bbox_pred = bbox_results['bbox_pred']
#         num_proposals_per_img = tuple(len(p) for p in proposals)
#         rois = rois.split(num_proposals_per_img, 0)
#         cls_score = cls_score.split(num_proposals_per_img, 0)

#         # some detector with_reg is False, bbox_pred will be None
#         if bbox_pred is not None:
#             # the bbox prediction of some detectors like SABL is not Tensor
#             if isinstance(bbox_pred, torch.Tensor):
#                 bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
#             else:
#                 bbox_pred = self.bbox_head.bbox_pred_split(
#                     bbox_pred, num_proposals_per_img)
#         else:
#             bbox_pred = (None, ) * len(proposals)

#         # apply bbox post-processing to each image individually
#         det_bboxes = []
#         det_labels = []
#         for i in range(len(proposals)):
#             det_bbox, det_label = self.bbox_head.get_bboxes(
#                 rois[i],
#                 cls_score[i],
#                 bbox_pred[i],
#                 img_shapes[i],
#                 scale_factors[i],
#                 rescale=rescale,
#                 cfg=rcnn_test_cfg)
#             det_bboxes.append(det_bbox)
#             det_labels.append(det_label)
#         return det_bboxes, det_labels

#     def simple_test(self, x, proposal_list, img_metas, rescale=False):
#         """Test without augmentation.

#         Args:
#             x (list[Tensor]): list of multi-level img features.
#             proposal_list (list[Tensors]): list of region proposals.
#             img_metas (list[dict]): list of image info dict where each dict
#                 has: 'img_shape', 'scale_factor', 'flip', and may also contain
#                 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#             rescale (bool): If True, return boxes in original image space.
#                 Default: False.

#         Returns:
#             dict[str, Tensor]: a dictionary of bbox_results.
#         """
#         assert self.with_bbox, 'Bbox head must be implemented.'

#         det_bboxes, det_labels = self.simple_test_bboxes(
#             x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        

#         # det_bboxes_sim, det_labels_sim = self.simple_test_bboxes(
#         #     x_sim, img_metas, proposal_list_sim, self.test_cfg, rescale=rescale)

#         # det_bboxes = [torch.cat([det_bboxes[i], det_bboxes_sim[i]]) for i in range(len(det_bboxes))]
#         # det_labels = [torch.cat([det_labels[i], det_labels_sim[i]]) for i in range(len(det_labels))]
#         bbox_results = [
#             rbbox2result(det_bboxes[i], det_labels[i],
#                          self.bbox_head.num_classes)
#             for i in range(len(det_bboxes))
#         ]
#         # if self.flag:
#         #     self.draw_tsne()

#         return bbox_results


#     # def update_prototype_feature(self, feats_ori, scores_ori, angles, gt_labels, thresh=0.7, epsilon=1e-6):
#     def update_prototype_feature(self, feats_ori, scores_ori, x_angle, y_angle, gt_labels, thresh=0.7, epsilon=1e-6):
#         # x_angle = angles[:, 0]
#         # y_angle = angles[:, 1]
#         # x = x_angle < thresh
#         # y = y_angle < thresh
#         # satisfied = torch.logical_and(x, y)
        
#         _, levelx = torch.max(x_angle, dim=-1)
#         _, levely = torch.max(y_angle, dim=-1)
#         angle_max = torch.max(levelx, levely)
#         # satisfied = angle_max < 3
#         satisfied = angle_max < 1
#         num_classes = scores_ori.shape[1] - 1

#         # _, level = torch.max(angles, dim=-1)
#         # satisfied = level == 0
#         # print('satisfied:', satisfied)
#         # satisfied = torch.tensor([0, 0, 1], dtype=torch.bool)
#         idx = torch.nonzero(satisfied).squeeze(dim=-1)
#         if idx.shape[0] == 0:
#             prototype_features = []
#             for c in range(0, num_classes):
#                 # tmp_feat = self.prototype_features[c].clone()
#                 # for l in self.extractor:
#                 #     tmp_feat = l(tmp_feat)
#                 prototype_feature = self.prototype_features[c].clone()
#                 # prototype_feature = self.relu(self.bn(prototype_feature))
#                 prototype_features.append(prototype_feature)
#             prototype_features = torch.stack(prototype_features, dim=0)
#             return prototype_features
#         feats = torch.index_select(feats_ori, 0, idx)
#         labels = torch.index_select(gt_labels, 0, idx)
#         scores = torch.index_select(scores_ori, 0, idx)
        
#         scores = F.softmax(scores, dim=-1)
#         all_class_feat = list()
#         for i in range(num_classes):
#             idx = torch.nonzero(labels == i).squeeze(dim=-1)
#             if idx.shape[0] != 0:
#                 feats_cls = torch.index_select(feats, 0, idx)
#                 feat_cls = torch.sum(feats_cls, dim=0)
#                 avg_feat_cls = feat_cls / idx.shape[0]
#             else:
#                 avg_feat_cls = torch.zeros_like(feats[0,:])
#             # cls_prob = scores[:, i].view(scores.size(0), 1)
#             # #pdb.set_trace()
#             # tmp_class_feat = feats * cls_prob
#             # class_feat = torch.sum(tmp_class_feat, dim=0) / (torch.sum(cls_prob) + epsilon)
#             all_class_feat.append(avg_feat_cls)
#         all_class_feat = torch.stack(all_class_feat, dim = 0)

#         # if self.prototype_features == None:
#         #     self.prototype_features = all_class_feat.detach().clone()
#             # self.prototype_features.requires_grad_()
            
#         for c in range(0, num_classes):
#             if torch.equal(self.prototype_features[c], torch.zeros_like(self.prototype_features[c])):
#                 # self.prototype_features[c] = all_class_feat[c].detach().clone()
#                 self.prototype_features[c] = all_class_feat[c].detach()
#                 continue
#             if torch.equal(all_class_feat[c], torch.zeros_like(all_class_feat[c])):
#                 continue
#             alpha = F.cosine_similarity(self.prototype_features[c], all_class_feat[c], dim=0).item()
#             alpha = 0.2
#             feature_updated = (1.0 - alpha) * self.prototype_features[c].detach() + alpha * all_class_feat[c]
#             # feature_updated = alpha * self.prototype_features[c] + (1 - alpha) * all_class_feat[c]
#             self.prototype_features[c] = feature_updated.detach()
        
#         # prototype_features = self.relu(self.bn(prototype_features))
#         prototype_features = self.get_prototype_features()
#         # matrix = torch.zeros((num_classes, num_classes))
#         # for a in range(0, num_classes):
#         #     for b in range(0, num_classes):
#         #         alpha = F.cosine_similarity(self.prototype_features[a], self.prototype_features[b], dim=0)
#         #         matrix[a,b] = alpha
#         # print(matrix)
#         return prototype_features

#     def get_prototype_features(self):
#         prototype_features = []
#         for c in range(0, len(self.prototype_features)):
#             # tmp_feat = self.prototype_features[c].clone()
#             # for l in self.extractor:
#             #     tmp_feat = l(tmp_feat)
#             # prototype_feature = self.prototype_features[c].clone()
#             prototype_feature = self.extractor(self.prototype_features[c].clone())
#             # prototype_feature = self.relu(self.bn(prototype_feature))
#             prototype_features.append(prototype_feature)
#         prototype_features = torch.stack(prototype_features, dim=0)
#         return prototype_features

#     def contrastive_loss(self, prototype_features):
#         simi = cal_simi(prototype_features, prototype_features)
#         mask = torch.ones_like(simi)
#         mask.fill_diagonal_(0)
#         bool_mask = mask == 1
#         abs_sim_masked = simi[bool_mask]
#         loss = torch.mean(abs_sim_masked)
#         return loss
    
#     def check_full(self):
#         if self.prototype_features == None:
#             return False
#         num_classes = self.prototype_features.shape[0]
#         for c in range(num_classes):
#             if torch.equal(self.prototype_features[c], torch.zeros_like(self.prototype_features[c])):
#                 return False
        
#         return True

        

# def get_prepared(feats, feats_flatten, scores, angles_pred, sampling_results, 
#                 #  simulated, angles
#                 ):
#     batch_size = len(sampling_results)
#     feats_list = feats.chunk(batch_size, dim = 0)
#     feats_flatten_list = feats_flatten.chunk(batch_size, dim = 0)
#     scores_list = scores.chunk(batch_size, dim = 0)
#     angles_pred_list = angles_pred.chunk(batch_size, dim = 0)
#     gt_idxs = [res.pos_assigned_gt_inds for res in sampling_results]
#     pos_cnts = [res.pos_inds.shape[0]  for res in sampling_results]
#     pos_gts = [res.pos_gt_labels for res in sampling_results]
#     pos_feats_list = []
#     pos_feats_flatten_list = []
#     pos_scores_list = []
#     pos_angles_pred_list = []
#     # pos_simulated_list = []
#     # pos_angles_list = []
#     # for i, (feat, feat_flatten, score, angle_pred, sim_label, angle, gt_idx, pos_cnt) in enumerate(
#     #     zip(feats_list, feats_flatten_list, scores_list, angles_pred_list, simulated, angles, gt_idxs, pos_cnts)):
#     for i, (feat, feat_flatten, score, angle_pred, gt_idx, pos_cnt) in enumerate(
#         zip(feats_list, feats_flatten_list, scores_list, angles_pred_list, gt_idxs, pos_cnts)):
#         pos_feats = feat[:pos_cnt, :, :, :]
#         pos_feats_flatten = feat_flatten[:pos_cnt, :]
#         pos_scores = score[:pos_cnt, :]
#         pos_angles_pred = angle_pred[:pos_cnt, :]
#         # pos_simulated = torch.index_select(sim_label, 0, gt_idx)
#         # pos_angles = torch.index_select(angle, 0, gt_idx)
#         pos_feats_list.append(pos_feats)
#         pos_feats_flatten_list.append(pos_feats_flatten)
#         pos_scores_list.append(pos_scores)
#         pos_angles_pred_list.append(pos_angles_pred)
#         # pos_simulated_list.append(pos_simulated)
#         # pos_angles_list.append(pos_angles)
#     pos_feats_all = torch.cat(pos_feats_list, 0)
#     pos_feats_flatten_all = torch.cat(pos_feats_flatten_list, 0)
#     pos_scores_all = torch.cat(pos_scores_list, 0)
#     pos_angles_pred_all = torch.cat(pos_angles_pred_list, 0)
#     # pos_simulated_all = torch.cat(pos_simulated_list, 0)
#     # pos_angles_all = torch.cat(pos_angles_list, 0)
#     pos_gts_all = torch.cat(pos_gts, 0)
#     # return pos_feats_all, pos_feats_flatten_all, pos_scores_all, pos_angles_pred_all, pos_simulated_all, pos_angles_all, pos_gts_all
#     return pos_feats_all, pos_feats_flatten_all, pos_scores_all, pos_angles_pred_all, pos_gts_all
#     # return pos_feats_all, pos_simulated_all, pos_angles_all