# # Copyright (c) OpenMMLab. All rights reserved.
# import torch

# from ..builder import ROTATED_DETECTORS
# from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
# from .two_stage import RotatedTwoStageDetector


# @ROTATED_DETECTORS.register_module()
# class OursRCNN(RotatedTwoStageDetector):
#     """Implementation of `OursRCNN for Object Detection.`__

#     __ https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf  # noqa: E501, E261.
#     """
#     # origin
#     # def __init__(self,
#     #              backbone,
#     #              rpn_head,
#     #              roi_head,
#     #              train_cfg,
#     #              test_cfg,
#     #              neck=None,
#     #              pretrained=None,
#     #              init_cfg=None):
#     #     super(OursRCNN, self).__init__(
#     #         backbone=backbone,
#     #         neck=neck,
#     #         rpn_head=rpn_head,
#     #         roi_head=roi_head,
#     #         train_cfg=train_cfg,
#     #         test_cfg=test_cfg,
#     #         pretrained=pretrained,
#     #         init_cfg=init_cfg)
#     def __init__(self,
#                  backbone,
#                  rpn_head,
#                  roi_head,
#                  train_cfg,
#                  test_cfg,
#                  neck=None,
#                  pretrained=None,
#                  init_cfg=None):
#         super(OursRCNN, self).__init__(
#             backbone=backbone,
#             neck=neck,
#             rpn_head=rpn_head,
#             roi_head=roi_head,
#             train_cfg=train_cfg,
#             test_cfg=test_cfg,
#             pretrained=pretrained,
#             init_cfg=init_cfg)
#         # self.backbone_sim = build_backbone(backbone)
#         # if rpn_head is not None:
#         #     rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
#         #     rpn_head_ = rpn_head.copy()
#         #     rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
#         #     self.rpn_head_sim = build_head(rpn_head_)
        

#     def forward_dummy(self, img):
#         """Used for computing network flops.

#         See `mmrotate/tools/analysis_tools/get_flops.py`
#         """
#         outs = ()
#         # backbone
#         x = self.extract_feat(img)
#         # rpn
#         if self.with_rpn:
#             rpn_outs = self.rpn_head(x)
#             outs = outs + (rpn_outs, )
#         proposals = torch.randn(1000, 6).to(img.device)
#         # roi_head
#         roi_outs = self.roi_head.forward_dummy(x, proposals)
#         outs = outs + (roi_outs, )
#         return outs
#     def forward_angle(self, img, img_metas, gt_bboxes, gt_labels, **kwargs):
#         x = self.extract_feat(img)
#         if self.with_rpn:
#             proposal_cfg = self.train_cfg.get('rpn_proposal',
#                                               self.test_cfg.rpn)
#             _, proposal_list = self.rpn_head.forward_train(
#                 x,
#                 img_metas,
#                 gt_bboxes,
#                 gt_labels=None,
#                 gt_bboxes_ignore=None,
#                 proposal_cfg=proposal_cfg,
#                 **kwargs)
#         else:
#             proposal_list = None

#         # angles_pre = self.roi_head.forward_angle(x, img_metas, proposal_list,
#         #                                          gt_bboxes, gt_labels, 
#         #                                          **kwargs)
#         x_pre, y_pre = self.roi_head.forward_angle(x, img_metas, proposal_list,
#                                                  gt_bboxes, gt_labels, 
#                                                  **kwargs)
#         # return angles_pre
#         return x_pre, y_pre
#     # origin
#     def forward_train(self,
#                       img,
#                       img_metas,
#                       gt_bboxes,
#                       gt_labels,
#                     #   simulated,
#                     #   angles,
#                       gt_bboxes_ignore=None,
#                       gt_masks=None,
#                       proposals=None,
#                       **kwargs):
#         """
#         Args:
#             img (Tensor): of shape (N, C, H, W) encoding input images.
#                 Typically these should be mean centered and std scaled.

#             img_metas (list[dict]): list of image info dict where each dict
#                 has: 'img_shape', 'scale_factor', 'flip', and may also contain
#                 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                 For details on the values of these keys see
#                 `mmdet/datasets/pipelines/formatting.py:Collect`.

#             gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                 shape (num_gts, 5) in [cx, cy, w, h, a] format.

#             gt_labels (list[Tensor]): class indices corresponding to each box

#             gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#                 boxes can be ignored when computing the loss.

#             gt_masks (None | Tensor) : true segmentation masks for each box
#                 used if the architecture supports a segmentation task.

#             proposals : override rpn proposals with custom proposals. Use when
#                 `with_rpn` is False.

#         Returns:
#             dict[str, Tensor]: a dictionary of loss components
#         """
#         x = self.extract_feat(img)

#         losses = dict()

#         if False:
#             sim_bboxes = []
#             sim_angles = []
#             for i in range(len(gt_bboxes)):
#                 sim_idx = simulated[i] == 1
#                 sim_bboxes.append(gt_bboxes[i][sim_idx])
#                 sim_angles.append(angles[i][sim_idx])
#                 gt_bboxes[i] = gt_bboxes[i][torch.logical_not(sim_idx)]
#                 gt_labels[i] = gt_labels[i][torch.logical_not(sim_idx)]
#                 simulated[i] = simulated[i][torch.logical_not(sim_idx)]
#                 angles[i] = angles[i][torch.logical_not(sim_idx)]
#         else:
#             sim_bboxes = None
#             sim_angles = None
        
#         # RPN forward and loss
#         if self.with_rpn:
#             proposal_cfg = self.train_cfg.get('rpn_proposal',
#                                               self.test_cfg.rpn)
#             rpn_losses, proposal_list = self.rpn_head.forward_train(
#                 x,
#                 img_metas,
#                 gt_bboxes,
#                 gt_labels=None,
#                 gt_bboxes_ignore=gt_bboxes_ignore,
#                 proposal_cfg=proposal_cfg,
#                 **kwargs)
#             losses.update(rpn_losses)
#         else:
#             proposal_list = proposals

#         roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
#                                                  gt_bboxes, gt_labels, 
#                                                 #  simulated, angles, 
#                                                  sim_bboxes, sim_angles, 
#                                                  gt_bboxes_ignore, gt_masks, 
#                                                  **kwargs)
#         # roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
#         #                                          gt_bboxes, gt_labels, 
#         #                                          gt_bboxes_ignore, gt_masks,
#         #                                          **kwargs)
#         losses.update(roi_losses)

#         # losses_for_classifier = {'loss_angle': losses['loss_angle']}

#         return losses
#         # return losses_for_classifier
    
#     # def forward_train(self,
#     #                   img,
#     #                   img_metas,
#     #                   gt_bboxes,
#     #                   gt_labels,
#     #                   simulated,
#     #                   angles,
#     #                   gt_bboxes_ignore=None,
#     #                   gt_masks=None,
#     #                   proposals=None,
#     #                   **kwargs):
#     #     """
#     #     Args:
#     #         img (Tensor): of shape (N, C, H, W) encoding input images.
#     #             Typically these should be mean centered and std scaled.

#     #         img_metas (list[dict]): list of image info dict where each dict
#     #             has: 'img_shape', 'scale_factor', 'flip', and may also contain
#     #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#     #             For details on the values of these keys see
#     #             `mmdet/datasets/pipelines/formatting.py:Collect`.

#     #         gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#     #             shape (num_gts, 5) in [cx, cy, w, h, a] format.

#     #         gt_labels (list[Tensor]): class indices corresponding to each box

#     #         gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#     #             boxes can be ignored when computing the loss.

#     #         gt_masks (None | Tensor) : true segmentation masks for each box
#     #             used if the architecture supports a segmentation task.

#     #         proposals : override rpn proposals with custom proposals. Use when
#     #             `with_rpn` is False.

#     #     Returns:
#     #         dict[str, Tensor]: a dictionary of loss components
#     #     """
#     #     x = self.extract_feat(img)
#     #     x_sim = self.extract_feat_sim(img)

#     #     losses = dict()
#     #     sim_gt_bboxes = []
#     #     sim_gt_labels = []
#     #     real_gt_bboxes = []
#     #     real_gt_labels = []
#     #     for sim_i, gts_i, labels_i in zip(simulated, gt_bboxes, gt_labels):
#     #         sim_idx = sim_i > 0
#     #         real_idx = sim_i == 0
#     #         sim_gt_bboxes.append(gts_i[sim_idx])
#     #         sim_gt_labels.append(labels_i[sim_idx])
#     #         real_gt_bboxes.append(gts_i[real_idx])
#     #         real_gt_labels.append(labels_i[real_idx])
#     #     # RPN forward and loss
#     #     if self.with_rpn:
#     #         proposal_cfg = self.train_cfg.get('rpn_proposal',
#     #                                           self.test_cfg.rpn)
#     #         rpn_losses, proposal_list = self.rpn_head.forward_train(
#     #             x,
#     #             img_metas,
#     #             real_gt_bboxes,
#     #             gt_labels=None,
#     #             gt_bboxes_ignore=gt_bboxes_ignore,
#     #             proposal_cfg=proposal_cfg,
#     #             **kwargs)

#     #         rpn_losses_sim, proposal_list_sim = self.rpn_head_sim.forward_train(
#     #             x_sim,
#     #             img_metas,
#     #             sim_gt_bboxes,
#     #             gt_labels=None,
#     #             gt_bboxes_ignore=gt_bboxes_ignore,
#     #             proposal_cfg=proposal_cfg,
#     #             **kwargs)
#     #         rpn_losses['loss_rpn_cls_sim'] = rpn_losses_sim['loss_rpn_cls']
#     #         rpn_losses['loss_rpn_bbox_sim'] = rpn_losses_sim['loss_rpn_bbox']
#     #         losses.update(rpn_losses)
#     #     else:
#     #         proposal_list = proposals

#     #     #concat proposals
#     #     # for i in range(len(proposal_list)):
#     #     #     proposal_list[i] = torch.cat(proposal_list[i], proposal_list_sim[i], dim=0)

#     #     roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list, x_sim, proposal_list_sim, 
#     #                                              real_gt_bboxes, real_gt_labels, sim_gt_bboxes, sim_gt_labels, gt_bboxes, gt_labels, simulated, angles, 
#     #                                              gt_bboxes_ignore, gt_masks,
#     #                                              **kwargs)
#     #     # roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
#     #     #                                          gt_bboxes, gt_labels, simulated, angles, 
#     #     #                                          gt_bboxes_ignore, gt_masks,
#     #     #                                          **kwargs)
#     #     losses.update(roi_losses)

#     #     # losses_for_classifier = {'loss_angle': losses['loss_angle']}

#     #     return losses
#     #     # return losses_for_classifier
    

#     def extract_feat_sim(self, img):
#         """Directly extract features from the backbone+neck."""
#         x = self.backbone_sim(img)
#         if self.with_neck:
#             x = self.neck(x)
#         return x
    
#     def simple_test(self, img, img_metas, proposals=None, rescale=False):
#         """Test without augmentation."""

#         assert self.with_bbox, 'Bbox head must be implemented.'
#         x = self.extract_feat(img)
#         # x_sim = self.extract_feat_sim(img)
#         if proposals is None:
#             proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
#             # proposal_list_sim = self.rpn_head_sim.simple_test_rpn(x_sim, img_metas)
#         else:
#             proposal_list = proposals

#         # return self.roi_head.simple_test(
#         #     x, proposal_list, img_metas, rescale=rescale)
#         return self.roi_head.simple_test(
#             x, proposal_list, img_metas, rescale=rescale)