# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .two_stage import RotatedTwoStageDetector


@ROTATED_DETECTORS.register_module()
class GlidingVertex(RotatedTwoStageDetector):
    """Implementation of `Gliding Vertex on the Horizontal Bounding Box for
    Multi-Oriented Object Detection <https://arxiv.org/pdf/1911.09358.pdf>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(GlidingVertex, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmrotate/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs
    def forward_angle(self, img, img_metas, gt_bboxes, gt_labels, **kwargs):
        x = self.extract_feat(img)
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            _, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=proposal_cfg,
                **kwargs)
        else:
            proposal_list = None

        # angles_pre = self.roi_head.forward_angle(x, img_metas, proposal_list,
        #                                          gt_bboxes, gt_labels, 
        #                                          **kwargs)
        x_pre, y_pre = self.roi_head.forward_angle(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, 
                                                 **kwargs)
        # return angles_pre
        return x_pre, y_pre
    # origin
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                    #   simulated,
                    #   angles,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        if False:
            sim_bboxes = []
            sim_angles = []
            for i in range(len(gt_bboxes)):
                sim_idx = simulated[i] == 1
                sim_bboxes.append(gt_bboxes[i][sim_idx])
                sim_angles.append(angles[i][sim_idx])
                gt_bboxes[i] = gt_bboxes[i][torch.logical_not(sim_idx)]
                gt_labels[i] = gt_labels[i][torch.logical_not(sim_idx)]
                simulated[i] = simulated[i][torch.logical_not(sim_idx)]
                angles[i] = angles[i][torch.logical_not(sim_idx)]
        else:
            sim_bboxes = None
            sim_angles = None
        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, 
                                                #  simulated, angles, 
                                                 sim_bboxes, sim_angles, 
                                                 gt_bboxes_ignore, gt_masks, 
                                                 **kwargs)
        # roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
        #                                          gt_bboxes, gt_labels, 
        #                                          gt_bboxes_ignore, gt_masks,
        #                                          **kwargs)
        losses.update(roi_losses)

        # losses_for_classifier = {'loss_angle': losses['loss_angle']}

        return losses
        # return losses_for_classifier

    def extract_feat_sim(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_sim(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        # x_sim = self.extract_feat_sim(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            # proposal_list_sim = self.rpn_head_sim.simple_test_rpn(x_sim, img_metas)
        else:
            proposal_list = proposals

        # return self.roi_head.simple_test(
        #     x, proposal_list, img_metas, rescale=rescale)
        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
