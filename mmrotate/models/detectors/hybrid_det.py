# Copyright (c) OpenMMLab. All rights reserved.
import torch
import warnings

from ..builder import ROTATED_DETECTORS, build_detector, build_backbone, build_head, build_neck
from .two_stage import RotatedTwoStageDetector
from mmdet.models.detectors.base import BaseDetector


@ROTATED_DETECTORS.register_module()
class HybridDet(BaseDetector):
# class HybridDet(RotatedTwoStageDetector):

    def __init__(self,
                 model_h,
                 model_r,
                 train_cfg,
                 test_cfg,
                 pretrained=None,
                 init_cfg=None
                 ):
        super(HybridDet, self).__init__(init_cfg)
        
        self.model_h = build_detector(model_h)
        self.model_r = build_detector(model_r)
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes_h,
                      gt_bboxes_r,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        losses = dict()
        loss_h = self.model_h.forward_train(img,
                      img_metas,
                      gt_bboxes_h,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs)
        losses.update(loss_h)
        loss_r = self.model_r.forward_train(img,
                      img_metas,
                      gt_bboxes_r,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs)
        
        for key in loss_r.keys():
            losses[key + '_r'] = loss_r[key]
        
        return losses
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        output_h = self.model_h.simple_test(img, img_metas, proposals=None, rescale=True)
        
        output_r = self.model_r.simple_test(img, img_metas, proposals=None, rescale=True)
        
        
        # return [{'r': output_r[0]}]
        return [{'h': output_h[0], 'r': output_r[0]}]
    
    
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
    
    def forward_gt(self, img, img_metas, gt_bboxes_h, gt_bboxes_r, gt_labels, **kwargs):
        x_h, x_r = self.extract_feat(img)
        
        scores_h = self.model_h.roi_head.forward_gt(x_h, img_metas, None,
                                                 gt_bboxes_h, gt_labels, 
                                                 **kwargs)
        
        scores_r = self.model_r.roi_head.forward_gt(x_r, img_metas, None,
                                                 gt_bboxes_r, gt_labels, 
                                                 **kwargs)
        # return angles_pre
        return scores_h, scores_r
    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        
        x_h = self.model_h.backbone(img)
        x_r = self.model_r.backbone(img)
        
        x_h = self.model_h.neck(x_h)
        x_r =self.model_r.neck(x_r)
        
        return x_h, x_r
    
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

@ROTATED_DETECTORS.register_module()
class HybridDet2(BaseDetector):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head_h,
                 rpn_head_r,
                 roi_head_h,
                 roi_head_r,
                 train_cfg_h,
                 train_cfg_r,
                 test_cfg_h,
                 test_cfg_r,
                 train_cfg,
                 test_cfg,
                 pretrained=None,
                 init_cfg=None
                 ):
        super(HybridDet2, self).__init__(init_cfg)
        
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head_h is not None:
            rpn_train_cfg = train_cfg_h.rpn if train_cfg_h is not None else None
            rpn_head_ = rpn_head_h.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg_h.rpn)
            self.rpn_head_h = build_head(rpn_head_)
        
        if rpn_head_r is not None:
            rpn_train_cfg = train_cfg_r.rpn if train_cfg_r is not None else None
            rpn_head_ = rpn_head_r.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg_r.rpn)
            self.rpn_head_r = build_head(rpn_head_)

        if roi_head_h is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg_h.rcnn if train_cfg_h is not None else None
            roi_head_h.update(train_cfg=rcnn_train_cfg)
            roi_head_h.update(test_cfg=test_cfg_h.rcnn)
            roi_head_h.pretrained = pretrained
            self.roi_head_h = build_head(roi_head_h)

        if roi_head_r is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg_r.rcnn if train_cfg_r is not None else None
            roi_head_r.update(train_cfg=rcnn_train_cfg)
            roi_head_r.update(test_cfg=test_cfg_r.rcnn)
            roi_head_r.pretrained = pretrained
            self.roi_head_r = build_head(roi_head_r)

        self.train_cfg_h = train_cfg_h
        self.test_cfg_h = test_cfg_h
        self.train_cfg_r = train_cfg_r
        self.test_cfg_r = test_cfg_r

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head_h') and self.roi_head_h.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head_h') and self.roi_head_h.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head_h') and self.rpn_head_h is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head_h') and self.roi_head_h is not None


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes_h,
                      gt_bboxes_r,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()
        losses_h = dict()
        losses_r = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg_h = self.train_cfg_h.get('rpn_proposal',
                                              self.test_cfg_h.rpn)
            rpn_losses_h, proposal_list_h = self.rpn_head_h.forward_train(
                x,
                img_metas,
                gt_bboxes_h,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg_h,
                **kwargs)
            losses_h.update(rpn_losses_h)
        else:
            proposal_list_h = proposals

        roi_losses_h = self.roi_head_h.forward_train(x, img_metas, proposal_list_h,
                                                 gt_bboxes_h, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses_h.update(roi_losses_h)


        if self.with_rpn:
            proposal_cfg_r = self.train_cfg_r.get('rpn_proposal',
                                              self.test_cfg_r.rpn)
            rpn_losses_r, proposal_list_r = self.rpn_head_r.forward_train(
                x,
                img_metas,
                gt_bboxes_r,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg_r,
                **kwargs)
            losses_r.update(rpn_losses_r)
        else:
            proposal_list_r = proposals

        roi_losses_r = self.roi_head_r.forward_train(x, img_metas, proposal_list_r,
                                                 gt_bboxes_r, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses_r.update(roi_losses_r)

        losses.update(losses_h)
        for key in losses_r.keys():
            losses[key + '_r'] = losses_r[key]
        return losses
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

            
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list_h = self.rpn_head_h.simple_test_rpn(x, img_metas)
            proposal_list_r = self.rpn_head_r.simple_test_rpn(x, img_metas)
        else:
            proposal_list_h = proposals
            proposal_list_r = proposals

        output_h = self.roi_head_h.simple_test(
            x, proposal_list_h, img_metas, rescale=rescale)
        
        output_r = self.roi_head_r.simple_test(
            x, proposal_list_r, img_metas, rescale=rescale)
        
        
        # return [{'r': output_r[0]}]
        return [{'h': output_h[0], 'r': output_r[0]}]
    
    
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
    
    def forward_gt(self, img, img_metas, gt_bboxes, gt_labels, **kwargs):
        x = self.extract_feat(img)
        
        scores = self.roi_head.forward_gt(x, img_metas, None,
                                                 gt_bboxes, gt_labels, 
                                                 **kwargs)
        # return angles_pre
        return scores
    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # return None
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
