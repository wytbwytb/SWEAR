# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrotate.core import rbbox2roi
from mmdet.core import bbox2roi
import torch.nn as nn
import torch.nn.functional as F
from ..builder import ROTATED_HEADS, build_loss
from .rotate_standard_roi_head import RotatedStandardRoIHead


@ROTATED_HEADS.register_module()
class OrientedStandardRoIHead(RotatedStandardRoIHead):
    """Oriented RCNN roi head including one bbox head."""

    def __init__(self, bbox_roi_extractor=None, bbox_head=None, shared_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None, version='oc'):
        super().__init__(bbox_roi_extractor, bbox_head, shared_head, train_cfg, test_cfg, pretrained, init_cfg, version)
        # self.mse = build_loss(dict(type='MSELoss', loss_weight=1.0, reduction='sum'))

    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
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
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
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
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
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
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        # bbox_results['bbox_pred'], bbox_results['var_pred'], rois,
                                        *bbox_targets)
        # loss_bbox['loss_mse'] = loss_mse / 10000
        
        # loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
        #                                 bbox_results['bbox_pred'], bbox_results['bbox_pred_h'], rois,
        #                                 *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

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
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        # cls_score, bbox_pred, bbox_pred_h = self.bbox_head(bbox_feats)
        # cls_score, bbox_pred, var_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
            # cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, var_pred=var_pred)
            # cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, bbox_pred_h=bbox_pred_h)
        return bbox_results

    def refine_pred(self, x, proposals, bbox_pred):
        refine_bbox = [self.bbox_head.bbox_coder.decode(proposal, pred) for proposal, pred in zip(proposals, bbox_pred)]
        
        refine_rois = rbbox2roi(refine_bbox)
        bbox_results = self._bbox_forward(x, refine_rois)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        refine_rois = refine_rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)
        
        return refine_rois, cls_score, bbox_pred

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

        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # rois, cls_score, bbox_pred = self.refine_pred(x, proposals, bbox_pred)
        
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
    
    def forward_gt(self, x, img_metas, proposal_list, gt_bboxes, gt_labels):
        rois = rbbox2roi([bboxes for bboxes in gt_bboxes])
        
        # rois = bbox2roi([bboxes for bboxes in gt_bboxes])
        bbox_results = self._bbox_forward(x, rois)
        # img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        # scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        cls_score = bbox_results['cls_score']
        scores = F.softmax(
                cls_score, dim=-1)
        # result = [np.zeros((0, 6), dtype=np.float32) for _ in range(self.bbox_head.num_classes)]
        # for s, gt in zip(scores, gt_bboxes):
        #     # result[gt]
        #     results[l][label] = np.concatenate((results[l][label], bbox))
        return scores
    
    def forward_box(self, x, proposals, img_metas):
        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        
        scores = cls_score.softmax(dim=1)
        bboxes = self.bbox_head.bbox_coder.decode(rois[:, 1:], bbox_pred)
        
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        scores = scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bboxes is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.split(num_proposals_per_img, 0)
            else:
                bboxes = self.bbox_head.bbox_pred_split(
                    bboxes, num_proposals_per_img)
        else:
            bboxes = (None, ) * len(proposals)
        
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
        
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=False,
                cfg=self.test_cfg)
                
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            
            
        return bboxes, scores, det_bboxes, det_labels

@ROTATED_HEADS.register_module()
class RefinedOrientedStandardRoIHead(RotatedStandardRoIHead):
    """Oriented RCNN roi head including one bbox head."""
    def __init__(self, bbox_roi_extractor=None, bbox_head=None, shared_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None, version='oc'):
        super().__init__(bbox_roi_extractor, bbox_head, shared_head, train_cfg, test_cfg, pretrained, init_cfg, version)
        self.mse = build_loss(dict(type='MSELoss', loss_weight=1.0, reduction='sum'))
        self.fuse_conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
    
    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
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
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
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
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
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
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
   
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        
        # loss_bbox_our = self.ours_module2(x, sampling_results, bbox_results, bbox_targets)
        loss_bbox_our = self.ours_module(x, sampling_results, bbox_results, bbox_targets)
        loss_bbox.update(loss_bbox_our)
        # loss_bbox['loss_angle'] = loss_angle
        # loss_bbox['loss_angle2'] = loss_angle2
        
        # loss_bbox2 = self.bbox_head.loss(cls_score,
        #                                 bbox_pred, rois[idx],
        #                                 *pos_bbox_targets)
        # loss_bbox['loss_mse'] = loss_mse
        # loss_bbox['loss_sim'] = loss_sim
        # loss_bbox['loss_cls_pos'] = loss_bbox2['loss_cls']
        # loss_bbox['loss_bbox_pos'] = loss_bbox2['loss_bbox']
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

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
        # refined_bbox_feats = self.BA(bbox_feats)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        # cls_score, bbox_pred, bbox_pred_h = self.bbox_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
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

        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)
        
        # rois, cls_score, bbox_pred = self.refine_pred_circle(x, proposals, bbox_pred, cls_score)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        
        
        return det_bboxes, det_labels
    
    def forward_box(self, x, img_metas, proposals, rcnn_test_cfg):
        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)
        
        bboxes = self.bbox_head.decode(proposals, bbox_pred)
        
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=False,
                cfg=self.test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        
        return bboxes, cls_score, det_bboxes, det_labels
    
    def forward_gt(self, x, img_metas, proposal_list, gt_bboxes, gt_labels):
        rois = rbbox2roi([bboxes for bboxes in gt_bboxes])
        
        # rois = bbox2roi([bboxes for bboxes in gt_bboxes])
        bbox_results = self._bbox_forward(x, rois)
        # img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        # scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        cls_score = bbox_results['cls_score']
        scores = F.softmax(
                cls_score, dim=-1)
        # result = [np.zeros((0, 6), dtype=np.float32) for _ in range(self.bbox_head.num_classes)]
        # for s, gt in zip(scores, gt_bboxes):
        #     # result[gt]
        #     results[l][label] = np.concatenate((results[l][label], bbox))
        return scores

    def ours_module(self, x, sampling_results, bbox_results, bbox_targets):
        loss_bbox = {}
        pos_ids = bbox_targets[0] < torch.max(bbox_targets[0])
        pos_bbox_pred = bbox_results['bbox_pred'][pos_ids]
        pos_rois = rbbox2roi([res.pos_bboxes for res in sampling_results])
        neg_rois = rbbox2roi([res.neg_bboxes for res in sampling_results])
        pos_gt_rois = rbbox2roi([res.pos_gt_bboxes for res in sampling_results])
        pos_gt_labels = torch.cat([res.pos_gt_labels for res in sampling_results], dim=0)
        
        bbox = self.bbox_head.bbox_coder.decode(pos_rois[:, 1:], pos_bbox_pred)
        bbox_rois = rbbox2roi([bbox])
        
        # pos_roi_feats, pos_gt_roi_feats = self.bbox_roi_extractor(
        #     x[:self.bbox_roi_extractor.num_inputs], pos_rois, pos_gt_rois)
        bbox_roi_feats, pos_gt_roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], bbox_rois, pos_gt_rois)
        
        neg_roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], neg_rois
        )
        
        # pos_roi_feats_flatten = self.bbox_head.forward_flatten_feat(pos_roi_feats)
        bbox_roi_feats_flatten = self.bbox_head.forward_flatten_feat(bbox_roi_feats)
        pos_gt_roi_feats_flatten = self.bbox_head.forward_flatten_feat(pos_gt_roi_feats)
        neg_roi_feats_flatten = self.bbox_head.forward_flatten_feat(neg_roi_feats)
        
        weight = bbox_roi_feats_flatten.new_ones(bbox_roi_feats_flatten.shape)

        is_gt = torch.cat([res.pos_is_gt for res in sampling_results])
        weight[is_gt > 0, :] = 0.0
        
        ###
        rand_neg_idx = torch.randint(0, neg_roi_feats_flatten.shape[0], (1,))
        concat_feats_flatten = torch.cat([pos_gt_roi_feats_flatten, neg_roi_feats_flatten[rand_neg_idx]], dim=0)
        concat_lables = torch.cat([pos_gt_labels, pos_gt_labels.new_full((1,), -1)])
        
        not_equal_matrix = concat_lables.unsqueeze(0) != concat_lables.unsqueeze(1)
        true_indices = [torch.nonzero(line).reshape(-1) for line in not_equal_matrix[:-1]]
        selected_indices =  torch.cat([torch.index_select(idxes, 0, torch.randint(0, idxes.shape[0], size=(1,)).cuda()) for idxes in true_indices])
        rand_contrast_gt = torch.index_select(concat_feats_flatten, 0, selected_indices)
        # loss_mse = self.mse(pos_roi_feats_flatten, pos_gt_roi_feats_flatten, weight)
        # sim = F.cosine_similarity(pos_roi_feats_flatten[is_gt == 0], 
        #                           pos_gt_roi_feats_flatten[is_gt == 0]) if torch.any(is_gt == 0) else pos_roi_feats_flatten.new_zeros(1)
        
        sim = F.cosine_similarity(bbox_roi_feats_flatten[is_gt == 0], 
                                  pos_gt_roi_feats_flatten[is_gt == 0]) if torch.any(is_gt == 0) else bbox_roi_feats_flatten.new_zeros(1)
        
        sim_con = F.cosine_similarity(bbox_roi_feats_flatten[is_gt == 0], 
                                  rand_contrast_gt[is_gt == 0]) if torch.any(is_gt == 0) else bbox_roi_feats_flatten.new_zeros(1)
        
        loss_sim = 1 - torch.sum(sim) / len(sim) + torch.sum(sim_con) / len(sim_con)
        # loss_sim = 1 - torch.sum(sim) / len(sim)
        loss_bbox['loss_sim'] = loss_sim
        
        # feats_flatten_concat = torch.stack([pos_roi_feats_flatten, pos_gt_roi_feats_flatten], dim=1)
        # feats_flatten = self.fuse_conv(feats_flatten_concat).squeeze()
        # angle_pred = self.bbox_head.forward_predict_angle(feats_flatten)

        # idx = (bbox_targets[0] < torch.max(bbox_targets[0]))
        # pos_bbox_targets = [target[idx] for target in bbox_targets]
        
        # loss_angle, loss_angle2 = self.bbox_head.loss_angle(angle_pred, bbox_results['bbox_pred'], rois, *bbox_targets)

        return loss_bbox
    
    def ours_module2(self, x, sampling_results, bbox_results, bbox_targets):
        loss_bbox = {}
        
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        refine_bbox = [self.bbox_head.bbox_coder.decode(rois[:, 1:], bbox_pred)]
        refine_rois = rbbox2roi(refine_bbox)
        refine_bbox_results = self._bbox_forward(x, refine_rois)

        loss_bbox = self.bbox_head.loss(refine_bbox_results['cls_score'],
                                        refine_bbox_results['bbox_pred'], refine_rois,
                                        *bbox_targets)
        
        new_loss_bbox = {}
        new_loss_bbox['loss_refine_cls'] = loss_bbox['loss_cls']
        new_loss_bbox['loss_refine_bbox'] = loss_bbox['loss_bbox']
        
        return new_loss_bbox
    
    
    
    def refine_pred(self, x, proposals, bbox_pred):
        refine_bbox = [self.bbox_head.bbox_coder.decode(proposal, pred) for proposal, pred in zip(proposals, bbox_pred)]
        
        rois = rbbox2roi(refine_bbox)
        bbox_results = self._bbox_forward(x, rois)

        # ori_rois = rbbox2roi(proposals)
        # ori_bbox_results = self._bbox_forward(x, ori_rois)
        # roi_feats_flatten = self.bbox_head.forward_flatten_feat(bbox_results['bbox_feats'])
        # ori_roi_feats_flatten = self.bbox_head.forward_flatten_feat(ori_bbox_results['bbox_feats'])

        # feats_flatten_concat = torch.stack([roi_feats_flatten, ori_roi_feats_flatten], dim=1)
        # feats_flatten = self.fuse_conv(feats_flatten_concat).squeeze()
        # angle_pred = self.bbox_head.forward_predict_angle(feats_flatten)

        
        # cls_score = ori_bbox_results['cls_score']
        # bbox_pred = ori_bbox_results['bbox_pred']
        # bbox_pred = torch.cat([bbox_pred[:, :-1], angle_pred], dim=1)
        # rois = ori_rois

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)
        
        return rois, cls_score, bbox_pred

    def refine_pred_circle(self, x, proposals, bbox_pred, cls_score, max_circle=10, conf_thr=0.9):
        bbox_pred = torch.cat(bbox_pred, dim=0)
        proposals = torch.cat(proposals, dim=0)
        cls_score = torch.cat(cls_score, dim=0)
        global_proposals = proposals.clone()
        global_bbox_pred = bbox_pred.clone()
        global_cls_score = cls_score.clone()

        ori_bbox_pred = bbox_pred.new_ones(bbox_pred.shape)
        ori_cls_score = cls_score.new_ones(cls_score.shape)
        refine_ids = bbox_pred.new_ones(bbox_pred.shape[0], dtype=torch.bool)
        
        circle_iter = 0
        while circle_iter < max_circle:
            cnt = torch.numel(ori_bbox_pred[refine_ids])
            cha = torch.abs(global_bbox_pred - ori_bbox_pred).sum() / torch.numel(ori_bbox_pred[refine_ids])
            if cha < 0.01 or cnt == 0:
                break
            ori_bbox_pred = global_bbox_pred.clone()
            ori_cls_score = global_cls_score.clone()
            scores = F.softmax(
                ori_cls_score, dim=-1)
            maxscore, _ = torch.max(scores, dim=-1)
            refine_ids = maxscore < conf_thr
            
            refine_bbox = [self.bbox_head.bbox_coder.decode(proposals, bbox_pred)]

            rois = rbbox2roi(refine_bbox)
            bbox_results = self._bbox_forward(x, rois)
            
            proposals = torch.cat(refine_bbox, dim=0)
            bbox_pred = bbox_results['bbox_pred']
            cls_score = bbox_results['cls_score']
            circle_iter += 1
            
            global_proposals[refine_ids, :-1] = torch.cat(refine_bbox, dim=0)[refine_ids].clone()
            global_bbox_pred[refine_ids] = bbox_results['bbox_pred'][refine_ids].clone()
            global_cls_score[refine_ids] = bbox_results['cls_score'][refine_ids].clone()
        
        print(circle_iter)
        rois = rbbox2roi([global_proposals])
        # cls_score = bbox_results['cls_score']
        # bbox_pred = bbox_results['bbox_pred']
        cls_score = global_cls_score
        bbox_pred = global_bbox_pred
        
        num_proposals_per_img = len(proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)
        
        return rois, cls_score, bbox_pred      
    
class Boundary_Aggregation(nn.Module):
    def __init__(self, in_channels):
        super(Boundary_Aggregation, self).__init__()
        self.conv = nn.Conv2d(in_channels * 5, in_channels, 1)

    def forward(self, x_batch: torch.tensor):
        in_channels, height, width = x_batch.size()[1:4]
        x_clk_rot = torch.rot90(x_batch, -1, [2, 3])
        x1 = self.up_to_bottom(x_batch, in_channels, height)
        x2 = self.bottom_to_up(x_batch, in_channels, height)
        x3 = self.left_to_right(x_clk_rot, in_channels, width)
        x4 = self.right_to_left(x_clk_rot, in_channels, width)
        x_con = torch.cat((x_batch, x1, x2, x3, x4), 1)
        x_merge = self.conv(x_con)
        return x_merge

    def left_to_right(self, x_clk_rot: torch.tensor, in_channels: int,
                      height: int):
        x = torch.clone(x_clk_rot)
        x = self.up_to_bottom(x, in_channels, height)
        x = torch.rot90(x, 1, [2, 3])
        return x

    def right_to_left(self, x_clk_rot: torch.tensor, in_channels: int,
                      height: int):
        x = torch.clone(x_clk_rot)
        x = self.bottom_to_up(x, in_channels, height)
        x = torch.rot90(x, 1, [2, 3])
        return x

    def bottom_to_up(self, x_raw: torch.tensor, in_channels: int, height: int):
        x = torch.clone(x_raw)
        for i in range(height - 1, -1, -1):
            x[:, :, i] = torch.max(x[:, :, i:], 2, True)[0].squeeze(2)
        return x

    def up_to_bottom(self, x_raw: torch.tensor, in_channels: int, height: int):
        x = torch.clone(x_raw)
        for i in range(height):
            x[:, :, i] = torch.max(x[:, :, :i + 1], 2, True)[0].squeeze(2)
        return x