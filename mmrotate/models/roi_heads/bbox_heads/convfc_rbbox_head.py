# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer
from mmdet.core import multi_apply

from mmrotate.core import obb2xyxy
from mmrotate.core import build_bbox_coder
from ...builder import ROTATED_HEADS, build_loss
from .rotated_bbox_head import RotatedBBoxHead


@ROTATED_HEADS.register_module()
class RotatedConvFCBBoxHead(RotatedBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg

    Args:
        num_shared_convs (int, optional): number of ``shared_convs``.
        num_shared_fcs (int, optional): number of ``shared_fcs``.
        num_cls_convs (int, optional): number of ``cls_convs``.
        num_cls_fcs (int, optional): number of ``cls_fcs``.
        num_reg_convs (int, optional): number of ``reg_convs``.
        num_reg_fcs (int, optional): number of ``reg_fcs``.
        conv_out_channels (int, optional): output channels of convolution.
        fc_out_channels (int, optional): output channels of fc.
        conv_cfg (dict, optional): Config of convolution.
        norm_cfg (dict, optional): Config of normalization.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(RotatedConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm1d(1024)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches

        # x = self.relu(self.bn(x))

        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@ROTATED_HEADS.register_module()
class RotatedShared2FCBBoxHead(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RotatedShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        # _, self.angle_fcs, _ = self._add_conv_fc_branch(
        #         0, 2, self.shared_out_channels)
        self.fc_angle = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=2)
        
    def forward_flatten_feat(self, x):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches

        x_flatten = x
        return x_flatten
    
    def forward_predict(self, x):
        x_cls = x
        x_reg = x
        
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        # for fc_h in self.reg_fcs_h:

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        
        # angle_pred = self.fc_angle(x_flatten)
        return cls_score, bbox_pred

    def forward_predict_angle(self, x):
        x_angle = x
        # for fc in self.angle_fcs:
        #     x_angle = self.relu(fc(x_angle))
        angle_pred = self.fc_angle(x_angle)
        return angle_pred

    def loss_angle(self, 
             angle_pred,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        
        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        if angle_pred.shape[0] != bbox_pred.shape[0]: ### pos vs all
            assert angle_pred.shape[0] == pos_inds.nonzero().shape[0]
        pos_bbox_targets = bbox_targets[pos_inds.type(torch.bool)]
        pos_bbox_weights = bbox_weights[pos_inds.type(torch.bool)]
        pos_rois = rois[pos_inds.type(torch.bool)]
        # class_agnostic
        pos_bbox_pred = bbox_pred.view(
                bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
        

        # origin_bbox_pred = bbox_pred.clone()
        pos_modified_pred = torch.cat([pos_bbox_pred[:, :-1], angle_pred], dim=1)
        # do not perform bounding box regression for BG anymore.
        
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`,
            # `GIouLoss`, `DIouLoss`) is applied directly on
            # the decoded bounding boxes, it decodes the
            # already encoded coordinates to absolute format.
            pos_modified_pred = self.bbox_coder.decode(pos_rois[:, 1:], pos_modified_pred)
            pos_bbox_pred = self.bbox_coder.decode(pos_rois[:, 1:], pos_bbox_pred)
        # if self.reg_class_agnostic:
        #     pos_bbox_pred = bbox_pred.view(
        #         bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
        # else:
        #     pos_bbox_pred = bbox_pred.view(
        #         bbox_pred.size(0), -1,
        #         5)[pos_inds.type(torch.bool),
        #             labels[pos_inds.type(torch.bool)]]

        pos_angle_weights = pos_bbox_weights.new_zeros(pos_bbox_weights.shape)
        pos_angle_weights[:, -1] = 1.0

        loss_angle = self.loss_bbox(
            pos_modified_pred,
            pos_bbox_targets,
            pos_angle_weights,
            avg_factor=bbox_targets.size(0),
            reduction_override=reduction_override)

        pos_bbox_targets[:, -1] = pos_bbox_pred[:, -1]
        loss_angle2 = self.loss_bbox(
            pos_modified_pred,
            pos_bbox_targets,
            pos_angle_weights,
            avg_factor=bbox_targets.size(0),
            reduction_override=reduction_override)
        
        # loss_angle2 = self.loss_bbox(
        #     pos_bbox_pred,
        #     pos_bbox_targets,
        #     pos_bbox_weights,
        #     avg_factor=bbox_targets.size(0),
        #     reduction_override=reduction_override)
        return loss_angle, loss_angle2

@ROTATED_HEADS.register_module()
class RotatedShared2FCAccessBBoxHead(RotatedShared2FCBBoxHead):
    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches

        # x = self.relu(self.bn(x))

        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        angle_pred = self.forward_predict_angle(x)
        return cls_score, bbox_pred, x, angle_pred
    
    def forward_cls(self, x):
        x_cls = x
        
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        
        # angle_pred = self.fc_angle(x_flatten)
        return cls_score
    
    def forward_reg(self, x):
        x_reg = x

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        # for fc_h in self.reg_fcs_h:

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        
        # angle_pred = self.fc_angle(x_flatten)
        return bbox_pred

@ROTATED_HEADS.register_module()
class RefinedRotatedShared2FCBBoxHead(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, version, bbox_coder_h, loss_bbox_h, fc_out_channels=1024, *args, **kwargs):
        super(RefinedRotatedShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.bbox_coder_h = build_bbox_coder(bbox_coder_h)
        out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
        self.loss_bbox_h = build_loss(loss_bbox_h)
        self.fc_reg_h = build_linear_layer(
            self.reg_predictor_cfg,
            in_features=self.reg_last_dim,
            out_features=out_dim_reg)
        self.version = version
        # self.fc_angle = nn.Linear(in_features=1024, out_features=2)

        
    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches

        x_flatten = x

        # x = self.relu(self.bn(x))

        x_cls = x
        x_reg = x
        x_reg_h = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        # for fc_h in self.reg_fcs_h:

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        bbox_pred_h = self.fc_reg_h(x_reg_h) if self.with_reg else None
        # angle_pred = self.fc_angle(x_flatten)
        return cls_score, bbox_pred, bbox_pred_h

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (torch.Tensor): Contains all the positive boxes,
                has shape (num_pos, 5), the last dimension 5
                represents [cx, cy, w, h, a].
            neg_bboxes (torch.Tensor): Contains all the negative boxes,
                has shape (num_neg, 5), the last dimension 5
                represents [cx, cy, w, h, a].
            pos_gt_bboxes (torch.Tensor): Contains all the gt_boxes,
                has shape (num_gt, 5), the last dimension 5
                represents [cx, cy, w, h, a].
            pos_gt_labels (torch.Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(torch.Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
                - bbox_weights(torch.Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 5).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 5)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 5)
        bbox_targets_h = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights_h = pos_bboxes.new_zeros(num_samples, 4)
        proposal = pos_bboxes.new_zeros(num_samples, 5)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
                pos_gt_bboxes_h = obb2xyxy(pos_gt_bboxes, self.version)
                pos_bboxes_h = obb2xyxy(pos_bboxes, self.version)
                pos_bbox_targets_h = self.bbox_coder_h.encode(
                    pos_bboxes_h, pos_gt_bboxes_h)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
            bbox_targets_h[:num_pos, :] = pos_bbox_targets_h
            bbox_weights_h[:num_pos, :] = 1
            proposal[:num_pos, :] = pos_bboxes
            
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, bbox_targets_h, bbox_weights_h, proposal

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 5),  the last dimension 5
                represents [cx, cy, w, h, a].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 5) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 5), the last dimension 4 represents
                  [cx, cy, w, h, a].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights, bbox_targets_h, bbox_weights_h, proposal = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_targets_h = torch.cat(bbox_targets_h, 0)
            bbox_weights_h = torch.cat(bbox_weights_h, 0)
            proposal = torch.cat(proposal, 0)
        return labels, label_weights, bbox_targets, bbox_weights, bbox_targets_h, bbox_weights_h, proposal

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             bbox_pred_h,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             bbox_targets_h,
             bbox_weights_h,
             proposal,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                    pos_bbox_pred_h = bbox_pred_h.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    pos_bbox_pred_h = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                pos_proposals = proposal[pos_inds.type(torch.bool)]
                pos_bbox = self.bbox_coder.decode(pos_proposals, pos_bbox_pred)
                pos_bbox_h = obb2xyxy(pos_bbox, self.version)
                pos_proposals_h = obb2xyxy(pos_proposals, self.version)
                pos_bbox_pred_rh = self.bbox_coder_h.encode(pos_proposals_h, pos_bbox_h)
                # bbox_targets_h = bbox_targets_h(pos_inds.type(torch.bool))
                pos_gts_h = self.bbox_coder_h.decode(pos_proposals_h, bbox_targets_h[pos_inds.type(torch.bool)])
                # losses['loss_bbox_h'] = self.loss_bbox(
                #     pos_bbox_pred_h,
                #     bbox_targets_h[pos_inds.type(torch.bool)],
                #     bbox_weights_h[pos_inds.type(torch.bool)],
                #     avg_factor=bbox_targets_h.size(0),
                #     reduction_override=reduction_override)
                # losses['loss_bbox_h'] = self.loss_bbox(
                #     pos_bbox_pred_h,
                #     bbox_targets_h[pos_inds.type(torch.bool)],
                #     bbox_weights_h[pos_inds.type(torch.bool)],
                #     avg_factor=bbox_targets_h.size(0),
                #     reduction_override=reduction_override)
                # losses['loss_bbox_rh'] = self.loss_bbox(
                #     pos_bbox_pred_h,
                #     pos_bbox_pred_rh,
                #     bbox_weights_h[pos_inds.type(torch.bool)],
                #     avg_factor=bbox_targets_h.size(0),
                #     reduction_override=reduction_override)
                losses['loss_bbox_rh'] = self.loss_bbox_h(
                    pos_bbox_h,
                    pos_gts_h,
                    bbox_weights_h[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets_h.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

@ROTATED_HEADS.register_module()
class RotatedKFIoUShared2FCBBoxHead(RotatedConvFCBBoxHead):
    """KFIoU RoI head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RotatedKFIoUShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function."""
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                bbox_pred_decode = self.bbox_coder.decode(
                    rois[:, 1:], bbox_pred)
                bbox_targets_decode = self.bbox_coder.decode(
                    rois[:, 1:], bbox_targets)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                    pos_bbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred_decode.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    pos_bbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred_decode.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    pred_decode=pos_bbox_pred_decode,
                    targets_decode=bbox_targets_decode[pos_inds.type(
                        torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses


@ROTATED_HEADS.register_module()
class KLRotatedShared2FCBBoxHead(RotatedShared2FCBBoxHead):

    def __init__(self, fc_out_channels=1024, num_var_convs=0, num_var_fcs=0, *args, **kwargs):
        super().__init__(fc_out_channels, *args, **kwargs)
        out_dim_var = (5 if self.reg_class_agnostic else 4 *
                           self.num_classes)
        self.num_var_convs = num_var_convs
        self.num_var_fcs = num_var_fcs
        
        self.var_convs, self.var_fcs, self.var_last_dim = \
            self._add_conv_fc_branch(
                self.num_var_convs, self.num_var_fcs, self.shared_out_channels)
        
        self.fc_var = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.var_last_dim,
                out_features=out_dim_var)
    
    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_var = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
            
        for conv in self.var_convs:
            x_var = conv(x_var)
        if x_var.dim() > 2:
            if self.with_avg_pool:
                x_var = self.avg_pool(x_var)
            x_var = x_var.flatten(1)
        for fc in self.var_fcs:
            x_var = self.relu(fc(x_var))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        var_pred = self.fc_var(x_var)
        return cls_score, bbox_pred, var_pred


    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'var_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             var_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_targets = self.bbox_coder.decode(rois[:, 1:], bbox_targets)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                pos_var_pred = var_pred.view(
                        var_pred.size(0), 5)[pos_inds.type(torch.bool)]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    pos_var_pred,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                # if losses['loss_bbox']
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses