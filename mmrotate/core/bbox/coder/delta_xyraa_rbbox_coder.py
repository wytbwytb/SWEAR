# Copyright (c) OpenMMLab. All rights reserved.
# Modified from jbwang1997: https://github.com/jbwang1997/OBBDetection
from locale import DAY_2
import mmcv
import numpy as np
import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder

from ..builder import ROTATED_BBOX_CODERS
from ..transforms import obb2poly, obb2xyxy, poly2obb, xyxy2poly, poly2obb_cc, obb2poly_cc


@ROTATED_BBOX_CODERS.register_module()
class DeltaXYRAARBBoxCoder(BaseBBoxCoder):
    """Mid point offset coder. This coder encodes bbox (x, y, w, h, a) into \
    delta (dx, dy, dr, da1, da2) and decodes delta (dx, dy, dr, da1, da2) \
    back to oriented bbox (x, y, w, h, a).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 angle_range='oc'):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.version = angle_range

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 5
        assert gt_bboxes.size(-1) == 5
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds,
                                    self.version)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `bboxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (torch.Tensor): Encoded offsets with respect to each
                roi. Has shape (B, N, 5) or (N, 5).
                Note N = num_anchors * W * H when rois is a grid of anchors.

            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 6), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        assert bboxes.size(-1) == 5
        assert pred_bboxes.size(-1) == 5
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    wh_ratio_clip, self.version)

        return decoded_bboxes


@mmcv.jit(coderize=True)
def bbox2delta(proposals,
               gt,
               means=(0., 0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1., 1.),
               version='oc'):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h, a, b of proposals w.r.t ground
    truth bboxes to get regression target. This is the inverse function of
    :func:`delta2bbox`.

    Args:
        proposals (torch.Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (torch.Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.
        version (str, optional): Angle representations. Defaults to 'oc'.

    Returns:
        Tensor: deltas with shape (N, 6), where columns represent dx, dy,
            dw, dh, da, db.
    """
    proposals = proposals.float()
    gt = gt.float()

    proposals_poly = obb2poly(proposals, version)
    proposals_cc = poly2obb_cc(proposals_poly)

    px = proposals_cc[..., 0]
    py = proposals_cc[..., 1]
    pr = proposals_cc[..., 2]
    pa1 = proposals_cc[..., 3]
    pa2 = proposals_cc[..., 4]

    gt_poly = obb2poly(gt, version)
    gt_cc = poly2obb_cc(gt_poly)
    
    gx = gt_cc[..., 0]
    gy = gt_cc[..., 1]
    gr = gt_cc[..., 2]
    ga1 = gt_cc[..., 3]
    ga2 = gt_cc[..., 4]

    dx = (gx - px) / pr
    dy = (gy - py) / pr
    dr = torch.log(gr / pr)
    da1 = torch.fmod((ga1 - pa1), np.pi) / np.pi
    # da2 = torch.fmod((ga2 - pa2), np.pi) / np.pi
    da2 = torch.fmod((ga2 - pa2), (np.pi - pa1)) / (np.pi - pa1)
    deltas = torch.stack([dx, dy, dr, da1, da2], dim=-1)
    # if torch.any(torch.isnan(deltas)):
    #     print(torch.any(torch.isnan(dx)),torch.any(torch.isnan(dy)),torch.any(torch.isnan(dr)),torch.any(torch.isnan(da1)),torch.any(torch.isnan(da2)))
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


@mmcv.jit(coderize=True)
def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1., 1.),
               r_ratio_clip=16 / 1000,
               version='oc'):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas
    are network outputs used to shift/scale those boxes. This is the inverse
    function of :func:`bbox2delta`.


    Args:
        rois (torch.Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (torch.Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        version (str, optional): Angle representations. Defaults to 'oc'.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5), where 5
           represent cx, cy, w, h, a.
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0]
    dy = denorm_deltas[:, 1]
    dr = denorm_deltas[:, 2]
    da1 = denorm_deltas[:, 3]
    da2 = denorm_deltas[:, 4]
    
    max_ratio = np.abs(np.log(r_ratio_clip))
    dr = dr.clamp(min=-max_ratio, max=max_ratio)
    da1 = da1.clamp(min=-1, max=1)
    da2 = da2.clamp(min=-1, max=1)

    rois_poly = obb2poly(rois)
    rois_cc = poly2obb_cc(rois_poly)
    px = rois_cc[..., 0]
    py = rois_cc[..., 1]
    pr = rois_cc[..., 2]
    pa1 = rois_cc[..., 3]
    pa2 = rois_cc[..., 4]

    gx = px + pr * dx
    gy = py + pr * dy
    gr = pr * dr.exp()
    ga1 = pa1 + np.pi * da1
    # ga2 = pa2 + np.pi * da2
    ga2 = pa2 + (np.pi - pa1) * da2
    
    proposals_cc = torch.stack([gx, gy, gr, ga1, ga2], dim=-1)
    proposals_poly = obb2poly_cc(proposals_cc)
    proposals_obb = poly2obb(proposals_poly)

    return proposals_obb
    # Compute center of each roi
    polys = torch.stack([ga, y1, x2, gb, _ga, y2, x1, _gb], dim=-1)

    center = torch.stack([gx, gy, gx, gy, gx, gy, gx, gy], dim=-1)
    center_polys = polys - center
    diag_len = torch.sqrt(center_polys[..., 0::2] * center_polys[..., 0::2] +
                          center_polys[..., 1::2] * center_polys[..., 1::2])
    max_diag_len, _ = torch.max(diag_len, dim=-1, keepdim=True)
    diag_scale_factor = max_diag_len / diag_len
    center_polys = center_polys * diag_scale_factor.repeat_interleave(
        2, dim=-1)
    rectpolys = center_polys + center
    obboxes = poly2obb(rectpolys, version)
    return obboxes
