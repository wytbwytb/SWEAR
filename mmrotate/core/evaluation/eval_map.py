# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import get_context

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmcv.utils import print_log
from mmdet.core import average_precision
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmcv.ops import batched_nms
from mmrotate.core.bbox.transforms import obb2xyxy, hbb2obb
from terminaltables import AsciiTable


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 5). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0],
                  dtype=bool), np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        return tp, fp

    ious = box_iou_rotated(
        torch.from_numpy(det_bboxes).float(),
        torch.from_numpy(gt_bboxes).float()).numpy()
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp

def tpfp_default_h(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 5). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0],
                  dtype=bool), np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        return tp, fp
        # return tp, fp, det_r_ious, det_h_ious

    # ious = box_iou_rotated(
    #     torch.from_numpy(det_bboxes).float(),
    #     torch.from_numpy(gt_bboxes).float()).numpy()
    ious = bbox_overlaps(
        det_bboxes, gt_bboxes, use_legacy_coordinate=True)
    # det = torch.from_numpy(det_bboxes)[:, :5]
    # gt = torch.from_numpy(gt_bboxes)
    # det_h = obb2xyxy(det, version='le90')
    # det_r = hbb2obb(det_h, version='le90')
    # gt_h = obb2xyxy(gt, version='le90')
    # ious = bbox_overlaps(
    #     det_h.numpy(), gt.numpy(), use_legacy_coordinate=True)
    # ious_h = box_iou_rotated(
    #     det_r.float(),
    #     gt.float()).numpy()
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                        # det_r_ious.append(ious_max[i])
                        # det_h_ious.append(ious_h[i,matched_gt])
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp
    # return tp, fp, det_r_ious, det_h_ious


def tpfp_default_level(level_cnt, 
                 det_bboxes,
                 gt_bboxes,
                 gt_levels,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 5). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    # level_cnt = 2
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    tp_mask = [np.ones((num_dets), dtype=bool) for _ in range(level_cnt)]
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        tp_all = [tp for _ in range(level_cnt)]
        fp_all = [fp for _ in range(level_cnt)]
        
        return tp_all, fp_all, tp_mask
    
    ious = box_iou_rotated(
        torch.from_numpy(det_bboxes).float(),
        torch.from_numpy(gt_bboxes).float()).numpy()
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            matched_gt = ious_argmax[i]
            if gt_ignore_inds[matched_gt] or ious_max[i] == 0:
                continue
            level = gt_levels[matched_gt]
            # for the other level, this tp don't count
            for j in range(level_cnt-1):
                tp_mask[(level+j+1) % level_cnt][i] = False
            # tp_mask[(level+1) % level_cnt][i] = False
            if ious_max[i] >= iou_thr:
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    tp_all = [tp[:,tp_mask[i]] for i in range(level_cnt)]
    fp_all = [fp[:,tp_mask[i]] for i in range(level_cnt)]
    return tp_all, fp_all, tp_mask

def tpfp_default_angle(level_cnt, 
                 det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 5). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    # level_cnt = 2
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    angles = gt_bboxes[:,-1]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    tp_mask = [np.ones((num_dets), dtype=bool) for _ in range(level_cnt)]
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        tp_all = [tp for _ in range(level_cnt)]
        fp_all = [fp for _ in range(level_cnt)]
        
        return tp_all, fp_all, tp_mask
    
    ious = box_iou_rotated(
        torch.from_numpy(det_bboxes).float(),
        torch.from_numpy(gt_bboxes).float()).numpy()
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            matched_gt = ious_argmax[i]
            angle = angles[matched_gt]
            level = cal_angle_level(level_cnt, angle)
            # print(level)
            # for the other 2 level, this tp don't count
            for j in range(level_cnt-1):
                tp_mask[(level+j) % level_cnt][i] = False
            # tp_mask[(level+1) % level_cnt][i] = False
            if ious_max[i] >= iou_thr:
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    tp_all = [tp[:,tp_mask[i]] for i in range(level_cnt)]
    fp_all = [fp[:,tp_mask[i]] for i in range(level_cnt)]
    return tp_all, fp_all, tp_mask

def tpfp_default_angle_level(level_cnt, 
                 det_bboxes,
                 gt_bboxes,
                 gt_levels,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    
    # an indicator of ignored gts
    # level_cnt = 2
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    angles = gt_bboxes[:,-1]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    tp_mask_1 = [np.ones((num_dets), dtype=bool) for _ in range(level_cnt)]
    tp_mask_2 = [np.ones((num_dets), dtype=bool) for _ in range(level_cnt)]
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        tp_all = [tp for _ in range(level_cnt)]
        fp_all = [fp for _ in range(level_cnt)]
        
        return tp_all, fp_all, tp_mask_1, tp_all, fp_all, tp_mask_2
    
    ious = box_iou_rotated(
        torch.from_numpy(det_bboxes).float(),
        torch.from_numpy(gt_bboxes).float()).numpy()
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            matched_gt = ious_argmax[i]
            angle = angles[matched_gt]
            angle_level = cal_angle_level(level_cnt, angle)
            level = gt_levels[matched_gt]
            # print(level)
            # for the other 2 level, this tp don't count
            if level == 1:
                for j in range(level_cnt-1):
                    tp_mask_1[(angle_level+j) % level_cnt][i] = False
                for j in range(level_cnt):
                    tp_mask_2[(angle_level+j) % level_cnt][i] = False
            else:
                for j in range(level_cnt-1):
                    tp_mask_2[(angle_level+j) % level_cnt][i] = False
                for j in range(level_cnt):
                    tp_mask_1[(angle_level+j) % level_cnt][i] = False
            # tp_mask[(level+1) % level_cnt][i] = False
            if ious_max[i] >= iou_thr:
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    tp_all_1 = [tp[:,tp_mask_1[i]] for i in range(level_cnt)]
    fp_all_1 = [fp[:,tp_mask_1[i]] for i in range(level_cnt)]
    tp_all_2 = [tp[:,tp_mask_2[i]] for i in range(level_cnt)]
    fp_all_2 = [fp[:,tp_mask_2[i]] for i in range(level_cnt)]
    return tp_all_1, fp_all_1, tp_mask_1, tp_all_2, fp_all_2, tp_mask_2


def get_cls_results_level(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    cls_gts_level = []
    cls_gts_x = []
    cls_gts_y = []
    cls_gts_xy_max = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id

        # for simulated test
        save_inds = ann['simulated'] > 0
        ignore_inds = ann['simulated'] == 0
        gt_save_inds = gt_inds & save_inds
        gt_ignore_inds = gt_inds & ignore_inds
        
        cls_gts.append(ann['bboxes'][gt_save_inds, :])
        cls_gts_level.append(ann['simulated'][gt_save_inds])
        x = ann['angles'][gt_save_inds, 0] / 30
        cls_gts_x.append(x.astype(int))
        y = ann['angles'][gt_save_inds, 1] / 30
        cls_gts_y.append(y.astype(int))

        # avg_new
        avg = (ann['angles'][gt_save_inds, 0] + ann['angles'][gt_save_inds, 1]) / 2
        avg_l12 = avg / 50
        avg_l23 = avg / 70
        avg_l3 = avg_l23 >= 1
        avg_l12[avg_l3] = 2.0
        cls_gts_xy_max.append(avg_l12.astype(int))

        # Max(x,y)_new
        # x_l12 = ann['angles'][gt_save_inds, 0] / 50
        # x_l23 = ann['angles'][gt_save_inds, 0] / 70
        # x_l3 = x_l23 >= 1
        # x = x_l12
        # x[x_l3] = 2.0

        # y_l12 = ann['angles'][gt_save_inds, 1] / 50
        # y_l23 = ann['angles'][gt_save_inds, 1] / 70
        # y_l3 = y_l23 >= 1
        # y = y_l12
        # y[y_l3] = 2.0
        # # max = np.maximum(x, y)
        # max = x
        # # max = (ann['angles'][gt_save_inds, 0] + ann['angles'][gt_save_inds, 1]) / 2 / 30 
        # cls_gts_xy_max.append(max.astype(int))
        
        # horizontal rotate
        # z_l = (ann['angles'][gt_save_inds, 2] - 1) / 120
        # cls_gts_xy_max.append(z_l.astype(int))


        # y_l12 = ann['angles'][gt_save_inds, 1] / 50
        # y_l23 = ann['angles'][gt_save_inds, 1] / 70
        # y_l3 = y_l23 >= 1
        # y = y_l12
        # y[y_l3] = 2.0
        # # max = np.maximum(x, y)
        # max = x
        # # max = (ann['angles'][gt_save_inds, 0] + ann['angles'][gt_save_inds, 1]) / 2 / 30 
        # cls_gts_xy_max.append(max.astype(int))
        
        cls_gts_ignore.append(ann['bboxes'][gt_ignore_inds, :])

        # origin
        # cls_gts.append(ann['bboxes'][gt_inds, :])
        # cls_gts_level.append(ann['simulated'][gt_inds])
        # x = ann['angles'][gt_inds, 0] / 3
        # cls_gts_x.append(x.astype(int))
        # y = ann['angles'][gt_inds, 1] / 3
        # cls_gts_y.append(y.astype(int))

        # # Max(x,y)_new
        # # x_l12 = ann['angles'][gt_inds, 0] / 5
        # # x_l23 = ann['angles'][gt_inds, 0] / 7
        # # x_l3 = x_l23 >= 1
        # # x = x_l12
        # # x[x_l3] = 2.0

        # # y_l12 = ann['angles'][gt_inds, 1] / 5
        # # y_l23 = ann['angles'][gt_inds, 1] / 7
        # # y_l3 = y_l23 >= 1
        # # y = y_l12
        # # y[y_l3] = 2.0
        # # max = np.maximum(x, y)

        # # avg_new
        # avg = (ann['angles'][gt_inds, 0] + ann['angles'][gt_inds, 1]) / 2
        # avg_l12 = avg / 50
        # avg_l23 = avg / 70
        # avg_l3 = avg_l23 >= 1
        # avg_l12[avg_l3] = 2.0
        # cls_gts_xy_max.append(avg_l12.astype(int))

        # # max = (ann['angles'][gt_inds, 0] + ann['angles'][gt_inds, 1]) / 2 / 3
        # # cls_gts_xy_max.append(max.astype(int))

        # if ann.get('labels_ignore', None) is not None:
        #     ignore_inds = ann['labels_ignore'] == class_id
        #     cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])

        # else:
        #     cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))

    return cls_dets, cls_gts, cls_gts_level, cls_gts_x, cls_gts_y, cls_gts_xy_max, cls_gts_ignore

def get_cls_results_hlevel(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    cls_gts_level = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        
        horizontal_angle = ann['bboxes'][gt_inds, -1]
        
        ### 根据angle-version重新判断
        ## le90
        level = (horizontal_angle + np.pi / 2) / (np.pi / 3)
        
        ## oc
        # level = (horizontal_angle / (np.pi / 6))
        if np.any(level == 3):
            print(1)
        
        cls_gts.append(ann['bboxes'][gt_inds, :])
        cls_gts_level.append(level.astype(int))
        
        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])

        else:
            cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))


    return cls_dets, cls_gts, cls_gts_level, cls_gts_ignore


def get_cls_results(det_results, annotations, class_id, mode=None):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    
    if mode == 'h':
        bbox_type = 'bboxes_h'
        bboxes_ignore_type = 'bboxes_h_ignore'
    elif mode == 'r':
        bbox_type = 'bboxes_r'
        bboxes_ignore_type = 'bboxes_r_ignore'
        if annotations[0].get(bbox_type, None) is None:
            bbox_type = 'bboxes'
            bboxes_ignore_type = 'bboxes_ignore'
    else:
        bbox_type = 'bboxes'
        bboxes_ignore_type = 'bboxes_ignore'
        
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann[bbox_type][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann[bboxes_ignore_type][ignore_inds, :])

        else:
            cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))

    return cls_dets, cls_gts, cls_gts_ignore

def get_cls_results_ignore(det_results, annotations, class_id, simulated_select):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id

        save_inds = ann['simulated'] == simulated_select
        ignore_inds = ann['simulated'] == 1 - simulated_select
        gt_save_inds = gt_inds & save_inds
        gt_ignore_inds = gt_inds & ignore_inds

        cls_gts.append(ann['bboxes'][gt_save_inds, :])

        if ann.get('labels_ignore', None) is not None:
            # ignore_inds = ann['labels_ignore'] == class_id
            # cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
            cls_gts_ignore.append(ann['bboxes'][gt_ignore_inds, :])

        else:
            cls_gts_ignore.append(ann['bboxes'][gt_ignore_inds, :])

    return cls_dets, cls_gts, cls_gts_ignore

def eval_rbbox_map_for_2(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):

    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)

    print('only for real')
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results_ignore(
            det_results, annotations, i, 0)

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))

        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # # choose real gts
        # ids = tp > 0 or fp > 0
        # cls_dets = cls_dets[ids]
        # tp = tp[ids]
        # fp = fp[ids]
        

        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    print('only for simulated')
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results_ignore(
            det_results, annotations, i, 1)

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))

        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results

def eval_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    
    if len(det_results[0]) != len(dataset):
        num_classes = len(dataset)
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        cls_name = dataset[i]
        select_cls = ['umbrella', 
               'OCbottle', 'glassbottle', 'metalbottle',
               'knife', 'gun']
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i, 'h')

        # compute tp and fp for each image with multiple processes
        
        tp = []
        fp = []

        for j in range(num_imgs):
            tp_j, fp_j= tpfp_default_h(cls_dets[j], cls_gts[j],
             cls_gts_ignore[j], iou_thr, area_ranges)
            # tp_i, fp_i, mask_i = tpfp_default_angle(level_cnt, cls_dets[i], cls_gts[i],
            #  cls_gts_ignore[i], iou_thr, area_ranges)
            tp.append(tp_j)
            fp.append(fp_j)

        # tpfp = pool.starmap(
        #     tpfp_default_h,
        #     zip(cls_dets, cls_gts, cls_gts_ignore,
        #         [iou_thr for _ in range(num_imgs)],
        #         [area_ranges for _ in range(num_imgs)]))
        # tp, fp, det_r_ious, det_h_ious = tuple(zip(*tpfp))

        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results

def eval_hybrid_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):

    assert len(det_results) == len(annotations)
    det_results_h = [result['h'] for result in det_results]
    det_results_r = [result['r'] for result in det_results]
    
    _ = eval_map(det_results_h, annotations, scale_ranges=scale_ranges, iou_thr=iou_thr, dataset=dataset, logger=logger, nproc=nproc)
    _ = eval_rbbox_map(det_results_r, annotations, scale_ranges=scale_ranges, iou_thr=iou_thr, dataset=dataset, logger=logger, nproc=nproc)
    # det = torch.from_numpy(det_bboxes)[:, :5]
    # det_results_r2h = [torch.from_numpy(bboxes)[:, 5] for bboxes in det_results_r]
    # multiclass_nms()

    num_imgs = len(det_results_h)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results_h[0])  # positive class num
    
    if len(det_results_h[0]) != len(dataset):
        num_classes = len(dataset)
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        cls_name = dataset[i]
        select_cls = ['umbrella', 
               'OCbottle', 'glassbottle', 'metalbottle',
               'knife', 'gun']
        # get gt and det bboxes of this class
        cls_dets_h, cls_gts_h, cls_gts_h_ignore = get_cls_results(
            det_results_h, annotations, i, 'h')
        
        cls_dets_r, cls_gts_r, cls_gts_r_ignore = get_cls_results(
            det_results_r, annotations, i, 'r')

        # compute tp and fp for each image with multiple processes
        
        tp = []
        fp = []
        cls_dets_hybrid = []

        for j in range(num_imgs):
            scores_h = torch.from_numpy(cls_dets_h[j][:, -1])
            boxes_h = torch.from_numpy(cls_dets_h[j][:, :4])
            scores_r = torch.from_numpy(cls_dets_r[j][:, -1])
            boxes_r = torch.from_numpy(cls_dets_r[j][:, :5])
            boxes_r2h = obb2xyxy(boxes_r, version='le90')

            scores_hybrid = torch.cat([scores_h, scores_r], dim=0)
            boxes_hybrid = torch.cat([boxes_h, boxes_r2h], dim=0)
            labels = torch.zeros_like(scores_hybrid)
            if scores_hybrid.shape[0] > 0:
                cls_dets_hybrid_j, keep = batched_nms(boxes_hybrid, scores_hybrid, labels, nms_cfg=dict(type='nms', iou_threshold=0.5))
                cls_dets_hybrid_j = cls_dets_hybrid_j.numpy()
            else:
                cls_dets_hybrid_j = cls_dets_h[j]
            cls_dets_hybrid.append(cls_dets_hybrid_j)
            # scores_keep = scores_hybrid[keep].reshape(-1,1)
            # b = boxes_hybrid[keep]
            # cls_dets_hybrid = torch.cat([boxes_keep,scores_keep], dim=-1)
            
            tp_j, fp_j = tpfp_default_h(cls_dets_hybrid_j, cls_gts_h[j],
             cls_gts_h_ignore[j], iou_thr, area_ranges)
            
            # tp_i, fp_i = tpfp_default(cls_dets_r[i], cls_gts_r[i],
            #  cls_gts_r_ignore[i], iou_thr, area_ranges)
            # tp_i, fp_i, mask_i = tpfp_default_angle(level_cnt, cls_dets[i], cls_gts[i],
            #  cls_gts_ignore[i], iou_thr, area_ranges)
            tp.append(tp_j)
            fp.append(fp_j)

        # tpfp = pool.starmap(
        #     tpfp_default_h,
        #     zip(cls_dets, cls_gts, cls_gts_ignore,
        #         [iou_thr for _ in range(num_imgs)],
        #         [area_ranges for _ in range(num_imgs)]))
        # tp, fp, det_r_ious, det_h_ious = tuple(zip(*tpfp))

        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts_h):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets_hybrid)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results




def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    
    if len(det_results[0]) != len(dataset):
        num_classes = len(dataset)
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        select_cls = ['umbrella', 
               'OCbottle', 'glassbottle', 'metalbottle',
               'knife', 'gun']
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i, 'r')

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))

        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results

def eval_rbbox_map_level(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    level_cnt = 3

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []

    tp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    fp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    score_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]

    for i in range(num_classes):
        # if dataset[i] != 'battery':
        #     continue
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_level, _, _, _, cls_gts_ignore = get_cls_results_level(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tp = [[] for _ in range(level_cnt)]
        fp = [[] for _ in range(level_cnt)]
        mask = [[] for _ in range(level_cnt)]
        for i in range(num_imgs):
            tp_i, fp_i, mask_i = tpfp_default_level(level_cnt, cls_dets[i], cls_gts[i],
             cls_gts_level[i], cls_gts_ignore[i], iou_thr, area_ranges)
            # tp_i, fp_i, mask_i = tpfp_default_angle(level_cnt, cls_dets[i], cls_gts[i],
            #  cls_gts_ignore[i], iou_thr, area_ranges)
            for j in range(level_cnt):
                tp[j].append(tp_i[j])
                fp[j].append(fp_i[j])
                mask[j].append(mask_i[j])
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                level = cls_gts_level[_]
                if level.shape[0] == 0:
                    continue
                else:
                    for i in range(level.shape[0]):
                        num_gts[level[i]-1][0] += 1
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        score = cls_dets[:,-1:]
        score = score.T
        # num_dets = cls_dets.shape[0]
        # sort_inds = np.argsort(-cls_dets[:, -1])
        tp = [np.hstack(tp[i]) for i in range(level_cnt)]
        fp = [np.hstack(fp[i]) for i in range(level_cnt)]
        mask = [np.hstack(mask[i]) for i in range(level_cnt)]
        scores = [score[:,mask[i]] for i in range(level_cnt)]
        for j in range(level_cnt):
            tp_all[j] = np.hstack([tp_all[j],tp[j]])
            fp_all[j] = np.hstack([fp_all[j],fp[j]])
            num_gts_all[j] += num_gts[j]
            score_all[j] = np.hstack([score_all[j], scores[j]])
        
    pool.close()
    # calculate recall and precision with tp and fp
    for i in range(level_cnt):
        tp = tp_all[i]
        fp = fp_all[i]
        score = score_all[i]
        num_gts = num_gts_all[i]

        num_dets = tp.shape[1]
        sort_inds = np.argsort(-score)
        tp = tp[:, sort_inds].squeeze(axis = 0)
        fp = fp[:, sort_inds].squeeze(axis = 0)
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, ('level1', 'level2', 'level3'), area_ranges, logger=logger)

    return mean_ap, eval_results

def eval_rbbox_map_xlevel(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    level_cnt = 3

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []

    tp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    fp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    score_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]

    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, _, cls_gts_x, _, _, cls_gts_ignore = get_cls_results_level(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tp = [[] for _ in range(level_cnt)]
        fp = [[] for _ in range(level_cnt)]
        mask = [[] for _ in range(level_cnt)]
        for i in range(num_imgs):
            tp_i, fp_i, mask_i = tpfp_default_level(level_cnt, cls_dets[i], cls_gts[i],
             cls_gts_x[i], cls_gts_ignore[i], iou_thr, area_ranges)
            # tp_i, fp_i, mask_i = tpfp_default_angle(level_cnt, cls_dets[i], cls_gts[i],
            #  cls_gts_ignore[i], iou_thr, area_ranges)
            for j in range(level_cnt):
                tp[j].append(tp_i[j])
                fp[j].append(fp_i[j])
                mask[j].append(mask_i[j])
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                level = cls_gts_x[_]
                if level.shape[0] == 0:
                    continue
                else:
                    for i in range(level.shape[0]):
                        num_gts[level[i]][0] += 1
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        score = cls_dets[:,-1:]
        score = score.T
        # num_dets = cls_dets.shape[0]
        # sort_inds = np.argsort(-cls_dets[:, -1])
        tp = [np.hstack(tp[i]) for i in range(level_cnt)]
        fp = [np.hstack(fp[i]) for i in range(level_cnt)]
        mask = [np.hstack(mask[i]) for i in range(level_cnt)]
        scores = [score[:,mask[i]] for i in range(level_cnt)]
        for j in range(level_cnt):
            tp_all[j] = np.hstack([tp_all[j],tp[j]])
            fp_all[j] = np.hstack([fp_all[j],fp[j]])
            num_gts_all[j] += num_gts[j]
            score_all[j] = np.hstack([score_all[j], scores[j]])
        
    pool.close()
    # calculate recall and precision with tp and fp
    for i in range(level_cnt):
        tp = tp_all[i]
        fp = fp_all[i]
        score = score_all[i]
        num_gts = num_gts_all[i]

        num_dets = tp.shape[1]
        sort_inds = np.argsort(-score)
        tp = tp[:, sort_inds].squeeze(axis = 0)
        fp = fp[:, sort_inds].squeeze(axis = 0)
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, ['level1', 'level2', 'level3', 'level4', 'level5', 'level6', 'level7', 'level8', 'level9'], area_ranges, logger=logger)

    return mean_ap, eval_results

def eval_rbbox_map_ylevel(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    level_cnt = 3

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []

    tp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    fp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    score_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]

    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, _, _, cls_gts_y, _, cls_gts_ignore = get_cls_results_level(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tp = [[] for _ in range(level_cnt)]
        fp = [[] for _ in range(level_cnt)]
        mask = [[] for _ in range(level_cnt)]
        for i in range(num_imgs):
            tp_i, fp_i, mask_i = tpfp_default_level(level_cnt, cls_dets[i], cls_gts[i],
             cls_gts_y[i], cls_gts_ignore[i], iou_thr, area_ranges)
            # tp_i, fp_i, mask_i = tpfp_default_angle(level_cnt, cls_dets[i], cls_gts[i],
            #  cls_gts_ignore[i], iou_thr, area_ranges)
            for j in range(level_cnt):
                tp[j].append(tp_i[j])
                fp[j].append(fp_i[j])
                mask[j].append(mask_i[j])
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                level = cls_gts_y[_]
                if level.shape[0] == 0:
                    continue
                else:
                    for i in range(level.shape[0]):
                        num_gts[level[i]][0] += 1
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        score = cls_dets[:,-1:]
        score = score.T
        # num_dets = cls_dets.shape[0]
        # sort_inds = np.argsort(-cls_dets[:, -1])
        tp = [np.hstack(tp[i]) for i in range(level_cnt)]
        fp = [np.hstack(fp[i]) for i in range(level_cnt)]
        mask = [np.hstack(mask[i]) for i in range(level_cnt)]
        scores = [score[:,mask[i]] for i in range(level_cnt)]
        for j in range(level_cnt):
            tp_all[j] = np.hstack([tp_all[j],tp[j]])
            fp_all[j] = np.hstack([fp_all[j],fp[j]])
            num_gts_all[j] += num_gts[j]
            score_all[j] = np.hstack([score_all[j], scores[j]])
        
    pool.close()
    # calculate recall and precision with tp and fp
    for i in range(level_cnt):
        tp = tp_all[i]
        fp = fp_all[i]
        score = score_all[i]
        num_gts = num_gts_all[i]

        num_dets = tp.shape[1]
        sort_inds = np.argsort(-score)
        tp = tp[:, sort_inds].squeeze(axis = 0)
        fp = fp[:, sort_inds].squeeze(axis = 0)
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, ['level1', 'level2', 'level3', 'level4', 'level5', 'level6', 'level7', 'level8', 'level9'], area_ranges, logger=logger)

    return mean_ap, eval_results

def eval_rbbox_map_maxlevel(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    level_cnt = 3

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []

    tp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    fp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    score_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
    select_cls = ['metalbottle', 
               'glassbottle', 
               'pressure',]
    for i in range(num_classes):
        # get gt and det bboxes of this class
        # if not dataset[i] in select_cls:
        #     continue
        cls_dets, cls_gts, _, _,  _, cls_gts_xy_max, cls_gts_ignore = get_cls_results_level(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tp = [[] for _ in range(level_cnt)]
        fp = [[] for _ in range(level_cnt)]
        mask = [[] for _ in range(level_cnt)]
        for i in range(num_imgs):
            tp_i, fp_i, mask_i = tpfp_default_level(level_cnt, cls_dets[i], cls_gts[i],
             cls_gts_xy_max[i], cls_gts_ignore[i], iou_thr, area_ranges)
            # tp_i, fp_i, mask_i = tpfp_default_angle(level_cnt, cls_dets[i], cls_gts[i],
            #  cls_gts_ignore[i], iou_thr, area_ranges)
            for j in range(level_cnt):
                tp[j].append(tp_i[j])
                fp[j].append(fp_i[j])
                mask[j].append(mask_i[j])
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                level = cls_gts_xy_max[_]
                if level.shape[0] == 0:
                    continue
                else:
                    for i in range(level.shape[0]):
                        num_gts[level[i]][0] += 1
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        score = cls_dets[:,-1:]
        score = score.T
        # num_dets = cls_dets.shape[0]
        # sort_inds = np.argsort(-cls_dets[:, -1])
        tp = [np.hstack(tp[i]) for i in range(level_cnt)]
        fp = [np.hstack(fp[i]) for i in range(level_cnt)]
        mask = [np.hstack(mask[i]) for i in range(level_cnt)]
        scores = [score[:,mask[i]] for i in range(level_cnt)]
        for j in range(level_cnt):
            tp_all[j] = np.hstack([tp_all[j],tp[j]])
            fp_all[j] = np.hstack([fp_all[j],fp[j]])
            num_gts_all[j] += num_gts[j]
            score_all[j] = np.hstack([score_all[j], scores[j]])
        
    pool.close()
    # calculate recall and precision with tp and fp
    for i in range(level_cnt):
        tp = tp_all[i]
        fp = fp_all[i]
        score = score_all[i]
        num_gts = num_gts_all[i]

        num_dets = tp.shape[1]
        sort_inds = np.argsort(-score)
        tp = tp[:, sort_inds].squeeze(axis = 0)
        fp = fp[:, sort_inds].squeeze(axis = 0)
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, ['level1', 'level2', 'level3', 'level4', 'level5', 'level6', 'level7', 'level8', 'level9'], area_ranges, logger=logger)

    return mean_ap, eval_results



def eval_rbbox_map_angle(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    level_cnt = 3

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []

    tp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    fp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    score_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]

    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tp = [[] for _ in range(level_cnt)]
        fp = [[] for _ in range(level_cnt)]
        mask = [[] for _ in range(level_cnt)]
        for i in range(num_imgs):
            # tp_i, fp_i, mask_i = tpfp_default_level(level_cnt, cls_dets[i], cls_gts[i],
            #  cls_gts_level[i], cls_gts_ignore[i], iou_thr, area_ranges)
            tp_i, fp_i, mask_i = tpfp_default_angle(level_cnt, cls_dets[i], cls_gts[i],
             cls_gts_ignore[i], iou_thr, area_ranges)
            for j in range(level_cnt):
                tp[j].append(tp_i[j])
                fp[j].append(fp_i[j])
                mask[j].append(mask_i[j])
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
        # cls_gts_level = cls_gts[:,-1]
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                level = bbox[:,-1]
                if level.shape[0] == 0:
                    continue
                else:
                    for i in range(level.shape[0]):
                        angle_level = cal_angle_level(level_cnt, level[i])
                        num_gts[angle_level-1][0] += 1
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        score = cls_dets[:,-1:]
        score = score.T
        # num_dets = cls_dets.shape[0]
        # sort_inds = np.argsort(-cls_dets[:, -1])
        tp = [np.hstack(tp[i]) for i in range(level_cnt)]
        fp = [np.hstack(fp[i]) for i in range(level_cnt)]
        mask = [np.hstack(mask[i]) for i in range(level_cnt)]
        scores = [score[:,mask[i]] for i in range(level_cnt)]
        for j in range(level_cnt):
            tp_all[j] = np.hstack([tp_all[j],tp[j]])
            fp_all[j] = np.hstack([fp_all[j],fp[j]])
            num_gts_all[j] += num_gts[j]
            score_all[j] = np.hstack([score_all[j], scores[j]])
        
    pool.close()
    # calculate recall and precision with tp and fp
    for i in range(level_cnt):
        tp = tp_all[i]
        fp = fp_all[i]
        score = score_all[i]
        num_gts = num_gts_all[i]

        num_dets = tp.shape[1]
        sort_inds = np.argsort(-score)
        tp = tp[:, sort_inds].squeeze(axis = 0)
        fp = fp[:, sort_inds].squeeze(axis = 0)
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results

def eval_rbbox_map_angle_level(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    
    assert len(det_results) == len(annotations)

    level_cnt = 3

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []

    tp_all_1 = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    fp_all_1 = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    score_all_1 = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    num_gts_all_1 = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]

    tp_all_2 = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    fp_all_2 = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    score_all_2 = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    num_gts_all_2 = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]

    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_level, cls_gts_ignore = get_cls_results_level(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tp_1 = [[] for _ in range(level_cnt)]
        fp_1 = [[] for _ in range(level_cnt)]
        mask_1 = [[] for _ in range(level_cnt)]

        tp_2 = [[] for _ in range(level_cnt)]
        fp_2 = [[] for _ in range(level_cnt)]
        mask_2 = [[] for _ in range(level_cnt)]

        for i in range(num_imgs):
            # tp_i, fp_i, mask_i = tpfp_default_level(level_cnt, cls_dets[i], cls_gts[i],
            #  cls_gts_level[i], cls_gts_ignore[i], iou_thr, area_ranges)
            tp_i_1, fp_i_1, mask_i_1, tp_i_2, fp_i_2, mask_i_2,= tpfp_default_angle_level(level_cnt, cls_dets[i], cls_gts[i],
             cls_gts_level[i], cls_gts_ignore[i], iou_thr, area_ranges)
            for j in range(level_cnt):
                tp_1[j].append(tp_i_1[j])
                fp_1[j].append(fp_i_1[j])
                mask_1[j].append(mask_i_1[j])
                tp_2[j].append(tp_i_2[j])
                fp_2[j].append(fp_i_2[j])
                mask_2[j].append(mask_i_2[j])
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts_1 = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
        num_gts_2 = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
        # cls_gts_level = cls_gts[:,-1]
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                level = cls_gts_level[_]
                a_level = bbox[:,-1]
                
                if level.shape[0] == 0:
                    continue
                else:
                    for i in range(level.shape[0]):
                        angle_level = cal_angle_level(level_cnt, a_level[i])
                        if level[i] == 1:
                            num_gts_1[angle_level-1][0] += 1
                        else:
                            num_gts_2[angle_level-1][0] += 1
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        score = cls_dets[:,-1:]
        score = score.T
        # num_dets = cls_dets.shape[0]
        # sort_inds = np.argsort(-cls_dets[:, -1])
        tp_1 = [np.hstack(tp_1[i]) for i in range(level_cnt)]
        fp_1 = [np.hstack(fp_1[i]) for i in range(level_cnt)]
        mask_1 = [np.hstack(mask_1[i]) for i in range(level_cnt)]
        scores_1 = [score[:,mask_1[i]] for i in range(level_cnt)]

        tp_2 = [np.hstack(tp_2[i]) for i in range(level_cnt)]
        fp_2 = [np.hstack(fp_2[i]) for i in range(level_cnt)]
        mask_2 = [np.hstack(mask_2[i]) for i in range(level_cnt)]
        scores_2 = [score[:,mask_2[i]] for i in range(level_cnt)]

        for j in range(level_cnt):
            tp_all_1[j] = np.hstack([tp_all_1[j],tp_1[j]])
            fp_all_1[j] = np.hstack([fp_all_1[j],fp_1[j]])
            num_gts_all_1[j] += num_gts_1[j]
            score_all_1[j] = np.hstack([score_all_1[j], scores_1[j]])

            tp_all_2[j] = np.hstack([tp_all_2[j],tp_2[j]])
            fp_all_2[j] = np.hstack([fp_all_2[j],fp_2[j]])
            num_gts_all_2[j] += num_gts_2[j]
            score_all_2[j] = np.hstack([score_all_2[j], scores_2[j]])
        
    pool.close()
    # calculate recall and precision with tp and fp
    print('level1')
    for i in range(level_cnt):
        tp = tp_all_1[i]
        fp = fp_all_1[i]
        score = score_all_1[i]
        num_gts = num_gts_all_1[i]

        num_dets = tp.shape[1]
        sort_inds = np.argsort(-score)
        tp = tp[:, sort_inds].squeeze(axis = 0)
        fp = fp[:, sort_inds].squeeze(axis = 0)
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    eval_results = []
    print('level2')
    for i in range(level_cnt):
        tp = tp_all_2[i]
        fp = fp_all_2[i]
        score = score_all_2[i]
        num_gts = num_gts_all_2[i]

        num_dets = tp.shape[1]
        sort_inds = np.argsort(-score)
        tp = tp[:, sort_inds].squeeze(axis = 0)
        fp = fp[:, sort_inds].squeeze(axis = 0)
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)


    return mean_ap, eval_results

def eval_rbbox_map_horizontal_level(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    level_cnt = 3

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []

    tp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    fp_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    score_all = [np.empty(shape=(1,0)) for _ in range(level_cnt)]
    num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]

    for i in range(num_classes):
        # if dataset[i] != 'battery':
        #     continue
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_level, cls_gts_ignore = get_cls_results_hlevel(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tp = [[] for _ in range(level_cnt)]
        fp = [[] for _ in range(level_cnt)]
        mask = [[] for _ in range(level_cnt)]
        for i in range(num_imgs):
            tp_i, fp_i, mask_i = tpfp_default_level(level_cnt, cls_dets[i], cls_gts[i],
             cls_gts_level[i], cls_gts_ignore[i], iou_thr, area_ranges)
            # tp_i, fp_i, mask_i = tpfp_default_angle(level_cnt, cls_dets[i], cls_gts[i],
            #  cls_gts_ignore[i], iou_thr, area_ranges)
            for j in range(level_cnt):
                tp[j].append(tp_i[j])
                fp[j].append(fp_i[j])
                mask[j].append(mask_i[j])
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                level = cls_gts_level[_]
                if level.shape[0] == 0:
                    continue
                else:
                    for i in range(level.shape[0]):
                        num_gts[level[i] % level_cnt][0] += 1
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        score = cls_dets[:,-1:]
        score = score.T
        # num_dets = cls_dets.shape[0]
        # sort_inds = np.argsort(-cls_dets[:, -1])
        tp = [np.hstack(tp[i]) for i in range(level_cnt)]
        fp = [np.hstack(fp[i]) for i in range(level_cnt)]
        mask = [np.hstack(mask[i]) for i in range(level_cnt)]
        scores = [score[:,mask[i]] for i in range(level_cnt)]
        for j in range(level_cnt):
            tp_all[j] = np.hstack([tp_all[j],tp[j]])
            fp_all[j] = np.hstack([fp_all[j],fp[j]])
            num_gts_all[j] += num_gts[j]
            score_all[j] = np.hstack([score_all[j], scores[j]])
        
    pool.close()
    # calculate recall and precision with tp and fp
    for i in range(level_cnt):
        tp = tp_all[i]
        fp = fp_all[i]
        score = score_all[i]
        num_gts = num_gts_all[i]

        num_dets = tp.shape[1]
        sort_inds = np.argsort(-score)
        tp = tp[:, sort_inds].squeeze(axis = 0)
        fp = fp[:, sort_inds].squeeze(axis = 0)
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, ('level1', 'level2', 'level3'), area_ranges, logger=logger)

    return mean_ap, eval_results

def eval_rbbox_map_level_class(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    
    assert len(det_results) == len(annotations)

    level_cnt = 3

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []

    # tp_all = [np.empty(shape=(1,0)) for _ in range(3)]
    # fp_all = [np.empty(shape=(1,0)) for _ in range(3)]
    # score_all = [np.empty(shape=(1,0)) for _ in range(3)]
    # num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(3)]
    select_cls = ['metalbottle', 
               'glassbottle', 
               'pressure',]
    # select_cls = ('battery', 'electronicequipment', 'glassbottle',
    #            'gun', 'knife', 'lighter', 'metalbottle',
    #            'OCbottle', 'pressure', 'umbrella')
    for cnt in range(num_classes):
        if not dataset[cnt] in select_cls:
            continue
        # get gt and det bboxes of this class
        
        cls_dets, cls_gts, _, _, _,cls_gts_xy_max, cls_gts_ignore = get_cls_results_level(
            det_results, annotations, cnt)

        # compute tp and fp for each image with multiple processes
        tp = [[] for _ in range(level_cnt)]
        fp = [[] for _ in range(level_cnt)]
        mask = [[] for _ in range(level_cnt)]
        for i in range(num_imgs):
            tp_i, fp_i, mask_i = tpfp_default_level(level_cnt, cls_dets[i], cls_gts[i], cls_gts_xy_max[i], cls_gts_ignore[i], iou_thr, area_ranges)
            for j in range(level_cnt):
                tp[j].append(tp_i[j])
                fp[j].append(fp_i[j])
                mask[j].append(mask_i[j])
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                level = cls_gts_xy_max[_]
                if level.shape[0] == 0:
                    continue
                else:
                    for i in range(level.shape[0]):
                        num_gts_all[level[i]][0] += 1
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts_all[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        score = cls_dets[:,-1:]
        score = score.T
        # num_dets = cls_dets.shape[0]
        # sort_inds = np.argsort(-cls_dets[:, -1])
        tp_all = [np.hstack(tp[i]) for i in range(level_cnt)]
        fp_all = [np.hstack(fp[i]) for i in range(level_cnt)]
        mask = [np.hstack(mask[i]) for i in range(level_cnt)]
        score_all = [score[:,mask[i]] for i in range(level_cnt)]
        for j in range(level_cnt):
            tp = tp_all[j]
            fp = fp_all[j]
            score = score_all[j]
            num_gts = num_gts_all[j]
            num_dets = tp.shape[1]
            sort_inds = np.argsort(-score)
            tp = tp[:, sort_inds].squeeze(axis = 0)
            fp = fp[:, sort_inds].squeeze(axis = 0)
            tp = np.cumsum(tp, axis=1)
            fp = np.cumsum(fp, axis=1)
            eps = np.finfo(np.float32).eps
            recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
            precisions = tp / np.maximum((tp + fp), eps)
            # calculate AP
            if scale_ranges is None:
                recalls = recalls[0, :]
                precisions = precisions[0, :]
                num_gts = num_gts.item()
            mode = 'area' if not use_07_metric else '11points'
            ap = average_precision(recalls, precisions, mode)
            eval_results.append({
                'num_gts': num_gts,
                'num_dets': num_dets,
                'recall': recalls,
                'precision': precisions,
                'ap': ap
            })
        if scale_ranges is not None:
        # shape (num_classes, num_scales)
            all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
            all_num_gts = np.vstack(
                [cls_result['num_gts'] for cls_result in eval_results])
            mean_ap = []
            for i in range(num_scales):
                if np.any(all_num_gts[:, i] > 0):
                    mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
                else:
                    mean_ap.append(0.0)
        else:
            aps = []
            for cls_result in eval_results:
                if cls_result['num_gts'] > 0:
                    aps.append(cls_result['ap'])
            mean_ap = np.array(aps).mean().item() if aps else 0.0
        print(dataset[cnt])
        print_map_summary(
            mean_ap, eval_results, ['level1', 'level2', 'level3'], area_ranges, logger=logger)
        eval_results = []
        
    pool.close()

    return mean_ap, eval_results


def eval_rbbox_map_xlevel_class(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    
    assert len(det_results) == len(annotations)

    level_cnt = 3

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []

    # tp_all = [np.empty(shape=(1,0)) for _ in range(3)]
    # fp_all = [np.empty(shape=(1,0)) for _ in range(3)]
    # score_all = [np.empty(shape=(1,0)) for _ in range(3)]
    # num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(3)]

    for cnt in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, _, cls_gts_x, _,cls_gts_ignore = get_cls_results_level(
            det_results, annotations, cnt)

        # compute tp and fp for each image with multiple processes
        tp = [[] for _ in range(level_cnt)]
        fp = [[] for _ in range(level_cnt)]
        mask = [[] for _ in range(level_cnt)]
        for i in range(num_imgs):
            tp_i, fp_i, mask_i = tpfp_default_level(level_cnt, cls_dets[i], cls_gts[i], cls_gts_x[i], cls_gts_ignore[i], iou_thr, area_ranges)
            for j in range(level_cnt):
                tp[j].append(tp_i[j])
                fp[j].append(fp_i[j])
                mask[j].append(mask_i[j])
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                level = cls_gts_x[_]
                if level.shape[0] == 0:
                    continue
                else:
                    for i in range(level.shape[0]):
                        num_gts_all[level[i]][0] += 1
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts_all[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        score = cls_dets[:,-1:]
        score = score.T
        # num_dets = cls_dets.shape[0]
        # sort_inds = np.argsort(-cls_dets[:, -1])
        tp_all = [np.hstack(tp[i]) for i in range(level_cnt)]
        fp_all = [np.hstack(fp[i]) for i in range(level_cnt)]
        mask = [np.hstack(mask[i]) for i in range(level_cnt)]
        score_all = [score[:,mask[i]] for i in range(level_cnt)]
        for j in range(level_cnt):
            tp = tp_all[j]
            fp = fp_all[j]
            score = score_all[j]
            num_gts = num_gts_all[j]
            num_dets = tp.shape[1]
            sort_inds = np.argsort(-score)
            tp = tp[:, sort_inds].squeeze(axis = 0)
            fp = fp[:, sort_inds].squeeze(axis = 0)
            tp = np.cumsum(tp, axis=1)
            fp = np.cumsum(fp, axis=1)
            eps = np.finfo(np.float32).eps
            recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
            precisions = tp / np.maximum((tp + fp), eps)
            # calculate AP
            if scale_ranges is None:
                recalls = recalls[0, :]
                precisions = precisions[0, :]
                num_gts = num_gts.item()
            mode = 'area' if not use_07_metric else '11points'
            ap = average_precision(recalls, precisions, mode)
            eval_results.append({
                'num_gts': num_gts,
                'num_dets': num_dets,
                'recall': recalls,
                'precision': precisions,
                'ap': ap
            })
        if scale_ranges is not None:
        # shape (num_classes, num_scales)
            all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
            all_num_gts = np.vstack(
                [cls_result['num_gts'] for cls_result in eval_results])
            mean_ap = []
            for i in range(num_scales):
                if np.any(all_num_gts[:, i] > 0):
                    mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
                else:
                    mean_ap.append(0.0)
        else:
            aps = []
            for cls_result in eval_results:
                if cls_result['num_gts'] > 0:
                    aps.append(cls_result['ap'])
            mean_ap = np.array(aps).mean().item() if aps else 0.0
        print(dataset[cnt])
        print_map_summary(
            mean_ap, eval_results, ['level1', 'level2', 'level3'], area_ranges, logger=logger)
        eval_results = []
        
    pool.close()

    return mean_ap, eval_results

def eval_rbbox_map_ylevel_class(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    
    assert len(det_results) == len(annotations)

    level_cnt = 3

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []

    # tp_all = [np.empty(shape=(1,0)) for _ in range(3)]
    # fp_all = [np.empty(shape=(1,0)) for _ in range(3)]
    # score_all = [np.empty(shape=(1,0)) for _ in range(3)]
    # num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(3)]

    for cnt in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, _, _, cls_gts_y, cls_gts_ignore = get_cls_results_level(
            det_results, annotations, cnt)

        # compute tp and fp for each image with multiple processes
        tp = [[] for _ in range(level_cnt)]
        fp = [[] for _ in range(level_cnt)]
        mask = [[] for _ in range(level_cnt)]
        for i in range(num_imgs):
            tp_i, fp_i, mask_i = tpfp_default_level(level_cnt, cls_dets[i], cls_gts[i], cls_gts_y[i], cls_gts_ignore[i], iou_thr, area_ranges)
            for j in range(level_cnt):
                tp[j].append(tp_i[j])
                fp[j].append(fp_i[j])
                mask[j].append(mask_i[j])
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts_all = [np.zeros(num_scales, dtype=int) for _ in range(level_cnt)]
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                level = cls_gts_y[_]
                if level.shape[0] == 0:
                    continue
                else:
                    for i in range(level.shape[0]):
                        num_gts_all[level[i]][0] += 1
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts_all[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        score = cls_dets[:,-1:]
        score = score.T
        # num_dets = cls_dets.shape[0]
        # sort_inds = np.argsort(-cls_dets[:, -1])
        tp_all = [np.hstack(tp[i]) for i in range(level_cnt)]
        fp_all = [np.hstack(fp[i]) for i in range(level_cnt)]
        mask = [np.hstack(mask[i]) for i in range(level_cnt)]
        score_all = [score[:,mask[i]] for i in range(level_cnt)]
        for j in range(level_cnt):
            tp = tp_all[j]
            fp = fp_all[j]
            score = score_all[j]
            num_gts = num_gts_all[j]
            num_dets = tp.shape[1]
            sort_inds = np.argsort(-score)
            tp = tp[:, sort_inds].squeeze(axis = 0)
            fp = fp[:, sort_inds].squeeze(axis = 0)
            tp = np.cumsum(tp, axis=1)
            fp = np.cumsum(fp, axis=1)
            eps = np.finfo(np.float32).eps
            recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
            precisions = tp / np.maximum((tp + fp), eps)
            # calculate AP
            if scale_ranges is None:
                recalls = recalls[0, :]
                precisions = precisions[0, :]
                num_gts = num_gts.item()
            mode = 'area' if not use_07_metric else '11points'
            ap = average_precision(recalls, precisions, mode)
            eval_results.append({
                'num_gts': num_gts,
                'num_dets': num_dets,
                'recall': recalls,
                'precision': precisions,
                'ap': ap
            })
        if scale_ranges is not None:
        # shape (num_classes, num_scales)
            all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
            all_num_gts = np.vstack(
                [cls_result['num_gts'] for cls_result in eval_results])
            mean_ap = []
            for i in range(num_scales):
                if np.any(all_num_gts[:, i] > 0):
                    mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
                else:
                    mean_ap.append(0.0)
        else:
            aps = []
            for cls_result in eval_results:
                if cls_result['num_gts'] > 0:
                    aps.append(cls_result['ap'])
            mean_ap = np.array(aps).mean().item() if aps else 0.0
        print(dataset[cnt])
        print_map_summary(
            mean_ap, eval_results, ['level1', 'level2', 'level3'], area_ranges, logger=logger)
        eval_results = []
        
    pool.close()

    return mean_ap, eval_results




def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)  #定义
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        if cls_result['precision'].size > 0:
            precisions[:, i] = np.array(cls_result['precision'], ndmin=2)[:, -1]  #添加值
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'precision', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{precisions[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)


def cal_angle_level(cnt, angle):
    angle_per_range = np.pi / cnt
    level = int((angle + np.pi / 2) / angle_per_range) + 1
    assert level >= 1 and level <= cnt
    return level
