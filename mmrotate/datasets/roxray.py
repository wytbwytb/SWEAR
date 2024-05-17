# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from mmrotate.core.evaluation.eval_map import eval_rbbox_map_level, eval_rbbox_map_level_class, eval_rbbox_map_angle, eval_rbbox_map_angle_level, eval_map 
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class ROXrayDataset(CustomDataset):
    """ROXray dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES = ('battery', 'electronicequipment', 'glassbottle',
               'gun', 'knife', 'lighter', 'metalbottle',
               'OCbottle', 'pressure', 'umbrella')

    PALETTE = [(165, 42, 42), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (0,0,255)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty

        super(ROXrayDataset, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains ROXray annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        self.gun_imgids = set()
        
        for ann_file in ann_files:
            data_info = {}
            img_id = osp.split(ann_file)[1][:-4]
            img_name = img_id + '.jpg'
            data_info['filename'] = img_name
            data_info['ann'] = {}
            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            gt_bboxes_h = []
            gt_level = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            gt_polygons_ignore = []
            gt_level_ignore = []

            if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                continue

            with open(ann_file) as f:
                s = f.readlines()
                for si in s:
                    bbox_info = si.split()
                    poly = np.array(bbox_info[-8:], dtype=np.float32)
                    # bbox = np.array(bbox_info[2:6], dtype=np.float32)
                    bbox = np.array(bbox_info[3:7], dtype=np.float32)
                    # xmin, ymin, xmax, ymax = bbox_info[2:6]
                    # a = [xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax]
                    # poly = np.array(a, dtype=np.float32)
                    try:
                        x, y, w, h, a = poly2obb_np(poly, self.version)
                    except:  # noqa: E722
                        continue
                    # level = bbox_info[1]
                    cls_name = bbox_info[1]
                    # if cls_name == 'gun':
                    #     continue
                    # cls_name = bbox_info[2]
                    difficulty = 0
                    label = cls_map[cls_name]
                    if difficulty > self.difficulty:
                        pass
                    else:
                        gt_bboxes.append([x, y, w, h, a])
                        gt_bboxes_h.append(bbox)
                        gt_labels.append(label)
                        gt_polygons.append(poly)
                        # gt_level.append(level)
                    if cls_name == 'gun':
                        self.gun_imgids.add(img_id)

            if gt_bboxes:
                data_info['ann']['bboxes'] = np.array(
                    gt_bboxes, dtype=np.float32)
                data_info['ann']['labels'] = np.array(
                    gt_labels, dtype=np.int64)
                data_info['ann']['bboxes_h'] = np.array(
                    gt_bboxes_h, dtype=np.float32)
                data_info['ann']['polygons'] = np.array(
                    gt_polygons, dtype=np.float32)
                data_info['ann']['level'] = np.array(
                    gt_level, dtype=np.int64)
            else:
                data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                        dtype=np.float32)
                data_info['ann']['labels'] = np.array([], dtype=np.int64)
                data_info['ann']['bboxes_h'] = np.zeros((0, 4),
                                                        dtype=np.float32)
                data_info['ann']['polygons'] = np.zeros((0, 8),
                                                        dtype=np.float32)
                data_info['ann']['level'] = np.array([], dtype=np.int64)

            if gt_polygons_ignore:
                data_info['ann']['bboxes_ignore'] = np.array(
                    gt_bboxes_ignore, dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    gt_labels_ignore, dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.array(
                    gt_polygons_ignore, dtype=np.float32)
                data_info['ann']['level_ignore'] = np.array(
                    gt_level_ignore, dtype=np.int64)
            else:
                data_info['ann']['bboxes_ignore'] = np.zeros(
                    (0, 5), dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    [], dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.zeros(
                    (0, 8), dtype=np.float32)
                data_info['ann']['level_ignore'] = np.array(
                    [], dtype=np.int64)
                
            
            data_infos.append(data_info)
            
        self.gun_imgids = list(self.gun_imgids)
        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None
    
        file = osp.join(submission_dir, 'results'+'.txt')
        file_objs = open(file, 'w')
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]

            for j in range(len(self.CLASSES)):
                dets = result[j]
                cls = self.CLASSES[j]

                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    txt_element = [img_id+'.jpg', cls, str(bbox[-1])
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]
                    file_objs.writelines(' '.join(txt_element) + '\n')

        file_objs.close()

    
    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=0.5,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
        else:
            raise NotImplementedError

        return eval_results
