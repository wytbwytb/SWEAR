# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator
from mmcv.parallel import MMDataParallel
from mmcv import Config, DictAction
from mmcv.ops import nms_rotated
from mmdet.datasets import build_dataset
from mmrotate.models import build_detector
from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmcv.runner import init_dist
from mmcv.image import tensor2imgs

from mmrotate.core.bbox import rbbox_overlaps

CLASSES = ('pressure', 'umbrella', 'lighter',
               'OCbottle', 'glassbottle', 'battery', 'metalbottle',
               'knife', 'electronicequipment')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument(
        '--prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        '--save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='plasma',
        help='theme of the matrix color map')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=None,
        help='nms IoU threshold, only applied when users want to change the'
        'nms IoU threshold.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def analysis(mode, 
            model, 
            data_loader, 
            dataset,
            results,
            save_path,
            score_thr=0,
            nms_iou_thr=None,
            tp_iou_thr=0.5):
   
    num_classes = len(dataset.CLASSES)
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)
    prog_bar = mmcv.ProgressBar(len(results))

    assert mode in ['find_not_detect', 'misclassification']
    if mode == 'find_not_detect':
        path = os.path.join(save_path, 'not_detect')
    elif mode == 'misclassification':
        path = os.path.join(save_path, 'miscls')
    else:
        return 0

    PALETTE = getattr(dataset, 'PALETTE', None)
    # for idx, per_img_res in enumerate(results):
    for idx, data in enumerate(data_loader):
        img_tensor = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        img = imgs[0]
        img_meta = img_metas[0]
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        per_img_res = results[idx]
        if isinstance(per_img_res, tuple):
            res_bboxes, _ = per_img_res
        else:
            res_bboxes = per_img_res
        ann = dataset.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        labels = ann['labels']
        filename = dataset.data_infos[idx]['filename']
        print(filename)
        analysis_per_img(mode, model, PALETTE, filename, img_show, gt_bboxes, labels, res_bboxes, path, 
                             score_thr, tp_iou_thr, nms_iou_thr)
        prog_bar.update()
    return confusion_matrix


def analysis_per_img(mode, 
                    model,
                    PALETTE, 
                    filename, 
                    img_show,
                    gt_bboxes,
                    gt_labels,
                    result,
                    save_path, 
                    score_thr=0,
                    tp_iou_thr=0.5,
                    nms_iou_thr=None):
    
    true_positives = np.zeros_like(gt_labels)
    
    gt_bboxes = torch.from_numpy(gt_bboxes).float()
    for det_label, det_bboxes in enumerate(result):
        miscls = np.zeros(det_bboxes.shape[0])
        results = np.array([[0,0,0,0,0,0]], dtype=np.float32)
        det_bboxes = torch.from_numpy(det_bboxes).float()
        if nms_iou_thr:
            det_bboxes, _ = nms_rotated(
                det_bboxes[:, :5],
                det_bboxes[:, -1],
                nms_iou_thr,
                score_threshold=score_thr)
        ious = rbbox_overlaps(det_bboxes[:, :5], gt_bboxes)
        for i, det_bbox in enumerate(det_bboxes):
            score = det_bbox[5]
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                        else:
                            miscls[i] = 1
                            if len(det_bbox.shape) == 1:
                                det_bbox = np.expand_dims(det_bbox, 0)
                            
                            results = np.concatenate((results, det_bbox))
                if det_match == 0:  # BG FP
                    miscls[i] = 1
                    det_bbox = np.expand_dims(det_bbox, 0)
                    results = np.concatenate((results, det_bbox))

        if mode == 'misclassification':  # BG FP
            cls = CLASSES[det_label]
            out_file = os.path.join(save_path, cls, filename)
            result_cls = [np.array([[0,0,0,0,0,0]], dtype=np.float32) for _ in range(len(CLASSES))]
            if results.shape[0] == 1:
                continue 
            result_cls[det_label] = results
            model.module.show_result(
                        img_show,
                        result_cls,
                        bbox_color=PALETTE,
                        text_color=PALETTE,
                        mask_color=PALETTE,
                        show=False,
                        out_file=out_file,
                        score_thr=0)
        
    if mode == 'find_not_detect':
        results = [np.array([[0,0,0,0,0,0]], dtype=np.float32) for i in range(len(CLASSES))]
        for num_tp, gt_label, bbox in zip(true_positives, gt_labels, gt_bboxes):
            if num_tp == 0:  # FN
                bbox = np.append(bbox,1.0)
                bbox = np.expand_dims(bbox,0)
                results[gt_label] = np.concatenate((results[gt_label], bbox))
            
        for i, cls in enumerate(CLASSES):
            out_file = os.path.join(save_path, cls, filename)
            result_cls = [np.array([[0,0,0,0,0,0]], dtype=np.float32) for i in range(len(CLASSES))]
            if results[i].shape[0] == 1:
                continue 
            result_cls[i] = results[i]
            model.module.show_result(
                            img_show,
                            result_cls,
                            bbox_color=PALETTE,
                            text_color=PALETTE,
                            mask_color=PALETTE,
                            show=False,
                            out_file=out_file,
                            score_thr=0)
        
    

def main():
    args = parse_args()

    args.config = 'configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_roxray_le90.py'
    args.prediction_path = 'results.pkl'
    args.save_dir = '/media/datasets/gpu17_models/mmrotate/oriented_rcnn/oriented_rcnn_r50_fpn_1x_roxray_le90'
    args.show = True

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    results = mmcv.load(args.prediction_path)
    assert isinstance(results, list)
    if isinstance(results[0], list):
        pass
    elif isinstance(results[0], tuple):
        results = [result[0] for result in results]
    else:
        raise TypeError('invalid type of prediction results')


    distributed = False
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model)
    model.eval()

    # analysis('find_not_detect', model, data_loader, dataset, results, args.save_dir,
    #                 args.score_thr,
    #                 args.nms_iou_thr,
    #                 args.tp_iou_thr)
    
    analysis('misclassification', model, data_loader, dataset, results, args.save_dir,
                    args.score_thr,
                    args.nms_iou_thr,
                    args.tp_iou_thr)
    


    # confusion_matrix_l1, confusion_matrix_l2 = calculate_confusion_matrix_level(dataset, results,
    #                                               args.score_thr,
    #                                               args.nms_iou_thr,
    #                                               args.tp_iou_thr)


if __name__ == '__main__':
    main()
