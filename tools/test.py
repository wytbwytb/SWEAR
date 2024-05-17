# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import numpy
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn

from mmcv.image import tensor2imgs
from mmcv.ops import box_iou_rotated
from mmcv.parallel import MMDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, replace_ImageToTensor

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import (build_ddp, build_dp, compat_cfg, get_device,
                            setup_multi_processes)


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
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
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    ##
    # args.config = '/home/wtb/mmrotate/configs/ours/ours_r50_fpn_3x_roxray_p_3000_le90.py'
    # args.checkpoint = '/media/datasets/gpu17_models/mmrotate/ours/ours_r50_fpn_3x_roxray_p_le90/mAP671.pth'
    # args.eval = 'mAP' 
    
    args.config = '/home/wtb/mmrotate/configs/ours/ours_r50_fpn_3x_roxray_p_3000_le90.py'
    args.checkpoint = '/media/datasets/gpu17_models/mmrotate/ours/ours_r50_fpn_3x_roxray_p_le90/0323_mAP736.pth'
    args.eval = 'mAP' 
    # args.config = '/home/wtb/mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_3x_roxray_3000_le90.py'
    # args.checkpoint = '/media/datasets/gpu17_models/mmrotate/oriented_rcnn/oriented_rcnn_r50_fpn_3x_roxray_3000_le90/epoch_35.pth'
    # args.eval = 'mAP'
    # args.config = 'configs/sasm_reppoints/sasm_reppoints_r50_fpn_3x_roxray_p_le90.py'
    # args.checkpoint = '/media/datasets/gpu17_models/mmrotate/models/sasm_reppoints_r50_fpn_3x_roxray_p_oc/latest.pth'
    # args.eval = 'mAP' 
    # args.config = '/home/wtb/mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py'
    # args.checkpoint = '/media/datasets/gpu17_models/mmrotate/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/latest.pth'
    # args.eval = 'mAP'
    # args.config = 'configs/hybrid/hybrid_r50_fpn_3x_roxray_b_le90.py'
    # args.checkpoint = '/media/datasets/gpu17_models/mmrotate/hybrid/hybrid_r50_fpn_3x_roxray_b_le90/epoch_28.pth'
    # args.eval = 'mAP'  
    # args.show_dir = '/media/datasets/gpu17_models/mmrotate/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/vis'
    # args.show_dir = '/media/datasets/roxray/test/vis'
    # args.config = './configs/oriented_reppoints/oriented_reppoints_r50_fpn_3x_roxray_le90.py'
    # args.checkpoint = '/media/datasets/gpu17_models/mmrotate/oriented_reppoints/oriented_reppoints_r50_fpn_3x_roxray_le90/814.pth'
    # args.eval = 'mAP' 
    
    # args.format_only = './'
    # args.out = 'results.pkl'
    # args.show_dir = '/media/datasets/gpu17_models/mmrotate/oriented_rcnn/oriented_rcnn_r50_fpn_3x_roxray_le90/vis'
    ##
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    if args.format_only and cfg.mp_start_method != 'spawn':
        warnings.warn(
            '`mp_start_method` in `cfg` is set to `spawn` to use CUDA '
            'with multiprocessing when formatting output result.')
        cfg.mp_start_method = 'spawn'

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if 'samples_per_gpu' in cfg.data.test:
            warnings.warn('`samples_per_gpu` in `test` field of '
                          'data will be deprecated, you should'
                          ' move it to `test_dataloader` field')
            test_dataloader_default_args['samples_per_gpu'] = \
                cfg.data.test.pop('samples_per_gpu')
        if test_dataloader_default_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
            if 'samples_per_gpu' in ds_cfg:
                warnings.warn('`samples_per_gpu` in `test` field of '
                              'data will be deprecated, you should'
                              ' move it to `test_dataloader` field')
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        test_dataloader_default_args['samples_per_gpu'] = samples_per_gpu
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is None and cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        b = osp.dirname(args.config).split('/')[-1]
        args.work_dir = osp.join('/media/datasets/gpu17_models/mmrotate/', b ,
                                osp.splitext(osp.basename(args.config))[0])

    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    cfg.device = get_device()
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None or cfg.device == 'npu':
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    print(args.show_dir)

    ## show ground truth
    # if not distributed:
    #     model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    # show_score_thr=0.3
    # # dir = '/media/datasets/rotate_labels_buaa/test/vis'
    # dir = args.show_dir
    # model.eval()
    # results = []
    # dataset = data_loader.dataset
    # PALETTE = getattr(dataset, 'PALETTE', None)
    # prog_bar = mmcv.ProgressBar(len(dataset))
    # for i, data in enumerate(data_loader):
    #     # with torch.no_grad():
    #     #     result = model(return_loss=False, rescale=True, **data)[0]

    #     # batch_size = len(result)
    #     batch_size = len(data['img'])
    #     if args.show or dir:
    #         if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
    #             img_tensor = data['img'][0]
    #         else:
    #             img_tensor = data['img'][0].data[0]
    #         img_metas = data['img_metas'][0].data[0]
            
    #         bboxes = dataset.data_infos[i]['ann']['bboxes']
    #         labels = dataset.data_infos[i]['ann']['labels']
    #         # levels = dataset.data_infos[i]['ann']['level']
    #         results = [numpy.array([[0,0,0,0,0,0]], dtype=numpy.float32) for i in range(len(dataset.CLASSES))]
    #         # results = [numpy.array([[0,0,0,0,0]], dtype=numpy.float32) for i in range(len(dataset.CLASSES))]
    #         # results = [[numpy.array([[0,0,0,0,0,0]], dtype=numpy.float32) for i in range(len(dataset.CLASSES))] for _ in range(2)]
    #         # for j, (bbox, label, level) in enumerate(zip(bboxes, labels, levels)):
    #         #     # print(j)
    #         #     bbox = numpy.append(bbox,level)
    #         #     bbox = numpy.expand_dims(bbox,0)
    #         #     l = int(level) - 1
    #         #     results[l][label] = numpy.concatenate((results[l][label], bbox))
    #         for j, (bbox, label) in enumerate(zip(bboxes, labels)):
    #             # print(j)
    #             bbox = numpy.append(bbox, 1)
    #             bbox = numpy.expand_dims(bbox,0)
    #             # l = int(level) - 1
    #             results[label] = numpy.concatenate((results[label], bbox))
    #         filename = dataset.data_infos[i]['filename']
    #         filename2 = img_metas[0]['ori_filename']
    #         assert filename == filename2

    #         imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    #         assert len(imgs) == len(img_metas)

    #         for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
    #             print(img_meta['ori_filename'])
    #             h, w, _ = img_meta['img_shape']
    #             img_show = img[:h, :w, :]

    #             ori_h, ori_w = img_meta['ori_shape'][:-1]
    #             img_show = mmcv.imresize(img_show, (ori_w, ori_h))

    #             # for each class
    #             ## level
    #             # for level in range(2):
    #             #     for i, cls in enumerate(dataset.CLASSES):
    #             #         out_dir = os.path.join(dir, str(level + 1), cls)
    #             #         if out_dir:
    #             #             out_file = osp.join(out_dir, img_meta['ori_filename'])
    #             #         else:
    #             #             out_file = None

    #             #         result_cls = [numpy.array([[0,0,0,0,0,0]], dtype=numpy.float32) for i in range(len(dataset.CLASSES))]
    #             #         if results[level][i].shape[0] == 1:
    #             #             continue 
    #             #         result_cls[i] = results[level][i]
    #             #         model.module.show_result(
    #             #             img_show,
    #             #             result_cls,
    #             #             bbox_color=PALETTE,
    #             #             text_color=PALETTE,
    #             #             mask_color=PALETTE,
    #             #             show=args.show,
    #             #             out_file=out_file,
    #             #             score_thr=show_score_thr)

    #             # for i, cls in enumerate(dataset.CLASSES):
    #             #     out_dir = os.path.join(dir, cls)
    #             #     if out_dir:
    #             #         out_file = osp.join(out_dir, img_meta['ori_filename'])
    #             #     else:
    #             #         out_file = None
                    

    #             #     result_cls = [numpy.array([[0,0,0,0,0,0]], dtype=numpy.float32) for i in range(len(dataset.CLASSES))]
    #             #     # if results[i].shape[0] == 1:
    #             #     if results[i].shape[0] > 1 and result[i].shape[0] > 0:
    #             #         ious = box_iou_rotated(
    #             #             torch.from_numpy(results[i]).float(),
    #             #             torch.from_numpy(result[i]).float()).numpy()
    #             #         # for each det, the max iou with all gts
    #             #         ious_max = ious.max(axis=1)
    #             #         results[i] = results[i][ious_max < 0.5]
    #             #         # for each det, which gt overlaps most with it
    #             #         # ious_argmax = ious.argmax(axis=1)
    #             #     if results[i].shape[0] == 1:
    #             #     # if result[i].shape[0] == 0:
    #             #         continue 
    #             #     result_cls[i] = results[i]
    #             #     model.module.show_result(
    #             #         img_show,
    #             #         result_cls,
    #             #         bbox_color=PALETTE,
    #             #         text_color=PALETTE,
    #             #         mask_color=PALETTE,
    #             #         show=args.show,
    #             #         out_file=out_file,
    #             #         score_thr=show_score_thr)
                    
    #             out_dir = dir
    #             if out_dir:
    #                 out_file = osp.join(out_dir, img_meta['ori_filename'])
    #             else:
    #                 out_file = None
    #             model.module.show_result(
    #                 img_show,
    #                 result,
    #                 bbox_color=PALETTE,
    #                 text_color=PALETTE,
    #                 mask_color=PALETTE,
    #                 show=args.show,
    #                 out_file=out_file,
    #                 score_thr=show_score_thr)

    # #
    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, args.format_only)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()
