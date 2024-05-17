import glob
import os.path as osp
import os
import numpy as np
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from tools.test import main
import matplotlib.pyplot as plt

def main():
    ann_folder = '/media/datasets/rotate_labels_buaa/test/annotations'
    classes = ('pressure', 'umbrella', 'lighter',
               'OCbottle', 'glassbottle', 'battery', 'metalbottle',
               'knife', 'electronicequipment')
    cal(ann_folder, classes, 'le90')

def cal(ann_folder,classes,version):
        """
            Args:
                ann_folder: folder that contains ROXray annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(classes)
                   }  # in mmdet v2.0 label is 0-based
        cls_level = [{c: [] for c in classes} for _ in range(2)]
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        
        for ann_file in ann_files:
            data_info = {}
            img_id = osp.split(ann_file)[1][:-4]
            img_name = img_id + '.jpg'
            data_info['filename'] = img_name
            data_info['ann'] = {}


            with open(ann_file) as f:
                s = f.readlines()
                for si in s:
                    bbox_info = si.split()
                    poly = np.array(bbox_info[-8:], dtype=np.float32)
                    try:
                        x, y, w, h, a = poly2obb_np(poly, version)
                    except:  # noqa: E722
                        continue
                    level = bbox_info[1]

                    cls_name = bbox_info[2]
                    rate = max(h / w, w / h)
                    cls_level[int(level)-1][cls_name].append(rate)


            data_infos.append(data_info)
        for c in classes:
            plt.figure()
            name = c
            plt.title(name)
            for i in range(2):
                ratios = cls_level[i][c]
                r = sorted(ratios)
                plt.subplot(1,2,i+1)
                plt.scatter([k for k in range(len(r))],r)
            plt.savefig('/media/datasets/rotate_labels_buaa/test/figs/'+name+'.jpg')
        img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos


if __name__ == '__main__':
    main()