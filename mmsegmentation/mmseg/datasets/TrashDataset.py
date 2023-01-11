import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose, CustomLoadAnnotations


from .custom import CustomDataset
@DATASETS.register_module()
class TrashDataset(CustomDataset):

       CLASS = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
       "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

       PALETTE = None
       def __init__(self,
                     pipeline, 
                     coco_json_path, 
                     is_valid,
                     img_dir, 
                     img_suffix='.jpg', 
                     ann_dir=None, 
                     seg_map_suffix='.png', 
                     split=None,
                     data_root=None, 
                     test_mode=False, 
                     ignore_index=255, 
                     reduce_zero_label=False,
                     classes=None, 
                     palette=None, 
                     gt_seg_map_loader_cfg=None, 
                     file_client_args=dict(backend='disk')):
              super().__init__(
                            pipeline,
                            img_dir,
                            img_suffix,
                            ann_dir,
                            seg_map_suffix,
                            split,
                            data_root,
                            test_mode,
                            ignore_index,
                            reduce_zero_label,
                            classes,
                            palette,
                            gt_seg_map_loader_cfg=None,
                            file_client_args=file_client_args)
              self.coco_json_path=coco_json_path
              self.is_valid=is_valid
              self.gt_seg_map_loader=CustomLoadAnnotations(self.coco_json_path)

       def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
              img_infos = []
              for img in self.gt_seg_map_loader.get_coco().imgs.values():
                     img_info=dict(filename=img['file_name'])
                     img_infos.append(img_info)
                     img_info['ann']=dict(coco_image_id=img['id'])
              return img_infos
       
       def get_ann_info(self, idx):
              return self.img_infos[idx]['ann']

       # results['seg_prefix] = ann_dir
       # results[ann_info]['seg_map'] = batch_03/0103.png
       # filename = results['seg_prefix] + results[ann_info]['seg_map']
       def pre_pipeline(self, results):
              results['seg_fields'] = []
              results['img_prefix'] = self.img_dir
              if self.custom_classes:
                     results['label_mapo'] = self.label_map