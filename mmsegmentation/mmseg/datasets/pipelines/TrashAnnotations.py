import os.path as osp

from pycocotools.coco import COCO
import mmcv
import numpy as np

from ..builder import PIPELINES

@PIPELINES.register_module()
class CustomLoadAnnotations(object):
    def __init__(self,
                coco_json_path,
                reduce_zero_label=False,
                file_client_args=dict(backend='disk'),
                imdecode_backend='pillow'):
        self.coco=COCO(coco_json_path)
        self.reduce_zero_label=reduce_zero_label
        self.file_client_args=file_client_args.copy()
        self.file_client=None
        self.imdecode_backend=imdecode_backend
    
    def __call__(self, results):
        coco_ind = results['ann_info']['coco_image_id']
        image_info = self.coco.loadImgs(coco_ind)[0]
        ann_inds = self.coco.getAnnIds(coco_ind)
        anns = self.coco.loadAnns(ann_inds)
        anns = list(sorted(anns, key=lambda x:-x['area']))

        gt_semantic_seg = np.zeros((image_info['height'], image_info['width']))
        for ann in anns:
            gt_semantic_seg[self.coco.annToMask(ann)==1] = ann['category_id']
        gt_semantic_seg = gt_semantic_seg.astype(np.int64)

        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        
        if self.reduce_zero_label:
            gt_semantic_seg[gt_semantic_seg==0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg==254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def get_coco(self):
        return self.coco