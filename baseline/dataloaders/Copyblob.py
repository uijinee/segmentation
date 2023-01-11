from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2, os
import numpy as np
import random


class Copyblob:
    def __init__(self, train_json_path, cl, p):
        self.coco = COCO(train_json_path)
        self.img_ids = self.coco.getImgIds(catIds=cl)
        self.img_infos = self.coco.loadImgs(self.img_ids)
        self.cl = cl
        self.p = p
        self.category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
                                'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

    def __call__(self, image):
        app = random.choices((True, False), weights = [self.p, 1-self.p])
        if not app:
            return image
        src_img_info = random.choice(self.img_infos)
        src_image = cv2.imread(os.path.join('/opt/ml/input/data', src_img_info['file_name']))
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        src_image /= 255.0

        src_ann_id = self.coco.getAnnIds(imgIds=src_img_info['id'])
        src_ann = self.coco.loadAnns(src_ann_id)
        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)
        
        src_mask = np.zeros((src_img_info["height"], src_img_info["width"]))
        src_ann = sorted(src_ann, key=lambda idx : idx['area'], reverse=True)
        for i in range(len(src_ann)):
            className = self.get_classname(src_ann[i]['category_id'], cats)
            pixel_value = self.category_names.index(className)
            src_mask[self.coco.annToMask(src_ann[i]) == 1] = pixel_value
        src_mask = src_mask.astype(np.int8)

        self.copyblob(src, dst=image, self.cl, 0)
        
    def get_classname(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"

#   copyblob(src_img=images[i], src_mask=masks[i], dst_img=images[rand_idx], dst_mask=masks[rand_idx], src_class=4, dst_class=0)
    def copyblob(self, src_img, src_mask, dst_img, dst_mask, src_class, dst_class):
        """ copy src blob and paste to any dst blob"""
        mask_y, mask_x = src_mask.size()
        """ get src object's min index"""
        src_idx = np.where(src_mask==src_class)
        
        src_idx_sum = list(src_idx[0][i] + src_idx[1][i] for i in range(len(src_idx[0])))
        src_idx_sum_min_idx = np.argmin(src_idx_sum)        
        src_idx_min = src_idx[0][src_idx_sum_min_idx], src_idx[1][src_idx_sum_min_idx]
        
        """ get dst object's random index"""
        dst_idx = np.where(dst_mask==dst_class)
        rand_idx = np.random.randint(len(dst_idx[0]))
        target_pos = dst_idx[0][rand_idx], dst_idx[1][rand_idx] 
        
        src_dst_offset = tuple(map(lambda x, y: x - y, src_idx_min, target_pos))
        dst_idx = tuple(map(lambda x, y: x - y, src_idx, src_dst_offset))
        
        for i in range(len(dst_idx[0])):
            dst_idx[0][i] = (min(dst_idx[0][i], mask_y-1))
        for i in range(len(dst_idx[1])):
            dst_idx[1][i] = (min(dst_idx[1][i], mask_x-1))
        
        dst_mask[dst_idx] = src_class
        dst_img[:, dst_idx[0], dst_idx[1]] = src_img[:, src_idx[0], src_idx[1]]