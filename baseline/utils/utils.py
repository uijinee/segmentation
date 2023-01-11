# # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import random
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                        minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu


def add_hist(hist, label_trues, label_preds, n_class):
    """
        stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist




def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


# def label_accuracy_score(label_trues, label_preds, n_class):
#     """Returns accuracy score evaluation result.
#       - overall accuracy
#       - mean accuracy
#       - mean IU
#       - fwavacc
#     """
#     hist = np.zeros((n_class, n_class))
#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
#     acc = np.diag(hist).sum() / hist.sum()
#     with np.errstate(divide='ignore', invalid='ignore'):
#         acc_cls = np.diag(hist) / hist.sum(axis=1)
#     acc_cls = np.nanmean(acc_cls)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         iu = np.diag(hist) / (
#             hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
#         )
#     mean_iu = np.nanmean(iu)
#     freq = hist.sum(axis=1) / hist.sum()
#     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#     return acc, acc_cls, mean_iu, fwavacc, iu


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_classname(classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"

def get_random_img_mask(coco, cls, transform):
    img_ids = coco.getImgIds(catIds=cls)
    img_infos = coco.loadImgs(img_ids)
    category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
                    'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    img_info = random.choice(img_infos)
    image = cv2.imread(os.path.join('/opt/ml/input/data', img_info['file_name']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0

    ann_id = coco.getAnnIds(imgIds=img_info['id'])
    ann = coco.loadAnns(ann_id)
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    
    mask = np.zeros((img_info["height"], img_info["width"]))
    ann = sorted(ann, key=lambda idx : idx['area'], reverse=True)
    for i in range(len(ann)):
        className = get_classname(ann[i]['category_id'], cats)
        pixel_value = category_names.index(className)
        mask[coco.annToMask(ann[i]) == 1] = pixel_value
    mask = mask.astype(np.int8)

    if transform is not None:
        transformed = transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
    return image, mask

def copyblob(coco, transform, dst_img, dst_mask, src_class, dst_class, p=0.5):
    app = random.choices((True, False), weights = [p, 1-p])
    if not app:
        pass
    src_img, src_mask = get_random_img_mask(coco, src_class, transform)

    mask_hist_src, _ = np.histogram(src_mask.numpy().ravel(), 11, [0, 11])
    mask_hist_dst, _ = np.histogram(dst_mask.numpy().ravel(), 11, [0, 11])

    if mask_hist_src[src_class] != 0 and mask_hist_dst[dst_class] != 0:
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